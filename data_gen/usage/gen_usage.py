from __future__ import annotations
import argparse, os, json, random, math
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import xxhash, yaml

from data_gen.common.io_utils import save_csv, save_parquet
from data_gen.common.minio_utils import upload_file, upload_folder


def _rng(seed_text: str) -> random.Random:
    return random.Random(xxhash.xxh64_intdigest(seed_text))

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def _lognorm_bytes(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))

def _heavy_tail_mix(rng: random.Random, base_val: float, ht_cfg: dict) -> float:
    p = float(ht_cfg.get("prob", 0.0))
    if rng.random() >= p:
        return base_val
    alpha = float(ht_cfg.get("pareto_alpha", 2.0))
    scale = float(ht_cfg.get("pareto_scale", 1.0))
    u = max(1e-12, rng.random())
    tail = scale * (u ** (-1.0 / alpha))
    return base_val * tail

def _cap_by_bw(bytes_val: float, bw_mbps: int | None, pct_cap: float, seconds: int) -> float:
    if bw_mbps is None:
        return bytes_val
    cap = bw_mbps * 1_000_000 / 8 * seconds * pct_cap
    return min(bytes_val, cap)

def _build_daypart_map(cfg: dict) -> Dict[int, Tuple[float, float]]:
    m = {}
    prof = cfg["daypart_profile"]
    for block in prof.values():
        dl = float(block["dl"]); ul = float(block["ul"])
        for h in block["hours"]:
            m[int(h)] = (dl, ul)
    for h in range(24):
        if h not in m:
            m[h] = (1.0, 1.0)
    return m

def _choose_granularity_5min(total_services: int, rows_target: int, ratio_max: float) -> Tuple[int, int]:
    """
    5-minute mix:
      rows/day ≈ 24*N_hourly + 288*N_5min
               = 24N + (288-24)*m = 24N + 264*m
    => m ≈ (rows_target - 24N) / 264
    clamp [0, N*ratio_max]
    """
    base = 24 * total_services
    m = max(0, (rows_target - base) // 264)
    m = int(min(m, total_services * ratio_max))
    return m, total_services - m

def _make_ids_pool(seed: str, sid: str, max_mac: int, max_dev: int) -> Tuple[List[str], List[str]]:
    rng = _rng(f"{seed}|ids|{sid}")
    macs = [f"MAC{rng.randint(0, 16**10):010X}" for _ in range(rng.randint(0, max_mac))]
    devs = [f"DEV{rng.randint(0, 16**10):010X}" for _ in range(rng.randint(0, max_dev))]
    return macs, devs

def _apply_jitter(ts: datetime, rng: random.Random, max_sec: int) -> datetime:
    if max_sec <= 0:
        return ts
    return ts + timedelta(seconds=rng.randint(0, max_sec))

def _gen_for_service_day(
    rng: random.Random,
    service_id: str,
    customer_id: str,
    the_day: date,
    granularity: str,                       # "5min" | "hour"
    dp_map: Dict[int, Tuple[float,float]],
    weekend_boost: Dict[str, float],
    cam_ul_night_boost: float,
    evening_stream_boost: float,
    ul_mu: float, ul_sigma: float,
    dl_mu: float, dl_sigma: float,
    bw_dl: int|None,
    bw_ul: int|None,
    cap_dl_pct: float, cap_ul_pct: float,
    session_p: float, mac_p: float, dev_p: float,
    mac_pool: List[str], dev_pool: List[str],
    outage: Tuple[int,int]|None,            # (start_idx, length) in 5-min or hour units
    spike_idx: int|None,
    spike_mult_range: Tuple[float,float],
    # volatility & tails:
    ht_cfg: dict,
    ar1_phi: float,
    ar1_shock_sigma: float,
    daily_vol_sigma: float,
    jitter_max: int,
    random_drop_pct: float,
) -> pd.DataFrame:
    if granularity == "5min":
        intervals = 288; step = timedelta(minutes=5); seconds = 300
    else:  # "hour"
        intervals = 24;  step = timedelta(hours=1);   seconds = 3600

    rows = []
    t0 = datetime(the_day.year, the_day.month, the_day.day)
    is_weekend = t0.weekday() >= 5
    w_dl = float(weekend_boost.get("dl", 1.0)) if is_weekend else 1.0
    w_ul = float(weekend_boost.get("ul", 1.0)) if is_weekend else 1.0

    # AR(1) state trong log-space
    z_dl = 0.0
    z_ul = 0.0
    day_vol_dl = math.exp(rng.gauss(0.0, daily_vol_sigma))
    day_vol_ul = math.exp(rng.gauss(0.0, daily_vol_sigma))

    for i in range(intervals):
        # random drop rời rạc
        if rng.random() < random_drop_pct:
            continue

        cur = t0 + i*step
        hour = cur.hour
        dl_fac, ul_fac = dp_map[hour]

        # boosts
        if hour in (18,19,20,21,22,23):
            dl_fac *= evening_stream_boost
        if hour in (0,1,2,3,4,5):
            ul_fac *= cam_ul_night_boost

        # base lognormal + heavy tail
        base_dl = _lognorm_bytes(rng, dl_mu, dl_sigma)
        base_ul = _lognorm_bytes(rng, ul_mu, ul_sigma)
        base_dl = _heavy_tail_mix(rng, base_dl, ht_cfg)
        base_ul = _heavy_tail_mix(rng, base_ul, ht_cfg)

        # AR(1) multipliers
        z_dl = ar1_phi * z_dl + rng.gauss(0.0, ar1_shock_sigma)
        z_ul = ar1_phi * z_ul + rng.gauss(0.0, ar1_shock_sigma)
        ar_mult_dl = math.exp(z_dl)
        ar_mult_ul = math.exp(z_ul)

        # compose
        dl = base_dl * dl_fac * w_dl * day_vol_dl * ar_mult_dl
        ul = base_ul * ul_fac * w_ul * day_vol_ul * ar_mult_ul

        # spike?
        if spike_idx is not None and i == spike_idx:
            mult = rng.uniform(*spike_mult_range)
            dl *= mult; ul *= mult

        # cap theo băng thông
        dl = _cap_by_bw(dl, bw_dl, cap_dl_pct, seconds)
        ul = _cap_by_bw(ul, bw_ul, cap_ul_pct, seconds)

        # outage
        if outage is not None:
            s_o, l_o = outage
            if s_o <= i < s_o + l_o:
                dl = 0.0; ul = 0.0

        # optional ids
        session_id = f"SESS{xxhash.xxh64_intdigest(f'{service_id}|{cur.isoformat()}')%10_000_000:07d}" if rng.random() < session_p else None
        device_mac = rng.choice(mac_pool) if (mac_pool and rng.random() < mac_p) else None
        device_id  = rng.choice(dev_pool) if (dev_pool and rng.random() < dev_p) else None

        # jitter timestamp
        ts_out = _apply_jitter(cur, rng, jitter_max)
        rows.append({
            "service_id": service_id,
            "customer_id": customer_id,
            "timestamp": ts_out.isoformat(),
            "date": ts_out.date().isoformat(),
            "hour": ts_out.hour,
            "uplink_bytes": int(ul),
            "downlink_bytes": int(dl),
            "device_mac": device_mac,
            "device_id": device_id,
            "session_id": session_id,
        })

    return pd.DataFrame(rows)


def gen_usage(cfg: dict, keys_csv: str, profile_csv: str|None, start_date: str, end_date: str,
              out_dir: str, seed: str, upload: bool):
    # keys (1 row / service_id)
    keys = (pd.read_csv(keys_csv, usecols=["customer_id","service_id"])
              .groupby("service_id", as_index=False).first())

    # join bandwidth từ profile (nếu có)
    if profile_csv and os.path.exists(profile_csv):
        prof = pd.read_parquet(profile_csv) if profile_csv.endswith(".parquet") else pd.read_csv(profile_csv)
        prof = prof[["service_id","bandwidth_dl","bandwidth_ul"]]
        keys = keys.merge(prof, on="service_id", how="left")

    total_services = len(keys)

    # cfg shortcuts
    scale   = cfg["scale"]; model = cfg["model"]; ids = cfg["id_fields"]; outopt = cfg["output"]
    dp_map  = _build_daypart_map(model); weekend = model.get("weekend_boost", {"dl":1.0,"ul":1.0})
    cap_dl  = float(model["bw_utilization_cap"]["dl_pct"]); cap_ul = float(model["bw_utilization_cap"]["ul_pct"])
    ul_mu   = float(model["ul_lognorm"]["mu"]); ul_sigma = float(model["ul_lognorm"]["sigma"])
    dl_mu   = float(model["dl_lognorm"]["mu"]); dl_sigma = float(model["dl_lognorm"]["sigma"])
    cam_boost = float(model.get("camera_ul_night_boost", 1.0))
    stream_boost = float(model.get("evening_stream_boost", 1.0))
    spike_prob = float(model.get("spike_per_day_prob", 0.0))
    spike_range = tuple(model.get("spike_multiplier_range", [2.5,6.0]))
    outage_prob = float(model.get("outage_per_day_prob", 0.0))
    out_min_min, out_min_max = model.get("outage_minutes_range", [10,90])
    out_h_min, out_h_max = model.get("outage_hours_range", [1,4])

    # volatility & tails
    ht_cfg = model.get("heavy_tail", {"prob":0.0})
    ar_phi_lo, ar_phi_hi = model.get("ar1", {}).get("phi_range", [0.6,0.9])
    ar_shock_sigma = float(model.get("ar1", {}).get("shock_sigma", 0.5))
    daily_vol_sigma = float(model.get("ar1", {}).get("daily_vol_sigma", 0.3))
    jitter_max = int(model.get("jitter_seconds_max", 0))
    rand_drop = float(model.get("random_drop_pct_per_day", 0.0))

    sess_p = float(ids.get("session_presence_rate", 0.35))
    mac_p  = float(ids.get("device_mac_presence_rate", 0.55))
    dev_p  = float(ids.get("device_id_presence_rate", 0.35))
    mac_max = int(ids.get("device_mac_per_service_max", 6))
    dev_max = int(ids.get("device_id_per_service_max", 3))

    minute_ratio_max = float(scale.get("minute_cohort_max_ratio", 0.60))
    rows_target = int(scale.get("rows_per_day_target", 1_000_000))
    force_gran = scale.get("force_granularity", None)  # "5min"|"hour"|None

    # per-service id pools
    mac_pools, dev_pools = {}, {}
    for sid in keys["service_id"]:
        macs, devs = _make_ids_pool(seed, sid, mac_max, dev_max)
        mac_pools[sid] = macs; dev_pools[sid] = devs

    # date range
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date, "%Y-%m-%d").date()

    for d in _daterange(start, end):
        # cohort sizing
        if force_gran == "5min":
            m_cnt, h_cnt = total_services, 0
        elif force_gran == "hour":
            m_cnt, h_cnt = 0, total_services
        else:
            m_cnt, h_cnt = _choose_granularity_5min(total_services, rows_target, minute_ratio_max)

        # choose 5min cohort indices
        idx = np.arange(total_services)
        rs  = np.random.default_rng(xxhash.xxh64_intdigest(f"{seed}|choice|{d.isoformat()}"))
        rs.shuffle(idx)
        five_set = set(idx[:m_cnt])

        # generate per service
        day_chunks = []
        for pos, r in enumerate(keys.itertuples(index=False)):
            sid, cid = r.service_id, r.customer_id
            bw_dl = None if (not hasattr(r,'bandwidth_dl') or pd.isna(r.bandwidth_dl)) else int(r.bandwidth_dl)
            bw_ul = None if (not hasattr(r,'bandwidth_ul') or pd.isna(r.bandwidth_ul)) else int(r.bandwidth_ul)

            gran = "5min" if (pos in five_set) else "hour"
            rng_row = _rng(f"{seed}|{sid}|{d.isoformat()}|{gran}")

            # spike & outage
            spike_idx = None
            if rng_row.random() < spike_prob:
                spike_idx = rng_row.randint(0, 287 if gran=="5min" else 23)
            outage = None
            if rng_row.random() < outage_prob:
                if gran == "5min":
                    # map phút -> bucket 5’
                    bmin = max(1, int(math.ceil(out_min_min / 5)))
                    bmax = max(bmin, int(math.ceil(out_min_max / 5)))
                    start_o = rng_row.randint(0, 287)
                    length  = min(287 - start_o, rng_row.randint(bmin, bmax))
                else:
                    start_o = rng_row.randint(0, 23)
                    length  = min(23 - start_o, rng_row.randint(out_h_min, out_h_max))
                outage = (start_o, length)

            # phi riêng cho service (AR1)
            phi_service = rng_row.uniform(ar_phi_lo, ar_phi_hi)

            df_one = _gen_for_service_day(
                rng=rng_row,
                service_id=sid, customer_id=cid, the_day=d, granularity=gran,
                dp_map=dp_map, weekend_boost=weekend,
                cam_ul_night_boost=cam_boost, evening_stream_boost=stream_boost,
                ul_mu=ul_mu, ul_sigma=ul_sigma, dl_mu=dl_mu, dl_sigma=dl_sigma,
                bw_dl=bw_dl, bw_ul=bw_ul, cap_dl_pct=cap_dl, cap_ul_pct=cap_ul,
                session_p=sess_p, mac_p=mac_p, dev_p=dev_p,
                mac_pool=mac_pools[sid], dev_pool=dev_pools[sid],
                outage=outage, spike_idx=spike_idx, spike_mult_range=spike_range,
                ht_cfg=ht_cfg,
                ar1_phi=phi_service, ar1_shock_sigma=ar_shock_sigma, daily_vol_sigma=daily_vol_sigma,
                jitter_max=jitter_max, random_drop_pct=rand_drop,
            )
            day_chunks.append(df_one)

        day_df = pd.concat(day_chunks, ignore_index=True)

        # --- write RAW (đúng schema) ---
        part = f"date={d.strftime('%Y%m%d')}"
        out_raw_dir = os.path.join(out_dir, "usage_raw_5min_hour", part)
        csv_fp = save_csv(day_df, out_raw_dir, "usage_raw.csv", index=False)
        pq_dir = save_parquet(day_df.assign(date_part=d.strftime('%Y%m%d')),
                              os.path.join(out_dir, "usage_raw_5min_hour"),
                              "usage_raw.parquet", partition_cols=["date_part"])
        print(f"[OK] RAW {d} -> rows={len(day_df):,}")

        # --- aggregates: HOURLY ---
        if cfg["output"].get("write_hourly", True):
            hourly = (day_df.groupby(["service_id","customer_id","date","hour"], as_index=False)
                          [["uplink_bytes","downlink_bytes"]].sum())
            out_h_dir = os.path.join(out_dir, "usage_hourly", part)
            save_csv(hourly, out_h_dir, "usage_hourly.csv", index=False)
            save_parquet(hourly.assign(date_part=d.strftime('%Y%m%d')),
                         os.path.join(out_dir, "usage_hourly"),
                         "usage_hourly.parquet", partition_cols=["date_part"])

        # upload nếu bật
        if upload:
            upload_file(csv_fp, f"usage/usage_raw_5min_hour/{part}/usage_raw.csv", content_type="text/csv")
            upload_folder(os.path.join(out_dir, "usage_raw_5min_hour", "usage_raw.parquet"), "usage/usage_raw_5min_hour")
            if cfg["output"].get("write_hourly", True):
                upload_folder(os.path.join(out_dir, "usage_hourly", "usage_hourly.parquet"), "usage/usage_hourly")


def main():
    ap = argparse.ArgumentParser(description="Generate Usage Logs (5-min & hourly mix) with heavy variance; RAW + hourly aggregates")
    ap.add_argument("--config", default="data_gen/usage/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--profile_csv", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    keys_csv = args.keys_csv or cfg["paths"]["keys_csv"]
    profile_csv = args.profile_csv or cfg["paths"].get("profile_csv")
    start = args.start or cfg["date_range"]["start"]
    end   = args.end   or cfg["date_range"]["end"]
    out_dir = args.out or cfg["paths"]["out_dir"]
    seed = args.seed or cfg.get("seed", "usage-5min-2025")
    upload = args.upload or bool(cfg["paths"].get("upload_to_minio", False))

    os.makedirs(out_dir, exist_ok=True)
    gen_usage(cfg, keys_csv, profile_csv, start, end, out_dir, seed, upload)

if __name__ == "__main__":
    main()
