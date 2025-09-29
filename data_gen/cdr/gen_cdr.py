from __future__ import annotations
import argparse, os, math, random
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import yaml, xxhash

from data_gen.common.io_utils import save_csv, save_parquet
from data_gen.common.minio_utils import upload_file, upload_folder


def _rng(seed_text: str) -> random.Random:
    return random.Random(xxhash.xxh64_intdigest(seed_text))

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def _poisson_stable(lam: float, seed_text: str) -> int:
    if lam <= 0: return 0
    seed_int = xxhash.xxh64_intdigest(seed_text) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(seed_int)
    return int(rng.poisson(lam))

def _weighted_choice(rng: random.Random, items: List[Tuple[Any, float]]):
    tot = sum(w for _, w in items)
    r = rng.uniform(0, tot); c = 0.0
    for v, w in items:
        c += w
        if r <= c: return v
    return items[-1][0]

def _e164_from_seed(seed_text: str) -> str:
    # +84 + 9~10 digits (deterministic)
    d = xxhash.xxh64_intdigest(seed_text) % 10_000_000_000
    return "+84" + f"{d:010d}"[-9:]  # 9–10 tuỳ nhà mạng (ở đây lấy 9 số cuối)

def _cdr_id(seed_text: str) -> str:
    return f"CDR{xxhash.xxh64_intdigest(seed_text)%10_000_000:07d}"


def _load_keys(keys_csv: str) -> pd.DataFrame:
    # keys_csv phải có customer_id, service_id
    df = pd.read_csv(keys_csv, usecols=["customer_id","service_id"])
    # 1 hàng / service_id
    return df.groupby("service_id", as_index=False).first()

def _load_crm(crm_csv: str | None) -> pd.DataFrame | None:
    if not crm_csv or not os.path.exists(crm_csv): return None
    df = pd.read_csv(crm_csv)
    cols = [c for c in df.columns if c in ("customer_id","service_id","primary_msisdn","msisdn_list","province","district","ward","address")]
    return df[cols]

def _attach_msisdn(keys: pd.DataFrame, crm: pd.DataFrame | None, use_crm_rate: float) -> pd.DataFrame:
    out = keys.copy()
    if crm is None:
        out["msisdn"] = out.apply(lambda r: _e164_from_seed(f"{r.service_id}|{r.customer_id}"), axis=1)
        return out

    crm_small = crm.copy()
    # chọn msisdn: 90% primary_msisdn nếu có, 10% lấy ngẫu nhiên từ msisdn_list (nếu có)
    def pick_msisdn(row) -> str:
        rng = _rng(f"msisdn|{row.get('service_id','')}|{row.get('customer_id','')}")
        primary = row.get("primary_msisdn", None)
        lst = row.get("msisdn_list", None)
        if pd.isna(primary) and (pd.isna(lst) or not str(lst).strip()):
            return _e164_from_seed(f"{row.get('service_id')}|{row.get('customer_id')}")
        use_primary = rng.random() < use_crm_rate
        if use_primary and pd.notna(primary) and str(primary).strip():
            return str(primary).strip()
        # từ msisdn_list dạng "num1;num2"
        if pd.notna(lst) and str(lst).strip():
            cand = [s.strip() for s in str(lst).replace(",", ";").split(";") if s.strip()]
            if cand:
                return rng.choice(cand)
        # fallback
        return _e164_from_seed(f"{row.get('service_id')}|{row.get('customer_id')}")

    crm_small["msisdn"] = crm_small.apply(pick_msisdn, axis=1)
    # ưu tiên match theo service_id, fallback theo customer_id nếu không có
    out = out.merge(crm_small[["service_id","msisdn","province","district","ward"]], on="service_id", how="left")
    # nếu service_id không có trong CRM, cố gắng join theo customer_id
    miss = out["msisdn"].isna()
    if miss.any():
        crm_by_cust = crm_small.groupby("customer_id", as_index=False).first()[["customer_id","msisdn","province","district","ward"]]
        out = out.merge(crm_by_cust, on="customer_id", how="left", suffixes=("","_bycust"))
        out.loc[miss, "msisdn"]  = out.loc[miss, "msisdn_bycust"]
        out.loc[miss, "province"]= out.loc[miss, "province_bycust"]
        out.loc[miss, "district"]= out.loc[miss, "district_bycust"]
        out.loc[miss, "ward"]    = out.loc[miss, "ward_bycust"]
        out = out.drop(columns=[c for c in out.columns if c.endswith("_bycust")], errors="ignore")
    # fill msisdn còn thiếu
    out["msisdn"] = out.apply(lambda r: _e164_from_seed(f"{r.service_id}|{r.customer_id}") if pd.isna(r["msisdn"]) else r["msisdn"], axis=1)
    return out


def _build_topology(cfg: dict, seed: str):
    rng = _rng(f"{seed}|topology")
    cities_cfg = cfg["topology"]["cities"]
    city_weights = cfg["topology"].get("city_weights", {c["name"]: 1.0 for c in cities_cfg})
    # flatten weights
    items = []
    for c in cities_cfg:
        items.append((c, float(city_weights.get(c["name"], 1.0))))
    # build dict: city_name -> (clusters: [ {cluster_id, lac, cells: [cell_ids...]} ])
    topo = {}
    for c in cities_cfg:
        name = c["name"]; lac_prefix = int(c["lac_prefix"])
        n_clusters = int(c["clusters"]); a,b = c["cells_per_cluster"]
        clusters = []
        for k in range(n_clusters):
            cluster_id = f"{name[:3].upper()}_{k:04d}"
            lac = lac_prefix*100 + (k % 100)   # đơn giản: lac_prefixxx
            n_cells = rng.randint(int(a), int(b))
            cells = [f"{cluster_id}_C{j:03d}" for j in range(n_cells)]
            clusters.append({"cluster_id": cluster_id, "lac": lac, "cells": cells})
        topo[name] = clusters
    # sampler city by weights
    totalw = sum(w for _,w in items)
    def sample_city(rr: random.Random):
        r = rr.uniform(0,totalw); cum=0.0
        for it,w in items:
            cum += w
            if r<=cum: return it["name"]
        return items[-1][0]["name"]
    return topo, sample_city

def _choose_home_work_cluster(rng: random.Random, topo: dict, city: str) -> Tuple[dict, dict]:
    clusters = topo[city]
    home = rng.choice(clusters)
    # work cluster: có thể trùng hoặc khác; 70% khác cụm
    if rng.random() < 0.7 and len(clusters) > 1:
        work = rng.choice([c for c in clusters if c["cluster_id"] != home["cluster_id"]])
    else:
        work = home
    return home, work


def _pick_cell(rng: random.Random, cluster: dict) -> Tuple[str,int,str]:
    cell_id = rng.choice(cluster["cells"])
    lac = cluster["lac"]
    return cell_id, lac, cluster["cluster_id"]

def _duration_seconds(rng: random.Random, etype: str, dur_cfg: dict) -> int:
    lo, hi = dur_cfg.get("clip",[1,3600])
    if etype == "voice":
        mu, sigma = dur_cfg["voice_lognorm"]["mu"], dur_cfg["voice_lognorm"]["sigma"]
    elif etype == "data":
        mu, sigma = dur_cfg["data_lognorm"]["mu"], dur_cfg["data_lognorm"]["sigma"]
    else:
        return int(dur_cfg.get("sms_fixed_seconds", 2))
    v = math.exp(rng.gauss(mu, sigma))
    return int(max(lo, min(hi, v)))

def _mk_imei(seed: str) -> str:
    # 15-digit pseudo IMEI (Luhn not enforced)
    n = xxhash.xxh64_intdigest(seed) % 10**14
    return f"{n:014d}0"


def gen_cdr(cfg: dict, keys_csv: str, crm_csv: str | None, out_dir: str, seed: str, upload: bool):
    # load sources
    keys = _load_keys(keys_csv)
    crm  = _load_crm(crm_csv)
    subs = _attach_msisdn(keys, crm, float(cfg["ids"].get("use_crm_msisdn_rate", 0.9)))
    total_subs = len(subs)

    print(f"[INFO] total_subscribers≈{total_subs:,}")

    # topology
    topo, sample_city = _build_topology(cfg, seed)

    # persona
    mix = cfg["persona_mix"]["weights"]
    profs = cfg["persona_mix"]["profiles"]
    persona_items = [(name, float(w)) for name,w in mix.items()]

    # event volume
    vol = cfg["volume"]
    ev_per_1k = float(vol["events_per_1k_subs"])
    wdm = [float(x) for x in vol["weekday_multiplier"]]
    hourw = {int(h): float(w) for h,w in vol["hour_weights"].items()}
    etype_w = vol["event_type_weights"]
    etype_items = [(k,float(v)) for k,v in etype_w.items()]
    dur_cfg = cfg["volume"]["duration"]

    # ids config
    imei_rate = float(cfg["ids"]["imei_presence_rate"])
    imei_chg_week = float(cfg["ids"]["imei_change_prob_per_week"])
    miss = cfg.get("missingness", {})

    # assign per-subscriber persona, home/work cluster, city, imei
    persona_map = {}
    home_map = {}
    work_map = {}
    city_map = {}
    imei_map = {}
    rng_assign = _rng(f"{seed}|assign")
    for r in subs.itertuples(index=False):
        sid = r.service_id; msisdn = r.msisdn
        rr = _rng(f"{seed}|{sid}|{msisdn}|persona")
        persona = _weighted_choice(rr, persona_items)
        city = None
        # nếu CRM có province => map city đơn giản theo chuỗi
        if hasattr(r, "province") and pd.notna(r.province):
            p = str(r.province).lower()
            if "hồ chí minh" in p or "ho chi minh" in p or "hcm" in p: city = "HoChiMinh"
            elif "đà nẵng" in p or "da nang" in p: city = "DaNang"
            else: city = "HaNoi"
        else:
            city = sample_city(rr)
        home, work = _choose_home_work_cluster(rr, topo, city)
        imei = _mk_imei(f"{seed}|{msisdn}|{persona}")

        key = r.service_id
        persona_map[key] = persona
        home_map[key] = home
        work_map[key] = work
        city_map[key] = city
        imei_map[key] = imei

    # date loop
    d0 = datetime.strptime(cfg["date_range"]["start"], "%Y-%m-%d").date()
    d1 = datetime.strptime(cfg["date_range"]["end"], "%Y-%m-%d").date()

    for d in _daterange(d0, d1):
        weekday = d.weekday()
        lam = (ev_per_1k/1000.0) * total_subs * wdm[weekday]
        # Poisson stable cho tổng event
        total_events = _poisson_stable(lam, f"{seed}|{d.isoformat()}|n_events")
        print(f"[INFO] {d} target_events≈{int(lam):,}, sampled={total_events:,}")

        # build per-hour categorical sampling
        hours, weights = zip(*sorted(hourw.items()))
        weights = np.array(weights, dtype=float); weights = weights/weights.sum()

        rows = []
        # sample service indices for each event (uniform over subscribers)
        # để tái lập: seed theo ngày
        rng_np = np.random.default_rng(xxhash.xxh64_intdigest(f"{seed}|pick|{d.isoformat()}"))
        svc_idx = rng_np.integers(low=0, high=total_subs, size=total_events)

        subs_arr = subs[["service_id","customer_id","msisdn"]].to_numpy()

        for i, idx in enumerate(svc_idx):
            service_id, customer_id, msisdn = subs_arr[idx]
            persona = persona_map[service_id]
            home = home_map[service_id]; work = work_map[service_id]

            rr = _rng(f"{seed}|{service_id}|{d.isoformat()}|{i}")
            # chọn hour theo weights, rồi minute/second uniform
            hour = int(np.random.choice(hours, p=weights))
            minute = rr.randint(0,59); second = rr.randint(0,59)
            ts = datetime(d.year, d.month, d.day, hour, minute, second)

            # quyết định vị trí (home/work/transit/field) theo persona & giờ
            prof = profs[persona]
            wh = prof["work_hours"]
            # cửa sổ làm việc có thể qua đêm (night_shift)
            in_work = False
            if wh[0] <= wh[1]:
                in_work = (hour >= int(wh[0]) and hour < int(wh[1]))
            else:
                # qua ngày: 22..24 or 0..6
                in_work = (hour >= int(wh[0]) or hour < int(wh[1]))

            # rời nhà?
            leave_p = prof["leave_home_prob_weekend"] if weekday>=5 else prof["leave_home_prob_weekday"]
            left_today = (rr.random() < leave_p)

            # số cụm di chuyển trong ngày
            min_mv, max_mv = prof["day_move_clusters"]
            n_mv = rr.randint(min_mv, max_mv) if left_today else 0

            # simple routing: nếu trong work_hours & left_today → work; nếu không & n_mv>0 → chọn ngẫu nhiên
            if in_work and left_today:
                cluster = work
            else:
                cluster = home
                if n_mv>0 and rr.random() < 0.35:
                    # ghé cụm khác trong cùng city
                    rng_city = _rng(f"{seed}|{city_map[service_id]}|{service_id}|mv")
                    cluster = rng_city.choice([c for c in topo[city_map[service_id]] if c["cluster_id"]!=home["cluster_id"]])

            cell_id, lac, cluster_id = _pick_cell(rr, cluster)
            # miss probs
            if rr.random() < float(cfg["missingness"].get("missing_cell_prob", 0.005)):
                cell_id = None
            if rr.random() < float(cfg["missingness"].get("missing_lac_prob", 0.01)):
                lac = None

            # event type
            etype = _weighted_choice(rr, [(k,float(v)) for k,v in etype_w.items()])

            # duration
            durs = _duration_seconds(rr, etype, dur_cfg)

            # imei present?
            imei_out = None
            if rr.random() < float(imei_rate):
                # đổi máy theo tuần?
                week_id = int((d - d0).days // 7)
                change = (rr.random() < float(imei_chg_week))
                imei_seed = f"{seed}|{service_id}|{week_id if not change else (week_id+1)}"
                imei_curr = _mk_imei(imei_seed)
                if rr.random() > float(cfg["missingness"].get("missing_imei_if_present_prob", 0.05)):
                    imei_out = imei_curr

            rows.append({
                "cdr_id": _cdr_id(f"{service_id}|{msisdn}|{ts.isoformat()}|{i}"),
                "msisdn": msisdn,
                "timestamp": ts.isoformat(),
                "cell_id": cell_id,
                "cell_cluster_id": cluster_id,
                "location_area_code": None if lac is None else int(lac),
                "event_type": etype,      # data | voice | sms
                "duration": int(durs),
                "imei": imei_out
            })

        out_df = pd.DataFrame(rows)

        # --- write partition by date ---
        part = f"date={d.strftime('%Y%m%d')}"
        out_day_dir = os.path.join(cfg["paths"]["out_dir"], "cdr_raw", part)
        csv_fp = save_csv(out_df, out_day_dir, "cdr_raw.csv", index=False)
        pq_dir = save_parquet(out_df.assign(date_part=d.strftime('%Y%m%d')),
                              os.path.join(cfg["paths"]["out_dir"], "cdr_raw"),
                              "cdr_raw.parquet", partition_cols=["date_part"])

        print(f"[OK] {d} events={len(out_df):,} -> {csv_fp}")

        if cfg["paths"].get("upload_to_minio", False):
            upload_file(csv_fp, f"cdr/cdr_raw/{part}/cdr_raw.csv", content_type="text/csv")
            upload_folder(os.path.join(cfg["paths"]["out_dir"], "cdr_raw", "cdr_raw.parquet"),
                          "cdr/cdr_raw")

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Generate Mobility/CDR with realistic mobility personas & cell topology")
    ap.add_argument("--config", default="data_gen/cdr/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--crm_csv",  default=None)
    ap.add_argument("--start",    default=None)
    ap.add_argument("--end",      default=None)
    ap.add_argument("--seed",     default=None)
    ap.add_argument("--out",      default=None)
    ap.add_argument("--upload",   action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    if args.keys_csv: cfg["paths"]["keys_csv"] = args.keys_csv
    if args.crm_csv is not None: cfg["paths"]["crm_csv"] = args.crm_csv
    if args.start: cfg["date_range"]["start"] = args.start
    if args.end:   cfg["date_range"]["end"]   = args.end
    if args.out:   cfg["paths"]["out_dir"]    = args.out
    if args.seed:  cfg["seed"] = args.seed
    if args.upload: cfg["paths"]["upload_to_minio"] = True

    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
    gen_cdr(cfg, cfg["paths"]["keys_csv"], cfg["paths"].get("crm_csv"),
            cfg["paths"]["out_dir"], cfg.get("seed","cdr-2025"),
            bool(cfg["paths"].get("upload_to_minio", False)))

if __name__ == "__main__":
    main()
