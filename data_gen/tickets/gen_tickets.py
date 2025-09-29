from __future__ import annotations
import argparse, os, random, math
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

def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm (lam moderate)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1

def _weighted_choice(rng: random.Random, items: List[Dict[str, Any]], key="weight") -> Dict[str, Any]:
    ws = [float(x.get(key, 1.0)) for x in items]
    total = sum(ws) or 1.0
    r = rng.uniform(0, total); cum = 0.0
    for it, w in zip(items, ws):
        cum += w
        if r <= cum:
            return it
    return items[-1]

def _ticket_id(seed_text: str) -> str:
    return f"TCK{xxhash.xxh64_intdigest(seed_text)%10_000_000:07d}"

def _choose_hour(rng: random.Random, hour_weights: Dict[str, float]) -> int:
    items = [(int(h), float(w)) for h, w in hour_weights.items()]
    total = sum(w for _, w in items)
    r = rng.uniform(0, total); cum = 0.0
    for h, w in items:
        cum += w
        if r <= cum: return h
    return items[-1][0]

def _lognorm_hours(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))


def _load_keys(keys_csv: str) -> pd.DataFrame:
    df = pd.read_csv(keys_csv, usecols=["customer_id","service_id"])
    # 1 hàng / service_id
    return df.groupby("service_id", as_index=False).first()

def _load_profile(profile_csv: str | None) -> pd.DataFrame | None:
    if not profile_csv or not os.path.exists(profile_csv): return None
    if profile_csv.endswith(".parquet"):
        df = pd.read_parquet(profile_csv)
    else:
        df = pd.read_csv(profile_csv)
    return df[["service_id","cpe_old_device_flag"]]

def _load_subs(subs_csv: str | None) -> pd.DataFrame | None:
    if not subs_csv or not os.path.exists(subs_csv): return None
    df = pd.read_csv(subs_csv, usecols=["service_id","multi_site_flag"])
    # chọn kỳ active gần snapshot? — ở đây chỉ cần flag tổng quát; lấy max theo service
    df = df.groupby("service_id", as_index=False)["multi_site_flag"].max()
    return df

def _prepare_service_attrs(keys: pd.DataFrame, prof: pd.DataFrame | None, subs: pd.DataFrame | None) -> pd.DataFrame:
    df = keys.copy()
    if prof is not None:
        df = df.merge(prof, on="service_id", how="left")
    else:
        df["cpe_old_device_flag"] = 0
    if subs is not None:
        df = df.merge(subs, on="service_id", how="left")
    else:
        df["multi_site_flag"] = 0
    df["cpe_old_device_flag"] = df["cpe_old_device_flag"].fillna(0).astype(int)
    df["multi_site_flag"] = df["multi_site_flag"].fillna(0).astype(int)
    return df

def _topic_with_boost(rng: random.Random, topics: List[Dict[str, Any]], row: pd.Series, corr_cfg: dict) -> Dict[str, Any]:
    # áp boost có điều kiện theo profile/subs
    items = []
    for t in topics:
        w = float(t.get("weight", 1.0))
        name = t.get("name")
        if corr_cfg.get("use_profile", False) and name == "wifi_coverage" and int(row.get("cpe_old_device_flag", 0)) == 1:
            w *= float(corr_cfg.get("wifi_old_device_boost", 1.5))
        if corr_cfg.get("use_subscriptions", False) and name == "POS" and int(row.get("multi_site_flag", 0)) == 1:
            w *= float(corr_cfg.get("pos_multisite_boost", 1.6))
        items.append({**t, "weight": w})
    return _weighted_choice(rng, items, key="weight")

def _maybe(arr: List[str], p: float, rng: random.Random) -> List[str]:
    if rng.random() > p or not arr:
        return []
    # 1..min(3,len(arr)) keywords
    k = rng.randint(1, min(3, len(arr)))
    return rng.sample(arr, k)

def _desc(rng: random.Random, arr: List[str], p: float) -> str | None:
    if rng.random() > p or not arr: return None
    return rng.choice(arr)

def gen_tickets_from_keys(cfg: dict, keys_csv: str, out_dir: str, seed: str,
                          profile_csv: str | None, subs_csv: str | None, upload: bool):
    # load sources
    keys = _load_keys(keys_csv)
    prof = _load_profile(profile_csv)
    subs = _load_subs(subs_csv)
    svc = _prepare_service_attrs(keys, prof, subs)
    total_services = len(svc)

    # config shortcuts
    vol = cfg["volume"]
    topics_cfg = cfg["topics"]["catalog"]
    miss = cfg.get("missingness", {})
    base_unresolved = float(cfg.get("resolution", {}).get("unresolved_rate", 0.12))
    hour_weights = vol.get("peak_hours_weights", {str(h): 1/24 for h in range(24)})
    weekday_mults = vol.get("weekday_multiplier", [1.0]*7)
    corr = cfg.get("correlation", {})

    # date range
    d0 = datetime.strptime(cfg["date_range"]["start"], "%Y-%m-%d").date()
    d1 = datetime.strptime(cfg["date_range"]["end"], "%Y-%m-%d").date()

    # precompute service array for fast sampling
    svc_arr = svc[["service_id","customer_id","cpe_old_device_flag","multi_site_flag"]].to_numpy()

    for d in _daterange(d0, d1):
        rng_day = _rng(f"{seed}|{d.isoformat()}")
        weekday = d.weekday()
        lam = (vol["base_per_1k_services"] / 1000.0) * total_services * float(weekday_mults[weekday])
        n_tickets = _poisson(rng_day, lam)

        # prepare hour categorical sampling
        hours = [int(h) for h in hour_weights.keys()]
        weights = np.array([float(w) for w in hour_weights.values()], dtype=float)
        weights = weights / weights.sum()

        rows: List[dict] = []
        for i in range(n_tickets):
            # pick service uniformly
            idx = rng_day.randint(0, total_services - 1)
            service_id, customer_id, old_flag, multi_site = svc_arr[idx]
            rng_row = _rng(f"{seed}|{service_id}|{d.isoformat()}|{i}")

            # topic + conditional boosts
            topic = _topic_with_boost(rng_row, topics_cfg, pd.Series({
                "cpe_old_device_flag": old_flag,
                "multi_site_flag": multi_site
            }), corr)

            # create time (hour weighted) + minute/second uniform
            hour = rng_row.choices(hours, weights=weights, k=1)[0]
            minute = rng_row.randint(0, 59); second = rng_row.randint(0, 59)
            create_dt = datetime(d.year, d.month, d.day, hour, minute, second)

            # resolution
            topic_resolved_rate = float(topic.get("resolved_rate", 1.0))
            resolved_flag = 1 if (rng_row.random() < topic_resolved_rate * (1.0 - base_unresolved)) else 0

            close_time = None
            if resolved_flag == 1:
                mu = float(topic.get("sla_hours_lognorm", {}).get("mu", 1.5))
                sigma = float(topic.get("sla_hours_lognorm", {}).get("sigma", 0.6))
                dur_hours = _lognorm_hours(rng_row, mu, sigma)
                # 10% spill-over qua ngày sau để thực tế hơn
                if rng_row.random() < 0.10:
                    dur_hours += rng_row.uniform(1.0, 6.0)
                close_dt = create_dt + timedelta(hours=dur_hours)
                # missing close_time dù đã resolved
                if rng_row.random() > float(miss.get("missing_close_prob_if_resolved", 0.10)):
                    close_time = close_dt.isoformat()

            # site visit theo topic
            site_visit_flag = 1 if (rng_row.random() < float(topic.get("site_visit_rate", 0.05))) else 0

            # optional fields
            kw_prob = float(miss.get("keywords_presence_rate", 0.7))
            desc_prob = float(miss.get("description_presence_rate", 0.92))
            keywords = _maybe(topic.get("keywords", []), kw_prob, rng_row)
            description = _desc(rng_row, topic.get("descriptions", []), desc_prob)

            rows.append({
                "ticket_id": _ticket_id(f"{service_id}|{create_dt.isoformat()}|{i}"),
                "customer_id": customer_id,
                "service_id": service_id,
                "create_time": create_dt.isoformat(),
                "close_time": close_time,
                "topic_group": topic["name"],
                "keywords": ",".join(keywords) if keywords else None,
                "description": description,
                "site_visit_flag": site_visit_flag,
                "resolved_flag": resolved_flag,
            })

        out_df = pd.DataFrame(rows)

        # --- write partitions by date ---
        part = f"date={d.strftime('%Y%m%d')}"
        out_day_dir = os.path.join(cfg["paths"]["out_dir"], "tickets_raw", part)
        csv_fp = save_csv(out_df, out_day_dir, "tickets_raw.csv", index=False)
        pq_dir = save_parquet(out_df.assign(date_part=d.strftime('%Y%m%d')),
                              os.path.join(cfg["paths"]["out_dir"], "tickets_raw"),
                              "tickets_raw.parquet", partition_cols=["date_part"])

        print(f"[OK] {d} tickets={len(out_df):,} -> {csv_fp}")

        # upload (optional)
        if cfg["paths"].get("upload_to_minio", False):
            upload_file(csv_fp, f"tickets/tickets_raw/{part}/tickets_raw.csv", content_type="text/csv")
            upload_folder(os.path.join(cfg["paths"]["out_dir"], "tickets_raw", "tickets_raw.parquet"),
                          "tickets/tickets_raw")


def main():
    ap = argparse.ArgumentParser(description="Generate Tickets/Helpdesk RAW with realistic topic mix and missingness")
    ap.add_argument("--config", default="data_gen/tickets/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--profile_csv", default=None)
    ap.add_argument("--subs_csv", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    # override từ CLI nếu có
    if args.keys_csv:   cfg["paths"]["keys_csv"] = args.keys_csv
    if args.profile_csv is not None: cfg["paths"]["profile_csv"] = args.profile_csv
    if args.subs_csv is not None:    cfg["paths"]["subs_csv"] = args.subs_csv
    if args.start: cfg["date_range"]["start"] = args.start
    if args.end:   cfg["date_range"]["end"]   = args.end
    if args.out:   cfg["paths"]["out_dir"]    = args.out
    if args.seed:  cfg["seed"] = args.seed
    if args.upload: cfg["paths"]["upload_to_minio"] = True

    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)

    gen_tickets_from_keys(
        cfg=cfg,
        keys_csv=cfg["paths"]["keys_csv"],
        out_dir=cfg["paths"]["out_dir"],
        seed=cfg.get("seed", "tickets-2025"),
        profile_csv=cfg["paths"].get("profile_csv"),
        subs_csv=cfg["paths"].get("subs_csv"),
        upload=bool(cfg["paths"].get("upload_to_minio", False)),
    )

if __name__ == "__main__":
    main()
