from __future__ import annotations
import argparse, os, random, math, glob
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
    # Knuth
    L = math.exp(-lam)
    k = 0; p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _poisson_stable(lam: float, seed_text: str) -> int:
    """Poisson sampler ổn định cho λ lớn, dùng NumPy RNG với seed từ xxhash."""
    if lam <= 0:
        return 0
    seed_int = xxhash.xxh64_intdigest(seed_text) & 0xFFFFFFFFFFFFFFFF  # uint64
    rng = np.random.default_rng(seed_int)
    return int(rng.poisson(lam))


def _geometric_count(rng: random.Random, mean_val: float) -> int:
    # mean = (1-p)/p => p = 1/(mean+1)
    p = 1.0 / (mean_val + 1.0)
    # draw until first success
    k = 1
    while rng.random() > p:
        k += 1
    return k

def _weighted_choice(rng: random.Random, items: List[Tuple[Any, float]]):
    total = sum(w for _, w in items)
    r = rng.uniform(0, total); cum = 0.0
    for v, w in items:
        cum += w
        if r <= cum:
            return v
    return items[-1][0]

def _choose_hour(rng: random.Random, hour_weights: Dict[str, float]) -> int:
    items = [(int(h), float(w)) for h, w in hour_weights.items()]
    return _weighted_choice(rng, items)

def _lognorm_seconds(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))

def _heavy_tail_mix(rng: random.Random, base_val: float, cfg: dict) -> float:
    p = float(cfg.get("prob", 0.0))
    if rng.random() >= p: return base_val
    alpha = float(cfg.get("pareto_alpha", 1.8))
    scale = float(cfg.get("pareto_scale", 1.5))
    u = max(1e-12, rng.random())
    tail = scale * (u ** (-1.0 / alpha))
    return base_val * tail

def _ticket_files_for_day(tickets_dir: str, d: date) -> List[str]:
    # pattern: tickets_raw/date=YYYYMMDD/tickets_raw.csv|parquet
    p1 = os.path.join(tickets_dir, f"date={d.strftime('%Y%m%d')}", "tickets_raw.csv")
    p2 = os.path.join(tickets_dir, f"date={d.strftime('%Y%m%d')}", "tickets_raw.parquet")
    return [p for p in (p1, p2) if os.path.exists(p)]

def _load_tickets_window(tickets_dir: str, start: date, end: date) -> pd.DataFrame:
    frames = []
    for d in _daterange(start, end):
        for fp in _ticket_files_for_day(tickets_dir, d):
            if fp.endswith(".parquet"):
                frames.append(pd.read_parquet(fp))
            else:
                frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _mk_session_id(service_id: str, start_iso: str, seq: int) -> str:
    return f"SESS{xxhash.xxh64_intdigest(f'{service_id}|{start_iso}|{seq}')%10_000_000:07d}"

def _flatten_catalog(cfg: dict) -> Tuple[List[Tuple[Tuple[str, str], float]], List[Tuple[Tuple[str, str], float]]]:
    """
    returns:
      general_items: [((domain, topic), weight)]
      security_items: [((domain, topic), weight)]
    """
    gen = []
    for name, obj in cfg["topic_mix"]["general_domains"].items():
        w = float(obj.get("weight", 1.0))
        topic = obj.get("topic", name)
        for d in obj.get("domains", []):
            gen.append(((d, topic), w / max(1, len(obj.get("domains", [])))))

    sec = []
    for name, obj in cfg["topic_mix"]["security_domains"].items():
        w = float(obj.get("weight", 1.0))
        topic = obj.get("topic", name)
        for d in obj.get("domains", []):
            sec.append(((d, topic), w / max(1, len(obj.get("domains", [])))))

    return gen, sec

def _long_tail_domain(rng: random.Random, words: List[str], tlds: List[str]) -> str:
    n = rng.randint(2, 3)
    name = "-".join(rng.sample(words, n))
    return name + rng.choice(tlds)

def _load_keys(keys_csv: str) -> pd.DataFrame:
    df = pd.read_csv(keys_csv, usecols=["customer_id","service_id"])
    return df.groupby("service_id", as_index=False).first()

def _prepare_ticket_boosts(cfg: dict, d: date) -> Dict[str, Dict[str, float]]:
    """
    build map: service_id -> topic -> boost_weight
    dựa trên tickets trong lookback_days
    """
    tdir = cfg["paths"].get("tickets_dir")
    if not tdir:
        return {}
    lookback = int(cfg["correlation"].get("lookback_days", 7))
    start = d - timedelta(days=lookback)
    tickets = _load_tickets_window(tdir, start, d)
    if tickets.empty:
        return {}
    mapping = cfg["correlation"].get("mapping_ticket_to_topic", {})
    boost = float(cfg["correlation"].get("boost_factor", 1.8))

    out: Dict[str, Dict[str, float]] = {}
    for r in tickets.itertuples(index=False):
        sid = getattr(r, "service_id", None)
        tg  = getattr(r, "topic_group", None)
        if not sid or not tg: continue
        topic = mapping.get(str(tg), None)
        if not topic: continue
        out.setdefault(sid, {})
        out[sid][topic] = max(out[sid].get(topic, 1.0), boost)
    return out

def gen_browsing(cfg: dict, keys_csv: str, out_dir: str, seed: str, upload: bool):
    keys = _load_keys(keys_csv)
    total_services = len(keys)

    vol = cfg["volume"]
    idcfg = cfg["id_fields"]
    dur = cfg["duration_model"]
    mix = cfg["topic_mix"]
    long_tail = cfg["long_tail"]

    # precompute catalogs
    gen_items, sec_items = _flatten_catalog(cfg)
    mix_weights = mix["weights"]  # dict

    # convenience lists for sampling
    gen_pairs = gen_items
    sec_pairs = sec_items
    lt_words = long_tail.get("words", [])
    lt_tlds  = long_tail.get("tlds", [])
    lt_topic = long_tail.get("topic", "other")

    # date range
    d0 = datetime.strptime(cfg["date_range"]["start"], "%Y-%m-%d").date()
    d1 = datetime.strptime(cfg["date_range"]["end"], "%Y-%m-%d").date()

    # prepare array for fast service sampling
    svc_arr = keys[["service_id","customer_id"]].to_numpy()

    for d in _daterange(d0, d1):
        rng_day = _rng(f"{seed}|{d.isoformat()}")

        # ticket-based boosts
        boosts = _prepare_ticket_boosts(cfg, d)  # sid -> {topic: boost}

        # sessions/day via Poisson
        weekday = d.weekday()
        lam = (vol["sessions_per_1k_services"] / 1000.0) * total_services * float(vol["weekday_multiplier"][weekday])
        n_sessions = _poisson_stable(lam, f"{seed}|{d.isoformat()}|sessions")


        hour_weights = vol["start_hour_weights"]
        intr_lo, intr_hi = vol.get("inter_event_seconds", [5, 120])

        rows: List[dict] = []

        for sidx in range(n_sessions):
            # choose a service
            i = rng_day.randint(0, total_services - 1)
            service_id, customer_id = svc_arr[i]
            rng_sess = _rng(f"{seed}|{service_id}|{d.isoformat()}|sess{sidx}")

            hour = _choose_hour(rng_sess, hour_weights)
            minute = rng_sess.randint(0, 59); second = rng_sess.randint(0, 59)
            start_dt = datetime(d.year, d.month, d.day, hour, minute, second)
            session_id = _mk_session_id(service_id, start_dt.isoformat(), sidx) if (rng_sess.random() < idcfg.get("session_presence_rate", 0.9)) else None

            # events in session
            events_n = max(1, _geometric_count(rng_sess, float(vol.get("events_per_session_mean", 5.0))))
            cur_ts = start_dt

            # build topic-bucket choice with boosts
            w_gen = float(mix_weights.get("general_web", 0.7))
            w_lt  = float(mix_weights.get("long_tail", 0.25))
            w_sec = float(mix_weights.get("security_cam_family", 0.05))

            # apply ticket boost if any
            if service_id in boosts:
                # if a mapped topic exists, push some mass to security/tech/shopping accordingly by slightly scaling groups
                # Simple strategy: if any security-like topic in boosts, increase w_sec; if shopping/tech, bump general a bit
                if any(t in ("camera","CCTV","doorbell","alarm","smart_home") for t in boosts[service_id].keys()):
                    w_sec *= max(boosts[service_id].values())
                if any(t in ("shopping","tech") for t in boosts[service_id].keys()):
                    w_gen *= max(boosts[service_id].values())
                # re-normalize softly
                total = w_gen + w_lt + w_sec
                w_gen, w_lt, w_sec = w_gen/total, w_lt/total, w_sec/total

            for e in range(events_n):
                # pick bucket
                bucket = _weighted_choice(rng_sess, [("general", w_gen), ("longtail", w_lt), ("security", w_sec)])

                if bucket == "general" and gen_pairs:
                    (domain, topic), _ = _weighted_choice(rng_sess, gen_pairs), None
                    # _weighted_choice above returns (val,weight); we just passed pairs with weights embedded → fix:
                    # redefine helper for pairs with embedded weight:
                # Workaround: sample manually for gen_pairs
                if bucket == "general":
                    total_w = sum(w for (_, _topic), w in gen_pairs)
                    r = rng_sess.uniform(0, total_w); cum = 0.0
                    domain, topic = None, None
                    for (d_pair, t_pair), w in gen_pairs:
                        cum += w
                        if r <= cum:
                            domain, topic = d_pair, t_pair
                            break

                elif bucket == "security" and sec_pairs:
                    total_w = sum(w for (_, _topic), w in sec_pairs)
                    r = rng_sess.uniform(0, total_w); cum = 0.0
                    domain, topic = None, None
                    for (d_pair, t_pair), w in sec_pairs:
                        cum += w
                        if r <= cum:
                            domain, topic = d_pair, t_pair
                            break
                else:
                    domain = _long_tail_domain(rng_sess, lt_words, lt_tlds)
                    topic = lt_topic

                # duration
                dur_mu = float(dur["lognorm"]["mu"]); dur_sigma = float(dur["lognorm"]["sigma"])
                seconds = _lognorm_seconds(rng_sess, dur_mu, dur_sigma)
                seconds = _heavy_tail_mix(rng_sess, seconds, dur.get("heavy_tail", {}))
                lo, hi = dur.get("clip_seconds", [1, 1800])
                seconds = max(lo, min(hi, seconds))
                duration_val = int(seconds) if (rng_sess.random() < float(cfg["id_fields"].get("duration_presence_rate", 0.97))) else None

                rows.append({
                    "customer_id": customer_id,
                    "service_id": service_id,
                    "timestamp": cur_ts.isoformat(),
                    "domain": domain,
                    "topic": topic,
                    "session_id": session_id,
                    "duration": duration_val
                })

                # next event time
                gap = rng_sess.randint(int(intr_lo), int(intr_hi))
                cur_ts = cur_ts + timedelta(seconds=gap)

        out_df = pd.DataFrame(rows)

        # --- write partition by date ---
        part = f"date={d.strftime('%Y%m%d')}"
        out_day_dir = os.path.join(cfg["paths"]["out_dir"], "browsing_raw", part)
        csv_fp = save_csv(out_df, out_day_dir, "browsing_raw.csv", index=False)
        pq_dir = save_parquet(out_df.assign(date_part=d.strftime('%Y%m%d')),
                              os.path.join(cfg["paths"]["out_dir"], "browsing_raw"),
                              "browsing_raw.parquet", partition_cols=["date_part"])
        print(f"[OK] {d} sessions={n_sessions:,} events={len(out_df):,} -> {csv_fp}")

        if cfg["paths"].get("upload_to_minio", False):
            upload_file(csv_fp, f"browsing/browsing_raw/{part}/browsing_raw.csv", content_type="text/csv")
            upload_folder(os.path.join(cfg["paths"]["out_dir"], "browsing_raw", "browsing_raw.parquet"),
                          "browsing/browsing_raw")


def main():
    ap = argparse.ArgumentParser(description="Generate Browsing/App Topics (large-scale, session-based, long-tail domains)")
    ap.add_argument("--config", default="data_gen/browsing/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    if args.keys_csv: cfg["paths"]["keys_csv"] = args.keys_csv
    if args.start:    cfg["date_range"]["start"] = args.start
    if args.end:      cfg["date_range"]["end"]   = args.end
    if args.out:      cfg["paths"]["out_dir"]    = args.out
    if args.seed:     cfg["seed"] = args.seed
    if args.upload:   cfg["paths"]["upload_to_minio"] = True

    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)

    gen_browsing(
        cfg=cfg,
        keys_csv=cfg["paths"]["keys_csv"],
        out_dir=cfg["paths"]["out_dir"],
        seed=cfg.get("seed", "browsing-2025"),
        upload=bool(cfg["paths"].get("upload_to_minio", False)),
    )

if __name__ == "__main__":
    main()
