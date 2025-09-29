from __future__ import annotations
import argparse, os, random, json
from typing import List, Dict, Any
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import xxhash, yaml

from data_gen.common.io_utils import save_csv, save_parquet
from data_gen.common.minio_utils import upload_file, upload_folder


def _rng(seed_text: str) -> random.Random:
    return random.Random(xxhash.xxh64_intdigest(seed_text))

def _weighted_choice(rng: random.Random, items: List[Dict[str, Any]], key="weight") -> Dict[str, Any]:
    if not items: return {}
    ws = [float(it.get(key, 1.0)) for it in items]
    tot = sum(ws) or 1.0
    r = rng.uniform(0, tot); cum = 0.0
    for it, w in zip(items, ws):
        cum += w
        if r <= cum: return it
    return items[-1]

def _choose_from_hist(rng: random.Random, hist: Dict[str, float]) -> int:
    items = list(hist.items())
    tot = sum(w for _, w in items) or 1.0
    r = rng.uniform(0, tot); cum = 0.0
    for label, w in items:
        cum += w
        if r <= cum:
            lo, hi = [int(x) for x in label.split("-")]
            return rng.randint(lo, hi)
    lo, hi = [int(x) for x in items[-1][0].split("-")]
    return rng.randint(lo, hi)

def _gender_from_ratio(rng: random.Random, ratio: Dict[str, float]) -> str:
    p_m = float(ratio.get("M", 0.5))
    return "M" if rng.random() < p_m else "F"

def _household_size_from_probs(rng: random.Random, probs: Dict[str, float]) -> int:
    items = [(int(k), float(v)) for k, v in probs.items()]
    tot = sum(v for _, v in items) or 1.0
    r = rng.uniform(0, tot); cum = 0.0
    for k, v in items:
        cum += v
        if r <= cum: return k
    return items[-1][0]

def _e164_vn(seed_text: str) -> str:
    rnd = xxhash.xxh64_intdigest(seed_text)
    head = 90 + (rnd % 10)          # giả lập 09x
    body = (rnd // 10) % 10_000_000
    return f"+84{head}{body:07d}"

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _pick_location(rng: random.Random, geo_cfg: dict) -> tuple[str, str, str]:
    cities = geo_cfg.get("cities", [])
    city = _weighted_choice(rng, cities)
    city_name = city.get("name", "Thành phố")
    d_items = city.get("districts", [])
    dist = _weighted_choice(rng, d_items)
    dist_name = dist.get("name", "Quận/Huyện")
    wards = dist.get("wards", ["Phường 1"])
    ward_name = rng.choice(wards) if wards else "Phường 1"
    return city_name, dist_name, ward_name

def _pick_housing(rng: random.Random, housing_cfg: dict) -> tuple[str, int]:
    items = housing_cfg.get("types", [])
    if items and isinstance(items[0], str):
        weighted = [{"name": n, "weight": 1.0} for n in items]
    else:
        weighted = items
    chosen = _weighted_choice(rng, weighted) if weighted else {"name": "nhà_phố", "weight": 1.0}
    name = chosen.get("name", "nhà_phố")
    corner = 1 if name == "góc" else 0
    return name, corner


def gen_crm_from_keys(keys_csv: str, snapshot_month: str, cfg: dict, seed_override: str | None = None) -> pd.DataFrame:
    keys = pd.read_csv(keys_csv)  # customer_id, account_id, service_id
    first_per_customer = keys.groupby("customer_id", as_index=False).first()

    demo   = cfg.get("demographics", {})
    geo    = cfg.get("geography", {})
    housing_cfg = cfg.get("housing", {})
    msisdn_cfg  = cfg.get("msisdn", {})
    svc_cfg     = cfg.get("service", {})

    seed = seed_override or cfg.get("seed", "crm-2025")
    snap_dt = datetime.strptime(snapshot_month, "%Y-%m").date().replace(day=1)

    rows: List[dict] = []
    for _, r in first_per_customer.iterrows():
        customer_id = r["customer_id"]; account_id = r["account_id"]; service_id = r["service_id"]
        rng = _rng(f"{seed}|{customer_id}")

        # demographics
        age = _choose_from_hist(rng, demo.get("age_histogram", {"18-70": 1.0}))
        gender = _gender_from_ratio(rng, demo.get("gender_ratio", {"M":0.5,"F":0.5}))
        hh_size = _household_size_from_probs(rng, demo.get("household_size_probs", {"3":1.0}))
        member_ages = [rng.randint(int(demo.get("member_age_min", 1)), int(demo.get("member_age_max", 85))) for _ in range(hh_size)]
        comp = f"{hh_size}_members"
        change_back = rng.randint(0, int(demo.get("household_change_days_back_max", 365)))
        household_change_date = (snap_dt - relativedelta(days=change_back)).isoformat()

        # location
        province, district, ward = _pick_location(rng, geo)
        address_no = 10 + (xxhash.xxh64_intdigest(customer_id) % 90)
        address = f"Số {address_no}, {ward}, {district}, {province}"

        # housing
        housing_type, corner_house_flag = _pick_housing(rng, housing_cfg)

        # msisdn (có thể vắng mặt)
        ms_presence = rng.random() < float(msisdn_cfg.get("presence_rate", 1.0))
        if ms_presence:
            primary_msisdn = _e164_vn(customer_id)
            extra_cnt = rng.randint(0, int(msisdn_cfg.get("extra_per_customer_max", 2)))
            msisdn_list = [_e164_vn(f"{customer_id}|{k}") for k in range(extra_cnt)]
        else:
            primary_msisdn = None
            msisdn_list = []

        # segment & service start
        segments = demo.get("segments", ["mass","value","premium"])
        segment = segments[xxhash.xxh64_intdigest(customer_id+'s') % max(1, len(segments))]
        back_min = int(svc_cfg.get("service_start_days_back_min", 30))
        back_max = int(svc_cfg.get("service_start_days_back_max", 900))
        back_days = rng.randint(back_min, back_max)
        service_start_date = (snap_dt - relativedelta(days=back_days)).isoformat()

        # dob suy từ age
        today = date.today()
        dob = (today - relativedelta(years=age, months=rng.randint(0,11), days=rng.randint(0,27))).isoformat()

        rows.append({
            "customer_id": customer_id,
            "account_id": account_id,
            "service_id": service_id,  # primary_service_id trong raw
            "full_name": f"User {customer_id[-6:]}",
            "dob": dob,
            "age": age,
            "gender": gender,
            "household_composition": comp,
            "household_member_age": json.dumps(member_ages, ensure_ascii=False),
            "household_change_date": household_change_date,
            "address": address,
            "primary_msisdn": primary_msisdn,
            "msisdn_list": json.dumps(msisdn_list, ensure_ascii=False),
            "district": district,
            "ward": ward,
            "province": province,
            "housing_type": housing_type,
            "corner_house_flag": corner_house_flag,
            "segment": segment,
            "service_start_date": service_start_date,
            "snapshot_month": snap_dt.strftime("%Y%m"),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate CRM RAW from seed keys with realistic distributions")
    ap.add_argument("--config", default="data_gen/crm/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--month", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)

    keys_csv = args.keys_csv or cfg["paths"]["keys_csv"]
    month    = args.month or cfg.get("snapshot_month", "2025-05")
    out_dir  = args.out or cfg["paths"]["out_dir"]
    seed     = args.seed or cfg.get("seed", "crm-2025")
    do_upload = args.upload or bool(cfg["paths"].get("upload_to_minio", False))

    df = gen_crm_from_keys(keys_csv, month, cfg, seed_override=seed)

    part = f"month={month.replace('-','')}"
    local_dir = os.path.join(out_dir, part)
    csv_fp = save_csv(df, local_dir, "crm_raw.csv", index=False)
    pq_dir = save_parquet(df, out_dir, "crm_raw.parquet", partition_cols=["snapshot_month"])

    print(f"[OK] CSV  -> {csv_fp}")
    print(f"[OK] PARQ -> {pq_dir}")

    if do_upload:
        upload_file(csv_fp, f"crm/{part}/crm_raw.csv", content_type="text/csv")
        upload_folder(pq_dir, "crm")

if __name__ == "__main__":
    main()
