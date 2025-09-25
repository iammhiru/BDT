import os
import sys
import uuid
import hashlib
import random
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    import yaml
    import numpy as np
    import pandas as pd
    from faker import Faker
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install pyyaml numpy pandas faker pyarrow")
    sys.exit(1)

# -----------------------------
# Helpers
# -----------------------------

def sha1_row(d: Dict[str, Any], keys: List[str]) -> str:
    s = "|".join(str(d.get(k, "")) for k in keys)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ts_between(start: dt.datetime, end: dt.datetime) -> dt.datetime:
    delta = end - start
    secs = random.randint(0, max(1, int(delta.total_seconds())))
    return start + dt.timedelta(seconds=secs)


def pick_weighted(items: List[Any], weights: List[float]):
    """
    Nhận vào một danh sách các mục và trọng số tương ứng, trả về một mục được chọn ngẫu nhiên dựa trên trọng số.
    items: Danh sách các mục để chọn.
    weights: Danh sách các trọng số tương ứng với mỗi mục.
    Trả về: Một mục được chọn ngẫu nhiên từ danh sách items dựa trên
    """
    return random.choices(items, weights=weights, k=1)[0]


# -----------------------------
# Core generator
# -----------------------------

@dataclass
class AdminUnit:
    city: str
    city_code: str
    district: str
    district_code: str
    ward: str
    ward_code: str


def build_address(fake: Faker, street_names: List[str], au: AdminUnit, housing_type: str, cfg: Dict[str, Any]) -> str:
    # All proper nouns pulled from YAML
    if housing_type == cfg['enums']['housing_types']['apartment']:
        apt_no_min, apt_no_max = cfg['corpora']['apartment_no_range']
        apt_no = random.randint(apt_no_min, apt_no_max)
        block = random.choice(cfg['corpora']['apartment_blocks'])
        street = random.choice(street_names)
        return f"Căn {apt_no}, Block {block}, Đường {street}, {au.ward}, {au.district}, {au.city}"
    elif housing_type == cfg['enums']['housing_types']['shophouse']:
        lot = random.randint(*cfg['corpora']['shophouse_lot_range'])
        kdt = random.choice(cfg['corpora']['urban_areas'])
        return f"Lô SH-{lot:02d}, {kdt}, {au.ward}, {au.district}, {au.city}"
    else:
        no = random.randint(*cfg['corpora']['street_no_range'])
        street = random.choice(street_names)
        return f"Số {no}, Đường {street}, {au.ward}, {au.district}, {au.city}"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # normalize weights
    def norm(d):
        s = sum(float(v) for v in d.values())
        return {k: float(v)/s for k, v in d.items()}
    cfg['distributions']['housing_type_mix']['global'] = norm(cfg['distributions']['housing_type_mix']['global'])
    return cfg


def seed_admin_units(cfg: Dict[str, Any]) -> List[AdminUnit]:
    aus: List[AdminUnit] = []
    for city in cfg["admin_units"]["cities"]:
        city_name = city["name"]
        city_code = city.get("code", city_name[:3].upper())
        for dist in city["districts"]:
            dname = dist["name"]
            dcode = dist.get("code", f"{city_code}-{dname[:3].upper()}")
            for ward in dist["wards"]:
                wname = ward["name"]
                wcode = ward.get("code", f"{dcode}-{wname[:3].upper()}")
                aus.append(AdminUnit(city_name, city_code, dname, dcode, wname, wcode))
    random.shuffle(aus)
    return aus


def sample_area_floors(housing_type: str, cfg: Dict[str, Any]):
    rng = cfg["distributions"]["area_floors"][housing_type]
    area = int(np.random.uniform(rng["area_min"], rng["area_max"]))
    fl_rng = rng.get("floors", [1, 5])
    floors = int(np.random.uniform(fl_rng[0], fl_rng[1]))
    return area, max(1, floors)


def sample_age_band(cfg: Dict[str, Any]):
    bands = list(cfg["distributions"]["age_band"].keys())
    weights = list(cfg["distributions"]["age_band"].values())
    return pick_weighted(bands, weights)


def sample_children_elderly(age_band: str, housing_type: str, cfg: Dict[str, Any]):
    rules = cfg["distributions"]["household_members"]
    rule = rules.get(age_band, {"children": [0, 1.0], "elderly": [0, 1.0]})
    def draw(vals_probs):
        vals, probs = vals_probs
        return int(np.random.choice(vals, p=probs))
    ch = draw(rule["children"]) if isinstance(rule["children"], list) or isinstance(rule["children"], tuple) else 0
    el = draw(rule["elderly"]) if isinstance(rule["elderly"], list) or isinstance(rule["elderly"], tuple) else 0
    # small housing_type effect (optional bump for certain types)
    if housing_type in (cfg['enums']['housing_types']['street_house'], cfg['enums']['housing_types']['townhouse'], cfg['enums']['housing_types']['apartment']) and ch == 0 and random.random() < 0.2:
        ch = 1
    return int(ch), int(el)


def sample_corner(housing_type: str, cfg: Dict[str, Any]):
    rate_map = cfg["distributions"]["corner_rate"]
    rate = rate_map.get(housing_type, 0.0)
    return random.random() < rate


def gen_ids(hh_idx: int) -> Dict[str, str]:
    household_id = f"HH-{hh_idx:06d}"
    account_id = f"ACC-{hh_idx:06d}"
    subscriber_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"
    return {"household_id": household_id, "account_id": account_id, "subscriber_id": subscriber_id}


def sample_multi_service(cfg: Dict[str, Any]):
    tiers = cfg["distributions"]["multi_service_count"]
    choices = [int(k) for k in tiers.keys()]
    weights = [float(v) for v in tiers.values()]
    return int(pick_weighted(choices, weights))


def to_parquet(df: pd.DataFrame, out_dir: str, dt_str: str, base_name: str):
    path = os.path.join(out_dir, f"dt={dt_str}")
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{base_name}_{dt_str}.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    return file_path


def to_csv(df: pd.DataFrame, out_dir: str, dt_str: str, base_name: str):
    path = os.path.join(out_dir, f"dt={dt_str}")
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{base_name}_{dt_str}.csv")
    df.to_csv(file_path, index=False)
    return file_path

def gen_extras(fake: Faker, cfg: Dict[str, Any], housing_type: str) -> Dict[str, Any]:
    email = fake.email()
    gender = random.choice(cfg['enums']['gender'])
    marital_status = random.choice(cfg['enums']['marital_status'])
    id_card = f"{random.randint(*cfg['corpora']['id_card_range'])}"
    tax_code = f"{random.randint(*cfg['corpora']['tax_code_range'])}"
    building_name = None
    apartment_block = None
    if housing_type == cfg['enums']['housing_types']['apartment']:
        building_name = random.choice(cfg['corpora']['apartment_buildings'])
        apartment_block = random.choice(cfg['corpora']['apartment_blocks'])
    move_in_date = (dt.date.today() - dt.timedelta(days=random.randint(*cfg['corpora']['move_in_days_range']))).isoformat()
    marketing_opt_in = random.choice([True, False])
    note = random.choice(cfg['corpora']['notes'])
    lat_rng = cfg['corpora']['lat_range']
    lon_rng = cfg['corpora']['lon_range']
    lat = round(random.uniform(*lat_rng), 6)
    lon = round(random.uniform(*lon_rng), 6)
    return {
        "email": email,
        "gender": gender,
        "marital_status": marital_status,
        "id_card": id_card,
        "tax_code": tax_code,
        "building_name": building_name,
        "apartment_block": apartment_block,
        "move_in_date": move_in_date,
        "marketing_opt_in": marketing_opt_in,
        "latitude": lat,
        "longitude": lon,
        "note": note,
    }


def inject_string_noise(val: Any, cfg_noise: Dict[str, Any]) -> Any:
    if not isinstance(val, str):
        return val
    if random.random() < cfg_noise.get("whitespace_noise_rate", 0.0):
        val = f"  {val}  ".replace(" ", "  ")
    if random.random() < cfg_noise.get("casing_mixed_rate", 0.0):
        val = ''.join(ch.upper() if random.random()<0.5 else ch.lower() for ch in val)
    return val


def inject_field_missing(row: Dict[str, Any], cfg_noise: Dict[str, Any], fields: List[str]):
    miss_rate = cfg_noise.get("missing_rate", 0.0)
    for f in fields:
        if random.random() < miss_rate:
            row[f] = None


def inject_code_errors(row: Dict[str, Any], cfg_noise: Dict[str, Any]):
    if random.random() < cfg_noise.get("invalid_code_rate", 0.0):
        row["ward_code"] = (row.get("ward_code") or "") + "_X"
    if random.random() < cfg_noise.get("invalid_code_rate", 0.0):
        row["district_code"] = (row.get("district_code") or "") + "_BAD"


def inject_phone_bad_format(row: Dict[str, Any], cfg_noise: Dict[str, Any]):
    if random.random() < cfg_noise.get("phone_bad_format_rate", 0.0):
        phone = row.get("phone") or ""
        row["phone"] = phone.replace(" ", "").replace("-", "").replace("+84", "84") + "x"


def inject_out_of_order_timestamps(row: Dict[str, Any], cfg_noise: Dict[str, Any]):
    if random.random() < cfg_noise.get("out_of_order_ts_rate", 0.0):
        ca = row.get("created_at")
        ua = row.get("updated_at")
        if ca and ua:
            row["created_at"], row["updated_at"] = ua, ca


def inject_corner_inconsistency(row: Dict[str, Any], cfg_noise: Dict[str, Any], cfg: Dict[str, Any]):
    if random.random() < cfg_noise.get("corner_inconsistent_rate", 0.0):
        sh = cfg['enums']['housing_types']['street_house']
        shop = cfg['enums']['housing_types']['shophouse']
        if row.get("housing_type") not in (sh, shop):
            row["corner_flag"] = True
        else:
            row["corner_flag"] = not bool(row.get("corner_flag"))


def apply_noise(row: Dict[str, Any], cfg_noise: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    inject_field_missing(row, cfg_noise, ["corner_flag","age_band","children_count","elderly_count","email","id_card"]) 
    inject_code_errors(row, cfg_noise)
    inject_phone_bad_format(row, cfg_noise)
    inject_out_of_order_timestamps(row, cfg_noise)
    inject_corner_inconsistency(row, cfg_noise, cfg)
    for f in ["service_address","ward","district","city","full_name"]:
        row[f] = inject_string_noise(row.get(f), cfg_noise)
    return row


def create_duplicates(rows: List[Dict[str, Any]], cfg_noise: Dict[str, Any]) -> List[Dict[str, Any]]:
    dup_rate = cfg_noise.get("duplicate_rate", 0.0)
    near_dup_rate = cfg_noise.get("near_duplicate_rate", 0.0)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(r)
        if random.random() < dup_rate:
            out.append(dict(r))
        if random.random() < near_dup_rate:
            nd = dict(r)
            nd["subscriber_id"] = f"SUB-{uuid.uuid4().hex[:8].upper()}"
            if isinstance(nd.get("service_address"), str):
                nd["service_address"] = nd["service_address"] + "  "
            out.append(nd)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate CRM raw dataset (Bronze)")
    parser.add_argument("--config", default="crm_gen.yaml", help="Path to YAML config")
    parser.add_argument("--today", default=None, help="Partition date YYYY-MM-DD (default: today)")
    parser.add_argument("--rows", type=int, default=None, help="Override number of households")
    parser.add_argument("--csv", action="store_true", help="Also write CSV output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fake = Faker(cfg.get('faker_locale', 'vi_VN'))
    random.seed(cfg.get("seed", 13))
    np.random.seed(cfg.get("seed", 13))

    dt_today = args.today or dt.date.today().isoformat()

    # Seeds
    aus = seed_admin_units(cfg)
    street_names = cfg['corpora']['street_names']

    # Size
    n_households = args.rows or cfg["size"]["households"]

    # Housing mix
    h_mix = cfg["distributions"]["housing_type_mix"]["global"]

    rows = []
    hh_idx = 1

    # Time ranges
    created_min = dt.datetime.fromisoformat(cfg["time"]["created_min"])
    created_max = dt.datetime.fromisoformat(cfg["time"]["created_max"])

    # Enums shortcuts
    h_enum = cfg['enums']['housing_types']

    for hh in range(n_households):
        au = random.choice(aus)
        h_types = list(h_mix.keys())
        h_weights = list(h_mix.values())
        housing_type = pick_weighted(h_types, h_weights)

        area, floors = sample_area_floors(housing_type, cfg)
        corner = sample_corner(housing_type, cfg)
        if corner and housing_type in (h_enum['street_house'], h_enum['shophouse']):
            area = int(area * np.random.uniform(1.2, 1.3))

        age_band = sample_age_band(cfg)
        children_count, elderly_count = sample_children_elderly(age_band, housing_type, cfg)

        ids = gen_ids(hh_idx)
        k_services = sample_multi_service(cfg)

        for s in range(k_services):
            service_id = f"{ids['household_id']}-S{s+1:02d}"
            full_name = fake.name()
            phone = fake.phone_number()
            service_address = build_address(fake, street_names, au, housing_type, cfg)

            created_at = ts_between(created_min, created_max)
            updated_at = created_at + dt.timedelta(days=random.randint(0, 90))

            row = {
                "subscriber_id": ids['subscriber_id'] if s == 0 else f"SUB-{uuid.uuid4().hex[:8].upper()}",
                "account_id": ids['account_id'],
                "household_id": ids['household_id'],
                "service_id": service_id,
                "full_name": full_name,
                "phone": phone,
                "service_address": service_address,
                "ward": au.ward,
                "ward_code": au.ward_code,
                "district": au.district,
                "district_code": au.district_code,
                "city": au.city,
                "city_code": au.city_code,
                "housing_type": housing_type,
                "corner_flag": corner if housing_type in (h_enum['street_house'], h_enum['shophouse']) else False,
                "dwelling_area_m2": int(area),
                "floors": int(floors),
                "age_band": age_band,
                "children_count": int(children_count),
                "elderly_count": int(elderly_count),
                "multi_service_count": int(k_services),
                "created_at": created_at.isoformat(timespec='seconds'),
                "updated_at": updated_at.isoformat(timespec='seconds'),
                "src": cfg["source"],
                "ingest_ts": dt.datetime.now().isoformat(timespec='seconds'),
                "load_dt": dt_today,
            }
            # Extras
            if cfg.get('extras', {}).get('enable', True):
                row.update(gen_extras(fake, cfg, housing_type))
            # Noise
            row = apply_noise(row, cfg.get("noise", {}), cfg)
            # Hash
            row["src_hash"] = sha1_row(row, [
                "subscriber_id","account_id","household_id","service_id",
                "service_address","ward_code","district_code","city_code",
                "housing_type","corner_flag","dwelling_area_m2","floors",
                "age_band","children_count","elderly_count"
            ])
            rows.append(row)
        hh_idx += 1

    # Duplicates
    rows = create_duplicates(rows, cfg.get("noise", {}))

    df = pd.DataFrame(rows)

    # Output
    out_dir = cfg["output"]["local_dir"].rstrip('/') + "/crm"
    base = cfg["output"].get("base_name", "crm")

    parquet_path = to_parquet(df, out_dir, dt_today, base)
    csv_path = None
    if args.csv:
        csv_path = to_csv(df, out_dir, dt_today, base)

    print("Written Parquet:", parquet_path)
    if csv_path:
        print("Written CSV:", csv_path)


if __name__ == "__main__":
    main()
