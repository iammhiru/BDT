from __future__ import annotations
import argparse, os, json, random, calendar
from typing import Dict, List, Any, Tuple
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import yaml, xxhash

from data_gen.common.io_utils import save_csv, save_parquet
from data_gen.common.minio_utils import upload_file, upload_folder


def _rng(seed_text: str) -> random.Random:
    return random.Random(xxhash.xxh64_intdigest(seed_text))

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _weighted_choice(rng: random.Random, items: List[Dict[str, Any]], key="weight") -> Dict[str, Any]:
    if not items: return {}
    ws = [float(it.get(key, 1.0)) for it in items]
    tot = sum(ws) or 1.0
    r = rng.uniform(0, tot); cum = 0.0
    for it, w in zip(items, ws):
        cum += w
        if r <= cum: return it
    return items[-1]

def _days_in_month(ym: str) -> List[str]:
    # ym = "YYYY-MM"
    y, m = [int(x) for x in ym.split("-")]
    _, ndays = calendar.monthrange(y, m)
    return [f"{y:04d}-{m:02d}-{d:02d}" for d in range(1, ndays + 1)]

def _choose_wifi_mean(std_cfg: Dict[str, Any], dl: int) -> Tuple[float, float]:
    buckets = std_cfg.get("by_bandwidth", [])
    for b in buckets:
        if dl <= int(b["max_dl"]):
            return float(b["mean"]), float(b["std"])
    return float(std_cfg.get("default_mean", 6)), float(std_cfg.get("default_std", 3))


def _pick_plan_for_service(
    rng: random.Random,
    service_id: str,
    snap_dt: date,
    subs_df: pd.DataFrame | None,
    plans_catalog: List[Dict[str, Any]],
    missingness: Dict[str, float],
) -> Tuple[str | None, str | None, int | None, int | None]:
    """
    Trả về (plan_id, plan_name, dl, ul).
    - Ưu tiên lấy từ Subscriptions: kỳ hiệu lực tại snapshot.
    - Nếu không có, rút từ catalog theo trọng số.
    - Thêm missingness nhẹ theo config.
    """
    plan_id = plan_name = None
    dl = ul = None

    if subs_df is not None and not subs_df.empty:
        # kỳ hiệu lực: start <= snapshot AND (end is NULL OR end >= snapshot)
        eff = subs_df[
            (subs_df["service_id"] == service_id) &
            (pd.to_datetime(subs_df["subscription_start_date"]) <= pd.Timestamp(snap_dt)) &
            (
                subs_df["subscription_end_date"].isna() |
                (pd.to_datetime(subs_df["subscription_end_date"]) >= pd.Timestamp(snap_dt))
            )
        ].copy()
        # nếu nhiều kỳ thỏa, lấy kỳ có start_date lớn nhất
        if not eff.empty:
            eff["start_ts"] = pd.to_datetime(eff["subscription_start_date"])
            eff = eff.sort_values("start_ts", ascending=False).iloc[0]
            plan_id   = eff.get("plan_id")
            plan_name = eff.get("plan_name")
            dl        = int(eff.get("bandwidth_dl")) if pd.notna(eff.get("bandwidth_dl")) else None
            ul        = int(eff.get("bandwidth_ul")) if pd.notna(eff.get("bandwidth_ul")) else None

    if plan_id is None or plan_name is None or dl is None or ul is None:
        choice = _weighted_choice(rng, plans_catalog) if plans_catalog else {"plan_id":"PLN001","name":"Fiber 100","dl":100,"ul":100}
        plan_id   = plan_id   or choice.get("plan_id")
        plan_name = plan_name or choice.get("name")
        dl        = dl        if dl is not None else int(choice.get("dl", 100))
        ul        = ul        if ul is not None else int(choice.get("ul", 100))

    # missingness
    if rng.random() < float(missingness.get("plan_id_null_rate", 0.0)):   plan_id = None
    if rng.random() < float(missingness.get("plan_name_null_rate", 0.0)): plan_name = None
    if rng.random() < float(missingness.get("bandwidth_null_rate", 0.0)): dl = None; ul = None

    return plan_id, plan_name, dl, ul

def _pick_cpe(
    rng: random.Random,
    device_cfg: Dict[str, Any],
    install_dt: date,
    snap_dt: date
) -> Tuple[str, str, str, str | None, int]:
    """
    Trả về (cpe_id, cpe_model, firmware, replacement_date, old_flag)
    """
    models = device_cfg.get("models", [])
    fw_map = device_cfg.get("firmware_versions", {})
    old_rate = float(device_cfg.get("old_device_rate", 0.1))
    repl_rate = float(device_cfg.get("replacement_rate", 0.05))
    repl_model_change_rate = float(device_cfg.get("replacement_model_change_rate", 0.5))

    # chọn model & firmware
    model = _weighted_choice(rng, models).get("model", "ONT-A1") if models else "ONT-A1"
    fw_list = fw_map.get(model, ["v1.0.0"])
    firmware = rng.choice(fw_list)

    # CPE ID
    digits = xxhash.xxh64_intdigest(f"{model}|{install_dt.isoformat()}")
    cpe_id = f"CPE{digits % 1_000_000:06d}"

    # old flag (độc lập gần đúng)
    old_flag = 1 if rng.random() < old_rate else 0

    # replacement?
    replacement_date = None
    if rng.random() < repl_rate:
        # replacement phải nằm trong (install, snapshot)
        min_after = int(cfg_dates.get("replace_days_after_install_min", 60))
        max_after = int(cfg_dates.get("replace_days_after_install_max", 700))
        delta_days = rng.randint(min_after, max_after)
        replacement = install_dt + relativedelta(days=delta_days)
        if replacement >= snap_dt:
            replacement = snap_dt - relativedelta(days=1)
        if replacement > install_dt:
            replacement_date = replacement.isoformat()
            # cơ hội thay đổi model/firmware khi thay
            if rng.random() < repl_model_change_rate:
                model2 = _weighted_choice(rng, models).get("model", model) if models else model
                fw_list2 = fw_map.get(model2, fw_list)
                model = model2
                firmware = rng.choice(fw_list2)

    return cpe_id, model, firmware, replacement_date, old_flag


def gen_profile_from_keys(
    keys_csv: str,
    snapshot_month: str,
    cfg: dict,
    subs_csv: str | None = None,
    seed_override: str | None = None,
) -> pd.DataFrame:
    keys = pd.read_csv(keys_csv)  # customer_id, account_id, service_id
    # lấy 1 row đại diện customer_id cho mỗi service_id (seed có thể có duplicate mapping theo account)
    keys_first = keys.groupby("service_id", as_index=False).first()

    snap_dt = datetime.strptime(snapshot_month, "%Y-%m").date().replace(day=1)
    rng_global = _rng(seed_override or cfg.get("seed", "profile-2025"))

    # (optional) subscriptions để backfill plan
    subs_df = None
    if subs_csv and os.path.exists(subs_csv):
        subs_df = pd.read_csv(subs_csv)

    plans_catalog = cfg.get("plans", {}).get("catalog", [])
    device_cfg    = cfg.get("device", {})
    wifi_cfg      = cfg.get("wifi_clients", {})
    miss_cfg      = cfg.get("missingness", {})
    global cfg_dates
    cfg_dates = cfg.get("dates", {})
    inst_min = int(cfg_dates.get("install_days_back_min", 30))
    inst_max = int(cfg_dates.get("install_days_back_max", 900))

    days_list = _days_in_month(snapshot_month)

    rows: List[dict] = []
    for _, r in keys_first.iterrows():
        service_id  = r["service_id"]
        customer_id = r["customer_id"]
        rng = _rng(f"{snapshot_month}|{service_id}")

        # install date (before snapshot)
        install_back = rng.randint(inst_min, inst_max)
        cpe_install_date = (snap_dt - relativedelta(days=install_back))

        # plan fields
        plan_id, plan_name, dl, ul = _pick_plan_for_service(
            rng, service_id, snap_dt, subs_df, plans_catalog, miss_cfg
        )

        # CPE
        cpe_id, cpe_model, fw, cpe_repl_date, old_flag = _pick_cpe(rng, device_cfg, cpe_install_date, snap_dt)

        # wifi clients daily (JSON string) — có thể NULL
        wifi_json = None
        if rng.random() >= float(wifi_cfg.get("missing_rate", 0.0)):
            mean, std = _choose_wifi_mean(wifi_cfg, dl if dl is not None else 100)
            weekend_boost = float(wifi_cfg.get("weekend_boost", 1.0))
            floor_zero = bool(wifi_cfg.get("floor_zero", True))
            counts: Dict[str, int] = {}
            for d in days_list:
                y, m, dd = [int(x) for x in d.split("-")]
                is_weekend = date(y, m, dd).weekday() >= 5
                mu = mean * (weekend_boost if is_weekend else 1.0)
                val = rng.gauss(mu, std)
                if floor_zero and val < 0: val = 0
                counts[d] = int(round(val))
            wifi_json = json.dumps(counts, ensure_ascii=False)

        rows.append({
            "service_id": service_id,
            "customer_id": customer_id,
            "plan_id": plan_id,
            "plan_name": plan_name,
            "bandwidth_dl": dl,
            "bandwidth_ul": ul,
            "cpe_id": cpe_id,
            "cpe_model": cpe_model,
            "cpe_firmware_version": fw,
            "cpe_install_date": cpe_install_date.isoformat(),
            "cpe_replacement_date": cpe_repl_date,
            "cpe_old_device_flag": old_flag,
            "wifi_clients_count_daily": wifi_json,   # JSON string {date: count} hoặc NULL
            "snapshot_month": snap_dt.strftime("%Y%m"),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate Broadband/Service Profile RAW with realistic CPE + wifi daily counts")
    ap.add_argument("--config", default="data_gen/service_profile/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--subs_csv", default=None)
    ap.add_argument("--month", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    keys_csv = args.keys_csv or cfg["paths"]["keys_csv"]
    subs_csv = args.subs_csv  or cfg["paths"].get("subs_csv")
    month    = args.month or cfg.get("snapshot_month", "2025-05")
    out_dir  = args.out or cfg["paths"]["out_dir"]
    seed     = args.seed or cfg.get("seed", "profile-2025")
    do_upload = args.upload or bool(cfg["paths"].get("upload_to_minio", False))

    df = gen_profile_from_keys(keys_csv, month, cfg, subs_csv=subs_csv, seed_override=seed)

    part = f"month={month.replace('-','')}"
    local_dir = os.path.join(out_dir, part)
    csv_fp = save_csv(df, local_dir, "service_profile_raw.csv", index=False)
    pq_dir = save_parquet(df, out_dir, "service_profile_raw.parquet", partition_cols=["snapshot_month"])

    print(f"[OK] CSV  -> {csv_fp}")
    print(f"[OK] PARQ -> {pq_dir}")
    print(f"[STATS] services={df['service_id'].nunique():,} rows={len(df):,} "
          f"wifi_non_null={df['wifi_clients_count_daily'].notna().mean():.2%} "
          f"cpe_replaced={df['cpe_replacement_date'].notna().mean():.2%}")

    if do_upload:
        upload_file(csv_fp, f"service_profile/{part}/service_profile_raw.csv", content_type="text/csv")
        upload_folder(pq_dir, "service_profile")

if __name__ == "__main__":
    main()
