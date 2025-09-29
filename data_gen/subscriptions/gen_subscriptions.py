from __future__ import annotations
import argparse, os, json, random
from typing import Dict, Any, List, Tuple
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

def _hist_choice_int(rng: random.Random, hist: Dict[str, float]) -> int:
    items = [(int(k), float(v)) for k, v in hist.items()]
    tot = sum(v for _, v in items) or 1.0
    r = rng.uniform(0, tot); cum = 0.0
    for k, v in items:
        cum += v
        if r <= cum: return k
    return items[-1][0]

def _sub_id_for(service_id: str, ordinal: int) -> str:
    # ví dụ: SRV000012345 -> SUB000012345-v02
    digits = "".join(ch for ch in service_id if ch.isdigit())
    return f"SUB{digits}-v{ordinal:02d}"


def _build_periods_for_service(
    rng: random.Random,
    snapshot: date,
    periods_hist: Dict[str, float],
    contract_hist: Dict[str, float],
    gap_hist: Dict[str, float],
    active_rate: float,
) -> List[Tuple[date, date | None]]:
    """
    Xây danh sách (start_date, end_date) cho 1 service, theo thứ tự thời gian tăng dần.
    - n_periods từ periods_hist
    - Độ dài mỗi kỳ từ contract_hist (tháng)
    - Gap giữa kỳ từ gap_hist (tháng)
    - Xác suất kỳ cuối active qua snapshot theo active_rate
    """
    n = _hist_choice_int(rng, periods_hist)
    periods: List[Tuple[date, date | None]] = []

    # Xây ngược từ kỳ mới nhất về quá khứ, sau đó đảo lại
    current_end = None
    # kỳ cuối: active?
    last_active = rng.random() < active_rate

    for i in range(n, 0, -1):
        months = _hist_choice_int(rng, contract_hist)
        if i == n:  # kỳ mới nhất
            if last_active:
                end_date = None
                start_date = snapshot - relativedelta(months=months, days=rng.randint(0, 27))
            else:
                # kết thúc trước snapshot
                end_date = snapshot - relativedelta(days=rng.randint(1, 28))
                start_date = end_date - relativedelta(months=months, days=rng.randint(0, 27))
            current_end = start_date
        else:
            # quá khứ: end = current_start - gap
            gap_m = _hist_choice_int(rng, gap_hist)
            end_date = current_end - relativedelta(months=gap_m, days=rng.randint(0, 5))
            start_date = end_date - relativedelta(months=months, days=rng.randint(0, 27))
            current_end = start_date

        periods.append((start_date, end_date))

    periods.reverse()
    return periods


def gen_subscriptions_from_keys(
    keys_csv: str,
    snapshot_month: str,
    cfg: dict,
    crm_csv: str | None = None,
    seed_override: str | None = None,
) -> pd.DataFrame:
    keys = pd.read_csv(keys_csv)  # columns: customer_id, account_id, service_id
    snap_dt = datetime.strptime(snapshot_month, "%Y-%m").date().replace(day=1)
    seed = seed_override or cfg.get("seed", "subs-2025")

    # config
    periods_hist  = cfg.get("periods", {}).get("per_service_hist", {"1":1.0})
    contract_hist = cfg.get("dates",   {}).get("contract_months_hist", {"12":1.0})
    gap_hist      = cfg.get("dates",   {}).get("gap_months_hist", {"0":1.0})
    active_rate   = float(cfg.get("dates", {}).get("active_rate", 1.0))

    life = cfg.get("lifecycle", {})
    upgrade_prob = float(life.get("upgrade_event_prob", 0.1))
    up_min_m = int(life.get("upgrade_after_months_min", 3))
    up_max_m = int(life.get("upgrade_after_months_max", 12))
    miss_up_prob = float(life.get("missing_upgrade_date_prob", 0.0))

    addr_change_prob = float(life.get("address_change_prob", 0.05))
    ac_min_m = int(life.get("address_change_after_months_min", 6))
    ac_max_m = int(life.get("address_change_after_months_max", 18))
    miss_ac_prob = float(life.get("missing_address_change_date_prob", 0.0))

    multi_site_prob = float(life.get("multi_site_prob", 0.1))
    ms_count_hist   = life.get("multi_site_count_hist", {"1":1.0})

    plans = cfg.get("plans", {}).get("catalog", [])
    addr_cfg = cfg.get("addressing", {})
    backfill_rate = float(addr_cfg.get("backfill_from_crm_rate", 0.0))
    force_null_rate = float(addr_cfg.get("force_null_rate", 0.0))

    miss_cfg = cfg.get("missingness", {})
    plan_id_null_rate   = float(miss_cfg.get("plan_id_null_rate", 0.0))
    plan_name_null_rate = float(miss_cfg.get("plan_name_null_rate", 0.0))
    bw_null_rate        = float(miss_cfg.get("bandwidth_null_rate", 0.0))

    # (optional) address từ CRM
    crm_addr: Dict[str, str] = {}
    if crm_csv and os.path.exists(crm_csv):
        crm_df = pd.read_csv(crm_csv, usecols=["customer_id","address"])
        crm_addr = dict(zip(crm_df["customer_id"], crm_df["address"]))

    rows: List[dict] = []

    # Lặp theo service (đảm bảo 1 service_id → N subscription_id)
    for service_id, grp in keys.groupby("service_id"):
        # cùng service -> customer/account là duy nhất trong seed (nếu có nhiều, chọn first)
        r0 = grp.iloc[0]
        customer_id = r0["customer_id"]
        account_id  = r0["account_id"]

        rng_service = _rng(f"{seed}|{service_id}")
        periods = _build_periods_for_service(
            rng_service, snap_dt, periods_hist, contract_hist, gap_hist, active_rate
        )

        # address backfill quyết định 1 lần cho service (thực tế thường giữ địa chỉ/hồ sơ)
        addr_val = None
        if crm_addr and rng_service.random() < backfill_rate:
            addr_val = crm_addr.get(customer_id)
        if rng_service.random() < force_null_rate:
            addr_val = None

        # sinh từng kỳ
        for ord_idx, (start_date, end_date) in enumerate(periods, start=1):
            rng_row = _rng(f"{seed}|{service_id}|{ord_idx}")
            sub_id = _sub_id_for(service_id, ord_idx)

            # Plan selection (weighted)
            plan = _weighted_choice(rng_row, plans) if plans else {"plan_id":"PLN001","name":"Fiber 100","dl":100,"ul":100}
            plan_id   = plan.get("plan_id")
            plan_name = plan.get("name")
            bw_dl     = int(plan.get("dl", 100))
            bw_ul     = int(plan.get("ul", 100))

            # missingness cho plan/bandwidth
            if rng_row.random() < plan_id_null_rate:
                plan_id = None
            if rng_row.random() < plan_name_null_rate:
                plan_name = None
            if rng_row.random() < bw_null_rate:
                bw_dl = None; bw_ul = None

            # Events: upgrade/address change
            # Phạm vi hợp lệ cho ngày sự kiện phải nằm trong [start_date, end_date) nếu có end; nếu active (end=None), trong [start_date, snapshot)
            def _event_date_between(min_m: int, max_m: int) -> str | None:
                # chọn tháng offset
                offset_m = rng_row.randint(min_m, max_m)
                base = start_date + relativedelta(months=offset_m, days=rng_row.randint(0, 27))
                last_allowed = (end_date - relativedelta(days=1)) if end_date else (snap_dt - relativedelta(days=1))
                if base >= last_allowed:
                    base = last_allowed
                if base <= start_date:
                    base = start_date + relativedelta(days=1)
                return base.isoformat()

            upgrade_flag = (rng_row.random() < upgrade_prob)
            upgrade_date = _event_date_between(up_min_m, up_max_m) if upgrade_flag else None
            if upgrade_flag and rng_row.random() < miss_up_prob:
                upgrade_date = None

            address_change_flag = (rng_row.random() < addr_change_prob)
            address_change_date = _event_date_between(ac_min_m, ac_max_m) if address_change_flag else None
            if address_change_flag and rng_row.random() < miss_ac_prob:
                address_change_date = None

            multi_site_flag = (rng_row.random() < multi_site_prob)
            multi_site_count = _hist_choice_int(rng_row, ms_count_hist) if multi_site_flag else 1

            rows.append({
                "subscription_id": sub_id,
                "customer_id": customer_id,
                "service_id": service_id,
                "plan_id": plan_id,
                "plan_name": plan_name,
                "bandwidth_dl": bw_dl,
                "bandwidth_ul": bw_ul,
                "subscription_start_date": start_date.isoformat(),
                "subscription_end_date": end_date.isoformat() if end_date else None,
                "upgrade_event_flag": 1 if upgrade_flag else 0,
                "upgrade_date": upgrade_date,
                "address": addr_val,
                "multi_site_flag": 1 if multi_site_flag else 0,
                "multi_site_count": multi_site_count,
                "address_change_flag": 1 if address_change_flag else 0,
                "address_change_date": address_change_date,
                "snapshot_month": snap_dt.strftime("%Y%m"),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate Service Subscriptions RAW (realistic, multi-period per service)")
    ap.add_argument("--config", default="data_gen/subscriptions/config.yaml")
    ap.add_argument("--keys_csv", default=None)
    ap.add_argument("--crm_csv", default=None)
    ap.add_argument("--month", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    keys_csv = args.keys_csv or cfg["paths"]["keys_csv"]
    crm_csv  = args.crm_csv  or cfg["paths"].get("crm_csv")
    month    = args.month or cfg.get("snapshot_month", "2025-05")
    out_dir  = args.out or cfg["paths"]["out_dir"]
    seed     = args.seed or cfg.get("seed", "subs-2025")
    do_upload = args.upload or bool(cfg["paths"].get("upload_to_minio", False))

    df = gen_subscriptions_from_keys(keys_csv, month, cfg, crm_csv=crm_csv, seed_override=seed)

    part = f"month={month.replace('-','')}"
    local_dir = os.path.join(out_dir, part)
    csv_fp = save_csv(df, local_dir, "subscriptions_raw.csv", index=False)
    pq_dir = save_parquet(df, out_dir, "subscriptions_raw.parquet", partition_cols=["snapshot_month"])

    print(f"[OK] CSV  -> {csv_fp}")
    print(f"[OK] PARQ -> {pq_dir}")
    print(f"[STATS] services={df['service_id'].nunique():,} "
          f"subs={df['subscription_id'].nunique():,} "
          f"avg_subs_per_service={df.groupby('service_id')['subscription_id'].nunique().mean():.2f}")

    if do_upload:
        upload_file(csv_fp, f"subscriptions/{part}/subscriptions_raw.csv", content_type="text/csv")
        upload_folder(pq_dir, "subscriptions")

if __name__ == "__main__":
    main()
