from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple
import random
import pandas as pd
import xxhash

from data_gen.common.io_utils import save_csv, save_parquet
from data_gen.common.minio_utils import upload_folder

def _rng(seed: str) -> random.Random:
    return random.Random(xxhash.xxh64_intdigest(seed))

def _parse_probs(s: str, default: Dict[int, float]) -> Dict[int, float]:
    """
    "1:0.7,2:0.25,3:0.05" -> {1:0.7,2:0.25,3:0.05} (tự động chuẩn hoá tổng=1)
    """
    if not s:
        return default
    out: Dict[int, float] = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[int(k.strip())] = float(v.strip())
    tot = sum(out.values())
    if abs(tot - 1.0) > 1e-9:
        out = {k: v / tot for k, v in out.items()}
    return out

def _sample_with_probs(rng: random.Random, probs: Dict[int, float]) -> int:
    r = rng.random()
    cum = 0.0
    for k, p in sorted(probs.items()):
        cum += p
        if r <= cum:
            return k
    return max(probs.keys())

def _mk_id(prefix: str, num: int, width: int) -> str:
    return f"{prefix}{num:0{width}d}"

def gen_keys(
    n_customers: int,
    acc_probs: Dict[int, float],
    svc_probs: Dict[int, float],
    seed: str,
) -> pd.DataFrame:
    """
    Sinh bảng khóa sạch: customer_id → account_id → service_id (1→N→N).
    - Ngẫu nhiên theo phân phối nhưng tái lập bằng seed.
    - Đảm bảo duy nhất tuyệt đối cho mọi ID.
    """
    rng = _rng(f"keys|{seed}")
    rows: List[dict] = []

    cust_counter = 0
    acc_counter  = 0
    svc_counter  = 0

    for _ in range(n_customers):
        customer_id = _mk_id("CUST", cust_counter, 7)
        cust_counter += 1

        acc_n = _sample_with_probs(rng, acc_probs)
        for _a in range(acc_n):
            account_id = _mk_id("ACC", acc_counter, 8)
            acc_counter += 1

            svc_n = _sample_with_probs(rng, svc_probs)
            for _s in range(svc_n):
                service_id = _mk_id("SRV", svc_counter, 9)
                svc_counter += 1
                rows.append({
                    "customer_id": customer_id,
                    "account_id":  account_id,
                    "service_id":  service_id
                })

    df = pd.DataFrame(rows, columns=["customer_id", "account_id", "service_id"])
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate clean 1→N→N keys (customer→account→service)")
    ap.add_argument("--customers", type=int, default=50_000)

    ap.add_argument("--acc_probs", type=str, default="1:0.65,2:0.28,3:0.07",
                    help='VD: "1:0.65,2:0.28,3:0.07"')

    ap.add_argument("--svc_probs", type=str, default="1:0.72,2:0.23,3:0.05",
                    help='VD: "1:0.72,2:0.23,3:0.05"')

    ap.add_argument("--seed", type=str, default="2025-keys")
    ap.add_argument("--out", default="data/keys")
    ap.add_argument("--basename", default="seed_keys_clean")
    args = ap.parse_args()

    acc_probs = _parse_probs(args.acc_probs, {1:0.65, 2:0.28, 3:0.07})
    svc_probs = _parse_probs(args.svc_probs, {1:0.72, 2:0.23, 3:0.05})

    df = gen_keys(
        n_customers=args.customers,
        acc_probs=acc_probs,
        svc_probs=svc_probs,
        seed=args.seed,
    )

    csv_fp = save_csv(df, args.out, f"{args.basename}.csv", index=False)
    pq_fp  = save_parquet(df, args.out, f"{args.basename}.parquet")

    n_customers = df["customer_id"].nunique()
    n_accounts  = df["account_id"].nunique()
    n_services  = df["service_id"].nunique()
    print(f"[OK] keys CSV -> {csv_fp}")
    print(f"[OK] keys PQ  -> {pq_fp}")
    print(f"[STATS] customers={n_customers:,} accounts={n_accounts:,} services={n_services:,}")

    if os.environ.get("UPLOAD_TO_MINIO", "false").lower() == "true":
            prefix = f"keys/{args.basename}"
            upload_folder(args.out, prefix)
            print(f"[UPLOAD] s3://$MINIO_BUCKET/{prefix}/ (uploaded folder {args.out})")
            
if __name__ == "__main__":
    main()
