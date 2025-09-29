from __future__ import annotations
import os
from typing import Optional, List
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, out_dir: str, filename: str, index: bool=False) -> str:
    ensure_dir(out_dir)
    fp = os.path.join(out_dir, filename if filename.endswith(".csv") else f"{filename}.csv")
    df.to_csv(fp, index=index)
    return fp

def save_parquet(df: pd.DataFrame, out_dir: str, filename: str, partition_cols: Optional[List[str]] = None) -> str:
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds
    ensure_dir(out_dir)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if not partition_cols:
        fp = os.path.join(out_dir, filename if filename.endswith(".parquet") else f"{filename}.parquet")
        pq.write_table(table, fp, compression="snappy")
        return fp

    base_dir = os.path.join(out_dir, filename.replace(".parquet", ""))
    ensure_dir(base_dir)

    fmt = ds.ParquetFileFormat()
    opts = fmt.make_write_options(compression="snappy")

    ds.write_dataset(
        data=table,
        base_dir=base_dir,
        format="parquet",
        partitioning=partition_cols,
        existing_data_behavior="overwrite_or_ignore",
        file_options=opts,
    )
    return base_dir
