from __future__ import annotations
import os
from typing import Optional
from minio import Minio

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "raw-stage")

def _client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)

def ensure_bucket(bucket: Optional[str]=None):
    cli = _client(); b = bucket or MINIO_BUCKET
    if not cli.bucket_exists(b):
        cli.make_bucket(b)

def upload_file(local_path: str, remote_path: str, bucket: Optional[str]=None, content_type: Optional[str]=None):
    cli = _client(); b = bucket or MINIO_BUCKET
    ensure_bucket(b); cli.fput_object(b, remote_path, local_path, content_type=content_type)

def upload_folder(local_dir: str, remote_prefix: str, bucket: Optional[str]=None):
    cli = _client(); b = bucket or MINIO_BUCKET
    ensure_bucket(b)
    for root, _, files in os.walk(local_dir):
        for f in files:
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, start=local_dir).replace("\\","/")
            rp = f"{remote_prefix.rstrip('/')}/{rel}"
            cli.fput_object(b, rp, lp)
