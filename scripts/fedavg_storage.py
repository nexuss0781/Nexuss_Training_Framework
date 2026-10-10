"""Minimal S3 and local-file storage adapters for FedAvg data/checkpoint handoff."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import urlparse


class ObjectStore:
    def __init__(self, uri: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            if not parsed.netloc:
                raise ValueError("S3 storage URI must include a bucket")
            self.kind = "s3"
            self.bucket = parsed.netloc
            self.prefix = parsed.path.strip("/")
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("install boto3 from training/fedavg/requirements.txt to use S3") from exc
            self.client = boto3.client("s3")
            self.root = None
        elif parsed.scheme == "file":
            self.kind = "file"
            self.bucket = None
            self.prefix = ""
            self.client = None
            self.root = Path(parsed.path).expanduser().resolve()
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("storage URI must use s3:// or file://")

    @staticmethod
    def _validate_key(key: str) -> str:
        path = PurePosixPath(key)
        if not key or path.is_absolute() or any(part in {"", ".", ".."} for part in key.split("/")):
            raise ValueError(f"unsafe object key: {key!r}")
        return str(path)

    def _path(self, key: str) -> Path:
        assert self.root is not None
        safe = self._validate_key(key)
        path = (self.root / safe).resolve()
        if self.root not in path.parents:
            raise ValueError("object path escaped storage root")
        return path

    def _s3_key(self, key: str) -> str:
        safe = self._validate_key(key)
        return f"{self.prefix}/{safe}" if self.prefix else safe

    def read_key(self, key: str) -> bytes:
        if self.kind == "file":
            return self._path(key).read_bytes()
        response = self.client.get_object(Bucket=self.bucket, Key=self._s3_key(key))
        return response["Body"].read()

    def read_external_key(self, key: str) -> bytes:
        """Read a manifest/data key in the bucket, outside the per-run prefix."""
        safe = self._validate_key(key)
        if self.kind == "file":
            assert self.root is not None
            return (self.root / safe).read_bytes()
        response = self.client.get_object(Bucket=self.bucket, Key=safe)
        return response["Body"].read()

    def download_external_to(self, key: str, destination: Path) -> None:
        safe = self._validate_key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "file":
            assert self.root is not None
            source = self.root / safe
            with source.open("rb") as src, destination.open("wb") as dst:
                _copy_stream(src, dst)
            return
        response = self.client.get_object(Bucket=self.bucket, Key=safe)
        with response["Body"] as src, destination.open("wb") as dst:
            _copy_stream(src, dst)

    def exists(self, key: str) -> bool:
        if self.kind == "file":
            return self._path(key).is_file()
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._s3_key(key))
            return True
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def put_bytes(self, key: str, payload: bytes) -> None:
        if self.kind == "file":
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            return
        # S3's conditional write makes run outputs immutable and prevents accidental overwrite.
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._s3_key(key),
            Body=payload,
            IfNoneMatch="*",
        )


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        destination.write(chunk)
