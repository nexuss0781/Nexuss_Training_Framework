#!/usr/bin/env python3
"""Partition a UTF-8 JSONL corpus into 19 stable, disjoint, checksummed shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from fedavg_core import sha256_file


def shard_for_record(record_id: str, shard_count: int = 19) -> int:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def prepare_shards(
    source_path: Path,
    output_dir: Path,
    dataset_id: str,
    revision: str,
    source: str,
    license_name: str,
    object_prefix: str,
    shard_count: int = 19,
) -> Path:
    if shard_count != 19:
        raise ValueError("NTF FedAvg requires exactly 19 shards")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    for label, value in (("dataset_id", dataset_id), ("revision", revision), ("source", source), ("license", license_name)):
        if not value.strip():
            raise ValueError(f"{label} must be non-empty provenance")
    prefix = object_prefix.strip("/")
    if not prefix or ".." in Path(prefix).parts:
        raise ValueError("object_prefix must be a non-empty safe object-key prefix")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"shard-{shard:02d}.jsonl" for shard in range(shard_count)]
    existing = [path for path in [output_dir / "manifest.json", *shard_paths] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prepared dataset files; choose a fresh output directory: {existing[0]}")
    counts = [0] * shard_count

    # SQLite's unique primary key detects duplicate IDs without retaining the full ID set in RAM.
    with tempfile.TemporaryDirectory(prefix="ntf-shard-ids-") as temp_dir:
        database = Path(temp_dir) / "ids.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE record_ids (id TEXT PRIMARY KEY)")
        streams = [path.open("w", encoding="utf-8", newline="\n") for path in shard_paths]
        try:
            with source_path.open("r", encoding="utf-8") as source_stream:
                for line_number, line in enumerate(source_stream, start=1):
                    if not line.strip():
                        raise ValueError(f"blank JSONL line {line_number} is not allowed")
                    try:
                        record: Any = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON on source line {line_number}: {exc}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"source line {line_number} must be a JSON object")
                    raw_id = record.get("id")
                    text = record.get("text")
                    if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or not str(raw_id):
                        raise ValueError(f"source line {line_number} needs a non-empty string/integer id")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"source line {line_number} needs non-empty text")
                    record_id = str(raw_id)
                    try:
                        connection.execute("INSERT INTO record_ids(id) VALUES (?)", (record_id,))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate record id {record_id!r} on source line {line_number}") from exc
                    shard = shard_for_record(record_id, shard_count)
                    record["id"] = record_id
                    streams[shard].write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    counts[shard] += 1
            connection.commit()
        finally:
            for stream in streams:
                stream.close()
            connection.close()

    empty = [f"{shard:02d}" for shard, count in enumerate(counts) if count == 0]
    if empty:
        raise ValueError("source corpus does not populate every worker shard: " + ", ".join(empty))
    shards = [
        {
            "id": shard,
            "object_key": f"{prefix}/shards/shard-{shard:02d}.jsonl",
            "sha256": sha256_file(shard_paths[shard]),
            "record_count": counts[shard],
        }
        for shard in range(shard_count)
    ]
    manifest = {
        "schema_version": 1,
        "dataset": {
            "dataset_id": dataset_id,
            "revision": revision,
            "source": source,
            "license": license_name,
            "record_count": sum(counts),
            "content_sha256": sha256_file(source_path),
        },
        "partition_algorithm": "sha256-record-id-modulo-v1",
        "shard_count": shard_count,
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source JSONL with unique id and text fields")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True, help="immutable source revision/version")
    parser.add_argument("--source", required=True, help="source URL or approved registry identifier")
    parser.add_argument("--license", required=True)
    parser.add_argument("--object-prefix", required=True, help="S3 key prefix where shard files will be uploaded")
    args = parser.parse_args()
    manifest_path = prepare_shards(
        args.input, args.output_dir, args.dataset_id, args.revision,
        args.source, args.license, args.object_prefix,
    )
    print(f"prepared 19 disjoint shard files and manifest: {manifest_path}")


if __name__ == "__main__":
    main()
