#!/usr/bin/env python3
"""Tiny deterministic NTF worker example: count tokens for JSONL records in one shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[a-z0-9']+")


def iter_shard_records(input_path: Path, shard: int, shard_count: int) -> Iterable[dict[str, Any]]:
    """Yield records whose zero-based line number maps to this round-robin shard."""
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line_number % shard_count != shard:
                continue
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number + 1} must contain an object")
            yield record


def count_record_tokens(records: Iterable[dict[str, Any]]) -> tuple[int, Counter[str]]:
    total_records = 0
    counts: Counter[str] = Counter()
    for record in records:
        text = record.get("text", "")
        if not isinstance(text, str):
            raise ValueError("each demo record's 'text' field must be a string")
        total_records += 1
        counts.update(TOKEN_RE.findall(text.lower()))
    return total_records, counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(shard: int, shard_count: int, input_path: Path, output_dir: Path) -> Path:
    if shard_count < 1:
        raise ValueError("--shards must be at least 1")
    if not 0 <= shard < shard_count:
        raise ValueError("--shard must be in the range [0, --shards)")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    record_count, counts = count_record_tokens(iter_shard_records(input_path, shard, shard_count))
    result = {
        "schema_version": 1,
        "shard_id": shard,
        "shard_count": shard_count,
        "record_count": record_count,
        "source_sha256": sha256_file(input_path),
        "word_counts": dict(sorted(counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, help="zero-based shard index")
    parser.add_argument("--shards", type=int, required=True, help="total shard count")
    parser.add_argument("--input", type=Path, default=Path("data/demo.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    args = parser.parse_args()
    output = run(args.shard, args.shards, args.input, args.output_dir)
    print(f"shard {args.shard}/{args.shards} wrote {output}")


if __name__ == "__main__":
    main()
