#!/usr/bin/env python3
"""Validate per-shard outputs and reduce demo word-count maps into one summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        results.append(data)
    return results


def merge_results(results: list[dict[str, Any]], expected_shards: int) -> dict[str, Any]:
    if expected_shards < 1:
        raise ValueError("expected_shards must be at least 1")
    seen: set[int] = set()
    combined_counts: Counter[str] = Counter()
    total_records = 0
    source_hashes: set[str] = set()

    for result in results:
        if result.get("schema_version") != 1:
            raise ValueError("worker result has an unsupported schema_version")
        shard = result.get("shard_id")
        if not isinstance(shard, int) or not 0 <= shard < expected_shards:
            raise ValueError(f"invalid shard_id: {shard!r}")
        if shard in seen:
            raise ValueError(f"duplicate result for shard {shard:02d}")
        seen.add(shard)
        if result.get("shard_count") != expected_shards:
            raise ValueError(f"shard {shard:02d} reports the wrong shard_count")
        record_count = result.get("record_count")
        if not isinstance(record_count, int) or record_count < 0:
            raise ValueError(f"invalid record_count for shard {shard:02d}")
        word_counts = result.get("word_counts")
        if not isinstance(word_counts, dict):
            raise ValueError(f"word_counts for shard {shard:02d} must be an object")
        for token, count in word_counts.items():
            if not isinstance(token, str) or not isinstance(count, int) or count < 0:
                raise ValueError(f"invalid word-count entry in shard {shard:02d}")
            combined_counts[token] += count
        total_records += record_count
        digest = result.get("source_sha256")
        if isinstance(digest, str):
            source_hashes.add(digest)

    missing = sorted(set(range(expected_shards)) - seen)
    if missing:
        formatted = ", ".join(f"{shard:02d}" for shard in missing)
        raise ValueError(f"missing worker result(s): {formatted}")
    if len(source_hashes) > 1:
        raise ValueError("workers processed inputs with different SHA-256 digests")

    return {
        "schema_version": 1,
        "worker_count": expected_shards,
        "completed_shards": sorted(seen),
        "total_records": total_records,
        "source_sha256": next(iter(source_hashes), None),
        "word_counts": dict(sorted(combined_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = merge_results(load_results(args.input_dir), args.expected_shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
