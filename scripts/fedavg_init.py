#!/usr/bin/env python3
"""Create the one shared, deterministic round-0 checkpoint for a FedAvg run."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "fedavg"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from model import ByteGPT  # noqa: E402
from fedavg_core import (  # noqa: E402
    canonical_json,
    effective_config_digest,
    load_config,
    sha256_bytes,
    validate_manifest,
)
from fedavg_storage import ObjectStore  # noqa: E402


def initialize(storage_uri: str, manifest_key: str, config_path: Path, rounds: int, local_steps: int, run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty, single run identifier")
    config = load_config(config_path)
    if rounds < 1 or local_steps < 1:
        raise ValueError("rounds and local_steps must be positive")
    store = ObjectStore(storage_uri)
    manifest_bytes = store.read_external_key(manifest_key)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_manifest(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    config_sha = effective_config_digest(config, rounds, local_steps, manifest_sha)
    architecture_sha = sha256_bytes(canonical_json(config["model"]))

    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = ByteGPT(config["model"]).cpu()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    checkpoint = buffer.getvalue()
    checkpoint_sha = sha256_bytes(checkpoint)

    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "algorithm": "fedavg-local-sgd-sample-weighted-v1",
        "rounds": rounds,
        "local_steps": local_steps,
        "worker_count": 19,
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "architecture_sha256": architecture_sha,
        "initial_checkpoint_sha256": checkpoint_sha,
        "seed": config["seed"],
        "optimizer_state_policy": "reset-per-round; aggregate-model-weights-only",
    }
    initial_meta = {
        "schema_version": 1,
        "run_id": run_id,
        "round": 0,
        "kind": "global",
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": None,
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "architecture_sha256": architecture_sha,
        "worker_count": 19,
        "aggregation": None,
    }
    # Metadata is written after each checkpoint and is the readiness marker.
    store.put_bytes("run.json", canonical_json(run_record) + b"\n")
    store.put_bytes("global/round-000.pt", checkpoint)
    store.put_bytes("global/round-000.json", canonical_json(initial_meta) + b"\n")
    print(json.dumps(run_record, indent=2, sort_keys=True))
    return run_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-uri", required=True, help="run-specific s3://bucket/prefix or file:// URI")
    parser.add_argument("--manifest-key", required=True, help="dataset manifest object key in the same bucket")
    parser.add_argument("--config", type=Path, default=ROOT / "training/fedavg/config.json")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-steps", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    initialize(args.storage_uri, args.manifest_key, args.config, args.rounds, args.local_steps, args.run_id)


if __name__ == "__main__":
    main()
