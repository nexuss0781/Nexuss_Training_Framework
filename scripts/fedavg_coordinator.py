#!/usr/bin/env python3
"""Coordinate round barriers, validate updates, and publish iterative FedAvg checkpoints."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "fedavg"))

import torch  # noqa: E402
from fedavg_core import (  # noqa: E402
    aggregate_states,
    canonical_json,
    evaluate_worker_jobs,
    effective_config_digest,
    load_config,
    sha256_bytes,
    validate_manifest,
)
from fedavg_storage import ObjectStore  # noqa: E402

API_VERSION = "2022-11-28"


def fetch_jobs(api_url: str, repository: str, run_id: str, attempt: str, token: str) -> list[dict[str, Any]]:
    try:
        owner, repo = repository.split("/", 1)
    except ValueError as exc:
        raise ValueError("GITHUB_REPOSITORY must be owner/repository") from exc
    base = (
        f"{api_url.rstrip('/')}/repos/{quote(owner)}/{quote(repo)}"
        f"/actions/runs/{quote(run_id)}/attempts/{quote(attempt)}/jobs"
    )
    jobs: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urlencode({"per_page": 100, "page": page})
        request = Request(
            f"{base}?{query}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "nexus-training-framework-fedavg-coordinator",
            },
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload.get("jobs", [])
        jobs.extend(current)
        if len(current) < 100:
            return jobs
        page += 1


def read_actions_jobs() -> list[dict[str, Any]]:
    names = ("GITHUB_API_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_TOKEN")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing required Actions environment values: " + ", ".join(missing))
    return fetch_jobs(
        os.environ["GITHUB_API_URL"], os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"],
        os.environ["GITHUB_TOKEN"],
    )


def validate_run(store: ObjectStore, manifest_key: str, config_path: Path, rounds: int, local_steps: int) -> tuple[dict, dict, dict]:
    run_record = json.loads(store.read_key("run.json").decode("utf-8"))
    config = load_config(config_path)
    manifest_bytes = store.read_external_key(manifest_key)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_manifest(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    if run_record.get("rounds") != rounds or run_record.get("local_steps") != local_steps:
        raise ValueError("coordinator round/local-step configuration differs from initializer")
    if run_record.get("worker_count") != 19 or run_record.get("manifest_sha256") != manifest_sha:
        raise ValueError("run worker count or manifest digest differs from initializer")
    if run_record.get("config_sha256") != effective_config_digest(config, rounds, local_steps, manifest_sha):
        raise ValueError("coordinator training config differs from initializer")
    if run_record.get("architecture_sha256") != sha256_bytes(canonical_json(config["model"])):
        raise ValueError("coordinator model architecture differs from initializer")
    initial_meta = json.loads(store.read_key("global/round-000.json").decode("utf-8"))
    initial_checkpoint = store.read_key("global/round-000.pt")
    if initial_meta.get("round") != 0 or initial_meta.get("kind") != "global":
        raise ValueError("invalid initial global checkpoint metadata")
    initial_sha = sha256_bytes(initial_checkpoint)
    if initial_sha != initial_meta.get("checkpoint_sha256") or initial_sha != run_record.get("initial_checkpoint_sha256"):
        raise ValueError("initial global checkpoint checksum mismatch")
    for key in ("run_id", "config_sha256", "manifest_sha256", "architecture_sha256"):
        if initial_meta.get(key) != run_record.get(key):
            raise ValueError(f"initial global checkpoint {key} mismatch")
    if initial_meta.get("parent_checkpoint_sha256") is not None:
        raise ValueError("initial checkpoint must not have a parent")
    return run_record, manifest, config


def check_job_health() -> tuple[dict[int, dict], list[int], bool]:
    jobs = read_actions_jobs()
    recognized, missing, failures, all_succeeded = evaluate_worker_jobs(jobs, 19)
    if failures:
        raise RuntimeError("worker job failure(s): " + "; ".join(failures))
    return dict(recognized), missing, all_succeeded


def wait_for_updates(store: ObjectStore, run_record: dict, manifest: dict, config: dict, round_index: int, poll_seconds: int, timeout_seconds: int) -> list[dict]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        jobs, missing_jobs, all_succeeded = check_job_health()
        ready: list[dict] = []
        missing_markers: list[int] = []
        runtime_profiles: set[tuple[Any, ...]] = set()
        for shard in range(19):
            base = f"updates/round-{round_index:03d}/shard-{shard:02d}"
            if not store.exists(base + ".json"):
                missing_markers.append(shard)
                continue
            metadata = json.loads(store.read_key(base + ".json").decode("utf-8"))
            checkpoint_bytes = store.read_key(base + ".pt")
            if sha256_bytes(checkpoint_bytes) != metadata.get("checkpoint_sha256"):
                raise ValueError(f"round {round_index}, shard {shard:02d}: update checksum mismatch")
            shard_entry = next(item for item in manifest["shards"] if item["id"] == shard)
            expected = {
                "schema_version": 1,
                "kind": "local_update",
                "run_id": run_record["run_id"],
                "round": round_index,
                "shard_id": shard,
                "worker_count": 19,
                "config_sha256": run_record["config_sha256"],
                "manifest_sha256": run_record["manifest_sha256"],
                "architecture_sha256": run_record["architecture_sha256"],
                "shard_data_sha256": shard_entry["sha256"],
                "dataset_id": manifest["dataset"]["dataset_id"],
                "dataset_revision": manifest["dataset"]["revision"],
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"round {round_index}, shard {shard:02d}: {key} mismatch")
            if type(metadata.get("sample_count")) is not int or metadata["sample_count"] <= 0:
                raise ValueError(f"round {round_index}, shard {shard:02d}: invalid sample_count")
            steps = metadata.get("steps_completed")
            if type(steps) is not int or not 1 <= steps <= run_record["local_steps"]:
                raise ValueError(f"round {round_index}, shard {shard:02d}: invalid steps_completed")
            if not steps <= metadata["sample_count"] <= steps * config["batch_size"]:
                raise ValueError(f"round {round_index}, shard {shard:02d}: sample_count is inconsistent with local steps")
            runtime_profile = (
                metadata.get("torch_version"), metadata.get("cuda_version"),
                metadata.get("gpu_name"), metadata.get("device"),
            )
            if any(value is None for value in runtime_profile):
                raise ValueError(f"round {round_index}, shard {shard:02d}: incomplete runtime profile")
            runtime_profiles.add(runtime_profile)
            ready.append({"shard_id": shard, "sample_count": metadata["sample_count"], "metadata": metadata, "checkpoint_bytes": checkpoint_bytes})
        if not missing_markers:
            if len(runtime_profiles) != 1:
                raise ValueError(f"round {round_index}: worker PyTorch/CUDA/device profiles differ")
            return ready
        if not missing_jobs and all_succeeded:
            raise RuntimeError("all worker jobs succeeded but update markers are missing for shards: " + ", ".join(f"{i:02d}" for i in missing_markers))
        completed = 19 - len(missing_markers)
        print(f"round {round_index + 1}: {completed}/19 update markers present; {len(missing_jobs)} worker jobs not visible")
        time.sleep(min(poll_seconds, max(1, int(deadline - time.monotonic()))))
    raise TimeoutError(f"timed out waiting for round {round_index} updates")


def publish_aggregate(store: ObjectStore, run_record: dict, updates: list[dict], round_index: int, parent_meta: dict) -> dict:
    parent_sha = parent_meta["checkpoint_sha256"]
    parent_bytes = store.read_key(f"global/round-{round_index:03d}.pt")
    if sha256_bytes(parent_bytes) != parent_sha:
        raise ValueError(f"round {round_index}: parent global checkpoint checksum mismatch")
    parent_state = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=True)
    fedavg_inputs = []
    for update in updates:
        metadata = update["metadata"]
        if metadata.get("parent_checkpoint_sha256") != parent_sha:
            raise ValueError(f"round {round_index}, shard {update['shard_id']:02d} did not start from the common global checkpoint")
        state = torch.load(io.BytesIO(update["checkpoint_bytes"]), map_location="cpu", weights_only=True)
        fedavg_inputs.append({
            "shard_id": update["shard_id"],
            "sample_count": update["sample_count"],
            "state_dict": state,
        })
    global_state, total_samples = aggregate_states(fedavg_inputs, 19, reference_state=parent_state)
    buffer = io.BytesIO()
    torch.save(global_state, buffer)
    checkpoint = buffer.getvalue()
    checkpoint_sha = sha256_bytes(checkpoint)
    next_round = round_index + 1
    next_meta = {
        "schema_version": 1,
        "run_id": run_record["run_id"],
        "round": next_round,
        "kind": "global",
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": parent_sha,
        "config_sha256": run_record["config_sha256"],
        "manifest_sha256": run_record["manifest_sha256"],
        "architecture_sha256": run_record["architecture_sha256"],
        "worker_count": 19,
        "aggregation": {
            "algorithm": "sample-count-weighted FedAvg",
            "contributing_workers": 19,
            "total_sample_count": total_samples,
            "per_shard_sample_counts": {f"{u['shard_id']:02d}": u["sample_count"] for u in updates},
            "optimizer_state_policy": "not aggregated; each worker starts a fresh AdamW optimizer per round",
        },
    }
    store.put_bytes(f"global/round-{next_round:03d}.pt", checkpoint)
    store.put_bytes(f"global/round-{next_round:03d}.json", canonical_json(next_meta) + b"\n")
    print(f"published global round {next_round}; weighted examples={total_samples}; sha256={checkpoint_sha}")
    return next_meta


def wait_for_worker_success(poll_seconds: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        jobs = read_actions_jobs()
        _, missing, failures, all_succeeded = evaluate_worker_jobs(jobs, 19)
        if failures:
            raise RuntimeError("worker job failure(s): " + "; ".join(failures))
        if all_succeeded:
            print("all 19 worker jobs completed successfully")
            return
        print(f"final status: {19 - len(missing)}/{19} worker jobs visible; waiting for successful completion")
        time.sleep(min(poll_seconds, max(1, int(deadline - time.monotonic()))))
    raise TimeoutError("timed out waiting for all worker jobs to succeed")


def coordinate(storage_uri: str, manifest_key: str, config_path: Path, rounds: int, local_steps: int, poll_seconds: int, timeout_seconds: int) -> dict:
    if rounds < 1 or local_steps < 1 or poll_seconds < 1 or timeout_seconds < 1:
        raise ValueError("rounds, local_steps, poll, and timeout values must be positive")
    store = ObjectStore(storage_uri)
    run_record, manifest, config = validate_run(store, manifest_key, config_path, rounds, local_steps)
    previous_meta = json.loads(store.read_key("global/round-000.json").decode("utf-8"))
    for round_index in range(rounds):
        updates = wait_for_updates(store, run_record, manifest, config, round_index, poll_seconds, timeout_seconds)
        previous_meta = publish_aggregate(store, run_record, updates, round_index, previous_meta)
    wait_for_worker_success(poll_seconds, timeout_seconds)
    final = {
        "schema_version": 1,
        "run_id": run_record["run_id"],
        "algorithm": run_record["algorithm"],
        "completed_rounds": rounds,
        "worker_count": 19,
        "final_checkpoint": f"global/round-{rounds:03d}.pt",
        "final_checkpoint_sha256": previous_meta["checkpoint_sha256"],
        "config_sha256": run_record["config_sha256"],
        "manifest_sha256": run_record["manifest_sha256"],
        "architecture_sha256": run_record["architecture_sha256"],
    }
    store.put_bytes("final.json", canonical_json(final) + b"\n")
    print(json.dumps(final, indent=2, sort_keys=True))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-uri", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training/fedavg/config.json")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-steps", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="per-round and final worker-status timeout")
    args = parser.parse_args()
    coordinate(
        args.storage_uri, args.manifest_key, args.config, args.rounds,
        args.local_steps, args.poll_seconds, args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
