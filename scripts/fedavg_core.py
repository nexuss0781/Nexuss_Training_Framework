"""Validation and aggregation primitives for the NTF iterative FedAvg engine."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
WORKER_NAME_PREFIX = "NTF FedAvg worker shard "


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid training config from {path}: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: Any) -> None:
    required = {
        "schema_version", "worker_count", "rounds", "local_steps", "batch_size",
        "learning_rate", "weight_decay", "max_sequence_length", "seed",
        "require_cuda", "deterministic_algorithms", "model",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"config keys must be exactly: {', '.join(sorted(required))}")
    if config["schema_version"] != 1 or config["worker_count"] != 19:
        raise ValueError("schema_version must be 1 and worker_count must remain 19")
    for key in ("rounds", "local_steps", "batch_size", "max_sequence_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_sequence_length"] < 2:
        raise ValueError("max_sequence_length must be at least 2")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(config["require_cuda"], bool) or not isinstance(config["deterministic_algorithms"], bool):
        raise ValueError("hardware and determinism flags must be booleans")
    for key in ("learning_rate", "weight_decay"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
    if config["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    model = config["model"]
    model_keys = {"name", "vocab_size", "d_model", "n_heads", "n_layers", "max_sequence_length", "dropout"}
    if not isinstance(model, dict) or set(model) != model_keys:
        raise ValueError(f"model keys must be exactly: {', '.join(sorted(model_keys))}")
    if model["name"] != "ntf-byte-gpt-14m-v1" or model["vocab_size"] != 256:
        raise ValueError("unsupported model architecture or tokenizer")
    for key in ("d_model", "n_heads", "n_layers", "max_sequence_length"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if model["d_model"] % model["n_heads"]:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if model["max_sequence_length"] != config["max_sequence_length"]:
        raise ValueError("model and training max_sequence_length must match")
    dropout = model["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")


def effective_config_digest(config: dict[str, Any], rounds: int, local_steps: int, manifest_sha256: str) -> str:
    if type(rounds) is not int or rounds < 1 or type(local_steps) is not int or local_steps < 1:
        raise ValueError("rounds and local_steps overrides must be positive integers")
    if not HEX_SHA256.fullmatch(manifest_sha256):
        raise ValueError("manifest SHA-256 is malformed")
    effective = {
        "algorithm": "fedavg-local-sgd-sample-weighted-v1",
        "config": config,
        "rounds": rounds,
        "local_steps": local_steps,
        "manifest_sha256": manifest_sha256,
        "optimizer_state_policy": "reset-per-round; aggregate-model-weights-only",
    }
    return sha256_bytes(canonical_json(effective))


def validate_manifest(manifest: Any, expected_shards: int = 19) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest must be a schema_version=1 JSON object")
    if type(expected_shards) is not int or expected_shards != 19:
        raise ValueError("NTF FedAvg requires exactly 19 workers")
    if manifest.get("shard_count") != expected_shards:
        raise ValueError("manifest shard_count must equal 19")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("manifest dataset provenance must be an object")
    for key in ("dataset_id", "revision", "source", "license"):
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise ValueError(f"manifest dataset.{key} is required for provenance")
    total = dataset.get("record_count")
    if type(total) is not int or total < expected_shards:
        raise ValueError("dataset.record_count must include at least one record per worker")
    if not isinstance(dataset.get("content_sha256"), str) or not HEX_SHA256.fullmatch(dataset["content_sha256"]):
        raise ValueError("dataset.content_sha256 must be a lowercase SHA-256 digest")
    if manifest.get("partition_algorithm") != "sha256-record-id-modulo-v1":
        raise ValueError("unsupported dataset partition algorithm")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != expected_shards:
        raise ValueError("manifest must contain exactly 19 shard entries")
    seen: set[int] = set()
    object_keys: set[str] = set()
    total_shard_records = 0
    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError("each manifest shard entry must be an object")
        shard = entry.get("id")
        if type(shard) is not int or not 0 <= shard < expected_shards:
            raise ValueError(f"invalid shard id: {shard!r}")
        if shard in seen:
            raise ValueError(f"duplicate manifest shard id {shard:02d}")
        seen.add(shard)
        key = entry.get("object_key")
        if not isinstance(key, str) or not key.strip() or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid object_key for shard {shard:02d}")
        if key in object_keys:
            raise ValueError("manifest shard object keys must be unique")
        object_keys.add(key)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
            raise ValueError(f"invalid shard SHA-256 for shard {shard:02d}")
        record_count = entry.get("record_count")
        if type(record_count) is not int or record_count < 1:
            raise ValueError(f"shard {shard:02d} must declare at least one record")
        total_shard_records += record_count
    if seen != set(range(expected_shards)):
        raise ValueError("manifest shard IDs must cover exactly 00 through 18")
    if total_shard_records != total:
        raise ValueError("sum of per-shard record_count values must equal dataset.record_count")


def aggregate_states(
    updates: Sequence[Mapping[str, Any]],
    expected_shards: int = 19,
    reference_state: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Return sample-count-weighted FedAvg of compatible floating-point model states."""
    import torch

    if expected_shards != 19:
        raise ValueError("NTF FedAvg requires exactly 19 worker updates")
    seen: set[int] = set()
    total_samples = 0
    baseline: Mapping[str, Any] | None = reference_state
    validated: list[tuple[int, Mapping[str, Any]]] = []
    for update in updates:
        shard = update.get("shard_id")
        if type(shard) is not int or not 0 <= shard < expected_shards:
            raise ValueError(f"invalid worker shard_id: {shard!r}")
        if shard in seen:
            raise ValueError(f"duplicate worker update for shard {shard:02d}")
        seen.add(shard)
        count = update.get("sample_count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"shard {shard:02d} sample_count must be a positive integer")
        state = update.get("state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"shard {shard:02d} state_dict must be a non-empty mapping")
        if baseline is None:
            baseline = state
        elif set(state) != set(baseline):
            raise ValueError(f"model key mismatch in shard {shard:02d}")
        for name, tensor in state.items():
            if not isinstance(name, str) or not torch.is_tensor(tensor):
                raise ValueError(f"invalid tensor entry in shard {shard:02d}")
            if not tensor.is_floating_point():
                raise ValueError(f"non-floating model tensor {name!r} cannot be FedAvg-averaged")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"non-finite tensor {name!r} in shard {shard:02d}")
        if baseline is not state:
            for name in baseline:
                first, current = baseline[name], state[name]
                if first.shape != current.shape:
                    raise ValueError(f"model shape mismatch for {name!r} in shard {shard:02d}")
                if first.dtype != current.dtype:
                    raise ValueError(f"model dtype mismatch for {name!r} in shard {shard:02d}")
        total_samples += count
        validated.append((count, state))
    missing = sorted(set(range(expected_shards)) - seen)
    if missing:
        raise ValueError("missing worker update(s): " + ", ".join(f"{s:02d}" for s in missing))
    assert baseline is not None
    averaged: dict[str, Any] = {}
    for name, reference in baseline.items():
        accumulator = torch.zeros(reference.shape, dtype=torch.float64, device="cpu")
        for count, state in validated:
            accumulator.add_(state[name].detach().to(device="cpu", dtype=torch.float64), alpha=count)
        output = (accumulator / total_samples).to(dtype=reference.dtype)
        if not torch.isfinite(output).all().item():
            raise ValueError(f"FedAvg produced a non-finite tensor for {name!r}")
        averaged[name] = output
    return averaged, total_samples


def evaluate_worker_jobs(jobs: Sequence[Mapping[str, Any]], expected_shards: int = 19) -> tuple[dict[int, Mapping[str, Any]], list[int], list[str], bool]:
    """Inspect exact worker names; report missing IDs, failures, and all-success status."""
    recognized: dict[int, Mapping[str, Any]] = {}
    failures: list[str] = []
    for job in jobs:
        name = str(job.get("name", ""))
        if not name.startswith(WORKER_NAME_PREFIX) or len(name) != len(WORKER_NAME_PREFIX) + 2:
            continue
        suffix = name[len(WORKER_NAME_PREFIX):]
        if not suffix.isdigit():
            continue
        shard = int(suffix)
        if not 0 <= shard < expected_shards:
            continue
        if shard in recognized:
            failures.append(f"duplicate Actions job for shard {shard:02d}")
            continue
        recognized[shard] = job
        if job.get("status") == "completed" and job.get("conclusion") != "success":
            failures.append(f"shard {shard:02d} ended with {job.get('conclusion')}")
    missing = sorted(set(range(expected_shards)) - set(recognized))
    all_succeeded = not missing and all(
        job.get("status") == "completed" and job.get("conclusion") == "success"
        for job in recognized.values()
    ) and not failures
    return recognized, missing, failures, all_succeeded
