#!/usr/bin/env python3
"""Train one assigned data shard through iterative FedAvg rounds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "fedavg"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from model import ByteGPT  # noqa: E402
from fedavg_core import (  # noqa: E402
    canonical_json,
    effective_config_digest,
    load_config,
    sha256_bytes,
    sha256_file,
    validate_manifest,
)
from fedavg_storage import ObjectStore  # noqa: E402


def seed_for_round(seed: int, round_index: int, shard: int) -> int:
    payload = f"ntf-fedavg-v1:{seed}:{round_index}:{shard}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def prepare_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def iter_examples(path: Path, max_sequence_length: int) -> Iterator[tuple[list[int], list[int]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank shard JSONL line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not isinstance(record.get("text"), str):
                raise ValueError(f"shard line {line_number} must contain string id and text fields")
            tokens = list(record["text"].encode("utf-8"))
            width = max_sequence_length + 1
            for start in range(0, len(tokens), max_sequence_length):
                chunk = tokens[start : start + width]
                if len(chunk) >= 2:
                    yield chunk[:-1], chunk[1:]


def make_batch(examples: list[tuple[list[int], list[int]]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(x) for x, _ in examples)
    inputs = torch.zeros((len(examples), length), dtype=torch.long)
    targets = torch.full((len(examples), length), -100, dtype=torch.long)
    for row, (source, target) in enumerate(examples):
        inputs[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        targets[row, : len(target)] = torch.tensor(target, dtype=torch.long)
    return inputs.to(device), targets.to(device)


def checked_global(store: ObjectStore, round_index: int, run_record: dict) -> tuple[dict, str]:
    checkpoint_key = f"global/round-{round_index:03d}.pt"
    metadata_key = f"global/round-{round_index:03d}.json"
    # Metadata is the publication marker and is written only after the checkpoint.
    metadata = json.loads(store.read_key(metadata_key).decode("utf-8"))
    checkpoint = store.read_key(checkpoint_key)
    digest = sha256_bytes(checkpoint)
    if metadata.get("round") != round_index or metadata.get("kind") != "global":
        raise ValueError(f"invalid global checkpoint metadata for round {round_index}")
    if metadata.get("checkpoint_sha256") != digest:
        raise ValueError(f"global checkpoint checksum mismatch for round {round_index}")
    for key in ("run_id", "config_sha256", "manifest_sha256", "architecture_sha256"):
        if metadata.get(key) != run_record.get(key):
            raise ValueError(f"global checkpoint {key} does not match this run")
    if round_index == 0:
        if digest != run_record.get("initial_checkpoint_sha256"):
            raise ValueError("round-0 checkpoint differs from the initializer's recorded digest")
        if metadata.get("parent_checkpoint_sha256") is not None:
            raise ValueError("round-0 checkpoint must not have a parent")
    elif not metadata.get("parent_checkpoint_sha256"):
        raise ValueError("later global checkpoint is missing parent checkpoint provenance")
    return torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True), digest


def train_local_round(
    model: ByteGPT,
    data_path: Path,
    config: dict,
    device: torch.device,
    local_steps: int,
    round_index: int,
    shard: int,
) -> tuple[dict, int, int]:
    local_seed = seed_for_round(config["seed"], round_index, shard)
    prepare_determinism(local_seed, config["deterministic_algorithms"])
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    examples = iter_examples(data_path, config["max_sequence_length"])
    samples_processed = 0
    steps_completed = 0
    for _ in range(local_steps):
        batch: list[tuple[list[int], list[int]]] = []
        for _ in range(config["batch_size"]):
            try:
                batch.append(next(examples))
            except StopIteration:
                break
        if not batch:
            break
        inputs, targets = make_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss).item():
            raise ValueError(f"non-finite local loss in round {round_index}, shard {shard:02d}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        samples_processed += len(batch)
        steps_completed += 1
    if samples_processed < 1:
        raise ValueError(f"shard {shard:02d} produced no training sequences")
    model.cpu()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return state, samples_processed, steps_completed


def run_worker(
    storage_uri: str,
    manifest_key: str,
    config_path: Path,
    shard: int,
    rounds: int,
    local_steps: int,
    poll_seconds: int = 15,
    checkpoint_timeout_seconds: int = 1800,
    run_id: str | None = None,
) -> None:
    if type(shard) is not int or not 0 <= shard < 19:
        raise ValueError("shard must be between 0 and 18")
    if rounds < 1 or local_steps < 1 or poll_seconds < 1 or checkpoint_timeout_seconds < 1:
        raise ValueError("rounds, local_steps, poll, and timeout values must be positive")
    config = load_config(config_path)
    store = ObjectStore(storage_uri)
    run_record = json.loads(store.read_key("run.json").decode("utf-8"))
    if run_id is not None and run_record.get("run_id") != run_id:
        raise ValueError("run ID mismatch")
    if run_record.get("rounds") != rounds or run_record.get("local_steps") != local_steps:
        raise ValueError("worker round/local-step configuration differs from the initializer")
    manifest_bytes = store.read_external_key(manifest_key)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_manifest(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha != run_record.get("manifest_sha256"):
        raise ValueError("dataset manifest changed after run initialization")
    if effective_config_digest(config, rounds, local_steps, manifest_sha) != run_record.get("config_sha256"):
        raise ValueError("deterministic worker configuration differs from run initialization")
    if sha256_bytes(canonical_json(config["model"])) != run_record.get("architecture_sha256"):
        raise ValueError("model architecture differs from the initialized global model")
    if config["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("this production config requires CUDA; configure a GPU runner or explicitly change require_cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    shard_entry = next(item for item in manifest["shards"] if item["id"] == shard)
    with tempfile.TemporaryDirectory(prefix=f"ntf-shard-{shard:02d}-") as temp_dir:
        shard_path = Path(temp_dir) / f"shard-{shard:02d}.jsonl"
        store.download_external_to(shard_entry["object_key"], shard_path)
        if sha256_file(shard_path) != shard_entry["sha256"]:
            raise ValueError(f"shard {shard:02d} content checksum mismatch")
        with shard_path.open("rb") as shard_stream:
            actual_records = sum(1 for _ in shard_stream)
        if actual_records != shard_entry["record_count"]:
            raise ValueError(f"shard {shard:02d} record_count does not match its manifest")
        model = ByteGPT(config["model"])
        for round_index in range(rounds):
            deadline = time.monotonic() + checkpoint_timeout_seconds
            while not store.exists(f"global/round-{round_index:03d}.json"):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for global round {round_index} checkpoint")
                time.sleep(poll_seconds)
            state, parent_sha = checked_global(store, round_index, run_record)
            model.load_state_dict(state, strict=True)
            next_state, sample_count, steps_completed = train_local_round(
                model, shard_path, config, device, local_steps, round_index, shard,
            )
            buffer = io.BytesIO()
            torch.save(next_state, buffer)
            update_bytes = buffer.getvalue()
            update_sha = sha256_bytes(update_bytes)
            update_base = f"updates/round-{round_index:03d}/shard-{shard:02d}"
            update_meta = {
                "schema_version": 1,
                "kind": "local_update",
                "run_id": run_record["run_id"],
                "round": round_index,
                "shard_id": shard,
                "worker_count": 19,
                "sample_count": sample_count,
                "steps_completed": steps_completed,
                "checkpoint_sha256": update_sha,
                "parent_checkpoint_sha256": parent_sha,
                "config_sha256": run_record["config_sha256"],
                "manifest_sha256": manifest_sha,
                "architecture_sha256": run_record["architecture_sha256"],
                "shard_data_sha256": shard_entry["sha256"],
                "dataset_id": manifest["dataset"]["dataset_id"],
                "dataset_revision": manifest["dataset"]["revision"],
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "cpu",
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "optimizer": "AdamW; reset at the start of every round",
            }
            store.put_bytes(update_base + ".pt", update_bytes)
            # A metadata object appears last and is the complete-update marker.
            store.put_bytes(update_base + ".json", canonical_json(update_meta) + b"\n")
            print(f"round {round_index + 1}/{rounds}: shard {shard:02d} processed {sample_count} examples on {device}")
    print(f"worker shard {shard:02d} completed all {rounds} rounds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-uri", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "training/fedavg/config.json")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-steps", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--checkpoint-timeout-seconds", type=int, default=1800)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_worker(
        args.storage_uri, args.manifest_key, args.config, args.shard,
        args.rounds, args.local_steps, args.poll_seconds,
        args.checkpoint_timeout_seconds, args.run_id,
    )


if __name__ == "__main__":
    main()
