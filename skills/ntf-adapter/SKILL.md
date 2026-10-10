---
name: ntf-adapter
description: Adapt the Nexus Training Framework's 19-worker iterative FedAvg workflow, dataset manifest, object-storage coordination, and validation checks. Use when asked to install, customize, debug, or extend this NTF starter in a project.
---

# NTF Adapter Skill

NTF is a manual-only 19-worker iterative federated-averaging training scaffold. Preserve exactly 19 shard workers and one coordinator unless the user explicitly requests another topology. The coordinator is separate from the worker matrix. Do not claim that arbitrary independent model runs can be merged.

## First inspect

1. Inspect repository language, branch/worktree status, workflows, data contract, secrets policy, storage, and test commands.
2. Read the root README, `docs/adapting.md`, `.github/workflows/ntf.yml`, `training/fedavg/config.json`, and FedAvg scripts.
3. Confirm whether the task is only code preparation or explicitly authorizes a training run. Never start a training run merely to test code; synthetic tensors and fixture data are sufficient for unit tests.
4. Preserve unrelated branches, tags, work, model artifacts, and user changes.

## FedAvg invariants

- Every round begins with all 19 workers loading the identical published global checkpoint and verifying its checksum and run/config/architecture/manifest identity.
- Every worker consumes only one assigned dataset shard. The training architecture and effective configuration are identical across shards; only data assignment and deterministic per-shard/per-round random seed differ.
- The coordinator must fail closed on failed/missing/duplicate workers, incomplete metadata, inconsistent parent checkpoint, manifest/config mismatch, tensor key/shape/dtype mismatch, non-finite values, or invalid sample counts.
- Aggregate model tensors using sample-count-weighted FedAvg, `sum(n_i * local_weights_i) / sum(n_i)`, where `n_i` is the actual number of local sequence examples processed for that update. Do not average optimizer state; this implementation resets each local optimizer per round.
- Publish the next global checkpoint only after all 19 compatible updates succeed. Workers must wait for it before starting the next round. Never average diverged full-training runs or independent initializations.
- Checkpoint/state format, model architecture, tokenization, data manifest, deterministic configuration, and software pins are part of the compatibility identity. Require new tests for changes.

## Data and storage contract

- Source JSONL records require unique IDs and non-empty text. Use `scripts/prepare_shards.py` to partition by a stable 19-way ID hash and generate manifest provenance plus per-shard/source checksums. It rejects duplicate source IDs and empty shards.
- Keep data outside Git. Review licensing, privacy, source revision, and retention before uploading the shard files and manifest to private object storage.
- Workers validate exact manifest coverage, per-shard SHA-256 and record count; the coordinator checks data/config/architecture digests across every update. The manifest is trusted as the immutable statement of the partition produced by the preparation tool; the coordinator does not re-download all workers' raw data.
- Actions jobs have isolated filesystems. Use the documented S3 barrier for intermediate round state; ordinary artifacts cannot provide a live round-by-round synchronization channel. Restrict OIDC role permissions to required dataset/run prefixes and configure lifecycle retention.

## Runner/Actions boundary

- Standard GitHub-hosted runner labels such as `ubuntu-latest` are CPU-only. Production config requires CUDA and fails closed without it. Configure 19 compatible GPU-capable workers; do not imply a standard hosted runner provides GPUs.
- Configure `worker_runner_json` and `coordinator_runner_json` with valid JSON arrays of runner labels. Runner labels do not reserve availability; account concurrency may queue jobs.
- Keep Actions token permissions least-privilege: workers need source read and S3 role access; coordinator needs Actions read, source read, and S3 role access; no repository-content write permission is necessary.
- Keep workflow triggers manual-only unless the user explicitly requests automatic training. Document execution time, storage/transfer cost, data privacy, and current runner/account limits.

## Work procedure and validation

1. Make the smallest project-specific change while retaining the algorithm invariants.
2. Add or update tests for 19-way coverage, unique IDs, manifest provenance, deterministic config, common checkpoint, weighted math, tensor compatibility, missing/failed workers, and write-once checkpoint behavior.
3. Run `python3 -m unittest discover -s tests -v`. Tests must not launch model training or a GitHub workflow unless the user explicitly authorized a non-production run.
4. Validate workflow YAML and static Python compilation. State clearly whether an actual Actions run/training occurred.
5. Keep output checkpoints and release tags unpublished unless the user has approved the exact external release action.
6. For a release candidate, provide the exact proposed title, tag, target branch, notes payload, and commit. Do not publish the tag/release before confirmation.
