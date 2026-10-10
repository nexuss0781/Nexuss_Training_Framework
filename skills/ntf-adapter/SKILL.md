---
name: ntf-adapter
description: Adapt the Nexus Training Framework's 19-worker/1-coordinator GitHub Actions pattern to an existing software, data, or ML workflow. Use when asked to install, customize, debug, or extend this NTF starter in a project.
---

# NTF Adapter Skill

You are adapting an existing project's workflow to the Nexus Training Framework (NTF), a GitHub Actions **fan-out/fan-in** starter. Preserve the core shape—19 matrix workers and one coordinator—unless the user explicitly asks to change the shard count. Never claim that unrelated independently trained model weights can be merged generically.

## First inspect

1. Inspect the target repository's language, build/test commands, current workflows, data format, secrets policy, and existing artifact conventions before editing.
2. Read the NTF `README.md`, `docs/adapting.md`, `.github/workflows/ntf.yml`, and worker/reducer scripts.
3. Identify what work can be split into independent, deterministic shards and what the correct final operation is. If there is no valid decomposition or reducer, explain the limitation rather than inventing one.
4. Preserve unrelated user changes. Do not overwrite an existing workflow; merge the pattern into it or add a clearly named workflow.

## Adaptation contract

Keep these responsibilities distinct:

- **Partition:** Given a run/input identity and shard IDs `0..18`, assign every intended work item exactly once (unless purposeful replication is part of the algorithm). Prefer a stable manifest of shard-specific files/objects for large inputs.
- **Worker:** Consume only its assigned work; be idempotent where possible; write a versioned output with the shard ID, input identity/checksum, and enough metadata to verify it.
- **Monitor:** Keep one coordinator job that tracks the 19 named worker jobs for the current workflow run. If job names change, update the monitor's matching rule and tests together. Its token should have only `actions: read` plus the minimum other permissions needed.
- **Artifact handoff:** Use unique, predictable artifact names per shard. GitHub-hosted jobs do not share a workspace. Avoid uploading secrets or unnecessarily large intermediates.
- **Reducer:** Validate result count, unique shard IDs, schemas, common input identity, and worker success before applying a domain-specific deterministic reducer. Preserve provenance in the final artifact.

## ML and aggregation gate

Before implementing model-weight aggregation, verify and document all of the following with project-specific evidence:

- identical model architecture and tensor names/shapes;
- the same agreed initial checkpoint/version and compatible tokenization/configuration;
- a named aggregation/training algorithm (for example, a properly specified federated averaging round), including the weighting and optimizer-state policy;
- validation that checks model quality, numerical integrity, and provenance.

A shared architecture by itself does **not** make checkpoints mergeable. Do not average arbitrary checkpoints, do not call independent weight averaging "frontier model training," and do not publish/deploy aggregated weights without explicit validation. Ordinary independent GitHub Actions jobs do not provide synchronized distributed gradients or guaranteed GPUs/inter-worker networking. For a non-ML workflow, reduce its actual outputs instead (counts, test reports, candidate rankings, or assembled files).

## GitHub Actions constraints to preserve

- The sample matrix has exactly 19 shards and `max-parallel: 19`; the coordinator is a twentieth job, not an extra worker.
- Concurrency is a ceiling subject to plan/account/repository limits and other active workflows, not a reservation.
- Hosted-runner hardware depends on repository visibility and runner label. The common 4 CPU / 16 GB / 14 GB Linux spec applies to standard public-repository runners in current documentation; private standard runners may have fewer resources.
- Each hosted job has a six-hour maximum. Budget for setup, data transfer, work, polling, and reduction.
- Jobs are isolated. Large data should be partitioned and fetched efficiently; downloading the same 1 GB input in all 19 jobs multiplies transfer volume.
- Review API permissions, secret exposure, billing/minutes, artifact retention, and data license/privacy before enabling production runs.

## Work procedure

1. Propose the smallest safe project-specific change; list files and any behavior that would change.
2. Implement the partition/data adapter, worker, result schema, and reducer. Keep the orchestration contract stable.
3. Update workflow paths, runtime/dependencies, environment variables, artifact names, and monitor matching only as needed. Keep shard count 19 unless authorized otherwise.
4. Add tests for exact coverage/no duplicates, missing shards, malformed outputs, common input identity, and reduction correctness. Add tests for monitor job-name matching whenever it changes.
5. Run local unit tests and a small local end-to-end run. Validate workflow YAML and actions syntax as far as available; state clearly if a real GitHub run was not performed.
6. Summarize changed files, tests, required GitHub secrets/settings, data/cost implications, and any unresolved production validation.

## Safe defaults

- Do not broaden workflow triggers or publish/deploy results without explicit user intent.
- Use least-privilege token permissions and never print tokens or secrets.
- Use idempotent shard outputs and fail loudly on missing/duplicate workers.
- Do not assume current GitHub plan limits are permanent; link to official docs.
- If the requested task needs GPUs, multi-node collectives, >6-hour jobs, shared high-throughput storage, or an invalid model merge, explain the mismatch and suggest a suitable training platform/protocol rather than disguising Actions as a cluster.
