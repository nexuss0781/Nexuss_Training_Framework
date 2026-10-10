# Adapting NTF to a project

This note is for the project maintainer or the AI agent using `skills/ntf-adapter/SKILL.md`.

## 1. Decide whether the task parallelizes cleanly

A useful shard has bounded input, independent work, a predictable output, and a defined join operation.

| Good first fits | Requires careful design | Usually a poor fit for ordinary hosted Actions |
|---|---|---|
| Independent test suites, batch conversion, document/media preprocessing, evaluation sets, static analysis, parameter sweeps | Federated learning rounds with compatible models, large data requiring object-store credentials, outputs that need ranking/deduplication | Synchronized multi-node gradient training, large frontier-model pretraining, GPU-heavy training without GPU runners, tasks needing a shared local disk or continuous low-latency worker communication |

Do not parallelize a fundamentally sequential workflow merely to use 19 jobs. Startup, environment installation, data transfer, and coordination can outweigh the work.

## 2. Define a partition contract

Document the immutable input version and a mapping from each work item to exactly one shard. Prefer a manifest such as:

```json
{
  "schema_version": 1,
  "input_version": "dataset-2026-10-11",
  "shard_count": 19,
  "shards": [
    {"id": 0, "objects": ["part-000", "part-019"]},
    {"id": 1, "objects": ["part-001", "part-020"]}
  ]
}
```

The example is illustrative, not a full 19-entry manifest. In production, record all shard entries, size/checksum, expected work counts, and the source revision. A hash-based assignment gives stable item coverage; balancing by bytes, tokens, or estimated runtime may be better than balancing record counts.

The included demo uses `line_index % 19` and the small checked-in `data/demo.jsonl`. Every runner checks out that file and scans it. For a large dataset, replace this adapter so a worker fetches only its assigned partition, rather than downloading all source data in each runner.

## 3. Adapt worker and result schema

A worker should accept a shard ID and total count, then write a result such as:

```json
{
  "schema_version": 1,
  "shard_id": 4,
  "shard_count": 19,
  "input_version": "immutable-version-or-digest",
  "items_seen": 1250,
  "output": "shard-specific-result"
}
```

Use compact summaries when possible. Large outputs can be sent to object storage with an authenticated, short-lived reference; the artifact can carry checksums and metadata instead of a huge file. Make retries safe: same run/shard should not silently overwrite another shard's data.

## 4. Adapt reducer and verification

The reducer must match the job's real semantics. Examples:

- **Counts/metrics:** sum counts and recompute global metrics from sufficient statistics; do not average shard percentages without weighting.
- **Files:** assemble only with a deterministic order and collision policy.
- **Search/evaluation candidates:** gather candidates and run a global ranking or validation step.
- **ML:** implement only a specified, compatible aggregation/training algorithm. Same architecture alone is insufficient. Validate resulting quality against a baseline and keep the source checkpoint/version and algorithm metadata.

Always reject missing or duplicate shard IDs, unexpected input versions, schema mismatches, invalid checksums, and failed workers. Retain per-shard outputs for debugging for an appropriate period, while considering quota and privacy.

## 5. Adapt the workflow carefully

1. Keep the matrix worker display name prefix `NTF worker shard ` or change `WORKER_NAME_RE` in `scripts/monitor.py` and its tests at the same time.
2. Keep exactly 19 entries for the canonical 19–1 mode. The coordinator is a separate job, so it can overlap when capacity permits.
3. Set only the permissions required. The monitor needs Actions read permission to list jobs; it does not need write access.
4. Configure dependencies, caching, input retrieval, and runtime for the language/framework. Do not store credentials in source or artifacts.
5. Adjust timeouts below GitHub's hosted-job maximum and plan for concurrent job-minutes, data egress, artifact transfer, and private-repository billing.
6. Add triggers deliberately. The example is manual-only; a push or schedule trigger may start 20 jobs repeatedly.
7. Pin third-party actions to reviewed commit SHAs for hardened/production environments, and periodically update the pins.

## 6. Before rollout

- Run the local tests: `python3 -m unittest discover -s tests -v`.
- Run one small full end-to-end sample and compare against a sequential baseline.
- Test one worker failure, missing output, duplicate output, and retry behavior.
- Check whether the selected GitHub plan/repository has sufficient concurrency, Actions minutes, artifact storage, and network access.
- Confirm data privacy, license, retention, and secret exposure requirements.
- Treat the first GitHub-hosted run as an integration test; local success does not validate GitHub's API permissions, policy, or actual runner capacity.
