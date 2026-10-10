# Adapting NTF FedAvg safely

This guide accompanies the manual-only 19-worker protocol in the root [README](../README.md). Read it before configuring a run; changing dataset segments is supported, changing the model or aggregation contract requires code review and new tests.

## Data contract

- Source: UTF-8 JSONL; every row is an object with a unique string/integer `id` and non-empty `text`.
- Partition: `scripts/prepare_shards.py` applies `SHA256(id) mod 19`; duplicate IDs fail. The produced manifest records immutable dataset ID/revision, source, license, source SHA-256, total and per-shard record counts, shard object keys, and per-file SHA-256.
- Before upload, review the source rights, redaction, privacy, allowed retention, dataset revision, and generated manifest. Do not commit the data or secrets.
- Upload shard files and `manifest.json` to the same private S3 bucket used by the workflow. The manifest is immutable for the run. Workers verify the exact manifest and their own shard content before training.
- Empty shards are rejected. Each shard also needs enough usable text to produce at least one next-byte sequence; a worker with no training examples fails the entire round.

The preparation tool proves unique IDs over the source corpus and assigns each row to exactly one shard. Once staged remotely, the coordinator validates the manifest structure and hashes; it trusts the immutable manifest and does not download every other worker's full data shard to re-prove the partition.

## Training and aggregation contract

1. Initialize a single random-weight byte-GPT checkpoint using the versioned architecture and deterministic seed.
2. In round `r`, every worker loads the same `global/round-r.pt` and checks its SHA-256 and run/config/architecture identity.
3. Each worker trains up to the configured local-step count on only its manifest shard. The local AdamW optimizer is reset for each round. `sample_count` is the number of sequence examples actually consumed by that local update.
4. All 19 worker update markers must exist, correspond to successful Actions worker jobs, descend from the same parent checkpoint, and agree on the round, run, model, config, and manifest digests.
5. `scripts/fedavg_core.py` requires identical model key sets, shapes, floating dtypes, and finite tensor values. It computes the sample-count-weighted mean and publishes only after all 19 updates pass validation.
6. The next-round checkpoint is published to S3 before workers proceed. The final checkpoint is accompanied by `final.json`; it is not evaluated or published as a usable model automatically.

Do not aggregate optimizer states, average independently initialized full runs, change a local architecture, or skip a worker and renormalize the rest. A changed state dict, tokenizer, software pin, shard manifest, or training configuration requires a distinct reviewed run/config digest.

## Runner and storage setup

- Standard GitHub-hosted runner labels are CPU-only. The checked-in `require_cuda: true` setting prevents the workers from silently using CPUs. Select actual compatible GPU labels for the worker matrix and verify identical CUDA/PyTorch environments and available memory.
- Every Actions worker/coordinator is isolated. S3 is the round-barrier synchronization channel; GitHub artifacts are not used for intermediate checkpoints because jobs cannot consume another job's final artifact while it is still training.
- Use GitHub OIDC and a role restricted to the dataset prefix and unique run prefix. The coordinator also needs read-only Actions API access to fail fast on failed workers. No write permission to repository contents is required.
- Set an S3 lifecycle for run data; checkpoints, local updates, and data egress can dominate cost. Restrict object reads and logs because model states and data may be sensitive.
- Runner labels, training rounds, local steps, region, bucket/prefix, manifest key, and round timeout are explicit workflow inputs. `max-parallel` stays at 19; account concurrency may queue work.

## Preflight and release checks

- Run `python3 -m unittest discover -s tests -v`; tests use only synthetic tensor values and fixtures, never invoke training.
- Verify manifest checksums and metadata against the approved source version.
- Verify role trust, least-privilege S3 permissions, OIDC subject, runner labels, CUDA/PyTorch compatibility, memory/VRAM, disk, runtime, and retention configuration.
- Use a separate authorized non-production dataset/role/prefix for any future end-to-end training validation. This release-candidate preparation intentionally does not start any training workflow.
- Evaluate the resulting checkpoint against a held-out, provenance-recorded set and baseline before making any quality claim or deployment decision. FedAvg convergence/quality is data- and optimization-dependent.
- Review Actions runner, concurrency, minutes, and storage costs for the current repository/account rather than relying on plan assumptions.
