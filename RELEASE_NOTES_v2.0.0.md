# Nexus Training Framework v2.0.0

**Proposed tag:** `v2.0.0`
**Proposed target:** `Training`
**Status:** Prepared on the `Training` branch; not tagged or published.

## Highlights

- Replace the word-count fan-out demo as the primary workflow with an explicitly dispatched 19-worker iterative federated-averaging (FedAvg) training path.
- Add a shared approximately 14M-parameter byte-level causal Transformer architecture and deterministic configuration, with configurable global rounds and per-round local optimizer steps.
- Initialize workers from one common global checkpoint per round; validate all 19 local updates and publish the next global checkpoint only after sample-count-weighted FedAvg succeeds.
- Add dataset sharding/provenance tooling, immutable S3 checkpoint synchronization, run/config/architecture/manifest fingerprints, tensor compatibility checks, and failed-worker detection.
- Make runner labels, dataset manifest, object-storage location, region, and coordination timeout configurable in a manual-only GitHub Actions workflow.
- Document that standard GitHub-hosted runner labels are CPU-only and real training throughput needs properly configured GPU-capable workers.
- Add offline tests for synthetic tensor aggregation and validation. No training run is included in this release preparation.

## Operational notes

- The workflow requires a private S3 bucket, GitHub OIDC role configuration, an immutable 19-shard manifest, and 19 compatible GPU-capable workers. A CPU worker fails closed under the default `require_cuda` setting.
- The reference architecture/configuration is a training starting point, not a trained or validated model and not a frontier model. Run a separate evaluation before making any quality or deployment claim.
- The old `main` branch and `1.0.0` tag are unchanged. The old LitGPT smoke example remains separate from the FedAvg workflow.
