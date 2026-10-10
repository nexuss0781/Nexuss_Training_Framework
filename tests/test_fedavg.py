from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "training" / "fedavg"))

import fedavg_core  # noqa: E402
from fedavg_storage import ObjectStore  # noqa: E402
from prepare_shards import prepare_shards  # noqa: E402
from model import ByteGPT  # noqa: E402
from fedavg_coordinator import publish_aggregate, validate_run  # noqa: E402
from fedavg_init import initialize  # noqa: E402


class FedAvgCoreTests(unittest.TestCase):
    @staticmethod
    def config() -> dict:
        return {
            "schema_version": 1,
            "worker_count": 19,
            "rounds": 3,
            "local_steps": 4,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_sequence_length": 16,
            "seed": 7,
            "require_cuda": True,
            "deterministic_algorithms": True,
            "model": {
                "name": "ntf-byte-gpt-14m-v1",
                "vocab_size": 256,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "max_sequence_length": 16,
                "dropout": 0.0,
            },
        }

    @staticmethod
    def manifest() -> dict:
        return {
            "schema_version": 1,
            "dataset": {
                "dataset_id": "synthetic-test",
                "revision": "test-revision-1",
                "source": "unit-test fixture",
                "license": "test-only",
                "record_count": 19,
                "content_sha256": "a" * 64,
            },
            "partition_algorithm": "sha256-record-id-modulo-v1",
            "shard_count": 19,
            "shards": [
                {"id": i, "object_key": f"synthetic/shard-{i:02d}.jsonl", "sha256": f"{i + 1:064x}", "record_count": 1}
                for i in range(19)
            ],
        }

    @staticmethod
    def updates() -> list[dict]:
        result = []
        for shard in range(19):
            count = 3 if shard == 1 else 1
            value = 10.0 if shard == 1 else 2.0
            result.append({
                "shard_id": shard,
                "sample_count": count,
                "state_dict": {"weight": torch.tensor([value], dtype=torch.float32)},
            })
        return result

    def test_small_synthetic_model_forward_shape(self) -> None:
        model = ByteGPT(self.config()["model"]).eval()
        with torch.no_grad():
            logits = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 4, 256))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_manifest_requires_exact_19_shard_coverage_and_provenance(self) -> None:
        manifest = self.manifest()
        fedavg_core.validate_manifest(manifest)
        missing = self.manifest()
        missing["shards"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 19"):
            fedavg_core.validate_manifest(missing)
        duplicate = self.manifest()
        duplicate["shards"][18]["id"] = 17
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fedavg_core.validate_manifest(duplicate)
        bad_provenance = self.manifest()
        bad_provenance["dataset"]["license"] = ""
        with self.assertRaisesRegex(ValueError, "license"):
            fedavg_core.validate_manifest(bad_provenance)
        mismatched_total = self.manifest()
        mismatched_total["dataset"]["record_count"] = 20
        with self.assertRaisesRegex(ValueError, "sum of per-shard"):
            fedavg_core.validate_manifest(mismatched_total)

    def test_configuration_fingerprint_is_canonical_and_sensitive(self) -> None:
        config = self.config()
        fedavg_core.validate_config(config)
        digest = fedavg_core.effective_config_digest(config, 3, 4, "b" * 64)
        reversed_top = dict(reversed(list(config.items())))
        self.assertEqual(digest, fedavg_core.effective_config_digest(reversed_top, 3, 4, "b" * 64))
        self.assertNotEqual(digest, fedavg_core.effective_config_digest(config, 4, 4, "b" * 64))
        self.assertNotEqual(digest, fedavg_core.effective_config_digest(config, 3, 4, "c" * 64))
        bad = dict(config)
        bad["worker_count"] = 18
        with self.assertRaisesRegex(ValueError, "remain 19"):
            fedavg_core.validate_config(bad)

    def test_sample_count_weighted_fedavg_math(self) -> None:
        averaged, total_samples = fedavg_core.aggregate_states(self.updates(), 19)
        # 18 shards contribute 1*2.0 and shard 1 contributes 3*10.0: 66 / 21.
        self.assertEqual(total_samples, 21)
        self.assertAlmostEqual(float(averaged["weight"].item()), 66 / 21, places=6)
        self.assertEqual(averaged["weight"].dtype, torch.float32)

    def test_common_checkpoint_integrity_and_next_round_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store_root = root / "objects"
            store = ObjectStore(store_root.as_uri())
            manifest_bytes = fedavg_core.canonical_json(self.manifest()) + b"\n"
            store.put_bytes("datasets/manifest.json", manifest_bytes)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self.config()), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                run_record = initialize(store_root.as_uri(), "datasets/manifest.json", config_path, 2, 3, "synthetic-run")
            validated_run, manifest, config = validate_run(store, "datasets/manifest.json", config_path, 2, 3)
            self.assertEqual(validated_run["run_id"], run_record["run_id"])
            self.assertEqual(manifest["shard_count"], 19)

            initial_meta = json.loads(store.read_key("global/round-000.json"))
            initial_state = torch.load(io.BytesIO(store.read_key("global/round-000.pt")), weights_only=True)
            first_key = next(iter(initial_state))
            updates = []
            for update in self.updates():
                synthetic_state = {key: value.clone() for key, value in initial_state.items()}
                synthetic_state[first_key].fill_(float(update["state_dict"]["weight"].item()))
                buffer = io.BytesIO()
                torch.save(synthetic_state, buffer)
                updates.append({
                    "shard_id": update["shard_id"],
                    "sample_count": update["sample_count"],
                    "metadata": {"parent_checkpoint_sha256": initial_meta["checkpoint_sha256"]},
                    "checkpoint_bytes": buffer.getvalue(),
                })
            next_meta = publish_aggregate(store, run_record, updates, 0, initial_meta)
            self.assertEqual(next_meta["round"], 1)
            self.assertEqual(next_meta["parent_checkpoint_sha256"], initial_meta["checkpoint_sha256"])
            next_state = torch.load(io.BytesIO(store.read_key("global/round-001.pt")), weights_only=True)
            self.assertAlmostEqual(float(next_state[first_key].flatten()[0].item()), 66 / 21, places=6)

            store._path("global/round-000.pt").write_bytes(b"tampered synthetic checkpoint")
            with self.assertRaisesRegex(ValueError, "initial global checkpoint checksum mismatch"):
                validate_run(store, "datasets/manifest.json", config_path, 2, 3)

    def test_aggregator_rejects_missing_duplicate_and_invalid_sample_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            fedavg_core.aggregate_states(self.updates()[:-1], 19)
        duplicated = self.updates()
        duplicated[-1] = dict(duplicated[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fedavg_core.aggregate_states(duplicated, 19)
        invalid_count = self.updates()
        invalid_count[0]["sample_count"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fedavg_core.aggregate_states(invalid_count, 19)

    def test_aggregator_rejects_key_shape_dtype_and_nonfinite_mismatches(self) -> None:
        updates = self.updates()
        updates[18]["state_dict"] = {"other": torch.tensor([2.0])}
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            fedavg_core.aggregate_states(updates, 19)
        updates = self.updates()
        updates[18]["state_dict"] = {"weight": torch.tensor([2.0, 3.0])}
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            fedavg_core.aggregate_states(updates, 19)
        updates = self.updates()
        updates[18]["state_dict"] = {"weight": torch.tensor([2.0], dtype=torch.float64)}
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            fedavg_core.aggregate_states(updates, 19)
        updates = self.updates()
        updates[18]["state_dict"] = {"weight": torch.tensor([float("nan")])}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fedavg_core.aggregate_states(updates, 19)
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            fedavg_core.aggregate_states(self.updates(), 19, reference_state={"different": torch.tensor([2.0])})

    def test_failed_actions_workers_are_detected(self) -> None:
        jobs = [
            {"name": f"NTF FedAvg worker shard {i:02d}", "status": "completed", "conclusion": "success"}
            for i in range(19)
        ]
        jobs[4] = {"name": "NTF FedAvg worker shard 04", "status": "completed", "conclusion": "failure"}
        recognized, missing, failures, succeeded = fedavg_core.evaluate_worker_jobs(jobs, 19)
        self.assertEqual(len(recognized), 19)
        self.assertEqual(missing, [])
        self.assertIn("shard 04 ended with failure", failures)
        self.assertFalse(succeeded)

    def test_write_once_checkpoint_storage_and_safe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ObjectStore(Path(temp_dir).as_uri())
            store.put_bytes("global/round-000.pt", b"synthetic checkpoint")
            self.assertEqual(store.read_key("global/round-000.pt"), b"synthetic checkpoint")
            with self.assertRaises(FileExistsError):
                store.put_bytes("global/round-000.pt", b"overwrite")
            with self.assertRaises(ValueError):
                store.put_bytes("../escape", b"bad")


class ShardPreparationTests(unittest.TestCase):
    def test_partition_has_19_complete_disjoint_shards_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            rows = [{"id": f"record-{i:04d}", "text": f"synthetic text example {i}"} for i in range(1900)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = prepare_shards(
                source, root / "prepared", "test-corpus", "rev-1", "test source", "test license", "data/test-corpus",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fedavg_core.validate_manifest(manifest)
            observed_ids = []
            for item in manifest["shards"]:
                shard_path = root / "prepared" / "shards" / f"shard-{item['id']:02d}.jsonl"
                self.assertEqual(fedavg_core.sha256_file(shard_path), item["sha256"])
                records = [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), item["record_count"])
                observed_ids.extend(record["id"] for record in records)
            self.assertEqual(len(observed_ids), len(rows))
            self.assertEqual(len(set(observed_ids)), len(rows))

    def test_duplicate_record_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "duplicate.jsonl"
            source.write_text('{"id":"same","text":"one"}\n{"id":"same","text":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate record id"):
                prepare_shards(source, root / "prepared", "duplicate-test", "rev", "source", "license", "data/duplicate")


if __name__ == "__main__":
    unittest.main()
