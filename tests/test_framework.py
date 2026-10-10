from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import demo_worker  # noqa: E402
import monitor  # noqa: E402
import reduce as reducer  # noqa: E402


class WorkerTests(unittest.TestCase):
    def test_round_robin_shards_cover_each_line_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "items.jsonl"
            rows = [{"id": str(i), "text": f"row number {i}"} for i in range(41)]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            observed: list[str] = []
            for shard in range(19):
                records = list(demo_worker.iter_shard_records(input_path, shard, 19))
                observed.extend(record["id"] for record in records)
                self.assertLessEqual(len(records), 3)

            self.assertEqual(sorted(observed, key=int), [str(i) for i in range(41)])
            self.assertEqual(len(observed), len(set(observed)))

    def test_worker_rejects_out_of_range_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "items.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                demo_worker.run(19, 19, input_path, Path(temp_dir) / "out")


class ReducerTests(unittest.TestCase):
    @staticmethod
    def result(shard: int, count: int = 1) -> dict:
        return {
            "schema_version": 1,
            "shard_id": shard,
            "shard_count": 2,
            "record_count": count,
            "source_sha256": "abc123",
            "word_counts": {"nexus": shard + 1},
        }

    def test_reduces_all_shards(self) -> None:
        merged = reducer.merge_results([self.result(0), self.result(1, 3)], 2)
        self.assertEqual(merged["worker_count"], 2)
        self.assertEqual(merged["completed_shards"], [0, 1])
        self.assertEqual(merged["total_records"], 4)
        self.assertEqual(merged["word_counts"], {"nexus": 3})

    def test_rejects_missing_or_duplicate_shards(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            reducer.merge_results([self.result(0)], 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reducer.merge_results([self.result(0), self.result(0), self.result(1)], 2)

    def test_rejects_different_source_digests(self) -> None:
        second = self.result(1)
        second["source_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different SHA-256"):
            reducer.merge_results([self.result(0), second], 2)


class MonitorTests(unittest.TestCase):
    @staticmethod
    def worker_job(shard: int, status: str = "completed", conclusion: str = "success") -> dict:
        return {
            "name": f"NTF worker shard {shard:02d}",
            "status": status,
            "conclusion": conclusion,
        }

    def test_monitor_recognizes_exact_worker_names_and_ignores_coordinator(self) -> None:
        jobs = [self.worker_job(i) for i in range(19)]
        jobs.append({"name": "NTF coordinator / live monitor + reducer", "status": "in_progress"})
        recognized, missing, succeeded = monitor.evaluate_jobs(jobs, 19)
        self.assertEqual(len(recognized), 19)
        self.assertEqual(missing, [])
        self.assertTrue(succeeded)

    def test_monitor_waits_for_missing_and_in_progress_workers(self) -> None:
        jobs = [self.worker_job(i) for i in range(18)]
        jobs.append(self.worker_job(18, status="in_progress", conclusion=""))
        recognized, missing, succeeded = monitor.evaluate_jobs(jobs, 19)
        self.assertEqual(missing, [])
        self.assertFalse(succeeded)
        recognized, missing, succeeded = monitor.evaluate_jobs(jobs[:-1], 19)
        self.assertEqual(missing, [18])
        self.assertFalse(succeeded)

    def test_monitor_reports_worker_failure_after_completion(self) -> None:
        jobs = [self.worker_job(i) for i in range(19)]
        jobs[7] = self.worker_job(7, conclusion="failure")
        _, missing, succeeded = monitor.evaluate_jobs(jobs, 19)
        self.assertEqual(missing, [])
        self.assertFalse(succeeded)


if __name__ == "__main__":
    unittest.main()
