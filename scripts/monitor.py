#!/usr/bin/env python3
"""Poll the current GitHub Actions run until every expected NTF worker has completed."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

WORKER_NAME_RE = re.compile(r"^NTF worker shard (\d{2})$")
API_VERSION = "2022-11-28"


def evaluate_jobs(jobs: list[dict[str, Any]], expected_shards: int) -> tuple[dict[int, dict[str, Any]], list[int], bool]:
    """Return recognized jobs, missing shard IDs, and whether all recognized jobs succeeded."""
    recognized: dict[int, dict[str, Any]] = {}
    for job in jobs:
        match = WORKER_NAME_RE.fullmatch(str(job.get("name", "")))
        if not match:
            continue
        shard = int(match.group(1))
        if 0 <= shard < expected_shards:
            recognized[shard] = job

    missing = [shard for shard in range(expected_shards) if shard not in recognized]
    all_completed = not missing and all(job.get("status") == "completed" for job in recognized.values())
    all_succeeded = all_completed and all(job.get("conclusion") == "success" for job in recognized.values())
    return recognized, missing, all_succeeded


def fetch_jobs(api_url: str, repository: str, run_id: str, attempt: str, token: str) -> list[dict[str, Any]]:
    owner, repo = repository.split("/", 1)
    base = (
        f"{api_url.rstrip('/')}/repos/{quote(owner)}/{quote(repo)}"
        f"/actions/runs/{quote(run_id)}/attempts/{quote(attempt)}/jobs"
    )
    page = 1
    jobs: list[dict[str, Any]] = []
    while True:
        query = urlencode({"per_page": 100, "page": page})
        request = Request(
            f"{base}?{query}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "nexus-training-framework-monitor",
            },
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page_jobs = payload.get("jobs", [])
        jobs.extend(page_jobs)
        if len(page_jobs) < 100:
            return jobs
        page += 1


def monitor(
    api_url: str,
    repository: str,
    run_id: str,
    attempt: str,
    token: str,
    expected_shards: int,
    poll_seconds: int,
    timeout_seconds: int,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    poll_number = 0
    while time.monotonic() < deadline:
        poll_number += 1
        try:
            jobs = fetch_jobs(api_url, repository, run_id, attempt, token)
        except HTTPError as exc:
            # Permission/configuration errors should not be disguised as transient network issues.
            print(f"GitHub API returned HTTP {exc.code}: {exc.reason}", file=sys.stderr)
            return False
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"poll {poll_number}: transient API/network error: {exc}; retrying", file=sys.stderr)
            time.sleep(min(poll_seconds, max(1, int(deadline - time.monotonic()))))
            continue

        recognized, missing, all_succeeded = evaluate_jobs(jobs, expected_shards)
        completed = sum(job.get("status") == "completed" for job in recognized.values())
        print(
            f"poll {poll_number}: {completed}/{expected_shards} workers completed; "
            f"{len(missing)} worker job(s) not yet visible"
        )
        if not missing and completed == expected_shards:
            failed = [
                f"shard {shard:02d}: {job.get('conclusion')}"
                for shard, job in sorted(recognized.items())
                if job.get("conclusion") != "success"
            ]
            if failed:
                print("worker failure(s): " + "; ".join(failed), file=sys.stderr)
                return False
            if all_succeeded:
                print(f"all {expected_shards} workers succeeded")
                return True
        time.sleep(min(poll_seconds, max(1, int(deadline - time.monotonic()))))

    print(f"timed out waiting for {expected_shards} worker jobs", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-shards", type=int, default=19)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=20_400)
    args = parser.parse_args()
    if args.expected_shards < 1 or args.poll_seconds < 1 or args.timeout_seconds < 1:
        parser.error("all numeric arguments must be positive")

    env_names = ("GITHUB_API_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_TOKEN")
    missing_env = [name for name in env_names if not os.environ.get(name)]
    if missing_env:
        print("missing required environment variables: " + ", ".join(missing_env), file=sys.stderr)
        return 2

    succeeded = monitor(
        api_url=os.environ["GITHUB_API_URL"],
        repository=os.environ["GITHUB_REPOSITORY"],
        run_id=os.environ["GITHUB_RUN_ID"],
        attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        token=os.environ["GITHUB_TOKEN"],
        expected_shards=args.expected_shards,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
