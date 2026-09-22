#!/usr/bin/env python3
"""Measure jevjudge round-trip latency for Switchyard-shaped judge requests.

    python scripts/bench_judge.py --url http://127.0.0.1:8090 --n 50 --mode capability

Works against the mock stack (scripts/e2e.sh leaves nothing running; start jevjudge yourself)
and against the real TypeSafe API (start jevjudge with TYPESAFE_API_KEY set). Reports p50/p95
and the average Jev-side latency jevjudge measured, plus billable input tokens per verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
SWITCHYARD_PROMPTS = HERE.parent / "docs" / "switchyard-prompts"

CAPABILITY_TASKS = [
    "Add a --verbose flag to the CLI that prints each processed file name. Tests are in tests/test_cli.py.",
    "Reverse engineer the undocumented .dat format produced by the legacy tool and write a reader that matches its checksums.",
    "Fix the failing unit test in utils/date.py; the expected output is given in the test.",
    "Migrate the auth service from sessions to JWT across the API, workers, and admin UI without downtime.",
    "Rename the variable `tmp` to `buffer` in parser.py.",
    "Implement a rate limiter matching the behaviour of the production one, whose config we do not have.",
    "Write a SQL query returning the top 10 customers by revenue last quarter; schema is in schema.sql.",
    "Find and fix the intermittent deadlock in the job scheduler.",
]


def load_prompt(name: str) -> str:
    path = SWITCHYARD_PROMPTS / name
    return path.read_text() if path.exists() else "You are a judge. Return exactly one JSON object matching the response schema."


def capability_request(task: str) -> dict:
    schema = json.loads((SWITCHYARD_PROMPTS / "capability-classifier.schema.json").read_text())
    return {
        "model": "jev-latest",
        "messages": [{"role": "system", "content": load_prompt("capability-classifier.prompt.md")}, {"role": "user", "content": task}],
        "max_tokens": 4096,
        "response_format": schema,
    }


def escalation_request(task: str) -> dict:
    schema = json.loads((SWITCHYARD_PROMPTS / "escalation.schema.json").read_text())
    transcript = (
        "Conversation turn 6; showing the last 6 of 12 messages after the task framing.\n"
        f"[user (task)] {task}\n[assistant] tool_call bash(pytest -q)\n[tool] 3 failed, 10 passed\n"
        "[assistant] editing config.py\n[assistant] tool_call bash(pytest -q)\n[tool] 3 failed, 10 passed (same errors)\n"
        "[assistant] editing config.py again\n[assistant] tool_call bash(pytest -q)\n[tool] 3 failed, 10 passed (same errors)"
    )
    return {
        "model": "jev-latest",
        "messages": [{"role": "system", "content": load_prompt("escalation.prompt.md")}, {"role": "user", "content": transcript}],
        "max_tokens": 4096,
        "response_format": schema,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--mode", choices=["capability", "escalation"], default="capability")
    args = ap.parse_args()

    build = capability_request if args.mode == "capability" else escalation_request
    reqs = [build(CAPABILITY_TASKS[i % len(CAPABILITY_TASKS)]) for i in range(args.n)]
    sem = asyncio.Semaphore(args.concurrency)
    lat: list[float] = []
    jev_lat: list[float] = []
    tokens: list[int] = []
    verdicts: list[dict] = []

    async with httpx.AsyncClient(timeout=60) as http:
        async def one(req: dict) -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await http.post(f"{args.url}/v1/chat/completions", json=req)
                lat.append((time.perf_counter() - t0) * 1000)
                r.raise_for_status()
                body = r.json()
                verdicts.append(json.loads(body["choices"][0]["message"]["content"]) if body["choices"][0]["message"]["content"].startswith("{") else {"abstain": True})
                meta = body.get("jevjudge") or {}
                if "jev_latency_ms" in meta:
                    jev_lat.append(float(meta["jev_latency_ms"]))
                tokens.append(int(body.get("usage", {}).get("prompt_tokens", 0)))

        t0 = time.perf_counter()
        await asyncio.gather(*(one(r) for r in reqs))
        wall = time.perf_counter() - t0

    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
    print(f"mode={args.mode} n={args.n} concurrency={args.concurrency} wall={wall:.2f}s")
    print(f"round-trip ms: p50={p(0.5):.0f} p95={p(0.95):.0f} max={lat[-1]:.0f}")
    if jev_lat:
        print(f"jev-side ms:   mean={statistics.mean(jev_lat):.0f} p95={sorted(jev_lat)[int(0.95*len(jev_lat))-1]:.0f}")
    if tokens:
        mean_tokens = statistics.mean(tokens)
        print(f"input tokens/verdict: mean={mean_tokens:.0f}  -> ${mean_tokens * 0.042 / 1e6:.6f} per verdict at $0.042/M (output free)")
    print("sample verdicts:")
    for v in verdicts[: min(4, len(verdicts))]:
        print("  ", json.dumps(v)[:160])


if __name__ == "__main__":
    asyncio.run(main())
