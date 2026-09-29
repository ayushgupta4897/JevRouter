"""Where a decision's time goes, and what connection reuse is worth.

Measures, against the real Jev API from wherever this runs:

1. the bare network round trip to Jev's host (a request that does no Jev work);
2. a full decision on a warm connection, and how much of it is our own code;
3. 8 concurrent decisions, first on a cold pool and then on a warm one;
4. decisions after idle gaps, with httpx's default 5 s keep-alive vs. our 300 s pool.

Numbers depend on where you run it: the network round trip is the biggest single component, so
deploy close to Jev's region and re-run this from there.

    set -a; . ./.env; set +a
    .venv/bin/python experiments/judge_latency.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "routerctl" / "src"))
sys.path.insert(0, str(ROOT.parent / "jevjudge" / "src"))

from jevjudge.cascade import Judge, JudgeConfig  # noqa: E402
from jevjudge.client import JevClient, JevClientConfig  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.decide import decide  # noqa: E402
from routerctl.messages import to_libsy_messages  # noqa: E402
from routerctl.schema import Route  # noqa: E402

ROUTE = compile_route_algorithm(Route.model_validate({"name": "e", "policy": "escalation",
                                                      "models": {"weak": {"id": "m1", "client": "x"}, "strong": {"id": "m2", "client": "x"}}}))
REQUEST = {"model": "e", "stream": False, "messages": to_libsy_messages([{"role": "user", "content": "Customer: I was charged twice for invoice 4417."}])}


def judge_with(keepalive_s: float | None) -> Judge:
    config = JevClientConfig.from_env()
    http = httpx.AsyncClient(timeout=config.timeout_s) if keepalive_s is None else None  # None: httpx defaults (5 s)
    if keepalive_s is not None:
        config.keepalive_s = keepalive_s
    return Judge(JevClient(config, http=http), JudgeConfig.from_env())


async def one(judge: Judge) -> float:
    return (await decide(ROUTE, judge, REQUEST)).latency_ms


async def main() -> None:
    out: dict = {}
    base = JevClientConfig.from_env().base_url

    async with httpx.AsyncClient() as h:
        await h.get(base + "/", timeout=20)
        rtt = []
        for _ in range(6):
            t = time.perf_counter()
            await h.get(base + "/", timeout=20)
            rtt.append((time.perf_counter() - t) * 1000)
    out["network_round_trip_ms_p50"] = round(statistics.median(rtt))

    judge = judge_with(300)
    cold = await one(judge)
    warm = [await one(judge) for _ in range(6)]
    out["decision_cold_ms"] = round(cold)
    out["decision_warm_ms_p50"] = round(statistics.median(warm))
    await judge.aclose()

    judge = judge_with(300)
    batches = []
    for _ in range(3):
        lat = await asyncio.gather(*(one(judge) for _ in range(8)))
        batches.append(round(statistics.median(lat)))
    out["concurrent_8_batches_p50_ms"] = batches  # first batch opens 8 connections
    await judge.aclose()

    for label, keepalive in (("httpx_default_5s", None), ("pool_300s", 300.0)):
        judge = judge_with(keepalive)
        await one(judge)
        gaps = {}
        for idle in (10, 30):
            await asyncio.sleep(idle)
            gaps[f"after_{idle}s_idle_ms"] = round(await one(judge))
        out[label] = gaps
        await judge.aclose()

    for key, value in out.items():
        print(f"{key:<32} {value}")
    path = ROOT / "results" / "judge-latency.json"
    path.write_text(json.dumps({"run_date": time.strftime("%Y-%m-%d"), "measured_from": "Claude Code cloud container (via egress proxy)", **out}, indent=2))
    print(f"-> {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
