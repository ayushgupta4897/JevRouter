"""Does the `complexity` fix actually route better? Decision accuracy, before vs. after.

Real Jev decisions only -- no model answers, so the whole run costs pennies of judge calls. Three
arms on every item:

* ``switchyard``  Switchyard's packaged coding-agent rubric *and* jevjudge's original
                  harness-worded success question -- exactly what `complexity` did before the fix.
* ``general``     the new default rubric (routerctl/rubrics.py), nothing route-specific.
* ``criteria``    the general rubric plus the route's plain-English `weak_when` / `strong_when`.

Two item sets:

* the original 54 complexity items (code review, finance, SQL) whose misroutes motivated the fix
  -- the wording was written *after* reading those failures, so treat these numbers as in-sample;
* 48 held-out items (experiments/datasets/complexity-holdout.yaml), written before the new rubric
  was ever evaluated, including an ops/analytics domain no criteria were written for -- the honest
  measure of generalisation.

    set -a; . ./.env; set +a
    .venv/bin/python experiments/complexity_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "routerctl" / "src"))
sys.path.insert(0, str(ROOT.parent / "jevjudge" / "src"))

from jevjudge import compiler as jev_compiler  # noqa: E402
from jevjudge.cascade import build_judge_from_env  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.decide import decide  # noqa: E402
from routerctl.messages import to_libsy_messages  # noqa: E402
from routerctl.schema import Route  # noqa: E402

ORIGINAL_SUCCESS_WORDING = (
    "SUCCESS: the EFFICIENT (cheaper) agent completes the whole task correctly on one "
    "fresh run under the actual harness, tools, and budget, as judged by the final "
    "verifier. Use only evidence in the task and the capability rules in "
    "judge_instructions. Answer with the probability of SUCCESS."
)
NEW_SUCCESS_WORDING = jev_compiler.PROFILES["CapabilityClassifierDecision"]["instructions"]["p_solve"]

# Per-domain route settings: the thresholds the experiment teams already use, and the criteria a
# team would plausibly write for its own traffic (the `criteria` arm only).
DOMAINS = {
    "sql": {
        "base_threshold": 0.5,
        "weak_when": "A single-table select, filter, sort, count, sum, insert, update, delete, or rename.",
        "strong_when": "The query needs window functions, several joins with deduplication or double-counting risk, "
                       "recursive CTEs, time-window or funnel logic, gaps-and-islands, or careful NULL and edge-case handling.",
    },
    "code": {
        "base_threshold": 0.5,
        "weak_when": "A naming, formatting, style, docstring, import, or lint-level issue.",
        "strong_when": "The issue involves concurrency, idempotency, caching or consistency across processes, security, "
                       "money or numeric precision, failure and retry behaviour, or data correctness under load.",
    },
    "finance": {
        "base_threshold": 0.6,
        "weak_when": "A single arithmetic step: a percentage, growth rate, conversion, runway, interest, average, or total.",
        "strong_when": "Multi-period or multi-scenario modelling, reconciling metrics that move in different directions, "
                       "dilution or cap-table maths, hedging, or cohort and unit-economics analysis.",
    },
    "ops": {"base_threshold": 0.5},  # held-out only: no criteria written, on purpose
}
ORIGINAL_SETS = {"code-review": "code", "financial-analysis": "finance", "sql-query-generation": "sql"}


def route(domain: str, arm: str) -> Route:
    cfg = DOMAINS[domain]
    policy = {"policy": "complexity", "base_threshold": cfg["base_threshold"], "threshold_step": 0.1,
              "models": {"weak": {"id": "weak", "client": "openrouter"}, "strong": {"id": "strong", "client": "openrouter"}}}
    if arm == "switchyard":
        policy["rubric"] = "coding_agent"
    if arm == "criteria":
        policy.update({k: cfg[k] for k in ("weak_when", "strong_when") if k in cfg})
    return Route.model_validate({"name": f"eval/{domain}", **policy})


def items() -> list[dict]:
    out = []
    for name, domain in ORIGINAL_SETS.items():
        for it in yaml.safe_load((ROOT / "datasets" / f"{name}.yaml").read_text()):
            out.append({**it, "domain": domain, "set": "original"})
    for it in yaml.safe_load((ROOT / "datasets" / "complexity-holdout.yaml").read_text()):
        out.append({**it, "set": "holdout"})
    return out


async def run_arm(arm: str, dataset: list[dict], judge, sem: asyncio.Semaphore) -> list[dict]:
    compiled = {d: compile_route_algorithm(route(d, arm)) for d in DOMAINS}

    async def one(it: dict) -> dict:
        async with sem:
            request = {"model": "eval", "stream": False, "messages": to_libsy_messages([{"role": "user", "content": it["text"]}])}
            decision = await decide(compiled[it["domain"]], judge, request)
            label = "weak" if decision.selected_model == "weak" else "strong"
            return {"id": it["id"], "set": it["set"], "domain": it["domain"], "expected": it["expected"], "label": label,
                    "correct": label == it["expected"], "confidence": decision.judge_confidence, "judge_error": decision.judge_error}

    # jevjudge's success question is process-global; set it for this arm, and run arms one at a time.
    jev_compiler.PROFILES["CapabilityClassifierDecision"]["instructions"]["p_solve"] = (
        ORIGINAL_SUCCESS_WORDING if arm == "switchyard" else NEW_SUCCESS_WORDING
    )
    return list(await asyncio.gather(*(one(it) for it in dataset)))


def accuracy(rows: list[dict], **where) -> tuple[int, int]:
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return sum(r["correct"] for r in sel), len(sel)


async def main() -> None:
    dataset, judge, sem = items(), build_judge_from_env(), asyncio.Semaphore(8)
    results = {}
    for arm in ("switchyard", "general", "criteria"):
        results[arm] = await run_arm(arm, dataset, judge, sem)
        print(f"done: {arm}", flush=True)
    results["general_repeat"] = await run_arm("general", dataset, judge, sem)  # judge stability check
    await judge.aclose()

    def fmt(c):
        return f"{c[0]}/{c[1]} ({c[0] / c[1]:.0%})" if c[1] else "-"
    print(f"\n{'':<24}{'switchyard':>16}{'general':>16}{'criteria':>16}")
    for label, where in [("original, all", {"set": "original"}),
                         *[(f"  original {d}", {"set": "original", "domain": d}) for d in ("code", "finance", "sql")],
                         ("HOLDOUT, all", {"set": "holdout"}),
                         *[(f"  holdout {d}", {"set": "holdout", "domain": d}) for d in ("code", "finance", "sql", "ops")]]:
        print(f"{label:<24}" + "".join(f"{fmt(accuracy(results[a], **where)):>16}" for a in ("switchyard", "general", "criteria")))
    for arm in ("switchyard", "general", "criteria"):
        rows = results[arm]
        under = sum(1 for r in rows if r["expected"] == "strong" and r["label"] == "weak")
        over = sum(1 for r in rows if r["expected"] == "weak" and r["label"] == "strong")
        print(f"{arm:<11} hard->weak (quality risk): {under:<3} easy->strong (wasted spend): {over}")
    same = sum(a["label"] == b["label"] for a, b in zip(results["general"], results["general_repeat"]))
    print(f"judge stability: general arm gave the same decision on {same}/{len(dataset)} items when re-run")
    errors = sum(1 for arm in results.values() for r in arm if r["judge_error"])
    print(f"judge errors: {errors}")
    out = ROOT / "results" / "complexity-eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
