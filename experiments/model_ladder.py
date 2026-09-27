"""Deferent across a ten-model ladder, ultra-cheap to strong, all through OpenRouter.

The question here is about our own router, not anyone else's: when a route's tiers run from
models costing cents per million tokens up to frontier-class ones, how close does routing get to
the strong model's quality while paying mostly cheap-model prices?

The 12 route policies, datasets, thresholds and Jev judge are the same as run_experiments.py.
Each route's tiers are swapped for models from this ladder (always cheap -> strong within a
route), 10 models in all, OpenRouter choosing the host:

    ultra-cheap  openai/gpt-oss-20b, qwen/qwen3.7-flash, z-ai/glm-5.3-flash,
                 mistralai/ministral-8b-2512, deepseek/deepseek-v4.1-flash
    cheap        openai/gpt-6-luna, deepseek/deepseek-v4-pro
    mid          google/gemini-3.8-flash
    strong       anthropic/claude-sonnet-5, openai/gpt-6-sol

Every item is routed by Deferent and answered by the model it picked (cost = OpenRouter's billed
`usage.cost`). On every third item (72 in all) the same prompt also goes to two baselines:

    always-cheapest   the route's cheapest tier for everything (what quality costs you without routing)
    always-strongest  the route's strongest tier for everything (what you'd pay without routing)

All three answers are graded for adequacy by Jev and scored 1-5 side by side, shuffled, by
Gemini 3.1 Pro (not in any route's roster; two Gemini Flash models are, which is a family
closeness worth knowing about).

    set -a; . ./.env; set +a
    .venv/bin/python experiments/model_ladder.py --max-budget-usd 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from multiprovider import Answer, Budget, Item, jev_grade, openrouter, side_by_side  # noqa: E402
from run_experiments import EXPERIMENTS, actual_label, load_dataset  # noqa: E402

from jevjudge.cascade import build_judge_from_env  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.decide import decide  # noqa: E402
from routerctl.messages import to_libsy_messages  # noqa: E402
from routerctl.schema import TeamFile  # noqa: E402

OUT = ROOT / "results" / "model-ladder"

# OpenRouter list prices, USD per 1M tokens (input, output), checked 2026-09-27.
PRICE = {
    "openai/gpt-oss-20b": (0.018, 0.09),
    "qwen/qwen3.7-flash": (0.03, 0.13),
    "z-ai/glm-5.3-flash": (0.045, 0.14),
    "mistralai/ministral-8b-2512": (0.15, 0.15),
    "deepseek/deepseek-v4.1-flash": (0.035, 0.29),
    "openai/gpt-6-luna": (0.10, 0.50),
    "deepseek/deepseek-v4-pro": (0.348, 0.696),
    "google/gemini-3.8-flash": (0.75, 3.75),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "openai/gpt-6-sol": (2.00, 10.00),
}

# Per experiment: the original tier's model id -> this ladder's model. Cheap -> strong in every route.
LADDER = {
    "support-ticket-triage": {"gpt-5.6-luna": "qwen/qwen3.7-flash", "gpt-5.6-terra": "openai/gpt-6-luna",
                              "gpt-5.6-sol": "google/gemini-3.8-flash", "gpt-6-astra": "anthropic/claude-sonnet-5"},
    "exec-assistant": {"gpt-5.6-luna": "openai/gpt-oss-20b", "gpt-5.6-terra": "z-ai/glm-5.3-flash",
                       "gpt-5.6-sol": "deepseek/deepseek-v4-pro", "gpt-6-astra": "openai/gpt-6-sol"},
    "post-call-transcript": {"gpt-5.6-luna": "mistralai/ministral-8b-2512", "gpt-5.6-sol": "deepseek/deepseek-v4-pro"},
    "meeting-notes": {"gpt-5.6-luna": "z-ai/glm-5.3-flash", "gpt-5.6-terra": "openai/gpt-6-luna"},
    "legal-contract-review": {"gpt-5.6-luna": "qwen/qwen3.7-flash", "gpt-6-astra": "anthropic/claude-sonnet-5"},
    "resume-screening": {"gpt-5.6-luna": "deepseek/deepseek-v4.1-flash", "gpt-5.6-sol": "google/gemini-3.8-flash"},
    "coding-agent": {"gpt-5.6-terra": "deepseek/deepseek-v4.1-flash", "gpt-6-astra": "anthropic/claude-sonnet-5"},
    "security-incident-response": {"gpt-5.6-sol": "deepseek/deepseek-v4-pro", "gpt-6-astra": "openai/gpt-6-sol"},
    "code-review": {"gpt-5.6-luna": "openai/gpt-oss-20b", "gpt-5.6-sol": "deepseek/deepseek-v4-pro"},
    "financial-analysis": {"gpt-5.6-terra": "openai/gpt-6-luna", "gpt-6-astra": "openai/gpt-6-sol"},
    "sql-query-generation": {"gpt-5.6-luna": "qwen/qwen3.7-flash", "gpt-5.6-terra": "deepseek/deepseek-v4-pro"},
    "general-assistant": {"gpt-5.6-terra": "mistralai/ministral-8b-2512", "gpt-6-astra": "google/gemini-3.8-flash"},
}


def laddered_team(name: str) -> TeamFile:
    raw = yaml.safe_load((ROOT / "teams" / f"{name}.yaml").read_text())
    for route in raw["routes"]:
        for ref in route["models"].values():
            ref["id"], ref["client"] = LADDER[name][ref["id"]], "openrouter"
    return TeamFile.model_validate(raw)


async def run_experiment(name: str, route_name: str, judge, http: httpx.AsyncClient, budget: Budget, sem: asyncio.Semaphore) -> dict:
    route = next(r for r in laddered_team(name).routes if r.name == route_name)
    compiled = compile_route_algorithm(route)
    roster = sorted(compiled.model_by_id, key=lambda m: PRICE[m][1])
    cheapest, strongest = roster[0], roster[-1]

    async def one(idx: int, entry: dict) -> Item:
        async with sem:
            prompt = entry["text"]
            decision = await decide(compiled, judge, {"model": route_name, "stream": False, "messages": to_libsy_messages([{"role": "user", "content": prompt}])})
            label = actual_label(compiled, decision)
            item = Item(entry["id"], entry["expected"], label, label == entry["expected"], decision.judge_confidence)
            item.routed = await openrouter(http, budget, "deferent", decision.selected_model, prompt, decision.selected_extra_body)
            for model, extra in zip(decision.fallback_model_ids, decision.fallback_extra_bodies):
                if item.routed.error in (None, "empty visible answer", "budget exhausted"):
                    break  # answered, or the model itself came back empty -- not a provider failure
                failed = item.routed.model
                item.routed = await openrouter(http, budget, "deferent", model, prompt, extra)
                item.routed.fallback_from = failed
            await jev_grade(judge, prompt, item.routed)
            if idx % 3 == 0:
                item.head_to_head = list(await asyncio.gather(
                    openrouter(http, budget, "always-cheapest", cheapest, prompt),
                    openrouter(http, budget, "always-strongest", strongest, prompt),
                ))
                for answer in item.head_to_head:
                    await jev_grade(judge, prompt, answer)
                await side_by_side(http, budget, prompt, [item.routed, *item.head_to_head], seed=zlib.crc32(entry["id"].encode()))
            return item

    items = await asyncio.gather(*(one(i, e) for i, e in enumerate(load_dataset(name))))
    return {
        "name": name, "route_name": route_name, "policy": route.policy.policy, "roster": roster,
        "accuracy": sum(i.correct for i in items) / len(items),
        "routed_cost_usd": sum(i.routed.cost_usd for i in items if i.routed),
        "models_used": {m: sum(1 for i in items if i.routed and i.routed.model == m) for m in roster},
        "items": [asdict(i) for i in items],
    }


def summarize(results: list[dict]) -> dict:
    print(f"\n{'experiment':<28}{'policy':<12}{'acc':>6}{'routed $':>10}  models used")
    for r in results:
        used = ", ".join(f"{m.split('/')[1]}×{n}" for m, n in r["models_used"].items() if n)
        print(f"{r['name']:<28}{r['policy']:<12}{r['accuracy']:>6.0%}{r['routed_cost_usd']:>10.4f}  {used}")

    arms: dict[str, list[dict]] = {"always-cheapest": [], "deferent": [], "always-strongest": []}
    for r in results:
        for i in r["items"]:
            if i["head_to_head"]:
                for a in [i["routed"], *i["head_to_head"]]:
                    arms[a["router"]].append(a)
    print("\nSame prompts, three ways (every third item of every experiment):")
    table = {}
    for arm, answers in arms.items():
        scores = [a["score"] for a in answers if a["score"] is not None]
        jev = [a["jev_adequate"] for a in answers if a["jev_adequate"] is not None]
        table[arm] = {
            "prompts": len(answers), "cost_usd": sum(a["cost_usd"] for a in answers),
            "mean_score": statistics.mean(scores) if scores else None, "score_ge_4": sum(s >= 4 for s in scores), "scored": len(scores),
            "jev_adequate": [sum(jev), len(jev)], "failed_or_empty": sum(1 for a in answers if a["error"]),
        }
        t = table[arm]
        print(f"  {arm:<18} cost=${t['cost_usd']:.4f}  mean score={t['mean_score']:.2f}  >=4: {t['score_ge_4']}/{t['scored']}  "
              f"Jev adequate={t['jev_adequate'][0]}/{t['jev_adequate'][1]}  failed/empty={t['failed_or_empty']}")
    all_models: dict[str, int] = {}
    for r in results:
        for m, n in r["models_used"].items():
            all_models[m] = all_models.get(m, 0) + n
    return {"head_to_head": table, "models_used": all_models, "routed_cost_usd": sum(r["routed_cost_usd"] for r in results)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-budget-usd", type=float, default=4.0)
    parser.add_argument("--only", help="Comma-separated experiment names (default: all 12).")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out", default=str(OUT), help="Results directory (default: results/model-ladder).")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget, judge, sem = Budget(args.max_budget_usd), build_judge_from_env(), asyncio.Semaphore(args.concurrency)
    only = set(args.only.split(",")) if args.only else None
    results = []
    async with httpx.AsyncClient() as http:
        for name, route_name in EXPERIMENTS:
            if only and name not in only:
                continue
            print(f"== {name} -- spent ${budget.spent:.3f} of ${budget.cap:.2f}", flush=True)
            result = await run_experiment(name, route_name, judge, http, budget, sem)
            (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
            results.append(result)
    await judge.aclose()
    summary = summarize(results)
    summary["total_spend_usd"] = budget.spent
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntotal spend: ${budget.spent:.4f} of ${budget.cap:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
