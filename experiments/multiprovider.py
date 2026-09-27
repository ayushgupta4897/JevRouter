"""Multi-provider routing, and a head-to-head against OpenRouter's own routers.

Two questions, both about why build our own router rather than use someone else's:

1. **Does routing still pay when the tiers span providers?** The same 12 route policies as
   run_experiments.py (same datasets, same Jev judge, same thresholds), but every tier swapped for
   a model on a *different* inference provider, each pinned to that provider through the route's
   own `extra_body` -- which decision-only mode now hands to the gateway with every decision:

       gpt-5.6-luna  -> deepseek/deepseek-v4.1-flash  Fireworks, then Together
       gpt-5.6-terra -> google/gemini-3.8-flash       Google AI Studio, then Vertex
       gpt-5.6-sol   -> anthropic/claude-sonnet-5      Anthropic
       gpt-6-astra   -> anthropic/claude-opus-5.5      Anthropic

   Deferent decides (real Jev); this script plays the gateway: it calls the decision's model with
   its extra_body and, like a real gateway, moves down the decision's fallbacks if a provider
   errors. Cost is OpenRouter's own billed `usage.cost`, not an estimate.

2. **Head-to-head.** A fixed subset of the same prompts (every third item: 6 per experiment, 72
   total) sent four ways: Deferent, `openrouter/auto`, `typesafe/jev-router`, and "always the
   route's top tier" (no routing at all -- the measured baseline, not an extrapolation). Every
   answer gets Jev's adequacy verdict, and all four answers to a prompt are scored 1-5 side by
   side, in shuffled order, by Gemini 3.1 Pro -- a model none of the contestants uses, because Jev
   and the Jev Router come from the same company, and our strong tiers are Claude models.

    set -a; . ./.env; set +a
    .venv/bin/python experiments/multiprovider.py --max-budget-usd 8
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_experiments import EXPERIMENTS, actual_label, judge_adequate, load_dataset  # noqa: E402

from jevjudge.cascade import build_judge_from_env  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.decide import decide  # noqa: E402
from routerctl.messages import to_libsy_messages  # noqa: E402
from routerctl.schema import TeamFile  # noqa: E402

OPENROUTER = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4000  # several of these are reasoning models; 1200 left some hard answers empty in a probe
OUT = ROOT / "results" / "multiprovider"


def pinned(model: str, providers: list[str]) -> dict:
    # Restrict OpenRouter to these providers, in this order: provider failover *we* chose, not
    # whichever host OpenRouter would pick.
    return {"id": model, "client": "openrouter", "extra_body": {"provider": {"order": providers, "allow_fallbacks": False}}}


# Old tier -> new (model, providers, input $/1M, output $/1M). Prices are the first provider's own
# OpenRouter endpoint price, checked 2026-09-27 -- used only for tier ordering and the
# extrapolated baseline; every routed call's cost is OpenRouter's billed usage.cost.
# (A first run used gpt-oss-120b@Together as the mid tier: it is cheaper than the "cheap" tier,
# which inverted two routes' ordering. See docs/DECISION.md section 13.)
TIERS = {
    "gpt-5.6-luna": ("deepseek/deepseek-v4.1-flash", ["fireworks", "together"], 0.22, 0.66),
    "gpt-5.6-terra": ("google/gemini-3.8-flash", ["google-ai-studio", "google-vertex"], 0.75, 3.75),
    "gpt-5.6-sol": ("anthropic/claude-sonnet-5", ["anthropic"], 2.00, 10.00),
    "gpt-6-astra": ("anthropic/claude-opus-5.5", ["anthropic"], 4.00, 20.00),
}
PRICE = {model: (inp, out) for model, _, inp, out in TIERS.values()}
GRADER = "google/gemini-3.1-pro-preview"  # a contestant in nothing
H2H_ROUTERS = ["openrouter/auto", "typesafe/jev-router"]


def remapped_team(name: str) -> TeamFile:
    raw = yaml.safe_load((ROOT / "teams" / f"{name}.yaml").read_text())
    for route in raw["routes"]:
        for ref in route["models"].values():
            model, providers, _, _ = TIERS[ref["id"]]
            ref.update(copy.deepcopy(pinned(model, providers)))
    return TeamFile.model_validate(raw)


class Budget:
    def __init__(self, cap: float):
        self.cap, self.spent = cap, 0.0

    def ok(self) -> bool:
        return self.spent < self.cap


@dataclass
class Answer:
    router: str
    model: str | None
    provider: str | None
    cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: float
    text: str
    error: str | None = None
    fallback_from: str | None = None  # set when the gateway had to fall back from this model
    jev_adequate: bool | None = None
    score: int | None = None  # 1-5 from the side-by-side grader (head-to-head prompts only)


@dataclass
class Item:
    item_id: str
    expected: str
    actual_label: str
    correct: bool
    judge_confidence: float | None
    routed: Answer | None = None
    head_to_head: list[Answer] = field(default_factory=list)  # openrouter/auto, jev-router, always-top-tier


async def openrouter(http: httpx.AsyncClient, budget: Budget, router_label: str, model: str, prompt: str, extra_body: dict | None = None) -> Answer:
    if not budget.ok():
        return Answer(router_label, None, None, 0.0, 0, 0, 0.0, "", error="budget exhausted")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS, "usage": {"include": True}, **(extra_body or {})}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "deferent-experiments"}
    started = time.perf_counter()
    try:
        resp = await http.post(f"{OPENROUTER}/chat/completions", json=body, headers=headers, timeout=240)
        payload = resp.json()
    except (httpx.HTTPError, ValueError) as error:
        return Answer(router_label, model, None, 0.0, 0, 0, 0.0, "", error=repr(error))
    latency = (time.perf_counter() - started) * 1000
    if resp.status_code != 200 or "choices" not in payload:
        return Answer(router_label, model, None, 0.0, 0, 0, latency, "", error=f"HTTP {resp.status_code}: {json.dumps(payload)[:300]}")
    usage = payload.get("usage") or {}
    cost = float(usage.get("cost") or 0.0)
    budget.spent += cost
    text = payload["choices"][0]["message"].get("content") or ""
    answer = Answer(router_label, payload.get("model"), payload.get("provider"), cost, int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0), latency, text)
    if not text.strip():
        answer.error = "empty visible answer"
    return answer


async def jev_grade(judge, task: str, answer: Answer) -> None:
    if answer.error or not answer.text.strip():
        answer.jev_adequate = False  # a failed or empty answer is not adequate
        return
    try:
        answer.jev_adequate = await judge_adequate(judge, task, answer.text)
    except Exception:  # noqa: BLE001 -- a grading failure leaves the field None, it doesn't abort the run
        answer.jev_adequate = None


SIDE_BY_SIDE = """Several responses to the same task follow, labelled {labels}. Score each from 1 to 5:
5 = excellent: correct, complete, genuinely useful; 4 = good, minor gaps; 3 = acceptable but
noticeably weaker; 2 = significant errors or omissions; 1 = wrong, off-topic, or empty.
Judge substance, not length or formatting. Be strict about correctness.

Task:
{task}

{responses}

Reply with only a JSON object mapping each label to its score, e.g. {example}."""


async def side_by_side(http: httpx.AsyncClient, budget: Budget, task: str, answers: list[Answer], seed: int) -> None:
    import random
    order = list(range(len(answers)))
    random.Random(seed).shuffle(order)  # position bias: no contestant is always "A"
    labels = [chr(ord("A") + i) for i in range(len(answers))]
    blocks = "\n\n".join(f"=== Response {labels[k]} ===\n{(answers[i].text or '(empty response)')[:10000]}" for k, i in enumerate(order))
    prompt = SIDE_BY_SIDE.format(labels=", ".join(labels), task=task, responses=blocks, example=json.dumps({l: 4 for l in labels}))
    graded = await openrouter(http, budget, "grader", GRADER, prompt, {"reasoning": {"effort": "low"}, "temperature": 0})
    raw = graded.text.strip().strip("`").removeprefix("json").strip()
    try:
        scores = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        return
    for k, i in enumerate(order):
        value = scores.get(labels[k])
        answers[i].score = int(value) if isinstance(value, (int, float)) else None


async def run_experiment(name: str, route_name: str, judge, http: httpx.AsyncClient, budget: Budget, sem: asyncio.Semaphore) -> dict:
    team = remapped_team(name)
    route = next(r for r in team.routes if r.name == route_name)
    compiled = compile_route_algorithm(route)
    roster = sorted(compiled.model_by_id, key=lambda m: PRICE[m][1])
    frontier = compiled.model_by_id[roster[-1]]
    dataset = load_dataset(name)

    async def one(idx: int, entry: dict) -> Item:
        async with sem:
            prompt = entry["text"]
            decision = await decide(compiled, judge, {"model": route_name, "stream": False, "messages": to_libsy_messages([{"role": "user", "content": prompt}])})
            label = actual_label(compiled, decision)
            item = Item(entry["id"], entry["expected"], label, label == entry["expected"], decision.judge_confidence)
            # The gateway's side: call exactly what the decision says, extra_body included.
            item.routed = await openrouter(http, budget, "deferent", decision.selected_model, prompt, decision.selected_extra_body)
            for model, extra in zip(decision.fallback_model_ids, decision.fallback_extra_bodies):
                if item.routed.error in (None, "empty visible answer", "budget exhausted"):
                    break  # answered, or the model itself came back empty -- not a provider failure
                failed = item.routed.model
                item.routed = await openrouter(http, budget, "deferent", model, prompt, extra)
                item.routed.fallback_from = failed
            await jev_grade(judge, prompt, item.routed)
            if idx % 3 == 0:  # head-to-head subset: 6 prompts per experiment, spread across the dataset
                rivals = [openrouter(http, budget, router, router, prompt) for router in H2H_ROUTERS]
                rivals.append(openrouter(http, budget, "always-top-tier", frontier.id, prompt, frontier.extra_body))
                item.head_to_head = list(await asyncio.gather(*rivals))
                for answer in item.head_to_head:
                    await jev_grade(judge, prompt, answer)
                await side_by_side(http, budget, prompt, [item.routed, *item.head_to_head], seed=zlib.crc32(entry["id"].encode()))
            return item

    items = await asyncio.gather(*(one(i, e) for i, e in enumerate(dataset)))
    priced = [i for i in items if i.routed and not i.routed.error]
    real = sum(i.routed.cost_usd for i in priced)
    fin, fout = PRICE[frontier.id]
    frontier_equiv = sum((i.routed.input_tokens * fin + i.routed.output_tokens * fout) / 1e6 for i in priced)
    return {
        "name": name, "route_name": route_name, "policy": route.policy.policy, "roster": roster,
        "accuracy": sum(i.correct for i in items) / len(items),
        "real_cost_usd": real, "frontier_equivalent_usd": frontier_equiv,
        "savings_pct": (1 - real / frontier_equiv) * 100 if frontier_equiv else 0.0,
        "items": [asdict(i) for i in items],
    }


def summarize(results: list[dict]) -> dict:
    total_real = sum(r["real_cost_usd"] for r in results)
    total_frontier = sum(r["frontier_equivalent_usd"] for r in results)
    print(f"\n{'experiment':<28}{'policy':<12}{'acc':>6}{'real $':>10}{'frontier $':>12}{'savings':>9}")
    for r in results:
        print(f"{r['name']:<28}{r['policy']:<12}{r['accuracy']:>6.0%}{r['real_cost_usd']:>10.4f}{r['frontier_equivalent_usd']:>12.4f}{r['savings_pct']:>8.1f}%")
    blended = (1 - total_real / total_frontier) * 100 if total_frontier else 0.0
    print(f"{'TOTAL':<28}{'':<12}{'':>6}{total_real:>10.4f}{total_frontier:>12.4f}{blended:>8.1f}%")

    contestants = ["deferent", *H2H_ROUTERS, "always-top-tier"]
    h2h: dict[str, list[dict]] = {c: [] for c in contestants}
    models: dict[str, dict[str, int]] = {c: {} for c in contestants}
    for r in results:
        for i in r["items"]:
            if not i["head_to_head"]:
                continue
            for a in [i["routed"], *i["head_to_head"]]:
                h2h[a["router"]].append(a)
                key = a["model"] or "error"
                models[a["router"]][key] = models[a["router"]].get(key, 0) + 1
    print("\nHead-to-head on the same prompts:")
    table = {}
    for router in contestants:
        answers = h2h[router]
        jev = [a["jev_adequate"] for a in answers if a["jev_adequate"] is not None]
        scores = [a["score"] for a in answers if a["score"] is not None]
        cost = sum(a["cost_usd"] for a in answers)
        failed = sum(1 for a in answers if a["error"])
        table[router] = {
            "prompts": len(answers), "cost_usd": cost, "failed_or_empty": failed,
            "jev_adequate": [sum(jev), len(jev)],
            "mean_score": sum(scores) / len(scores) if scores else None, "scored": len(scores),
            "score_ge_4": sum(1 for x in scores if x >= 4),
        }
        mean = f"{table[router]['mean_score']:.2f}" if scores else "n/a"
        print(f"  {router:<20} prompts={len(answers):<3} cost=${cost:<8.4f} mean score={mean} (>=4: {table[router]['score_ge_4']}/{len(scores)})  "
              f"Jev adequate={sum(jev)}/{len(jev)}  failed/empty={failed}")
    for router in contestants:
        print(f"  {router} used: {dict(sorted(models[router].items(), key=lambda kv: -kv[1]))}")
    return {
        "blended_savings_pct": blended, "real_cost_usd": total_real, "frontier_equivalent_usd": total_frontier,
        "head_to_head": table, "head_to_head_models": models,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-budget-usd", type=float, default=8.0)
    parser.add_argument("--only", help="Comma-separated experiment names (default: all 12).")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set")
    OUT.mkdir(parents=True, exist_ok=True)
    budget, judge, sem = Budget(args.max_budget_usd), build_judge_from_env(), asyncio.Semaphore(args.concurrency)
    only = set(args.only.split(",")) if args.only else None
    results = []
    async with httpx.AsyncClient() as http:
        for name, route_name in EXPERIMENTS:
            if only and name not in only:
                continue
            print(f"== {name} ({route_name}) -- spent ${budget.spent:.3f} of ${budget.cap:.2f}", flush=True)
            result = await run_experiment(name, route_name, judge, http, budget, sem)
            (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
            print(f"   accuracy {result['accuracy']:.0%}, real ${result['real_cost_usd']:.4f}, savings {result['savings_pct']:.1f}%", flush=True)
            results.append(result)
    await judge.aclose()
    summary = summarize(results)
    summary["total_spend_usd"] = budget.spent
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntotal spend: ${budget.spent:.4f} of ${budget.cap:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
