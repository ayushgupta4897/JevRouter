#!/usr/bin/env python3
"""Run all 12 router-type experiments: real routing decisions (real Jev), real per-decision
completions (real OpenAI, for genuine measured cost -- not an assumed token count), and a small
real quality-parity spot check (routed model vs. the most capable model in that experiment's own
roster, graded by the same Jev judge) on 3 representative items per experiment.

    python experiments/run_experiments.py --max-budget-usd 5.00

Every model here is one already validated for real elsewhere in this repo (evals/, docs/DECISION.md):
gpt-5.6-luna/terra/sol, gpt-6-astra, and the real TypeSafe Jev judge. Routing decisions themselves
cost only a Jev judge call (a fraction of a cent) since this router never calls the target to
decide -- see docs/DECISION.md's decision-only section. The real completion calls here are purely
to measure genuine cost and quality, the same way evals/run_eval.py does for the model-comparison
harness; they are not part of what the router itself does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "evals"))
sys.path.insert(0, str(ROOT.parent / "routerctl" / "src"))
sys.path.insert(0, str(ROOT.parent / "jevjudge" / "src"))

from models import get as get_pricing  # noqa: E402  (evals/models.py -- real verified pricing)
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.compiler import load_team_file  # noqa: E402
from routerctl.decide import decide  # noqa: E402
from routerctl.messages import to_libsy_messages  # noqa: E402
from jevjudge.cascade import build_judge_from_env  # noqa: E402

EXPERIMENTS = [
    ("support-ticket-triage", "support/ticket-triage"),
    ("post-call-transcript", "experiment/post-call-transcript"),
    ("exec-assistant", "exec/email-triage"),
    ("meeting-notes", "ops/meeting-notes"),
    ("legal-contract-review", "legal/contract-review"),
    ("resume-screening", "hr/resume-screening"),
    ("coding-agent", "eng/coding-agent"),
    ("security-incident-response", "security/incident-response"),
    ("code-review", "eng/code-review"),
    ("financial-analysis", "finance/analysis"),
    ("sql-query-generation", "data/sql-query-generation"),
    ("general-assistant", "general/assistant"),
]

BASE_URL = "https://api.openai.com/v1"
API_KEY_ENV = "OPENAI_API_KEY"
# gpt-5.6-*/gpt-6-astra are reasoning models: hidden reasoning tokens count against
# max_completion_tokens before any visible text is emitted. A first run at 400 found 35 of 216
# items (16%, across every tier) came back with 400/400 output tokens and an EMPTY visible
# response -- the model spent its whole budget reasoning and never got to answer. That silently
# corrupts both cost (billed for tokens that produced nothing visible) and any quality read on
# those items. 1200 leaves comfortable room for reasoning plus a full answer even on this
# dataset's hardest prompts (financial modeling, multi-join SQL, distributed-systems bugs).
MAX_TOKENS = 1200
ADEQUACY_RUBRIC = (
    "The response is a competent, on-topic answer to the task: it addresses what was actually "
    "asked, contains no clear factual or logical errors, and doesn't omit something the task "
    "explicitly required. Minor style differences from an ideal answer don't count against it."
)


class Budget:
    def __init__(self, max_usd: float):
        self.max_usd = max_usd
        self.spent = 0.0

    def allow(self, estimate: float = 0.02) -> bool:
        return self.spent + estimate <= self.max_usd

    def record(self, actual: float) -> None:
        self.spent += actual


@dataclass
class ItemResult:
    item_id: str
    expected: str
    actual_label: str
    correct: bool
    selected_model: str
    selected_client: str
    judge_source: str | None
    judge_confidence: float | None
    judge_latency_ms: float
    completion_input_tokens: int = 0
    completion_output_tokens: int = 0
    completion_cost_usd: float = 0.0
    completion_latency_ms: float = 0.0
    completion_text: str | None = None
    completion_error: str | None = None


@dataclass
class SpotCheck:
    item_id: str
    routed_model: str
    frontier_model: str
    routed_adequate: bool | None
    frontier_adequate: bool | None
    routed_text: str | None = None
    frontier_text: str | None = None


@dataclass
class ExperimentResult:
    name: str
    route_name: str
    policy: str
    roster_ids: list[str] = field(default_factory=list)  # every model the route is CONFIGURED with, not just what was picked
    items: list[ItemResult] = field(default_factory=list)
    spot_checks: list[SpotCheck] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(1 for i in self.items if i.correct) / len(self.items) if self.items else 0.0

    @property
    def total_real_cost(self) -> float:
        return sum(i.completion_cost_usd for i in self.items)

    @property
    def frontier_only_cost(self) -> float:
        """What every item would have cost had it gone to the most expensive model this route is
        CONFIGURED with -- not just the models routing actually picked, since a route that (like
        `auto` here) stays on one tier across the whole sample would otherwise show 0% savings
        purely because nothing pricier was ever selected to compare against. Uses each item's
        REAL measured token counts against that fixed tier's real price -- the fair baseline
        (same tokens, priciest configured tier), not a guessed frontier token count."""
        pricing = get_pricing(max(self.roster_ids, key=lambda m: get_pricing(m).output_per_million))
        return sum(pricing.cost(i.completion_input_tokens, i.completion_output_tokens) for i in self.items)

    @property
    def savings_pct(self) -> float:
        frontier = self.frontier_only_cost
        return (1 - self.total_real_cost / frontier) * 100 if frontier > 0 else 0.0


def load_dataset(name: str) -> list[dict]:
    return yaml.safe_load((ROOT / "datasets" / f"{name}.yaml").read_text())


def actual_label(compiled, decision) -> str:
    tag = compiled.route.policy.policy
    if tag in ("intent", "escalation"):
        return decision.bucket
    if tag == "auto":
        return "efficient" if decision.selected_model == compiled.models["efficient"][0] else "capable"
    if tag == "complexity":
        return "weak" if decision.selected_model == compiled.models["efficient"][0] else "strong"
    raise AssertionError(tag)


async def call_model(http: httpx.AsyncClient, api_key: str, model: str, prompt: str) -> tuple[str, int, int, float]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": MAX_TOKENS}
    started = time.perf_counter()
    response = await http.post(f"{BASE_URL}/chat/completions", json=body, headers=headers, timeout=60.0)
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"] or ""
    usage = payload.get("usage") or {}
    return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), latency_ms


async def judge_adequate(judge, task: str, response_text: str) -> bool | None:
    request = {
        "model": "jev-latest",
        "messages": [{"role": "user", "content": f"Task: {task}\n\nResponse: {response_text}"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Adequacy", "schema": {"type": "object", "properties": {"adequate": {"type": "boolean", "description": ADEQUACY_RUBRIC}}, "required": ["adequate"], "additionalProperties": False}},
        },
    }
    outcome = await judge.judge(request)
    content = outcome.response["choices"][0]["message"]["content"]
    try:
        return bool(json.loads(content)["adequate"])
    except (json.JSONDecodeError, KeyError):
        return None


async def run_experiment(name: str, route_name: str, judge, http: httpx.AsyncClient, api_key: str, budget: Budget) -> ExperimentResult:
    team = load_team_file(ROOT / "teams" / f"{name}.yaml")
    route = next(r for r in team.routes if r.name == route_name)
    compiled = compile_route_algorithm(route)
    dataset = load_dataset(name)
    roster_ids = sorted(compiled.model_by_id, key=lambda m: get_pricing(m).output_per_million)
    result = ExperimentResult(name=name, route_name=route_name, policy=route.policy.policy, roster_ids=roster_ids)
    frontier_id = roster_ids[-1]

    for idx, item in enumerate(dataset):
        prompt = item["text"]
        request = {"model": "auto", "stream": False, "messages": to_libsy_messages([{"role": "user", "content": prompt}])}
        decision = await decide(compiled, judge, request)
        label = actual_label(compiled, decision)

        item_result = ItemResult(
            item_id=item["id"], expected=item["expected"], actual_label=label, correct=(label == item["expected"]),
            selected_model=decision.selected_model, selected_client=decision.selected_client,
            judge_source=decision.judge_source, judge_confidence=decision.judge_confidence, judge_latency_ms=decision.latency_ms,
        )

        if budget.allow():
            try:
                text, in_tok, out_tok, lat = await call_model(http, api_key, decision.selected_model, prompt)
                cost = get_pricing(decision.selected_model).cost(in_tok, out_tok)
                budget.record(cost)
                item_result.completion_input_tokens, item_result.completion_output_tokens = in_tok, out_tok
                item_result.completion_cost_usd, item_result.completion_latency_ms = cost, lat
                item_result.completion_text = text
            except httpx.HTTPError as error:
                item_result.completion_error = str(error)
        else:
            item_result.completion_error = "budget exhausted"
        result.items.append(item_result)

        # Quality-parity spot check on the first 3 items: routed model vs. this experiment's own
        # most capable/expensive roster model, both graded for adequacy by the same Jev judge.
        if idx < 3 and decision.selected_model != frontier_id and item_result.completion_text and budget.allow():
            try:
                frontier_text, in_tok, out_tok, _ = await call_model(http, api_key, frontier_id, prompt)
                budget.record(get_pricing(frontier_id).cost(in_tok, out_tok))
                routed_ok = await judge_adequate(judge, prompt, item_result.completion_text)
                frontier_ok = await judge_adequate(judge, prompt, frontier_text)
                result.spot_checks.append(SpotCheck(item["id"], decision.selected_model, frontier_id, routed_ok, frontier_ok, item_result.completion_text, frontier_text))
            except httpx.HTTPError as error:
                print(f"  [{name}] spot-check call failed for {item['id']}: {error}", file=sys.stderr)

    return result


def print_summary(results: list[ExperimentResult]) -> None:
    print("\n=== Experiment summary ===")
    header = f"{'experiment':<28}{'policy':<12}{'items':>7}{'accuracy':>10}{'real $':>10}{'frontier $':>12}{'savings':>9}"
    print(header)
    print("-" * len(header))
    total_real, total_frontier = 0.0, 0.0
    for r in results:
        total_real += r.total_real_cost
        total_frontier += r.frontier_only_cost
        print(f"{r.name:<28}{r.policy:<12}{len(r.items):>7}{r.accuracy:>10.0%}{r.total_real_cost:>10.4f}{r.frontier_only_cost:>12.4f}{r.savings_pct:>8.1f}%")
    print("-" * len(header))
    overall_savings = (1 - total_real / total_frontier) * 100 if total_frontier > 0 else 0.0
    print(f"{'TOTAL':<28}{'':<12}{sum(len(r.items) for r in results):>7}{'':<10}{total_real:>10.4f}{total_frontier:>12.4f}{overall_savings:>8.1f}%")

    print("\n=== Quality-parity spot checks (routed model vs. this experiment's own frontier tier) ===")
    for r in results:
        if not r.spot_checks:
            continue
        routed_ok = sum(1 for s in r.spot_checks if s.routed_adequate)
        frontier_ok = sum(1 for s in r.spot_checks if s.frontier_adequate)
        print(f"  {r.name:<28} routed adequate {routed_ok}/{len(r.spot_checks)}   frontier adequate {frontier_ok}/{len(r.spot_checks)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-budget-usd", type=float, default=5.00)
    parser.add_argument("--only", help="Comma-separated experiment names to run (default: all 12).")
    args = parser.parse_args()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"{API_KEY_ENV} not set")

    budget = Budget(args.max_budget_usd)
    judge = build_judge_from_env()
    only = set(args.only.split(",")) if args.only else None

    results: list[ExperimentResult] = []
    async with httpx.AsyncClient() as http:
        for name, route_name in EXPERIMENTS:
            if only and name not in only:
                continue
            print(f"== running {name} ({route_name}) -- budget remaining ${args.max_budget_usd - budget.spent:.3f}")
            result = await run_experiment(name, route_name, judge, http, api_key, budget)
            results.append(result)
            print(f"   accuracy {result.accuracy:.0%}, real cost ${result.total_real_cost:.4f}, "
                  f"frontier-equivalent ${result.frontier_only_cost:.4f}, savings {result.savings_pct:.1f}%")
            (ROOT / "results" / f"{name}.json").write_text(json.dumps({
                "name": result.name, "route_name": result.route_name, "policy": result.policy,
                "items": [asdict(i) for i in result.items], "spot_checks": [asdict(s) for s in result.spot_checks],
            }, indent=2))

    await judge.aclose()
    print_summary(results)
    print(f"\ntotal spend: ${budget.spent:.4f} of ${args.max_budget_usd:.2f} budget")


if __name__ == "__main__":
    asyncio.run(main())
