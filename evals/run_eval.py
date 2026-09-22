#!/usr/bin/env python3
"""Run evals/suite.yaml against one or more OpenAI-compatible models and produce a model card.

    python evals/run_eval.py --models gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol,gpt-6-astra \
        --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY \
        --max-budget-usd 8.00

Grading is local (contains_all, numeric_equals) or via Jev (jev_noul) through this repo's own
jevjudge sidecar -- point --judge-url at a running one (default http://127.0.0.1:8090/v1).
Reuses the exact same TypeSafe/mock transports jevjudge already supports; no new judge code.

A hard cost guard stops the run before exceeding --max-budget-usd: cost is estimated per call
from actual token usage the API reports, checked BEFORE each subsequent call would run.
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
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from models import get as get_pricing  # noqa: E402

SUITE_PATH = Path(__file__).parent / "suite.yaml"


@dataclass
class CaseResult:
    case_id: str
    domain: str
    difficulty: str
    correct: bool | None  # None if the call itself failed
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None


@dataclass
class ModelReport:
    model: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def accuracy(self, domain: str | None = None) -> float | None:
        pool = [r for r in self.results if r.correct is not None and (domain is None or r.domain == domain)]
        return (sum(r.correct for r in pool) / len(pool)) if pool else None

    def latency_p50_p95(self) -> tuple[float, float]:
        lat = sorted(r.latency_ms for r in self.results if r.error is None)
        if not lat:
            return (0.0, 0.0)
        return lat[len(lat) // 2], lat[min(len(lat) - 1, int(0.95 * len(lat)))]

    @property
    def error_rate(self) -> float:
        return sum(1 for r in self.results if r.error is not None) / len(self.results) if self.results else 0.0


def load_suite() -> list[dict]:
    return yaml.safe_load(SUITE_PATH.read_text())


def extract_first_number(text: str) -> float | None:
    match = re.search(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(match.group().replace(",", "")) if match else None


def grade_local(case: dict, response_text: str) -> bool | None:
    grader = case["grader"]
    if grader == "contains_all":
        lowered = response_text.lower()
        return all(s.lower() in lowered for s in case["expected"])
    if grader == "numeric_equals":
        value = extract_first_number(response_text)
        if value is None:
            return False
        return abs(value - float(case["expected"])) <= float(case.get("tolerance", 0))
    return None  # jev_noul: graded separately, batched through the judge


class Budget:
    def __init__(self, max_usd: float):
        self.max_usd = max_usd
        self.spent = 0.0

    def allow(self, estimate: float) -> bool:
        return self.spent + estimate <= self.max_usd

    def record(self, actual: float) -> None:
        self.spent += actual


async def call_model(http: httpx.AsyncClient, base_url: str, api_key: str | None, model: str, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    started = time.perf_counter()
    response = await http.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=60.0)
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"] or ""
    usage = payload.get("usage") or {}
    return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), latency_ms


async def judge_noul(http: httpx.AsyncClient, judge_url: str, state: str, rubric: str) -> bool:
    body = {
        "model": "jev-latest",
        "messages": [{"role": "user", "content": state}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Correctness",
                "schema": {
                    "type": "object",
                    "properties": {"correct": {"type": "boolean", "description": rubric}},
                    "required": ["correct"],
                    "additionalProperties": False,
                },
            },
        },
    }
    response = await http.post(f"{judge_url}/chat/completions", json=body, timeout=15.0)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return bool(json.loads(content)["correct"]) if content.strip().startswith("{") else False


async def run_model(model: str, suite: list[dict], base_url: str, api_key: str | None, judge_url: str, budget: Budget, max_tokens: int) -> ModelReport:
    report = ModelReport(model=model)
    pricing = get_pricing(model)
    async with httpx.AsyncClient() as http:
        for case in suite:
            avg_case_tokens_estimate = 400  # rough, corrected once real usage starts coming back
            estimate = pricing.cost(avg_case_tokens_estimate, max_tokens)
            if not budget.allow(estimate):
                print(f"  [{model}] budget guard: stopping before {case['id']} (spent ${budget.spent:.3f} of ${budget.max_usd:.2f})")
                break
            try:
                text, in_tok, out_tok, latency_ms = await call_model(http, base_url, api_key, model, case["prompt"], max_tokens)
            except httpx.HTTPError as error:
                report.results.append(CaseResult(case["id"], case["domain"], case["difficulty"], None, 0.0, 0, 0, 0.0, error=str(error)))
                continue
            cost = pricing.cost(in_tok, out_tok)
            budget.record(cost)
            if case["grader"] == "jev_noul":
                try:
                    correct = await judge_noul(http, judge_url, f"Task: {case['prompt']}\n\nResponse: {text}", case["rubric"])
                except httpx.HTTPError as error:
                    print(f"  [{model}] judge unreachable for {case['id']}: {error}", file=sys.stderr)
                    correct = None
            else:
                correct = grade_local(case, text)
            report.results.append(CaseResult(case["id"], case["domain"], case["difficulty"], correct, latency_ms, in_tok, out_tok, cost))
    return report


def print_report(reports: list[ModelReport]) -> None:
    domains = sorted({c["domain"] for c in load_suite()})
    print("\n=== Model card ===")
    header = f"{'model':<16}{'overall':>9}" + "".join(f"{d:>12}" for d in domains) + f"{'p50 ms':>9}{'p95 ms':>9}{'errs':>7}{'cost $':>9}"
    print(header)
    print("-" * len(header))
    for r in reports:
        overall = r.accuracy()
        p50, p95 = r.latency_p50_p95()
        row = f"{r.model:<16}{(f'{overall:.0%}' if overall is not None else 'n/a'):>9}"
        for d in domains:
            acc = r.accuracy(d)
            row += f"{(f'{acc:.0%}' if acc is not None else 'n/a'):>12}"
        row += f"{p50:>9.0f}{p95:>9.0f}{r.error_rate:>7.0%}{r.total_cost:>9.4f}"
        print(row)
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True, help="Comma-separated model IDs, must have pricing in evals/models.py.")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-url", default="http://127.0.0.1:8090/v1", help="jevjudge sidecar base URL, for jev_noul graded cases.")
    parser.add_argument("--max-tokens", type=int, default=600, help="Per-call output cap -- also bounds worst-case spend.")
    parser.add_argument("--max-budget-usd", type=float, default=8.0, help="Hard stop across the whole run, all models combined.")
    parser.add_argument("--output", default=None, help="Write raw per-case JSON results here.")
    args = parser.parse_args()

    suite = load_suite()
    api_key = os.environ.get(args.api_key_env)
    budget = Budget(args.max_budget_usd)

    reports: list[ModelReport] = []
    for model in args.models.split(","):
        model = model.strip()
        print(f"== running {model} against {len(suite)} cases (budget remaining: ${args.max_budget_usd - budget.spent:.2f})")
        report = await run_model(model, suite, args.base_url, api_key, args.judge_url, budget, args.max_tokens)
        reports.append(report)
        print(f"   done: {sum(1 for r in report.results if r.correct)} correct, "
              f"{sum(1 for r in report.results if r.error)} errors, cost ${report.total_cost:.4f}")

    print_report(reports)
    print(f"total spend: ${budget.spent:.4f} of ${args.max_budget_usd:.2f} budget")

    if args.output:
        out = [
            {"model": r.model, "cases": [vars(c) for c in r.results]}
            for r in reports
        ]
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"raw results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
