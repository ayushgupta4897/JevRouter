"""Cache-aware switching on Anthropic models, through OpenRouter -- the second provider family.

cache_validation.py established the mechanics on OpenAI (automatic caching, 30-minute TTL,
0.1x reads). Anthropic's caching is different in every parameter that matters: caching is
opt-in per request (`cache_control` breakpoints the gateway must send), the TTL is 5 minutes,
and Opus 5.5 reads cached tokens at 0.05x input. If the gate's pricing model is right, it has
to hold up here too with nothing changed but the prices in the config.

1. **Premise.** Same ~8K-token prefix to Sonnet 5, Sonnet 5 again, Haiku 4.5, then back to
   Sonnet 5. Read cached / written tokens off each response.
2. **Long context, hard start then easy tail** (the same shape as cache_validation.py phase 4),
   Opus 5.5 -> Sonnet 5 on a ~30K-token session: the policy's picks followed blindly vs. gated.
3. **OpenRouter's Auto Router on the same session** (with a `session_id`, as OpenRouter
   documents for sticky routing), to see what it picks turn by turn and what it costs.

Every dollar is OpenRouter's billed `usage.cost`. Each arm starts with its own random nonce so
no arm reuses another's cache.

    set -a; . ./.env; set +a
    .venv/bin/python experiments/cache_validation_anthropic.py --max-budget-usd 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import cache_validation as cv  # noqa: E402  (document, questions, Turn/Arm/show, scripted decisions)

from routerctl import cache  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.schema import CacheConfig, ModelPricing, Route  # noqa: E402

OPENROUTER = "https://openrouter.ai/api/v1"
MAX_TOKENS = 1500
OPUS, SONNET, HAIKU = "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5"
PIN = {"provider": {"order": ["anthropic"], "allow_fallbacks": False}}

# Anthropic list prices (USD per 1M tokens) as OpenRouter bills them, checked 2026-09-27; 5-minute
# cache writes are 1.25x input; minimum cacheable length per Anthropic's docs for each model.
PRICING = {
    OPUS: ModelPricing(input=4.0, cached_input=0.20, cache_write=5.0, output=20.0, cache_ttl_seconds=300, min_cacheable_tokens=512),
    SONNET: ModelPricing(input=2.0, cached_input=0.20, cache_write=2.5, output=10.0, cache_ttl_seconds=300, min_cacheable_tokens=1024),
    HAIKU: ModelPricing(input=1.0, cached_input=0.10, cache_write=1.25, output=5.0, cache_ttl_seconds=300, min_cacheable_tokens=4096),
}


class Budget:
    def __init__(self, cap: float):
        self.cap, self.spent = cap, 0.0


def with_breakpoints(messages: list[dict]) -> list[dict]:
    """Anthropic only caches up to explicit breakpoints, and placing them is the gateway's job:
    one on the system prompt (the big document) and one on the latest message, so each turn's
    whole prompt becomes the next turn's cached prefix."""
    out = []
    for idx, m in enumerate(messages):
        mark = m["role"] == "system" or idx == len(messages) - 1
        block = {"type": "text", "text": m["content"]}
        if mark:
            block["cache_control"] = {"type": "ephemeral"}
        out.append({"role": m["role"], "content": [block]})
    return out


async def complete(http: httpx.AsyncClient, budget: Budget, model: str, messages: list[dict], extra: dict | None = None) -> tuple[str, dict, str]:
    if budget.spent > budget.cap:
        raise cv.BudgetExceeded(f"spent ${budget.spent:.3f} of ${budget.cap:.2f}")
    body = {"model": model, "messages": with_breakpoints(messages), "max_tokens": MAX_TOKENS, "usage": {"include": True}, **(extra if extra is not None else PIN)}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "deferent-experiments"}
    resp = await http.post(f"{OPENROUTER}/chat/completions", json=body, headers=headers, timeout=240)
    payload = resp.json()
    if resp.status_code != 200 or "choices" not in payload:
        raise RuntimeError(f"{model}: HTTP {resp.status_code}: {json.dumps(payload)[:300]}")
    usage = payload.get("usage") or {}
    budget.spent += float(usage.get("cost") or 0)
    return payload["choices"][0]["message"].get("content") or "", usage, payload.get("model") or model


def turn_of(i: int, pick: str, served: str, prev: str | None, usage: dict, reason: str | None) -> cv.Turn:
    details = usage.get("prompt_tokens_details") or {}
    return cv.Turn(
        turn=i, policy_pick=pick, served_by=served, switched=prev is not None and served != prev,
        prompt_tokens=int(usage.get("prompt_tokens") or 0), cached_tokens=int(details.get("cached_tokens") or 0),
        cache_write_tokens=int(details.get("cache_write_tokens") or 0), completion_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0), gate_reason=reason,
    )


class Tracker:
    """Gateway-side session state, Anthropic flavour: what's warm is what was read + written."""

    def __init__(self):
        self.current: str | None = None
        self.last_at = 0.0
        self.warm: dict[str, int] = {}

    def session(self) -> cache.SessionState | None:
        if self.current is None:
            return None
        return cache.SessionState(current_model=self.current, cached_prefix_tokens=self.warm.get(self.current),
                                  idle_seconds=time.time() - self.last_at,
                                  other_warm_caches={m: t for m, t in self.warm.items() if m != self.current})

    def record(self, model: str, usage: dict) -> None:
        details = usage.get("prompt_tokens_details") or {}
        self.warm[model] = int(details.get("cached_tokens") or 0) + int(details.get("cache_write_tokens") or 0)
        self.current, self.last_at = model, time.time()


async def premise(http, budget) -> list[dict]:
    nonce, rows = secrets.token_hex(8), []
    for i, model in enumerate([SONNET, SONNET, HAIKU, SONNET]):
        messages = [{"role": "system", "content": cv.system_prompt(nonce)}, {"role": "user", "content": cv.QUESTIONS[i][0]}]
        _, usage, _ = await complete(http, budget, model, messages)
        details = usage.get("prompt_tokens_details") or {}
        rows.append({"call": i + 1, "model": model, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": details.get("cached_tokens"),
                     "cache_write_tokens": details.get("cache_write_tokens"), "cost_usd": usage.get("cost")})
        print(f"  premise {i + 1}: {rows[-1]}", flush=True)
    return rows


async def scripted(http, budget, picks: list[str], *, gated: bool) -> cv.Arm:
    route = Route.model_validate({"name": "v/anthropic", "policy": "complexity",
                                  "models": {"weak": {"id": SONNET, "client": "openrouter"}, "strong": {"id": OPUS, "client": "openrouter"}}})
    compiled = compile_route_algorithm(route)
    arm, tracker = cv.Arm(f"30K opus/sonnet {'gated' if gated else 'ungated'}"), Tracker()
    messages = [{"role": "system", "content": cv.system_prompt(secrets.token_hex(8), cv.LONG_DOCUMENT)}]
    for i, pick in enumerate(picks):
        messages.append({"role": "user", "content": cv.QUESTIONS[i % len(cv.QUESTIONS)][0]})
        model, reason = pick, None
        if gated:
            d = cache.apply(cv.scripted_decision(pick, compiled), compiled, tracker.session(), PRICING, CacheConfig(), messages)
            model, reason = d.selected_model, d.cache["reason"]
        text, usage, _ = await complete(http, budget, model, messages)
        arm.turns.append(turn_of(i + 1, pick, model, tracker.current, usage, reason))
        tracker.record(model, usage)
        messages.append({"role": "assistant", "content": text})
    return arm


async def auto_router(http, budget, turns: int) -> cv.Arm:
    arm, prev = cv.Arm("30K openrouter/auto (session_id)"), None
    session_id = f"deferent-{secrets.token_hex(6)}"
    messages = [{"role": "system", "content": cv.system_prompt(secrets.token_hex(8), cv.LONG_DOCUMENT)}]
    for i in range(turns):
        messages.append({"role": "user", "content": cv.QUESTIONS[i % len(cv.QUESTIONS)][0]})
        text, usage, served = await complete(http, budget, "openrouter/auto", messages, {"session_id": session_id})
        arm.turns.append(turn_of(i + 1, "openrouter/auto", served, prev, usage, None))
        prev = served
        messages.append({"role": "assistant", "content": text})
    return arm


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-budget-usd", type=float, default=1.5)
    parser.add_argument("--phases", default="1,2,3")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set")
    phases, budget = set(args.phases.split(",")), Budget(args.max_budget_usd)
    results: dict = {"run_date": time.strftime("%Y-%m-%d"), "prices": {m: p.model_dump() for m, p in PRICING.items()}}
    async with httpx.AsyncClient() as http:
        try:
            if "1" in phases:
                print("== phase 1: does a switch lose the cache on Anthropic too?", flush=True)
                results["premise"] = await premise(http, budget)
            if "2" in phases:
                picks = [OPUS, OPUS, SONNET, SONNET, SONNET, SONNET]
                print("== phase 2: ~30K tokens, hard start on Opus 5.5 then an easy tail the policy sends to Sonnet 5", flush=True)
                results["long_context"] = []
                for gated in (False, True):
                    arm = await scripted(http, budget, picks, gated=gated)
                    cv.show(arm)
                    results["long_context"].append({**arm.summary(), "turns_detail": [asdict(t) for t in arm.turns]})
            if "3" in phases:
                print("== phase 3: OpenRouter's Auto Router on the same kind of session", flush=True)
                arm = await auto_router(http, budget, 6)
                cv.show(arm)
                results["auto_router"] = {**arm.summary(), "turns_detail": [asdict(t) for t in arm.turns]}
        except cv.BudgetExceeded as error:
            print(f"!! stopped: {error}")
            results["stopped"] = str(error)
        except Exception as error:  # noqa: BLE001 -- keep what earlier phases measured
            print(f"!! failed: {error!r}")
            results["failed"] = repr(error)
    results["total_spend_usd"] = round(budget.spent, 5)
    out = ROOT / "results" / "cache-validation-anthropic.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\ntotal spend: ${budget.spent:.4f} of ${args.max_budget_usd:.2f} -> {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
