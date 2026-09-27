"""Real-provider validation of cache-aware switching (routerctl/cache.py), against OpenAI.

Deferent never calls a model itself, so this script plays the AI Gateway: it asks Deferent for a
decision, makes the real call to whichever model was chosen, and feeds the provider's real
`usage` back as the next turn's `session`. Every dollar figure below is real usage times the
real prices in ../clients.yaml -- nothing is estimated.

Four phases:

1. **Premise.** Does switching models really throw the cache away? Same long prefix sent to
   terra, terra again, luna, then back to terra; read `cached_tokens` off each response.
2. **Flip-flopping policy, gated vs. not.** A scripted per-turn verdict that bounces between a
   strong and a weak tier (the failure mode per-request routers have on borderline sessions),
   run twice on fresh caches: once followed blindly, once through the cache gate. Two pairs:
   sol/terra (gate should hold) and sol/luna (gate should let the downgrade through).
3. **End to end.** The real decision server with the real Jev judge on a complexity route,
   over a mixed easy/hard conversation -- once with `session` (gated), once without.

4. **Long context.** A ~30K-token session that opens hard on sol and then asks easy follow-ups
   the policy would send to terra -- where a cold write into terra costs real money.

Each arm gets its own random nonce at the very start of the prompt, so no arm can ride another
arm's cache. A hard spend cap aborts before any call that could cross it.

    set -a; . ./.env; set +a
    .venv/bin/python experiments/cache_validation.py --max-budget-usd 1.50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import secrets
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "routerctl" / "src"))
sys.path.insert(0, str(ROOT.parent / "jevjudge" / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from jevjudge.cascade import build_judge_from_env  # noqa: E402
from routerctl import cache  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.compiler import load_clients_file  # noqa: E402
from routerctl.decide import Decision  # noqa: E402
from routerctl.decision_server import build_app  # noqa: E402
from routerctl.schema import CacheConfig, Route  # noqa: E402

BASE_URL = "https://api.openai.com/v1"
MAX_TOKENS = 600  # reasoning models spend some of this thinking; enough for short answers
PRICING = load_clients_file(ROOT.parent / "clients.yaml").pricing


# ---------------------------------------------------------------------------- the conversation


def ops_log(lines: int = 420, seed: int = 7) -> str:
    """A deterministic ~7K-token payments-platform ops log with a few planted incidents."""
    rng = random.Random(seed)
    services = ["ledger", "auth", "payouts", "risk", "gateway", "notifier", "fx-rates", "kyc"]
    events = [
        "request completed in {ms}ms", "cache refresh ok ({n} keys)", "retrying upstream call (attempt {a})",
        "queue depth {n}", "p99 latency {ms}ms", "connection pool at {p}%", "batch {n} settled",
        "token refreshed for tenant t-{n}", "webhook delivered to merchant m-{n}",
    ]
    out = ["# payments-platform ops log, 2026-09-14, us-east-1. On-call: Priya Raman (primary), Tomas Eriksen (secondary).", ""]
    for i in range(lines):
        h, m, s = 13 + i // 360, (i // 6) % 60, (i * 10) % 60
        svc = rng.choice(services)
        level = rng.choices(["INFO", "WARN", "ERROR"], [85, 11, 4])[0]
        msg = rng.choice(events).format(ms=rng.randint(4, 2400), n=rng.randint(1, 9999), a=rng.randint(1, 5), p=rng.randint(10, 99))
        if i == 310:
            svc, level, msg = "fx-rates", "ERROR", "upstream rate feed returned stale snapshot (age 1840s); serving cached rates"
        if i == 318:
            svc, level, msg = "payouts", "ERROR", "payout batch 7731 priced with stale EUR/USD rate; 412 payouts affected"
        if i == 334:
            svc, level, msg = "risk", "WARN", "anomaly score spike on EUR corridor; auto-hold applied to 57 payouts"
        out.append(f"{h:02d}:{m:02d}:{s:02d} {level:<5} [{svc}] {msg}")
    return "\n".join(out)


DOCUMENT = ops_log()

# (question, is_hard) -- the scripted phases ignore is_hard; phase 3 lets Jev judge for itself.
QUESTIONS = [
    ("Who is the primary on-call engineer named in the log header?", False),
    ("Reconstruct the causal chain behind the payout problem around 13:51-13:55: which service failed first, how it propagated, and what the blast radius was. Then propose two concrete mitigations and the monitoring that would have caught it earlier.", True),
    ("What region is this log from?", False),
    ("Which payout batch number was affected?", False),
    ("Given the stale-rate incident, draft a short customer-facing incident summary for affected merchants, and separately list three follow-up engineering actions ranked by risk reduction per effort, justifying the ranking.", True),
    ("How many payouts did risk auto-hold?", False),
    ("What is the secondary on-call's name?", False),
    ("Argue for and against automatically reversing the 412 mispriced payouts versus issuing top-up payments, considering ledger integrity, merchant experience, and regulatory reporting. Recommend one.", True),
    ("What date is this log from?", False),
    ("Summarize the incident in one sentence.", False),
]


LONG_DOCUMENT = ops_log(lines=1700, seed=11)  # ~30K tokens: here a cold write is real money


def system_prompt(nonce: str, document: str = DOCUMENT) -> str:
    return f"[session {nonce}]\nYou are an SRE assistant. Answer concisely using the ops log below.\n\n{document}"


# ---------------------------------------------------------------------------- money


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    cap: float
    spent: float = 0.0

    def check(self, model: str, prompt_tokens_estimate: int) -> None:
        p = PRICING[model]
        worst = (prompt_tokens_estimate * (p.cache_write or p.input) + MAX_TOKENS * p.output) / 1e6
        if self.spent + worst > self.cap:
            raise BudgetExceeded(f"next call could cost ${worst:.4f}; spent ${self.spent:.4f} of ${self.cap:.2f}")


def real_cost(model: str, usage: dict) -> float:
    """Real usage times real prices. OpenAI's prompt_tokens includes cached and written tokens."""
    p = PRICING[model]
    details = usage.get("prompt_tokens_details") or {}
    prompt = int(usage.get("prompt_tokens", 0))
    read = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    plain = max(prompt - read - written, 0)
    return (read * p.cached_input + written * (p.cache_write or p.input) + plain * p.input + int(usage.get("completion_tokens", 0)) * p.output) / 1e6


@dataclass
class Turn:
    turn: int
    policy_pick: str
    served_by: str
    switched: bool
    prompt_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    completion_tokens: int
    cost_usd: float
    gate_reason: str | None = None


@dataclass
class Arm:
    name: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def switches(self) -> int:
        return sum(t.switched for t in self.turns)

    def summary(self) -> dict:
        prompt = sum(t.prompt_tokens for t in self.turns)
        cached = sum(t.cached_tokens for t in self.turns)
        return {
            "arm": self.name, "turns": len(self.turns), "switches": self.switches, "cost_usd": round(self.cost, 5),
            "cache_hit_rate": round(cached / prompt, 3) if prompt else 0.0,
            "models": [t.served_by for t in self.turns],
        }


# ---------------------------------------------------------------------------- the gateway


class Gateway:
    """The bits of an AI Gateway this test needs: make the call, track per-model cache state."""

    def __init__(self, http: httpx.AsyncClient, api_key: str, budget: Budget):
        self.http, self.api_key, self.budget = http, api_key, budget

    async def complete(self, model: str, messages: list[dict]) -> tuple[str, dict]:
        estimate = sum(len(m["content"]) for m in messages) // 4
        self.budget.check(model, estimate)
        # Same request shape every call (no per-call reasoning_effort changes): changing effort
        # mid-session is itself a cache-busting trap, and would confound this measurement.
        body = {"model": model, "messages": messages, "max_completion_tokens": MAX_TOKENS}
        resp = await self.http.post(f"{BASE_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        usage = payload.get("usage") or {}
        self.budget.spent += real_cost(model, usage)
        return payload["choices"][0]["message"]["content"] or "", usage


class SessionTracker:
    """What a gateway remembers per session to fill in Deferent's `session` field."""

    def __init__(self):
        self.current: str | None = None
        self.last_at: float | None = None
        self.warm: dict[str, int] = {}  # model -> prompt tokens it has cached for this session

    def session(self) -> cache.SessionState | None:
        if self.current is None:
            return None
        return cache.SessionState(
            current_model=self.current,
            cached_prefix_tokens=self.warm.get(self.current),
            idle_seconds=time.time() - self.last_at,
            other_warm_caches={m: t for m, t in self.warm.items() if m != self.current},
        )

    def record(self, model: str, usage: dict) -> None:
        prompt = int(usage.get("prompt_tokens", 0))
        # After this turn the model holds the whole prompt (read + written), if it was cacheable.
        self.warm[model] = prompt if prompt >= PRICING[model].min_cacheable_tokens else 0
        self.current, self.last_at = model, time.time()


def turn_record(i: int, pick: str, model: str, prev: str | None, usage: dict, reason: str | None) -> Turn:
    details = usage.get("prompt_tokens_details") or {}
    return Turn(
        turn=i, policy_pick=pick, served_by=model, switched=prev is not None and model != prev,
        prompt_tokens=int(usage.get("prompt_tokens", 0)), cached_tokens=int(details.get("cached_tokens") or 0),
        cache_write_tokens=int(details.get("cache_write_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens", 0)), cost_usd=real_cost(model, usage), gate_reason=reason,
    )


# ---------------------------------------------------------------------------- phases


async def phase_premise(gw: Gateway) -> list[dict]:
    nonce = secrets.token_hex(8)
    rows = []
    for i, model in enumerate(["gpt-5.6-terra", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-terra"]):
        messages = [{"role": "system", "content": system_prompt(nonce)}, {"role": "user", "content": QUESTIONS[i][0]}]
        _, usage = await gw.complete(model, messages)
        details = usage.get("prompt_tokens_details") or {}
        rows.append({"call": i + 1, "model": model, "prompt_tokens": usage.get("prompt_tokens"),
                     "cached_tokens": details.get("cached_tokens"), "cache_write_tokens": details.get("cache_write_tokens"),
                     "cost_usd": round(real_cost(model, usage), 5)})
        print(f"  premise call {i + 1}: {model:<14} {rows[-1]}")
    return rows


def scripted_decision(pick: str, compiled) -> Decision:
    ref = compiled.model_by_id[pick]
    others = [m for m in compiled.model_by_id if m != pick]
    return Decision(route=compiled.route.name, policy="complexity", selected_model=pick, selected_client=ref.client, bucket=None,
                    fallback_model_ids=others, fallback_clients=[compiled.model_by_id[m].client for m in others],
                    judge_source="scripted", judge_confidence=0.8, judge_error=None, latency_ms=0.0, outcome_id=None)


async def run_scripted(gw: Gateway, weak: str, strong: str, picks: list[str], *, gated: bool, document: str = DOCUMENT, label: str = "") -> Arm:
    route = Route.model_validate({"name": "validation/scripted", "policy": "complexity",
                                  "models": {"weak": {"id": weak, "client": "openai"}, "strong": {"id": strong, "client": "openai"}}})
    compiled = compile_route_algorithm(route)
    arm = Arm(f"{label}{strong}/{weak} {'gated' if gated else 'ungated'}")
    tracker = SessionTracker()
    messages = [{"role": "system", "content": system_prompt(secrets.token_hex(8), document)}]
    for i, pick in enumerate(picks):
        messages.append({"role": "user", "content": QUESTIONS[i % len(QUESTIONS)][0]})
        model, reason = pick, None
        if gated:
            d = cache.apply(scripted_decision(pick, compiled), compiled, tracker.session(), PRICING, CacheConfig(), messages)
            model, reason = d.selected_model, d.cache["reason"]
        text, usage = await gw.complete(model, messages)
        arm.turns.append(turn_record(i + 1, pick, model, tracker.current, usage, reason))
        tracker.record(model, usage)
        messages.append({"role": "assistant", "content": text})
    return arm


E2E_CLIENTS = """\
clients:
  openai: { format: openai_chat, base_url: "https://api.openai.com/v1", api_key_env: OPENAI_API_KEY }
pricing:
PRICING_ROWS
"""
E2E_TEAM = """\
team: validation
routes:
  - name: validation/sre
    policy: complexity
    models:
      weak: { id: gpt-5.6-terra, client: openai }
      strong: { id: gpt-5.6-sol, client: openai }
"""


async def run_e2e(gw: Gateway, client: TestClient, *, gated: bool) -> Arm:
    arm = Arm(f"e2e jev sol/terra {'gated' if gated else 'ungated'}")
    tracker = SessionTracker()
    messages = [{"role": "system", "content": system_prompt(secrets.token_hex(8))}]
    for i, (question, _hard) in enumerate(QUESTIONS[:8]):
        messages.append({"role": "user", "content": question})
        body = {"model": "validation/sre", "messages": messages}
        session = tracker.session() if gated else None
        if session:
            body["session"] = session.model_dump(exclude_none=True)
        decision = client.post("/v1/decide", json=body).json()
        model = decision["selected_model"]
        text, usage = await gw.complete(model, messages)
        arm.turns.append(turn_record(i + 1, decision["cache"]["policy_model"], model, tracker.current, usage, decision["cache"]["reason"]))
        tracker.record(model, usage)
        messages.append({"role": "assistant", "content": text})
    return arm


# ---------------------------------------------------------------------------- main


def show(arm: Arm) -> None:
    s = arm.summary()
    print(f"  {s['arm']:<34} switches={s['switches']:<2} cache_hit={s['cache_hit_rate']:.0%}  cost=${s['cost_usd']:.4f}")
    for t in arm.turns:
        print(f"      t{t.turn:<2} pick={t.policy_pick:<14} served={t.served_by:<14} prompt={t.prompt_tokens:<6} cached={t.cached_tokens:<6} "
              f"write={t.cache_write_tokens:<6} ${t.cost_usd:.4f}  {t.gate_reason or ''}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-budget-usd", type=float, default=1.50)
    parser.add_argument("--phases", default="1,2,3,4")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY not set")
    phases = set(args.phases.split(","))
    budget = Budget(args.max_budget_usd)
    results: dict = {"prices": {m: p.model_dump() for m, p in PRICING.items()}}

    async with httpx.AsyncClient() as http:
        gw = Gateway(http, api_key, budget)
        try:
            if "1" in phases:
                print("== phase 1: does a model switch really lose the cache?")
                results["premise"] = await phase_premise(gw)

            if "2" in phases:
                picks_template = ["S", "W", "S", "W", "W", "S", "W", "S", "W", "W"]  # a borderline, flip-flopping verdict
                results["scripted"] = []
                for weak, strong in [("gpt-5.6-terra", "gpt-5.6-sol"), ("gpt-5.6-luna", "gpt-5.6-sol")]:
                    picks = [strong if p == "S" else weak for p in picks_template]
                    print(f"== phase 2: scripted flip-flop {strong} <-> {weak}")
                    for gated in (False, True):
                        arm = await run_scripted(gw, weak, strong, picks, gated=gated)
                        show(arm)
                        results["scripted"].append({**arm.summary(), "turns_detail": [asdict(t) for t in arm.turns]})

            if "3" in phases:
                print("== phase 3: end to end -- real decision server, real Jev judge")
                with tempfile.TemporaryDirectory() as tmp:
                    pricing_yaml = "\n".join(f"  {m}: {json.dumps(p.model_dump())}" for m, p in PRICING.items())
                    (Path(tmp) / "clients.yaml").write_text(E2E_CLIENTS.replace("PRICING_ROWS", pricing_yaml))
                    (Path(tmp) / "teams").mkdir()
                    (Path(tmp) / "teams" / "validation.yaml").write_text(E2E_TEAM)
                    app = build_app(Path(tmp) / "teams", judge=build_judge_from_env())
                    results["e2e"] = []
                    with TestClient(app) as client:
                        for gated in (False, True):
                            arm = await run_e2e(gw, client, gated=gated)
                            show(arm)
                            results["e2e"].append({**arm.summary(), "turns_detail": [asdict(t) for t in arm.turns]})

            if "4" in phases:
                # The common agent shape: a hard opening on the strong model, then easy follow-ups
                # the policy would rather send to the cheaper tier -- on a long context.
                weak, strong = "gpt-5.6-terra", "gpt-5.6-sol"
                picks = [strong, strong, weak, weak, weak, weak]
                print(f"== phase 4: long context (~30K tokens), hard start then easy tail, {strong} -> {weak}")
                results["long_context"] = []
                for gated in (False, True):
                    arm = await run_scripted(gw, weak, strong, picks, gated=gated, document=LONG_DOCUMENT, label="30K ")
                    show(arm)
                    results["long_context"].append({**arm.summary(), "turns_detail": [asdict(t) for t in arm.turns]})
        except BudgetExceeded as error:
            print(f"!! stopped: {error}")
            results["stopped"] = str(error)
        except Exception as error:  # noqa: BLE001 -- still save what earlier phases measured (and paid for)
            print(f"!! failed: {error!r}")
            results["failed"] = repr(error)

    results["total_spend_usd"] = round(budget.spent, 5)
    out = ROOT / "results" / "cache-validation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\ntotal spend: ${budget.spent:.4f} of ${args.max_budget_usd:.2f}  ->  {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
