# Deferent

**Decides which model should answer a request. Never answers it.**

Deferent is a routing layer built on NVIDIA NeMo Switchyard (vendored, unmodified) with
TypeSafe's Jev as its judge instead of an LLM. You send it a request; it sends back the name of
the model that should handle it — cost tier, provider, everything your policy defines — and
stops there. Making the real call is your AI Gateway's job, not this router's. That split is the
whole design, not an implementation detail:

```
┌────────────────────────────────────────────────────────────────────┐
│ 1. Your app calls Deferent                                         │
│    POST /v1/decide  {"model": "deployment/foo", "messages": [...]} │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. Deferent asks Jev one cheap, typed question and gets            │
│    back a verdict -- not a model call. This is the only            │
│    network call Deferent makes on its own.                         │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. Deferent replies with a decision, never an answer:              │
│    {"selected_model": "gpt-5.6-luna", "selected_client": "openai"} │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 4. Your AI Gateway (Bifrost, LiteLLM, your own) takes that         │
│    decision and makes the real call. Deferent never does.          │
└────────────────────────────────────────────────────────────────────┘
```

Nothing here ever dials the model it recommends. That's not a missing feature — it's the point:
routing and execution are two jobs with two owners, so a routing bug can never turn into a stray
bill on someone else's model, and an execution outage can never look like a routing decision.

## Why a judge that isn't an LLM

Every routing policy below needs one small decision per request — "is this task hard," "which of
these N buckets fits," "has this run gone in circles" — expressed as a JSON verdict. Running an
LLM to answer that is what Switchyard normally does, and on the industry's own numbers ([LangChain's
Switchyard benchmark](https://www.langchain.com/blog/switchyard-agent-routing-benchmark)) the
judge alone eats **21% of routed spend** and adds **~700ms per turn**. Jev is a "System One"
model: it doesn't generate text, it answers typed questions (yes/no, multiple-choice, a score) in
one parallel pass, cheaply. A router's verdict *is* a set of typed questions, so this project
puts Jev where the LLM judge used to be. Full economics: [`docs/DECISION.md`](docs/DECISION.md) §7.

## The four routing policies

Every route in your YAML picks one. Each maps to a specific Switchyard mechanism underneath:

| `policy:` | Switchyard mechanism | How it decides | Judge calls |
|---|---|---|---|
| `auto` | `stage_router`, no classifier | A heuristic over tool-call/tool-result signals in the conversation — Switchyard's own zero-config default | none |
| `complexity` | `llm_classifier`, capability mode | One call: "can the weak model solve this whole task?" (a probability). Above your threshold → weak, else → strong | 1 |
| `intent` | `llm_classifier`, custom mode | One call: a multiple-choice pick among your named buckets, using each bucket's own `description:` as the criteria | 1 |
| `escalation` | reimplemented as custom mode (see below) | One call: reads the conversation you send and decides `continue` vs `escalate` on a repeated-failure pattern | 1 |

**Why `escalation` isn't Switchyard's native escalation router:** Switchyard's real one calls the
current-tier model itself to watch its live behavior before judging — which means *it* executes,
breaking the one rule this whole project exists to enforce. Deferent's `escalation` instead reads
whatever transcript your AI Gateway already has and judges that. Full account, including how this
was verified against the real algorithm: [`docs/DECISION.md`](docs/DECISION.md) §11.

Switchyard has other routers (`passthrough`, `random`, `composite`, `advisor_gate`,
`prefill_router`) that Deferent doesn't use — they don't fit a judge-driven, decision-only design.

## What Jev actually sees

Never your whole request verbatim. `jevjudge` (this repo's compiler) turns your route's own
`description:` text into Jev questions, and caps what's sent to a bounded character budget
(default 100k chars, configurable), trimming from the *middle* of a long conversation so the
opening task and the most recent turns both survive. Only the judge's input is bounded — the
model your AI Gateway actually calls still gets your full, untouched request. One real example
(`policy: intent`, a legal-review route):

```jsonc
// Deferent sends Jev one "choice" question, built from your two buckets' descriptions:
{"decision__target": {"type": "choice", "criteria": {
  "extraction": "Pull a specific, already-defined field... stated verbatim in the document.",
  "risk_flagging": "Judge whether a clause is unusual, one-sided, or risky..."
}}}

// Jev answers with a pick and a calibrated probability, not text:
{"decision__target": {"choice": "risk_flagging", "confidence": 0.97,
  "probabilities": {"risk_flagging": 0.99, "extraction": 0.01}}}
```

## Writing a route

```yaml
# teams/legal.yaml
team: legal
routes:
  - name: legal/contract-review
    policy: intent
    default: risk_flagging          # used if Jev is unreachable or unsure
    models:
      extraction:
        id: gpt-5.6-luna
        client: openai
        description: "Pull a stated field verbatim: a date, a party, an amount."
      risk_flagging:
        id: gpt-6-astra
        client: openai
        description: "Judge whether a clause is unusual or risky. Gets it wrong, real cost."
```

Two more, showing the range — a zero-config default, and per-model tuning:

```yaml
# smallest possible route
routes:
  - name: deployment/simple
    policy: auto
    models:
      efficient: { id: gpt-5.6-terra, client: openai }
      capable:   { id: gpt-6-astra,   client: openai }
```

```yaml
# a custom judge model and a provider-specific knob on one tier
routes:
  - name: deployment/deep-research
    policy: complexity
    judge: { id: gpt-5.6-luna, client: openai }
    models:
      weak:   { id: gpt-5.6-terra, client: openai }
      strong: { id: gpt-6-astra, client: openai, extra_body: { reasoning: { effort: high } } }
    base_threshold: 0.65
```

Full schema, every field, what "live" means, what a broken edit does: [`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md).
Twelve more real, worked examples across support, coding, legal, finance, HR, and security:
[`experiments/teams/`](experiments/teams/) and [`experiments/REPORT.md`](experiments/REPORT.md).

## Integrating it

One HTTP call. No SDK.

```bash
routerctl serve teams/ --port 4000

curl localhost:4000/v1/decide -d '{"model": "deployment/simple", "messages": [...]}'
# -> {"selected_model": "gpt-5.6-terra", "selected_client": "openai", ...}
```

Read `selected_model` / `selected_client`, then call that model however you already do. With an
OpenAI-compatible gateway like [Bifrost](https://github.com/maximhq/bifrost), that's a second,
completely unrelated HTTP call:

```python
decision = requests.post("http://localhost:4000/v1/decide",
                          json={"model": "deployment/simple", "messages": msgs}).json()
answer = requests.post("http://localhost:8080/v1/chat/completions",
                        json={"model": decision["selected_model"], "messages": msgs}).json()
```

Deferent never needs to know Bifrost (or LiteLLM, or your own gateway) exists, and vice versa.

**If you don't have a gateway yet**, `routerctl serve teams/ --mode proxy` runs the same config
as a real forwarding proxy instead (Switchyard itself, with live reload) — see
[`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md) for both modes.

## Cache-aware switching

Provider prompt caches are per model. Move a long session to a different model and the next
turn pays full price, plus a 1.25× cache-write premium on GPT-5.6+, to re-send the whole
conversation. A router that re-decides every turn can cost more than it saves.

So a decision can also weigh the cache. Tell Deferent which model served the previous turn and
how much of the conversation it has cached, and it prices the switch before recommending it:

```json
{"model": "deployment/coding-agent", "messages": [...],
 "session": {"current_model": "gpt-5.6-sol", "cached_prefix_tokens": 31176, "idle_seconds": 40}}
```

* **Downgrades have to pay for themselves.** Deferent compares staying warm against paying a cold
  first turn on the cheaper model, over the next few turns. If switching doesn't save money, it
  holds the current model.
* **Upgrades need a decisive verdict.** A borderline judge verdict won't throw away a warm cache.
  A clear one will.
* **A cold cache is free to leave.** First turn, idle past the provider's TTL, or a prefix too
  short to cache: the policy decides alone.
* **A judge outage holds the current model.** It doesn't fall back to the default tier, which
  would pay a cold write on every request until the judge recovers.

Every response has a `cache` block that says what happened and why, so an override is never
silent. For example: `{"reason": "downgrade_not_worth_losing_cache", "held_current_model": true,
"policy_model": "gpt-5.6-terra", "stay_cost_usd": 0.1241, "switch_cost_usd": 0.1388}`. The
policy's own pick stays first in `fallback_model_ids`. Leave out `session` and nothing changes.

The gateway already has what it needs. On OpenAI, `cached_prefix_tokens` is the previous turn's
`usage.prompt_tokens`. Prices live in `clients.yaml` under `pricing:`, and the config is in
[`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md#cache-aware-switching).

Tested against real OpenAI calls (`experiments/cache_validation.py`, real usage × real prices):

* **Caches are per model.** A switch showed 0 cached tokens.
* **Switching back within the TTL is warm.** Flip-flopping between two already-warm models turned
  out cheap; the first cold write is what costs money.
* **Long context:** a 31K-token session with a hard start and easy follow-ups cost **14% less**
  gated. The four follow-ups alone cost **37% less**, and they stayed on the stronger model.
* **Short context:** on an 8K-token context the gate correctly allowed every switch.

The full record is in [`docs/DECISION.md` §12](docs/DECISION.md#12-cache-aware-switching-2026-09-27).

## Quickstart

```bash
scripts/build.sh                  # builds switchyard-server + bindings + jevjudge + routerctl, ~3 min
source .venv/bin/activate

pytest jevjudge routerctl -q      # 127 tests: compilers, live-reload, fail-open, cache gate, every policy
routerctl validate teams/          # schema check + a real switchyard-server --dry-run

scripts/e2e_routerctl_decide.sh   # proves decision-only end to end (target genuinely never dialed)
scripts/e2e.sh                    # 12 routing-decision checks against mock Jev + mock upstream
```

Needs Rust (via [rustup](https://rustup.rs), version pinned by the vendored toolchain file),
Python ≥ 3.11, and [`uv`](https://docs.astral.sh/uv/). Everything Switchyard needs is vendored
under `vendor/switchyard/` at a pinned commit — no `cargo install --git`, nothing unvendored.

## What's in the repo

| Path | What |
|---|---|
| `routerctl/` | The router. `--mode decide` (default): embeds Switchyard's routing algorithms directly and returns a decision, never a completion. `--mode proxy`: compiles to a real Switchyard deployment with live reload, for teams without a gateway yet |
| `jevjudge/` | Compiles a route's schema into Jev questions and Jev's answers back into a verdict; the confidence gate and fallback cascade |
| `vendor/switchyard/` | Vendored Switchyard source, pinned commit, unmodified (`vendor/NOTICE.md`) |
| `teams/*.yaml`, `experiments/teams/*.yaml` | 14 real route configs across both this repo's shipped examples and a 12-use-case study |
| `evals/` | A concise real-model comparison harness — accuracy, cost, latency per model |
| `docs/DECISION.md` | The full design record: why Switchyard, why Jev, every real bug found and how, the decision-only pivot, economics |
| `docs/TEAM_CONFIG.md` | The complete YAML schema |
| `docs/EVALS.md` | How to decide which model is actually better, at what, for how much |
| `experiments/REPORT.md` | 216 real routing decisions across 12 business use cases: real cost savings, every disagreement individually investigated |

## Status

Validated against the real TypeSafe API and real OpenAI models (not just mocks) — see
[`docs/DECISION.md`](docs/DECISION.md) for the full record, including every real bug found along
the way and how each was caught. In short: routing decisions are real and tested; two production
gaps (an unenforced context window on `escalation`, and a judge outage that used to be fatal
instead of degrading gracefully) were found and fixed, with tests pinning both. Switchyard is
pre-1.0, vendored at a pinned commit; `scripts/update_vendor.sh` re-vendors when you want to move.
