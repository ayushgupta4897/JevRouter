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

## How Deferent is different

Most routers answer one question once per request: *which model is best for this message?* They
use an LLM or an embedding classifier to answer it, and many then make the call themselves.
Deferent is built around what that misses:

| | A typical per-request router | Deferent |
|---|---|---|
| **Who judges** | An LLM call. By LangChain's Switchyard benchmark, that's 21% of routed spend and ~700ms a turn | Jev: typed yes/no, pick-one or score questions in one cheap pass. No text generation |
| **What it does with the answer** | Often makes the model call too | Only decides. Your gateway calls the model, so a routing bug can't run up a bill |
| **What it looks at** | The current message | The whole session: an agent's tool failures (`auto`), whether a run is stuck (`escalation`), and which model already holds the conversation in its prompt cache |
| **Prompt cache** | Ignored. Switching models mid-session silently re-sends everything at full price | Priced in dollars. It switches only when the switch pays for itself |
| **When the judge is down** | Errors, or falls back to a default model | Falls back safely and says so. Mid-session it holds the current model instead of paying for a cold switch |
| **Who configures it** | A platform team, in code | Each team, in its own YAML. Plain-English bucket descriptions become the judge's criteria, and edits go live in seconds |
| **Can you see why?** | Rarely | Every decision carries the judge's confidence and, when the session matters, a `cache` block with the reason and the dollars on each side |

The examples below are real decisions, taken from [`experiments/`](experiments/) and
[`experiments/cache_validation.py`](experiments/cache_validation.py) runs against the real Jev
and OpenAI APIs.

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
| `auto` | `stage_router`, no classifier | Reads the agent's tool-call history: repeated errors push up a tier, steady productive edits keep it down | none |
| `complexity` | `llm_classifier`, capability mode | Asks one question, "can the cheaper model do this whole task?", and compares the answer (a probability) with your threshold | 1 |
| `intent` | `llm_classifier`, custom mode | Asks one pick-one question across your named buckets, using each bucket's `description:` as the criteria | 1 |
| `escalation` | reimplemented as custom mode (see below) | Reads the transcript your gateway sends and answers `continue` or `escalate` based on a repeated-failure pattern | 1 |

### `auto`: watch what the agent is doing

No judge call. It routes on the tool traffic already in the conversation:

| Conversation so far | Routed to |
|---|---|
| "Prove that √2 is irrational." (plain chat, no tools yet) | `gpt-5.6-terra` (efficient) |
| The agent ran `python -m app` three times and got the same `ModuleNotFoundError` traceback each time | **`gpt-6-astra` (capable)**: it's stuck, so it gets the stronger model |
| The agent wrote two files, edited one, and `pytest` passed | `gpt-5.6-terra`: it's making progress, so there's no reason to pay more |

One weak signal isn't enough. An agent that has only read six files stays efficient, because
Switchyard wants a second, corroborating signal before it escalates. `auto` never reads what the
user is asking for, so a hard question in plain chat stays efficient. Use `complexity` when the
content itself should decide.

### `complexity`: forecast the difficulty before running

A finance route with terra and astra, threshold 0.6 (financial answers feed real decisions):

| Request | Jev's forecast | Routed to | Real cost |
|---|---|---|---|
| "What is 15% of $340,000?" | 96% that terra can do it | `gpt-5.6-terra` | $0.00025 |
| "If churn is 2% monthly, what's the annualized churn rate?" | 82% | `gpt-5.6-terra` | $0.0017 |
| "Assess whether our pricing is value-based or cost-plus, using three years of price changes against three years of delivery costs…" | 18% | **`gpt-6-astra`** | $0.026 |

It isn't perfect. Jev gave terra 74% on a cap-table modelling question, and terra got it wrong.
[`experiments/REPORT.md`](experiments/REPORT.md) walks through every miss like that one.

### `intent`: route by what the request is

You name the buckets and describe them in plain English. Those descriptions become the judge's
criteria:

| Route | Request | Bucket, model |
|---|---|---|
| legal review | "What is the governing law?" (on a one-line clause) | `extraction`, `gpt-5.6-luna` |
| legal review | "Client shall indemnify… from any and all claims… including claims arising from Vendor's own…" | `risk_flagging`, **`gpt-6-astra`** |
| support triage | "Where can I download last month's invoice as a PDF?" | `simple_faq`, `gpt-5.6-luna` |
| support triage | "I was charged $49 but I'm on the $29 plan." | `billing_dispute`, `gpt-5.6-terra` |
| support triage | "I think my API key leaked. I found requests from an IP range we don't use." | `account_security`, **`gpt-6-astra`** |
| post-call transcript | "What is the order number?" | `extraction`, `gpt-5.6-luna` |
| post-call transcript | "This is the third time I've called about the same broken feature… considering cancelling." | `analysis`, `gpt-5.6-sol` |

### `escalation`: judge the run, not the request

Start on the cheaper model and move up only when the transcript shows the run going nowhere:

| Transcript | Verdict, model |
|---|---|
| "Test failed → fixed the off-by-one → all tests pass → committed." | continue, `gpt-5.6-terra` |
| "Build fails with a missing import → added it, *same error, same line* → tried another path, *same error*." | **escalate, `gpt-6-astra`** |
| Security: "Alert was a known scanner false positive → closed with the standard note." | continue, `gpt-5.6-sol` |
| Security: "Isolation script failed silently → retried, same silent failure → manual procedure, same outcome." | **escalate, `gpt-6-astra`** |

**Why `escalation` isn't Switchyard's native escalation router:** Switchyard's version calls the
current-tier model itself to watch how it behaves before judging. That means *it* makes model
calls, which breaks the one rule this project exists to enforce. Deferent's `escalation` reads
the transcript your AI Gateway already has and judges that instead. The full account, including
how this was checked against the real algorithm, is in [`docs/DECISION.md`](docs/DECISION.md) §11.

Switchyard has other routers (`passthrough`, `random`, `composite`, `advisor_gate`,
`prefill_router`) that Deferent doesn't use. They don't fit a judge-driven, decision-only design.

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

One route can span providers. Here every tier sits with a different provider, and each is pinned
to it through `extra_body`, which arrives with every decision as `selected_extra_body`. Your
gateway sends it along with the call.

```yaml
# three inference providers, one route
routes:
  - name: support/triage
    policy: intent
    default: technical
    models:
      faq:       { id: deepseek/deepseek-v4.1-flash, client: openrouter, description: "...",
                   extra_body: { provider: { order: [fireworks], allow_fallbacks: false } } }
      billing:   { id: openai/gpt-oss-120b, client: openrouter, description: "...",
                   extra_body: { provider: { order: [together], allow_fallbacks: false } } }
      technical: { id: anthropic/claude-sonnet-5, client: openrouter, description: "...",
                   extra_body: { provider: { order: [anthropic] } } }
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

Read `selected_model` / `selected_client` (plus `selected_extra_body`, if the route sets one), then
call that model however you already do. With an
OpenAI-compatible gateway like [Bifrost](https://github.com/maximhq/bifrost), that's a second,
completely unrelated HTTP call:

```python
decision = requests.post("http://localhost:4000/v1/decide",
                          json={"model": "deployment/simple", "messages": msgs}).json()
answer = requests.post("http://localhost:8080/v1/chat/completions",
                        json={"model": decision["selected_model"], "messages": msgs}).json()
```

Deferent never needs to know Bifrost (or LiteLLM, or your own gateway) exists, and vice versa.
For multi-turn sessions, also send which model answered last. See
[cache-aware switching](#cache-aware-switching).

**If you don't have a gateway yet**, `routerctl serve teams/ --mode proxy` runs the same config
as a real forwarding proxy instead (Switchyard itself, with live reload) — see
[`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md) for both modes.

## Cache-aware switching

**The hidden cost of switching models.** Providers cache the conversation a model has already
read, and each model has its own cache. Staying on the same model means the history costs 10% of
the normal input price. Switching to a different model means the new model has never seen it, so
you pay to send all of it again, and on GPT-5.6+ that re-send costs 125%. A router that re-decides
every message and ignores this can spend more than it saves.

Deferent prices the switch before recommending it. Here's what that looks like:

**1. On a strong model, then an easy question arrives: stay.** A 31K-token incident review opens
with two hard questions on `gpt-5.6-sol`. Then come follow-ups the policy considers easy, like "What
region is this log from?". It says "use the cheaper `gpt-5.6-terra`". But terra has never seen the
31K-token log, and sending it costs $0.079 just to get started. Sol already has it cached. Deferent
stays on sol. Measured on real calls, the four follow-ups cost **$0.065 instead of $0.103 (37%
less)**, and they were answered by the *stronger* model. The same test on Claude (Opus 5.5 →
Sonnet 5, about 45K tokens) came out **45% less**. Anthropic's caching works differently (explicit
breakpoints, a 5-minute TTL), and the only config change was the prices.

**2. …unless the cheaper model is cheap enough anyway: switch.** Same situation, but the cheaper
option is `gpt-5.6-luna`. Luna re-reading the whole conversation from scratch still costs less
than sol reading it from cache. So Deferent switches (`downgrade_saves_money`). This is the "premium
trap" per-session routers fall into when they refuse to step down just to protect the cache.
Because Deferent prices the choice instead of banning it, it doesn't fall in.

**3. Short conversations: switch freely.** On an 8K-token chat, the price gap between models
outweighs one cold start, so Deferent allowed every switch the policy asked for. It added no
overhead and no false holds. Below the provider's minimum (1,024 tokens on OpenAI) nothing is
cached at all.

**4. Going back to a model you just left is cheap.** Real calls: terra, then luna, then back to
terra. Terra still had 7,723 of 7,737 tokens cached, because OpenAI keeps a cache warm for 30
minutes. The gateway reports this in `other_warm_caches`, and Deferent prices the switch back as
warm.

**5. A borderline verdict won't throw away a warm cache.** If the policy wants to move up but
Jev's verdict is close to a coin flip, Deferent stays put. A decisive verdict moves up at once,
because upgrades are about quality, not cost. This damping keeps the router from bouncing
between tiers on every message.

**6. The judge goes down mid-session: hold.** Without this, every request would fall back to the
default tier and pay a cold switch until Jev recovered. The session stays where it is instead.

### Using it

Your gateway adds one optional field. It already has every number it needs:

```json
{"model": "deployment/coding-agent", "messages": [...],
 "session": {"current_model": "gpt-5.6-sol", "cached_prefix_tokens": 31176, "idle_seconds": 40}}
```

| Field | What to send |
|---|---|
| `current_model` | The model that answered the previous message. |
| `cached_prefix_tokens` | On OpenAI, the previous response's `usage.prompt_tokens`. |
| `idle_seconds` | Time since the previous message. |

Every response then says what happened and why:

```json
"cache": {"reason": "downgrade_not_worth_losing_cache", "held_current_model": true,
          "policy_model": "gpt-5.6-terra", "stay_cost_usd": 0.1241, "switch_cost_usd": 0.1388}
```

When Deferent holds, the policy's own pick stays first in `fallback_model_ids`. Leave out
`session` and routing behaves exactly as before.

**What changes in your YAML:** nothing, unless you want it to. Cache-aware switching is on for
every route with sensible defaults. You can tune it per route with an optional `cache:` block,
for example `horizon_turns: 10` for long agent sessions. Model prices sit once, in the
platform's `clients.yaml` under `pricing:`. All fields:
[`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md#cache-aware-switching). Design, ecosystem survey
and every measurement: [`docs/DECISION.md` §12](docs/DECISION.md#12-cache-aware-switching-2026-09-27).

## Head-to-head with OpenRouter's routers

We ran the same 72 prompts four ways:
* **Deferent**, with its tiers spread across Fireworks, Google and Anthropic
* **OpenRouter's Auto Router**
* **OpenRouter's Jev Router**
* **Always the most expensive model**, with no routing

Gemini 3.1 Pro scored all four answers 1–5 side by side, in shuffled order. None of the
contestants used that model.

| | Cost | Mean score |
|---|---|---|
| Always the top model | $0.90 | 4.82 |
| **Deferent** | $0.54 | 4.58 |
| OpenRouter Auto | $0.08 | 4.40 |
| Jev Router | $0.06 | 4.14 |

What the numbers support:

* **Deferent vs the top model:** we cut cost by 40%, and quality dropped by 0.24.
* **Deferent vs Jev Router:** our answers were measurably better, winning on 28 prompts and losing
  on 5. Jev Router sent most prompts to an undisclosed "stealth" model that currently costs $0.
* **Deferent vs Auto:** on one-off prompts, Auto is much cheaper, and at this sample size its
  quality can't be told apart from ours.

Most of that cost gap comes from which models we put in the route (Claude Opus as the top tier),
not from routing mistakes. Owning the router means we can put the same cheap, strong models in
our routes.

This test doesn't measure the reasons we're building our own router:
* choosing our own providers, AI gateway and agent harness
* routing at every step of an agent run
* cache-aware sessions
* learning from our own outcome data

Full method and caveats: [`docs/DECISION.md` §13](docs/DECISION.md#13-multiple-providers-and-a-head-to-head-with-openrouter-2026-09-27).

## Quickstart

```bash
scripts/build.sh                  # builds switchyard-server + bindings + jevjudge + routerctl, ~3 min
source .venv/bin/activate

pytest jevjudge routerctl -q      # 138 tests: compilers, live-reload, fail-open, cache gate, every policy
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
