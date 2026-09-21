# JevRouter

**NVIDIA NeMo Switchyard as the router. TypeSafe Jev as the judge. No fork.**

Switchyard decides which model serves each LLM call. Its best-performing algorithms (capability
classification, trajectory escalation, custom N-way policies) all consult an *LLM judge* that
returns a small JSON verdict. That judge was the router's tax: in LangChain's Switchyard benchmark
it consumed **21% of routed spend** and added **~700 ms per turn**, and it runs on every turn until
a session escalates.

Jev is a "System One" model: it does not generate text, it answers typed questions
(`noul` yes/no, `choice`, `score`) about a state in one parallel pass, in ~70–500 ms, at
$0.042 per million input tokens with free output, and returns calibrated probabilities.
A router's verdict *is* a set of typed questions. So this repo puts Jev where the judge was.

```
                    ┌──────────────────────────────┐
  agent / SDK ───▶  │  switchyard-server (unforked)│ ──▶ capable model (Opus, Sol, …)
  OpenAI/Anthropic  │  llm_classifier: capability  │ ──▶ efficient model (GLM, Nemotron, …)
  wire format       │                  escalation  │
                    │                  custom N-way│
                    └──────────────┬───────────────┘
                     judge call    │  OpenAI chat + response_format JSON Schema
                                   ▼
                    ┌──────────────────────────────┐        ┌──────────────────────────┐
                    │  jevjudge (this repo)        │ ─────▶ │ Jev  POST /v1/systemone  │
                    │  JSON Schema ─▶ Jev questions│ ◀───── │ (TypeSafe · OpenRouter · │
                    │  Jev answers ─▶ JSON verdict │        │  self-hosted openjev/kev)│
                    │  confidence gate ─▶ cascade  │──┐     └──────────────────────────┘
                    └──────────────────────────────┘  └──▶ fallback LLM judge (optional)
```

Switchyard sees an ordinary OpenAI-compatible judge that returns schema-valid JSON. Everything
else in Switchyard (translation, session affinity, escalation streaks, fallbacks, metrics,
benchmark harness) keeps working unchanged. The full reasoning, alternatives, prior art, and
risks are in [`docs/DECISION.md`](docs/DECISION.md).

## What is in the box

| Path | What |
|---|---|
| `jevjudge/` | Python package: the sidecar (`jevjudge`), the schema→questions compiler, the confidence cascade, offline mocks, tests |
| `switchyard/routes.*.toml` | Switchyard deployments for capability, escalation, and custom modes with Jev as `classifier_target` |
| `scripts/e2e.sh` | Boots mock upstream + mock Jev + jevjudge + the real `switchyard-server` and asserts 12 routing decisions. No API keys |
| `scripts/bench_judge.py` | p50/p95 latency and $/verdict for Switchyard-shaped judge requests (mock or real Jev) |
| `examples/embedded_libsy.py` | The no-sidecar path: a Python harness drives Switchyard's `libsy` algorithms and serves the judge call with Jev in-process |
| `docs/DECISION.md` | Why Switchyard, why not fork it, why a sidecar, what Jev can and cannot answer, economics, risks, roadmap |
| `docs/switchyard-prompts/` | Verbatim copies of Switchyard's packaged judge prompts and verdict schemas (the exact bytes the judge receives) |

## How a verdict is compiled

Switchyard's judge request is a system prompt (the rubric) + a message window + a
`response_format` JSON Schema. `jevjudge` maps each schema property to a Jev primitive:

| Schema property | Jev question | Decoded as |
|---|---|---|
| `boolean` | `noul` | `p >= 0.5` (threshold configurable) |
| `number` with `minimum: 0, maximum: 1` | `noul` | the probability itself |
| `string` with `enum` (≤255) | `choice` | selected label; criteria harvested from `- label: text` lines in the prompt |
| `integer` with a range ≤ 10, or numeric `enum` | `score` | nearest level |
| `string` without `enum` (`reason`, `crux`) | not asked | deterministic template, e.g. `jevjudge: p_escalate=0.870` |
| `const` / single-value `enum` | not asked | the literal |
| nested `object` | recursed | reassembled |

Profiles refine known schemas. For Switchyard's `CapabilityClassifierDecision`,
`capability_boundary` is **derived** from `primary_rule` (SUP→supported, UNC→uncertain,
LIM→unsupported, none→unmatched) rather than asked separately, because Switchyard rejects
verdicts where the two disagree. The capability card's rule lines become the `choice` criteria.
For `EscalationVerdict`, `escalate` becomes one `noul` with explicit true/false criteria.

The **gate confidence** of a verdict is the minimum over its decided fields (`choice`/`score`
confidence, `2·|p−0.5|` for a `noul`). Below `JEVJUDGE_MIN_CONFIDENCE`, jevjudge can
**fallback** (forward the identical request to an LLM judge), **abstain** (reply with non-JSON
so Switchyard fails open to its configured default/strong target), or **return** anyway (default).

## Quickstart (offline, no keys)

```bash
# 1. Python side
python -m venv .venv && . .venv/bin/activate
pip install -e 'jevjudge[dev]'
pytest jevjudge -q                       # 17 tests: compiler on the real Switchyard schemas + ASGI round trips

# 2. Switchyard server (Rust ≥ 1.96; the release binary builds in ~3 min)
cargo install --locked --git https://github.com/NVIDIA-NeMo/Switchyard.git --branch main switchyard-server

# 3. Full stack, deterministic mock Jev
SWITCHYARD_SERVER=$(command -v switchyard-server) PYTHON=.venv/bin/python scripts/e2e.sh
```

Expected: `passed=12 failed=0`, covering easy→weak / hard→strong / ambiguous→abstain→strong in
capability mode; weak→weak→strong latch (and no judge call after latching) in escalation mode;
4-way custom routing including low-confidence→abstain→`default_target`.

## Run it with real Jev

```bash
export TYPESAFE_API_KEY=...                       # console.typesafe.ai ($5 free credit at signup)
jevjudge --port 8090                              # OpenAI-compatible judge on :8090

# any of the three deployments; swap the mock upstream for OpenRouter/NVIDIA in the TOML
switchyard-server --config switchyard/routes.escalation.toml --port 4000

export ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_MODEL=escalation ANTHROPIC_API_KEY=placeholder
claude                                             # Claude Code now routes through Switchyard + Jev
```

Other Jev transports, same sidecar:

| Transport | Env |
|---|---|
| TypeSafe native (default) | `TYPESAFE_API_KEY`, optional `TYPESAFE_BASE_URL` |
| OpenRouter Decisions (alpha; wire format nearly identical, **unverified here**) | `JEVJUDGE_TRANSPORT=openrouter OPENROUTER_API_KEY=... JEVJUDGE_MODEL=typesafe/jev-latest` |
| Self-hosted clone serving `/v1/systemone` (openjev-sglang, kev, litjev, Decider) | `TYPESAFE_BASE_URL=http://your-host:port TYPESAFE_API_KEY=anything` |

Judge knobs (all env): `JEVJUDGE_MIN_CONFIDENCE` (default 0), `JEVJUDGE_LOW_CONFIDENCE_ACTION`
= `return|fallback|abstain`, `JEVJUDGE_FALLBACK_BASE_URL` / `_API_KEY` / `_MODEL` (an
OpenAI-compatible `/v1` base, e.g. OpenRouter + `google/gemini-3.5-flash`),
`JEVJUDGE_RUBRIC_PLACEMENT=state|drop`, `JEVJUDGE_MAX_STATE_CHARS` (default 100k; Jev's
state limit is 32k tokens), `JEVJUDGE_TIMEOUT_S`, `JEVJUDGE_PROFILE` (force a profile).

Every judge reply carries `x-jevjudge-source` (`jev|fallback|abstain`),
`x-jevjudge-confidence`, `x-jevjudge-latency-ms`, and a `jevjudge` object with the raw
probabilities so you can log calibration. `GET /v1/stats` on the sidecar aggregates them.

## Measured on the mock stack (this container, 4 vCPU)

These numbers exclude Jev itself (the mock answers in ~3 ms); they bound the sidecar's overhead
and give the *billable* token count per verdict, which is what the cost estimate needs.

| Verdict | jevjudge round trip p50 / p95 | Input tokens / verdict | Cost / verdict at $0.042/M |
|---|---|---|---|
| Capability (`p_solve` + `primary_rule`) | 18 ms / 56 ms | ~1,840 | ~$0.00008 |
| Escalation (`escalate`) | 25 ms / 54 ms | ~2,750 | ~$0.00012 |

For scale: LangChain's Switchyard escalation run spent about $0.64 on the judge over ~913 judged
turns (≈ $0.0007 per verdict, ~700 ms each) with Gemini 3.1 Flash Lite. Jev's list price puts the
same verdict at roughly a tenth of that, and TypeSafe/independent measurements put its round trip
at ~0.3 s. The sidecar adds tens of milliseconds. Real-key numbers: run
`scripts/bench_judge.py --url http://127.0.0.1:8090 --n 50 --mode escalation`.

## Status and honest caveats

* **Plumbing is validated end to end; routing quality is not yet measured with real Jev.**
  Everything above ran through the real Switchyard binary against a deterministic mock Jev.
  With a `TYPESAFE_API_KEY` the same commands exercise the real model; the next step is the
  Terminal-Bench 2.1 subset Switchyard ships (`benchmark/`) with Jev vs. the LLM judge.
* Jev's calibration is vendor-claimed and only partly independently checked; one community PR
  that replaced Switchyard's capability judge with a Jev `choice` lost 3 of 20 tasks vs. Opus.
  That is exactly why the confidence gate and cascade exist. Start with `fallback` to an LLM
  judge and tighten as your own logs show Jev's probabilities hold up.
* Free-text verdict fields (`reason`, `crux`) are templated, not written. Switchyard only
  requires them non-empty; anything that reads them for humans will see `jevjudge: …`.
* Switchyard is pre-1.0 (APIs move); OpenRouter's Decisions transport is implemented from its
  published description but not exercised here.

## Where this goes next

1. Real-Jev validation and a routing-accuracy comparison against the LLM judge on Switchyard's
   benchmark subset (the harness and profiles are already in the Switchyard repo).
2. A **multi-signal escalation profile**: Jev evaluates all questions in one call at marginal
   cost, so the trajectory judge can ask `looping`, `false_progress`, `drift`, `desperation`,
   `external_blocker` separately and combine them, which a text judge could never afford per turn.
3. Upstreaming: a native `format = "typesafe_systemone"` LLM client in `switchyard-llm-client`
   would remove the sidecar hop; Switchyard issue #723 and PRs #739/#762 are the live threads.
4. Self-hosting an API-compatible Jev clone (kev/openjev) behind the same sidecar for
   zero-marginal-cost judging, and distilling a task-specific classifier from Jev's own logs.
