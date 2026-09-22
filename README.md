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

On top of that, **`routerctl`** lets any team write their own routing policy as a few lines of
YAML — pick weak/strong/intent-bucket models, save the file, and it's live in seconds, with no
platform restart and no risk to any other team's routes. See
[**Multi-team routing**](#multi-team-routing-routerctl) below and
[`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md) for the schema.

## What is in the box

| Path | What |
|---|---|
| `vendor/switchyard/` | **Vendored source**, pinned commit (`vendor/switchyard/VENDORED_COMMIT`). Not a fork: unmodified, rebuilt by `scripts/build.sh`. See `vendor/NOTICE.md` |
| `jevjudge/` | Python package: the sidecar (`jevjudge`), the schema→questions compiler, the confidence cascade, offline mocks, tests |
| `routerctl/` | Python package: compiles per-team YAML into Switchyard TOML, and serves it with watch→recompile→validate→health-check→swap live reload |
| `teams/*.yaml`, `clients.yaml` | Real example configs: a voice-AI team (`intent` and `complexity` policies) and the platform default (`auto`, `escalation`) |
| `switchyard/routes.*.toml` | Hand-written Switchyard deployments for capability, escalation, and custom modes — the reference `routerctl` compiles down to |
| `evals/` | A concise (24-case), runnable model-comparison harness: accuracy per domain, cost, latency, graded automatically (exact-match or Jev-as-judge) |
| `scripts/build.sh` | One command: builds `switchyard-server`, its Python bindings, `jevjudge`, and `routerctl` into a project-local `.venv` |
| `scripts/e2e.sh` | Boots mock upstream + mock Jev + jevjudge + the real `switchyard-server` and asserts 12 routing decisions. No API keys |
| `scripts/e2e_routerctl.sh` | Proves live reload: edits a team's YAML while serving and asserts the change takes effect with no restart, and that a bad edit never goes live |
| `scripts/bench_judge.py` | p50/p95 latency and $/verdict for Switchyard-shaped judge requests (mock or real Jev) |
| `scripts/update_vendor.sh` | Re-vendor from a newer Switchyard commit or branch |
| `examples/embedded_libsy.py` | The no-sidecar path: a Python harness drives Switchyard's `libsy` algorithms and serves the judge call with Jev in-process |
| `docs/DECISION.md` | Why Switchyard, why not fork it, why a sidecar, what Jev can and cannot answer, economics, risks, roadmap |
| `docs/TEAM_CONFIG.md` | The YAML schema: all four policies, model definitions, what "live" means and what it costs |
| `docs/EVALS.md` | The eval direction: public benchmarks vs. a small own set vs. operational metrics, and why each stays separate |
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

## Multi-team routing (`routerctl`)

A team writes a route as YAML — pick a policy, name some models, save:

```yaml
# teams/voice-ai.yaml
team: voice-ai
routes:
  - name: deployment/transcription
    policy: intent                # auto | complexity | escalation | intent
    default: analysis
    models:
      extraction: { id: gpt-5.6-luna, client: openai, description: "field lookups: names, dates, amounts" }
      analysis:   { id: gpt-5.6-sol,  client: openai, description: "summarizing, sentiment, compliance risk" }
```

`routerctl` compiles every team's YAML in `teams/` (plus the platform-managed `clients.yaml`)
into one Switchyard TOML — the exact shape already validated in this README — and Jev judges
every `intent`/`complexity`/`escalation` route by default, at no extra config.

**"Live" is real**, not aspirational: Switchyard's server has no config hot-reload (checked
against the vendored source), so `routerctl serve` watches every file and, on a change,
recompiles, runs a genuine `switchyard-server --dry-run`, boots a new process, health-checks it,
and only then atomically swaps a thin proxy over to it before draining the old one:

```bash
routerctl validate teams/          # schema + a real --dry-run; no API keys needed
routerctl serve teams/ --port 4000  # edit any file in teams/ -- it's live within a few seconds
```

`scripts/e2e_routerctl.sh` proves this end to end: a live edit takes effect with no restart, a
broken edit is rejected and the last-good config keeps serving every team, and it recovers once
fixed — 6/6 passing. Full schema (all four policies, model definitions, per-route judge
overrides): [`docs/TEAM_CONFIG.md`](docs/TEAM_CONFIG.md).

## Choosing models: evals

`evals/` is a concise, runnable comparison harness — 24 hand-written cases across coding,
extraction, finance, and general reasoning, graded automatically (exact-match, or Jev-as-judge
for open-ended correctness) — plus the direction for the two things it deliberately doesn't try
to replace: public benchmarks for absolute model quality, and operational metrics (cost,
latency, availability) for what needs no ground truth at all. See
[`docs/EVALS.md`](docs/EVALS.md) for why these stay three separate layers, and
[`evals/run_eval.py`](evals/run_eval.py) `--help` to run it.

## Quickstart (one clone, offline, no keys)

Everything the router needs lives in this repository: Switchyard's source is vendored under
`vendor/switchyard/` (pinned commit, unmodified — see `vendor/NOTICE.md`), so there is no
`cargo install --git ...` step and no dependency on a released `nemo-switchyard` package, which
as of this writing lags Switchyard's own Python API.

Requires: [Rust via rustup](https://rustup.rs) (the vendored `rust-toolchain.toml` pins the exact
version and rustup installs it automatically), Python ≥ 3.11, and [`uv`](https://docs.astral.sh/uv/)
(falls back to `venv`/`pip` if absent).

```bash
scripts/build.sh          # builds switchyard-server + its Python bindings + jevjudge + routerctl, ~3 min

source .venv/bin/activate
pytest jevjudge routerctl -q     # 18 + 20 tests: compilers on real schemas, ASGI round trips, config validation

routerctl validate teams/         # the example team configs, schema + a real Switchyard --dry-run
scripts/e2e.sh                    # boots mock upstream + mock Jev + jevjudge + the real switchyard-server
scripts/e2e_routerctl.sh          # proves live reload: edit a team's config while serving, no restart
```

Expected: `passed=12 failed=0` from `e2e.sh`, covering easy→weak / hard→strong /
ambiguous→abstain→strong in capability mode; weak→weak→strong latch (and no judge call after
latching) in escalation mode; 4-way custom routing including low-confidence→abstain→`default_target`.
`passed=6 failed=0` from `e2e_routerctl.sh`: a live edit takes effect with no restart, a broken
edit is rejected and the platform keeps serving the last-good config, and it recovers once fixed.
All scripts default to this build's own binary and venv; pass `SWITCHYARD_SERVER=`/`PYTHON=` to
point at something else.

## Run it with real Jev

This path is verified, not aspirational: `REAL_JEV=1 scripts/e2e.sh` runs the identical 12-check
suite above against the live TypeSafe API instead of the mock — see **Status and honest
caveats** below for what that run found and fixed.

```bash
source .venv/bin/activate
export TYPESAFE_API_KEY=...                       # console.typesafe.ai ($5 free credit at signup)
jevjudge --port 8090                              # OpenAI-compatible judge on :8090

# any of the three deployments; swap the mock upstream for OpenRouter/NVIDIA in the TOML
vendor/switchyard/target/release/switchyard-server --config switchyard/routes.escalation.toml --port 4000

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

* **Validated end to end against the real TypeSafe API**, not just the mock: all three routes
  through the real, self-built `switchyard-server`, real Jev (`jev-1.13.0`) as the judge —
  11/11 checks (see `scripts/e2e.sh`; run it with `REAL_JEV=1`). Routing-*accuracy* — is the
  decision actually the right one on real tasks — is a separate, not-yet-run measurement; see
  the judge-only eval below.
* **Two real bugs found and fixed by that validation**, both worth knowing if you extend this:
  - *Gate confidence double-counted an already-discounted uncertainty.* Real Jev answered
    "print hello world in python" with `p_solve = 0.96` at confidence 0.92 (decisive), but its
    `primary_rule` choice split across several plausible labels at confidence 0.43 (no rule in
    the capability card covers "trivially easy"). Taking the minimum confidence over *both*
    fields abstained an obviously-easy task and sent it to the expensive model — even though
    Switchyard's own policy already widens the threshold `p_solve` must clear when
    `primary_rule` is uncertain (`threshold_step`), so gating on it too was double-counting.
    Fixed: the `CapabilityClassifierDecision` profile now gates on `p_solve` alone
    (`Compiled.gate_fields`), matching what Switchyard's policy actually thresholds on.
  - *The offline mock's keyword cues matched Switchyard's own rubric text, not just the
    conversation.* The escalation rubric's teaching prose legitimately uses "loop" and "doomed"
    to explain the pattern to a judge; since the whole rubric ships as `judge_instructions` in
    Jev's `state`, the mock's naive substring scan matched on every call regardless of actual
    content. Real Jev doesn't do keyword matching, so this never affected it — but it made the
    mock's escalation checks pass for the wrong reason. Fixed: the mock now scans only the
    conversation portion of state.
* **What real Jev's judgment looked like, honestly** (see `scripts/e2e.sh`'s `soft_expect`
  checks, which report rather than gate on these): given the same command failing identically
  twice, Jev was genuinely borderline (confidence 0.42, correctly triggering abstain) rather
  than confidently escalating — it needed a third or fourth identical failure to confirm,
  reading the rubric's "escalate only on a clear pattern... never on a single failed command"
  conservatively. And on a phrase engineered to carry no routing cue ("something with no
  routing cue"), Jev still picked a bucket with 0.86 confidence rather than expressing
  calibrated uncertainty — a real, useful data point: Jev's confidence reflects its own
  conviction, not whether a human would call the input ambiguous. Both are exactly why the
  confidence gate is configurable and the fallback/abstain cascade exists, not reasons to
  distrust the plumbing.
* Free-text verdict fields (`reason`, `crux`) are templated, not written. Switchyard only
  requires them non-empty; anything that reads them for humans will see `jevjudge: …`.
* Switchyard is pre-1.0 (APIs move); vendoring it at a pinned commit means this repo won't break
  under you, but also won't pick up upstream fixes until `scripts/update_vendor.sh` is run.
  OpenRouter's Decisions transport is implemented from its published description but not
  exercised here.
* Routing *accuracy* on real tasks — as opposed to "the API call works and the plumbing behaves
  sensibly", which is now verified — needs the judge-only eval and the Terminal-Bench 2.1
  subset Switchyard ships (`vendor/switchyard/benchmark/`), run with Jev vs. the LLM judge.
* **`routerctl`'s live reload is validated mechanically** (`scripts/e2e_routerctl.sh`, 6/6): a
  live edit takes effect without restart, a broken edit is rejected while the platform keeps
  serving every team's last-good routes, and it recovers once fixed. Two scope boundaries worth
  knowing: a swap costs a health-check round trip (treat "live" as seconds, not instantaneous),
  and the confidence gate / fallback judge is still one shared setting on the platform's
  `jevjudge` sidecar, not yet per-route — a team can pick a different judge model entirely
  (`judge:` in their YAML) but not yet a different confidence threshold from every other team.
* **The `evals/` harness has been run for real against `gpt-5.6-luna/terra/sol` and
  `gpt-6-astra`** (24 cases each, `python evals/run_eval.py --models
  gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol,gpt-6-astra`). Converged result:

  | model | overall | p50 latency | cost (24 cases) |
  |---|---|---|---|
  | gpt-5.6-terra | 100% | 1210ms | $0.0192 |
  | gpt-6-astra | 100% | 1979ms | $0.0939 |
  | gpt-5.6-luna | 96% | 1293ms | $0.0023 |
  | gpt-5.6-sol | 96% | 1579ms | $0.0413 |

  Getting there found **seven real bugs** — two in the harness's own API/grading code
  (`max_tokens` rejected by every model tested; grading extracted the first number in a
  step-by-step answer instead of the last, plus a follow-up LaTeX-brace edge case in that same
  fix), and four in the hand-written suite's own content (a self-contradictory calendar premise,
  an ambiguous logic puzzle, a compound OR-rubric, and a confusingly double-hedged rubric) —
  full narrative in `docs/DECISION.md` §10.1. None of these were models being wrong; every one
  surfaced by reading the raw response before accepting "model failed" at face value.
* **The real router was then run end to end**: `routerctl serve teams/` against real
  `teams/voice-ai.yaml`/`teams/platform.yaml`, real OpenAI models, and real Jev. This found an
  **eighth real bug** — `compile_routes` emitted every client in `clients.yaml` into the compiled
  TOML regardless of use, so a real (non-`--dry-run`) launch demanded a working API key for a
  client zero routes referenced. Fixed so only referenced clients are emitted or required
  (`docs/DECISION.md` §10.2). With that fixed, real traffic through `deployment/transcription`
  correctly bucketed an extraction prompt to `gpt-5.6-luna` and an analysis prompt to
  `gpt-5.6-sol`; real traffic through `deployment/qa-summary` correctly kept a trivial question,
  a multi-step finance calculation, and Einstein's five-house riddle all on the weak tier
  (`gpt-5.6-terra`) rather than over-escalating — consistent with `terra` scoring 100% above.
  Total real OpenAI spend across every experiment in this project: **~$0.72 of a $10 budget.**

## Where this goes next

1. Turn the real eval run's output into the committed `model-cards.json` `docs/EVALS.md`
   describes (`--output` already writes the raw per-case JSON) so `teams/*.yaml` authors have
   real numbers to pick models from, refreshed on a schedule instead of a one-off run.
2. Real-Jev routing-*accuracy* comparison against the LLM judge on Switchyard's benchmark subset
   (the harness and profiles are already in the Switchyard repo) — the judge-only eval design in
   `docs/EVALS.md` is the cheap first pass before a full routed run.
3. Per-route confidence-gate configuration: today `JEVJUDGE_MIN_CONFIDENCE` and the
   fallback/abstain cascade are one shared setting on the platform sidecar; a team should be able
   to tune how cautious *their* route's judge is without affecting anyone else's.
4. A **multi-signal escalation profile**: Jev evaluates all questions in one call at marginal
   cost, so the trajectory judge can ask `looping`, `false_progress`, `drift`, `desperation`,
   `external_blocker` separately and combine them, which a text judge could never afford per turn.
5. Upstreaming: a native `format = "typesafe_systemone"` LLM client in `switchyard-llm-client`
   would remove the sidecar hop; Switchyard issue #723 and PRs #739/#762 are the live threads.
6. Self-hosting an API-compatible Jev clone (kev/openjev) behind the same sidecar for
   zero-marginal-cost judging, and distilling a task-specific classifier from Jev's own logs.
