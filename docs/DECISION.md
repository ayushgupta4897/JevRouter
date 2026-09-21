# Decision: Switchyard as the router foundation, Jev as its judge, integrated as a sidecar

*Written 2026-09-21, six days after Jev's launch. Everything below was checked against the
Switchyard source at `main` (v0.3.0) and the `typesafe-sdk` 0.7.1 wire schemas on that date.*

## 1. The question

Complexity- and intent-based model routing has been hard to ship because the only thing that
classified well enough (an LLM) cost and delayed too much to sit in front of every call. This
is the same shape as early RAG: everyone hand-rolled chunkers and rerankers because the model
that could have done it was too expensive per call. Jev changes the per-decision economics by
two orders of magnitude. The question was: **what router foundation should we build on, and
where exactly does Jev go?**

## 2. What Switchyard actually is (from the code, not the blog)

NVIDIA NeMo Switchyard (Apache-2.0, Rust, pre-1.0) supersedes the deprecated
`NVIDIA-AI-Blueprints/llm-router`. It is three layers:

* **`switchyard-libsy`** – I/O-free routing algorithms. An `Algorithm` yields `Step::CallModel`
  (a model call the host must serve) and `Step::Done` (the pick). A `Classifier` scores
  targets by `Category` (`efficient`, `capable`, `judge`, `any`, or your own names).
* **`switchyard-llm-client` + `switchyard-translation`** – OpenAI Chat / Responses / Anthropic
  Messages in any combination.
* **`switchyard-server`** (standalone proxy), a **NeMo Relay plugin**, a **LiteLLM plugin**
  (decision-only; it cannot serve judge calls, so no classifier modes there), and **Python
  bindings** (`nemo-switchyard`, built from source) that expose the same algorithms.

The judge-backed algorithms all funnel through one helper, `JudgeClassifier` in
`algorithms/util/llm_judge.rs`:

1. build a request: system prompt = the packaged rubric, messages = a window of the
   conversation, `output.response_format` = a strict JSON Schema, `stream = false`;
2. `driver.call_model(request, models_for(Category::Judge))`, i.e. an ordinary LLM client
   with `format = openai_chat | anthropic_messages | openai_responses`;
3. parse `completion_text` as JSON (fence-stripped), validate against the schema, hand the
   verdict to a deterministic *policy* (thresholds, streaks, JSON-pointer target selector).

The three modes that use it and what their verdicts contain:

| Mode | Verdict schema | Policy |
|---|---|---|
| `capability` | `crux: string`, `primary_rule: enum[10]`, `capability_boundary: enum[4]`, `p_solve: 0..1` | route weak iff `p_solve ≥ base_threshold + steps·threshold_step`; invalid ⇒ strong |
| `escalation` | `escalate: bool`, `reason: string` | serve weak reply; `confirmations` consecutive `true` latch the session to strong |
| `custom` | any inner schema you write | JSON pointer selects a model group; unparseable ⇒ `default_target` |

Two facts matter for the design. First, the packaged capability prompt literally asks the LLM
to *forecast a calibrated probability* ("interpret probabilities as natural frequencies").
LLM-emitted probabilities are known to be poorly calibrated; an independent Jev test found that
when Jev said 0.8 the label agreed ~80% of the time, while a generative model writing "0.8"
agreed about half the time. Second, the judge is invoked on **every turn** in escalation mode
until the session latches, so its latency and price are paid per turn, not per task.

Benchmarks in the Switchyard repo and its partners' write-ups:

| Setup | Accuracy | Cost | Judge share |
|---|---|---|---|
| Terminal-Bench 2.1, Opus 4.8 alone | 76.0% | $98.06 | — |
| TB2.1, escalation (Opus 4.8 / GLM 5.2, DeepSeek judge) | 75.7% | $85.00 | n/a |
| TB2.1, capability (Task) | 71.2% | $79.32 | n/a |
| LangChain 145 agent tasks, Opus 4.8 alone | 86.0% | $11.45 | — |
| LangChain, escalation (Opus 4.8 / Nemotron 3.5, Gemini 3.1 Flash Lite judge) | 80.0% | $3.00 | **21.2%**, ~700 ms/turn |

LangChain's authors called the judge "the most obviously improvable part".

## 3. What Jev actually is (from the SDK, not the launch post)

`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer`, body
`{"model": "jev-latest", "state": str|object|array, "questions": {name: question}}`.

| Question | Request | Answer |
|---|---|---|
| `noul` | `instructions`, optional `criteria.{true,false}` | `noul: 0..1` |
| `choice` | `instructions`, `criteria: {label: description|null}` (≤255) | `choice`, `confidence`, `probabilities{label}` |
| `score` | `instructions`, `criteria: [level0, level1, …]` (≤10) | `score` (expected value), `confidence`, `legend`, `probabilities` |

Limits: 32k tokens of state, 64k total; 1,200 req/min; 70–500 ms (vendor), ~0.3 s median
(independent). Price $0.042/M input, output free. All questions in one request are answered in
one parallel pass, so a second question costs only its own tokens and no extra latency.

Jev cannot generate strings, explain, do arithmetic, or reason multi-step. It can be
"confidently wrong": the no-hallucination claim is about output *shape*, not truth.
Independent checks report ~5× faster and ~8.6× cheaper than Mistral Small 4 on a real task
(not the 193×/445× marketing multiples), 67.8% on TypeSafe's own 4-workflow eval (ties Sonnet 5,
trails Opus 5 at 73.1%), and good but not perfect calibration ("overconfident on an unseen
priority rule" in one out-of-distribution study).

Within a week of launch: OpenRouter serves it at `POST /api/alpha/decisions` (alpha), Vercel's
AI Gateway "auto" router defaults to it, Cloudflare Workers AI lists it, and there are at
least ten open-source reproductions, several **wire-compatible with `/v1/systemone`**
(openjev-sglang on Qwen3.6-35B-A3B, kev 0.8–9B, litjev, Decider). Those are the "20× cheaper in
four days" options: same client, `TYPESAFE_BASE_URL` repointed, you pay for GPUs instead of tokens.

## 4. Prior art on exactly this combination

| Where | What | Gap |
|---|---|---|
| Switchyard issue #723 (maintainers) | Asks for a TypeSafe-backed classifier; requires libsy stay I/O-free, creds from env, probabilities in telemetry | Design not settled |
| PR #739 (open, Sep 17) | New route type `type_safe_classifier`, new `switchyard-typesafe-client` crate, Choice over option labels, ~250–320 ms measured | Capability-style only; CodeRabbit nits; unmerged |
| PR #762 (open) | Choice over configured target names, averages three option orders to cancel order bias, keeps probabilities | Benchmark: **14/20 tasks vs 17/20 for Opus** (−3) at −18% cost; unmerged |
| PR #724 (docs) | 10-sample smoke test: both judges 100%, Jev faster | No implementation |
| `inkwell-finance/jev-switchyard`, `LeonardSEO/switchyard` | Forks replacing the capability classifier with Jev; the latter cascades Jev → chat judge → keywords | Forks drift from upstream; full benchmark "still needs to be run" |
| DevelopersIO write-up | Standalone: 40/40 on a toy 4-tier set, 0.64 s median, $0.000025/call | Not integrated |
| `langchain_typesafe` `ModelRouterMiddleware` | Jev picks a model inside a LangChain agent | LangChain-only, no escalation/affinity |

Nobody has (a) a zero-fork integration, (b) touched **escalation** mode, the mode with the
best accuracy retention and the highest judge cost, or (c) a confidence-gated fallback in the
upstream proposals, despite #762's accuracy regression showing why one is needed.

## 5. Options considered

| | A. Fork Switchyard / native Rust `type_safe_classifier` | **B. Unforked Switchyard + Jev judge sidecar** | C. Own Python router calling Jev directly | D. LangChain `ModelRouterMiddleware` |
|---|---|---|---|---|
| Works today with released Switchyard | no (source build of a fork) | **yes** | n/a | n/a |
| Covers capability / escalation / custom / composite / stage-classifier | only what the fork implements (#739: capability-like) | **all judge-backed modes** | must reimplement | none of Switchyard's |
| Keeps translation, affinity, streaks, fallbacks, metrics, benchmark harness | yes | **yes** | no | no |
| Survives upstream churn (pre-1.0) | merge conflicts | **yes; sidecar is a plain HTTP judge** | yes | yes |
| Honors maintainers' I/O-free-libsy constraint | depends | **yes** | n/a | n/a |
| Portable to other routers / gateways | no | **yes: anything that calls an OpenAI-compatible judge** | no | no |
| Extra hop latency | 0 | ~5–20 ms (measured) | 0 | 0 |
| Confidence cascade | would need adding | **built in** | would need adding | no |
| Self-hosted Jev clones | needs client support | **env var** | yes | no |

**Decision: B.** A native Rust client is still the right *eventual* home for the capability
path (it removes the hop and gives the runner first-class telemetry), and the sidecar's schema
compiler is the spec for it. Until #723 settles, B is strictly more useful and carries no fork
debt.

## 6. Design

### 6.1 The schema → questions compiler (`jevjudge/compiler.py`)

Deterministic, schema-driven, with optional profiles:

* `boolean` → `noul`; `number ∈ [0,1]` → `noul` (value = probability); `string enum` → `choice`;
  `integer` range ≤ 10 or numeric enum → `score`; `const`/single enum → literal; nested objects
  recurse (question names are `a__b`); local `$ref` resolved.
* Free-text fields are templated (`reason: "jevjudge: p_escalate=0.870"`). Switchyard only checks
  they are non-empty.
* **Criteria harvesting**: `- LABEL [tag]: text` lines in the judge prompt become `choice`
  criteria. Switchyard's capability card is written exactly that way, and custom-mode prompts
  can adopt it. `choice` options without a harvested line are sent by name only.
* **Profiles** keyed by `json_schema.name`:
  * `CapabilityClassifierDecision`: tailored instructions for `p_solve` (SUCCESS definition) and
    `primary_rule`; `capability_boundary` is *derived* from `primary_rule`'s prefix, never asked,
    because Switchyard's `TaskClassifierVerdict::is_valid()` rejects a mismatch.
  * `EscalationVerdict`: `escalate` gets the distilled trouble-pattern definition plus explicit
    true/false criteria.
  * Everything else (custom mode's `switchyard_classifier_response`) uses the generic rules.
* `json_object` mode (for providers without JSON-Schema support) is handled by parsing the
  schema Switchyard appends to the prompt.

### 6.2 State

`{"judge_instructions": <system prompt>, "conversation": [{role, content}, …]}`. The rubric is in
state once, not per question, to keep billable tokens flat. Oldest non-task messages are dropped
first to stay under `JEVJUDGE_MAX_STATE_CHARS` (default 100k chars ≈ 25k tokens, under Jev's 32k
state limit; Switchyard's escalation window is capped at 18k chars anyway).

### 6.3 Confidence gate and cascade (`jevjudge/cascade.py`)

Gate confidence = min over decided fields (`choice`/`score` confidence, `2·|p−0.5|` for a noul).
Below `JEVJUDGE_MIN_CONFIDENCE`:

* `fallback`: forward the **identical** OpenAI request to `JEVJUDGE_FALLBACK_BASE_URL` with
  `JEVJUDGE_FALLBACK_MODEL` (an LLM judge). Because the sidecar already speaks the judge's wire
  format, the fallback is a pure pass-through and the LLM judge sees exactly what it would have
  seen without Jev.
* `abstain`: reply with non-JSON. Switchyard's decoders treat that as fail-open: capability
  ⇒ `strong_target`, custom ⇒ `default_target`, escalation ⇒ serve the buffered weak reply and
  hold the streak. This turns Jev's uncertainty into the router's own safe default with zero
  new code in the router.
* `return` (default): hand back Jev's verdict, flag `low_confidence` in metadata.

Jev transport errors also fall back when a fallback is configured, otherwise surface as 502 and
Switchyard fails open per its own rules.

### 6.4 Embedded path (`jevjudge/libsy.py`, `examples/embedded_libsy.py`)

For harnesses that embed `libsy` in Python (the README "Path 2"), `serve_judge_call` converts
libsy's normalized request (instruction blocks, content blocks, `output.response_format`) to the
OpenAI shape and answers the `CallModel` in-process. Session identity is a
`x-switchyard-session-id` header on `run_stream`, same as the server.

## 7. Economics, with the arithmetic shown

Per verdict, measured billable input on Switchyard's real prompts through this sidecar:
capability ≈ 1,840 tokens, escalation ≈ 2,750 tokens.

* Jev list price: 1,840 × $0.042/M ≈ **$0.00008**; 2,750 × $0.042/M ≈ **$0.00012**. Output free.
* LangChain's measured judge spend: 21.2% of $3.00 = $0.64 over 145 tasks × 6.3 calls ≈ 913
  judged turns ⇒ **≈ $0.0007 per verdict** on Gemini 3.1 Flash Lite, ~700 ms each.
* So per verdict Jev is roughly **6–9× cheaper** than the cheapest LLM judge they used and
  **2–3× faster** (0.3 s vs 0.7 s), *not* 400×: the judge prompt is ~2k tokens either way and
  Flash-Lite-class models are already cheap. The 400× figure compares against frontier models.
* What that buys: the judge line drops from 21% of routed spend to ~2–3%; per-turn routing
  overhead drops by ~0.4 s across every unlatched turn; and, more importantly, the marginal cost
  of a *second* question is now only its tokens with no added latency, which is what enables
  multi-signal routing (section 9).

Self-hosted clones move this to fixed GPU cost; a 4B–9B decision model on one small GPU serves
thousands of verdicts per minute.

## 8. Risks and mitigations

| Risk | Evidence | Mitigation in this repo |
|---|---|---|
| Jev's forecast of `p_solve` on hard coding tasks is worse than an LLM's | #762 lost 3/20 tasks; Jev trails Opus 5 on TypeSafe's own eval | confidence gate + `fallback` to an LLM judge; abstain ⇒ Switchyard's safe default; log probabilities via `jevjudge` metadata for calibration audits |
| Calibration not independently proven | "no paper, no calibration curve"; OOD overconfidence reported | same; start with a conservative gate (0.5–0.6) and tighten from your logs |
| Free-text fields templated | Switchyard requires non-empty only | documented; a `reason` template still carries the probability |
| Criteria harvesting misses a prompt style | heuristic regex | falls back to label-only options; profiles can pin criteria explicitly |
| Switchyard pre-1.0 API churn | README says so | the sidecar depends only on the OpenAI judge contract, which is the stable part |
| OpenRouter Decisions wire drift (alpha) | "nearly identical", unverified here | transport isolated behind one flag; TypeSafe native is default |
| Judge outage | — | Switchyard `timeout_ms` on the judge client; sidecar returns 502; router fails open |

## 9. Roadmap

1. **Validate with a real key** (`scripts/e2e.sh` with `REAL_JEV=1`, `scripts/bench_judge.py`),
   then run Switchyard's `benchmark/` TB2.1 subset with the Jev judge vs. the LLM judge on the
   escalation profile, tracking accuracy, cost, judge share, and calibration (Brier/ECE from
   the logged probabilities against task outcomes).
2. **Multi-signal escalation profile**: ask `looping`, `false_progress`, `drift`, `desperation`,
   `external_blocker` as separate nouls in the same call; combine with a small rule (e.g. escalate
   iff any of the first four ≥ τ and `external_blocker` < τ′). This needs a custom policy in
   Switchyard's escalation shell, so it doubles as the upstream contribution.
3. **Upstream**: propose `format = "typesafe_systemone"` for `switchyard-llm-client` using this
   compiler as the spec, aligned with #723's I/O boundary. Until then the sidecar is the adapter.
4. **Composite/stage**: `composite` and `stage_router` accept a classifier; the same sidecar
   serves them unchanged (not exercised in e2e yet).
5. **Distillation**: after the routing feature is proven, log Jev's inputs/outputs and train a
   task-specific encoder classifier (Laya/ModernBERT class) for zero-marginal-cost judging.

## 10. Sources

Switchyard: repo README, `docs/routing_algorithms/*.md`, `crates/libsy/src/algorithms/util/llm_judge.rs`,
`crates/libsy/src/prompts/*`, `benchmark/routing-profiles/*`; NVIDIA blog "Route AI Agent Workloads
Across Models with NVIDIA NeMo Switchyard"; LangChain "How many of your agent's calls actually need
a frontier model?"; Switchyard issue #723, PRs #724, #739, #762.
Jev: typesafe.ai launch post; `typesafe-sdk` 0.7.1 (`_schemas/models.py`, generated from
api.typesafe.ai/openapi.json); docs.typesafe.ai/api; Pydantic AI TypeSafe docs; LangChain "Building a
harness with Jev"; APIMaster "Jev vs LLMs"; "The Jev File" independent checks; systemonemodels.org
alternatives index; OpenRouter Decisions endpoint notes; DevelopersIO "replacing model routing with
TypeSafe (Jev)"; Sean Goedecke, "System One models can train their own replacements".
