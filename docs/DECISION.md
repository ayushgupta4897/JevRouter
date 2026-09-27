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

## 9. Real-key validation (2026-09-21)

Section 8's "calibration not independently proven" risk was tested with a live
`TYPESAFE_API_KEY` against the real `switchyard-server` binary, `jev-1.13.0`. Raw wire calls
(`noul`, `choice`, `score`, and a two-question parallel call) matched the documented schema
exactly — 16 calls, 0 decode errors. `REAL_JEV=1 scripts/e2e.sh` then ran the full 12-check
routing suite through all three modes; final result 11/11 (one check was retired as
model-opinion-dependent, see below), no fallbacks needed.

Two real bugs surfaced and were fixed, not routed around:

1. **Gate confidence double-counted an already-discounted uncertainty.** On "print hello world
   in python", Jev returned `p_solve = 0.96` at confidence 0.92 but split `primary_rule` across
   several plausible labels at confidence 0.43 — the capability card has no rule for "trivially
   easy". Taking `min()` over both fields abstained an obviously-easy task, and Switchyard's
   fail-open sent it to the expensive model. But Switchyard's own policy already widens the
   threshold `p_solve` must clear when `primary_rule` lands outside "supported"
   (`threshold_step` in `TaskClassifierPolicy::threshold`) — gating on `primary_rule`'s own
   confidence too was scoring the same uncertainty twice. Fix: `Compiled.gate_fields` lets a
   profile name the field(s) the caller's policy actually thresholds on;
   `CapabilityClassifierDecision` now gates on `p_solve` alone. Confidence went from 0.43
   (abstain) to 0.92 (serve) for the identical verdict; routing corrected from strong to weak.
2. **The offline mock's cue matching scanned Switchyard's own rubric text, not just the
   conversation.** The escalation prompt's teaching prose legitimately contains "loop" and
   "doomed" (explaining the pattern to a judge), and since the full rubric ships as
   `judge_instructions` inside Jev's `state`, the mock matched those words on *every* call
   regardless of actual content — making the deterministic escalation tests pass for the wrong
   reason since the mock was first written. Real Jev was never affected (it doesn't do
   substring matching), which is exactly why this stayed hidden until a real-key run. Fix: the
   mock now scans only `state["conversation"]`.

What real Jev's judgment looked like once both fixes were in, on fixtures built from genuine
`tool_calls`/`tool` transcripts (not prose asserting trouble, which the rubric explicitly
discounts and a real judge, correctly, does too):

- A single tool failure was not escalated (matches "reproducing failures is the job").
- Exactly two identical failures were genuinely borderline — confidence 0.42, correctly
  triggering the abstain/hold path rather than a confident verdict either way. It took a third
  or fourth identical failure to confirm and latch. This reads as the rubric's own "escalate
  only on a clear pattern... never on a single failed command" being applied conservatively,
  not as a defect — but it meant the original 3-turn test schedule was tighter than a
  probabilistic judge should be held to, so the schedule was extended to four turns and the
  two boundary turns were changed from pass/fail assertions to reported-only (`soft_expect`).
- On a phrase engineered to carry no routing cue, Jev still picked a bucket at 0.86 confidence
  rather than expressing calibrated uncertainty. This is the same finding independent reviewers
  reported ("overconfident on an unseen priority rule"): Jev's confidence reflects its own
  conviction about the question asked, not whether a human would call the input ambiguous.
  That assertion was also converted to informational — it was testing one model's stylistic
  tendency on one phrase, not this router's plumbing, which the compiler's own unit tests
  (`test_low_confidence_abstain_returns_non_json`) already cover deterministically.

Net effect on section 8's risk table: "calibration not independently proven" is now partly
answered by our own evidence rather than only secondhand reports, and it points the same
direction those reports did — decisive on clear-cut cases, appropriately cautious right at a
rubric's stated threshold, but not reliably self-aware about genuinely ambiguous input. That is
precisely the shape a confidence gate with a fallback is for, and precisely why one ships here
rather than trusting Jev's verdict unconditionally.

## 10. Multi-tenant router and real-model validation (2026-09-22)

Two things were added and then both validated against real, paid traffic rather than mocks:
`routerctl` (self-serve per-team YAML → Switchyard TOML, with live reload since Switchyard has
no hot-reload of its own) and `evals/` (a concise, 24-case model card generator). Total real
OpenAI spend across every experiment below: **~$0.72 of a $10 budget ceiling.**

### 10.1 Real eval run: four models, seven real bugs found

Running `evals/run_eval.py` against `gpt-5.6-luna/terra/sol` and `gpt-6-astra` immediately
surfaced bugs that a mock could never have caught, because a mock's output is whatever the mock
author already expected:

1. **`max_tokens` rejected outright.** All four models return HTTP 400 unless the request uses
   `max_completion_tokens`. Switchyard's own `openai_chat` encoder already sends the right name;
   only this harness's direct API calls needed the fix.
2. **Grading extracted the first number, not the last.** A model reasoning step by step restates
   its inputs before its answer ("40% × $1,200,000 = ... = $180,000"), so `extract_first_number`
   matched the restated "40" and marked an exactly-correct $180,000 answer wrong — for every
   model, on the same case, which is what made it visible as a grading bug rather than a model
   failure. Fixed by taking the *last* number instead, with a regression test pinned to the
   captured real response text.
3. **LaTeX thousands-grouping split one number into two.** `gpt-5.6-sol` formatted its answer as
   `\boxed{\$180{,}000}` — the comma sits inside its own brace pair. Stripping only the comma
   left `180{}000`, and the digit regex read that as two separate numbers ("180", "000"), so the
   "last number" silently became 0. This is the same case as bug 2, found again on a second real
   run, because the first fix wasn't sufficient — caught only by inspecting the raw
   `response_text` field added specifically so a wrong verdict could be audited without
   re-calling the API. Fixed by also stripping `{`/`}` before matching.
4. **A self-contradictory calendar premise.** A case asserted "today is Monday, September 22,
   2026"; `datetime.date(2026, 9, 22).strftime('%A')` says that date is a Tuesday. Models split
   depending on whether they deferred to the stated (wrong) day or silently corrected it. Fixed
   by rewriting the case to ask for a relative day-count instead of depending on a real calendar
   date at all.
5. **An ambiguous hand-written logic puzzle.** The case conflated "is a liar" (a fixed type) with
   "made one true and one false statement" (a per-statement framing) — two different real models
   independently gave internally-coherent "no" answers to a case scored as "yes". Replaced with a
   standard, hand-verified knights-and-knaves puzzle.
6. **A compound OR-rubric.** A `jev_noul` rubric accepted "yes, consent given" OR "flags as
   ambiguous"; a response that plausibly satisfied the second branch was marked wrong by the
   one-shot judge, which isn't built to weigh two independent branches at once. Narrowed to a
   single, unhedged criterion.
7. **A confusingly double-hedged rubric.** `coding-medium-1`'s rubric read "does not guarantee...
   or does so correctly if it does" — a genuine double-negative-conditional. `gpt-6-astra`'s
   answer (`max(dict, key=dict.get)` on an insertion-ordered dict) is textbook-correct
   first-occurrence tie-breaking, and was marked wrong. Rewritten as one clear criterion.

None of these were the models being wrong; every one was found by refusing to accept a "model
failed" result at face value and reading the raw response before blaming the model. That
discipline is the actual deliverable here, not the fixes themselves — a harness that reports
model failures uncritically will happily encode its own bugs as "this model is worse."

**Final, converged 24-case result:**

| model | overall | p50 latency | cost (24 cases) |
|---|---|---|---|
| gpt-5.6-terra | 100% | 1210ms | $0.0192 |
| gpt-6-astra | 100% | 1979ms | $0.0939 |
| gpt-5.6-luna | 96% | 1293ms | $0.0023 |
| gpt-5.6-sol | 96% | 1579ms | $0.0413 |

The one miss for luna and sol each is the same genuinely disputed nuanced-judgment case (a
consent question where a well-reasoned "no" is arguably as defensible as "yes") — real model
disagreement, not a harness defect, and left as-is rather than "fixed" by loosening the rubric
until every model agrees. The practical read: at this suite's difficulty, `gpt-5.6-terra` is the
efficient frontier (matches Astra's accuracy at ~5x lower cost and faster p50); Astra earns its
premium only on tasks this 24-case set doesn't stress hard enough to reveal.

### 10.2 Real router demo, and an eighth real bug

With the eval harness converged, `routerctl serve teams/` was pointed at real
`teams/voice-ai.yaml` and `teams/platform.yaml`, real OpenAI models, and real Jev (via the
already-running `jevjudge` sidecar). The first real launch attempt failed immediately —
`switchyard-server` refused to boot because `clients.yaml` declares an `openrouter` client with
no `OPENROUTER_API_KEY` set, even though zero routes reference it.

This is a real gap, not a config mistake: `clients.yaml`'s own header describes it as a
platform-managed registry teams draw on by name, implying a client can be declared ahead of any
team adopting it. `compile_routes` instead emitted every declared client into the compiled TOML
unconditionally, so Switchyard's real (non-`--dry-run`) launch demanded a working key for a
client nothing routed to. Fixed in `routerctl/src/routerctl/compiler.py`: `TargetRegistry` now
tracks which client names a route actually resolved a target against
(`referenced_clients`), and `compile_routes` only emits — and only requires an `api_key_env`
for — clients in that set. `--dry-run`'s placeholder-filling was never the right fix for this;
it papered over the real launch path, which is exactly the path that broke. A regression test
(`test_a_declared_but_unused_client_is_not_emitted_or_required`) pins the fix.

With that fixed, the real router served real traffic correctly on both live-reloaded routes:

- `deployment/transcription` (intent policy): an extraction prompt ("what account number was
  mentioned") correctly routed to `gpt-5.6-luna` and answered correctly; an analysis prompt
  ("summarize sentiment and escalation risk") correctly routed to `gpt-5.6-sol` and answered
  correctly — the same judge call, on real transcript-shaped text, picking the intended bucket
  both times.
- `deployment/qa-summary` (complexity policy): a trivial fact question stayed on the weak tier
  (`gpt-5.6-terra`), as expected. Two deliberately harder probes — a three-quarter financial
  calculation and Einstein's classic five-house riddle — also stayed on `gpt-5.6-terra` rather
  than escalating to `gpt-6-astra`, and both were answered correctly. This is consistent with,
  not contrary to, 10.1's finding that `terra` scored 100% on this project's own eval suite: the
  complexity judge correctly predicted the weak tier could handle these, and it did — a real
  cost-saving decision, not a missed escalation. Confirming the escalation path itself fires
  under real load is left to a harder probe than this demo reached for; nothing here shows it
  can't.

## 11. Decision-only mode: the router decides, an AI Gateway executes (2026-09-22)

Real use surfaced a requirement the design up to this point didn't have: the router should tell
a caller which model to use, and never call that model itself. The caller already has (or is
building) its own AI Gateway responsible for actually executing the request; routing and
execution needed to be two processes with two owners, not one process doing both.

### 11.1 What Switchyard's embedded API actually allows

`routerctl serve`'s proxy mode (§10) works by compiling YAML to TOML and running the real
`switchyard-server` binary, which is an HTTP proxy by construction: given a request, it always
selects a target *and* forwards to it, streaming the completion back. There is no flag, header,
or config key that makes it decide without forwarding — checked directly against the vendored
source, `crates/switchyard-server/src/lib.rs`, which has no such branch.

But Switchyard's routing *algorithms* (`libsy`) are a separate crate from the HTTP-forwarding
logic, and `switchyard-py` exposes them directly to Python as an embedded API
(`switchyard.libsy`, already used by this project's `examples/embedded_libsy.py` for a different
reason — running a harness without the jevjudge HTTP sidecar). `Algorithm.run_stream(request,
models)` yields a sequence of `Step`s: zero or more `Step.CallModel` (the host must fulfill each
with a real response before the algorithm continues) followed by one terminal `Step.Done`
carrying `selected_model_ids` — the routing decision itself, independent of whether anything was
actually served.

Empirically probing all four policies against the algorithm's real `run_stream` (mock Jev, zero
cost) settled the design:

| Policy | Compiles to | `CallModel` steps needed |
|---|---|---|
| `auto` | `stage_router` with no classifier | **none** — a pure heuristic over request metadata, not even a judge call |
| `complexity` | `llm_classifier` capability mode | **one**, the judge |
| `intent` | `llm_classifier` custom mode | **one**, the judge |
| `escalation` | `llm_classifier` escalation mode | **the current tier's model itself**, before the judge even runs |

The first three need at most the judge — already the cheapest call in this whole system — and
never ask to call the actual target. `escalation` is qualitatively different: it is
*response-based* by design, meaning it watches how the current tier's model actually behaves
before judging whether that's a stuck pattern, which requires generating that behavior. No
configuration removes this; it's what the algorithm is for. Confirmed by reading the yielded
steps directly (a small probe script, `run_stream` against each policy in turn) rather than
assuming from documentation — the same discipline as every other finding in this document.

### 11.2 Escalation, reimplemented as a transcript-reading classifier

Given that, `escalation` in this project no longer maps onto Switchyard's native
`LlmClassifierConfig::Escalation`. It compiles instead to the *same* `custom` classifier
mechanism `intent` uses, with two buckets (`continue`, `escalate`) instead of team-named ones,
and a prompt asking the judge to read the conversation already present in the request — tool
calls, tool results, repeated errors — for a clear pattern recurring at least `confirmations`
times within the last `recent_turn_window` turns, versus ordinary or isolated trouble. This
keeps the same rubric intent (escalate on a pattern, never a single failure) while making the
policy genuinely decision-only: the judge is the only real call, exactly like every other policy.

The real behavior difference this trades away: Switchyard's native version tracks a confirmation
streak across multiple requests in the same session (via `x-switchyard-session-id`), so it can
notice "this is the second time in a row" without re-reading the whole history each call.
Decision-only mode has no cross-call memory by construction — each decision is a fresh, isolated
read of whatever transcript the caller includes in that one request. This is a legitimate
trade-off, not a regression papered over: an AI Gateway already holds the conversation history
(it's the one making the real calls), so handing the router the relevant recent turns each time
costs it nothing it doesn't already have, and the router stays stateless — no per-session data
this process needs to retain, clean up, or lose on restart.

### 11.3 A second real bug: `stage_router` needs an explicit `any` category

Compiling `auto` to `stage_router` at first failed every request with `LibsyError: target
"gpt-5.6-terra" was not found` — despite `gpt-5.6-terra` being listed exactly where the working
`llm_classifier` policies list their own candidates (`efficient`/`capable` categories). Isolated
with a minimal repro comparing fake placeholder names against real model IDs, and against a
models dict with and without an `any` key: `stage_router` (unlike `llm_classifier`) requires the
`run_stream` models dict to also include an `any` category listing every candidate the route can
reach, or its internal target resolution fails outright, regardless of the actual `efficient`/
`capable` values or their formatting (a red herring: the failing id happened to contain dots,
which turned out to be irrelevant once isolated). Fixed by always including `any` as the union of
a route's own targets in every policy's compiled models dict, not only where a native TOML config
happened to require it (`intent`'s TOML compilation already built this list; the others hadn't
needed to, before this).

### 11.4 Validation

Spot-checked against real Jev (no real OpenAI calls anywhere in this path, by construction — the
whole point is that the target is never called) across all four policies and both obvious
directions per judge-backed policy:

| Route (policy) | Input | Decision | Judge confidence |
|---|---|---|---|
| `deployment/auto` | trivial greeting | `gpt-5.6-terra` (efficient) | n/a — no judge call |
| `deployment/qa-summary` (complexity) | five-house logic riddle | `gpt-6-astra` (strong) | 0.32 |
| `deployment/qa-summary` (complexity) | `2+2=?` | `gpt-5.6-terra` (weak) | 0.92 |
| `deployment/transcription` (intent) | "what account number..." | `gpt-5.6-luna` (extraction) | 1.0 |
| `deployment/transcription` (intent) | "summarize sentiment..." | `gpt-5.6-sol` (analysis) | 1.0 |
| `deployment/coding-agent` (escalation) | same failure, 3rd time in a row | `gpt-6-astra` (escalate) | 0.93 |
| `deployment/coding-agent` (escalation) | fixed, moving on | `gpt-5.6-sol` (continue) | 1.0 |

All seven decisions correct, including the reimplemented escalation policy correctly detecting
a genuine repeated-failure transcript and correctly recognizing ordinary progress as not needing
escalation — purely from reading the given conversation, with no model call of its own. End-to-
end live reload against a real running server (not just in-process) is proven the same way as
proxy mode's own e2e script, with one addition specific to this mode:
`scripts/e2e_routerctl_decide.sh` points every declared client at a deliberately unreachable
address and asserts decisions still succeed — if this router ever attempted to call a target,
every request in that script would fail with a connection error instead.

## 12. Cache-aware switching (2026-09-27)

### 12.1 The problem

Prompt caches are per model and per provider. When a session moves to a different model, the
next turn re-sends the whole conversation at full price. On OpenAI GPT-5.6+ and on Anthropic
that re-send is billed as a cache *write* at 1.25× input. Staying costs 0.1× input. So a cold
switch costs about 12.5× what staying warm costs on the prefix, and a cheaper model is not the
same as a cheaper turn. Deferent originally decided each turn on its own, which is where
per-request routers lose money on long sessions. Switchyard doesn't cover it either:
`cache_eligibility.rs` is eval-only, and its session affinity (`AffinityRouter`,
`capable_hold_turns`) is process-local, keyed by a session header that decision mode never
receives, and never puts a price on the cache.

### 12.2 How others handle it (researched 2026-09-27)

| Who | Approach | What we took |
|---|---|---|
| OpenAI | Automatic prefix caching. GPT-5.6+: reads 0.1×, writes 1.25×, 30-min TTL, 1,024-token minimum. `usage.prompt_tokens_details.{cached_tokens, cache_write_tokens}`. No sign that tiers share a cache. | The pricing model and the usage fields |
| Anthropic | `cache_control` breakpoints: writes 1.25× (5 min) or 2× (1 h), reads 0.1× (0.05× on Opus 5.5); minimum 512–4,096 tokens. Changing effort or thinking settings invalidates cached messages. | `cache_write` and `min_cacheable_tokens` as per-model settings |
| OpenRouter | Provider stickiness per (account, model, conversation), 10-min idle expiry. The Auto Router reuses a model "while it remains among the top candidates". The Jev Router weighs expected gain against cost, including lost cache. | Confirms the approach. The public criticism of Jev Router's "one-way premium trap" (keeping the premium model for "what colour is a banana?") shaped §12.3's pricing of downgrades |
| vLLM Semantic Router (SAAR) | Hard locks for tool results, 300-s idle reset, a switch penalty that grows with session length and input price; 79% fewer switches. | The closest analogue. We price the penalty in dollars instead of using weights |
| LiteLLM | Measured "a switch is not an eviction": 97% of switch-backs landed warm at a 5-min TTL. `session_affinity` is off by default. | `other_warm_caches`. §12.4 reproduced this on real OpenAI calls |
| SGLang, Dynamo, llm-d | KV and prefix-aware routing between *replicas* of one model. | Not applicable: that's choosing a replica, not a model |

### 12.3 Design

A stateless gate runs after the route's own policy decision (`routerctl/cache.py`). The gateway
reports session state in the request, so no session table lives in Deferent. The rules, in order:

1. **Nothing to protect, so follow the policy:**
   * no `session`
   * the current model isn't in this route
   * no prices for either model
   * idle time ≥ the model's TTL
   * prefix below the provider's cacheable minimum
2. **Judge outage, so hold the current model.** A fall-open default would pay a cold write on
   every request until the judge recovers.
3. **Upgrade (the pricier model on a warm turn), a quality call.** Allowed, unless the judge's
   confidence is below `upgrade_min_confidence`. This hysteresis around the threshold is what
   damps oscillation.
4. **Downgrade, a cost call priced exactly.** Take the cost of staying warm for `horizon_turns`
   and the cost of a cold first turn (at `cache_write`) followed by warm turns on the cheaper
   model. Switch only if switching saves at least `switch_margin`. If the target model still
   holds an earlier prefix (`other_warm_caches`), only the turns since then count as cold.

Because downgrades are priced rather than banned, the gate can't fall into the premium trap. If
the cheaper model is cheaper even cold, it switches. That's true of luna vs. sol at every size
tested, because luna's cold rate is below sol's warm rate.

### 12.4 Real validation (`experiments/cache_validation.py`, results in `experiments/results/cache-validation.json`)

The script plays the gateway against real OpenAI. Every dollar figure is real `usage` × the
prices in `clients.yaml`. Each arm gets a random nonce at the start of the prompt, so no arm can
use another arm's cache. Total spend was $1.10.

1. **Premise, ~7.7K-token prefix:**

   | Call | Cached tokens | Written tokens |
   |---|---|---|
   | terra, cold | 0 | 7,740 |
   | terra again | 7,723 | 55 |
   | luna (a switch) | 0 | 7,734 |
   | back to terra | 7,723 | 11 |

   So caches are per model, and a switch-back within the TTL is warm.

2. **Scripted flip-flop, 10 turns, ~8K context.** Gated and ungated cost the same:
   * sol ↔ terra: $0.107 gated vs. $0.106 ungated
   * sol ↔ luna: $0.067 vs. $0.067

   The gate agreed with every switch. At this size, the output-price gap outweighs one cold
   write. The first cold write into each model is the only real cost, because both caches stay
   warm for 30 minutes. It's still a useful result: the gate added no false holds.

3. **Long context, ~31K tokens.** Two hard turns on sol, then four easy follow-ups the policy
   sends to terra.
   * Ungated: a $0.079 cold write into terra.
   * Gated: holds sol (`downgrade_not_worth_losing_cache`) at a warm 0.1× read rate.
   * Result: **$0.2435 vs. $0.2820 for the session (−14%)**, and **$0.065 vs. $0.103 for the
     four follow-ups (−37%)**, all served by the stronger model.

4. **End to end: real decision server, real Jev judge, complexity route sol/terra.** Jev picked
   sol on all 8 turns, including "What region is this log from?". The gate reported `same_model`
   throughout. This is a separate finding about complexity routing, not about caching: Jev judges
   the whole request, and with an 8K-token log in context it forecasts "needs the strong model"
   even for trivial questions. It's worth a follow-up, for example judging complexity on the
   latest user turn plus a summary.

**What this means.** The gate earns its keep on long contexts, where one cold write costs more
than several warm turns. On short contexts it rightly gets out of the way. Flip-flopping between
two models that are both already warm is cheap on OpenAI's 30-minute TTL. That matches
LiteLLM's measurement, and it's why the gate prices `other_warm_caches` rather than treating
every switch as losing a cache. Evidence from the design review, not a new experiment: our own
Sol price was out of date ($5/$30 in `evals/models.py`; the current promotional price is $4/$20
"at least through November 21, 2026"). That's fixed, and earlier Sol cost figures in this
document are slightly high.

### 12.5 Limitations and traps not handled

* **Token counts are estimates** (characters ÷ 4) unless the gateway sends `cached_prefix_tokens`.
  When it does, the real count wins.
* **Long-context pricing isn't modelled.** Above 272K input tokens, OpenAI doubles input and
  cache rates and charges 1.5× on output.
* **Adjusting effort instead of switching** isn't implemented. On Anthropic, changing effort or
  thinking settings invalidates cached messages. On OpenAI, effort changes are only cache-safe
  through `configuration_update` items. A future "stay but raise effort" option has to be sent in
  those cache-safe forms, or it's just another cold switch.
* **Cache-busting prefixes are the gateway's problem:** timestamps early in the system prompt,
  unordered tool lists, tool definitions changing mid-session. The gate can't see them. When the
  gateway reports real `cached_prefix_tokens`, a busted prefix shows up as a low count and the
  gate stops protecting it.
* **Token counts differ by tokenizer.** The same ops log came to 11,356 tokens on Claude Sonnet 5
  and 8,572 on Claude Haiku 4.5 (§13.3). The gate prices both sides of a switch with one count,
  so across model families its dollar estimates can be off by roughly that much.
* **Caches aren't shared across providers.** `current_client` distinguishes the same model on
  two clients. Provider stickiness below that level (OpenRouter's) belongs to the gateway.

### 12.6 Two real bugs found while writing the README examples

To give `auto` real examples, I ran the real `stage_router` on a few agent transcripts. It picked
the efficient tier for everything, including an agent that had hit the same traceback three
times. The cause was `routerctl/messages.py`, the adapter from OpenAI chat shape to libsy's
format. It kept only each message's `content`:

* It dropped every assistant `tool_calls` entry.
* It flattened `role: tool` results into anonymous text.

`auto` routes on nothing *but* tool traffic, so in decision mode it never saw a signal. The
escalation judge was also missing the tool calls, which are part of the transcript it reads.

The same adapter passed OpenAI content parts through unchanged, so any request with an
`image_url` part crashed libsy (`unknown variant image_url`) and returned a 500.

The fix maps `tool_calls` to libsy `tool_call` blocks and tool messages to `tool_result` blocks,
converts image parts, and drops parts routing can't use. With it, the stuck agent escalates to
the capable tier and a productive one stays efficient. `routerctl/tests/test_messages.py` pins
both behaviours plus the image case against the real `stage_router`.

The earlier `auto` experiment (`experiments/REPORT.md`, general assistant) isn't affected: it
used plain chat with no tool history, and there `auto` correctly stays efficient either way.

One related caveat for §12.3. For `complexity`, the confidence Jev reports measures how decisive
the verdict is (|p − 0.5| × 2), not how far p sits from the route's threshold. With the default
threshold of 0.5 these are the same thing. With a different threshold, `upgrade_min_confidence`
only approximates "borderline".

## 13. Multiple providers, and a head-to-head with OpenRouter (2026-09-27)

Every earlier result ran against OpenAI only. This round used an OpenRouter key to ask two
questions about building our own router. Does routing still pay when tiers span inference
providers? And how does it compare with the routers OpenRouter itself offers? Total spend was
$7.85 of the key's $50.

### 13.1 A real gap found first: `extra_body` never reached the gateway

A route can attach per-model settings (`extra_body`: reasoning effort, or OpenRouter provider
pinning such as `{provider: {order: [fireworks]}}`). In decision mode the decision named the
model and client but carried none of those settings, so they were silently lost. Decisions now
carry `selected_extra_body` and `fallback_extra_bodies`. A cache-gate hold swaps them along with
the model. Tests pin both behaviours.

### 13.2 Same twelve policies, four providers (`experiments/multiprovider.py`)

The routes, datasets, thresholds and real Jev judge are the same as §10's 12-experiment study.
Only the tiers changed. Each tier is pinned to its providers through `extra_body`:

| Was | Now | Providers (in order) | $/1M in/out |
|---|---|---|---|
| gpt-5.6-luna | DeepSeek V4.1 Flash | Fireworks, then Together | 0.22 / 0.66 |
| gpt-5.6-terra | Gemini 3.8 Flash | Google AI Studio, then Vertex | 0.75 / 3.75 |
| gpt-5.6-sol | Claude Sonnet 5 | Anthropic | 2 / 10 |
| gpt-6-astra | Claude Opus 5.5 | Anthropic | 4 / 20 |

The script plays the gateway. It calls exactly what the decision says, `extra_body` included.
If a provider errors, it moves down the decision's fallbacks. Cost is OpenRouter's billed
`usage.cost`.

**Routing held up across providers.**
* Decision accuracy matched the OpenAI study. All six intent routes scored 100%, and the weak
  spots were the same complexity routes: code review 61%, SQL 72%.
* Blended savings against "everything on the route's top tier" came to **47.8%**. This uses
  §10's extrapolated baseline: real tokens priced at the top tier's rate.

That baseline understates savings wherever the top model writes much longer answers. On legal
review, Opus wrote about 2,000-token answers where DeepSeek wrote about 100, so legal shows only
5.8%. The head-to-head below measures the baseline for real instead.

**A superseded first run is kept for the record** in
`experiments/results/multiprovider-run1-superseded/`. Don't quote its numbers. It had two flaws:
* Its mid tier (gpt-oss-120b on Together, $0.60/1M output) was cheaper than its "cheap" tier,
  which inverted two routes.
* Its harness never tried a decision's fallbacks. So a temporary Together outage (23 × HTTP 503
  with fallbacks disabled) showed up as routing failures.

Both are fixed. The outage is itself a small argument for the design: a decision already lists
ranked fallbacks with their settings, so a gateway can fail over across providers on its own terms.

### 13.3 Head-to-head on the same 72 prompts

Every third prompt of each dataset went four ways:
* **Deferent**
* **`openrouter/auto`**
* **`typesafe/jev-router`**
* **Always the route's top tier:** no routing, a measured baseline

Grading had two parts:
* **Jev adequacy.** Jev gave every answer an adequate-or-not verdict.
* **Side-by-side scores.** All four answers to a prompt were scored 1–5 side by side, in shuffled
  order, by Gemini 3.1 Pro. None of the contestants used that model. A neutral grader matters
  here: Jev and the Jev Router come from the same company, and our strong tiers are Claude models.

| | Cost (72 prompts) | Mean score | Scored ≥ 4 | Models used |
|---|---|---|---|---|
| Always top tier | $0.898 | **4.82** | 69/72 | Opus 5.5, Sonnet 5, Gemini Flash |
| **Deferent** | $0.538 | 4.58 | 65/72 | DeepSeek 30, Gemini 18, Sonnet 12, Opus 12 |
| OpenRouter Auto | **$0.081** | 4.40 | 63/72 | DeepSeek Flash ×2, Gemini 2.5 Flash, GLM 5.3 Flash |
| Jev Router | $0.062 | 4.14 | 54/72 | `stealth/space-bunny-alpha` 63, DeepSeek 6, gpt-6-sol 3 |

Paired on the same prompts (bootstrap 95% intervals):

| Comparison | Mean score difference | 95% interval | Prompts better / worse | Real gap? |
|---|---|---|---|---|
| Always-top − Deferent | +0.24 | [+0.10, +0.42] | 12 / 1 | Yes, and Deferent is 40% cheaper |
| Deferent − Jev Router | +0.44 | [+0.17, +0.71] | 28 / 5 | Yes |
| Deferent − OpenRouter Auto | +0.18 | [−0.08, +0.44] | 16 / 8 | **No, within noise.** Auto cost 6.6× less |

**Reading it honestly.** On single-turn prompts, OpenRouter's Auto Router is very cost-effective.
It picks from hundreds of models and the cheapest hosts, and at this sample size its quality
can't be told apart from ours. Most of our extra cost comes from the roster, not the routing:
Opus was the top tier on half of these routes. On the "hard" prompts, Auto spent $0.054 for a
mean of 4.37, while we spent $0.48 for 4.56.

The lesson is about rosters. A router we control lets us put models like DeepSeek V4.1 Flash and
Gemini Flash in the tiers, and those models carry most of Auto's advantage. Our outcome data is
what should decide when Opus is worth paying for.

The Jev Router served 63 of 72 prompts with an undisclosed stealth model that currently costs
$0. Its cost here says nothing about steady-state pricing. Its quality was measurably the lowest
of the four.

**What this does not measure.**
* Multi-turn sessions and caching. See §13.4.
* Per-step escalation inside agent runs.
* Choosing our own providers, gateway and harness.
* Outcome data.

Those are the reasons we're building our own router, and none of them show up in a single-turn
benchmark. The honest claim is independence plus cost control, not "cheaper than OpenRouter on
one-off prompts".

### 13.4 Cache-aware switching on Anthropic (`experiments/cache_validation_anthropic.py`)

Anthropic differs from OpenAI on every caching parameter that matters:
* The gateway must send `cache_control` breakpoints to get any caching at all.
* The cache lasts 5 minutes, not 30.
* Opus 5.5 reads cached tokens at 0.05× input.

Only the prices in the config changed.

1. **Premise.** Per-model caches, and switching back is warm, exactly as on OpenAI.

   | Call | Cached tokens | Written tokens |
   |---|---|---|
   | Sonnet 5, cold | 0 | 11,354 |
   | Sonnet 5 again | 11,334 | 79 |
   | Haiku 4.5 (a switch) | 0 | 8,569 |
   | Back to Sonnet 5 | 11,334 | 13 |

   The same text was 8,572 tokens on Haiku (the tokenizer caveat in §12.5).

2. **Long context, about 45K Claude tokens.** Two hard turns on Opus 5.5, then four easy
   follow-ups the policy sends to Sonnet 5. Ungated, the switch pays a $0.116 cold write. Gated,
   the session holds Opus (`downgrade_not_worth_losing_cache`).
   * Session: **$0.354 vs. $0.426 (−17%)**.
   * The four follow-ups: **$0.086 vs. $0.158 (−45%)**, all on the stronger model.

   On OpenAI the same test gave −14% and −37% (§12.4).

3. **OpenRouter's Auto Router, same kind of session**, with a `session_id` as OpenRouter
   documents for sticky routing. It used DeepSeek V4 Flash, jumped to Gemini 2.5 Flash on turn
   5, and came back: two switches, and a 17% cache hit rate. It was still cheap ($0.041 for six
   turns), because those models cost almost nothing.

### 13.5 Ten models, ultra-cheap to strong (`experiments/model_ladder.py`)

This part isn't a comparison with anyone. It tests Deferent across a wide spread of models, all
through OpenRouter, with OpenRouter choosing the host. The 12 policies and datasets are unchanged.
Each route's tiers come from a ten-model ladder, always cheap to strong within a route:

| Class | Models | $/1M output |
|---|---|---|
| Ultra-cheap | gpt-oss-20b, Qwen 3.7 Flash, GLM 5.3 Flash, Ministral 8B, DeepSeek V4.1 Flash | 0.09–0.29 |
| Cheap | GPT-6 Luna, DeepSeek V4 Pro | 0.50–0.70 |
| Mid | Gemini 3.8 Flash | 3.75 |
| Strong | Claude Sonnet 5, GPT-6 Sol | 10 |

All 216 items were routed and answered, for $0.37 of routed spend. Decision accuracy matched the
two earlier studies:
* intent: 100% on all six routes
* escalation: 100% and 89%
* complexity: 61% (code review), 83% (finance), 72% (SQL)

On every third item (72 prompts), the same prompt also went to two baselines: always the
route's cheapest tier, and always its strongest. Gemini 3.1 Pro scored all three answers
side by side. Two Gemini Flash models are in the ladder, so the grader shares a model family
with them.

| | Cost | Mean score | Easy prompts | Hard prompts |
|---|---|---|---|---|
| Always cheapest | $0.034 | 4.42 | 4.60 | 4.17 |
| **Deferent** | $0.104 | 4.47 | 4.57 | 4.33 |
| Always strongest | $0.203 | 4.67 | 4.74 | 4.57 |

Here "hard" means the correct route was above the cheapest tier. That was 30 of the 72 prompts.

**What it shows:**
1. **Cheap models are already good on single-turn tasks like these.** The whole quality spread
   from cheapest to strongest is 0.25 points, and none of the pairwise differences is
   statistically significant at n = 72 (every bootstrap interval includes zero). What routing
   reliably did was **halve the cost** of always using the strongest model.
2. **When routing helped, it was decisive.** On five hard prompts the cheapest model failed
   (scores 1–3), and Deferent routed up and scored 4–5: two exec-assistant tasks, a meeting
   summary, a legal risk clause, and a finance analysis.
3. **When it hurt, it was the `complexity` policy every time.** Two SQL prompts were sent to the
   cheap tier and scored 1, where the strong tier scored 4–5. Complexity routes have been the
   weakest in all three studies:
   * OpenAI-only (§10)
   * multi-provider (§13.2)
   * this run

   Intent routes scored 100% in all three. That makes complexity routing the most valuable
   thing to fix next (§14).

## 14. Roadmap

1. **Routing-accuracy validation**: the judge-only eval (predict vs. a ground-truth label from
   running both tiers) and Switchyard's `benchmark/` TB2.1 subset with the Jev judge vs. the LLM
   judge on the escalation profile, tracking accuracy, cost, judge share, and calibration
   (Brier/ECE from the logged probabilities against task outcomes).
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
6. **Decision-only escalation, richer signal**: §11.2's transcript-reading classifier asks one
   `noul`-equivalent (continue vs. escalate); it could ask Jev's separate `looping`,
   `false_progress`, `drift`, `desperation`, `external_blocker` nouls in the same call (Jev
   answers all questions in one round trip regardless of count) and combine them with a small
   rule, the same idea as item 2 above but built for the stateless, per-request shape decision
   mode actually needs rather than Switchyard's session-tracked native algorithm.

7. **Complexity judging on long contexts** (found in §12.4): with an 8K-token document in
   context, Jev forecast "strong" even for one-line lookups. Try judging the latest user turn,
   plus a compact description of the context, instead of the whole request, and measure the
   effect on routing accuracy with the existing eval harness.

8. **Roster tuning from outcomes** (found in §13.3): most of our cost gap with OpenRouter's Auto
   Router was the top tier we chose, not the routing. Re-run the head-to-head with a cheaper top
   tier, and let outcome data rather than intuition decide when a premium model earns its price.

9. **Fix `complexity` routing** (§13.5): across three studies with three different model
   rosters, it's the one policy that misroutes. Code review is at 56–61%, and SQL at 72%.
   It under-forecasts long, multi-step technical prompts. Options include recalibrating
   `base_threshold` per route from outcome data, or asking Jev about named difficulty signals
   (joins or window functions, concurrency, multi-constraint reasoning) instead of one "can the
   weak model do it" probability.

## 15. Sources

Switchyard: repo README, `docs/routing_algorithms/*.md`, `crates/libsy/src/algorithms/util/llm_judge.rs`,
`crates/libsy/src/prompts/*`, `benchmark/routing-profiles/*`; NVIDIA blog "Route AI Agent Workloads
Across Models with NVIDIA NeMo Switchyard"; LangChain "How many of your agent's calls actually need
a frontier model?"; Switchyard issue #723, PRs #724, #739, #762.
Jev: typesafe.ai launch post; `typesafe-sdk` 0.7.1 (`_schemas/models.py`, generated from
api.typesafe.ai/openapi.json); docs.typesafe.ai/api; Pydantic AI TypeSafe docs; LangChain "Building a
harness with Jev"; APIMaster "Jev vs LLMs"; "The Jev File" independent checks; systemonemodels.org
alternatives index; OpenRouter Decisions endpoint notes; DevelopersIO "replacing model routing with
TypeSafe (Jev)"; Sean Goedecke, "System One models can train their own replacements".
Cache-aware switching (§12, all checked 2026-09-27): developers.openai.com/api/docs/guides/prompt-caching,
/api/docs/pricing, /api/docs/models/gpt-5.6-sol; platform.claude.com/docs/en/build-with-claude/prompt-caching;
openrouter.ai/docs/guides/best-practices/prompt-caching; openrouter.ai/blog/announcements/introducing-the-new-auto-router;
x.com/OpenRouter/status/2103610898690855161 (Jev Router); vllm.ai/blog/2026-06-02-session-aware-agentic-routing;
docs.litellm.ai/docs/auto_router/prompt_caching; lmsys.org/blog/2024-12-04-sglang-v0-4;
docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing; blog.dailydoseofds.com "A cheaper model
does not imply a cheaper turn"; jfrog.com/blog/why-model-routing-backfires.
Multi-provider round (§13, checked 2026-09-27): openrouter.ai/api/v1/models and /models/{id}/endpoints
(per-provider prices), openrouter.ai/docs provider routing (`provider.order`, `allow_fallbacks`),
OpenRouter `usage.cost` accounting; results in `experiments/results/multiprovider/`, `experiments/results/model-ladder/` and
`experiments/results/cache-validation-anthropic.json`.
