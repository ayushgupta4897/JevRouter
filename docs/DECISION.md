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

## 11. Roadmap

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

## 12. Sources

Switchyard: repo README, `docs/routing_algorithms/*.md`, `crates/libsy/src/algorithms/util/llm_judge.rs`,
`crates/libsy/src/prompts/*`, `benchmark/routing-profiles/*`; NVIDIA blog "Route AI Agent Workloads
Across Models with NVIDIA NeMo Switchyard"; LangChain "How many of your agent's calls actually need
a frontier model?"; Switchyard issue #723, PRs #724, #739, #762.
Jev: typesafe.ai launch post; `typesafe-sdk` 0.7.1 (`_schemas/models.py`, generated from
api.typesafe.ai/openapi.json); docs.typesafe.ai/api; Pydantic AI TypeSafe docs; LangChain "Building a
harness with Jev"; APIMaster "Jev vs LLMs"; "The Jev File" independent checks; systemonemodels.org
alternatives index; OpenRouter Decisions endpoint notes; DevelopersIO "replacing model routing with
TypeSafe (Jev)"; Sean Goedecke, "System One models can train their own replacements".
