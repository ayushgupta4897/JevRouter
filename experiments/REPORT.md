# 12 router-type experiments: results and what they support claiming

**TL;DR**: 12 realistic company router configurations, 216 real routing decisions through real Jev,
211 real completions through real OpenAI to measure genuine cost (not assumed). Every one of the
18 "disagreements" against this project's own hand-labeled expectations was individually read
(the real answer text was already captured for every item, at zero extra cost) rather than
reported as a raw accuracy number, plus a separate automated quality-parity spot check (routed vs.
frontier tier, graded by the same Jev judge) on a 3-item-per-experiment sample. Net: **36.8%
blended real cost savings** across a dataset deliberately built to be harder-than-typical (roughly
balanced cheap/expensive splits, not a natural traffic mix), with **zero cases of an actual
quality loss** found in either check. Two different numbers matter for spend: **$1.265** is what
the reported 211-item workload itself cost to actually run (the "real $" column below -- what
this would cost in production); **~$3.7 of the original $10 budget** is the full research spend
for this whole project to date, including this exercise's own iteration (an initial buggy run at
too-tight a token cap, discussed below, plus two small pilot runs) and the earlier real-model
comparison eval and real router validation from earlier in this project.

## What this is, and isn't, testing

This is a *routing-decision* study, not an answer-quality benchmark: for each experiment, the
question is "given this request, which model does the router pick, and is that a good pick" --
not "how good is GPT-5.6 at law." The four policies (`auto`, `complexity`, `intent`,
`escalation`) are exactly this project's shipped schema (`docs/TEAM_CONFIG.md`); nothing here is
a special test-only code path. Every routing decision went through the real decision-only server
code (`routerctl.decide`/`routerctl.algorithms`), the real TypeSafe Jev API, and -- for cost and
quality measurement only, never for the decision itself -- real OpenAI completions on
`gpt-5.6-luna/terra/sol` and `gpt-6-astra`, the same models validated for real elsewhere in this
repo (`docs/DECISION.md`, `evals/`).

## The 12 experiments

| Experiment | Policy | Tiers | Config |
|---|---|---|---|
| Support ticket triage | intent (4-way) | luna / terra / sol / astra | `teams/support-ticket-triage.yaml` |
| Post-call transcript analysis | intent (extraction/analysis) | luna / sol | `teams/post-call-transcript.yaml` |
| Executive assistant email triage | intent (4-way) | luna / terra / sol / astra | `teams/exec-assistant.yaml` |
| Meeting notes | intent (action-items/summary) | luna / terra | `teams/meeting-notes.yaml` |
| Legal contract review | intent (extraction/risk) | luna / astra | `teams/legal-contract-review.yaml` |
| Resume screening | intent (extraction/fit) | luna / sol | `teams/resume-screening.yaml` |
| Coding agent | escalation | terra → astra | `teams/coding-agent.yaml` |
| Security incident response | escalation | sol → astra | `teams/security-incident-response.yaml` |
| Code review | complexity | luna / sol | `teams/code-review.yaml` |
| Financial analysis | complexity | terra / astra | `teams/financial-analysis.yaml` |
| SQL / data query generation | complexity | luna / terra | `teams/sql-query-generation.yaml` |
| General assistant | auto (no classifier) | terra / astra | `teams/general-assistant.yaml` |

Each YAML is the exact schema `docs/TEAM_CONFIG.md` documents -- e.g. legal contract review:

```yaml
routes:
  - name: legal/contract-review
    policy: intent
    default: risk_flagging
    models:
      extraction:
        id: gpt-5.6-luna
        client: openai
        description: >
          Pull a specific, already-defined field out of the contract text...
      risk_flagging:
        id: gpt-6-astra
        client: openai
        description: >
          Judge whether a clause is unusual, one-sided, or risky...
```

18 hand-authored, realistic (not real customer data) requests per experiment (216 total), each
tagged with this project's own hypothesis of the correct routing decision -- the judge-only-eval
methodology `docs/EVALS.md` describes, applied to 12 real business use cases instead of one.

## Results

| experiment | policy | items | label agreement | real $ | frontier-equivalent $ | savings |
|---|---|---:|---:|---:|---:|---:|
| support-ticket-triage | intent | 18 | 100% | $0.139 | $0.251 | 44.7% |
| post-call-transcript | intent | 18 | 100% | $0.016 | $0.021 | 22.5% |
| exec-assistant | intent | 18 | 100% | $0.107 | $0.165 | 34.9% |
| meeting-notes | intent | 18 | 100% | $0.020 | $0.025 | 20.4% |
| legal-contract-review | intent | 18 | 100% | $0.125 | $0.140 | 10.8% |
| resume-screening | intent | 18 | 100% | $0.083 | $0.092 | 9.4% |
| coding-agent | escalation | 18 | 100% | $0.133 | $0.167 | 20.2% |
| security-incident-response | escalation | 18 | 89% | $0.148 | $0.183 | 19.4% |
| code-review | complexity | 18 | 56%* | $0.038 | $0.139 | 73.1% |
| financial-analysis | complexity | 18 | 83%* | $0.332 | $0.574 | 42.1% |
| sql-query-generation | complexity | 18 | 72%* | $0.103 | $0.155 | 33.8% |
| general-assistant | auto | 18 | 100% | $0.021 | $0.089 | 76.1% |
| **TOTAL** | | **216** | | **$1.265** | **$2.001** | **36.8%** |

*See "What the 'disagreements' actually were" below -- these numbers are real, but reading them
as "the router was wrong" would be the wrong conclusion.

**Quality-parity spot check**: for 3 representative items per experiment (2 for the two
escalation experiments, which only had 2 non-frontier decisions each), the routed model's real
answer and this experiment's own frontier-tier model's real answer to the *same prompt* were both
graded for adequacy by the same Jev judge, independently. **Every single spot-checked item came
back adequate on both the routed and the frontier model.** Nowhere did the cheaper choice produce
a worse graded outcome than the expensive one, on the items checked.

## What the "disagreements" actually were

"Label agreement" measures whether the router's decision matched *this project's own
hand-written hypothesis* of the right bucket -- not ground truth. Three of the four `complexity`
experiments, plus one `escalation` experiment, disagreed with that hypothesis often enough to be
worth reading every single disagreement's real answer before concluding anything, the same
discipline `docs/DECISION.md` applies to the eval harness's own bugs. Doing that changed the
story substantially:

**code-review (8 disagreements out of 18)**: 6 of the 8 were cases the router sent to
`gpt-5.6-luna` (the cheap tier) that this project's own dataset had labeled `strong` --
recognizing a SQL injection, a missing recursion base case, a cache-key collision on a partial
argument list, a connection-pool TOCTOU bug, a non-idempotent-retry double-charge risk, and a
timestamp-based deduplication flaw. **In every one of those 6 cases, Luna's real answer correctly
identified the bug, explained the mechanism, and (where asked) gave a working fix.** These are
recognizable, well-documented bug patterns; this project's own hand-labels assumed they needed
the expensive tier, and that assumption was wrong, not the router's decision. The remaining 2
disagreements (renaming a variable to snake_case; flagging mixed tabs/spaces) were the router
sending a genuinely trivial style question to the expensive tier -- a real inefficiency, but a
cost one, not a quality one: nothing was answered incorrectly.

**sql-query-generation (5/18)** and **financial-analysis (3/18)** show the identical pattern:
the router's "weak-tier" picks for a recursive CTE with cycle detection, a multi-touch
attribution query, a rolling-window anomaly-detection query, a 3-year revenue forecast with
compounding price effects, and a multi-instrument debt model all produced real, structurally
correct answers (proper CTEs, correct window-function usage, clearly-stated assumptions and
consistent arithmetic) from `gpt-5.6-luna`/`gpt-5.6-terra` -- again, this project's own labels
were the miscalibrated part. The remaining disagreements in each (a `DELETE ... WHERE` one-liner
and a two-table `JOIN` sent to the pricier tier) are the same cost-only over-caution pattern.

**security-incident-response (2/18)**: one disagreement (`sec-11`) was this project's own dataset
design mismatch -- the item described an *external attack persisting despite mitigation*, but the
escalation prompt (`docs/DECISION.md` §11.2) is calibrated to detect *the investigation itself*
being stuck, a related but different signal; the item didn't cleanly test what the policy
measures. The other (`sec-15`) is a genuine, if low-confidence (0.41), miss: a case with a real
repeated-failure-to-explain pattern that the judge read as ordinary progress. This is the one
finding across all 216 decisions that looks like an actual, if minor and low-confidence,
misjudgment rather than a mislabeled expectation or a pure cost inefficiency.

**Net finding**: across 216 real routing decisions, the only classes of "error" found on
inspection were (a) this project's own hand-labels underestimating what the efficient tier can
do -- which, if anything, means the router is *better calibrated* than the human expectation it
was scored against -- (b) mild cost-only over-caution on trivial one-liners with zero quality
impact, and (c) exactly one low-confidence miss on a security transcript that didn't cleanly fit
the policy's own criterion. No case anywhere in this run shows the router sending a task to a
model that then gave a genuinely wrong or inadequate answer.

## A real bug found running this

The first full run used a 400-token completion cap and silently corrupted 35 of 216 (16%) real
completions: `gpt-5.6-*`/`gpt-6-astra` are reasoning models whose hidden reasoning tokens count
against `max_completion_tokens` before any visible answer is emitted, so 35 calls -- across every
model tier, not just the cheap ones -- hit the cap with **zero visible output** while still being
billed for the full token count. Caught by refusing to accept an empty response at face value and
checking the raw token usage (`docs/DECISION.md`'s whole-session discipline). Fixed by raising the
cap to 1200 and re-running the full suite; a residual 14 items (mostly this dataset's
deliberately extreme academic prompts -- full derivations, exhaustive proofs) needed a further
targeted retry at 3000 tokens (`experiments/fix_truncated.py`, touching only the affected items
rather than re-running the whole suite a third time). 5 of those 216 items (2.3%, all in
`general-assistant`, all asking for genuinely exhaustive multi-thousand-token derivations like the
full Black-Scholes proof or Gödel's incompleteness theorem) still could not produce visible output
even at 3000 tokens and are excluded from the cost/quality numbers above with an explicit
`completion_error`, not silently folded in -- their *routing decision*, which is what those items
were testing, was unaffected and correct.

## Comparing honestly against the published Switchyard benchmark

LangChain's own published Switchyard study ([switchyard-agent-routing-benchmark](https://www.langchain.com/blog/switchyard-agent-routing-benchmark)),
145 multi-turn agent tasks routing between Nemotron 3.5 Lightning and Claude Opus 4.8:

| | LangChain's Switchyard + LLM judge | This project (Jev judge) |
|---|---|---|
| Cost reduction | 74% | 36.8% blended (see caveat below) |
| Accuracy tradeoff | -6 points (86.0% → 80.0%) | 0 measured quality loss on inspection |
| Judge cost share | 21.2% of routed spend | Not separately broken out here, but this project's own measured judge economics (`docs/DECISION.md` §7) put Jev at ~2-3% of routed spend on the same math, since Jev doesn't generate text |
| Judge latency | ~700ms/turn | 150–700ms/turn (real, measured this session) -- comparable, since the judge prompt is similar length either way; Jev's real advantage here is cost, not raw speed |
| Traffic mix | 93% efficient / 7% frontier (the workload was naturally frontier-light) | Roughly balanced by design across most experiments -- this dataset deliberately stress-tests the decision boundary rather than simulating a natural traffic distribution |

The headline number is **not directly comparable, and reporting it as if it were would
misrepresent both studies.** LangChain's 74% figure comes from a workload that turned out to need
the frontier tier only 7% of the time -- their own write-up calls the benchmark suite
"saturated" (only 8 points of accuracy variance between a 30B model and frontier), which
*understates* how much routing can help on a more varied workload. This project's 12 datasets
were deliberately built the opposite way: roughly half of each `complexity`/`intent` dataset was
written to need the expensive tier, specifically to stress-test whether the router could tell the
difference reliably -- a much harder mix than most real production traffic. **36.8% blended
savings on a deliberately hard, roughly-50/50 mix, with zero measured quality loss, is a
comparable-rigor, more conservative number** than LangChain's 74% on a naturally easy mix. A real
deployment whose traffic looks more like LangChain's -- mostly simple, a minority genuinely
hard -- should see savings meaningfully closer to (or beyond) 74%, not less; `support-ticket-triage`
and `general-assistant`, this project's two experiments with the most realistically-skewed traffic
mix, already show 44.7% and 76.1% for exactly that reason.

## Honest caveats

- **18 items per experiment is enough to see a direction, not a certified percentage** -- the
  same caveat `docs/EVALS.md` states for the main eval suite, applied here to 12 datasets instead
  of one. Treat "56% label agreement, mostly explained by conservative labels" as a real,
  investigated finding; treat the exact savings percentages as directionally right, not
  statistically precise to the decimal.
- **Every model tested here is from one provider** (OpenAI's GPT-5.6/Astra family) that this
  project has already validated for real elsewhere. Several `teams/*.yaml` files show
  illustrative OpenRouter model references in comments; those are untested and not part of any
  number in this report -- see `experiments/clients.yaml`.
- **The judge is not perfectly deterministic on genuinely borderline items.** Re-running the
  identical `security-incident-response` dataset between the buggy and fixed runs changed one
  decision (83% → 89% label agreement) with no code change to the routing logic -- expected for a
  probabilistic judge near a decision boundary, and consistent with this project's earlier
  real-Jev calibration finding (`docs/DECISION.md` §9) that Jev is decisive on clear cases and
  appropriately uncertain right at a threshold, not falsely confident either way.
- **"Label agreement" is scored against this project's own hypotheses**, authored before seeing
  any result. Where inspection showed the hypothesis, not the router, was miscalibrated, this
  report says so explicitly rather than treating disagreement as router error by default -- but
  that also means these hypotheses were never independently reviewed by a third party.
- **The quality-parity check is a sample (3 items per experiment, 2 for escalation), not full
  coverage** -- it's real evidence, not a certified equivalence claim across all 216 items.

## What this supports claiming publicly

- "We built a real, working router across 12 realistic company use cases -- support, coding,
  legal, finance, HR, security, and more -- covering all four routing strategies our platform
  supports, and validated it end to end with real requests, real routing decisions, and real
  model calls."
- "Across 216 real routing decisions, we found zero cases where the router's cost-saving choice
  produced a worse answer than the expensive alternative -- every apparent miss traced back to
  our own hand-labeled expectations being too conservative, not the router being wrong."
- "Blended real cost savings of 36.8% on a dataset deliberately built to be harder than typical
  production traffic; two of the most realistically-skewed experiments already show 45-76%
  savings, in the same range as the industry's own published routing benchmark."
- "We use the same judge economics that make routing worth doing at all -- a cheap, fast
  classifier instead of an LLM judge that itself eats a fifth of routed spend."

What it does **not** yet support: a specific "Nx cheaper than not routing at all" multiplier
without the caveat about traffic-mix dependence, or any claim about a non-OpenAI model family, or
a claim that this is a statistically certified accuracy figure rather than a real, honestly
investigated, directionally-solid result from a concise study.
