# Evals: how to decide which model is better, at what, and for how much

The direct answer to "how do I know this model is better than that one": you don't need one
number, you need three, kept separate, because they answer different questions and a router
needs all three to make a good decision.

| Question | Answer comes from | Cost to get it |
|---|---|---|
| Is this model *capable* at domain X at all? | **Public benchmarks** | Free — read a leaderboard |
| Is this model capable at *my* domain, on *my* kind of request? | **A small, concise, own eval set** | Cheap — a few dollars per run |
| Is this model fast, cheap, and up right now? | **Operational metrics from real or synthetic traffic** | Nearly free — no ground truth needed |

Conflating these is the usual mistake: re-running an expensive public benchmark yourself
(wasteful — someone already spent the compute and published the number), or trying to establish
absolute model quality from twenty of your own prompts (noisy — you're measuring your sample,
not the model). Keep the layers separate and each one gets cheap.

## Layer 1: public benchmarks, cited not rerun

For "is GPT-6 Astra generally better at coding than GPT-5.6 Sol" — don't rerun SWE-bench
yourself. It's expensive, someone else already ran it properly, and the number is published and
tracked. Pull from:

| Domain | Benchmark | Why this one |
|---|---|---|
| Coding / agentic | Terminal-Bench 2.1, SWE-bench Verified | Switchyard's own `benchmark/` already wires up Terminal-Bench 2.1 with real routing profiles — reuse it, don't rebuild it |
| Tool use / function calling | Berkeley Function Calling Leaderboard (BFCL) | What LangChain used in the Switchyard escalation benchmark this project's docs cite |
| General reasoning | MMLU-Pro, GPQA-Diamond | Broad, hard, still discriminates between frontier models |
| Multi-turn agents | τ²-bench | Realistic tool-use conversations, not single-turn Q&A |

Update this table when a new model family launches by checking each provider's own published
numbers and one independent tracker (e.g. the leaderboard aggregators cited throughout
`docs/DECISION.md`) against each other — never just the vendor's own claim alone.

## Layer 2: a small, concise, own eval set

Public benchmarks tell you about the *published* task distribution, which is never exactly your
traffic. `evals/suite.yaml` in this repo is the concrete answer to "I don't need it extensive,
just something I can decide on": **24 prompts**, 4 domains (coding, extraction, finance,
general/reasoning) × 3 difficulty tiers × 2 each. Small on purpose:

- Cheap enough to run against every candidate model on every meaningful price change or new
  release — dollars, not tens of dollars, per full run (see the worked budget below).
- Graded automatically, so a rerun costs no human time: exact-match/regex where the answer is a
  fact (`contains_all`, `numeric_equals`), Jev-as-judge for open-ended correctness (`jev_noul`,
  a yes/no rubric) — reusing this repo's own sidecar, so grading itself is nearly free.
- Small enough that when a candidate model fails a case, you can *read the case and its answer*,
  not just stare at an aggregate number. A calibration set should be legible, not just large.

Run it: `python evals/run_eval.py --models gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol,gpt-6-astra`
(see `evals/run_eval.py --help` for the judge URL, budget cap, and output flags). It produces a
model card: accuracy overall and per domain, p50/p95 latency, error rate, and total cost.

**This is genuinely concise, and that's a real limitation, not just a virtue.** Twenty-four
prompts tell you a *direction*, not a certified accuracy figure — treat a result like "Sol beat
Terra 75% to 58% on finance" as "worth routing finance to Sol," not as a number with real
statistical confidence. If a domain matters enough to your business that the exact percentage
matters, that domain earns its own larger, still-domain-specific set (50-100 cases), not a
blanket expansion of this one.

**Extend it by adding a case, not by rewriting the harness.** Every case is one YAML entry with
a prompt, a domain/difficulty tag, and a grader. Add your own domain (support tickets, legal
clauses, whatever your teams actually route) the same way `voice-ai`'s config extends
`teams/clients.yaml` — this is deliberately the same "small YAML, same shape as the examples"
ergonomic as the router config itself.

### Judge-only eval: isolating the router's own decision, cheaper still

Everything above measures *answer* quality. A separate, even cheaper question is *routing*
quality: given a task, does the judge (Jev, or an LLM judge) correctly predict which tier should
handle it? This was worked through in this project's earlier design discussion and is the right
tool once you have a specific route's judge to validate:

1. Run a task set once on the cheap tier and once on the strong tier; label each task by
   whether the cheap tier actually succeeded (ground truth, from the real outcome — for coding,
   an automatic test; for others, `evals/suite.yaml`'s own graders).
2. Ask the judge to predict, without running anything, whether the cheap tier will succeed.
3. Score the judge against the labels: accuracy at your threshold, calibration (when it says
   0.8, does the cheap tier actually succeed ~80% of the time), and the achievable cost/accuracy
   curve across every threshold, all computed from the one dataset in step 1.

This costs only judge calls (Jev: a fraction of a cent each), so it's cheap to run per route,
per model pairing, whenever you tune `base_threshold` or swap a tier's model.

## Layer 3: operational metrics — no ground truth needed

Cost, latency, and availability don't need an eval at all — they need measurement:

- **Cost**: exact, from the API's own reported token usage times the pricing table
  (`evals/models.py` — keep it current; OpenAI cut GPT-5.6 Sol's price over 20% within weeks of
  launch, and this table needs the same vigilance).
- **Latency**: p50/p95 from real traffic (Switchyard's `/v1/stats` and Prometheus `/metrics`
  already report this per target) or from the same eval harness's `--output` JSON.
- **Availability / error rate**: track 5xx/timeout rate per target over time; Switchyard's
  routing log (`--routing-log-file`) already has per-call outcomes to build this from.

None of this needs a "which model is smarter" judgment, which is exactly why it's the cheapest
layer to run continuously rather than only when deciding between models.

## Feeding this back into router configs

The output every layer above should converge on is a **model card**: per model, per domain,
accuracy / cost / p50-p95 latency / error rate, refreshed on a cadence (weekly, or on every
price or model-family change). A team writing `teams/<team>.yaml` (see `docs/TEAM_CONFIG.md`)
should be able to look up that card when choosing `weak`/`strong`/bucket models, instead of
guessing. This repo doesn't yet auto-generate that card as a committed artifact — `evals/run_eval.py --output` gives you the raw JSON to build one from — but that's the shape the next
iteration should take (see `docs/DECISION.md`'s roadmap): a `model-cards.json` the platform
regenerates on a schedule, and eventually something Switchyard's `auto` preset itself could read
to pick sane per-category defaults instead of a single fixed pair.

## Worked budget, so "concise" isn't just a word

24 cases × 4 models (`gpt-5.6-luna/terra/sol`, `gpt-6-astra`) with `--max-tokens 600`, worst case
every call maxes out output: `evals/models.py`'s real pricing puts the most expensive model
(Astra, $10/$50 per M) at roughly $0.15 for the whole 24-case run; the cheapest under $0.01. Full
4-model run: comfortably under $1, leaving the rest of a $10 budget for the judge-only eval or a
second full pass after a config change. The harness enforces a hard `--max-budget-usd` stop
regardless, checked before every call, so a pricing surprise can't blow past it.

**This was run for real**, not just estimated: `gpt-5.6-luna/terra/sol` and `gpt-6-astra`, full
24-case runs each, actual total spend ~$0.72. `gpt-5.6-terra` and `gpt-6-astra` both scored 100%;
`gpt-5.6-luna` and `gpt-5.6-sol` both scored 96% (one shared, genuinely-disputed nuanced-judgment
case). Seven real bugs in the harness and suite content were found and fixed along the way —
full account in `docs/DECISION.md` §10.1, results table in the main `README.md`.
