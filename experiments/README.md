# Router-type experiments

Twelve realistic router configurations, one per common company use case, each with its own
hand-authored dataset, run for real through this repo's decision-only router
(`routerctl serve`'s default mode) and real Jev, with real completions used only to measure
genuine cost and quality -- not simulated. **See [`REPORT.md`](REPORT.md) for the full write-up
and results** (216 real routing decisions, 36.8% blended real cost savings, zero quality loss
found on inspecting every disagreement, and an honest comparison against the published
Switchyard/LangChain benchmark); this file covers what's here and how to reproduce it.

## Layout

| Path | What |
|---|---|
| `clients.yaml` | The real `openai` client plus an illustrative, untested `openrouter` entry -- see its own comments |
| `teams/*.yaml` | One real `routerctl` config per experiment, using the exact same schema as `docs/TEAM_CONFIG.md` |
| `datasets/*.yaml` | 18 hand-authored, realistic (not real customer data) requests per experiment, each tagged with this project's own hypothesis of the correct routing decision |
| `run_experiments.py` | Runs every experiment: real routing decisions via real Jev, real completions via real OpenAI to measure genuine cost/latency, and a 3-item quality-parity spot check per experiment (routed model vs. that experiment's own frontier tier, graded by the same Jev judge) |
| `fix_truncated.py` | One-off patch used after the first run: `gpt-5.6-*`/`gpt-6-astra` are reasoning models whose hidden reasoning tokens count against the completion cap, so a handful of the hardest prompts came back with zero visible output at 400/1200 tokens; this re-calls only those specific items at a higher cap rather than re-running everything |
| `cache_validation.py` | Real-OpenAI validation of cache-aware switching (see `docs/DECISION.md` §12.4). It plays the AI Gateway, feeds real `usage` back as `session`, and compares gated and ungated sessions on real cost. Results are in `results/cache-validation.json` |
| `multiprovider.py` | The same 12 policies with tiers across Fireworks, Together, Google and Anthropic (pinned through `extra_body`), plus a head-to-head against OpenRouter's Auto and Jev routers and an always-top-tier baseline, scored by a neutral grader. See `docs/DECISION.md` §13. Results are in `results/multiprovider/` |
| `model_ladder.py` | The 12 policies over ten OpenRouter models, from ultra-cheap to strong, with always-cheapest and always-strongest baselines on the same prompts. See `docs/DECISION.md` §13.5. Results are in `results/model-ladder/` |
| `auto_vs_escalation.py` | `auto` and `escalation` on the same six real transcripts. It exposed and re-validated the escalation blind spot (`docs/DECISION.md` §15). Results are in `results/auto-vs-escalation.json` |
| `complexity_eval.py` | Complexity-routing decision accuracy, before and after the rubric fix, on the original 54 items plus 48 held-out ones (`datasets/complexity-holdout.yaml`). Real Jev, pennies. See `docs/DECISION.md` §14 |
| `cache_validation_anthropic.py` | The cache-gate validation repeated on Claude models through OpenRouter. Results are in `results/cache-validation-anthropic.json` |
| `summarize.py` | Rebuilds the summary table straight from `results/*.json`, no re-run needed -- what `REPORT.md`'s numbers were generated from |
| `results/*.json` | Raw per-item results from the last run: decision, cost, latency, and the actual response text, so any number in the report is auditable back to source |
| `REPORT.md` | The write-up: methodology, per-experiment results, an honest read of every routing "disagreement" (not just the raw percentage), aggregate cost-savings, and a comparison against the published Switchyard/LangChain benchmark |

## Why this uses only the OpenAI family

Every model referenced for real here (`gpt-5.6-luna/terra/sol`, `gpt-6-astra`) is one already
validated end to end elsewhere in this repo, with real, verified pricing (`evals/models.py`) and
real measured behavior (`docs/DECISION.md`). This project's design goal is genuinely
provider-agnostic -- the decision-only router never dials the target model at all (see
`docs/DECISION.md`'s decision-only section), so adding an OpenRouter-listed model from any
provider costs nothing but a YAML edit and a real `OPENROUTER_API_KEY`. But claiming a specific
model's price or quality without having actually tested it would be exactly the kind of unverified
claim this project has avoided everywhere else (see the real-bug-hunting discipline in
`docs/DECISION.md`), so the illustrative OpenRouter references in a few `teams/*.yaml` comments
are marked as such, not presented as tested results.

## Reproducing

```bash
source .venv/bin/activate
routerctl validate experiments/teams --clients experiments/clients.yaml   # schema + dry-run
python experiments/run_experiments.py --max-budget-usd 5.00              # real Jev + real OpenAI
python experiments/run_experiments.py --only legal-contract-review       # a single experiment
```

Needs `TYPESAFE_API_KEY` (real Jev) and `OPENAI_API_KEY` in the environment (`.env`, matching the
rest of this repo). The hard `--max-budget-usd` guard stops the run before exceeding it, checked
before each completion call; routing decisions themselves cost only a Jev judge call regardless
(a fraction of a cent), since deciding never calls the target model.
