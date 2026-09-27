# Agent guide for this repo

Deferent is a **decision-only** model router. It returns which model should serve a request and
never calls that model. It's built on vendored NVIDIA NeMo Switchyard, with TypeSafe's Jev as the
judge. The internal packages are `routerctl` (the router) and `jevjudge` (the Jev adapter).

**Start with [`docs/HANDOVER.md`](docs/HANDOVER.md).** It covers current state, every
experiment, every bug fixed, performance, what we learned, and next steps.
[`docs/DECISION.md`](docs/DECISION.md) is the full design record, cited as §N everywhere.

## Commands

```bash
source .venv/bin/activate                              # after scripts/build.sh
pytest routerctl/tests jevjudge/tests evals/tests -q   # offline, ~5 s: run before every commit
routerctl validate teams/                              # schema + real switchyard-server --dry-run
routerctl validate experiments/teams/ --clients experiments/clients.yaml
scripts/e2e_routerctl_decide.sh                        # end-to-end, decision mode
scripts/e2e_routerctl.sh                               # end-to-end, proxy mode
```

## Rules

- **Decision-only is the invariant.** Nothing in `routerctl` may call a target model.
  `decide.UnexpectedRealCallError` guards this.
- **Keys** (`TYPESAFE_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) live only in the
  git-ignored `.env`.
  - Load them with `set -a; . ./.env; set +a`.
  - Never print them or commit them. Check `git diff --cached` before committing.
- **Real API spend** goes through experiment scripts with `--max-budget-usd`. Report the actual
  spend.
- **Every bug fix gets a regression test.** Every experiment gets:
  - a script
  - a results JSON under `experiments/results/`
  - a DECISION.md section with method, numbers and caveats

  Archive flawed runs as superseded; don't delete them.
- **Report honestly.** Say when differences aren't statistically significant, and say what a
  benchmark doesn't measure. Don't claim Deferent is cheaper than OpenRouter's routers on one-shot
  prompts; it isn't (§13.3).
- **Changing behaviour means updating the docs too.** Update the README if the pitch or examples
  change, `docs/TEAM_CONFIG.md` if the schema changes, and `docs/HANDOVER.md` if the project's
  state changes.
- **Public name:** Deferent. Don't rename internal packages without a reason.
