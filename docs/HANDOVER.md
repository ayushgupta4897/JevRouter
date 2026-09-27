# Handover: Deferent, state of the project

**Written 2026-09-27** for whoever picks this up next, human or agent. Read this first. It says
what exists, what has been measured, what broke and how it was fixed, what we learned, and what
to do next.

After this file, read in this order:
* [`README.md`](../README.md): the product pitch, with examples
* [`docs/TEAM_CONFIG.md`](TEAM_CONFIG.md): the YAML schema
* [`docs/DECISION.md`](DECISION.md): the full design record, with every experiment's method and
  caveats

---

## 1. What this is, in one paragraph

Deferent is a **decision-only model router**. It receives a request shaped like an OpenAI
chat-completions call and returns *which* model and client should serve it. It never calls that
model itself; the caller's AI gateway does. Deferent is built on NVIDIA NeMo Switchyard (vendored,
unmodified, under `vendor/switchyard/`) and uses TypeSafe's **Jev** as its judge. Jev is a
cheap, fast "System One" model that answers typed questions instead of generating text, so
routing can check every step without an LLM judge's cost or latency. Each team writes its own
YAML routes and picks one of four policies (`auto`, `complexity`, `intent`, `escalation`). The
public name is Deferent. The internal Python packages are still named `routerctl` and
`jevjudge`.

## 2. Current state at a glance

| Area | State | Evidence |
|---|---|---|
| Decision-only server (`routerctl serve`, `POST /v1/decide`) | Working, tested | 146 unit tests; `scripts/e2e_routerctl_decide.sh` 7/7 |
| Proxy mode (`--mode proxy`, real Switchyard server) | Working, tested | `scripts/e2e_routerctl.sh` 6/6; `routerctl validate` dry-runs |
| `intent` policy | Strongest policy: 100% decision accuracy in every study | DECISION §10, §13.2, §13.5 |
| `escalation` policy | Good: 89–100% | same |
| `auto` policy | Works **since the tool-call adapter fix** (§12.6). Before that it was blind in decision mode | `routerctl/tests/test_messages.py` |
| `complexity` policy | **Fixed on 2026-09-27** (§14). Held-out accuracy went from 90% to 98%, and misroutes fell from 22 to 6 | `experiments/complexity_eval.py` |
| Cache-aware switching | Built and validated on OpenAI and Anthropic | DECISION §12, §13.4 |
| Provider settings (`extra_body`) in decisions | Fixed 2026-09-27; previously dropped silently | `test_decide.py` |
| Outcome feedback loop | **Not built.** Every decision already carries an `outcome_id` to join against later | roadmap §15 |
| Production deployment | Not deployed anywhere. Nothing runs outside dev containers | — |

Branches and PRs:
* `main` has everything up to PR #3.
* Work after PR #3 is on `claude/jev-ai-router-exploration-m99f1j`, not yet merged:
  - multi-provider experiments
  - the ten-model ladder
  - the `extra_body` fix
  - the complexity fix
  - this document

## 3. Repository map

| Path | What it is |
|---|---|
| `routerctl/src/routerctl/` | The router |
| `routerctl/.../decision_server.py` | FastAPI `/v1/decide`, live reload, cache gate wiring |
| `routerctl/.../decide.py` | Runs one compiled algorithm to a `Decision`; judge fail-open; turn windowing |
| `routerctl/.../algorithms.py` | YAML policy → Switchyard algorithm. `escalation` is reimplemented as a transcript-reading custom classifier (DECISION §11.2) |
| `routerctl/.../rubrics.py` | The `complexity` capability rubric (§14) |
| `routerctl/.../cache.py` | Cache-aware switching gate (§12) |
| `routerctl/.../messages.py` | OpenAI chat shape → libsy blocks, including tool calls and images (§12.6) |
| `routerctl/.../schema.py` | The YAML schema (pydantic) |
| `routerctl/.../compiler.py` | YAML → Switchyard TOML for proxy mode |
| `jevjudge/` | Compiles a route's JSON schema into Jev questions and decodes Jev's answers; confidence gate and fallback cascade |
| `vendor/switchyard/` | Vendored Switchyard source, pinned. Refresh with `scripts/update_vendor.sh` |
| `clients.yaml`, `teams/*.yaml` | Shipped example config. `clients.yaml` also holds `pricing:` for the cache gate |
| `experiments/` | Every experiment script, dataset and result (§6 below). `experiments/README.md` indexes them |
| `evals/` | Model-comparison harness (accuracy, cost and latency per model) |
| `docs/DECISION.md` | The full design record, numbered by section. This file cites it as §N |

## 4. How to run things

```bash
scripts/build.sh                    # builds switchyard-server, bindings, jevjudge, routerctl (~3 min)
source .venv/bin/activate
pytest routerctl/tests jevjudge/tests evals/tests -q   # 154 tests, offline (mock Jev), ~5 s
routerctl validate teams/           # schema check + real switchyard-server --dry-run, no keys needed
routerctl serve teams/ --port 4000  # decision-only server
scripts/e2e_routerctl_decide.sh     # end-to-end, decision mode
scripts/e2e_routerctl.sh            # end-to-end, proxy mode
```

**Keys** live only in the git-ignored `.env`. **Never commit them, and never print them in logs
or chat.** Check `git diff --cached` before every commit.
* `TYPESAFE_API_KEY`: Jev
* `OPENAI_API_KEY`
* `OPENROUTER_API_KEY`: temporary, **expires 2026-10-04**

Load them with `set -a; . ./.env; set +a`. Every experiment script takes a `--max-budget-usd`
cap and stops before crossing it.

## 5. Architecture in five points

1. **Decision-only is the rule.** Nothing in `routerctl` calls a target model. `decide.py`
   raises `UnexpectedRealCallError` if an algorithm ever asks it to. The one network call a
   decision makes is to Jev.
2. **The response carries everything the gateway needs:**
   * `selected_model`, `selected_client`, `selected_extra_body`
   * ranked `fallback_model_ids` with their clients and settings
   * `judge_confidence`
   * `judge_error`: the judge failed and this decision is a fallback
   * `outcome_id`
   * a `cache` block
3. **Policies map to Switchyard mechanisms:**
   * `auto`: `stage_router`, which reads tool-call signals and makes no judge call
   * `complexity`: `llm_classifier`, capability mode, with our own rubric
   * `intent`: `llm_classifier`, custom mode, with bucket descriptions as criteria
   * `escalation`: our own custom classifier that reads the transcript, because Switchyard's
     native version calls the model itself
4. **The judge fails safe.** If Jev is unreachable, each policy falls back to its safe default
   and says so in `judge_error`. Mid-session with a warm cache, the cache gate holds the current
   model instead.
5. **Cache-aware switching is stateless.** The gateway sends `session` (current model, cached
   prefix tokens, idle time, other warm caches), and Deferent prices a switch against losing the
   cache before recommending it.

## 6. Every experiment so far

All costs are real billed amounts. "§" refers to `docs/DECISION.md`.

| # | Experiment | Script → results | Headline | Cost | Caveat |
|---|---|---|---|---|---|
| 1 | Model comparison eval | `evals/run_eval.py` | 4 OpenAI models compared on accuracy, cost and latency; 7 grading and harness bugs found (§10.1) | ~$1 | Small suite |
| 2 | 12 business routers, OpenAI | `experiments/run_experiments.py` → `results/*.json`; report `experiments/REPORT.md` | 216 real decisions. **36.8% blended savings.** No quality loss found in spot checks | $1.27 of workload | Harder-than-typical mix; savings baseline is extrapolated |
| 3 | Cache-aware switching, OpenAI | `experiments/cache_validation.py` → `results/cache-validation.json` | Caches are per model; switching back is warm. 31K-token session **−14%**, easy tail **−37%**. At 8K the gate correctly allows every switch | $1.10 | Scripted policy picks |
| 4 | 12 routers across providers | `experiments/multiprovider.py` → `results/multiprovider/` | Same decision accuracy with tiers on Fireworks, Together, Google and Anthropic. 47.8% extrapolated savings | ~$3.2 | First run superseded (tier inversion, no fallback), archived in `multiprovider-run1-superseded/` |
| 5 | Head-to-head with OpenRouter Auto and Jev Router | same script, 72 prompts | Deferent 4.58 at $0.54. Auto 4.40 at $0.08 (gap not significant). Jev Router 4.14 (significantly worse). Always-top 4.82 at $0.90 | incl. above | **Not a public claim.** One-shot prompts only; our roster had Opus on top |
| 6 | Cache-aware switching, Anthropic | `experiments/cache_validation_anthropic.py` → `results/cache-validation-anthropic.json` | Same mechanics as OpenAI. 45K-token session **−17%**, easy tail **−45%** | $0.88 | Tokenizers differ ~30% across Claude models |
| 7 | Ten-model ladder, ultra-cheap to strong | `experiments/model_ladder.py` → `results/model-ladder/` | Routing halves the cost of always-strongest. The cheap-to-strong quality spread is only 0.25, so no gap is significant at n=72. Misses were all in `complexity` | $0.99 | Single-turn prompts only |
| 8 | Complexity fix, decision accuracy | `experiments/complexity_eval.py` → `results/complexity-eval.json` | Held-out **90% → 96% (general) → 98% (with team criteria)**. Seen set 69% → 91%. Misroutes 22 → 6. Judge gave the same decision on 100/102 re-runs | pennies | Held-out prompts are more clear-cut than the originals |
| 9 | Complexity fix, answer quality | `experiments/model_ladder.py --out results/model-ladder-complexity-fix` | See §14.3 of DECISION.md | ~$0.2 | 3 routes, 18 prompts each |

## 7. Bugs found and fixed, and where the tests pin them

Found by running against real APIs, not mocks. Details are in the § cited.

| Bug | Impact | Fix | § |
|---|---|---|---|
| Jev gate double-counted `primary_rule` uncertainty | Easy tasks were abstained and sent to the expensive model | Gate only on `p_solve` | §9 |
| Grading and harness bugs (max_tokens name, first vs last number, LaTeX grouping, bad test cases) | Wrong eval verdicts | Seven fixes with regression tests | §10.1 |
| Declared-but-unused clients required API keys | Proxy mode wouldn't boot | Emit only referenced clients | §10.2 |
| `stage_router` needs an explicit `any` category | `auto` failed with "target not found" | Add `any` | §11.3 |
| 400-token cap truncated reasoning-model answers | 16% of experiment completions were empty | Cap 1200, plus a targeted retry | REPORT |
| `recent_turn_window` was decorative | Escalation judged whole histories | Enforced in `decide._windowed` | §11 |
| Judge outage was fatal (500) | Router down whenever Jev was down | Fail open with `judge_error` | §11 |
| Tool calls dropped by the message adapter | **`auto` never saw a signal**; images crashed decisions (500) | Map to libsy `tool_call`/`tool_result`/`image` blocks | §12.6 |
| `extra_body` never reached the gateway | Provider pinning and reasoning effort silently lost | `selected_extra_body`, `fallback_extra_bodies` | §13.1 |
| Cache field meant "read count", not "cached size" | A gateway would report 0 after a cold turn and the gate would lose the cache | Renamed to `cached_prefix_tokens`, documented | §12 |
| `complexity` used a coding-agent rubric for business prompts | ~⅓ misroutes in both directions | General rubric, plus `weak_when`/`strong_when` | §14 |
| Stale Sol price ($5/$30, now $4/$20) | Earlier Sol costs slightly overstated | `evals/models.py` | §12.4 |

## 8. Performance

* **Decision latency** (real Jev, 216 decisions):

  | Policy | p50 | p90 |
  |---|---|---|
  | `complexity` | 216 ms | 707 ms |
  | `intent` | 253 ms | 719 ms |
  | `escalation` | 573 ms | 663 ms |
  | `auto` | ~1 ms | — |

  `auto` makes no judge call. One intent outlier took 16 s, a Jev-side slow response. A gateway
  should put a timeout on `/v1/decide` and use the route default on timeout.
* **Judge cost** is negligible next to model spend: Jev is $0.042 per 1M input tokens.
* **Cost savings** depend on the traffic mix and the roster:
  * 36.8% (OpenAI, 12 routers)
  * 47.8% (multi-provider, extrapolated)
  * ~40–50% against always-strongest (measured head-to-heads)
  * −14% to −17% per long session from cache-aware switching, −37% to −45% on the easy tail
* **Spend to date:**

  | Account | Spent | Of |
  |---|---|---|
  | OpenAI | ~$4.8 | the $10 we set ourselves |
  | OpenRouter | ~$9.2 | the key's $50 |
  | TypeSafe (Jev) | cents | — |

## 9. What we learned

1. **Plain-English criteria beat clever rubrics.** `intent`, where a team describes each bucket,
   hit 100% everywhere. `complexity` improved most when given concrete criteria. Encourage teams
   to write `description:`, `weak_when:` and `strong_when:`.
2. **Test against real APIs.** Every significant bug in §7 was invisible to the mock-based tests.
3. **Cheap models are strong on single-turn tasks.** Across ten models, always-cheapest scored
   4.42/5 and always-strongest 4.67/5. Most of the saving comes from *which models are in the
   roster*. Routing's job is to catch the minority of prompts where cheap models fail badly; it
   caught 5 of them in the ladder.
4. **Cache economics are real but specific.**
   * Switching back to a model used within its TTL finds the cache warm. So flip-flopping
     between two warm models is cheap.
   * The expensive event is the *first* cold write of a long context into a new model.
   * The gate matters most on long contexts. On short ones it rightly gets out of the way.
5. **Don't claim we beat OpenRouter on price.** On one-shot prompts, OpenRouter Auto was 6.6×
   cheaper at statistically indistinguishable quality. It draws on hundreds of models and the
   cheapest hosts. The case for our own router is **independence plus cost control**:
   * our own providers, gateway and harness
   * decisions at every step of an agent run
   * cache-aware sessions
   * outcome data only we have

   That is the framing agreed with leadership.
6. **Complexity judging on long contexts over-escalates.** With an 8K-token document in
   context, Jev forecast "strong" even for trivial lookups (§12.4). This is still open.
7. **Harness bugs look like router bugs.** A test gateway that doesn't fall back, or a tier
   mapping that inverts prices, will make routing look broken. Check the harness first.

## 10. Known limitations

* Token counts in the cache gate are estimates (chars ÷ 4) unless the gateway sends real counts.
  Tokenizers also differ by about 30% across model families.
* The >272K-token long-context pricing tier isn't modelled.
* `pricing:` is keyed by model id. Two providers serving the same id at different prices can't
  both be priced, and provider-pinned variants share one entry.
* Adjusting effort instead of switching models isn't implemented. It must be sent in
  cache-safe forms or it breaks the cache too (§12.5).
* No outcome feedback. Thresholds and rosters are set by hand.
* Evaluations are single-turn datasets that we labelled ourselves. There's no production traffic
  sample yet.

## 11. What to do next, in priority order

1. **Merge the open branch** once reviewed (CI: `pytest` and `routerctl validate`).
2. **Outcome loop.** Add an endpoint to report an outcome against `outcome_id`, and a report of
   outcome rate by route, model and policy. Use it to set `base_threshold` and rosters from data.
   Start with one business outcome that's fast and clear, for example voice-call resolution.
3. **Complexity on long contexts** (learning 6): judge the latest turn plus a compact context
   summary. Measure it with `experiments/complexity_eval.py`, extended with long-context items.
4. **Roster tuning.** Re-run `model_ladder.py` with cheaper top tiers, and move each route to the
   cheapest roster whose quality holds.
5. **Per-provider pricing** keyed by `(client, model)`, for pinned providers.
6. **Production hardening:** deployment manifest, metrics (decision latency, judge errors, cache
   holds), and `/v1/decide` timeouts documented for gateway integrators.

## 12. Working conventions

* **Keep decision-only intact.** Any change that makes the router call a target model is a bug.
* **Every fix gets a regression test,** named after the behaviour.
* **Every experiment gets:**
  * a script with a `--max-budget-usd` cap
  * a results JSON under `experiments/results/`
  * a DECISION.md section with method, numbers and caveats

  If a run turns out flawed, archive it as superseded rather than deleting it.
* **Report honestly.** Say when a gap isn't statistically significant, and say what a benchmark
  doesn't measure.
* **Before committing:** run the tests, run `routerctl validate teams/`, and check that no key is
  staged.
