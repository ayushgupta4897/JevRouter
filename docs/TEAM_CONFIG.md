# Team routing config

Write a route, save the file, it's live in a few seconds. No platform ticket, no restart, and a
mistake in your file never affects any other team's routes.

```bash
routerctl validate teams/              # schema check + a real Switchyard --dry-run, no keys needed
routerctl serve teams/ --port 4000     # decision-only by default -- see below
```

`routerctl serve` defaults to **decision-only**: `POST /v1/decide` with `{"model": "<route
name>", "messages": [...]}` and it tells you which model and client should serve it —
`{"selected_model": "...", "selected_client": "...", ...}` — without ever calling that model.
Your own AI Gateway (or anyone's) makes the real request; this keeps deciding and executing as
two separate responsibilities, never one process doing both. Pass `--mode proxy` instead to run
the same config as a real forwarding proxy (Switchyard itself, with live reload) if you don't
have a gateway of your own yet — see the README's "Multi-team routing" section for both.

## The two files

**`clients.yaml`** (repo root, platform-managed) lists every upstream a route can use — a name,
a base URL, and which environment variable holds its key. You reference these by name; you never
put a credential in your own file.

**`teams/<your-team>.yaml`** (yours) is everything you own:

```yaml
team: voice-ai
owner: voice-ai-team@example.com     # optional

routes:
  - name: deployment/transcription   # the model id your clients call with
    policy: intent                   # see below
    # ...policy-specific fields...
```

A route's `name` is what a client puts in the `model` field of an ordinary OpenAI or Anthropic
request. Names must be unique across every team — `routerctl validate` fails loudly, naming both
files, if two teams pick the same one.

## The four policies

Every judge-backed policy below uses **Jev** (this project's own `jevjudge` sidecar) as its
classifier by default — nothing to configure, and it costs a fraction of a cent per decision. You
can point a route at a different judge with `judge: { id: ..., client: ... }` if you'd rather use
an LLM (see `README.md` for why you might: Jev is fast and cheap but not infallible, and the
sidecar's own confidence gate is what a real LLM fallback is for).

### `auto` — the zero-config default

Use this when you have no strong opinion yet. It's Switchyard's own tuned default: cheap first,
escalating off live signals in the conversation (tool errors, churn, exploration vs. production),
no judge call at all.

```yaml
- name: deployment/default
  policy: auto
  models:
    efficient: { id: gpt-5.6-terra, client: openai }
    capable:   { id: gpt-6-astra,   client: openai }
```

### `complexity` — judge the task before running it

A judge estimates whether the cheap model can do the whole task; a threshold decides. Good when
you can describe "easy" vs. "hard" for your domain but don't have (or want) an agentic loop to
watch.

```yaml
- name: deployment/qa-summary
  policy: complexity
  models:
    weak:   { id: gpt-5.6-terra, client: openai }
    strong: { id: gpt-6-astra,   client: openai }
  base_threshold: 0.55    # lowest solve-probability that still uses the weak model. 0-1, default 0.5
  threshold_step: 0.1     # extra caution the judge asks for when its own reasoning is shaky. default 0.1
  # optional, plain English: what your traffic looks like on each side
  weak_when: "A single arithmetic step: a percentage, growth rate, conversion, runway, or total."
  strong_when: "Multi-scenario modelling, reconciling metrics that move in different directions, or cap-table maths."
  rubric: general         # default. `coding_agent` = Switchyard's original rubric, for agents in a repo
```

`weak_when` and `strong_when` are optional but worth writing. They work like an `intent`
bucket's `description:`. The judge forecasts against a short rulebook of what the cheap model
can and can't handle, and your sentences replace its most generic rule on each side. In testing,
adding them took held-out decision accuracy from 96% to 98%, and cut easy prompts wrongly sent
to the strong model from 2 to 0 (`docs/DECISION.md` §14).

`rubric` picks that rulebook:
* `general` (the default) judges the task itself: steps, interacting constraints, and how likely
  a plausible answer is to be wrong.
* `coding_agent` is Switchyard's packaged rubric. It's written for agents working in a repository
  with tools and a verifier. Use it only for routes like that. On one-shot business prompts it
  misrouted about a third of the time.

Raising either number sends more traffic to the strong model. Start at the defaults and move
`base_threshold` up if the weak model is failing tasks it was routed, or down if too much traffic
lands on the strong model for tasks that turned out easy.

### `escalation` — judge the run, not the request

Every conversation starts on the cheap model. A judge reads the transcript so far — tool calls,
tool results, repeated errors — and escalates to the strong model once it sees a genuine pattern
of trouble, not a single failure. This is the mode to reach for when the same "how hard is this"
question can't be answered up front, but the run tells you as it goes.

```yaml
- name: deployment/coding-agent
  policy: escalation
  models:
    weak:   { id: gpt-5.6-sol,  client: openai }
    strong: { id: gpt-6-astra,  client: openai }
  confirmations: 2          # baked into the judge's prompt as "at least N times in a row"
  recent_turn_window: 28    # trailing messages the judge is told to focus on
```

**The two serving modes implement this differently, and it's worth knowing which you're
running.** Switchyard's native `escalation` classifier (used by `--mode proxy`) calls the
current tier's model itself to observe its live response, then judges that, and keeps a
confirmation streak across calls tied to an `x-switchyard-session-id` header. Decision-only mode
(the default) can't do that — calling the current tier would violate "the router never calls a
model" — so it instead judges the transcript already present in your request in one shot, with no
cross-call session state at all: `confirmations`/`recent_turn_window` become instructions inside
the judge's own prompt ("a pattern recurring at least N times, looking at the last M turns")
rather than counters Switchyard tracks for you. Practically: make sure your request's `messages`
actually include the recent tool calls/results/errors you want judged — decision mode has no
memory of previous requests to fall back on.

### `auto` or `escalation`?

Both start on the cheap model and move up when a run is going badly. They read the run
differently. Here are real decisions on the same transcripts (`experiments/auto_vs_escalation.py`):

| Transcript | `auto` | `escalation` |
|---|---|---|
| Coding agent hits the same traceback 3× | strong | strong |
| Coding agent: edits land, tests pass | cheap | cheap |
| Support chatbot breaks the same promise 3×, no tools | cheap | **strong** |
| One failure, then fixed | cheap | cheap |

* **`auto`** reads tool traffic: errors in tool results, spinning, and exploring vs. producing. It
  makes no judge call and takes about 1 ms. It's built for coding agents: it can't see trouble
  that isn't in a tool result, and it doesn't know tools outside its coding vocabulary.
* **`escalation`** asks the judge whether the transcript shows a repeated-failure pattern. It
  costs one judge call (about 0.2 s). It works for any conversation, and `confirmations` and
  `recent_turn_window` tune it.

Neither judges how hard a request is before anything has happened. For that, use `complexity`.

### `intent` — route by what the request actually is

A judge picks one of your named buckets by content. This is the shape for "if it's an extraction
request, use this model; if it's a computation, use that one" — the voice-AI example this schema
was built around:

```yaml
- name: deployment/transcription
  policy: intent
  default: analysis   # used when the judge is unsure or unreachable
  models:
    extraction:
      id: gpt-5.6-luna
      client: openai
      description: >
        Pulling a structured field straight out of the transcript: a name, date, amount, or
        yes/no outcome. No judgment call, the answer is either in the transcript or it isn't.
    analysis:
      id: gpt-5.6-sol
      client: openai
      description: >
        Summarizing the call, judging sentiment, or flagging a compliance risk -- anything
        that takes reading the whole conversation rather than looking up one fact.
```

At least two buckets, each with a `description`. **The description is the routing signal** — it
becomes the judge's own criterion for that bucket (literally: it's copied into the judge's
prompt, one line per bucket). A vague description gets vague routing; a specific one, naming the
actual kind of request that belongs there, is what makes this accurate. `default` is where a
request goes if the judge can't decide — pick your safer, more capable bucket, not your cheapest.

## Model definitions

Every `models.<role>` entry, in every policy, takes the same three fields:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | The exact model ID sent upstream, e.g. `gpt-5.6-sol`. |
| `client` | yes | A name from `clients.yaml`. |
| `description` | only for `intent` buckets | What routes here — see above. |
| `extra_body` | no | Provider-specific extras sent with the call, e.g. `{reasoning: {effort: high}}` or OpenRouter provider pinning `{provider: {order: [fireworks], allow_fallbacks: false}}`. In decision mode it comes back on every decision as `selected_extra_body` (and `fallback_extra_bodies`) for your gateway to send. |

The same `(id, client)` pair used by two different routes becomes **one shared target** — this is
deliberate (Switchyard would otherwise warn and silently collapse the duplicates itself; see
`routerctl.compiler.TargetRegistry`), so two routes both using `gpt-6-astra` on `openai` share a
connection pool rather than duplicating it. If you need genuinely different settings for the
"same" model in two routes, give it a second entry in `clients.yaml` instead of relying on
`extra_body` to distinguish them — `routerctl validate` will refuse to compile that ambiguity.

## Cache-aware switching

Decision-only mode. It's on for every route, and it does nothing until a request includes
`session`, which says which model served the previous turn. The gateway then gets a decision
that prices what a model switch would cost in lost prompt cache. The design and real-world
results are in `docs/DECISION.md` §12.

**Per route (optional).** The defaults suit most routes:

```yaml
  - name: deployment/coding-agent
    policy: escalation
    models: { ... }
    cache:
      enabled: true                 # false = always follow the policy
      horizon_turns: 10             # weigh a switch over this many upcoming turns (default 5)
      expected_output_tokens: 500   # per-turn output estimate for pricing (default 500)
      switch_margin: 0.10           # a downgrade must save >= 10% over the horizon (default)
      upgrade_min_confidence: 0.2   # a weaker verdict won't drop a warm cache to upgrade (default)
```

Raise `horizon_turns` for long agent sessions. It makes downgrades harder, because a cold start
costs you once and the saving comes back every turn after.

**Prices (platform-managed, `clients.yaml`).** Prices are in USD per 1M tokens, keyed by model
id. A switch involving a model with no price listed just follows the policy.

```yaml
pricing:
  gpt-5.6-terra: { input: 2.00, cached_input: 0.20, cache_write: 2.50, output: 12.00, cache_ttl_seconds: 1800 }
```

| Field | Default | Meaning |
|---|---|---|
| `input`, `cached_input`, `output` | required | Standard rates. `cached_input` must be ≤ `input`. |
| `cache_write` | none | Premium for writing uncached input into the cache: 1.25× input on OpenAI GPT-5.6+ and on Anthropic's 5-minute cache. Omit it where writes cost the normal input price. |
| `cache_ttl_seconds` | 300 | Idle time after which the cache counts as gone. OpenAI GPT-5.6+ is 1800. |
| `min_cacheable_tokens` | 1024 | Prompts shorter than this are never cached. It's 512–4096 on Anthropic, depending on model. |

**The request's `session` field.** Every field is optional, and unknown fields are rejected.

| Field | What the gateway sends |
|---|---|
| `current_model` | The model that served the previous turn. |
| `current_client` | Its client, if the route uses the same model on more than one. |
| `cached_prefix_tokens` | How much of the conversation that model now has cached: everything read from cache *plus* everything written. On OpenAI, that's the previous turn's `usage.prompt_tokens`. On Anthropic, it's `cache_read_input_tokens + cache_creation_input_tokens`. Don't send the read count alone: a cold turn reads 0 but still leaves the cache warm. |
| `idle_seconds` | Seconds since the previous turn. |
| `remaining_turns` | If you know how many turns are left, this overrides `horizon_turns`. |
| `other_warm_caches` | `{model_id: tokens}` for other models this session used within their TTL. Measured on real OpenAI calls: switching back to one is warm, so it's priced that way. |

Every decision's `cache.reason` is one of the following.

The gate followed the policy's pick when the reason is:
* `no_session`, `same_model`, `disabled`
* `current_not_in_route`, `no_pricing`
* `cache_expired`, `prefix_below_cache_minimum`
* `upgrade`, `downgrade_saves_money`

The gate held the current model when the reason is:
* `downgrade_not_worth_losing_cache`
* `upgrade_verdict_too_borderline`
* `judge_unavailable_hold`

When the gate holds, the policy's pick stays first in `fallback_model_ids`.

## What "live" actually means

Both modes watch every file and reject a bad edit while keeping your last good config serving —
a syntax error or a typo'd client name never reaches production, and only your routes would have
been affected if it had gone live, never another team's. How the swap itself happens differs:

* **`--mode decide`** (default): a config change recompiles in-process straight into
  `switchyard.libsy` algorithm objects and atomically swaps a dict — no subprocess, no health
  check, because there's no second process serving traffic to stand up.
* **`--mode proxy`**: Switchyard's server has no config hot-reload of its own — a change means a
  new process. `routerctl serve` handles that for you: on a change it recompiles, runs a real
  `switchyard-server --dry-run`, and only if that passes does it boot a new process, wait for its
  health check, and atomically swap a small proxy over to it before draining the old one. This
  costs about a second or two of health-check time, so treat "live" as "within a few seconds,"
  not instantaneous.

Confidence-gate tuning (how cautious Jev's fallback/abstain behavior is) is currently a
platform-wide setting on the shared `jevjudge` sidecar in both modes, not yet per-route — see
`docs/DECISION.md`'s roadmap.

## Validating before you save

```bash
routerctl validate teams/                 # everyone's routes, including yours
routerctl validate teams/ --skip-dry-run  # schema only, no Switchyard binary needed
```

`validate` doesn't need real API keys — it fills in a placeholder for `--dry-run`'s sake (which
makes no network calls), so this is safe to run in CI. Whether `routerctl serve` itself needs
real keys depends on the mode: **`--mode decide` never calls a client at all**, so it never needs
real keys for the models your routes name (only for the judge, if you point one at a paid LLM
fallback instead of the default Jev sidecar); **`--mode proxy`** does need a real key for every
client it actually forwards to, since it makes the real call.
