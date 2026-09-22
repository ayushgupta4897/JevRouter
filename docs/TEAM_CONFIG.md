# Team routing config

Write a route, save the file, it's live in a few seconds. No platform ticket, no restart, and a
mistake in your file never affects any other team's routes.

```bash
routerctl validate teams/              # schema check + a real Switchyard --dry-run, no keys needed
routerctl serve teams/ --port 4000     # runs it, watching teams/ for edits
```

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
```

Raising either number sends more traffic to the strong model. Start at the defaults and move
`base_threshold` up if the weak model is failing tasks it was routed, or down if too much traffic
lands on the strong model for tasks that turned out easy.

### `escalation` — judge the run, not the request

Every conversation starts on the cheap model. A judge watches the actual transcript — tool calls,
tool results, repeated errors — and escalates to the strong model once it sees a genuine pattern
of trouble, not a single failure. This is the mode to reach for when the same "how hard is this"
question can't be answered up front, but the run tells you as it goes.

```yaml
- name: deployment/coding-agent
  policy: escalation
  models:
    weak:   { id: gpt-5.6-sol,  client: openai }
    strong: { id: gpt-6-astra,  client: openai }
  confirmations: 2          # consecutive "escalate" verdicts required to switch tiers. default 2
  recent_turn_window: 28    # trailing messages the judge sees. default 28
```

Escalation needs a stable session so the streak survives across turns: send an
`x-switchyard-session-id` header with each request in the same conversation.

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
| `extra_body` | no | Provider-specific extras, e.g. `{reasoning: {effort: high}}`. |

The same `(id, client)` pair used by two different routes becomes **one shared target** — this is
deliberate (Switchyard would otherwise warn and silently collapse the duplicates itself; see
`routerctl.compiler.TargetRegistry`), so two routes both using `gpt-6-astra` on `openai` share a
connection pool rather than duplicating it. If you need genuinely different settings for the
"same" model in two routes, give it a second entry in `clients.yaml` instead of relying on
`extra_body` to distinguish them — `routerctl validate` will refuse to compile that ambiguity.

## What "live" actually means

Switchyard's server has no config hot-reload — a change means a new process. `routerctl serve`
handles that for you: it watches every file, and on a change it recompiles, runs a real
`switchyard-server --dry-run`, and only if that passes does it boot a new process, wait for its
health check, and atomically swap a small proxy over to it before draining the old one. Clients
never see a dropped connection, and a syntax error or a typo'd client name in your file is caught
before it ever reaches production — the platform keeps serving your last good config, and only
your routes would have been affected if it had gone live, never another team's.

Two costs worth knowing: a swap needs about a second or two of health-check time, so treat "live"
as "live within a few seconds," not instantaneous; and confidence-gate tuning (how cautious Jev's
fallback/abstain behavior is) is currently a platform-wide setting on the shared `jevjudge`
sidecar, not yet per-route — see `docs/DECISION.md`'s roadmap.

## Validating before you save

```bash
routerctl validate teams/                 # everyone's routes, including yours
routerctl validate teams/ --skip-dry-run  # schema only, no Switchyard binary needed
```

`validate` doesn't need real API keys — it fills in a placeholder for `--dry-run`'s sake (which
makes no network calls), so this is safe to run in CI. `routerctl serve` does need real keys for
any client it actually calls.
