# routerctl

Compiles simple per-team YAML into a routing decision. Two serving modes:

* **`--mode decide`** (the default): a decision-only service. `POST /v1/decide` returns which
  model and client should serve a request — never a completion. Embeds Switchyard's own `libsy`
  routing algorithms directly (no subprocess, no HTTP proxy); the shared Jev judge is the only
  real call any policy makes. An external AI Gateway executes the actual request.
* **`--mode proxy`**: compiles to the Switchyard TOML deployment `switchyard-server` runs, and
  serves it with watch-recompile-validate-swap live reload (no hot-reload in Switchyard itself,
  so this does a health-checked blue-green swap behind a thin proxy instead of a bare restart).
  For teams with no AI Gateway of their own yet.

See the repository's `docs/TEAM_CONFIG.md` for the YAML schema and `teams/` for real examples.

```bash
routerctl validate teams/                     # schema + Switchyard --dry-run, no server started
routerctl compile teams/ -o build/router.toml # (proxy-mode artifact) compile without serving
routerctl serve teams/ --port 4000            # decision-only; edit a file in teams/, it's live in seconds
routerctl serve teams/ --port 4000 --mode proxy
```
