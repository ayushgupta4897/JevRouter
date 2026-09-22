# routerctl

Compiles simple per-team YAML into the Switchyard TOML deployment `switchyard-server` runs, and
serves it with watch-recompile-validate-swap live reload (no hot-reload in Switchyard itself, so
this does a health-checked blue-green swap behind a thin proxy instead of a bare restart).

See the repository's `docs/TEAM_CONFIG.md` for the YAML schema and `teams/` for real examples.

```bash
routerctl validate teams/                    # schema + Switchyard --dry-run, no server started
routerctl compile teams/ -o build/router.toml
routerctl serve teams/ --port 4000            # live: edit a file in teams/, it's live in seconds
```
