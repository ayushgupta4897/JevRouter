"""Decision-only HTTP service: default `routerctl serve` mode.

Given a request shaped like an OpenAI chat-completions call (`model` set to a route name, e.g.
`deployment/transcription`, plus `messages`), returns *which* model should serve it -- never the
model's own completion. An external AI Gateway is expected to make the real call; this process
never does. See docs/DECISION.md's real-router section and README's "Decision-only mode".

Live reload is much simpler than the proxy mode's subprocess-swap dance (`supervisor.py`):
routes compile to in-process Python objects (`algorithms.compile_route_algorithm`), so a config
change just rebuilds a dict and swaps a reference -- no subprocess to boot or health-check.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jevjudge.cascade import Judge, build_judge_from_env

from . import cache
from .algorithms import CompiledRoute, compile_route_algorithm
from .compiler import ConfigError, load_clients_file, load_team_file
from .decide import UnexpectedRealCallError, decide
from .messages import to_libsy_messages
from .schema import ModelPricing

logger = logging.getLogger("routerctl.decision_server")


class DecisionRequest(BaseModel):
    model: str = Field(min_length=1, description="The route name, e.g. 'deployment/transcription'.")
    messages: list[dict] = Field(min_length=1)
    session: cache.SessionState | None = Field(
        default=None, description="Which model served the previous turn, for cache-aware switching.",
    )


class RouteTable:
    """The compiled routes currently being served, with hash-triggered reload from disk. A
    reload that fails to validate keeps the previous, last-good table live -- same guarantee
    proxy mode's supervisor makes, just without a second OS process to swap."""

    def __init__(self, teams_dir: Path, clients_path: Path | None = None):
        self.teams_dir = teams_dir
        self.clients_path = clients_path or (teams_dir.parent / "clients.yaml")
        self.routes: dict[str, CompiledRoute] = {}
        self.pricing: dict[str, ModelPricing] = {}
        self._content_hash: str | None = None

    def _hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.clients_path.read_bytes())
        for path in sorted(self.teams_dir.glob("*.yaml")):
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def load(self) -> None:
        # clients.yaml is still validated (catches typos, the reserved-name check, etc.) even
        # though a decision-only server never dials out to a client's base_url itself. Its
        # pricing table is what cache-aware switching prices a model switch with.
        clients = load_clients_file(self.clients_path)
        routes: dict[str, CompiledRoute] = {}
        team_paths = sorted(self.teams_dir.glob("*.yaml"))
        if not team_paths:
            raise ConfigError(f"no team YAML files found in {self.teams_dir}", file=str(self.teams_dir))
        for path in team_paths:
            team = load_team_file(path)
            for route in team.routes:
                if route.name in routes:
                    raise ConfigError(f"route name {route.name!r} is used twice", file=str(path))
                routes[route.name] = compile_route_algorithm(route)
        self.routes, self.pricing = routes, clients.pricing
        self._content_hash = self._hash()

    def maybe_reload(self) -> bool:
        if self._hash() == self._content_hash:
            return False
        try:
            self.load()
            logger.info("routes reloaded: %s", ", ".join(sorted(self.routes)))
            return True
        except Exception as error:  # noqa: BLE001 -- keep serving the last-good table regardless of cause
            logger.error("reload failed, keeping previous routes live: %s", error)
            return False


def build_app(teams_dir: Path, clients_path: Path | None = None, poll_seconds: float = 1.5, judge: Judge | None = None) -> FastAPI:
    table = RouteTable(teams_dir, clients_path)
    table.load()
    judge = judge or build_judge_from_env()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def watch_loop() -> None:
            while True:
                await asyncio.sleep(poll_seconds)
                table.maybe_reload()

        watcher = asyncio.create_task(watch_loop())
        try:
            yield
        finally:
            watcher.cancel()
            await judge.aclose()

    app = FastAPI(title="routerctl decision service", lifespan=lifespan)

    @app.get("/_routerctl/health")
    async def health() -> dict:
        return {"ok": True, "mode": "decide", "routes": sorted(table.routes)}

    @app.post("/v1/decide")
    async def decide_endpoint(req: DecisionRequest) -> dict:
        compiled = table.routes.get(req.model)
        if compiled is None:
            raise HTTPException(404, f"no route named {req.model!r}; known routes: {sorted(table.routes)}")
        request = {"model": req.model, "stream": False, "messages": to_libsy_messages(req.messages)}
        try:
            decision = await decide(compiled, judge, request)
        except UnexpectedRealCallError as error:
            # This is this project's own invariant breaking (algorithms.py should make it
            # structurally impossible), not a caller mistake -- logged with a stack trace so it
            # gets noticed, reported as a plain 500 without leaking internals to the client.
            logger.exception("decision-only invariant violated on route %r", req.model)
            raise HTTPException(500, "internal routing error") from error
        decision = cache.apply(decision, compiled, req.session, table.pricing, compiled.route.cache, req.messages)
        if decision.judge_error:
            held = (decision.cache or {}).get("held_current_model")
            fallback = "held the session's current model" if held else "fell back to its safe default"
            logger.warning("route %r: judge unreachable, %s: %s", req.model, fallback, decision.judge_error)
        return asdict(decision)

    return app


__all__ = ["RouteTable", "build_app"]
