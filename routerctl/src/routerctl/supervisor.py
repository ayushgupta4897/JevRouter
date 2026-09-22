"""Watch team YAML, recompile, validate, and swap into a live Switchyard process.

    client -> [proxy: fixed public port]  ->  switchyard-server (backend port A or B)
                       ^
                       |  swapped atomically once the new backend passes its health check
                supervisor loop (this module)

One config generation at a time is ever "current". A bad edit from any team fails validation
(schema, then `switchyard-server --dry-run`) and is logged and skipped -- the process already
serving traffic keeps serving the last-known-good compiled config, for every team, until the
bad file is fixed. There is no partial rollout: the compiled TOML is one file covering every
team's routes, so a swap always moves every route together, atomically, or not at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
from pathlib import Path

import httpx
import uvicorn

from .compiler import CompileResult, ConfigError, compile_directory
from .proxy import create_app

log = logging.getLogger("routerctl.supervisor")

HEALTH_TIMEOUT_S = 15.0
HEALTH_POLL_S = 0.25
DRAIN_TIMEOUT_S = 10.0


def _content_hash(teams_dir: Path, clients_path: Path) -> str:
    paths = sorted(teams_dir.glob("*.yaml")) + ([clients_path] if clients_path.exists() else [])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Backend:
    """One running `switchyard-server` subprocess and the port it's on."""

    def __init__(self, port: int, process: asyncio.subprocess.Process, log_path: Path):
        self.port = port
        self.process = process
        self.log_path = log_path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def wait_healthy(self, http: httpx.AsyncClient, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if self.process.returncode is not None:
                return False  # died before becoming healthy
            try:
                response = await http.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(HEALTH_POLL_S)
        return False

    async def drain_and_stop(self) -> None:
        if self.process.returncode is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=DRAIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("backend on port %d did not drain within %.0fs; killing it", self.port, DRAIN_TIMEOUT_S)
            self.process.kill()
            await self.process.wait()


async def _dry_run_ok(switchyard_server: str, result: CompileResult, build_dir: Path) -> tuple[bool, str]:
    # See cli.py's `_dry_run`: --dry-run still requires every api_key_env to exist, so fill in
    # placeholders for validation only. The real launch below (`_launch_backend`) does not do
    # this and inherits the supervisor's own real environment, so a genuinely missing key still
    # fails the health check as it should.
    env = {**os.environ, **{name: os.environ.get(name) or "routerctl-dry-run-placeholder" for name in result.api_key_envs}}
    check_path = build_dir / "_dry_run_check.toml"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(result.toml)
    process = await asyncio.create_subprocess_exec(
        switchyard_server, "--config", str(check_path), "--dry-run",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30.0)
    return process.returncode == 0, stdout.decode(errors="replace").strip()


async def _launch_backend(switchyard_server: str, toml_path: Path, port: int, log_path: Path) -> Backend:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as handle:
        process = await asyncio.create_subprocess_exec(
            switchyard_server, "--config", str(toml_path), "--host", "127.0.0.1", "--port", str(port),
            stdout=handle, stderr=asyncio.subprocess.STDOUT,
        )
    return Backend(port=port, process=process, log_path=log_path)


async def _watch_loop(
    app,
    teams_dir: Path,
    clients_path: Path,
    switchyard_server: str,
    poll_seconds: float,
    build_dir: Path,
    port_a: int,
    port_b: int,
) -> None:
    http = app.state.http
    current_backend: Backend | None = None
    current_hash: str | None = None
    last_seen_hash: str | None = None  # last hash we already logged/acted on, success or failure
    generation = 0
    next_port_toggle = port_a

    while True:
        try:
            content_hash = _content_hash(teams_dir, clients_path)
        except OSError as error:
            log.error("could not read config files: %s", error)
            await asyncio.sleep(poll_seconds)
            continue

        if content_hash == last_seen_hash:
            await asyncio.sleep(poll_seconds)
            continue
        last_seen_hash = content_hash

        try:
            result: CompileResult = compile_directory(teams_dir, clients_path)
        except ConfigError as error:
            log.error("config invalid, keeping generation %d live: %s: %s", generation, error.file, error)
            await asyncio.sleep(poll_seconds)
            continue

        ok, output = await _dry_run_ok(switchyard_server, result, build_dir)
        if not ok:
            log.error("switchyard-server --dry-run failed, keeping generation %d live:\n%s", generation, output)
            await asyncio.sleep(poll_seconds)
            continue

        generation += 1
        toml_path = build_dir / f"router.g{generation}.toml"
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        toml_path.write_text(result.toml)

        port = next_port_toggle
        next_port_toggle = port_b if port == port_a else port_a
        log_path = build_dir / "logs" / f"switchyard.g{generation}.log"
        log.info("generation %d: starting switchyard-server on port %d (%s)", generation, port, toml_path)
        new_backend = await _launch_backend(switchyard_server, toml_path, port, log_path)

        if not await new_backend.wait_healthy(http):
            log.error(
                "generation %d failed its health check on port %d; keeping the previous generation live. "
                "See %s for what it printed.", generation, port, log_path,
            )
            await new_backend.drain_and_stop()
            continue

        app.state.proxy.upstream = new_backend.base_url
        app.state.proxy.generation = generation
        routes = ", ".join(sorted(result.route_names))
        log.info("generation %d is live on port %d. routes: %s", generation, port, routes)

        if current_backend is not None:
            await current_backend.drain_and_stop()
        current_backend = new_backend
        current_hash = content_hash
        await asyncio.sleep(poll_seconds)

    # unreachable; loop exits only via cancellation from run_supervisor's shutdown handling
    _ = current_hash  # keep linters quiet about the final assignment being "unused"


def run_supervisor(
    teams_dir: Path,
    clients_path: Path | None,
    public_port: int,
    switchyard_server: str,
    poll_seconds: float,
    build_dir: Path,
) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    clients_path = clients_path or (teams_dir.parent / "clients.yaml")
    app = create_app()
    port_a, port_b = public_port + 1, public_port + 2

    async def main() -> None:
        watch_task = asyncio.create_task(
            _watch_loop(app, teams_dir, clients_path, switchyard_server, poll_seconds, build_dir, port_a, port_b)
        )
        config = uvicorn.Config(app, host="0.0.0.0", port=public_port, log_level="warning")
        server = uvicorn.Server(config)
        log.info(
            "routerctl proxy on :%d -> backends on :%d/:%d, watching %s (every %.1fs)",
            public_port, port_a, port_b, teams_dir, poll_seconds,
        )
        try:
            await server.serve()
        finally:
            watch_task.cancel()
            if app.state.proxy.upstream:
                log.info("shutting down; draining the live backend")
            # Best-effort: there's no direct handle to current_backend here since it lives in
            # the watch task's closure. A supervised deployment terminates this whole process
            # group on shutdown, and switchyard-server itself handles SIGTERM/Ctrl-C gracefully
            # (see vendor/switchyard/crates/switchyard-server/src/shutdown.rs) with the same
            # drain behavior _watch_loop uses between generations.

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0
