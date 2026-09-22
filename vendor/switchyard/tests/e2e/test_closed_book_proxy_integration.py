# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
PROXY_SRC = REPO / "benchmark" / "closed_book_proxy" / "proxy"


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            exc.stdout or "",
            exc.stderr or f"timed out after {timeout}s",
        )


def _compose(work_dir: Path, project: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return _run_command(["docker", "compose", "-p", project, *args], cwd=work_dir, timeout=timeout)


def _compose_service_container(work_dir: Path, project: str, service: str) -> str:
    result = _compose(work_dir, project, "ps", "-q", service, timeout=10)
    assert result.returncode == 0, result.stderr + result.stdout
    container = result.stdout.strip()
    assert container, f"no container id for {service}"
    return container


def _docker_exec(container: str, *args: str, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return _run_command(["docker", "exec", container, *args], timeout=timeout)


def _compose_logs(work_dir: Path, project: str) -> str:
    result = _compose(work_dir, project, "logs", "--no-color", timeout=10)
    return result.stdout + result.stderr


def _write_compose(work_dir: Path, network: str, *, closed_book: bool) -> None:
    shutil.copytree(PROXY_SRC, work_dir / "proxy")
    switchyard_server = (
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path == '/health':\n"
        "            self.send_response(200); self.end_headers(); self.wfile.write(b'ok\\n')\n"
        "        else:\n"
        "            self.send_response(404); self.end_headers()\n"
        "    def log_message(self, *args): pass\n"
        "ThreadingHTTPServer(('0.0.0.0', 4000), H).serve_forever()\n"
    )
    external_server = (
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path == '/health':\n"
        "            self.send_response(200); self.end_headers(); self.wfile.write(b'external-ok\\n')\n"
        "        else:\n"
        "            self.send_response(404); self.end_headers()\n"
        "    def log_message(self, *args): pass\n"
        "ThreadingHTTPServer(('0.0.0.0', 8080), H).serve_forever()\n"
    )
    compose = {
        "services": {
            "main": {
                "image": "curlimages/curl:8.10.1",
                "command": ["sh", "-c", "sleep infinity"],
                "depends_on": {"proxy": {"condition": "service_healthy"}},
                "environment": [
                    "HTTP_PROXY=http://proxy:3128",
                    "HTTPS_PROXY=http://proxy:3128",
                    "NO_PROXY=localhost,127.0.0.1,proxy",
                    "http_proxy=http://proxy:3128",
                    "https_proxy=http://proxy:3128",
                    "no_proxy=localhost,127.0.0.1,proxy",
                ],
                "networks": ["agent-internal"],
            },
            "proxy": {
                "build": {"context": "./proxy"},
                "environment": [
                    f"CLOSED_BOOK_MODE={1 if closed_book else 0}",
                    "OPENAI_BASE_URL=http://switchyard:4000/v1",
                    "SWITCHYARD_BASE_URL=http://switchyard:4000",
                    "ALLOWED_HOSTS=switchyard",
                    "VERIFIER_PROXY_TOKEN=test-token",
                ],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import socket;s=socket.create_connection(('127.0.0.1',3128),2);s.close()",
                    ],
                    "interval": "2s",
                    "timeout": "2s",
                    "retries": 30,
                },
                "networks": ["agent-internal", "switchyard-egress"],
            },
            "switchyard": {
                "image": "python:3.12-slim",
                "command": ["python", "-c", switchyard_server],
                "networks": ["switchyard-egress"],
            },
            "cheat": {
                "image": "python:3.12-slim",
                "command": ["python", "-c", external_server],
                "networks": {"switchyard-egress": {"aliases": ["raw.githubusercontent.com"]}},
            },
        },
        "networks": {
            "agent-internal": {"driver": "bridge", "internal": True},
            "switchyard-egress": {"external": True, "name": network},
        },
    }
    (work_dir / "docker-compose.yaml").write_text(yaml.safe_dump(compose, sort_keys=False))


def test_closed_book_proxy_allows_switchyard_service_and_blocks_cheat_source() -> None:
    if not _docker_compose_available():
        pytest.skip("Docker Compose is not available")

    run_id = uuid.uuid4().hex[:8]
    work_dir = REPO / "benchmark" / "tb_runs" / f".cb-it-{run_id}"
    work_dir.mkdir(parents=True)
    project = f"switchyard-closed-book-{run_id}"
    network = f"switchyard-closed-book-{run_id}"
    _write_compose(work_dir, network, closed_book=True)

    try:
        network_create = _run_command(["docker", "network", "create", network], timeout=30)
        assert network_create.returncode == 0, network_create.stderr + network_create.stdout
        up = _compose(work_dir, project, "up", "-d", "--build", timeout=180)
        assert up.returncode == 0, up.stderr + up.stdout
        main_container = _compose_service_container(work_dir, project, "main")

        result = _docker_exec(
            main_container,
            "sh",
            "-lc",
            "curl -fsS --max-time 10 http://switchyard:4000/health",
            timeout=15,
        )
        assert result.returncode == 0, _compose_logs(work_dir, project) + result.stderr

        blocked = _docker_exec(
            main_container,
            "sh",
            "-lc",
            "curl -sS -i --max-time 10 http://raw.githubusercontent.com:8080/health",
            timeout=15,
        )
        combined = blocked.stdout + blocked.stderr
        assert "403" in combined or "closed-book proxy denied outbound host" in combined

        verifier_open = _docker_exec(
            main_container,
            "sh",
            "-lc",
            (
                "HTTP_PROXY=http://verifier:test-token@proxy:3129 "
                "HTTPS_PROXY=http://verifier:test-token@proxy:3129 "
                "http_proxy=http://verifier:test-token@proxy:3129 "
                "https_proxy=http://verifier:test-token@proxy:3129 "
                "NO_PROXY= no_proxy= "
                "curl -fsS --max-time 10 http://cheat:8080/health"
            ),
            timeout=15,
        )
        assert verifier_open.returncode == 0, _compose_logs(work_dir, project) + verifier_open.stderr
        assert "external-ok" in verifier_open.stdout
    finally:
        _compose(work_dir, project, "down", "-v", "--remove-orphans")
        _run_command(["docker", "network", "rm", network], timeout=30)
        shutil.rmtree(work_dir, ignore_errors=True)


def test_open_book_proxy_allows_raw_github_hostname() -> None:
    if not _docker_compose_available():
        pytest.skip("Docker Compose is not available")

    run_id = uuid.uuid4().hex[:8]
    work_dir = REPO / "benchmark" / "tb_runs" / f".ob-it-{run_id}"
    work_dir.mkdir(parents=True)
    project = f"switchyard-open-book-{run_id}"
    network = f"switchyard-open-book-{run_id}"
    _write_compose(work_dir, network, closed_book=False)

    try:
        network_create = _run_command(["docker", "network", "create", network], timeout=30)
        assert network_create.returncode == 0, network_create.stderr + network_create.stdout
        up = _compose(work_dir, project, "up", "-d", "--build", timeout=180)
        assert up.returncode == 0, up.stderr + up.stdout
        main_container = _compose_service_container(work_dir, project, "main")

        allowed = _docker_exec(
            main_container,
            "sh",
            "-lc",
            "curl -fsS --max-time 10 http://raw.githubusercontent.com:8080/health",
            timeout=15,
        )
        assert allowed.returncode == 0, _compose_logs(work_dir, project) + allowed.stderr
        assert "external-ok" in allowed.stdout
    finally:
        _compose(work_dir, project, "down", "-v", "--remove-orphans")
        _run_command(["docker", "network", "rm", network], timeout=30)
        shutil.rmtree(work_dir, ignore_errors=True)
