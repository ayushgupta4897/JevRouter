# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROXY_DIR = REPO / "benchmark" / "closed_book_proxy" / "proxy"


def test_verifier_proxy_is_authenticated_mitmproxy_listener() -> None:
    entrypoint = (PROXY_DIR / "entrypoint.sh").read_text()
    dockerfile = (PROXY_DIR / "Dockerfile").read_text()

    assert 'VERIFIER_STATE_DIR="${STATE_DIR}/verifier"' in entrypoint
    assert "--listen-port 3129" in entrypoint
    assert '--proxyauth "verifier:${VERIFIER_PROXY_TOKEN}"' in entrypoint
    assert '--ignore-hosts ".*"' in entrypoint
    assert "verifier_proxy_pid=$!" in entrypoint

    assert "--listen-port 3128" in entrypoint
    assert "-s /opt/closed-book-proxy/rewriter.py" in entrypoint
    assert "agent_proxy_pid=$!" in entrypoint

    assert "verifier_proxy.py" not in dockerfile
    assert "EXPOSE 3128 3129" in dockerfile
