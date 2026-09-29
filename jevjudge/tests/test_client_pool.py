"""Connection reuse is most of Jev's latency: ~190 ms on a warm connection vs ~570 ms when a new
TLS connection has to be opened (docs/DECISION.md section 16). These pin the pool settings and
the warm-up behaviour that keep decisions on warm connections."""

from __future__ import annotations

import httpx

from jevjudge.client import JevClient, JevClientConfig


def test_idle_connections_are_kept_far_longer_than_httpx_default(monkeypatch):
    monkeypatch.delenv("JEVJUDGE_KEEPALIVE_S", raising=False)
    config = JevClientConfig.from_env()
    assert config.keepalive_s >= 60  # httpx's own default is 5 s, which made most bursty traffic cold
    client = JevClient(config)
    pool = client._http._transport._pool
    assert pool._keepalive_expiry == config.keepalive_s
    assert pool._max_keepalive_connections == config.pool_size


def test_keepalive_and_pool_size_are_configurable(monkeypatch):
    monkeypatch.setenv("JEVJUDGE_KEEPALIVE_S", "120")
    monkeypatch.setenv("JEVJUDGE_POOL_SIZE", "8")
    config = JevClientConfig.from_env()
    assert (config.keepalive_s, config.pool_size) == (120.0, 8)


async def test_warm_opens_the_requested_connections_without_calling_jev():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(404)  # any answer means the connection is up

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(JevClientConfig(base_url="https://jev.test", api_key="x"), http=http)
    assert await client.warm(3) == 3
    assert paths == ["/", "/", "/"]  # never the billed /v1/systemone endpoint


async def test_warm_never_raises_when_the_judge_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(JevClientConfig(base_url="https://jev.test", api_key="x"), http=http)
    assert await client.warm(4) == 0
