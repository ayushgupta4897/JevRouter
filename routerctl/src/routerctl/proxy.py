"""A thin reverse proxy whose upstream can be swapped atomically, at runtime.

This exists because Switchyard's server has no config hot-reload (confirmed against the
vendored source: no SIGHUP handler, no file watcher -- only Ctrl-C/SIGTERM graceful shutdown).
So a config change means a new `switchyard-server` process on a fresh config; this proxy is the
fixed point clients keep talking to (`routerctl serve`'s public port) while the supervisor
launches that new process, health-checks it, and only then calls `POST /_routerctl/swap` to
retarget every subsequent request. In-flight requests against the old upstream are unaffected;
the supervisor drains and stops the old process only after the swap.

Swap is local-only (loopback), by design: this proxy has no other authentication, so it must
never be reachable from anywhere the supervisor itself doesn't run.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@dataclass
class ProxyState:
    upstream: str | None = None  # e.g. "http://127.0.0.1:4001"; None until the first swap
    generation: int = 0  # bumped on every swap, for observability


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(http_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="routerctl proxy")
    app.state.proxy = ProxyState()
    app.state.http = http_client or httpx.AsyncClient(timeout=120.0)

    @app.get("/_routerctl/health")
    async def proxy_health() -> dict[str, object]:
        state: ProxyState = app.state.proxy
        return {"ok": state.upstream is not None, "upstream": state.upstream, "generation": state.generation}

    @app.post("/_routerctl/swap")
    async def swap(request: Request, body: dict[str, str]) -> JSONResponse:
        if not _is_loopback(request):
            return JSONResponse(status_code=403, content={"error": "swap is loopback-only"})
        upstream = body.get("upstream")
        if not upstream:
            return JSONResponse(status_code=422, content={"error": "body must be {'upstream': 'http://host:port'}"})
        state: ProxyState = app.state.proxy
        state.upstream = upstream.rstrip("/")
        state.generation += 1
        return JSONResponse({"ok": True, "upstream": state.upstream, "generation": state.generation})

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy_all(path: str, request: Request) -> Response:
        state: ProxyState = app.state.proxy
        if state.upstream is None:
            return JSONResponse(status_code=503, content={"error": "no backend is live yet"})
        upstream_url = f"{state.upstream}/{path}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        body = await request.body()
        client: httpx.AsyncClient = app.state.http

        upstream_request = client.build_request(
            request.method, upstream_url, params=request.query_params, headers=headers, content=body
        )
        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            return JSONResponse(status_code=502, content={"error": f"upstream unreachable: {error}"})

        response_headers = {k: v for k, v in upstream_response.headers.items() if k.lower() not in HOP_BY_HOP}

        async def body_stream():
            async for chunk in upstream_response.aiter_raw():
                yield chunk
            await upstream_response.aclose()

        return StreamingResponse(body_stream(), status_code=upstream_response.status_code, headers=response_headers)

    return app
