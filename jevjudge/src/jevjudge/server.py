"""OpenAI-compatible HTTP facade for the Jev judge.

    POST /v1/chat/completions   the judge (structured output in, JSON verdict out)
    GET  /v1/models             one model: the configured judge name
    GET  /v1/stats              counters (Jev calls, fallbacks, abstains, latency)
    GET  /health

Run:  jevjudge --port 8090      (or: python -m jevjudge.server --port 8090)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .cascade import CompileError, JevError, Judge, build_judge_from_env

log = logging.getLogger("jevjudge")


def create_app(judge: Judge | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _ensure_judge(app)
        try:
            yield
        finally:
            await app.state.judge.aclose()

    app = FastAPI(title="jevjudge", lifespan=lifespan)
    app.state.judge = judge  # None means: build from the environment on first use

    def _ensure_judge(app: FastAPI) -> Judge:
        if getattr(app.state, "judge", None) is None:
            app.state.judge = build_judge_from_env()
        return app.state.judge

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        j: Judge = _ensure_judge(request.app)
        return {"object": "list", "data": [{"id": j.jev.config.model, "object": "model", "owned_by": "jevjudge"}]}

    @app.get("/v1/stats")
    async def stats(request: Request) -> dict[str, Any]:
        j: Judge = _ensure_judge(request.app)
        return j.stats.snapshot()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        j: Judge = _ensure_judge(request.app)
        body = await request.json()
        try:
            outcome = await j.judge(body)
        except CompileError as error:
            return JSONResponse(status_code=422, content={"error": {"message": f"jevjudge cannot compile this verdict schema: {error}", "type": "invalid_request_error"}})
        except JevError as error:
            return JSONResponse(status_code=502, content={"error": {"message": str(error), "type": "upstream_error"}})
        headers = {
            "x-jevjudge-source": outcome.source,
            "x-jevjudge-confidence": "" if outcome.confidence is None else f"{outcome.confidence:.4f}",
            "x-jevjudge-latency-ms": f"{outcome.latency_ms:.1f}",
            "x-jevjudge-schema": outcome.schema_name,
        }
        log.info("judge schema=%s source=%s confidence=%s latency_ms=%.1f", outcome.schema_name, outcome.source, headers["x-jevjudge-confidence"], outcome.latency_ms)
        if body.get("stream"):
            return StreamingResponse(_sse(outcome.response), media_type="text/event-stream", headers=headers)
        return JSONResponse(content=outcome.response, headers=headers)

    return app


def _sse(response: dict[str, Any]) -> Any:
    """Emit the buffered verdict as a minimal chat-completions SSE stream."""
    content = response["choices"][0]["message"]["content"]
    base = {"id": response["id"], "object": "chat.completion.chunk", "created": response["created"], "model": response["model"]}

    def gen():
        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": response.get("usage")}) + "\n\n"
        yield "data: [DONE]\n\n"

    return gen()


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Jev-backed OpenAI-compatible judge")
    parser.add_argument("--host", default=os.environ.get("JEVJUDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JEVJUDGE_PORT", "8090")))
    parser.add_argument("--log-level", default=os.environ.get("JEVJUDGE_LOG_LEVEL", "info"))
    args = parser.parse_args(argv)
    import uvicorn

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("jevjudge.server:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
