"""Offline stand-ins so the whole routing stack can be exercised without any API key.

* ``typesafe`` mock: serves ``POST /v1/systemone`` with deterministic, keyword-driven answers.
  It is NOT a model. It exists to prove the plumbing (schema compile -> Jev wire -> verdict ->
  Switchyard policy -> target) end to end, and to make tests hermetic.
* ``upstream`` mock: serves ``POST /v1/chat/completions`` and echoes which model was called,
  so a routing decision is visible in the answer text. Supports streaming.

Run:  jevjudge-mock typesafe --port 8091
      jevjudge-mock upstream --port 8092
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Keyword cues the mock reads from the state. Tests and the e2e script use these.
HARD_CUES = ("[hard]", "[stuck]", "same error", "doomed", "loop")
EASY_CUES = ("[easy]", "trivial", "hello")
ROUTE_CUE = re.compile(r"route:([A-Za-z0-9_.-]+)")


def _state_text(state: Any) -> str:
    return json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state


def _jitter(seed: str) -> float:
    return (int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % 1000) / 1000.0 * 0.1


def _noul(name: str, instructions: str, text: str) -> float:
    lowered = text.lower()
    hard = any(c in lowered for c in HARD_CUES)
    easy = any(c in lowered for c in EASY_CUES)
    # `escalate`-style questions are true when the run looks stuck; `p_solve`-style questions
    # (success probability) are the opposite. Anything else: middling.
    inverted = "success" in instructions.lower() or name.endswith("p_solve")
    if hard:
        p = 0.12 if inverted else 0.88
    elif easy:
        p = 0.92 if inverted else 0.08
    else:
        p = 0.55 if inverted else 0.45
    return round(min(0.99, max(0.01, p + _jitter(name + text[:64]) - 0.05)), 3)


def _choice(name: str, criteria: dict[str, Any], text: str) -> dict[str, Any]:
    labels = list(criteria.keys())
    lowered = text.lower()
    pick = None
    for m in ROUTE_CUE.finditer(text):
        if m.group(1) in criteria:
            pick = m.group(1)
            break
    if pick is None:
        for label in labels:
            if f"[{label.lower()}]" in lowered:
                pick = label
                break
    confidence = 0.93 if pick else 0.42
    if pick is None:
        # Capability rules: hard -> LIM-2, easy -> SUP-1, otherwise UNC-1; generic: first label.
        if any(c in lowered for c in HARD_CUES) and "LIM-2" in criteria:
            pick, confidence = "LIM-2", 0.8
        elif any(c in lowered for c in EASY_CUES) and "SUP-1" in criteria:
            pick, confidence = "SUP-1", 0.85
        elif "UNC-1" in criteria:
            pick, confidence = "UNC-1", 0.5
        else:
            pick = labels[0]
    rest = max(0.0, 1.0 - confidence)
    probs = {label: (confidence if label == pick else rest / max(1, len(labels) - 1)) for label in labels}
    return {"type": "choice", "choice": pick, "confidence": confidence, "probabilities": probs}


def _score(criteria: list[Any], text: str) -> dict[str, Any]:
    n = len(criteria)
    lowered = text.lower()
    idx = n - 1 if any(c in lowered for c in HARD_CUES) else (0 if any(c in lowered for c in EASY_CUES) else n // 2)
    probs = {str(i): (0.8 if i == idx else 0.2 / max(1, n - 1)) for i in range(n)}
    score = sum(i * p for i, p in ((int(k), v) for k, v in probs.items()))
    return {"type": "score", "score": round(score, 3), "confidence": 0.8, "legend": {str(i): c for i, c in enumerate(criteria)}, "probabilities": probs}


def typesafe_mock_app() -> FastAPI:
    app = FastAPI(title="mock typesafe /v1/systemone")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"models": [{"name": "jev-mock", "description": "Deterministic keyword mock. Not a model.", "release_date": "2026-09-21"}]}

    @app.post("/v1/systemone")
    @app.post("/api/alpha/decisions")
    async def systemone(request: Request) -> Any:
        body = await request.json()
        questions = body.get("questions") or {}
        if not isinstance(questions, dict) or not questions:
            return JSONResponse(status_code=422, content={"detail": [{"loc": ["body", "questions"], "msg": "Field required", "type": "missing"}]})
        text = _state_text(body.get("state"))
        answers: dict[str, Any] = {}
        for name, q in questions.items():
            kind = q.get("type")
            instr = q.get("instructions")
            instr_text = instr if isinstance(instr, str) else json.dumps(instr or "")
            if kind == "noul":
                answers[name] = {"type": "noul", "noul": _noul(name, instr_text, text)}
            elif kind == "choice":
                answers[name] = _choice(name, q.get("criteria") or {}, text)
            elif kind == "score":
                answers[name] = _score(q.get("criteria") or [], text)
            else:
                return JSONResponse(status_code=422, content={"detail": [{"loc": ["body", "questions", name, "type"], "msg": "unknown type", "type": "value_error"}]})
        tokens = max(1, len(text) // 4 + len(json.dumps(questions)) // 4)
        return {"model": "jev-mock", "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": len(answers)}}

    return app


def upstream_mock_app() -> FastAPI:
    app = FastAPI(title="mock openai upstream")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        model = str(body.get("model") or "unknown")
        last_user = ""
        for m in reversed(body.get("messages") or []):
            if m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else json.dumps(c)[:80]
                break
        content = f"[served-by:{model}] echo: {last_user[:80]}"
        base = {"id": f"chatcmpl-mock-{int(time.time()*1000)}", "created": int(time.time()), "model": model}
        if body.get("stream"):
            def gen():
                yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}) + "\n\n"
                yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {**base, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="jevjudge offline mocks")
    parser.add_argument("kind", choices=["typesafe", "upstream"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    import uvicorn

    app = typesafe_mock_app() if args.kind == "typesafe" else upstream_mock_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
