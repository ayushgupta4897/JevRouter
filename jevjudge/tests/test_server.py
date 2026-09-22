"""End-to-end through ASGI: Switchyard-shaped judge request -> jevjudge -> mock Jev -> verdict."""

from __future__ import annotations

import json

import httpx
import pytest

from jevjudge.cascade import Judge, JudgeConfig
from jevjudge.client import JevClient, JevClientConfig
from jevjudge.mocks import typesafe_mock_app, upstream_mock_app
from jevjudge.server import create_app

from test_compiler import CAPABILITY_PROMPT, CAPABILITY_RF, ESCALATION_RF


def _judge(mock_transport: httpx.AsyncClient, **cfg) -> Judge:
    jev = JevClient(JevClientConfig(base_url="http://jev.mock", api_key="mock"), http=mock_transport)
    return Judge(jev, JudgeConfig(**cfg))


@pytest.fixture
def jev_http():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=typesafe_mock_app()), base_url="http://jev.mock")


def _capability_request(task: str) -> dict:
    return {
        "model": "jev-latest",
        "messages": [{"role": "system", "content": CAPABILITY_PROMPT}, {"role": "user", "content": task}],
        "max_tokens": 4096,
        "response_format": CAPABILITY_RF,
    }


async def test_capability_verdict_round_trip(jev_http):
    app = create_app(_judge(jev_http))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://judge") as c:
        r = await c.post("/v1/chat/completions", json=_capability_request("[easy] print hello world"))
        assert r.status_code == 200, r.text
        assert r.headers["x-jevjudge-source"] == "jev"
        body = r.json()
        verdict = json.loads(body["choices"][0]["message"]["content"])
        assert set(verdict) == {"crux", "primary_rule", "capability_boundary", "p_solve"}
        assert verdict["p_solve"] > 0.8
        assert verdict["primary_rule"] == "SUP-1" and verdict["capability_boundary"] == "supported"
        assert body["usage"]["prompt_tokens"] > 0
        assert body["jevjudge"]["evidence"]["p_solve"]["type"] == "noul"

        r = await c.post("/v1/chat/completions", json=_capability_request("[hard] reverse engineer the undocumented binary format"))
        verdict = json.loads(r.json()["choices"][0]["message"]["content"])
        assert verdict["p_solve"] < 0.3 and verdict["capability_boundary"] == "unsupported"

        stats = (await c.get("/v1/stats")).json()
        assert stats["jev_calls"] == 2 and stats["by_schema"]["CapabilityClassifierDecision"] == 2


async def test_escalation_verdict_and_streaming(jev_http):
    app = create_app(_judge(jev_http))
    req = {
        "model": "jev-latest",
        "stream": True,
        "messages": [{"role": "system", "content": "You are an escalation judge."}, {"role": "user", "content": "Conversation turn 5\n[assistant] retry\n[tool] same error again"}],
        "response_format": ESCALATION_RF,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://judge") as c:
        r = await c.post("/v1/chat/completions", json=req)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
        assert json.loads(content) == {"escalate": True, "reason": json.loads(content)["reason"]}
        assert json.loads(content)["reason"].startswith("jevjudge: p_escalate=")


async def test_low_confidence_abstain_returns_non_json(jev_http):
    app = create_app(_judge(jev_http, min_confidence=0.95, low_confidence_action="abstain"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://judge") as c:
        r = await c.post("/v1/chat/completions", json=_capability_request("ambiguous request with no cues"))
        assert r.headers["x-jevjudge-source"] == "abstain"
        content = r.json()["choices"][0]["message"]["content"]
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)  # Switchyard treats this as fail-open -> default target


async def test_low_confidence_falls_back_to_llm_judge(jev_http):
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream_mock_app()), base_url="http://llm.mock")
    judge = _judge(jev_http, min_confidence=0.95, low_confidence_action="fallback", fallback_base_url="http://llm.mock/v1", fallback_model="gemini-flash-lite")
    judge._http = upstream
    app = create_app(judge)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://judge") as c:
        r = await c.post("/v1/chat/completions", json=_capability_request("ambiguous request with no cues"))
        assert r.headers["x-jevjudge-source"] == "fallback"
        body = r.json()
        assert body["jevjudge"]["fallback_model"] == "gemini-flash-lite"
        assert "[served-by:gemini-flash-lite]" in body["choices"][0]["message"]["content"]


async def test_uncompilable_schema_is_422(jev_http):
    app = create_app(_judge(jev_http))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://judge") as c:
        r = await c.post("/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 422
