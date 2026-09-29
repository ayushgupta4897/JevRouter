"""End-to-end decide() against the free, offline mock Jev -- proves every compiled policy
reaches a decision without ever asking to call a real target model (the whole point of this
project's pivot to a decision-only router; see docs/DECISION.md)."""

from __future__ import annotations

import httpx
import pytest

from jevjudge.cascade import Judge, JudgeConfig
from jevjudge.client import JevClient, JevClientConfig
from jevjudge.mocks import typesafe_mock_app

from routerctl.algorithms import compile_route_algorithm
from routerctl.decide import decide
from routerctl.messages import to_libsy_messages
from routerctl.schema import Route


@pytest.fixture
def judge():
    mock_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=typesafe_mock_app()), base_url="http://jev.mock")
    jev = JevClient(JevClientConfig(base_url="http://jev.mock", api_key="mock"), http=mock_http)
    return Judge(jev, JudgeConfig())


def route(name: str, policy: dict) -> Route:
    return Route.model_validate({"name": name, **policy})


def request(text: str) -> dict:
    return {"model": "auto", "stream": False, "messages": to_libsy_messages([{"role": "user", "content": text}])}


async def test_auto_decides_with_no_judge_call_at_all(judge):
    compiled = compile_route_algorithm(route("r", {"policy": "auto", "models": {"efficient": {"id": "m-eff", "client": "openai"}, "capable": {"id": "m-cap", "client": "openai"}}}))
    decision = await decide(compiled, judge, request("hi"))
    assert decision.selected_model in {"m-eff", "m-cap"}
    assert decision.judge_source is None  # stage_router with no classifier makes zero LLM calls


async def test_complexity_decides_via_judge_only(judge):
    compiled = compile_route_algorithm(
        route("r", {"policy": "complexity", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    decision = await decide(compiled, judge, request("do a simple task"))
    assert decision.selected_model in {"m-weak", "m-strong"}
    assert decision.selected_client == "openai"
    assert decision.judge_source == "jev"
    assert decision.judge_confidence is not None


async def test_intent_decides_a_bucket_via_judge_only(judge):
    compiled = compile_route_algorithm(
        route(
            "r",
            {
                "policy": "intent",
                "default": "extraction",
                "models": {
                    "extraction": {"id": "m-extract", "client": "openai", "description": "pull one field verbatim from text"},
                    "analysis": {"id": "m-analyze", "client": "openai", "description": "judge sentiment or summarize"},
                },
            },
        )
    )
    decision = await decide(compiled, judge, request("what account number is mentioned?"))
    assert decision.bucket in {"extraction", "analysis"}
    assert decision.selected_model == ({"extraction": "m-extract", "analysis": "m-analyze"}[decision.bucket])
    assert decision.judge_source == "jev"


async def test_escalation_decides_continue_or_escalate_from_transcript_alone(judge):
    # No real model is ever called to "see how it's doing" -- the judge reads the transcript
    # already in the request, exactly like intent's custom classifier.
    compiled = compile_route_algorithm(
        route("r", {"policy": "escalation", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    decision = await decide(compiled, judge, request("same error again, third time in a row"))
    assert decision.bucket in {"continue", "escalate"}
    assert decision.selected_model == ({"continue": "m-weak", "escalate": "m-strong"}[decision.bucket])
    assert decision.judge_source == "jev"


async def test_fallback_model_ids_carry_the_rest_of_the_selection(judge):
    compiled = compile_route_algorithm(
        route("r", {"policy": "complexity", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    decision = await decide(compiled, judge, request("anything"))
    assert decision.selected_model not in decision.fallback_model_ids
    assert set(decision.fallback_model_ids) <= {"m-weak", "m-strong"}


async def test_decision_carries_each_models_extra_body_to_the_gateway(judge):
    # decision-only mode never makes the call, so provider pinning / reasoning effort set on a
    # model in YAML has to travel with the decision or it is silently lost
    pin = {"provider": {"order": ["fireworks"], "allow_fallbacks": False}}
    effort = {"reasoning": {"effort": "high"}}
    compiled = compile_route_algorithm(route("r", {"policy": "complexity", "models": {
        "weak": {"id": "m-weak", "client": "openai", "extra_body": pin},
        "strong": {"id": "m-strong", "client": "openai", "extra_body": effort},
    }}))
    decision = await decide(compiled, judge, request("anything"))
    expected = {"m-weak": pin, "m-strong": effort}
    assert decision.selected_extra_body == expected[decision.selected_model]
    assert decision.fallback_extra_bodies == [expected[m] for m in decision.fallback_model_ids]


async def test_escalation_judge_sees_the_tool_turns_not_just_the_opening_task(judge):
    # Regression: Switchyard's custom classifier shows the judge only the opening task and the
    # latest *user* message unless recent_turn_window reaches it. On a real agent transcript
    # the escalation judge never saw a single tool call or failure (DECISION.md section 15).
    import json as _json
    compiled = compile_route_algorithm(route("r", {"policy": "escalation", "models": {
        "weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}}))
    messages = [{"role": "user", "content": "Fix the failing build."}]
    for i in range(3):
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
             "function": {"name": "bash", "arguments": _json.dumps({"command": "python -m app"})}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"Traceback {i}: ModuleNotFoundError: No module named 'serde'"},
        ]
    seen: list[str] = []
    real_judge = judge.judge

    async def spy(request):
        seen.extend(str(m.get("content")) for m in request["messages"])
        return await real_judge(request)

    judge.judge = spy
    await decide(compiled, judge, {"model": "r", "stream": False, "messages": to_libsy_messages(messages)})
    transcript = "\n".join(seen)
    assert all(f"Traceback {i}" in transcript for i in range(3))
    assert "tool_call bash" in transcript
