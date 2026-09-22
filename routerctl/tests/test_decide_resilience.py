"""Two production-hardening fixes, tested directly:

1. `recent_turn_window` (escalation only) actually slices the message list sent to the judge --
   it used to be a prompt-text hint only, with the full conversation still sent regardless.
2. A judge that can't be reached fails the *step*, not the whole decision -- verified against
   real switchyard.libsy behavior (see decide.py's comment) that this falls back to each policy's
   own safe default (complexity -> the capable tier; intent/escalation -> default_target) rather
   than raising. The fallback is never silent: `Decision.judge_error` is always set when it fires.
"""

from __future__ import annotations

import httpx
import pytest

from jevjudge.cascade import Judge, JudgeConfig
from jevjudge.client import JevClient, JevClientConfig

from routerctl.algorithms import compile_route_algorithm
from routerctl.decide import _windowed, decide
from routerctl.messages import to_libsy_messages
from routerctl.schema import Route


def route(name: str, policy: dict) -> Route:
    return Route.model_validate({"name": name, **policy})


def msgs(n: int) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]} for i in range(n)]


# ---------------------------------------------------------------- _windowed


def test_windowed_keeps_everything_under_the_limit():
    request = {"model": "auto", "messages": msgs(5)}
    result = _windowed(request, recent_turn_window=10)
    assert result["messages"] == request["messages"]


def test_windowed_keeps_only_the_last_n_messages_over_the_limit():
    request = {"model": "auto", "messages": msgs(10)}
    result = _windowed(request, recent_turn_window=3)
    assert [m["content"][0]["text"] for m in result["messages"]] == ["turn 7", "turn 8", "turn 9"]


def test_windowed_is_a_noop_with_no_window_configured():
    request = {"model": "auto", "messages": msgs(50)}
    result = _windowed(request, recent_turn_window=None)
    assert result is request  # not even copied -- every non-escalation policy takes this path


def test_windowed_does_not_mutate_the_original_request():
    request = {"model": "auto", "messages": msgs(10)}
    original_messages = request["messages"]
    _windowed(request, recent_turn_window=3)
    assert request["messages"] is original_messages
    assert len(original_messages) == 10


async def test_escalation_route_actually_truncates_a_long_transcript(monkeypatch):
    # Wire compile_route_algorithm's recent_turn_window through decide() end to end: build a
    # 40-turn transcript, a route configured for a window of 5, and assert decide() actually
    # calls the real truncation helper with that route's window, on that exact request -- the
    # thing the switchyard.libsy Algorithm object then runs on (it's a Rust object; its own
    # methods can't be monkeypatched, so this checks the enforcement point directly instead).
    import routerctl.decide as decide_module

    compiled = compile_route_algorithm(
        route("r", {"policy": "escalation", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}, "recent_turn_window": 5})
    )
    assert compiled.recent_turn_window == 5

    calls: list[tuple[int, int | None]] = []
    real_windowed = decide_module._windowed

    def spy_windowed(request, recent_turn_window):
        calls.append((len(request["messages"]), recent_turn_window))
        return real_windowed(request, recent_turn_window)

    monkeypatch.setattr(decide_module, "_windowed", spy_windowed)

    mock_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_typesafe_mock()), base_url="http://jev.mock")
    judge = Judge(JevClient(JevClientConfig(base_url="http://jev.mock", api_key="mock"), http=mock_http), JudgeConfig())
    request = {"model": "auto", "stream": False, "messages": to_libsy_messages([{"role": "user", "content": f"turn {i}"} for i in range(40)])}
    await decide(compiled, judge, request)

    assert calls == [(40, 5)]


def _typesafe_mock():
    from jevjudge.mocks import typesafe_mock_app

    return typesafe_mock_app()


# ------------------------------------------------------------- fail-open


@pytest.fixture
def unreachable_judge():
    # A real (non-mock) transport pointed at a closed local port: connection-refused is
    # near-instant, so this stays fast despite max_retries.
    jev = JevClient(JevClientConfig(base_url="http://127.0.0.1:1", api_key="x", timeout_s=1.0, max_retries=1))
    return Judge(jev, JudgeConfig())


def request_with(text: str) -> dict:
    return {"model": "auto", "stream": False, "messages": to_libsy_messages([{"role": "user", "content": text}])}


async def test_complexity_fails_open_to_the_capable_tier_when_judge_is_unreachable(unreachable_judge):
    compiled = compile_route_algorithm(
        route("r", {"policy": "complexity", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    decision = await decide(compiled, unreachable_judge, request_with("anything"))
    assert decision.selected_model == "m-strong"  # capable tier -- the safer default
    assert decision.judge_error is not None
    assert decision.judge_source is None
    assert decision.judge_confidence is None


async def test_intent_fails_open_to_default_target_when_judge_is_unreachable(unreachable_judge):
    compiled = compile_route_algorithm(
        route(
            "r",
            {
                "policy": "intent",
                "default": "analysis",
                "models": {
                    "extraction": {"id": "m-extract", "client": "openai", "description": "extract"},
                    "analysis": {"id": "m-analyze", "client": "openai", "description": "analyze"},
                },
            },
        )
    )
    decision = await decide(compiled, unreachable_judge, request_with("anything"))
    assert decision.bucket == "analysis"
    assert decision.selected_model == "m-analyze"
    assert decision.judge_error is not None


async def test_escalation_fails_open_to_continue_when_judge_is_unreachable(unreachable_judge):
    compiled = compile_route_algorithm(
        route("r", {"policy": "escalation", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    decision = await decide(compiled, unreachable_judge, request_with("same error again"))
    assert decision.bucket == "continue"  # escalation's own default_target, see algorithms.py
    assert decision.selected_model == "m-weak"
    assert decision.judge_error is not None


async def test_judge_error_is_none_on_a_successful_decision():
    mock_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_typesafe_mock()), base_url="http://jev.mock")
    judge = Judge(JevClient(JevClientConfig(base_url="http://jev.mock", api_key="mock"), http=mock_http), JudgeConfig())
    compiled = compile_route_algorithm(
        route("r", {"policy": "complexity", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    decision = await decide(compiled, judge, request_with("do a simple task"))
    assert decision.judge_error is None
    assert decision.judge_source == "jev"
