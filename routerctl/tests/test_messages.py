"""The OpenAI-chat -> libsy adapter. Tool traffic has to survive it: `auto` routes on nothing
else, and an earlier version dropped it silently (see routerctl/messages.py)."""

from __future__ import annotations

import json

import httpx
import pytest

from jevjudge.cascade import Judge, JudgeConfig
from jevjudge.client import JevClient, JevClientConfig
from jevjudge.mocks import typesafe_mock_app

from routerctl.algorithms import compile_route_algorithm
from routerctl.decide import decide
from routerctl.messages import to_libsy_messages
from routerctl.schema import Route


def tool_turn(i: int, name: str, args: dict, result: str) -> list[dict]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": result},
    ]


def test_plain_text_becomes_a_text_block():
    assert to_libsy_messages([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_tool_calls_become_tool_call_blocks_with_parsed_arguments():
    [assistant, tool] = to_libsy_messages(tool_turn(0, "bash", {"command": "pytest"}, "1 failed"))
    assert assistant["content"] == [{"type": "tool_call", "id": "c0", "name": "bash", "arguments": {"command": "pytest"}}]
    assert tool == {"role": "tool", "content": [{"type": "tool_result", "tool_call_id": "c0", "content": [{"type": "text", "text": "1 failed"}], "is_error": None}]}


def test_assistant_text_and_tool_calls_are_both_kept():
    [msg] = to_libsy_messages([{"role": "assistant", "content": "Running tests.", "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": "{}"}}]}])
    assert [b["type"] for b in msg["content"]] == ["text", "tool_call"]


def test_unparseable_arguments_are_kept_as_a_string():
    [msg] = to_libsy_messages([{"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "not json"}}]}])
    assert msg["content"][0]["arguments"] == "not json"


def test_image_parts_convert_and_unsupported_parts_are_dropped():
    [msg] = to_libsy_messages([{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        {"type": "input_audio", "input_audio": {"data": "..", "format": "wav"}},
    ]}])
    assert [b["type"] for b in msg["content"]] == ["text", "image"]


def test_empty_content_still_yields_a_valid_message():
    [msg] = to_libsy_messages([{"role": "assistant", "content": None}])
    assert msg["content"] == [{"type": "text", "text": ""}]


# -------------------------------------------------------------- the real stage_router, end to end


@pytest.fixture
def judge():
    mock_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=typesafe_mock_app()), base_url="http://jev.mock")
    return Judge(JevClient(JevClientConfig(base_url="http://jev.mock", api_key="mock"), http=mock_http), JudgeConfig())


AUTO = compile_route_algorithm(Route.model_validate({"name": "r", "policy": "auto", "models": {"efficient": {"id": "m-eff", "client": "openai"}, "capable": {"id": "m-cap", "client": "openai"}}}))


async def route_of(judge, messages: list[dict]) -> str:
    decision = await decide(AUTO, judge, {"model": "r", "stream": False, "messages": to_libsy_messages(messages)})
    return decision.selected_model


async def test_auto_escalates_an_agent_stuck_on_the_same_error(judge):
    messages = [{"role": "user", "content": "Fix the failing build."}]
    for i in range(3):
        messages += tool_turn(i, "bash", {"command": "python -m app"}, "Traceback (most recent call last):\nModuleNotFoundError: No module named 'serde'")
    assert await route_of(judge, messages) == "m-cap"  # was m-eff while tool history was being dropped


async def test_auto_keeps_a_productive_agent_on_the_efficient_tier(judge):
    messages = [{"role": "user", "content": "Add the /health endpoint and tests."}]
    messages += tool_turn(0, "write_file", {"path": "src/server.py", "content": "..."}, "wrote 42 lines")
    messages += tool_turn(1, "edit", {"path": "src/server.py", "old": "a", "new": "b"}, "edited")
    messages += tool_turn(2, "bash", {"command": "pytest -q"}, "3 passed in 0.12s")
    assert await route_of(judge, messages) == "m-eff"


async def test_images_no_longer_break_a_decision(judge):
    messages = [{"role": "user", "content": [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]
    assert await route_of(judge, messages) in {"m-eff", "m-cap"}
