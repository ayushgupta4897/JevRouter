"""Adapt plain OpenAI-chat-shaped messages to libsy's normalized request shape.

`switchyard.libsy`'s `Algorithm.run_stream` rejects plain string content outright (verified: it
raises `invalid type: string "...", expected internally tagged enum ContentBlock`) -- every
message's `content` must already be a list of typed blocks (`{"type": "text", "text": ...}`).
An AI Gateway sending an ordinary OpenAI chat-completions body has plain string content, so this
is the one normalization this project's decision endpoint needs to do before calling libsy.

Tool traffic has to survive the trip, not just text. `auto` (Switchyard's stage_router) decides
*entirely* from an agent's tool-call history -- repeated failures, edits landing, exploration --
and the escalation judge reads it too. An earlier version of this adapter kept only `content`,
silently dropping every `tool_calls` entry and flattening tool results to anonymous text, so
`auto` never saw a single signal and picked the efficient tier for everything. OpenAI's shapes
map onto libsy's `ToolCall` / `ToolResult` blocks (vendor/switchyard/crates/protocol/src/llm.rs).
"""

from __future__ import annotations

import json
from typing import Any


def _arguments(raw: Any) -> Any:
    # OpenAI sends function arguments as a JSON string; libsy wants the parsed value.
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw if raw is not None else {}


def _part(part: Any) -> dict[str, Any] | None:
    """One OpenAI content part as a libsy block, or None for kinds routing can't use."""
    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if kind in {"text", "input_text", "output_text"} and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    if kind == "image_url":
        image = part.get("image_url") or {}
        url = image.get("url") if isinstance(image, dict) else image
        if isinstance(url, str):
            detail = image.get("detail") if isinstance(image, dict) else None
            return {"type": "image", "source": {"type": "url", "data": {"url": url, "detail": detail}}}
    if kind == "refusal" and isinstance(part.get("refusal"), str):
        return {"type": "refusal", "text": part["refusal"]}
    if kind in {"tool_call", "tool_result", "image", "reasoning"}:
        return part  # already libsy-shaped
    return None  # audio, files, provider-specific parts: not routing signals, and libsy would reject them


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return [block for block in (_part(p) for p in content) if block is not None]
    return []


def _convert(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role", "user")
    if role == "function":  # legacy OpenAI name for a tool result
        role = "tool"
    blocks = _as_blocks(message.get("content"))

    if role == "tool":
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": str(message.get("tool_call_id") or message.get("name") or ""),
            "content": blocks,
            "is_error": message["is_error"] if isinstance(message.get("is_error"), bool) else None,
        }
        return {"role": "tool", "content": [result]}

    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        blocks.append({
            "type": "tool_call",
            "id": str(call.get("id") or ""),
            "name": str(function.get("name") or call.get("name") or ""),
            "arguments": _arguments(function.get("arguments", call.get("arguments"))),
        })
    return {"role": role, "content": blocks or [{"type": "text", "text": ""}]}


def to_libsy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_convert(message) for message in messages]


__all__ = ["to_libsy_messages"]
