"""Serve a Switchyard `libsy` judge call with Jev in-process (no HTTP sidecar).

When you embed Switchyard's routing algorithms in your own Python harness (README "Path 2"),
`Algorithm.run_stream` yields `Step.CallModel` for the judge. Hand that call to
`serve_judge_call`; it converts libsy's normalized request to the OpenAI shape `Judge`
understands, asks Jev, and responds with a libsy aggregate response carrying the JSON verdict.
"""

from __future__ import annotations

import json
from typing import Any

from .cascade import Judge


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif kind == "tool_call":
            parts.append(f"tool_call {block.get('name', '?')}({json.dumps(block.get('arguments', {}))})")
        elif kind == "tool_result":
            parts.append(_blocks_text(block.get("content")))
        elif kind in {"refusal"} and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(p for p in parts if p)


def libsy_request_to_openai(request: dict[str, Any]) -> dict[str, Any]:
    """libsy normalized request -> OpenAI chat-completions request (messages + response_format)."""
    messages: list[dict[str, Any]] = []
    for instruction in request.get("instructions") or []:
        messages.append({"role": str(instruction.get("role") or "system"), "content": _blocks_text(instruction.get("content"))})
    for message in request.get("messages") or []:
        role = str(message.get("role") or "user")
        messages.append({"role": "tool" if role == "tool" else role, "content": _blocks_text(message.get("content"))})
    output = request.get("output") or {}
    return {
        "model": request.get("model") or "jev-latest",
        "messages": messages,
        "response_format": output.get("response_format"),
        "max_tokens": output.get("max_output_tokens"),
        "stream": False,
    }


def verdict_to_libsy_response(model: str, openai_response: dict[str, Any]) -> dict[str, Any]:
    content = openai_response["choices"][0]["message"]["content"]
    usage = openai_response.get("usage") or {}
    return {
        "model": model,
        "outputs": [{"role": "assistant", "content": [{"type": "text", "text": content}]}],
        "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
    }


async def serve_judge_call(judge: Judge, call: Any, *, LlmResponse: Any) -> dict[str, Any]:
    """Answer a libsy `ModelCall` with Jev. Returns jevjudge's metadata for logging."""
    openai_request = libsy_request_to_openai(call.request)
    outcome = await judge.judge(openai_request)
    call.respond(LlmResponse.Agg(verdict_to_libsy_response(call.models[0], outcome.response)))
    return {"source": outcome.source, "confidence": outcome.confidence, "latency_ms": outcome.latency_ms, "schema": outcome.schema_name}
