"""Turn an OpenAI chat-completions request into Jev `state`.

Jev accepts a string, object, or array as state. We send a small object:

    {"judge_instructions": <system prompt>, "conversation": [{"role": .., "content": ..}, ...]}

The judge's system prompt is the rubric the questions refer to (Switchyard's capability card
or trajectory-trouble patterns), so it belongs in state once rather than repeated per question.
The conversation is the window the router chose to show the judge, flattened to text.
"""

from __future__ import annotations

import json
from typing import Any

TRIM_MARKER = " ...[jevjudge trimmed] "


def _text_of_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind in {"text", "input_text", "output_text"} and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif kind == "image_url":
                    parts.append("[image]")
                elif kind == "refusal" and isinstance(block.get("refusal"), str):
                    parts.append(block["refusal"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


def _flatten_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = str(message.get("role") or "user")
    text = _text_of_content(message.get("content"))
    extras: list[str] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        extras.append(f"tool_call {fn.get('name', '?')}({fn.get('arguments', '')})")
    if role == "tool":
        entry: dict[str, Any] = {"role": "tool", "content": text}
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        return entry
    body = "\n".join(x for x in [text, *extras] if x)
    if not body:
        return None
    return {"role": role, "content": body}


def truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(limit - len(TRIM_MARKER), 20)
    head = keep * 2 // 3
    tail = keep - head
    return text[:head] + TRIM_MARKER + text[len(text) - tail :]


def build_state(
    messages: list[dict[str, Any]],
    *,
    rubric_placement: str = "state",
    max_state_chars: int = 100_000,
) -> tuple[dict[str, Any] | list[Any], str]:
    """Return (state, system_prompt).

    ``rubric_placement`` is ``state`` (default: system prompt included once in state under
    ``judge_instructions``), or ``drop`` (system prompt omitted from state; useful when the
    profile's question instructions already carry the rubric and you want the cheapest call).
    """
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"system", "developer"}:
            system_parts.append(_text_of_content(message.get("content")))
            continue
        flat = _flatten_message(message)
        if flat:
            conversation.append(flat)
    system_prompt = "\n\n".join(p for p in system_parts if p)

    # Budget: rubric gets at most a quarter; conversation the rest, trimmed oldest-first.
    if rubric_placement == "drop":
        rubric = ""
    else:
        rubric = truncate_middle(system_prompt, max_state_chars // 4)
    budget = max_state_chars - len(rubric)
    total = sum(len(m.get("content", "")) for m in conversation)
    while conversation and total > budget:
        if len(conversation) == 1:
            conversation[0]["content"] = truncate_middle(conversation[0]["content"], budget)
            break
        # Keep the opening task (index 0) if there is more than one message; drop the next oldest.
        dropped = conversation.pop(1 if len(conversation) > 1 else 0)
        total -= len(dropped.get("content", ""))

    state: dict[str, Any] = {"conversation": conversation}
    if rubric:
        state = {"judge_instructions": rubric, "conversation": conversation}
    return state, system_prompt
