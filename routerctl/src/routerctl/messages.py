"""Adapt plain OpenAI-chat-shaped messages to libsy's normalized request shape.

`switchyard.libsy`'s `Algorithm.run_stream` rejects plain string content outright (verified: it
raises `invalid type: string "...", expected internally tagged enum ContentBlock`) -- every
message's `content` must already be a list of typed blocks (`{"type": "text", "text": ...}`).
An AI Gateway sending an ordinary OpenAI chat-completions body has plain string content, so this
is the one normalization this project's decision endpoint needs to do before calling libsy.
"""

from __future__ import annotations

from typing import Any


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content  # assume already block-shaped; passed through as-is
    return [{"type": "text", "text": ""}]


def to_libsy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": message.get("role", "user"), "content": _as_blocks(message.get("content"))} for message in messages]


__all__ = ["to_libsy_messages"]
