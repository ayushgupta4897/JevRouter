#!/usr/bin/env python3
"""Embed Switchyard's routing in a Python harness and serve its judge with Jev in-process.

No HTTP sidecar: the harness drives `Algorithm.run_stream`, answers the judge `CallModel`
with `jevjudge.libsy.serve_judge_call`, and makes the answer call itself. Requires the
Switchyard Python bindings (`pip install git+https://github.com/NVIDIA-NeMo/Switchyard.git`).

    TYPESAFE_API_KEY=... python examples/embedded_libsy.py            # real Jev
    TYPESAFE_BASE_URL=http://127.0.0.1:8091 TYPESAFE_API_KEY=mock python examples/embedded_libsy.py   # mock

The answer models here are stubs that echo which tier served the turn.
"""

from __future__ import annotations

import asyncio
import json
import sys

from switchyard.libsy import (
    CustomClassifierConfig,
    EscalationClassifierConfig,
    LlmClassifierConfig,
    LlmResponse,
    Step,
    TaskClassifierConfig,
    algorithms,
)

from jevjudge.cascade import build_judge_from_env
from jevjudge.libsy import serve_judge_call

MODELS = {"judge": ["jev-latest"], "capable": ["strong-model"], "efficient": ["weak-model"], "any": ["strong-model", "weak-model"]}
CUSTOM_MODELS = {
    "judge": ["jev-latest"],
    "fast": ["fast-model"], "balanced": ["balanced-model"], "reasoning": ["reasoning-model"], "premium": ["premium-model"],
    "any": ["fast-model", "balanced-model", "reasoning-model", "premium-model"],
}


def user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


async def answer(model: str, request: dict) -> LlmResponse.Agg:
    """Stand-in for your real model client (OpenAI/Anthropic SDK, LiteLLM, ...)."""
    last = next((m for m in reversed(request["messages"]) if m["role"] == "user"), {"content": []})
    text = " ".join(b.get("text", "") for b in last["content"] if isinstance(b, dict))[:60]
    return LlmResponse.Agg({"model": model, "outputs": [{"role": "assistant", "content": [{"type": "text", "text": f"[served-by:{model}] {text}"}]}]})


async def route(judge, algorithm, request: dict, models: dict, session_id: str | None = None) -> str:
    # Session identity (escalation streaks, classify affinity) travels as a header, exactly as
    # the standalone server reads `x-switchyard-session-id`.
    headers = {"x-switchyard-session-id": session_id} if session_id else None
    async for step in algorithm.run_stream(request, models, headers=headers):
        match step:
            case Step.CallModel(call):
                if call.models[0] == "jev-latest":
                    meta = await serve_judge_call(judge, call, LlmResponse=LlmResponse)
                    print(f"    judge: {meta['schema']} via {meta['source']} confidence={meta['confidence']:.2f} {meta['latency_ms']:.0f}ms")
                else:  # escalation mode calls the weak model before judging its reply
                    call.respond(await answer(call.models[0], call.request))
            case Step.Done(outcome):
                model = outcome.selected_model_ids[0]
                response = outcome.response or await answer(model, outcome.request)
                match response:
                    case LlmResponse.Agg(agg):
                        return agg["outputs"][0]["content"][0]["text"]
    raise RuntimeError("algorithm ended without Done")


async def main() -> None:
    judge = build_judge_from_env()
    try:
        print("== capability mode")
        alg = algorithms.llm_classifier(LlmClassifierConfig.capability(config=TaskClassifierConfig(0.5, threshold_step=0.1)))
        for task in ["[easy] rename a variable in one file", "[hard] reproduce the undocumented legacy checksum exactly"]:
            print("  ", task, "->", await route(judge, alg, {"model": "auto", "stream": False, "messages": [user(task)]}, MODELS))

        print("== escalation mode (same session object -> streak carries across turns)")
        alg = algorithms.llm_classifier(LlmClassifierConfig.escalation(config=EscalationClassifierConfig(confirmations=2)))
        turns = [user("fix the failing test")]
        for i in range(3):
            turns += [assistant("retrying the same command"), user("[stuck] same error again")]
            req = {"model": "auto", "stream": False, "messages": turns}
            print(f"   turn {i+1} ->", await route(judge, alg, req, MODELS, session_id="esc-demo"))

        print("== custom mode (4-way Choice)")
        prompt = (
            "Choose the cheapest model group that can complete the request.\n"
            "- fast: greetings, lookups, one-line edits.\n- balanced: ordinary coding and writing.\n"
            "- reasoning: multi-step math, algorithms, proofs.\n- premium: high-stakes architecture or security.\n"
            "Return JSON matching the response schema supplied with the request."
        )
        schema = {"type": "object", "properties": {"decision": {"type": "object", "properties": {"target": {"type": "string", "enum": ["fast", "balanced", "reasoning", "premium"]}}, "required": ["target"], "additionalProperties": False}}, "required": ["decision"], "additionalProperties": False}
        alg = algorithms.llm_classifier(LlmClassifierConfig.custom(default_target="balanced", config=CustomClassifierConfig(prompt, schema, "/decision/target")))
        for task in ["route:fast hi", "route:reasoning prove sqrt(2) is irrational"]:
            print("  ", task, "->", await route(judge, alg, {"model": "auto", "stream": False, "messages": [user(task)]}, CUSTOM_MODELS))
    finally:
        await judge.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
