"""Drive one decision-only run of a compiled route's algorithm to a `Decision` -- never a
completion. The one real network call this makes is to the shared jevjudge classifier (cheap,
and the whole point of this project); the target model itself is never invoked here."""

from __future__ import annotations

import time
from dataclasses import dataclass

from switchyard.libsy import LlmResponse, Step

from jevjudge.cascade import Judge
from jevjudge.libsy import serve_judge_call

from .algorithms import CompiledRoute


class UnexpectedRealCallError(RuntimeError):
    """A compiled algorithm asked to call a real (non-judge) target model. Every policy in
    `algorithms.py` is built to never do this -- if this fires, that invariant broke, which is a
    bug in this project, not a runtime condition a caller should route around."""


@dataclass
class Decision:
    route: str
    policy: str
    selected_model: str
    selected_client: str
    bucket: str | None
    fallback_model_ids: list[str]
    fallback_clients: list[str]
    judge_source: str | None
    judge_confidence: float | None
    latency_ms: float
    outcome_id: str | None


async def decide(compiled: CompiledRoute, judge: Judge, request: dict, *, headers: dict[str, str] | None = None) -> Decision:
    started = time.perf_counter()
    judge_id = (compiled.models.get("judge") or [None])[0]
    judge_meta: dict | None = None

    async for step in compiled.algorithm.run_stream(request, compiled.models, headers=headers):
        match step:
            case Step.CallModel(call):
                requested = call.models[0] if call.models else None
                if judge_id is not None and requested == judge_id:
                    judge_meta = await serve_judge_call(judge, call, LlmResponse=LlmResponse)
                    continue
                raise UnexpectedRealCallError(
                    f"route {compiled.route.name!r} asked to call real target {call.models!r}; "
                    "every compiled policy here is decision-only and should never reach this"
                )
            case Step.Done(outcome):
                latency_ms = (time.perf_counter() - started) * 1000
                selected = outcome.selected_model_ids
                if not selected:
                    raise RuntimeError(f"route {compiled.route.name!r} produced an empty selection")
                primary_id, fallback_ids = selected[0], selected[1:]
                primary_ref = compiled.model_by_id[primary_id]
                fallback_refs = [compiled.model_by_id[i] for i in fallback_ids if i in compiled.model_by_id]
                bucket = compiled.bucket_by_model_id.get(primary_id) if compiled.bucket_by_model_id else None
                return Decision(
                    route=compiled.route.name,
                    policy=compiled.route.policy.policy,
                    selected_model=primary_ref.id,
                    selected_client=primary_ref.client,
                    bucket=bucket,
                    fallback_model_ids=[ref.id for ref in fallback_refs],
                    fallback_clients=[ref.client for ref in fallback_refs],
                    judge_source=judge_meta["source"] if judge_meta else None,
                    judge_confidence=judge_meta["confidence"] if judge_meta else None,
                    latency_ms=latency_ms,
                    outcome_id=outcome.metadata.outcome_id if outcome.metadata else None,
                )

    raise RuntimeError(f"route {compiled.route.name!r} algorithm ended without a terminal outcome")


__all__ = ["Decision", "UnexpectedRealCallError", "decide"]
