"""Drive one decision-only run of a compiled route's algorithm to a `Decision` -- never a
completion. The one real network call this makes is to the shared jevjudge classifier (cheap,
and the whole point of this project); the target model itself is never invoked here."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from switchyard.libsy import LlmResponse, Step

from jevjudge.cascade import Judge
from jevjudge.client import JevError
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
    judge_error: str | None  # set when the judge was unreachable and this decision fell back to
    # this route's own safe default (complexity -> capable tier; intent/escalation -> default_target)
    # instead of a real verdict -- Switchyard's own documented behavior for "unsure or unreachable",
    # verified empirically (see routerctl/tests/test_decide.py). Never silent: check this field.
    latency_ms: float
    outcome_id: str | None
    cache: dict | None = None  # set by cache.apply(): what the cache gate did and why
    # The route's per-model `extra_body` (reasoning effort, provider pinning, ...) for the gateway
    # to send with the call. Decision-only mode used to drop these silently: a route could say
    # `extra_body: {provider: {order: [fireworks]}}` and the gateway never heard about it.
    selected_extra_body: dict | None = None
    fallback_extra_bodies: list[dict | None] = field(default_factory=list)


def _windowed(request: dict, recent_turn_window: int | None) -> dict:
    """Escalation routes should judge only their configured recent_turn_window of turns -- not
    the whole history a long-running agent session might carry. Slicing here, on the actual
    message list, is the real enforcement; algorithms.py's escalation prompt also *says* "the
    most recent N turns" for the judge's benefit, but that was found to be decorative on its own
    (the full conversation was still being sent) until this was added."""
    if not recent_turn_window or len(request.get("messages") or []) <= recent_turn_window:
        return request
    return {**request, "messages": request["messages"][-recent_turn_window:]}


async def decide(compiled: CompiledRoute, judge: Judge, request: dict, *, headers: dict[str, str] | None = None) -> Decision:
    started = time.perf_counter()
    judge_id = (compiled.models.get("judge") or [None])[0]
    judge_meta: dict | None = None
    judge_error: str | None = None
    request = _windowed(request, compiled.recent_turn_window)

    async for step in compiled.algorithm.run_stream(request, compiled.models, headers=headers):
        match step:
            case Step.CallModel(call):
                requested = call.models[0] if call.models else None
                if judge_id is not None and requested == judge_id:
                    try:
                        judge_meta = await serve_judge_call(judge, call, LlmResponse=LlmResponse)
                    except JevError as error:
                        # Fail this one step, not the whole decision: verified empirically that
                        # Switchyard's own algorithms treat a failed judge call as "unsure or
                        # unreachable" and fall back to a safe default (complexity -> the capable
                        # tier; intent/escalation -> the route's own `default_target`) rather than
                        # aborting -- see test_decide.py. A caller must still be able to tell this
                        # happened, so it's surfaced on the Decision, never swallowed silently.
                        judge_error = str(error)
                        call.fail(error)
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
                    judge_error=judge_error,
                    latency_ms=latency_ms,
                    outcome_id=outcome.metadata.outcome_id if outcome.metadata else None,
                    selected_extra_body=primary_ref.extra_body,
                    fallback_extra_bodies=[ref.extra_body for ref in fallback_refs],
                )

    raise RuntimeError(f"route {compiled.route.name!r} algorithm ended without a terminal outcome")


__all__ = ["Decision", "UnexpectedRealCallError", "decide"]
