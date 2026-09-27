"""Build a `switchyard.libsy` embedded algorithm for one compiled `Route` -- a *decision-only*
one, by construction: every policy compiled here needs at most the shared jevjudge classifier
call (`auto`'s `stage_router` preset with no classifier needs none at all) and never asks its
caller to execute a real target model. See docs/DECISION.md's real-router section for why.

`escalation` is the one policy that does not map onto this directly: Switchyard's native
`LlmClassifierConfig.escalation` is *response-based* -- it calls the current tier's model itself
to observe its live behavior before judging whether to escalate, confirmed empirically (its
`run_stream` yields a `CallModel` for the weak model, not just the judge). That is fundamentally
incompatible with "the router decides, an external AI Gateway executes": there is no way to ask
"should we escalate" without either running a model ourselves or judging what already happened.
So `escalation` here is reimplemented as a `custom` classifier that judges the transcript already
present in the request (the Gateway's own prior turns, including any tool calls/results it has
already executed) for a repeated-failure pattern -- the same rubric intent, phrased as reading
history instead of watching a fresh call unfold. This keeps escalation genuinely decision-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from switchyard.libsy import Algorithm, CustomClassifierConfig, LlmClassifierConfig, TaskClassifierConfig, algorithms

from .rubrics import complexity_prompt
from .schema import AutoPolicy, ComplexityPolicy, EscalationPolicy, IntentPolicy, ModelRef, Route


@dataclass
class CompiledRoute:
    """A route ready to decide: its algorithm, the `run_stream` models dict, and enough of a
    reverse index to turn `selected_model_ids` (bare strings) back into `(id, client)` pairs."""

    route: Route
    algorithm: Algorithm
    models: dict[str, list[str]]
    model_by_id: dict[str, ModelRef]
    bucket_by_model_id: dict[str, str] | None = None  # intent / escalation-as-custom only
    recent_turn_window: int | None = None  # escalation only -- decide() slices messages to this


def _index(*refs: ModelRef) -> dict[str, ModelRef]:
    return {ref.id: ref for ref in refs}


def _target_selector_schema(bucket_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "object",
                "properties": {"target": {"type": "string", "enum": bucket_names}},
                "required": ["target"],
                "additionalProperties": False,
            }
        },
        "required": ["decision"],
        "additionalProperties": False,
    }


def _compile_custom(
    route: Route, judge_id: str, default_bucket: str, buckets: dict[str, ModelRef], prompt: str, *, recent_turn_window: int | None = None
) -> CompiledRoute:
    bucket_names = list(buckets)
    config = CustomClassifierConfig(prompt, _target_selector_schema(bucket_names), "/decision/target")
    algorithm = algorithms.llm_classifier(LlmClassifierConfig.custom(default_target=default_bucket, config=config))
    any_ids = [ref.id for ref in buckets.values()]
    models = {"judge": [judge_id], "any": any_ids, **{name: [ref.id] for name, ref in buckets.items()}}
    return CompiledRoute(
        route=route,
        algorithm=algorithm,
        models=models,
        model_by_id=_index(*buckets.values()),
        bucket_by_model_id={ref.id: name for name, ref in buckets.items()},
        recent_turn_window=recent_turn_window,
    )


def _intent_prompt(policy: IntentPolicy) -> str:
    lines = ["Choose the model group that best fits this request."]
    for bucket, model in policy.models.items():
        description = " ".join((model.description or "").split())
        lines.append(f"- {bucket}: {description}")
    lines.append("Return JSON matching the response schema supplied with the request.")
    return "\n".join(lines)


def _escalation_prompt(policy: EscalationPolicy) -> str:
    return (
        "You are shown a conversation that may include past tool calls and their results. Decide "
        "whether it shows a clear, repeated failure pattern -- the same kind of error or lack of "
        f"progress recurring at least {policy.confirmations} times in a row -- versus normal, "
        "productive work with at most an isolated failure. Escalate only on a clear pattern; never "
        f"on a single failed command or ordinary difficulty. Consider only the most recent "
        f"{policy.recent_turn_window} turns.\n"
        "- continue: no clear repeated-failure pattern; ordinary progress or an isolated issue.\n"
        "- escalate: a clear, repeated failure pattern spanning multiple turns.\n"
        "Return JSON matching the response schema supplied with the request."
    )


def compile_route_algorithm(route: Route) -> CompiledRoute:
    policy = route.policy

    if isinstance(policy, AutoPolicy):
        efficient, capable = policy.models["efficient"], policy.models["capable"]
        algorithm = algorithms.stage_router(picker="efficient_first", confidence_threshold=0.5)
        models = {"efficient": [efficient.id], "capable": [capable.id], "any": [efficient.id, capable.id]}
        return CompiledRoute(route=route, algorithm=algorithm, models=models, model_by_id=_index(efficient, capable))

    if isinstance(policy, ComplexityPolicy):
        weak, strong = policy.models["weak"], policy.models["strong"]
        config = TaskClassifierConfig(policy.base_threshold, threshold_step=policy.threshold_step, prompt=complexity_prompt(policy))
        algorithm = algorithms.llm_classifier(LlmClassifierConfig.capability(config=config))
        models = {"judge": [policy.judge.id], "efficient": [weak.id], "capable": [strong.id], "any": [weak.id, strong.id]}
        return CompiledRoute(route=route, algorithm=algorithm, models=models, model_by_id=_index(weak, strong))

    if isinstance(policy, IntentPolicy):
        return _compile_custom(route, policy.judge.id, policy.default, policy.models, _intent_prompt(policy))

    if isinstance(policy, EscalationPolicy):
        weak, strong = policy.models["weak"], policy.models["strong"]
        return _compile_custom(
            route, policy.judge.id, "continue", {"continue": weak, "escalate": strong}, _escalation_prompt(policy),
            recent_turn_window=policy.recent_turn_window,
        )

    raise AssertionError(f"unhandled policy type: {type(policy).__name__}")  # pragma: no cover


__all__ = ["CompiledRoute", "compile_route_algorithm"]
