"""Structural tests for compile_route_algorithm: correct models dict shape and reverse index,
for every policy, without needing to actually run the algorithm (see test_decide.py for that)."""

from __future__ import annotations

from routerctl.algorithms import compile_route_algorithm
from routerctl.schema import Route


def route(name: str, policy: dict) -> Route:
    return Route.model_validate({"name": name, **policy})


def test_auto_compiles_with_efficient_capable_and_any_categories():
    compiled = compile_route_algorithm(route("r", {"policy": "auto", "models": {"efficient": {"id": "m-eff", "client": "openai"}, "capable": {"id": "m-cap", "client": "openai"}}}))
    assert compiled.models["efficient"] == ["m-eff"]
    assert compiled.models["capable"] == ["m-cap"]
    assert set(compiled.models["any"]) == {"m-eff", "m-cap"}
    assert "judge" not in compiled.models  # auto's stage_router preset needs no classifier at all
    assert compiled.model_by_id["m-eff"].id == "m-eff"
    assert compiled.bucket_by_model_id is None


def test_complexity_compiles_with_judge_and_any_categories():
    compiled = compile_route_algorithm(
        route("r", {"policy": "complexity", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}, "base_threshold": 0.6})
    )
    assert compiled.models["judge"] == ["jev-latest"]
    assert compiled.models["efficient"] == ["m-weak"]
    assert compiled.models["capable"] == ["m-strong"]
    assert set(compiled.models["any"]) == {"m-weak", "m-strong"}


def test_intent_compiles_one_bucket_per_named_model_plus_any():
    compiled = compile_route_algorithm(
        route(
            "r",
            {
                "policy": "intent",
                "default": "b",
                "models": {
                    "a": {"id": "m-a", "client": "openai", "description": "does a"},
                    "b": {"id": "m-b", "client": "openai", "description": "does b"},
                },
            },
        )
    )
    assert compiled.models["a"] == ["m-a"]
    assert compiled.models["b"] == ["m-b"]
    assert set(compiled.models["any"]) == {"m-a", "m-b"}
    assert compiled.bucket_by_model_id == {"m-a": "a", "m-b": "b"}


def test_escalation_compiles_as_continue_escalate_custom_classifier():
    # Regression: Switchyard's native LlmClassifierConfig.escalation is response-based -- it
    # calls the current tier's model itself to observe its live behavior, verified empirically
    # (its run_stream yields a CallModel for the weak model, not just the judge). That breaks
    # "the router decides, an external gateway executes", so escalation compiles here as a
    # custom classifier reading the transcript already in the request instead.
    compiled = compile_route_algorithm(
        route("r", {"policy": "escalation", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}, "confirmations": 3})
    )
    assert compiled.models["continue"] == ["m-weak"]
    assert compiled.models["escalate"] == ["m-strong"]
    assert set(compiled.models["any"]) == {"m-weak", "m-strong"}
    assert compiled.bucket_by_model_id == {"m-weak": "continue", "m-strong": "escalate"}
