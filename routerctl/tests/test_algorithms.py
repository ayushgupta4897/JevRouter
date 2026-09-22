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


def test_intent_with_four_buckets_compiles_correctly():
    compiled = compile_route_algorithm(
        route(
            "r",
            {
                "policy": "intent",
                "default": "c",
                "models": {
                    "a": {"id": "m-a", "client": "openai", "description": "a"},
                    "b": {"id": "m-b", "client": "openai", "description": "b"},
                    "c": {"id": "m-c", "client": "openai", "description": "c"},
                    "d": {"id": "m-d", "client": "openai", "description": "d"},
                },
            },
        )
    )
    assert set(compiled.models["any"]) == {"m-a", "m-b", "m-c", "m-d"}
    assert compiled.bucket_by_model_id == {"m-a": "a", "m-b": "b", "m-c": "c", "m-d": "d"}
    assert {compiled.models[b][0] for b in ("a", "b", "c", "d")} == {"m-a", "m-b", "m-c", "m-d"}


def test_escalation_default_recent_turn_window_is_28_when_not_set():
    compiled = compile_route_algorithm(
        route("r", {"policy": "escalation", "models": {"weak": {"id": "m-weak", "client": "openai"}, "strong": {"id": "m-strong", "client": "openai"}}})
    )
    assert compiled.recent_turn_window == 28


def test_non_escalation_policies_never_set_a_recent_turn_window():
    for policy in (
        {"policy": "auto", "models": {"efficient": {"id": "m1", "client": "openai"}, "capable": {"id": "m2", "client": "openai"}}},
        {"policy": "complexity", "models": {"weak": {"id": "m1", "client": "openai"}, "strong": {"id": "m2", "client": "openai"}}},
        {"policy": "intent", "default": "a", "models": {"a": {"id": "m1", "client": "openai", "description": "a"}, "b": {"id": "m2", "client": "openai", "description": "b"}}},
    ):
        assert compile_route_algorithm(route("r", policy)).recent_turn_window is None


def test_same_model_id_for_both_tiers_does_not_crash_and_resolves_consistently():
    # Unusual (a team pointing both tiers at the same model), but not schema-forbidden --
    # model_by_id naturally collapses to one entry since both refs share the same id.
    compiled = compile_route_algorithm(
        route("r", {"policy": "complexity", "models": {"weak": {"id": "m-shared", "client": "openai"}, "strong": {"id": "m-shared", "client": "openai"}}})
    )
    assert compiled.models["efficient"] == compiled.models["capable"] == ["m-shared"]
    assert compiled.model_by_id["m-shared"].client == "openai"


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
