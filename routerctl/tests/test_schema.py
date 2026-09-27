"""Schema tests: what a team can and can't write, and why."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from routerctl.schema import ClientsFile, TeamFile


def test_intent_policy_parses_and_flattens_correctly():
    team = TeamFile.model_validate(
        {
            "team": "voice-ai",
            "routes": [
                {
                    "name": "deployment/transcription",
                    "policy": "intent",
                    "default": "analysis",
                    "models": {
                        "extraction": {"id": "gpt-5.6-luna", "client": "openai", "description": "field lookups"},
                        "analysis": {"id": "gpt-5.6-sol", "client": "openai", "description": "judgment calls"},
                    },
                }
            ],
        }
    )
    route = team.routes[0]
    assert route.name == "deployment/transcription"
    assert route.policy.policy == "intent"
    assert set(route.policy.models) == {"extraction", "analysis"}
    assert route.policy.default == "analysis"
    assert route.policy.judge.client == "jevjudge"  # default judge injected


def test_intent_requires_at_least_two_buckets():
    with pytest.raises(ValidationError, match="at least 2"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [
                    {
                        "name": "deployment/x",
                        "policy": "intent",
                        "default": "only",
                        "models": {"only": {"id": "m", "client": "openai", "description": "d"}},
                    }
                ],
            }
        )


def test_intent_buckets_need_a_description():
    with pytest.raises(ValidationError, match="missing for: b"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [
                    {
                        "name": "deployment/x",
                        "policy": "intent",
                        "default": "a",
                        "models": {
                            "a": {"id": "m1", "client": "openai", "description": "does a"},
                            "b": {"id": "m2", "client": "openai"},
                        },
                    }
                ],
            }
        )


def test_intent_default_must_be_a_real_bucket():
    with pytest.raises(ValidationError, match="not one of"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [
                    {
                        "name": "deployment/x",
                        "policy": "intent",
                        "default": "nope",
                        "models": {
                            "a": {"id": "m1", "client": "openai", "description": "d"},
                            "b": {"id": "m2", "client": "openai", "description": "d"},
                        },
                    }
                ],
            }
        )


def test_complexity_requires_weak_and_strong():
    with pytest.raises(ValidationError, match="needs models"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [{"name": "deployment/x", "policy": "complexity", "models": {"weak": {"id": "m", "client": "openai"}}}],
            }
        )


def test_complexity_threshold_bounds_are_enforced():
    with pytest.raises(ValidationError, match="must not exceed 1.0"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [
                    {
                        "name": "deployment/x",
                        "policy": "complexity",
                        "models": {"weak": {"id": "m1", "client": "openai"}, "strong": {"id": "m2", "client": "openai"}},
                        "base_threshold": 0.9,
                        "threshold_step": 0.2,
                    }
                ],
            }
        )


def test_auto_requires_efficient_and_capable():
    with pytest.raises(ValidationError, match="needs models"):
        TeamFile.model_validate(
            {"team": "x", "routes": [{"name": "deployment/x", "policy": "auto", "models": {"efficient": {"id": "m", "client": "openai"}}}]}
        )


def test_escalation_defaults_match_switchyard_benchmarked_values():
    team = TeamFile.model_validate(
        {
            "team": "x",
            "routes": [
                {
                    "name": "deployment/x",
                    "policy": "escalation",
                    "models": {"weak": {"id": "m1", "client": "openai"}, "strong": {"id": "m2", "client": "openai"}},
                }
            ],
        }
    )
    policy = team.routes[0].policy
    assert policy.confirmations == 2
    assert policy.recent_turn_window == 28


def test_route_name_must_be_toml_and_url_safe():
    with pytest.raises(ValidationError, match="must start with a letter or digit"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [
                    {
                        "name": "not a valid name!",
                        "policy": "auto",
                        "models": {"efficient": {"id": "m1", "client": "openai"}, "capable": {"id": "m2", "client": "openai"}},
                    }
                ],
            }
        )


def test_bucket_names_must_be_valid_choice_labels():
    with pytest.raises(ValidationError, match="must start with a letter"):
        TeamFile.model_validate(
            {
                "team": "x",
                "routes": [
                    {
                        "name": "deployment/x",
                        "policy": "intent",
                        "default": "a",
                        "models": {
                            "a": {"id": "m1", "client": "openai", "description": "d"},
                            "2bad": {"id": "m2", "client": "openai", "description": "d"},
                        },
                    }
                ],
            }
        )


def test_unknown_top_level_field_is_rejected():
    with pytest.raises(ValidationError):
        TeamFile.model_validate({"team": "x", "routes": [], "extra_stuff": True})


def test_clients_file_rejects_the_reserved_jevjudge_name():
    with pytest.raises(Exception, match="reserved"):
        ClientsFile.model_validate({"clients": {"jevjudge": {"base_url": "http://evil"}}})


def test_judge_can_be_overridden_per_route():
    team = TeamFile.model_validate(
        {
            "team": "x",
            "routes": [
                {
                    "name": "deployment/x",
                    "policy": "escalation",
                    "models": {"weak": {"id": "m1", "client": "openai"}, "strong": {"id": "m2", "client": "openai"}},
                    "judge": {"id": "gpt-5.6-terra", "client": "openai"},
                }
            ],
        }
    )
    judge = team.routes[0].policy.judge
    assert judge.id == "gpt-5.6-terra"
    assert judge.client == "openai"


# ---------------------------------------------------------------------------- cache-aware routing


def _route(**extra) -> dict:
    return {"name": "r", "policy": "complexity", "models": {"weak": {"id": "a", "client": "openai"}, "strong": {"id": "b", "client": "openai"}}, **extra}


def test_cache_block_stays_on_the_route_not_the_policy():
    route = TeamFile.model_validate({"team": "x", "routes": [_route(cache={"horizon_turns": 8, "switch_margin": 0.2})]}).routes[0]
    assert route.cache.horizon_turns == 8
    assert route.cache.switch_margin == 0.2
    assert route.policy.policy == "complexity"


def test_cache_defaults_to_enabled_when_omitted():
    route = TeamFile.model_validate({"team": "x", "routes": [_route()]}).routes[0]
    assert route.cache.enabled is True
    assert route.cache.horizon_turns == 5


@pytest.mark.parametrize("cache", [{"horizon_turns": 0}, {"switch_margin": 1.5}, {"upgrade_min_confidence": -0.1}, {"horizn_turns": 5}])
def test_cache_block_is_validated(cache):
    with pytest.raises(ValidationError):
        TeamFile.model_validate({"team": "x", "routes": [_route(cache=cache)]})


def test_pricing_parses_with_openai_style_cache_writes():
    clients = ClientsFile.model_validate({
        "clients": {"openai": {"base_url": "https://api.openai.com/v1"}},
        "pricing": {"gpt-5.6-terra": {"input": 2.0, "cached_input": 0.2, "cache_write": 2.5, "output": 12.0, "cache_ttl_seconds": 1800}},
    })
    terra = clients.pricing["gpt-5.6-terra"]
    assert (terra.cache_write, terra.cache_ttl_seconds, terra.min_cacheable_tokens) == (2.5, 1800, 1024)


def test_pricing_rejects_a_cached_rate_above_the_input_rate():
    with pytest.raises(ValidationError, match="cannot exceed"):
        ClientsFile.model_validate({"clients": {"openai": {"base_url": "x"}}, "pricing": {"m": {"input": 1.0, "cached_input": 2.0, "output": 1.0}}})


def test_pricing_is_optional():
    assert ClientsFile.model_validate({"clients": {"openai": {"base_url": "x"}}}).pricing == {}
