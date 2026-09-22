"""Compiler tests: schema objects in, correct Switchyard TOML out."""

from __future__ import annotations

import re

import pytest

from routerctl.compiler import ConfigError, compile_routes
from routerctl.schema import ClientsFile, TeamFile

CLIENTS = ClientsFile.model_validate(
    {"clients": {"openai": {"format": "openai_chat", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}}}
)


def team(name: str, routes: list[dict]) -> TeamFile:
    return TeamFile.model_validate({"team": name, "routes": routes})


def test_intent_route_compiles_with_harvestable_criteria_lines():
    t = team(
        "voice-ai",
        [
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
    )
    result = compile_routes(CLIENTS, [("voice-ai.yaml", t)])
    assert result.route_names["deployment/transcription"].team == "voice-ai"
    assert result.api_key_envs == {"OPENAI_API_KEY"}
    assert '[routes.deployment_transcription]' in result.toml
    assert 'id = "deployment/transcription"' in result.toml
    assert 'mode = "custom"' in result.toml
    assert "- extraction: field lookups" in result.toml  # harvest_criteria-compatible line
    assert "- analysis: judgment calls" in result.toml
    assert 'default_target = "analysis"' in result.toml
    assert '"enum": [\n            "extraction",\n            "analysis"\n          ]' in result.toml
    assert 'judge = ["jevjudge_jev_latest"]' in result.toml
    assert 'extraction = ["openai_gpt_5_6_luna"]' in result.toml
    assert 'any = ["openai_gpt_5_6_luna", "openai_gpt_5_6_sol"]' in result.toml
    # exactly one shared judge target, not one per route
    assert result.toml.count("[targets.jevjudge_jev_latest]") == 1


def test_complexity_and_escalation_compile_to_the_validated_shapes():
    t = team(
        "x",
        [
            {"name": "deployment/qa", "policy": "complexity", "models": {"weak": {"id": "gpt-5.6-terra", "client": "openai"}, "strong": {"id": "gpt-6-astra", "client": "openai"}}, "base_threshold": 0.55},
            {"name": "deployment/agent", "policy": "escalation", "models": {"weak": {"id": "gpt-5.6-sol", "client": "openai"}, "strong": {"id": "gpt-6-astra", "client": "openai"}}},
        ],
    )
    result = compile_routes(CLIENTS, [("x.yaml", t)])
    assert 'mode = "capability"' in result.toml
    assert "base_threshold = 0.55" in result.toml
    assert 'mode = "escalation"' in result.toml
    assert "confirmations = 2" in result.toml
    # gpt-6-astra is used by both routes: exactly one shared target, no duplicate-target warning
    assert result.toml.count("[targets.openai_gpt_6_astra]") == 1
    assert result.toml.count('strong_target = "openai_gpt_6_astra"') == 2


def test_auto_policy_compiles_to_switchyards_auto_preset():
    t = team("x", [{"name": "deployment/default", "policy": "auto", "models": {"efficient": {"id": "gpt-5.6-terra", "client": "openai"}, "capable": {"id": "gpt-6-astra", "client": "openai"}}}])
    result = compile_routes(CLIENTS, [("x.yaml", t)])
    assert 'type = "auto"' in result.toml
    assert 'capable_target = "openai_gpt_6_astra"' in result.toml
    assert 'efficient_target = "openai_gpt_5_6_terra"' in result.toml
    assert "[targets.jevjudge_jev_latest]" not in result.toml  # auto needs no judge


def test_duplicate_route_name_across_teams_is_a_clear_compile_error():
    a = team("team-a", [{"name": "deployment/shared", "policy": "auto", "models": {"efficient": {"id": "m1", "client": "openai"}, "capable": {"id": "m2", "client": "openai"}}}])
    b = team("team-b", [{"name": "deployment/shared", "policy": "auto", "models": {"efficient": {"id": "m3", "client": "openai"}, "capable": {"id": "m4", "client": "openai"}}}])
    with pytest.raises(ConfigError, match=re.escape("'deployment/shared'") + ".*a.yaml.*team-a.*b.yaml.*team-b"):
        compile_routes(CLIENTS, [("a.yaml", a), ("b.yaml", b)])


def test_same_model_and_client_with_different_extra_body_is_a_clear_compile_error():
    a = team(
        "team-a",
        [
            {
                "name": "deployment/a",
                "policy": "auto",
                "models": {
                    "efficient": {"id": "shared-model", "client": "openai", "extra_body": {"reasoning": {"effort": "low"}}},
                    "capable": {"id": "m2", "client": "openai"},
                },
            }
        ],
    )
    b = team(
        "team-b",
        [
            {
                "name": "deployment/b",
                "policy": "auto",
                "models": {
                    "efficient": {"id": "shared-model", "client": "openai", "extra_body": {"reasoning": {"effort": "high"}}},
                    "capable": {"id": "m2", "client": "openai"},
                },
            }
        ],
    )
    with pytest.raises(ConfigError, match="different extra_body settings"):
        compile_routes(CLIENTS, [("a.yaml", a), ("b.yaml", b)])


def test_same_model_and_client_with_matching_extra_body_shares_one_target():
    a = team(
        "team-a",
        [
            {
                "name": "deployment/a",
                "policy": "auto",
                "models": {
                    "efficient": {"id": "shared-model", "client": "openai", "extra_body": {"reasoning": {"effort": "high"}}},
                    "capable": {"id": "m2", "client": "openai"},
                },
            },
            {
                "name": "deployment/a2",
                "policy": "auto",
                "models": {
                    "efficient": {"id": "shared-model", "client": "openai", "extra_body": {"reasoning": {"effort": "high"}}},
                    "capable": {"id": "m2", "client": "openai"},
                },
            },
        ],
    )
    result = compile_routes(CLIENTS, [("a.yaml", a)])
    assert result.toml.count("[targets.openai_shared_model]") == 1
    assert 'extra_body.reasoning = { effort = "high" }' in result.toml


def test_a_declared_but_unused_client_is_not_emitted_or_required():
    # Regression: found against the real router. clients.yaml is a platform-managed registry
    # meant to let a client be declared before any team adopts it ("teams reference these by
    # name... they don't declare upstream connections themselves" -- clients.yaml's own header).
    # The compiler used to emit every declared client into the TOML regardless of use, so
    # Switchyard's real (non-dry-run) launch demanded an api_key_env for a client nothing
    # routed to -- an unrelated, unused entry blocked every real launch.
    clients = ClientsFile.model_validate(
        {
            "clients": {
                "openai": {"format": "openai_chat", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
                "openrouter": {"format": "openai_chat", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
            }
        }
    )
    t = team("x", [{"name": "deployment/x", "policy": "auto", "models": {"efficient": {"id": "m1", "client": "openai"}, "capable": {"id": "m2", "client": "openai"}}}])
    result = compile_routes(clients, [("x.yaml", t)])
    assert result.api_key_envs == {"OPENAI_API_KEY"}
    assert "[llm_clients.openai]" in result.toml
    assert "[llm_clients.openrouter]" not in result.toml
    assert "OPENROUTER_API_KEY" not in result.toml


def test_extra_body_renders_as_toml_inline_table():
    t = team("x", [{"name": "deployment/x", "policy": "auto", "models": {"efficient": {"id": "m1", "client": "openai", "extra_body": {"reasoning": {"effort": "high"}, "temperature": 0.2}}, "capable": {"id": "m2", "client": "openai"}}}])
    result = compile_routes(CLIENTS, [("x.yaml", t)])
    assert 'extra_body.reasoning = { effort = "high" }' in result.toml
    assert "extra_body.temperature = 0.2" in result.toml
