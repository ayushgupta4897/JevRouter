"""The `complexity` rubric fix: what the judge is asked to forecast against, in both modes.
Accuracy itself is measured against real Jev in experiments/complexity_eval.py (DECISION.md section 14)."""

from __future__ import annotations

import tomllib

from jevjudge.compiler import harvest_criteria

from routerctl.algorithms import compile_route_algorithm
from routerctl.compiler import compile_routes
from routerctl.rubrics import complexity_prompt
from routerctl.schema import ClientsFile, Route, TeamFile

RULES = ["SUP-1", "SUP-2", "SUP-3", "SUP-4", "SUP-5", "UNC-1", "UNC-2", "LIM-1", "LIM-2"]


def complexity(**extra) -> Route:
    return Route.model_validate({"name": "r", "policy": "complexity", **extra,
                                 "models": {"weak": {"id": "w", "client": "openai"}, "strong": {"id": "s", "client": "openai"}}})


def test_general_is_the_default_rubric():
    assert complexity().policy.rubric == "general"
    assert complexity_prompt(complexity().policy) is not None


def test_coding_agent_keeps_switchyards_packaged_rubric():
    assert complexity_prompt(complexity(rubric="coding_agent").policy) is None


def test_every_rule_id_is_a_harvestable_criterion_for_jev():
    # jevjudge turns `- SUP-1 [supported]: ...` lines into the criteria Jev picks primary_rule by;
    # every id in Switchyard's fixed schema enum must have one, or Jev chooses among blanks
    criteria = harvest_criteria(complexity_prompt(complexity().policy), RULES)
    assert set(criteria) == set(RULES)
    assert all(len(text) > 20 for text in criteria.values())


def test_general_rubric_is_about_the_task_not_a_repository():
    prompt = complexity_prompt(complexity().policy)
    for coding_agent_term in ("validator", "harness", "repository", "executable reference"):
        assert coding_agent_term not in prompt


def test_team_criteria_replace_the_most_generic_rule_on_each_side():
    policy = complexity(weak_when="A single-table select.", strong_when="Window functions or recursive CTEs.").policy
    criteria = harvest_criteria(complexity_prompt(policy), RULES)
    assert criteria["SUP-5"] == "A single-table select."
    assert criteria["LIM-2"] == "Window functions or recursive CTEs."
    assert criteria["LIM-1"].startswith("Correctness depends on multi-step reasoning")  # the rest stay


def test_decision_mode_compiles_with_the_prompt_override():
    compiled = compile_route_algorithm(complexity(strong_when="anything hard"))
    assert compiled.models["efficient"] == ["w"] and compiled.models["capable"] == ["s"]


def test_proxy_mode_emits_the_same_prompt_as_a_valid_toml_string():
    clients = ClientsFile.model_validate({"clients": {"openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}}})
    route = {"name": "r", "policy": "complexity", "strong_when": "Window functions.",
             "models": {"weak": {"id": "w", "client": "openai"}, "strong": {"id": "s", "client": "openai"}}}
    result = compile_routes(clients, [("t.yaml", TeamFile.model_validate({"team": "t", "routes": [route]}))])
    emitted = tomllib.loads(result.toml)["routes"]["r"]["prompt"]
    assert emitted == complexity_prompt(Route.model_validate(route).policy)


def test_proxy_mode_omits_the_prompt_for_coding_agent_routes():
    clients = ClientsFile.model_validate({"clients": {"openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}}})
    route = {"name": "r", "policy": "complexity", "rubric": "coding_agent",
             "models": {"weak": {"id": "w", "client": "openai"}, "strong": {"id": "s", "client": "openai"}}}
    result = compile_routes(clients, [("t.yaml", TeamFile.model_validate({"team": "t", "routes": [route]}))])
    assert "prompt" not in tomllib.loads(result.toml)["routes"]["r"]
