"""Compiler tests against the real Switchyard packaged schemas and a custom-mode schema."""

from __future__ import annotations

import json

import pytest

from jevjudge.compiler import (
    CompileError,
    compile_schema,
    decode_answers,
    extract_schema,
    harvest_criteria,
)

# Verbatim from NVIDIA-NeMo/Switchyard crates/libsy/src/prompts/capability-classifier/schema.json
CAPABILITY_RF = json.loads(
    """{
  "type": "json_schema",
  "json_schema": {
    "name": "CapabilityClassifierDecision",
    "strict": true,
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["crux", "primary_rule", "capability_boundary", "p_solve"],
      "properties": {
        "crux": {"type": "string", "minLength": 1},
        "primary_rule": {"type": "string", "enum": ["SUP-1","SUP-2","SUP-3","SUP-4","SUP-5","UNC-1","UNC-2","LIM-1","LIM-2","none"]},
        "capability_boundary": {"type": "string", "enum": ["supported", "uncertain", "unsupported", "unmatched"]},
        "p_solve": {"type": "number", "minimum": 0.0, "maximum": 1.0}
      }
    }
  }
}"""
)

CAPABILITY_PROMPT = """You are a task-level probability forecaster for a model router.

# Efficient-agent capability card

- SUP-1 [supported]: Route to the Efficient model when the task provides a complete output contract and a deterministic local validator.
- SUP-2 [supported]: Route to the Efficient model when all required inputs are available.
- UNC-1 [uncertain]: Treat the route as uncertain when multiple reasonable interpretations exist.
- LIM-2 [unsupported]: Prefer the Capable model when success depends on reproducing undocumented reference behavior.

# Output
Return exactly one JSON object matching the response schema supplied with the request.
"""

# Verbatim from crates/libsy/src/prompts/escalation/schema.json
ESCALATION_RF = json.loads(
    """{
  "type": "json_schema",
  "json_schema": {
    "name": "EscalationVerdict",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "escalate": {"type": "boolean", "description": "True when the run is likely doomed without escalation to the strong tier."},
        "reason": {"type": "string", "description": "One short sentence naming the trouble pattern, or stating why the run is progressing."}
      },
      "required": ["escalate", "reason"],
      "additionalProperties": false
    }
  }
}"""
)

# Custom mode wrapper as Switchyard builds it (from_inner_schema): name is fixed.
CUSTOM_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "switchyard_classifier_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "properties": {"target": {"type": "string", "enum": ["fast", "balanced", "reasoning", "premium"]}},
                    "required": ["target"],
                    "additionalProperties": False,
                }
            },
            "required": ["decision"],
            "additionalProperties": False,
        },
    },
}

CUSTOM_PROMPT = """Choose the best group for this request.
- fast: greetings, lookups, one-line edits.
- balanced: ordinary coding and writing.
- reasoning: multi-step math, algorithms, formal proofs.
- premium: high-stakes architecture or security decisions.
"""


def test_capability_schema_compiles_to_two_questions_and_derives_boundary():
    schema, name = extract_schema(CAPABILITY_RF, CAPABILITY_PROMPT)
    compiled = compile_schema(schema, schema_name=name, system_prompt=CAPABILITY_PROMPT)
    assert compiled.profile == "CapabilityClassifierDecision"
    assert set(compiled.questions) == {"primary_rule", "p_solve"}  # boundary derived, crux templated
    assert compiled.questions["p_solve"]["type"] == "noul"
    rule_q = compiled.questions["primary_rule"]
    assert rule_q["type"] == "choice"
    assert rule_q["criteria"]["SUP-1"].startswith("Route to the Efficient model when the task provides")
    assert rule_q["criteria"]["none"]  # profile-supplied
    assert rule_q["criteria"]["SUP-3"] is None  # not in this (abridged) prompt -> name only

    answers = {
        "primary_rule": {"type": "choice", "choice": "LIM-2", "confidence": 0.81, "probabilities": {"LIM-2": 0.81, "UNC-1": 0.1}},
        "p_solve": {"type": "noul", "noul": 0.23},
    }
    decoded = decode_answers(compiled, answers)
    v = decoded.verdict
    assert v["primary_rule"] == "LIM-2"
    assert v["capability_boundary"] == "unsupported"  # consistent with Switchyard's is_valid()
    assert v["p_solve"] == 0.23
    assert v["crux"].startswith("jevjudge: rule=LIM-2 p_solve=0.230")
    assert set(v) == {"crux", "primary_rule", "capability_boundary", "p_solve"}
    # gate confidence = min(choice confidence 0.81, noul confidence |0.23-0.5|*2 = 0.54)
    assert decoded.confidence == pytest.approx(0.54)


@pytest.mark.parametrize(
    "rule,boundary",
    [("SUP-4", "supported"), ("UNC-2", "uncertain"), ("LIM-1", "unsupported"), ("none", "unmatched")],
)
def test_boundary_derivation_matches_switchyard_validity_table(rule, boundary):
    schema, name = extract_schema(CAPABILITY_RF, CAPABILITY_PROMPT)
    compiled = compile_schema(schema, schema_name=name, system_prompt=CAPABILITY_PROMPT)
    decoded = decode_answers(
        compiled,
        {"primary_rule": {"type": "choice", "choice": rule, "confidence": 0.9, "probabilities": {rule: 0.9}}, "p_solve": {"type": "noul", "noul": 0.7}},
    )
    assert decoded.verdict["capability_boundary"] == boundary


def test_escalation_schema_is_one_noul_plus_template():
    schema, name = extract_schema(ESCALATION_RF, "judge prompt")
    compiled = compile_schema(schema, schema_name=name, system_prompt="judge prompt")
    assert list(compiled.questions) == ["escalate"]
    assert compiled.questions["escalate"]["type"] == "noul"
    assert "criteria" in compiled.questions["escalate"]  # profile-supplied true/false criteria
    decoded = decode_answers(compiled, {"escalate": {"type": "noul", "noul": 0.87}})
    assert decoded.verdict == {"escalate": True, "reason": "jevjudge: p_escalate=0.870"}
    assert decoded.confidence == pytest.approx(0.74)
    decoded = decode_answers(compiled, {"escalate": {"type": "noul", "noul": 0.2}})
    assert decoded.verdict["escalate"] is False


def test_custom_mode_nested_enum_becomes_choice_with_prompt_harvested_criteria():
    schema, name = extract_schema(CUSTOM_RF, CUSTOM_PROMPT)
    compiled = compile_schema(schema, schema_name=name, system_prompt=CUSTOM_PROMPT)
    assert compiled.profile == "generic"
    assert list(compiled.questions) == ["decision__target"]
    crit = compiled.questions["decision__target"]["criteria"]
    assert crit == {
        "fast": "greetings, lookups, one-line edits.",
        "balanced": "ordinary coding and writing.",
        "reasoning": "multi-step math, algorithms, formal proofs.",
        "premium": "high-stakes architecture or security decisions.",
    }
    decoded = decode_answers(
        compiled,
        {"decision__target": {"type": "choice", "choice": "reasoning", "confidence": 0.66, "probabilities": {"reasoning": 0.66, "balanced": 0.3, "fast": 0.02, "premium": 0.02}}},
    )
    assert decoded.verdict == {"decision": {"target": "reasoning"}}
    assert decoded.confidence == pytest.approx(0.66)


def test_unknown_choice_falls_back_to_best_known_probability():
    schema, name = extract_schema(CUSTOM_RF, CUSTOM_PROMPT)
    compiled = compile_schema(schema, schema_name=name, system_prompt=CUSTOM_PROMPT)
    decoded = decode_answers(
        compiled,
        {"decision__target": {"type": "choice", "choice": "nope", "confidence": 0.5, "probabilities": {"premium": 0.6, "fast": 0.4}}},
    )
    assert decoded.verdict["decision"]["target"] == "premium"


def test_json_object_mode_reads_schema_from_prompt():
    prompt = CAPABILITY_PROMPT + "\n\nReturn exactly one JSON object matching this JSON Schema:\n" + json.dumps(CAPABILITY_RF["json_schema"]["schema"], indent=2)
    schema, name = extract_schema({"type": "json_object"}, prompt)
    assert schema["properties"]["p_solve"]["maximum"] == 1.0
    compiled = compile_schema(schema, schema_name="CapabilityClassifierDecision", system_prompt=prompt)
    assert set(compiled.questions) == {"primary_rule", "p_solve"}


def test_generic_mapping_covers_score_integer_and_literal_fields():
    schema = {
        "type": "object",
        "properties": {
            "risk": {"type": "integer", "minimum": 1, "maximum": 5, "description": "How risky is the requested action?"},
            "needs_tools": {"type": "boolean"},
            "kind": {"type": "string", "const": "verdict"},
            "notes": {"type": "string", "minLength": 3},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["risk", "needs_tools", "kind", "notes", "tags"],
    }
    compiled = compile_schema(schema, schema_name="Anything")
    assert compiled.questions["risk"]["type"] == "score"
    assert len(compiled.questions["risk"]["criteria"]) == 5
    assert compiled.questions["needs_tools"]["type"] == "noul"
    decoded = decode_answers(
        compiled,
        {"risk": {"type": "score", "score": 3.4, "confidence": 0.7, "probabilities": {}}, "needs_tools": {"type": "noul", "noul": 0.5}},
    )
    assert decoded.verdict == {"risk": 4, "needs_tools": True, "kind": "verdict", "notes": "jevjudge: decided by Jev (no free text generated)", "tags": []}
    assert decoded.confidence == 0.0  # the 0.5 noul is maximally uncertain


def test_schema_with_nothing_decidable_is_rejected():
    with pytest.raises(CompileError):
        compile_schema({"type": "object", "properties": {"reason": {"type": "string"}}}, schema_name="x")


def test_harvest_handles_markdown_and_dash_variants():
    prompt = "* **fast** — cheap and quick\n- slow [tier]: heavy lifting\nnot a rule line"
    assert harvest_criteria(prompt, ["fast", "slow", "other"]) == {"fast": "cheap and quick", "slow": "heavy lifting"}
