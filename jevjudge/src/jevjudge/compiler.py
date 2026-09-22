"""Compile a judge's JSON Schema into Jev questions, and Jev answers back into a verdict.

Jev (TypeSafe's System One model) does not generate text. It answers a fixed set of typed
questions about a `state` in one parallel pass:

* ``noul``   -> probability that a statement is true
* ``choice`` -> a probability distribution over named options
* ``score``  -> an expected value over an ordered rubric

An LLM judge in a router is asked for a JSON object such as
``{"escalate": bool, "reason": str}`` or ``{"primary_rule": enum, "p_solve": 0..1, ...}``.
Almost every field in such a verdict is a decision, not prose, so it maps onto a Jev
primitive. The few free-text fields (``reason``, ``crux``) are filled with a deterministic
template so the verdict still validates against the schema.

The mapping is deterministic and schema-driven; optional *profiles* refine it for schemas
we know (Switchyard's packaged capability and escalation verdicts) by supplying better
question instructions, harvesting enum criteria from the judge's own system prompt, and
deriving fields that must stay consistent with each other.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

JsonSchema = dict[str, Any]

# Placeholder for free-text fields Jev cannot produce. Kept short and greppable.
DEFAULT_TEXT_PLACEHOLDER = "jevjudge: decided by Jev (no free text generated)"


class CompileError(ValueError):
    """The schema cannot be turned into Jev questions."""


@dataclass
class FieldPlan:
    """How one schema property is answered and reassembled."""

    path: tuple[str, ...]
    kind: str  # noul_bool | noul_prob | choice | score | literal | template | derived
    question: str | None = None  # Jev question name (None for literal/template/derived)
    options: list[Any] = field(default_factory=list)  # choice labels or score levels (in order)
    literal: Any = None  # literal value, or template string for `template`
    derive: dict[str, Any] | None = None  # derived: {"from": path, "map": {...}, "default": x}
    min_length: int = 0
    minimum: float | None = None

    @property
    def decides(self) -> bool:
        """True when this field carries a Jev decision (and so a confidence)."""
        return self.kind in {"noul_bool", "noul_prob", "choice", "score"}


@dataclass
class Compiled:
    """Jev questions plus the plan for reassembling the JSON verdict."""

    questions: dict[str, dict[str, Any]]
    plan: list[FieldPlan]
    schema_name: str
    profile: str
    # Flat field paths whose confidence the gate should consider. `None` (the default, and
    # every generic/unprofiled schema) means every decided field — safe when we don't know
    # which field the caller's policy actually thresholds on. A profile sets this when some
    # decided fields are diagnostic rather than decisive: Switchyard's capability policy
    # routes on `p_solve` alone and already widens its own threshold when `primary_rule`
    # lands on an uncertain or unmatched label (see `threshold_step` in llm_class.rs), so
    # gating on `primary_rule`'s own confidence too double-counts that same uncertainty and
    # can abstain a verdict whose actual routing signal (`p_solve`) was never in doubt.
    gate_fields: frozenset[str] | None = None


@dataclass
class Decoded:
    """A schema-shaped verdict plus per-field decision evidence."""

    verdict: dict[str, Any]
    confidence: float  # min over compiled.gate_fields (default: all deciding fields)
    evidence: dict[str, dict[str, Any]]


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------

_CAPABILITY_BOUNDARY_BY_PREFIX = {
    "SUP": "supported",
    "UNC": "uncertain",
    "LIM": "unsupported",
    "none": "unmatched",
}

PROFILES: dict[str, dict[str, Any]] = {
    # NVIDIA NeMo Switchyard, `llm_classifier` mode = "capability".
    # crates/libsy/src/prompts/capability-classifier/{prompt.md,schema.json}
    "CapabilityClassifierDecision": {
        "instructions": {
            "p_solve": (
                "SUCCESS: the EFFICIENT (cheaper) agent completes the whole task correctly on one "
                "fresh run under the actual harness, tools, and budget, as judged by the final "
                "verifier. Use only evidence in the task and the capability rules in "
                "judge_instructions. Answer with the probability of SUCCESS."
            ),
            "primary_rule": (
                "Which single capability rule from judge_instructions best describes the crux, "
                "the hardest material requirement for whole-task success? Pick `none` if no rule "
                "applies."
            ),
        },
        "extra_criteria": {"primary_rule": {"none": "No listed rule describes the crux."}},
        # capability_boundary must agree with primary_rule (Switchyard rejects inconsistent
        # verdicts), so derive it instead of asking a second, possibly disagreeing question.
        "derived": {
            "capability_boundary": {
                "from": "primary_rule",
                "prefix_map": _CAPABILITY_BOUNDARY_BY_PREFIX,
                "default": "unmatched",
            }
        },
        "templates": {"crux": "jevjudge: rule={primary_rule} p_solve={p_solve:.3f}"},
        # Switchyard thresholds on p_solve alone; primary_rule only widens that threshold
        # (threshold_step) when it lands outside "supported". Gate on p_solve so a genuinely
        # decisive p_solve isn't abstained just because ten rule labels were each plausible.
        "gate_fields": {"p_solve"},
    },
    # NVIDIA NeMo Switchyard, `llm_classifier` mode = "escalation".
    # crates/libsy/src/prompts/escalation/{prompt.md,schema.json}
    "EscalationVerdict": {
        "instructions": {
            "escalate": (
                "The run is likely DOOMED without escalation to the STRONG tier: the recent turns "
                "show a clear PATTERN of trouble (repetition or loops with no new information, "
                "false progress contradicted by tool output, drift away from the task, giving up "
                "or destructive flailing) of a kind a stronger model would fix. A single failed "
                "command, expected friction, or an external blocker no model can fix does NOT count."
            )
        },
        "noul_criteria": {
            "escalate": {
                "true": "Stuck in place; recent behaviour shows no mechanism for the next turns to differ.",
                "false": "Failing forward, healthy friction, still orienting, or blocked by something external.",
            }
        },
        "templates": {"reason": "jevjudge: p_escalate={escalate:.3f}"},
    },
}


# --------------------------------------------------------------------------------------
# Criteria harvesting from the judge prompt
# --------------------------------------------------------------------------------------

_RULE_LINE = re.compile(
    r"^\s*[-*]\s*(?:\*\*)?(?P<label>[A-Za-z0-9_./-]+)(?:\*\*)?\s*(?:\[[^\]]*\])?\s*[:—-]\s*(?P<text>.+?)\s*$"
)


def harvest_criteria(prompt: str, labels: list[str]) -> dict[str, str]:
    """Pull `- LABEL [tag]: description` lines out of a judge prompt for the given labels.

    Switchyard's capability prompt lists its rules exactly this way (``- SUP-1 [supported]: ...``)
    and a custom-mode prompt can adopt the same convention to describe each routing option.
    """
    wanted = {label.lower(): label for label in labels}
    found: dict[str, str] = {}
    for line in prompt.splitlines():
        match = _RULE_LINE.match(line)
        if not match:
            continue
        label = match.group("label").lower()
        if label in wanted and wanted[label] not in found:
            found[wanted[label]] = match.group("text")
    return found


# --------------------------------------------------------------------------------------
# Schema helpers
# --------------------------------------------------------------------------------------


def extract_schema(response_format: Any, system_prompt: str = "") -> tuple[JsonSchema, str]:
    """Return (inner JSON Schema, schema name) from an OpenAI `response_format`.

    Supports ``{"type":"json_schema","json_schema":{"name":..,"schema":{..}}}`` and, for
    ``{"type":"json_object"}``, a schema embedded in the system prompt after the phrase
    "matching this JSON Schema:" (Switchyard's ``response_format_type = "json_object"`` mode).
    """
    if isinstance(response_format, dict):
        kind = response_format.get("type")
        if kind == "json_schema":
            wrapper = response_format.get("json_schema") or {}
            schema = wrapper.get("schema")
            if not isinstance(schema, dict):
                raise CompileError("response_format.json_schema.schema is missing")
            return schema, str(wrapper.get("name") or "response")
        if kind == "json_object":
            schema = _schema_from_prompt(system_prompt)
            if schema is not None:
                return schema, str(schema.get("title") or "json_object")
            raise CompileError("json_object mode without an embedded JSON Schema in the prompt")
    schema = _schema_from_prompt(system_prompt)
    if schema is not None:
        return schema, str(schema.get("title") or "prompt_schema")
    raise CompileError("no JSON Schema found in response_format or system prompt")


def _schema_from_prompt(prompt: str) -> JsonSchema | None:
    marker = "JSON Schema:"
    idx = prompt.rfind(marker)
    if idx < 0:
        return None
    tail = prompt[idx + len(marker) :]
    start = tail.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(tail[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _resolve_ref(schema: JsonSchema, root: JsonSchema) -> JsonSchema:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    if not ref.startswith("#/"):
        raise CompileError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise CompileError(f"unresolvable $ref {ref!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise CompileError(f"$ref {ref!r} does not point at a schema object")
    merged = dict(node)
    merged.update({k: v for k, v in schema.items() if k != "$ref"})
    return merged


def _schema_type(schema: JsonSchema) -> str | None:
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        return non_null[0] if non_null else None
    return t


def _humanize(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").strip()


def _qname(path: tuple[str, ...]) -> str:
    return "__".join(path)


# --------------------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------------------


def compile_schema(
    schema: JsonSchema,
    *,
    schema_name: str = "response",
    system_prompt: str = "",
    profile_name: str | None = None,
    bool_threshold: float = 0.5,
    text_placeholder: str = DEFAULT_TEXT_PLACEHOLDER,
) -> Compiled:
    """Turn a JSON Schema for a judge verdict into Jev questions plus a reassembly plan."""
    profile_key = profile_name if profile_name is not None else schema_name
    profile = PROFILES.get(profile_key, {})
    plan: list[FieldPlan] = []
    questions: dict[str, dict[str, Any]] = {}

    root = schema
    _compile_object(
        schema,
        root,
        (),
        plan,
        questions,
        profile,
        system_prompt,
        text_placeholder,
    )
    if not any(p.decides for p in plan):
        raise CompileError("schema has no field Jev can decide (no boolean, enum, 0..1 number, or small integer range)")
    raw_gate_fields = profile.get("gate_fields")
    gate_fields = frozenset(raw_gate_fields) if raw_gate_fields else None
    return Compiled(
        questions=questions,
        plan=plan,
        schema_name=schema_name,
        profile=profile_key if profile else "generic",
        gate_fields=gate_fields,
    )


def _compile_object(
    schema: JsonSchema,
    root: JsonSchema,
    path: tuple[str, ...],
    plan: list[FieldPlan],
    questions: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    system_prompt: str,
    text_placeholder: str,
) -> None:
    schema = _resolve_ref(schema, root)
    props = schema.get("properties")
    if not isinstance(props, dict):
        raise CompileError(f"object at {'/'.join(path) or '<root>'} has no properties")
    derived_cfg = profile.get("derived", {})
    templates = profile.get("templates", {})
    for key, sub in props.items():
        if not isinstance(sub, dict):
            continue
        sub = _resolve_ref(sub, root)
        fpath = path + (key,)
        flat = ".".join(fpath)
        if key in derived_cfg and len(path) == 0:
            plan.append(FieldPlan(path=fpath, kind="derived", derive=derived_cfg[key]))
            continue
        if key in templates and len(path) == 0:
            plan.append(FieldPlan(path=fpath, kind="template", literal=templates[key], min_length=int(sub.get("minLength", 0) or 0)))
            continue
        if "const" in sub:
            plan.append(FieldPlan(path=fpath, kind="literal", literal=sub["const"]))
            continue
        stype = _schema_type(sub)
        if stype == "object" or (stype is None and "properties" in sub):
            _compile_object(sub, root, fpath, plan, questions, profile, system_prompt, text_placeholder)
            continue

        description = str(sub.get("description") or "")
        instructions = profile.get("instructions", {}).get(flat) or description or _humanize(key)
        enum = sub.get("enum")

        if isinstance(enum, list) and len(enum) == 1:
            plan.append(FieldPlan(path=fpath, kind="literal", literal=enum[0]))
            continue

        if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
            if len(enum) > 255:
                raise CompileError(f"{flat}: Jev choice supports at most 255 options, got {len(enum)}")
            criteria: dict[str, Any] = {v: None for v in enum}
            criteria.update(harvest_criteria(system_prompt, list(enum)))
            criteria.update(profile.get("extra_criteria", {}).get(flat, {}))
            qn = _qname(fpath)
            questions[qn] = {"type": "choice", "instructions": instructions, "criteria": criteria}
            plan.append(FieldPlan(path=fpath, kind="choice", question=qn, options=list(enum)))
            continue

        if isinstance(enum, list) and enum and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in enum):
            levels = sorted(enum)
            if len(levels) > 10:
                raise CompileError(f"{flat}: Jev score supports at most 10 levels, got {len(levels)}")
            qn = _qname(fpath)
            questions[qn] = {
                "type": "score",
                "instructions": instructions,
                "criteria": [f"{flat} = {lvl}" for lvl in levels],
            }
            plan.append(FieldPlan(path=fpath, kind="score", question=qn, options=levels))
            continue

        if stype == "boolean":
            qn = _qname(fpath)
            q: dict[str, Any] = {"type": "noul", "instructions": instructions}
            crit = profile.get("noul_criteria", {}).get(flat)
            if crit:
                q["criteria"] = crit
            questions[qn] = q
            plan.append(FieldPlan(path=fpath, kind="noul_bool", question=qn))
            continue

        if stype == "number":
            lo, hi = sub.get("minimum"), sub.get("maximum")
            if lo is not None and hi is not None and float(lo) == 0.0 and float(hi) == 1.0:
                qn = _qname(fpath)
                q = {"type": "noul", "instructions": instructions}
                crit = profile.get("noul_criteria", {}).get(flat)
                if crit:
                    q["criteria"] = crit
                questions[qn] = q
                plan.append(FieldPlan(path=fpath, kind="noul_prob", question=qn))
                continue
            plan.append(FieldPlan(path=fpath, kind="literal", literal=float(lo) if lo is not None else 0.0))
            continue

        if stype == "integer":
            lo, hi = sub.get("minimum"), sub.get("maximum")
            if lo is not None and hi is not None and 1 <= int(hi) - int(lo) <= 9:
                levels = list(range(int(lo), int(hi) + 1))
                qn = _qname(fpath)
                questions[qn] = {
                    "type": "score",
                    "instructions": instructions,
                    "criteria": [f"{flat} = {lvl}" for lvl in levels],
                }
                plan.append(FieldPlan(path=fpath, kind="score", question=qn, options=levels))
                continue
            plan.append(FieldPlan(path=fpath, kind="literal", literal=int(lo) if lo is not None else 0))
            continue

        if stype == "string":
            plan.append(
                FieldPlan(
                    path=fpath,
                    kind="template",
                    literal=text_placeholder,
                    min_length=int(sub.get("minLength", 0) or 0),
                )
            )
            continue

        if stype == "array":
            plan.append(FieldPlan(path=fpath, kind="literal", literal=[]))
            continue

        # Unknown or untyped: emit null so `required` still holds; schema validation may reject.
        plan.append(FieldPlan(path=fpath, kind="literal", literal=None))


# --------------------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------------------


def _set_path(obj: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = obj
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _get_path(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = obj
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def decode_answers(compiled: Compiled, answers: dict[str, Any], *, bool_threshold: float = 0.5) -> Decoded:
    """Reassemble a schema-shaped verdict from Jev's answers and compute a gate confidence.

    The gate confidence is the minimum confidence over the fields that matter for cascading:
    every decided field by default, or just ``compiled.gate_fields`` when the profile named a
    narrower set (see ``Compiled.gate_fields``). Every decided field's confidence is still
    recorded in ``evidence`` regardless, for logging and calibration audits.
    """
    verdict: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}
    field_confidences: dict[str, float] = {}
    raw_values: dict[str, Any] = {}

    # Pass 1: decided fields.
    for fp in compiled.plan:
        if not fp.decides:
            continue
        ans = answers.get(fp.question or "")
        if not isinstance(ans, dict):
            raise CompileError(f"Jev returned no answer for question {fp.question!r}")
        flat = ".".join(fp.path)
        if fp.kind in {"noul_bool", "noul_prob"}:
            p = float(ans.get("noul", 0.5))
            p = min(1.0, max(0.0, p))
            conf = abs(p - 0.5) * 2.0
            value: Any = (p >= bool_threshold) if fp.kind == "noul_bool" else p
            evidence[flat] = {"type": "noul", "p": p, "confidence": conf}
            raw_values[flat] = p
        elif fp.kind == "choice":
            choice = ans.get("choice")
            probs = ans.get("probabilities") or {}
            if choice not in fp.options:
                # Fall back to the best-probability option we know about.
                known = {k: float(v) for k, v in probs.items() if k in fp.options}
                choice = max(known, key=known.get) if known else fp.options[0]
            conf = float(ans.get("confidence", probs.get(choice, 0.0) if probs else 0.0))
            value = choice
            evidence[flat] = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": conf}
            raw_values[flat] = choice
        else:  # score
            score = float(ans.get("score", 0.0))
            idx = int(round(score))
            idx = min(max(idx, 0), len(fp.options) - 1)
            conf = float(ans.get("confidence", 0.0))
            value = fp.options[idx]
            evidence[flat] = {"type": "score", "score": score, "probabilities": ans.get("probabilities"), "confidence": conf}
            raw_values[flat] = score
        field_confidences[flat] = conf
        _set_path(verdict, fp.path, value)

    # Pass 2: literals, templates, derived.
    for fp in compiled.plan:
        if fp.decides:
            continue
        flat = ".".join(fp.path)
        if fp.kind == "literal":
            _set_path(verdict, fp.path, fp.literal)
        elif fp.kind == "template":
            text = _render_template(str(fp.literal), raw_values, verdict)
            if len(text) < fp.min_length:
                text = text.ljust(fp.min_length, ".")
            _set_path(verdict, fp.path, text)
        elif fp.kind == "derived":
            cfg = fp.derive or {}
            src = _get_path(verdict, tuple(str(cfg.get("from", "")).split(".")))
            value = cfg.get("default")
            if isinstance(src, str):
                prefix_map = cfg.get("prefix_map", {})
                for prefix, mapped in prefix_map.items():
                    if src == prefix or src.startswith(prefix):
                        value = mapped
                        break
                value = cfg.get("map", {}).get(src, value)
            _set_path(verdict, fp.path, value)
            evidence[flat] = {"type": "derived", "from": cfg.get("from"), "value": value}

    gated = field_confidences
    if compiled.gate_fields is not None:
        restricted = {name: conf for name, conf in field_confidences.items() if name in compiled.gate_fields}
        # A configured gate field that matched nothing (a profile/schema mismatch) is a bug in
        # the profile, not evidence of confidence — fall back to every decided field rather
        # than silently reporting 1.0.
        if restricted:
            gated = restricted
    confidence = min(gated.values()) if gated else 1.0
    return Decoded(verdict=verdict, confidence=confidence, evidence=evidence)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def _render_template(template: str, raw_values: dict[str, Any], verdict: dict[str, Any]) -> str:
    values: dict[str, Any] = {}
    # Flat keys first (top-level), so `{p_solve:.3f}` works for numeric raw values.
    for flat, raw in raw_values.items():
        values[flat.replace(".", "__")] = raw
    for key, val in verdict.items():
        values.setdefault(key, val)
    try:
        return template.format_map(_SafeDict(values))
    except (ValueError, TypeError):
        return template
