"""The capability rubric a `complexity` route asks Jev to forecast against.

Switchyard's capability classifier ships one rubric, written for *coding agents working in a
repository*: every rule is about validators, executable references, inspectable environments and
hidden repository state (vendor/switchyard/crates/libsy/src/prompts/capability-classifier/prompt.md).
A one-shot business request -- "write a cohort-retention query", "model this cap table" -- matches
none of those rules, so the forecast had nothing to anchor on. Across three studies and three
model rosters `complexity` was the one policy that misrouted, in both directions (docs/DECISION.md
sections 10, 13, 14).

`general` (the default) keeps Switchyard's whole mechanism -- the same output schema, the same
rule ids, the same threshold widening for uncertain / unsupported verdicts -- and swaps only the
rubric for one about the *task itself*. Teams can sharpen it in plain English with `weak_when` /
`strong_when`, the same kind of concrete criteria that make `intent` routing accurate.
`coding_agent` keeps Switchyard's original rubric, for routes that really do serve an agent in a
repository with tools and a verifier.
"""

from __future__ import annotations

from .schema import ComplexityPolicy

_GENERAL_RULES = {
    "SUP-1": "The answer is a direct lookup, restatement, or extraction of something stated in the request.",
    "SUP-2": "It is a single, well-defined calculation, conversion, or formula with every input given.",
    "SUP-3": (
        "It is a short, standard transformation: a simple query over one table, a rename, a formatting, "
        "naming, or style fix, a one-paragraph summary, or a short templated message."
    ),
    "SUP-4": "It asks for a common fact, definition, or convention that fits in a few sentences with no trade-offs to weigh.",
    "SUP-5": "It is a routine judgement with an obvious answer that a careful junior would get right.",
    "UNC-1": "It takes a few steps or conditions, but each one is standard and the result is easy to check.",
    "UNC-2": "It is ambiguous or underspecified enough that a confident, plausible answer could still be the wrong one.",
    "LIM-1": (
        "Correctness depends on multi-step reasoning across several interacting constraints: financial "
        "modelling or reconciliation, multi-table or windowed queries with edge cases, concurrency, "
        "distributed-systems or security failure analysis, experiment or causal analysis, or planning "
        "under competing constraints."
    ),
    "LIM-2": (
        "A subtle, plausible-but-wrong answer is likely: hidden edge cases, double counting, off-by-one "
        "or rounding traps, race conditions, or a judgement call where real expertise changes the answer."
    ),
}

_GENERAL_PROMPT = """You are a probability forecaster for a model router. You receive a request and
the capability card below, and forecast one binary event:

SUCCESS means the efficient (cheaper, smaller) model answers the whole request fully and correctly
in one attempt, so that a careful expert reviewing the answer would accept it. FAILURE means any
other outcome.

Judge the task itself: how many steps it takes, how many constraints interact, and how likely a
plausible-looking answer is to be wrong. A short request can be hard and a long one easy. Do not
reward or penalize a request for its length, its domain, or its formality.

# Assessment procedure

1. State the crux: the hardest thing the answer must get right.
2. Select the one rule below that best describes the crux. Use primary_rule=none and
   capability_boundary=unmatched when no rule applies. Rule ids are opaque labels.
3. Estimate p_solve last: the probability of SUCCESS, not a route recommendation or a cost
   judgement. Use the full range when justified.

# Efficient-model capability card

{rules}

# Output

Return exactly one JSON object matching the response schema supplied with the request. Do not
include markdown or commentary. p_solve must be between 0.00 and 1.00.
"""

_BOUNDARY = {"SUP": "supported", "UNC": "uncertain", "LIM": "unsupported"}


def complexity_prompt(policy: ComplexityPolicy) -> str | None:
    """The prompt override for this route, or None to keep Switchyard's packaged rubric."""
    if policy.rubric == "coding_agent":
        return None
    rules = dict(_GENERAL_RULES)
    # A team's own words replace the most generic rule on each side: they know their traffic.
    if policy.weak_when:
        rules["SUP-5"] = " ".join(policy.weak_when.split())
    if policy.strong_when:
        rules["LIM-2"] = " ".join(policy.strong_when.split())
    lines = [f"- {rule} [{_BOUNDARY[rule[:3]]}]: {text}" for rule, text in rules.items()]
    return _GENERAL_PROMPT.format(rules="\n".join(lines))


__all__ = ["complexity_prompt"]
