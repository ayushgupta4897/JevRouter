"""Grading logic tests. `test_last_number_not_first_*` are a regression for a real bug found
against the live API: a step-by-step response restates its inputs before its answer, so the
first number in the text is almost always an input being echoed, not the answer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_eval import extract_last_number, grade_local  # noqa: E402


def test_last_number_not_first_on_a_real_captured_response():
    # Captured verbatim from gpt-5.6-luna against finance-hard-1 (see docs/DECISION.md). The
    # model's answer, $180,000, is exactly right; extracting the first number ("40", from
    # "40%") wrongly failed this case for every model tested until this fix.
    text = (
        r"Gross profit: \(40\% \times \$1{,}200{,}000 = \$480{,}000\)"
        "\n\n"
        r"Operating expenses: \(25\% \times \$1{,}200{,}000 = \$300{,}000\)"
        "\n\n"
        r"**Operating income = \$480,000 - \$300,000 = \(\boxed{\$180,000}\)**"
    )
    assert extract_last_number(text) == 180000.0
    assert grade_local({"grader": "numeric_equals", "expected": 180000, "tolerance": 1000}, text) is True


def test_last_number_handles_latex_thousands_grouping_in_the_final_answer():
    # Captured verbatim from gpt-5.6-sol against finance-hard-1: the final boxed answer itself
    # used LaTeX's `{,}` grouping ("\boxed{\$180{,}000}"), not a plain comma. Stripping only the
    # comma character left "180{}000" -- the braces split one number into two digit runs ("180"
    # and "000"), so the "last number" silently became 0 instead of 180000. This is a real
    # answer, exactly right, that was marked wrong twice before this was caught.
    text = (
        r"Gross profit: \(1{,}200{,}000 \times 40\% = 480{,}000\)"
        "\n\n"
        r"Operating expenses: \(1{,}200{,}000 \times 25\% = 300{,}000\)"
        "\n\n"
        r"Operating income: \(480{,}000 - 300{,}000 = \boxed{\$180{,}000}\)"
    )
    assert extract_last_number(text) == 180000.0
    assert grade_local({"grader": "numeric_equals", "expected": 180000, "tolerance": 1000}, text) is True


def test_last_number_still_works_on_a_terse_reply():
    assert extract_last_number("445.99") == 445.99
    assert extract_last_number("The total due is $445.99.") == 445.99


def test_last_number_handles_commas_and_negative_values():
    assert extract_last_number("Revenue was $1,200,000 and the loss was -3,500.") == -3500.0


def test_last_number_returns_none_with_no_digits():
    assert extract_last_number("No numbers here at all.") is None


def test_grade_local_numeric_equals_respects_tolerance():
    case = {"grader": "numeric_equals", "expected": 100.0, "tolerance": 0.5}
    assert grade_local(case, "The answer is 100.4") is True
    assert grade_local(case, "The answer is 101.0") is False


def test_grade_local_contains_all_is_case_insensitive_and_requires_every_string():
    case = {"grader": "contains_all", "expected": ["INV-88213", "August"]}
    assert grade_local(case, "invoice inv-88213, dated in august") is True
    assert grade_local(case, "invoice inv-88213 only") is False


def test_grade_local_returns_none_for_jev_noul_graded_elsewhere():
    # jev_noul cases are graded through the judge, not grade_local -- it must not silently
    # invent a verdict for a grader type it doesn't handle.
    assert grade_local({"grader": "jev_noul", "rubric": "..."}, "anything") is None
