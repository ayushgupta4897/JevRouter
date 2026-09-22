#!/usr/bin/env python3
"""Rebuild the summary table (and quality-parity spot-check tally) from experiments/results/*.json
without re-running anything -- useful after a targeted patch like fix_truncated.py, and for
regenerating the numbers a report cites straight from the committed raw data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_experiments import EXPERIMENTS, ExperimentResult, ItemResult, SpotCheck, print_summary  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.compiler import load_team_file  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "evals"))
from models import get as get_pricing  # noqa: E402


def load_result(name: str, route_name: str) -> ExperimentResult:
    data = json.loads((ROOT / "results" / f"{name}.json").read_text())
    team = load_team_file(ROOT / "teams" / f"{name}.yaml")
    route = next(r for r in team.routes if r.name == route_name)
    compiled = compile_route_algorithm(route)
    roster_ids = sorted(compiled.model_by_id, key=lambda m: get_pricing(m).output_per_million)
    result = ExperimentResult(name=data["name"], route_name=data["route_name"], policy=data["policy"], roster_ids=roster_ids)
    result.items = [ItemResult(**i) for i in data["items"]]
    result.spot_checks = [SpotCheck(**s) for s in data["spot_checks"]]
    return result


if __name__ == "__main__":
    results = [load_result(name, route_name) for name, route_name in EXPERIMENTS]
    print_summary(results)
    excluded = sum(1 for r in results for i in r.items if i.completion_error)
    print(f"\nitems excluded from cost accounting (no usable completion): {excluded} of {sum(len(r.items) for r in results)}")
