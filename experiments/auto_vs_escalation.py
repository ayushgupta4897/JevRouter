"""`auto` (Switchyard's stage router) vs. `escalation`, on the same real transcripts.

Both start cheap and move up when a run is going badly, but they read the run differently:

* `auto` pattern-matches the *tool traffic*: error severity in tool results, spinning,
  exploring vs. producing. No judge call (~1 ms). Blind to anything that isn't a tool call.
* `escalation` asks Jev to read the *transcript* for a repeated-failure pattern. One judge call
  (~0.2 s on a warm connection). Works on any conversation, tools or not.

Six transcripts, each routed by both. Writing this is what exposed the bug in DECISION.md
section 15: the escalation judge had been seeing only the opening task on real agent transcripts.

    set -a; . ./.env; set +a
    .venv/bin/python experiments/auto_vs_escalation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "routerctl" / "src"))
sys.path.insert(0, str(ROOT.parent / "jevjudge" / "src"))

from jevjudge.cascade import build_judge_from_env  # noqa: E402
from routerctl.algorithms import compile_route_algorithm  # noqa: E402
from routerctl.decide import decide  # noqa: E402
from routerctl.messages import to_libsy_messages  # noqa: E402
from routerctl.schema import Route  # noqa: E402

TIERS = {"cheap": {"id": "cheap", "client": "openai"}, "strong": {"id": "strong", "client": "openai"}}
AUTO = compile_route_algorithm(Route.model_validate({"name": "a", "policy": "auto", "models": {"efficient": TIERS["cheap"], "capable": TIERS["strong"]}}))
ESCALATION = compile_route_algorithm(Route.model_validate({"name": "e", "policy": "escalation", "confirmations": 2,
                                                            "models": {"weak": TIERS["cheap"], "strong": TIERS["strong"]}}))


def tool_turn(i: int, name: str, args: dict, result: str) -> list[dict]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": result},
    ]


TRACEBACK = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'serde'"


def cases() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    m = [{"role": "user", "content": "Fix the failing build."}]
    for i in range(3):
        m += tool_turn(i, "bash", {"command": "python -m app"}, TRACEBACK)
    out["A. coding agent, same traceback 3x"] = m

    m = [{"role": "user", "content": "Add the /health endpoint and tests."}]
    m += tool_turn(0, "write_file", {"path": "src/server.py", "content": "..."}, "wrote 42 lines")
    m += tool_turn(1, "edit", {"path": "src/server.py", "old": "a", "new": "b"}, "edited")
    m += tool_turn(2, "bash", {"command": "pytest -q"}, "3 passed in 0.12s")
    out["B. coding agent, edits land, tests pass"] = m

    m = [{"role": "user", "content": "Investigate the suspicious login from 185.220.101.4 on the payments admin account."}]
    for i, src in enumerate(["firewall", "ids", "endpoint"]):
        m += tool_turn(i, "query_siem", {"source": src, "ip": "185.220.101.4"}, f"0 events matched in {src} logs for the last 24h.")
    out["C. security agent, custom tool, inconclusive 3x"] = m

    out["D. plain chat, hard question"] = [{"role": "user", "content": "Prove that the square root of 2 is irrational, rigorously."}]

    out["E. support chatbot, same broken promise 3x, no tools"] = [
        {"role": "user", "content": "Help me get a refund for order 7731."},
        {"role": "assistant", "content": "I've raised a refund request, you'll get it in 3-5 days."},
        {"role": "user", "content": "It's been 9 days, nothing came. You said 3-5 days."},
        {"role": "assistant", "content": "Sorry about that, I've raised the refund request again."},
        {"role": "user", "content": "This is the third time. Same promise every time and no money."},
    ]

    m = [{"role": "user", "content": "Fix the failing build."}]
    m += tool_turn(0, "bash", {"command": "python -m app"}, TRACEBACK)
    m += tool_turn(1, "bash", {"command": "pip install serde-lib && python -m app"}, "Server listening on :8000")
    out["F. coding agent, one failure then fixed"] = m
    return out


async def main() -> None:
    judge = build_judge_from_env()
    rows = []
    for name, messages in cases().items():
        request = {"model": "x", "stream": False, "messages": to_libsy_messages(messages)}
        a = await decide(AUTO, judge, request)
        e = await decide(ESCALATION, judge, request)
        rows.append({"case": name, "auto": a.selected_model, "auto_latency_ms": round(a.latency_ms, 1),
                     "escalation": e.selected_model, "escalation_confidence": e.judge_confidence,
                     "escalation_latency_ms": round(e.latency_ms, 1), "judge_error": e.judge_error})
        print(f"{name:<54} auto -> {a.selected_model:<7} escalation -> {e.selected_model:<7} (conf {e.judge_confidence})")
    await judge.aclose()
    out = ROOT / "results" / "auto-vs-escalation.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"-> {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
