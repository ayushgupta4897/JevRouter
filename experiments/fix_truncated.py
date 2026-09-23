#!/usr/bin/env python3
"""One-off patch: re-call only the items that came back with an EMPTY visible response at the
1200-token cap (hidden reasoning exhausted the budget) -- 14 of 216, all reasoning-model calls on
deliberately extreme-depth prompts (full mathematical proofs, multi-scenario financial models,
heavily-constrained SQL). Re-running the whole 216-item suite a third time to fix 14 items would
waste real budget on the 202 that are already correct; this touches only the affected entries.

If a retry at 3000 tokens is STILL empty, it's left as an explicit completion_error (excluded
from cost/quality accounting) rather than silently keeping a $0-content row that would corrupt
the numbers -- some of this dataset's items ask for genuinely exhaustive derivations that can
legitimately exceed even a generous token budget, and that's a real, reportable limitation, not
a bug to paper over.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "evals"))
from models import get as get_pricing  # noqa: E402

from run_experiments import BASE_URL, call_model, load_dataset  # noqa: E402

RETRY_TOKENS = 3000
TARGETS = {
    "financial-analysis": ["fin-06", "fin-07", "fin-13", "fin-18"],
    "general-assistant": ["gen-08", "gen-09", "gen-10", "gen-12", "gen-18"],
    "sql-query-generation": ["sql-06", "sql-08", "sql-11", "sql-16", "sql-18"],
}


async def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    async with httpx.AsyncClient() as http:
        for name, ids in TARGETS.items():
            path = ROOT / "results" / f"{name}.json"
            data = json.loads(path.read_text())
            dataset = {item["id"]: item["text"] for item in load_dataset(name)}
            by_id = {i["item_id"]: i for i in data["items"]}

            for rid in ids:
                item = by_id[rid]
                if (item.get("completion_text") or "").strip():
                    print(f"SKIP (already fixed)  {name}/{rid}")
                    continue
                model = item["selected_model"]
                prompt = dataset[rid]
                text, in_tok, out_tok, lat = await call_model(http, api_key, model, prompt, max_tokens=RETRY_TOKENS, timeout=180.0)
                if text.strip():
                    cost = get_pricing(model).cost(in_tok, out_tok)
                    item["completion_input_tokens"], item["completion_output_tokens"] = in_tok, out_tok
                    item["completion_cost_usd"], item["completion_latency_ms"] = cost, lat
                    item["completion_text"] = text
                    item["completion_error"] = None
                    print(f"FIXED   {name}/{rid}: {out_tok} tokens, {len(text)} chars")
                else:
                    item["completion_error"] = f"empty visible response even after {RETRY_TOKENS} tokens -- reasoning exhausted the budget"
                    item["completion_text"] = None
                    print(f"STILL EMPTY  {name}/{rid} at {RETRY_TOKENS} tokens -- marked as error, excluded from cost/quality accounting")

            path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
