"""Pricing for the models this eval run compares.

Verified against OpenAI's published pricing (developers.openai.com/api/docs/pricing) and
corroborating trackers, 2026-09-22. Update this table before trusting a cost column older than
that -- OpenAI cut GPT-5.6 Sol's price by over 20% once already (2026-08-24).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    model: str
    input_per_million: float
    output_per_million: float
    note: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1e6 * self.input_per_million + output_tokens / 1e6 * self.output_per_million


CATALOG: dict[str, ModelPricing] = {
    "gpt-5.6-luna": ModelPricing("gpt-5.6-luna", 0.20, 1.20, "cheapest GPT-5.6 tier"),
    "gpt-5.6-terra": ModelPricing("gpt-5.6-terra", 2.00, 12.00, "balanced GPT-5.6 tier"),
    "gpt-5.6-sol": ModelPricing("gpt-5.6-sol", 5.00, 30.00, "flagship GPT-5.6 tier; promo pricing through 2026-11-21"),
    "gpt-6-astra": ModelPricing("gpt-6-astra", 10.00, 50.00, "new flagship, launched 2026-09-03"),
    # Jev itself, for completeness when comparing router-judge cost, not answer quality.
    "jev-latest": ModelPricing("jev-latest", 0.042, 0.0, "TypeSafe System One; output is unmetered"),
}


def get(model: str) -> ModelPricing:
    if model not in CATALOG:
        raise KeyError(f"no pricing entry for {model!r}; add one to evals/models.py before running it")
    return CATALOG[model]
