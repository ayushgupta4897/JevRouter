"""Cache-aware switching: never throw away a warm prompt cache unless the switch pays for itself.

Provider prompt caches are per model (and per provider). The moment a session moves to a
different model, the next turn pays the full uncached price on the whole conversation so far,
so a router that re-decides every turn and bounces between tiers can cost more than simply
staying on the best model throughout. This module sits after a route's own policy decision
and can hold the session on its current model instead:

* **Downgrades are priced exactly.** Moving to a cheaper model only happens if, over the next
  ``horizon_turns``, the cheaper model -- paying a cold first turn -- still costs less than
  staying warm on the current one (by at least ``switch_margin``). Bounce-backs are where
  per-request routers leak money, and this part is fully computable from prices and tokens.
* **Upgrades are quality-driven, but need a decisive verdict.** A policy asking for a stronger
  model is making a quality call, not a cost one, so upgrades are allowed -- unless the judge's
  verdict is too borderline (below ``upgrade_min_confidence``) to justify dropping a warm cache.
  That's hysteresis around the policy's threshold, which is where oscillation comes from.
* **A cold cache is free to abandon.** First turn, idle past the provider's TTL, or a prefix
  below the provider's minimum cacheable length: there's nothing to protect, the policy wins.
* **A switch back isn't always cold.** If the gateway reports that the target model still holds
  an earlier prefix of this session (``other_warm_caches``), only the turns since then are
  priced as uncached.
* **A judge outage mid-session holds the current model** rather than falling open to the
  route's safe default and paying a cold switch on every request until the judge recovers.

Stateless by design: the caller (the AI Gateway) says which model served the previous turn and,
ideally, how much of the conversation that model has cached. No session table lives in this process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .algorithms import CompiledRoute
from .decide import Decision
from .schema import CacheConfig, ModelPricing

CHARS_PER_TOKEN = 4  # rough, provider-agnostic; session.cached_prefix_tokens gives exact numbers


class SessionState(BaseModel):
    """What the caller knows about the session so far. Every field is optional: with no
    ``current_model`` there is no cache to protect and routing behaves as it always has."""

    model_config = ConfigDict(extra="forbid")

    current_model: str | None = Field(default=None, description="Model that served the previous turn.")
    current_client: str | None = Field(default=None, description="Its client, if the route has more than one.")
    cached_prefix_tokens: int | None = Field(
        default=None, ge=0,
        description="How much of this conversation the current model now holds in its prompt "
        "cache: everything the previous turn read from cache *plus* what it wrote. On OpenAI that "
        "is the previous turn's usage.prompt_tokens (if >= 1,024); on Anthropic, "
        "cache_read_input_tokens + cache_creation_input_tokens. Omit to assume the whole prefix is "
        "warm. Don't pass the read count alone -- a cold turn reads 0 but leaves the cache warm.",
    )
    idle_seconds: float | None = Field(default=None, ge=0, description="Seconds since the previous turn.")
    remaining_turns: int | None = Field(default=None, ge=1, description="Overrides the route's horizon_turns.")
    other_warm_caches: dict[str, int] = Field(
        default_factory=dict,
        description="Other models this session used recently, within their cache TTL, mapped to the "
        "prompt tokens still cached there. A switch *back* to one of them isn't a cold start: the "
        "old prefix is still warm, only the turns since then are new.",
    )

    @field_validator("other_warm_caches")
    @classmethod
    def _non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(tokens < 0 for tokens in value.values()):
            raise ValueError("other_warm_caches token counts must be >= 0")
        return value


# ---------------------------------------------------------------------------- estimation


def _message_chars(message: dict[str, Any]) -> int:
    content = message.get("content")
    chars = 0
    if isinstance(content, str):
        chars += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chars += len(part["text"])
    if message.get("tool_calls"):
        chars += len(json.dumps(message["tool_calls"]))
    return chars


def estimate_tokens(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """(prefix_tokens, new_tokens): everything before the latest message is the cacheable
    prefix; the latest message is new input either way."""
    if not messages:
        return 0, 0
    prefix = sum(_message_chars(m) for m in messages[:-1]) // CHARS_PER_TOKEN
    return prefix, _message_chars(messages[-1]) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------- pricing


def turn_cost(p: ModelPricing, *, prefix: int, cached: int, new: int, output: int) -> float:
    """USD for one turn: the cached part of the prefix at the cached rate, every other input
    token at the input rate -- or at the cache-write rate, where the provider charges a premium
    to cache what it hasn't seen (Anthropic; OpenAI GPT-5.6+) and the prompt is long enough to
    be cached at all -- plus the output."""
    cached = min(cached, prefix)
    uncached = prefix - cached + new
    cacheable = p.cache_write is not None and prefix + new >= p.min_cacheable_tokens
    write_rate = p.cache_write if cacheable else p.input
    return (cached * p.cached_input + uncached * write_rate + output * p.output) / 1e6


def horizon_cost(p: ModelPricing, *, prefix: int, cached_now: int, new: int, output: int, turns: int) -> float:
    """USD over the next `turns` turns on one model. Each turn's prompt becomes the next turn's
    cached prefix (if long enough to cache); its output is appended uncached."""
    total = 0.0
    cached = cached_now
    for _ in range(turns):
        total += turn_cost(p, prefix=prefix, cached=cached, new=new, output=output)
        prompt = prefix + new
        cached = prompt if prompt >= p.min_cacheable_tokens else 0
        prefix = prompt + output
    return total


# ---------------------------------------------------------------------------- the gate


@dataclass
class CacheReport:
    reason: str
    held_current_model: bool
    policy_model: str
    current_model: str | None
    prefix_tokens: int | None = None
    cached_prefix_tokens: int | None = None
    target_cached_tokens: int | None = None
    horizon_turns: int | None = None
    stay_cost_usd: float | None = None
    switch_cost_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if v is not None}
        for key in ("stay_cost_usd", "switch_cost_usd"):
            if key in out:
                out[key] = round(out[key], 6)
        return out


def gate(
    decision: Decision,
    compiled: CompiledRoute,
    session: SessionState | None,
    pricing: dict[str, ModelPricing],
    config: CacheConfig,
    messages: list[dict[str, Any]],
) -> CacheReport:
    """Decide whether to follow the policy's pick or hold the session's current model."""
    pick = decision.selected_model
    current = session.current_model if session else None

    def follow(reason: str, **extra: Any) -> CacheReport:
        return CacheReport(reason=reason, held_current_model=False, policy_model=pick, current_model=current, **extra)

    def hold(reason: str, **extra: Any) -> CacheReport:
        return CacheReport(reason=reason, held_current_model=True, policy_model=pick, current_model=current, **extra)

    if not config.enabled:
        return follow("disabled")
    if not current:
        return follow("no_session")

    current_ref = compiled.model_by_id.get(current)
    if current_ref is None or (session.current_client and session.current_client != current_ref.client):
        return follow("current_not_in_route")  # can't hold a model this route doesn't serve
    if pick == current and decision.selected_client == current_ref.client:
        return follow("same_model")

    cur_p, new_p = pricing.get(current), pricing.get(pick)
    if cur_p is None or new_p is None:
        return follow("no_pricing")

    prefix, new = estimate_tokens(messages)
    if session.cached_prefix_tokens is not None:
        prefix = max(prefix, session.cached_prefix_tokens)  # a real count beats a chars/4 estimate
    if session.idle_seconds is not None and session.idle_seconds >= cur_p.cache_ttl_seconds:
        return follow("cache_expired", prefix_tokens=prefix)
    if prefix < cur_p.min_cacheable_tokens:
        return follow("prefix_below_cache_minimum", prefix_tokens=prefix)

    cached = min(session.cached_prefix_tokens, prefix) if session.cached_prefix_tokens is not None else prefix
    if cached < cur_p.min_cacheable_tokens:
        return follow("prefix_below_cache_minimum", prefix_tokens=prefix, cached_prefix_tokens=cached)

    if decision.judge_error:
        # No real verdict this turn: the policy only fell open to its safe default. With a warm
        # session on a model a real verdict chose earlier, staying is both the cheaper and the
        # better-informed choice.
        return hold("judge_unavailable_hold", prefix_tokens=prefix, cached_prefix_tokens=cached)

    turns = session.remaining_turns or config.horizon_turns
    out = config.expected_output_tokens
    # A model the session left recently may still hold most of the prefix (LiteLLM measured
    # 97%+ of switch-backs landing warm) -- only a genuinely cold target pays for all of it.
    target_warm = min(session.other_warm_caches.get(pick, 0), prefix)
    if target_warm < new_p.min_cacheable_tokens:
        target_warm = 0
    stay = horizon_cost(cur_p, prefix=prefix, cached_now=cached, new=new, output=out, turns=turns)
    switch = horizon_cost(new_p, prefix=prefix, cached_now=target_warm, new=new, output=out, turns=turns)
    numbers = dict(
        prefix_tokens=prefix, cached_prefix_tokens=cached, target_cached_tokens=target_warm or None,
        horizon_turns=turns, stay_cost_usd=stay, switch_cost_usd=switch,
    )

    # Direction by what a warm turn costs on each model: the pricier one is the "upgrade".
    cur_warm = turn_cost(cur_p, prefix=prefix, cached=prefix, new=new, output=out)
    new_warm = turn_cost(new_p, prefix=prefix, cached=prefix, new=new, output=out)
    if new_warm > cur_warm:
        confidence = decision.judge_confidence
        if confidence is not None and confidence < config.upgrade_min_confidence:
            return hold("upgrade_verdict_too_borderline", **numbers)
        return follow("upgrade", **numbers)

    if stay - switch >= config.switch_margin * stay:
        return follow("downgrade_saves_money", **numbers)
    return hold("downgrade_not_worth_losing_cache", **numbers)


def apply(
    decision: Decision,
    compiled: CompiledRoute,
    session: SessionState | None,
    pricing: dict[str, ModelPricing],
    config: CacheConfig,
    messages: list[dict[str, Any]],
) -> Decision:
    """The policy's decision with the cache gate applied -- `selected_model` is the final choice,
    and `cache` always says what the gate did and why, so it's never a silent override."""
    report = gate(decision, compiled, session, pricing, config, messages)
    if not report.held_current_model:
        return replace(decision, cache=report.as_dict())

    current = compiled.model_by_id[report.current_model]
    ranked = [decision.selected_model, *decision.fallback_model_ids]
    fallbacks = [compiled.model_by_id[m] for m in ranked if m != current.id and m in compiled.model_by_id]
    return replace(
        decision,
        selected_model=current.id,
        selected_client=current.client,
        bucket=compiled.bucket_by_model_id.get(current.id) if compiled.bucket_by_model_id else None,
        fallback_model_ids=[ref.id for ref in fallbacks],
        fallback_clients=[ref.client for ref in fallbacks],
        cache=report.as_dict(),
    )


__all__ = ["CacheReport", "SessionState", "apply", "estimate_tokens", "gate", "horizon_cost", "turn_cost"]
