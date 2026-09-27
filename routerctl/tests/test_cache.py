"""Cache-aware switching: the pricing math, every gate outcome, the Decision rewrite, and a
multi-turn simulation showing the gate stops a flip-flopping policy from losing money.

Prices here are synthetic test values chosen to exercise specific economics, not real
provider prices -- real ones live in clients.yaml."""

from __future__ import annotations

import pytest

from routerctl.algorithms import compile_route_algorithm
from routerctl.cache import SessionState, apply, estimate_tokens, gate, horizon_cost, turn_cost
from routerctl.decide import Decision
from routerctl.schema import CacheConfig, ModelPricing, Route

CHEAP = ModelPricing(input=0.2, cached_input=0.02, output=1.2)
MID = ModelPricing(input=2.0, cached_input=0.2, output=12.0)
PREMIUM = ModelPricing(input=5.0, cached_input=0.5, output=30.0)
PRICING = {"m-cheap": CHEAP, "m-mid": MID, "m-premium": PREMIUM}


def complexity_route(weak: str, strong: str, **cache) -> Route:
    data = {"name": "r", "policy": "complexity", "models": {"weak": {"id": weak, "client": "openai"}, "strong": {"id": strong, "client": "openai"}}}
    if cache:
        data["cache"] = cache
    return Route.model_validate(data)


def decision(model: str, *, confidence: float | None = 0.9, judge_error: str | None = None, fallbacks: list[str] | None = None) -> Decision:
    return Decision(
        route="r", policy="complexity", selected_model=model, selected_client="openai", bucket=None,
        fallback_model_ids=fallbacks or [], fallback_clients=["openai"] * len(fallbacks or []),
        judge_source="jev" if confidence is not None else None, judge_confidence=confidence,
        judge_error=judge_error, latency_ms=1.0, outcome_id=None,
    )


def conversation(prefix_chars: int, new_chars: int = 400) -> list[dict]:
    # prefix_chars / 4 ~= prefix tokens (see cache.CHARS_PER_TOKEN)
    return [
        {"role": "user", "content": "x" * (prefix_chars // 2)},
        {"role": "assistant", "content": "y" * (prefix_chars - prefix_chars // 2)},
        {"role": "user", "content": "z" * new_chars},
    ]


LONG = conversation(400_000)   # ~100K-token prefix: cache dominates
SHORT = conversation(2_000)    # ~500-token prefix: below the 1024-token cache minimum


# ------------------------------------------------------------------------ pricing math


def test_turn_cost_warm_uses_the_cached_rate():
    assert turn_cost(MID, prefix=100_000, cached=100_000, new=0, output=0) == pytest.approx(100_000 * 0.2 / 1e6)


def test_turn_cost_cold_pays_full_input_on_the_whole_prefix():
    assert turn_cost(MID, prefix=100_000, cached=0, new=0, output=0) == pytest.approx(100_000 * 2.0 / 1e6)


def test_turn_cost_cache_write_premium_applies_to_a_cold_cacheable_prefix():
    anthropic_like = ModelPricing(input=3.0, cached_input=0.3, output=15.0, cache_write=3.75)
    assert turn_cost(anthropic_like, prefix=100_000, cached=0, new=0, output=0) == pytest.approx(100_000 * 3.75 / 1e6)


def test_turn_cost_no_write_premium_below_the_cache_minimum():
    anthropic_like = ModelPricing(input=3.0, cached_input=0.3, output=15.0, cache_write=3.75)
    assert turn_cost(anthropic_like, prefix=500, cached=0, new=0, output=0) == pytest.approx(500 * 3.0 / 1e6)


def test_turn_cost_write_premium_covers_the_new_message_too():
    # OpenAI GPT-5.6+ / Anthropic bill every input token they haven't cached yet as a write
    openai_like = ModelPricing(input=2.0, cached_input=0.2, output=12.0, cache_write=2.5)
    cost = turn_cost(openai_like, prefix=10_000, cached=10_000, new=2_000, output=0)
    assert cost == pytest.approx((10_000 * 0.2 + 2_000 * 2.5) / 1e6)


def test_turn_cost_cached_is_capped_at_the_prefix():
    assert turn_cost(MID, prefix=1_000, cached=5_000, new=0, output=0) == pytest.approx(1_000 * 0.2 / 1e6)


def test_horizon_cost_cold_start_only_pays_uncached_once():
    one = horizon_cost(MID, prefix=100_000, cached_now=0, new=100, output=500, turns=1)
    three = horizon_cost(MID, prefix=100_000, cached_now=0, new=100, output=500, turns=3)
    # the second and third turns are warm, so they cost far less than the cold first one
    assert three - one < one


def test_horizon_cost_warm_start_is_cheaper_than_cold_start():
    warm = horizon_cost(MID, prefix=100_000, cached_now=100_000, new=100, output=500, turns=5)
    cold = horizon_cost(MID, prefix=100_000, cached_now=0, new=100, output=500, turns=5)
    assert warm < cold


def test_estimate_tokens_splits_prefix_from_the_latest_message():
    prefix, new = estimate_tokens([{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b" * 400}, {"role": "user", "content": "c" * 40}])
    assert (prefix, new) == (200, 10)


def test_estimate_tokens_counts_text_parts_and_tool_calls():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a" * 400}, {"type": "image_url", "image_url": {"url": "data:..."}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "user", "content": "hi"},
    ]
    prefix, _ = estimate_tokens(messages)
    assert prefix > 100  # text part + serialized tool call, image data ignored


# ------------------------------------------------------------------------ gate outcomes


def compiled_mid_premium(**cache):
    return compile_route_algorithm(complexity_route("m-mid", "m-premium", **cache))


def test_disabled_follows_the_policy():
    route = complexity_route("m-mid", "m-premium", enabled=False)
    r = gate(decision("m-mid"), compile_route_algorithm(route), SessionState(current_model="m-premium"), PRICING, route.cache, LONG)
    assert (r.reason, r.held_current_model) == ("disabled", False)


@pytest.mark.parametrize("session", [None, SessionState()])
def test_no_session_follows_the_policy(session):
    r = gate(decision("m-mid"), compiled_mid_premium(), session, PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("no_session", False)


def test_same_model_is_trivially_followed():
    r = gate(decision("m-premium"), compiled_mid_premium(), SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    assert r.reason == "same_model"


def test_current_model_outside_this_route_cannot_be_held():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-unknown"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("current_not_in_route", False)


def test_same_model_on_a_different_client_is_a_different_cache():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", current_client="openrouter"), PRICING, CacheConfig(), LONG)
    assert r.reason == "current_not_in_route"


def test_missing_pricing_follows_the_policy_rather_than_guessing():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium"), {"m-mid": MID}, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("no_pricing", False)


def test_idle_past_the_ttl_means_the_cache_is_already_gone():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", idle_seconds=301), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("cache_expired", False)


def test_idle_within_the_ttl_still_protects_the_cache():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", idle_seconds=120), PRICING, CacheConfig(), LONG)
    assert r.reason != "cache_expired"


def test_a_prefix_below_the_provider_minimum_was_never_cached():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium"), PRICING, CacheConfig(), SHORT)
    assert (r.reason, r.held_current_model) == ("prefix_below_cache_minimum", False)


def test_reported_cached_tokens_below_the_minimum_means_nothing_to_protect():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", cached_prefix_tokens=200), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("prefix_below_cache_minimum", False)


def test_judge_outage_holds_the_warm_session_instead_of_falling_open():
    r = gate(decision("m-premium", confidence=None, judge_error="unreachable"), compiled_mid_premium(), SessionState(current_model="m-mid"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("judge_unavailable_hold", True)


def test_confident_upgrade_is_allowed_even_though_it_drops_the_cache():
    r = gate(decision("m-premium", confidence=0.8), compiled_mid_premium(), SessionState(current_model="m-mid"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("upgrade", False)
    assert r.switch_cost_usd > r.stay_cost_usd  # it does cost more -- quality is the reason


def test_borderline_upgrade_holds_the_warm_cache():
    r = gate(decision("m-premium", confidence=0.1), compiled_mid_premium(), SessionState(current_model="m-mid"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("upgrade_verdict_too_borderline", True)


def test_upgrade_without_a_judge_trusts_the_policys_own_signals():
    # auto (stage_router) never calls a judge -- confidence is None, not low
    r = gate(decision("m-premium", confidence=None), compiled_mid_premium(), SessionState(current_model="m-mid"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("upgrade", False)


def test_upgrade_min_confidence_is_configurable():
    config = CacheConfig(upgrade_min_confidence=0.95)
    r = gate(decision("m-premium", confidence=0.8), compiled_mid_premium(), SessionState(current_model="m-mid"), PRICING, config, LONG)
    assert r.reason == "upgrade_verdict_too_borderline"


def test_downgrade_that_loses_money_over_the_horizon_is_held():
    # premium warm (~$0.5/M on 100K) vs mid cold ($2/M on 100K): the cold turn costs more than
    # the later warm savings recover within 5 turns -- the classic bounce-back money leak
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("downgrade_not_worth_losing_cache", True)
    assert r.switch_cost_usd > r.stay_cost_usd * 0.9


def test_downgrade_to_a_much_cheaper_model_pays_even_on_a_cold_cache():
    # cheap cold ($0.2/M) is below premium warm ($0.5/M): switching saves money immediately
    compiled = compile_route_algorithm(complexity_route("m-cheap", "m-premium"))
    r = gate(decision("m-cheap"), compiled, SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    assert (r.reason, r.held_current_model) == ("downgrade_saves_money", False)
    assert r.switch_cost_usd < r.stay_cost_usd


def test_a_longer_expected_session_can_make_a_downgrade_worth_it():
    short = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", remaining_turns=3), PRICING, CacheConfig(), LONG)
    long = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", remaining_turns=40), PRICING, CacheConfig(), LONG)
    assert short.held_current_model is True
    assert long.held_current_model is False and long.reason == "downgrade_saves_money"


def test_switch_margin_decides_razor_thin_downgrades():
    loose = CacheConfig(switch_margin=0.0, horizon_turns=12)
    strict = CacheConfig(switch_margin=0.5, horizon_turns=12)
    session = SessionState(current_model="m-premium")
    assert gate(decision("m-mid"), compiled_mid_premium(), session, PRICING, loose, LONG).held_current_model is False
    assert gate(decision("m-mid"), compiled_mid_premium(), session, PRICING, strict, LONG).held_current_model is True


def test_a_real_cached_count_overrides_a_low_prefix_estimate():
    # chars/4 says ~500 tokens (below the minimum), but the provider really cached 50K
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", cached_prefix_tokens=50_000), PRICING, CacheConfig(), SHORT)
    assert r.prefix_tokens == 50_000
    assert r.reason != "prefix_below_cache_minimum"  # priced, not dismissed as uncacheable
    assert r.stay_cost_usd is not None


def test_partial_cache_hit_weakens_the_case_for_staying():
    full = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    partial = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", cached_prefix_tokens=5_000), PRICING, CacheConfig(), LONG)
    assert partial.stay_cost_usd > full.stay_cost_usd


def test_switch_back_to_a_still_warm_model_is_priced_warm():
    # premium -> mid is held when mid is cold (see above), but if the session left mid a few
    # turns ago and its cache is still alive, going back only pays for the recent turns
    cold = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    warm = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", other_warm_caches={"m-mid": 95_000}), PRICING, CacheConfig(), LONG)
    assert cold.held_current_model is True
    assert (warm.reason, warm.held_current_model, warm.target_cached_tokens) == ("downgrade_saves_money", False, 95_000)
    assert warm.switch_cost_usd < cold.switch_cost_usd


def test_warm_cache_below_the_target_minimum_counts_as_cold():
    r = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", other_warm_caches={"m-mid": 500}), PRICING, CacheConfig(), LONG)
    assert r.target_cached_tokens is None
    assert r.held_current_model is True


def test_warm_cache_on_an_unrelated_model_changes_nothing():
    base = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    other = gate(decision("m-mid"), compiled_mid_premium(), SessionState(current_model="m-premium", other_warm_caches={"m-cheap": 90_000}), PRICING, CacheConfig(), LONG)
    assert other.switch_cost_usd == base.switch_cost_usd


# ------------------------------------------------------------------------ apply()


def test_apply_held_rewrites_selection_and_keeps_policy_pick_as_first_fallback():
    compiled = compiled_mid_premium()
    result = apply(decision("m-mid", fallbacks=["m-premium"]), compiled, SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    assert result.selected_model == "m-premium"
    assert result.fallback_model_ids == ["m-mid"]
    assert result.cache["held_current_model"] is True
    assert result.cache["policy_model"] == "m-mid"


def test_apply_followed_leaves_the_decision_and_explains_why():
    compiled = compiled_mid_premium()
    original = decision("m-mid", fallbacks=["m-premium"])
    result = apply(original, compiled, None, PRICING, CacheConfig(), LONG)
    assert result.selected_model == original.selected_model
    assert result.fallback_model_ids == original.fallback_model_ids
    assert result.cache == {"reason": "no_session", "held_current_model": False, "policy_model": "m-mid"}


def test_apply_held_on_intent_route_updates_the_bucket():
    route = Route.model_validate({
        "name": "r", "policy": "intent", "default": "analysis",
        "models": {
            "extraction": {"id": "m-mid", "client": "openai", "description": "extract"},
            "analysis": {"id": "m-premium", "client": "openai", "description": "analyze"},
        },
    })
    compiled = compile_route_algorithm(route)
    d = decision("m-mid", fallbacks=["m-premium"])
    d = Decision(**{**d.__dict__, "bucket": "extraction", "policy": "intent"})
    result = apply(d, compiled, SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    assert (result.selected_model, result.bucket) == ("m-premium", "analysis")


# ------------------------------------------------------------------------ simulation


def simulate(picks: list[str], *, gated: bool, prefix: int = 100_000, turn_growth: int = 800) -> tuple[int, float]:
    """Run a session where the policy's raw pick alternates; return (switches, total cost).
    Each turn is priced warm if it stays on the previous turn's model, cold if it switches."""
    compiled = compiled_mid_premium()
    config = CacheConfig()
    current, switches, total = None, 0, 0.0
    for pick in picks:
        chars = prefix * 4
        messages = conversation(chars)
        chosen = pick
        if gated:
            chosen = apply(decision(pick), compiled, SessionState(current_model=current), PRICING, config, messages).selected_model
        warm = chosen == current
        total += turn_cost(PRICING[chosen], prefix=prefix, cached=prefix if warm else 0, new=100, output=config.expected_output_tokens)
        switches += int(current is not None and chosen != current)
        current = chosen
        prefix += turn_growth
    return switches, total


def test_gate_stops_a_flip_flopping_policy_from_losing_money():
    # A borderline task: the policy's per-turn verdict alternates strong/weak for 10 turns.
    picks = ["m-premium", "m-mid"] * 5
    ungated_switches, ungated_cost = simulate(picks, gated=False)
    gated_switches, gated_cost = simulate(picks, gated=True)
    assert ungated_switches == 9
    assert gated_switches == 0          # starts on premium, never bounces down at this context size
    assert gated_cost < ungated_cost    # and it's cheaper than flip-flopping


def test_gate_is_cheaper_than_flip_flopping_but_not_just_always_premium_when_cheap_wins():
    # With a genuinely cheap weak tier, downgrades pay for themselves -- the gate allows them
    compiled = compile_route_algorithm(complexity_route("m-cheap", "m-premium"))
    r = apply(decision("m-cheap"), compiled, SessionState(current_model="m-premium"), PRICING, CacheConfig(), LONG)
    assert r.selected_model == "m-cheap"  # not a blanket "never downgrade" latch
