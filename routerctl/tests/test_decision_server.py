"""HTTP-level tests for the decision-only server: routing, 404s, and live reload -- all against
the free offline mock Jev, no subprocess and no network."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from jevjudge.cascade import Judge, JudgeConfig
from jevjudge.client import JevClient, JevClientConfig
from jevjudge.mocks import typesafe_mock_app

from routerctl.decision_server import build_app

CLIENTS_YAML = 'clients:\n  openai:\n    format: openai_chat\n    base_url: https://api.openai.com/v1\n    api_key_env: OPENAI_API_KEY\n'

TEAM_YAML = """\
team: x
routes:
  - name: deployment/qa
    policy: complexity
    models:
      weak: { id: m-weak, client: openai }
      strong: { id: m-strong, client: openai }
"""


def mock_judge() -> Judge:
    mock_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=typesafe_mock_app()), base_url="http://jev.mock")
    return Judge(JevClient(JevClientConfig(base_url="http://jev.mock", api_key="mock"), http=mock_http), JudgeConfig())


@pytest.fixture
def teams_dir(tmp_path):
    (tmp_path / "clients.yaml").write_text(CLIENTS_YAML)
    teams = tmp_path / "teams"
    teams.mkdir()
    (teams / "x.yaml").write_text(TEAM_YAML)
    return teams


def test_health_lists_routes(teams_dir):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        body = client.get("/_routerctl/health").json()
        assert body["ok"] is True
        assert body["mode"] == "decide"
        assert body["routes"] == ["deployment/qa"]


def test_decide_returns_a_real_decision_never_a_completion(teams_dir):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        resp = client.post("/v1/decide", json={"model": "deployment/qa", "messages": [{"role": "user", "content": "hello"}]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_model"] in {"m-weak", "m-strong"}
        assert body["selected_client"] == "openai"
        assert "choices" not in body  # never a completion -- see module docstring


def test_unknown_route_is_a_404_naming_known_routes(teams_dir):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        resp = client.post("/v1/decide", json={"model": "deployment/nope", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 404
        assert "deployment/qa" in resp.json()["detail"]


def test_empty_messages_is_a_422_not_a_500(teams_dir):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        resp = client.post("/v1/decide", json={"model": "deployment/qa", "messages": []})
        assert resp.status_code == 422


def test_empty_model_name_is_a_422(teams_dir):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        resp = client.post("/v1/decide", json={"model": "", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 422


def unreachable_judge() -> Judge:
    from jevjudge.cascade import JudgeConfig as _JudgeConfig

    return Judge(JevClient(JevClientConfig(base_url="http://127.0.0.1:1", api_key="x", timeout_s=1.0, max_retries=1)), _JudgeConfig())


def test_decide_still_returns_200_with_a_fallback_decision_when_the_judge_is_down(teams_dir):
    # A down judge is not a 5xx: the underlying algorithm fails open to its safe default (see
    # test_decide_resilience.py), and the caller gets a real, usable decision either way --
    # just one that says explicitly, via judge_error, that it's a fallback.
    app = build_app(teams_dir, judge=unreachable_judge())
    with TestClient(app) as client:
        resp = client.post("/v1/decide", json={"model": "deployment/qa", "messages": [{"role": "user", "content": "hello"}]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_model"] == "m-strong"  # complexity's safe default: the capable tier
        assert body["judge_error"] is not None


def test_route_table_reload_picks_up_a_new_route_and_rejects_a_broken_one(teams_dir):
    # build_app's background watch loop just calls RouteTable.maybe_reload() on a timer; this
    # exercises that same method directly rather than racing a real poll interval in a test.
    from routerctl.decision_server import RouteTable

    table = RouteTable(teams_dir)
    table.load()
    assert list(table.routes) == ["deployment/qa"]

    (teams_dir / "y.yaml").write_text(
        "team: y\nroutes:\n  - name: deployment/new\n    policy: auto\n    models:\n      efficient: { id: m1, client: openai }\n      capable: { id: m2, client: openai }\n"
    )
    assert table.maybe_reload() is True
    assert set(table.routes) == {"deployment/qa", "deployment/new"}

    (teams_dir / "y.yaml").write_text("team: y\nroutes: []\n")  # min_length=1 -> invalid
    assert table.maybe_reload() is False  # reload rejected
    assert set(table.routes) == {"deployment/qa", "deployment/new"}  # last-good table kept live


# ---------------------------------------------------------------------------- cache-aware switching

# Identical prices on both tiers: any switch away from a warm model is then pure cache loss, so
# the gate must hold it -- whichever tier the mock judge happens to pick. Keeps these tests
# deterministic without depending on the mock's verdict.
PRICED_CLIENTS_YAML = CLIENTS_YAML + """\
pricing:
  m-weak:   { input: 2.0, cached_input: 0.2, output: 12.0 }
  m-strong: { input: 2.0, cached_input: 0.2, output: 12.0 }
"""

LONG_HISTORY = [
    {"role": "user", "content": "x" * 40_000},       # ~10K tokens: well above the cache minimum
    {"role": "assistant", "content": "y" * 40_000},
    {"role": "user", "content": "and now?"},
]


@pytest.fixture
def priced_teams_dir(teams_dir):
    (teams_dir.parent / "clients.yaml").write_text(PRICED_CLIENTS_YAML)
    return teams_dir


def test_every_decision_says_what_the_cache_gate_did(teams_dir):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        body = client.post("/v1/decide", json={"model": "deployment/qa", "messages": [{"role": "user", "content": "hello"}]}).json()
        assert body["cache"] == {"reason": "no_session", "held_current_model": False, "policy_model": body["selected_model"]}


def test_switch_away_from_a_warm_model_is_held_and_explained(priced_teams_dir):
    app = build_app(priced_teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        pick = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY}).json()["selected_model"]
        other = "m-strong" if pick == "m-weak" else "m-weak"
        body = client.post("/v1/decide", json={
            "model": "deployment/qa", "messages": LONG_HISTORY,
            "session": {"current_model": other, "cached_prefix_tokens": 20_000, "idle_seconds": 30},
        }).json()
        assert body["selected_model"] == other
        assert body["fallback_model_ids"][0] == pick  # the policy's own pick stays available
        assert body["cache"]["held_current_model"] is True
        assert body["cache"]["reason"] == "downgrade_not_worth_losing_cache"
        assert body["cache"]["policy_model"] == pick
        assert body["cache"]["stay_cost_usd"] < body["cache"]["switch_cost_usd"]


def test_staying_on_the_same_model_is_followed(priced_teams_dir):
    app = build_app(priced_teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        pick = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY}).json()["selected_model"]
        body = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY, "session": {"current_model": pick}}).json()
        assert body["selected_model"] == pick
        assert body["cache"]["reason"] == "same_model"


def test_expired_cache_lets_the_policy_switch_freely(priced_teams_dir):
    app = build_app(priced_teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        pick = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY}).json()["selected_model"]
        other = "m-strong" if pick == "m-weak" else "m-weak"
        body = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY, "session": {"current_model": other, "idle_seconds": 3600}}).json()
        assert body["selected_model"] == pick
        assert body["cache"]["reason"] == "cache_expired"


def test_judge_outage_mid_session_holds_the_current_model(priced_teams_dir):
    app = build_app(priced_teams_dir, judge=unreachable_judge())
    with TestClient(app) as client:
        body = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY, "session": {"current_model": "m-weak"}}).json()
        assert body["judge_error"] is not None
        assert body["selected_model"] == "m-weak"  # not the fail-open capable tier
        assert body["cache"]["reason"] == "judge_unavailable_hold"


@pytest.mark.parametrize("session", [
    {"current_model": "m-weak", "surprise": 1},          # typo'd field: rejected, not ignored
    {"current_model": "m-weak", "cached_prefix_tokens": -1},
    {"current_model": "m-weak", "remaining_turns": 0},
    {"current_model": "m-weak", "other_warm_caches": {"m-strong": -5}},
])
def test_malformed_session_is_a_422(teams_dir, session):
    app = build_app(teams_dir, judge=mock_judge())
    with TestClient(app) as client:
        resp = client.post("/v1/decide", json={"model": "deployment/qa", "messages": LONG_HISTORY, "session": session})
        assert resp.status_code == 422


def test_pricing_reloads_with_clients_yaml(teams_dir):
    from routerctl.decision_server import RouteTable

    table = RouteTable(teams_dir)
    table.load()
    assert table.pricing == {}
    (teams_dir.parent / "clients.yaml").write_text(PRICED_CLIENTS_YAML)
    assert table.maybe_reload() is True
    assert set(table.pricing) == {"m-weak", "m-strong"}


def test_server_warms_the_judge_connection_pool_on_startup(teams_dir, monkeypatch):
    # A decision on a fresh TLS connection is ~3x slower than on a warm one (DECISION.md s16)
    monkeypatch.setenv("JEVJUDGE_WARM_CONNECTIONS", "3")
    monkeypatch.setenv("JEVJUDGE_PING_S", "0")
    judge = mock_judge()
    calls: list[int] = []
    real_warm = judge.jev.warm

    async def spy(connections: int = 4) -> int:
        calls.append(connections)
        return await real_warm(connections)

    judge.jev.warm = spy
    app = build_app(teams_dir, judge=judge)
    with TestClient(app) as client:
        assert client.get("/_routerctl/health").status_code == 200
    assert calls == [3]
