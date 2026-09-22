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
