#!/usr/bin/env bash
# Proves routerctl's decision-only mode (the default `routerctl serve`, no --mode flag needed):
# given a request, it returns which model/client should serve it and NEVER calls that model
# itself -- an external AI Gateway is expected to make the real call. Two things are proven:
#
#   1. The target is never dialed at all: every client in clients.yaml points at a deliberately
#      unreachable address, and decisions still succeed -- if this router ever tried to call the
#      target, every request here would fail with a connection error instead.
#   2. Live reload works exactly like proxy mode's, but in-process (no subprocess boot, no port
#      swap): edit a team's YAML while serving, and the new config is live within a few seconds.
#
# Uses the free, offline mock Jev (jevjudge.mocks typesafe), not real Jev -- this is about the
# decision-only mechanism and reload, not judge accuracy (see evals/ and docs/DECISION.md for
# real-Jev validation).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$( [[ -x .venv/bin/python ]] && echo "$PWD/.venv/bin/python" || command -v python3 )}"

FIXTURE="$(mktemp -d)"
LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$FIXTURE/teams" "$LOGDIR"
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$FIXTURE"; }
trap cleanup EXIT

JEV_MOCK_PORT=8098
PUBLIC_PORT=4200
UNREACHABLE="http://127.0.0.1:1/never-dialed"  # port 1 requires privileges routerctl won't have

cat > "$FIXTURE/clients.yaml" <<EOF
clients:
  never:
    format: openai_chat
    base_url: $UNREACHABLE
EOF

write_route() { # model id to use for auto's efficient tier
  cat > "$FIXTURE/teams/demo.yaml" <<EOF
team: demo
routes:
  - name: deployment/demo
    policy: auto
    models:
      efficient: { id: "$1", client: never }
      capable: { id: "model-capable", client: never }
  - name: deployment/qa
    policy: complexity
    models:
      weak: { id: "model-weak", client: never }
      strong: { id: "model-strong", client: never }
EOF
}

wait_for() { for _ in $(seq 1 80); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 0.25; done; echo "timeout waiting for $1"; exit 1; }
decide() { # route
  curl -sS "http://127.0.0.1:$PUBLIC_PORT/v1/decide" -H 'content-type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hello, quick task\"}]}"
}
selected() { "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["selected_model"])'; }

pass=0; fail=0
expect() { if [[ "$3" == *"$2"* ]]; then echo "PASS  $1 -> $2"; pass=$((pass+1)); else echo "FAIL  $1: expected '$2' in: $3"; fail=$((fail+1)); fi; }

echo "== starting mock Jev ($JEV_MOCK_PORT)"
"$PYTHON" -m jevjudge.mocks typesafe --port "$JEV_MOCK_PORT" >"$LOGDIR/mock-jev.log" 2>&1 & pids+=($!)
wait_for "http://127.0.0.1:$JEV_MOCK_PORT/health"

echo "== starting routerctl serve --mode decide on :$PUBLIC_PORT (fixture: $FIXTURE, every client points at an unreachable address)"
write_route "model-v1"
TYPESAFE_BASE_URL="http://127.0.0.1:$JEV_MOCK_PORT" TYPESAFE_API_KEY="mock" \
  "$PYTHON" -m routerctl.cli serve "$FIXTURE/teams" --clients "$FIXTURE/clients.yaml" \
  --port "$PUBLIC_PORT" --poll-seconds 1 >"$LOGDIR/routerctl.log" 2>&1 & pids+=($!)
wait_for "http://127.0.0.1:$PUBLIC_PORT/_routerctl/health"

echo; echo "== a decision succeeds even though every client is unreachable -- the target is never called"
r=$(decide deployment/demo); expect "auto decides without dialing the target" "model-v1" "$(echo "$r" | selected)"
r=$(decide deployment/qa); m=$(echo "$r" | selected)
expect "complexity decides via the judge, not the target" "1" "$([[ "$m" == "model-weak" || "$m" == "model-strong" ]] && echo 1 || echo 0)"
expect "response is a decision, never a completion" "0" "$(echo "$r" | grep -c '"choices"' || true)"

echo; echo "== editing teams/demo.yaml live (no restart, no subprocess swap) -> efficient tier becomes model-v1-updated"
write_route "model-v1-updated"
for _ in $(seq 1 40); do
  r=$(decide deployment/demo) || true
  [[ "$(echo "$r" | selected)" == "model-v1-updated" ]] && break
  sleep 0.5
done
expect "after live edit, no restart" "model-v1-updated" "$(echo "$r" | selected)"

echo; echo "== breaking the config (route with zero models, fails schema) -> must keep serving the last-good table"
cat > "$FIXTURE/teams/demo.yaml" <<'EOF'
team: demo
routes:
  - name: deployment/demo
    policy: auto
    models: {}
EOF
sleep 3
r=$(decide deployment/demo); expect "still serving the last-good table after a bad edit" "model-v1-updated" "$(echo "$r" | selected)"
bad_reload_attempts=$(grep -c "reload failed, keeping previous routes live" "$LOGDIR/routerctl.log" || true)
expect "bad edit was rejected, not silently applied" "1" "$([[ "$bad_reload_attempts" -ge 1 ]] && echo 1 || echo 0)"

echo; echo "== fixing it again -> live within a few seconds"
write_route "model-v1-recovered"
for _ in $(seq 1 40); do
  r=$(decide deployment/demo) || true
  [[ "$(echo "$r" | selected)" == "model-v1-recovered" ]] && break
  sleep 0.5
done
expect "recovers after the fix" "model-v1-recovered" "$(echo "$r" | selected)"

echo; echo "passed=$pass failed=$fail  (logs in $LOGDIR)"
[[ $fail -eq 0 ]]
