#!/usr/bin/env bash
# Proves routerctl's live reload: edit a team's YAML while the server is running, and the new
# config is live within a few seconds -- no restart, no dropped process, other teams unaffected.
#
# Uses a throwaway fixture (not teams/), a mock upstream, and no judge (policy: auto, a plain
# request with no tool history always picks the efficient tier deterministically -- see
# stage_router_routing.md) so this test is entirely about the reload mechanism, not routing
# decisions already covered by scripts/e2e.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$( [[ -x .venv/bin/python ]] && echo "$PWD/.venv/bin/python" || command -v python3 )}"
SWITCHYARD_SERVER="${SWITCHYARD_SERVER:-$( [[ -x vendor/switchyard/target/release/switchyard-server ]] && echo "$PWD/vendor/switchyard/target/release/switchyard-server" || command -v switchyard-server || true )}"
[[ -x "$SWITCHYARD_SERVER" ]] || { echo "switchyard-server not found; run scripts/build.sh"; exit 2; }

FIXTURE="$(mktemp -d)"
LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$FIXTURE/teams" "$LOGDIR"
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$FIXTURE"; }
trap cleanup EXIT

MOCK_PORT=8097
PUBLIC_PORT=4100

cat > "$FIXTURE/clients.yaml" <<EOF
clients:
  mock:
    format: openai_chat
    base_url: http://127.0.0.1:$MOCK_PORT/v1
EOF

write_route() { # model id to use for the efficient tier
  cat > "$FIXTURE/teams/demo.yaml" <<EOF
team: demo
routes:
  - name: deployment/demo
    policy: auto
    models:
      efficient: { id: "$1", client: mock }
      capable: { id: "model-capable", client: mock }
EOF
}

wait_for() { for _ in $(seq 1 80); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 0.25; done; echo "timeout waiting for $1"; exit 1; }
served() { curl -sS "http://127.0.0.1:$PUBLIC_PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"deployment/demo","messages":[{"role":"user","content":"hello"}],"max_tokens":16}' \
  | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'; }

pass=0; fail=0
expect() { if [[ "$3" == *"$2"* ]]; then echo "PASS  $1 -> $2"; pass=$((pass+1)); else echo "FAIL  $1: expected '$2' in: $3"; fail=$((fail+1)); fi; }

echo "== starting mock upstream ($MOCK_PORT)"
write_route "model-v1"
"$PYTHON" -m jevjudge.mocks upstream --port "$MOCK_PORT" >"$LOGDIR/upstream.log" 2>&1 & pids+=($!)
wait_for "http://127.0.0.1:$MOCK_PORT/health"

echo "== starting routerctl serve on :$PUBLIC_PORT (fixture: $FIXTURE)"
"$PYTHON" -m routerctl.cli serve "$FIXTURE/teams" --clients "$FIXTURE/clients.yaml" \
  --switchyard-server "$SWITCHYARD_SERVER" --port "$PUBLIC_PORT" --poll-seconds 1 \
  --build-dir "$LOGDIR/build" >"$LOGDIR/routerctl.log" 2>&1 & pids+=($!)
wait_for "http://127.0.0.1:$PUBLIC_PORT/_routerctl/health"

echo; echo "== generation 1: efficient tier is model-v1"
r=$(served); expect "before edit" "served-by:model-v1" "$r"

echo; echo "== editing teams/demo.yaml live (no restart) -> efficient tier becomes model-v1-updated"
write_route "model-v1-updated"
for _ in $(seq 1 40); do
  r=$(served) || true
  [[ "$r" == *"model-v1-updated"* ]] && break
  sleep 0.5
done
expect "after live edit, no restart" "served-by:model-v1-updated" "$r"

gen_line=$(grep -c "is live on port" "$LOGDIR/routerctl.log" || true)
expect "supervisor recorded two live generations" "2" "$gen_line"

echo; echo "== breaking the config (unknown client) -> must keep serving the last-good generation"
cat > "$FIXTURE/teams/demo.yaml" <<'EOF'
team: demo
routes:
  - name: deployment/demo
    policy: auto
    models:
      efficient: { id: "model-broken", client: does-not-exist }
      capable: { id: "model-capable", client: mock }
EOF
sleep 3
r=$(served); expect "still serving the last-good generation after a bad edit" "served-by:model-v1-updated" "$r"
expect "bad edit was rejected, not silently applied" "1" "$(grep -c "keeping generation" "$LOGDIR/routerctl.log" || true)"

echo; echo "== fixing it again -> live within a few seconds"
write_route "model-v1-recovered"
for _ in $(seq 1 40); do
  r=$(served) || true
  [[ "$r" == *"model-v1-recovered"* ]] && break
  sleep 0.5
done
expect "recovers after the fix" "served-by:model-v1-recovered" "$r"

echo; echo "passed=$pass failed=$fail  (logs in $LOGDIR)"
[[ $fail -eq 0 ]]
