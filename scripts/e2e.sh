#!/usr/bin/env bash
# End-to-end: mock upstream + mock Jev + jevjudge + real switchyard-server, no API keys.
#
#   SWITCHYARD_SERVER=/path/to/switchyard-server PYTHON=/path/to/venv/bin/python scripts/e2e.sh
#
# With a real key, run the same stack against TypeSafe instead of the mock:
#   TYPESAFE_API_KEY=... REAL_JEV=1 scripts/e2e.sh     (routing expectations then depend on Jev)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
SWITCHYARD_SERVER="${SWITCHYARD_SERVER:-$(command -v switchyard-server || true)}"
[[ -x "$SWITCHYARD_SERVER" ]] || { echo "switchyard-server not found; set SWITCHYARD_SERVER"; exit 2; }
LOGDIR="${LOGDIR:-$(mktemp -d)}"
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

wait_for() { local url=$1; for _ in $(seq 1 60); do curl -fsS "$url" >/dev/null 2>&1 && return 0; sleep 0.25; done; echo "timeout waiting for $url"; exit 1; }

echo "== starting mock upstream (8092)";   "$PYTHON" -m jevjudge.mocks upstream --port 8092 >"$LOGDIR/upstream.log" 2>&1 & pids+=($!)
if [[ -z "${REAL_JEV:-}" ]]; then
  echo "== starting mock Jev (8091)";      "$PYTHON" -m jevjudge.mocks typesafe --port 8091 >"$LOGDIR/jev.log" 2>&1 & pids+=($!)
  export TYPESAFE_BASE_URL=http://127.0.0.1:8091 TYPESAFE_API_KEY=mock
  wait_for http://127.0.0.1:8091/health
fi
echo "== starting jevjudge (8090) with confidence gate 0.5 -> abstain"
JEVJUDGE_MIN_CONFIDENCE="${JEVJUDGE_MIN_CONFIDENCE:-0.5}" JEVJUDGE_LOW_CONFIDENCE_ACTION="${JEVJUDGE_LOW_CONFIDENCE_ACTION:-abstain}" \
  "$PYTHON" -m jevjudge.server --port 8090 >"$LOGDIR/jevjudge.log" 2>&1 & pids+=($!)
echo "== starting switchyard-server (4000)"
"$SWITCHYARD_SERVER" --config switchyard/routes.e2e.toml --host 127.0.0.1 --port 4000 --routing-log-file "$LOGDIR/routing.jsonl" >"$LOGDIR/switchyard.log" 2>&1 & pids+=($!)
wait_for http://127.0.0.1:8092/health
wait_for http://127.0.0.1:8090/health
wait_for http://127.0.0.1:4000/health

pass=0; fail=0
chat() { # route, session, json-messages
  curl -sS http://127.0.0.1:4000/v1/chat/completions -H 'content-type: application/json' -H "x-switchyard-session-id: $2" \
    -d "{\"model\":\"$1\",\"messages\":$3,\"max_tokens\":64}"
}
expect() { # label, expected-substring, actual
  if [[ "$3" == *"$2"* ]]; then echo "PASS  $1 -> $2"; pass=$((pass+1)); else echo "FAIL  $1: expected '$2' in: $3"; fail=$((fail+1)); fi
}
served() { "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])'; }
judge_stats() { curl -sS http://127.0.0.1:8090/v1/stats; }

echo; echo "== capability mode (Jev forecasts p_solve + primary_rule; Switchyard applies thresholds)"
r=$(chat capability cap-1 '[{"role":"user","content":"[easy] print hello world in python"}]' | served); expect "easy task" "served-by:weak-model" "$r"
r=$(chat capability cap-2 '[{"role":"user","content":"[hard] reverse engineer the undocumented binary format and match its checksum"}]' | served); expect "hard task" "served-by:strong-model" "$r"
r=$(chat capability cap-3 '[{"role":"user","content":"refactor the parser module"}]' | served); expect "ambiguous -> jevjudge abstains -> Switchyard fails open to strong" "served-by:strong-model" "$r"

echo; echo "== escalation mode (weak first; two consecutive Jev 'escalate' verdicts latch to strong)"
t1='[{"role":"user","content":"fix the failing test"},{"role":"assistant","content":"running pytest"},{"role":"user","content":"[stuck] same error again after retry"}]'
r=$(chat escalation esc-1 "$t1" | served); expect "turn 1 (streak 1)" "served-by:weak-model" "$r"
t2='[{"role":"user","content":"fix the failing test"},{"role":"assistant","content":"running pytest"},{"role":"user","content":"[stuck] same error again after retry"},{"role":"assistant","content":"retrying"},{"role":"user","content":"[stuck] same error a third time, loop"}]'
r=$(chat escalation esc-1 "$t2" | served); expect "turn 2 (streak 2 -> latched)" "served-by:strong-model" "$r"
before=$(judge_stats | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["requests"])')
r=$(chat escalation esc-1 "$t2" | served); expect "turn 3 (latched, no judge call)" "served-by:strong-model" "$r"
after=$(judge_stats | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["requests"])')
expect "no judge call after latch" "$before" "$after"
r=$(chat escalation esc-2 '[{"role":"user","content":"[easy] hello, rename a variable"}]' | served); expect "healthy session stays weak" "served-by:weak-model" "$r"

echo; echo "== custom mode (4-way Jev Choice over model groups; criteria harvested from the prompt)"
r=$(chat custom cus-1 '[{"role":"user","content":"route:reasoning prove that sqrt(2) is irrational"}]' | served); expect "reasoning" "served-by:reasoning-model" "$r"
r=$(chat custom cus-2 '[{"role":"user","content":"route:fast hi"}]' | served); expect "fast" "served-by:fast-model" "$r"
r=$(chat custom cus-3 '[{"role":"user","content":"route:premium should we migrate the auth service"}]' | served); expect "premium" "served-by:premium-model" "$r"
r=$(chat custom cus-4 '[{"role":"user","content":"something with no routing cue"}]' | served); expect "low confidence -> abstain -> default_target" "served-by:balanced-model" "$r"

echo; echo "== jevjudge stats"; judge_stats; echo
echo "== last routing records"; tail -n 3 "$LOGDIR/routing.jsonl" 2>/dev/null | cut -c1-400 || true
echo; echo "passed=$pass failed=$fail  (logs in $LOGDIR)"
[[ $fail -eq 0 ]]
