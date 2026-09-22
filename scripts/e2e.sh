#!/usr/bin/env bash
# End-to-end: mock upstream + mock Jev + jevjudge + real switchyard-server, no API keys.
#
# Defaults to this repo's own build (run scripts/build.sh first). Override either half
# independently to test against something else:
#   SWITCHYARD_SERVER=/path/to/switchyard-server PYTHON=/path/to/venv/bin/python scripts/e2e.sh
#
# With a real key, run the same stack against TypeSafe instead of the mock:
#   TYPESAFE_API_KEY=... REAL_JEV=1 scripts/e2e.sh     (routing expectations then depend on Jev)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$( [[ -x .venv/bin/python ]] && echo "$PWD/.venv/bin/python" || command -v python3 )}"
SWITCHYARD_SERVER="${SWITCHYARD_SERVER:-$( [[ -x vendor/switchyard/target/release/switchyard-server ]] && echo "$PWD/vendor/switchyard/target/release/switchyard-server" || command -v switchyard-server || true )}"
[[ -x "$SWITCHYARD_SERVER" ]] || { echo "switchyard-server not found; run scripts/build.sh, or set SWITCHYARD_SERVER"; exit 2; }
[[ -x "$PYTHON" ]] || { echo "python not found; run scripts/build.sh, or set PYTHON"; exit 2; }
LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$LOGDIR"
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
soft_expect() { # label, expected-substring, actual -- reports, never fails the run
  # For a step whose outcome is a real, probabilistic model's judgment call rather than a
  # plumbing invariant (an intermediate escalation turn genuinely near the rubric's own
  # threshold; a phrase with no explicit routing cue that a real classifier may still form
  # an opinion on). Failing here would test one model's stylistic tendency, not this router.
  if [[ "$3" == *"$2"* ]]; then echo "INFO  $1 -> $2 (matched)"; else echo "INFO  $1 -> did not match '$2': $3"; fi
}
served() { "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])'; }
judge_stats() { curl -sS http://127.0.0.1:8090/v1/stats; }

echo; echo "== capability mode (Jev forecasts p_solve + primary_rule; Switchyard applies thresholds)"
r=$(chat capability cap-1 '[{"role":"user","content":"[easy] print hello world in python"}]' | served); expect "easy task" "served-by:weak-model" "$r"
r=$(chat capability cap-2 '[{"role":"user","content":"[hard] reverse engineer the undocumented binary format and match its checksum"}]' | served); expect "hard task" "served-by:strong-model" "$r"
r=$(chat capability cap-3 '[{"role":"user","content":"refactor the parser module"}]' | served); expect "ambiguous -> jevjudge abstains -> Switchyard fails open to strong" "served-by:strong-model" "$r"

echo; echo "== escalation mode (weak first; two consecutive Jev 'escalate' verdicts latch to strong)"
echo "   fixtures use genuine tool_calls/tool messages (a command failing identically 2-3x),"
echo "   not prose asserting trouble -- the escalation rubric explicitly discounts the latter,"
echo "   and so, correctly, does a real judge."
TASK='{"role":"user","content":"Fix the failing tests in tests/test_foo.py so pytest passes."}'
RUN1='{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"pytest -q\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"3 failed, 10 passed. AssertionError: expected 42, got None (test_foo.py:12)"}'
EDIT2='{"role":"assistant","content":"Updating the implementation.","tool_calls":[{"id":"c2","type":"function","function":{"name":"edit_file","arguments":"{\"path\":\"foo.py\"}"}}]},{"role":"tool","tool_call_id":"c2","content":"file edited"}'
RUN2='{"role":"assistant","content":null,"tool_calls":[{"id":"c3","type":"function","function":{"name":"bash","arguments":"{\"command\":\"pytest -q\"}"}}]},{"role":"tool","tool_call_id":"c3","content":"3 failed, 10 passed. Same error: AssertionError: expected 42, got None (test_foo.py:12)"}'
EDIT3='{"role":"assistant","content":"Trying another approach.","tool_calls":[{"id":"c4","type":"function","function":{"name":"edit_file","arguments":"{\"path\":\"foo.py\"}"}}]},{"role":"tool","tool_call_id":"c4","content":"file edited"}'
RUN3='{"role":"assistant","content":null,"tool_calls":[{"id":"c5","type":"function","function":{"name":"bash","arguments":"{\"command\":\"pytest -q\"}"}}]},{"role":"tool","tool_call_id":"c5","content":"3 failed, 10 passed. Same error again: AssertionError: expected 42, got None (test_foo.py:12). Loop."}'
EDIT4='{"role":"assistant","content":"One more attempt.","tool_calls":[{"id":"c6","type":"function","function":{"name":"edit_file","arguments":"{\"path\":\"foo.py\"}"}}]},{"role":"tool","tool_call_id":"c6","content":"file edited"}'
RUN4='{"role":"assistant","content":null,"tool_calls":[{"id":"c7","type":"function","function":{"name":"bash","arguments":"{\"command\":\"pytest -q\"}"}}]},{"role":"tool","tool_call_id":"c7","content":"3 failed, 10 passed. Same error yet again: AssertionError: expected 42, got None (test_foo.py:12). Fourth attempt, stuck in a loop."}'
t1="[$TASK,$RUN1]"
t2="[$TASK,$RUN1,$EDIT2,$RUN2]"
t3="[$TASK,$RUN1,$EDIT2,$RUN2,$EDIT3,$RUN3]"
t4="[$TASK,$RUN1,$EDIT2,$RUN2,$EDIT3,$RUN3,$EDIT4,$RUN4]"
r=$(chat escalation esc-1 "$t1" | served); expect "turn 1: first failure, not yet a pattern" "served-by:weak-model" "$r"
# Turns 2-3 are near the rubric's own "2+ times" threshold. A real judge weighing exactly two
# repeats against "never on a single failed command" can reasonably land on either side, and
# the mock's deterministic cue match does not need to agree with it turn-by-turn -- only the
# eventual outcome (does it ever confirm; does it then latch and stop calling the judge) is a
# plumbing property. Report these, don't gate the run on them.
r=$(chat escalation esc-1 "$t2" | served); soft_expect "turn 2: same error again" "served-by:weak-model" "$r"
r=$(chat escalation esc-1 "$t3" | served); soft_expect "turn 3: same error a third time" "served-by:strong-model" "$r"
r=$(chat escalation esc-1 "$t4" | served); expect "turn 4: same error a fourth time -> must be latched by now" "served-by:strong-model" "$r"
before=$(judge_stats | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["requests"])')
r=$(chat escalation esc-1 "$t4" | served); expect "turn 5 (latched, no judge call)" "served-by:strong-model" "$r"
after=$(judge_stats | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["requests"])')
expect "no judge call after latch" "$before" "$after"
r=$(chat escalation esc-2 '[{"role":"user","content":"[easy] hello, rename a variable"}]' | served); expect "healthy session stays weak" "served-by:weak-model" "$r"

echo; echo "== custom mode (4-way Jev Choice over model groups; criteria harvested from the prompt)"
r=$(chat custom cus-1 '[{"role":"user","content":"route:reasoning prove that sqrt(2) is irrational"}]' | served); expect "reasoning" "served-by:reasoning-model" "$r"
r=$(chat custom cus-2 '[{"role":"user","content":"route:fast hi"}]' | served); expect "fast" "served-by:fast-model" "$r"
r=$(chat custom cus-3 '[{"role":"user","content":"route:premium should we migrate the auth service"}]' | served); expect "premium" "served-by:premium-model" "$r"
# Whether an unhinted phrase reads as "ambiguous" is a property of the judge asked to score
# it, not of jevjudge -- the abstain/default_target mechanism itself is asserted directly and
# deterministically in jevjudge's own unit tests (test_low_confidence_abstain_returns_non_json).
r=$(chat custom cus-4 '[{"role":"user","content":"something with no routing cue"}]' | served); soft_expect "low confidence -> abstain -> default_target" "served-by:balanced-model" "$r"

echo; echo "== jevjudge stats"; judge_stats; echo
echo "== last routing records"; tail -n 3 "$LOGDIR/routing.jsonl" 2>/dev/null | cut -c1-400 || true
echo; echo "passed=$pass failed=$fail  (logs in $LOGDIR)"
[[ $fail -eq 0 ]]
