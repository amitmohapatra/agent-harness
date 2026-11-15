#!/usr/bin/env bash
# ANTHROPIC_KEY_FILE=<file> run.sh <arm> <run> <adapter> <variant> <meter-port|-> [extra args]
# One framework (arm), one fresh run of the acceptance run's five sets (long57, ordinary,
# sw16 @16000, sw8 @8000, swown) on the product path: the memory service on
# (`--history memory`) and agent-runs the run store (`--runs`), both this job's private stack
# (stack_env.sh, scripts/scenarios/stack.sh); MEMORY=off runs it with neither. SETS / CASES_OVERRIDE narrow it
# (the pilot). With a meter port, the model's requests go through meter.py (forwards and
# records the provider's usage; renames nothing). Used for `openai` (chat completions on
# Anthropic's OpenAI-compatible API either way) and `claude` (the CLI's ANTHROPIC_BASE_URL).
# Not for the LangChain adapters: on Anthropic's own URL the harness calls Anthropic's client
# (prompt cache kept), on any other URL the OpenAI-compatible one, so a meter would change
# their route. Kimi likewise picks its provider from the URL: both run unmetered.
set -uo pipefail
ARM=$1; RUN=$2; AD=$3; VAR=$4; PORT=$5; shift 5
C=/var/tmp/cons/compare; H=$C/agent-harness; V=/var/tmp/cons/full/agent-harness/.venv
S=$H/docs/research/compare-2026-11-15/scripts
STACK_ENV=${STACK_ENV:-/var/tmp/cons/compare-tmp/stack_env.sh}
K="${ANTHROPIC_KEY_FILE:?the file holding the key}"
B=/var/tmp/cons/compare-tmp; O=$B/out/${REPDIR:-full}/$ARM-$RUN; mkdir -p "$O/logs"
for v in $(env | grep -oE '^(CLAUDE[A-Z0-9_]*|CCR_[A-Z0-9_]*|ANTHROPIC_[A-Z0-9_]*|SESSION_INGRESS_URL)='); do unset "${v%=}"; done
unset LLM_GATEWAY MEMORY_URL TRELLIS_API_KEY RUNS_URL OPENAI_API_KEY OPENAI_BASE_URL
export PYTHONPATH=$H:$H/src:$C/agent-contracts/src:$C/agent-runs/sdk/python/src:$C/agent-memory-service/sdk/python/src:$C/bifrost-sdk/src
export PYTHONDONTWRITEBYTECODE=1
export LLM_BASE_URL=https://api.anthropic.com/v1/
if [ "$PORT" != "-" ]; then
  "$V/bin/python" "$S/meter.py" "$PORT" "$O/meter.jsonl" > "$O/logs/meter.log" 2>&1 &
  MP=$!; trap 'kill $MP 2>/dev/null' EXIT; sleep 2
  [ "$AD" != "claude" ] && export LLM_BASE_URL=http://127.0.0.1:$PORT/v1/
fi
cd "$H" || exit 1
for SET in ${SETS:-long57:cases57.txt:0.80:- ordinary:cases-ordinary.txt:0.40:- sw16:cases-sw16.txt:0.20:16000 sw8:cases-sw8.txt:0.20:8000 swown:cases-swown.txt:0.30:-}; do
  IFS=: read -r N F CAP WIN <<< "$SET"
  T=$O/tmp-$N; mkdir -p "$T/xdg" "$T/home"
  W=(); [ "$WIN" != "-" ] && W=(--window "$WIN")
  CASES="${CASES_OVERRIDE:-$(cat "$H/benchmark/cost/results/haiku-2026-11-15/sets/$F")}"
  # the key: read from its file into this one command's environment (never an argument),
  # never printed
  (
    if [ "$AD" = "claude" ]; then
      export HOME="$T/home" CLAUDE_RUN_ROOT="$T/claude" CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
      [ "$PORT" != "-" ] && export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
      ANTHROPIC_API_KEY="$(cat "$K")"; export ANTHROPIC_API_KEY
    fi
    [ "$AD" = "kimi" ] && export KIMI_BIN="$HOME/.kimi-code-npm/bin/kimi" KIMI_RUN_ROOT="$T/kimi"
    LLM_API_KEY="$(cat "$K")"; export LLM_API_KEY
    PRODUCT=()
    if [ "${MEMORY:-on}" = on ]; then
      . "$STACK_ENV"; . "$SCEN_HOME/keys.env"  # the stack's service key, never printed
      export MEMORY_URL=http://127.0.0.1:$SCEN_MEM_PORT RUNS_URL=http://127.0.0.1:$SCEN_RUNS_PORT
      export TRELLIS_API_KEY="$SCEN_KEY"
      unset SCEN_KEY SCEN_KEY2 SCEN_ADMIN_KEY SCEN_ADMIN_KEY2 SCEN_BOOTSTRAP_KEY
      PRODUCT=(--history memory --runs)
    fi
    # XDG_STATE_HOME: the harness's spool of undelivered memory writes, the arm's own (the
    # default, ~/.local/state/trellis, is shared by every process on the machine)
    export TMPDIR="$T" XDG_DATA_HOME="$T/xdg" XDG_STATE_HOME="$T/state" ACCOUNTED_OUT="$O/$N.accounted.json" SAID_OUT="$O/$N.said.json" \
      SERVICES_OUT="$O/$N.services.json"
    exec timeout 7200 "$V/bin/python" "$S/cx_compare.py" \
      --model claude-haiku-5-5 --adapter "$AD" --variant "$VAR" --cases "$CASES" --max-usd "$CAP" \
      "${W[@]}" "${PRODUCT[@]}" --out "$O/$N" "$@"
  ) > "$O/logs/$N.log" 2>&1
  echo "$N exit $?" >> "$O/logs/exits.txt"
  rm -rf "${T:?}"
  gzip -f "$O/$N.requests.jsonl" 2>/dev/null
done
echo done > "$O/DONE"
