#!/usr/bin/env bash
# ANTHROPIC_KEY_FILE=<file> acc.sh <arm-name> <run-no> [extra args]: one arm, one fresh run of the closing run's five sets
# (long57, ordinary, sw16 @16000, sw8 @8000, swown), memory off, on the haikuh worktree.
set -uo pipefail
ARM=$1; RUN=$2; shift 2
H=/var/tmp/cons/haikuh/agent-harness; V=/var/tmp/cons/full/agent-harness/.venv
K="${ANTHROPIC_KEY_FILE:?the file holding the key}"
B=/var/tmp/cons/haikuh-tmp; O=$B/out/${REPDIR:-acc}/$ARM-$RUN; mkdir -p $O/logs
for v in $(env | grep -oE '^(CLAUDE[A-Z0-9_]*|CCR_[A-Z0-9_]*|ANTHROPIC_[A-Z0-9_]*|SESSION_INGRESS_URL)='); do unset "${v%=}"; done
unset LLM_GATEWAY MEMORY_URL TRELLIS_API_KEY RUNS_URL OPENAI_API_KEY OPENAI_BASE_URL
export LLM_BASE_URL=https://api.anthropic.com/v1/
export PYTHONPATH=$H:$H/src PYTHONDONTWRITEBYTECODE=1
cd $H || exit 1
for SET in ${SETS:-long57:cases57.txt:0.30:- ordinary:cases-ordinary.txt:0.09:- sw16:cases-sw16.txt:0.10:16000 sw8:cases-sw8.txt:0.07:8000 swown:cases-swown.txt:0.10:-}; do
  IFS=: read -r N F CAP WIN <<< "$SET"
  T=$O/tmp-$N; mkdir -p $T/xdg
  W=(); [ "$WIN" != "-" ] && W=(--window "$WIN")
  CASES="${CASES_OVERRIDE:-$(cat $B/sets/$F)}"
  LLM_API_KEY="$(cat $K)" TMPDIR=$T XDG_DATA_HOME=$T/xdg timeout 5400 $V/bin/python -m benchmark.cost.complex \
    --model claude-haiku-5-5 --adapter ${ADAPTER:-react} --cases "$CASES" --max-usd $CAP "${W[@]}" \
    --out $O/$N "$@" > $O/logs/$N.log 2>&1
  rm -rf "${T:?}"
  gzip -f $O/$N.requests.jsonl 2>/dev/null
done
echo done > $O/DONE
