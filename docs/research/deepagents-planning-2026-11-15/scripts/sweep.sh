#!/usr/bin/env bash
# ANTHROPIC_KEY_FILE=<file> sweep.sh <name> <variant> <window|-> <cases> <cap>: one complex-suite run of
# create_deep_agent with Deep Agents' planning (TodoListMiddleware) beside middleware(), on Haiku.
set -uo pipefail
NAME=$1; VARIANT=$2; WIN=$3; CASES=$4; CAP=$5
D=/var/tmp/cons/decide2; H=$D/agent-harness; V=/var/tmp/cons/full/agent-harness/.venv
K="${ANTHROPIC_KEY_FILE:?the file holding the key}"
O=$D/out/$NAME; mkdir -p $O; T=$D-tmp/$NAME; mkdir -p $T/xdg
for v in $(env | grep -oE '^(CLAUDE[A-Z0-9_]*|CCR_[A-Z0-9_]*|ANTHROPIC_[A-Z0-9_]*|SESSION_INGRESS_URL)='); do unset "${v%=}"; done
unset LLM_GATEWAY MEMORY_URL TRELLIS_API_KEY RUNS_URL OPENAI_API_KEY OPENAI_BASE_URL
export LLM_BASE_URL=https://api.anthropic.com/v1/
export PYTHONPATH=$H/src:$H:$D/agent-contracts/src:$D/agent-runs/sdk/python/src:$D/agent-memory-service/sdk/python/src PYTHONDONTWRITEBYTECODE=1
cd $H || exit 1
W=(); [ "$WIN" != "-" ] && W=(--window "$WIN")
LLM_API_KEY="$(cat $K)" TMPDIR=$T XDG_DATA_HOME=$T/xdg timeout 3600 $V/bin/python -m benchmark.cost.complex \
  --model claude-haiku-5-5 --adapter deepagents --variant "$VARIANT" --cases "$CASES" --max-usd "$CAP" "${W[@]}" \
  --out $O/run > $O/log.txt 2>&1
echo "exit $?" >> $O/log.txt
rm -rf "${T:?}"
