#!/usr/bin/env bash
# run.sh <script.py> [args]: one case script with a scrubbed environment: the clones on
# PYTHONPATH, a home and TMPDIR of its own, the private stack's settings, and the Haiku key
# read from its file into the environment here (never printed). $T/pylib holds the docker SDK
# (uv pip --target, no deps), which the OpenAI Agents SDK's DockerSandboxClient imports.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=${SDKFULL_ROOT:-/var/tmp/cons/sdkfull}
T=${SDKFULL_TMP:-/var/tmp/cons/sdkfull-tmp}
K=${ANTHROPIC_KEY_FILE:?the file holding the key}
. "$T/scen_env.sh"
mkdir -p "$T/home" "$T/tmp"
cd "$HERE" || exit 1
exec env -i PATH="/opt/node22/bin:$HOME/.kimi-code-npm/bin:/usr/bin:/bin" HOME="$T/home" \
  LANG=C.UTF-8 TMPDIR="$T/tmp" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  NO_PROXY='localhost,127.0.0.1,api.anthropic.com' no_proxy='localhost,127.0.0.1,api.anthropic.com' \
  SSL_CERT_FILE="${SSL_CERT_FILE:-}" NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-}" \
  OPENAI_AGENTS_DISABLE_TRACING=1 SCEN_HOME="$SCEN_HOME" SCEN_MEM_PORT="$SCEN_MEM_PORT" \
  SCEN_RUNS_PORT="$SCEN_RUNS_PORT" KIMI_REAL_HOME="$HOME/.kimi-code" \
  PYTHONPATH="$HERE:$S/agent-harness:$S/agent-harness/src:$S/agent-contracts/src:$S/agent-memory-service/sdk/python/src:$S/agent-runs/sdk/python/src:$S/bifrost-sdk/src:$T/pylib" \
  ANTHROPIC_API_KEY="$(cat "$K")" \
  /var/tmp/cons/full/agent-harness/.venv/bin/python "$@"
