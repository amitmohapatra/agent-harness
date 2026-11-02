#!/usr/bin/env bash
# The private stack of the background load benchmark, then the benchmark with "$@":
# a fresh agent-runs database on the PostgreSQL at $BENCH_PG_ADMIN (pg_stat_statements loaded),
# agent-runs ($RUNS_SRC, run by $RUNS_PY) on 127.0.0.1:$RUNS_PORT with $RUNS_WORKERS processes,
# and a key registry stub at 127.0.0.1:$KEYS_PORT (one service key of tenant t1). Everything is
# stopped and the database dropped at the end.
#   RUNS_SRC=../agent-runs RUNS_PY=... benchmark/load/background_stack.sh --arm A --tasks 100
set -euo pipefail
: "${RUNS_SRC:?the agent-runs checkout}" "${RUNS_PY:?a python with agent-runs installed}"
ADMIN=${BENCH_PG_ADMIN:-postgresql://memory:memory@127.0.0.1:5432/postgres}
DB=${BENCH_DB:-runs_bench}; RUNS_PORT=${RUNS_PORT:-8090}; KEYS_PORT=${KEYS_PORT:-8091}
HERE=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-$HERE/.venv/bin/python}
psql "$ADMIN" -qc "drop database if exists $DB" -c "create database $DB"
export BENCH_PG=${ADMIN%/*}/$DB
psql "$BENCH_PG" -qc "create extension if not exists pg_stat_statements"
export RUNS__DATABASE__URL=postgresql+psycopg://${BENCH_PG#postgresql://}
export RUNS__SERVICE__PORT=$RUNS_PORT RUNS__SERVICE__WORKERS=${RUNS_WORKERS:-1}
export RUNS__SERVICE__LOG_LEVEL=WARNING MEMORY_URL=http://127.0.0.1:$KEYS_PORT
(cd "$RUNS_SRC" && $RUNS_PY -m alembic upgrade head >/dev/null)
$PY -c "
import http.server, json
class K(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == '/v1/keys/self'
        body = json.dumps({'key_id': 'k1', 'tenant_id': 't1', 'principal': 'svc', 'role': 'service', 'may_act_as': ['*']} if ok else {}).encode()
        self.send_response(200 if ok else 404); self.send_header('content-type', 'application/json'); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', $KEYS_PORT), K).serve_forever()" &
KEYS=$!
(cd "$RUNS_SRC" && exec $RUNS_PY -m agent_runs) > "${RUNS_LOG:-/dev/null}" 2>&1 &
API=$!
trap 'kill $API $KEYS 2>/dev/null; wait $API $KEYS 2>/dev/null; psql "$ADMIN" -qc "drop database if exists $DB with (force)"; true' EXIT
for _ in $(seq 60); do curl -fs -o /dev/null "http://127.0.0.1:$RUNS_PORT/health/ready" 2>/dev/null && break; sleep 0.5; done
curl -fs -o /dev/null "http://127.0.0.1:$RUNS_PORT/health/ready" || { echo "agent-runs did not start" >&2; exit 1; }
APIPIDS=$(pgrep -P $API | tr "\n" " " || true); WATCH=(--watch "api=$API"); for p in $APIPIDS; do WATCH+=(--watch "api-$p=$p"); done
export RUNS_URL=http://127.0.0.1:$RUNS_PORT TRELLIS_API_KEY=bench-key
unset LLM_GATEWAY OPENAI_API_KEY ANTHROPIC_API_KEY OTEL_EXPORTER_OTLP_ENDPOINT
cd "$HERE" && $PY -m benchmark.load.background "${WATCH[@]}" "$@"
