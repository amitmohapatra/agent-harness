#!/usr/bin/env bash
# stack.sh start|stop|status|reset|drop|stop-one <svc>|start-one <svc>|test [pytest args]
#
# A private stack for the user-scenario suite (tests/scenarios): its own Redis, the memory API
# (with its job worker, in one process) and agent-runs' API and ticker, from sibling checkouts,
# on databases of their own in an existing PostgreSQL cluster, with keys made for it alone
# (never printed). `test` runs the suite against it. Nothing here calls a model.
#
# Settings (environment, with their defaults):
#   SCEN_HOME=/var/tmp/scen         where it keeps its logs, pids, keys, blobs
#   SCEN_PG_PORT=5432               the PostgreSQL cluster (127.0.0.1, user postgres) it makes its databases in
#   SCEN_REDIS_PORT=56380 SCEN_MEM_PORT=18380 SCEN_RUNS_PORT=18390
#   SCEN_MEM_ROOT=../agent-memory-service SCEN_RUNS_ROOT=../agent-runs SCEN_CONTRACTS=../agent-contracts/src
#   SCEN_BIFROST=../bifrost-sdk/src
#   SCEN_SVC_PY (the memory service's python)  SCEN_RUNS_PY (agent-runs' python)  SCEN_PY (this repository's)
#   SCEN_MODELS                      the memory service's frozen models folder (embedding, NLI)
set -uo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
: "${SCEN_HOME:=/var/tmp/scen}" "${SCEN_PG_PORT:=5432}" "${SCEN_REDIS_PORT:=56380}"
: "${SCEN_MEM_PORT:=18380}" "${SCEN_RUNS_PORT:=18390}"
: "${SCEN_MEM_ROOT:=$HERE/../agent-memory-service}" "${SCEN_RUNS_ROOT:=$HERE/../agent-runs}"
: "${SCEN_CONTRACTS:=$HERE/../agent-contracts/src}" "${SCEN_BIFROST:=$HERE/../bifrost-sdk/src}"
: "${SCEN_SVC_PY:=$SCEN_MEM_ROOT/.venv/bin/python}" "${SCEN_RUNS_PY:=$SCEN_RUNS_ROOT/.venv/bin/python}"
: "${SCEN_PY:=$HERE/.venv/bin/python}" "${SCEN_MODELS:=$SCEN_MEM_ROOT/models}"
: "${SCEN_DB_MEM:=scen_memory}" "${SCEN_DB_RUNS:=scen_runs}" "${SCEN_TENANT:=scen}"
H=$SCEN_HOME
mkdir -p "$H"/{logs,run,tmp,home}
[ -f "$H/keys.env" ] && . "$H/keys.env"
psqlc() { psql -h 127.0.0.1 -p "$SCEN_PG_PORT" -U postgres -qAt "$@"; }
up() { curl -fsS --noproxy '*' -o /dev/null "http://127.0.0.1:$1/health/live" 2>/dev/null; }
alive() { [ -f "$H/run/$1.pid" ] && kill -0 "$(cat "$H/run/$1.pid")" 2>/dev/null; }
launch() { local name=$1; shift; setsid nohup "$@" >> "$H/logs/$name.log" 2>&1 < /dev/null & echo $! > "$H/run/$name.pid"; }

memory() {  # memory [migrate]: the memory API and its job worker (tests/scenarios/memory_api.py)
  local url="postgresql+psycopg://postgres@127.0.0.1:$SCEN_PG_PORT/$SCEN_DB_MEM" w=$H/memwork
  mkdir -p "$w/blob" "$w/qdrant"; ln -sfn "$SCEN_MODELS" "$w/models"
  local env=(PATH=/usr/bin:/bin HOME=$H/home LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    PYTHONPATH="$SCEN_MEM_ROOT/src:$SCEN_MEM_ROOT:$SCEN_MEM_ROOT/sdk/python/src:$SCEN_CONTRACTS:$SCEN_BIFROST"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY='*' TMPDIR=$H/tmp
    MEMORY__SERVICE__ENVIRONMENT=dev MEMORY__SERVICE__PORT=$SCEN_MEM_PORT MEMORY__SERVICE__WORKERS=1
    MEMORY__SERVICE__LOG_JSON=false MEMORY__AUTHENTICATION__BOOTSTRAP_ADMIN_KEY="$SCEN_BOOTSTRAP_KEY"
    MEMORY__DATABASE__URL="$url" MEMORY__DATABASE__DIRECT_URL="$url"
    MEMORY__CACHE__URL="redis://127.0.0.1:$SCEN_REDIS_PORT/0" MEMORY__BLOB__FILESYSTEM_ROOT="$w/blob"
    MEMORY__TASKS__METRICS_PORT=$((SCEN_MEM_PORT+100)) MH_QDRANT_PATH="$w/qdrant")
  if [ "${1:-}" = migrate ]; then
    (cd "$SCEN_MEM_ROOT" && env -i "${env[@]}" "$SCEN_SVC_PY" -m alembic upgrade head) &&
    (cd "$w" && env -i "${env[@]}" "$SCEN_SVC_PY" -m memory_service.adapters.tasks.procrastinate_schema || true)
    return
  fi
  cd "$w" && exec env -i "${env[@]}" "$SCEN_SVC_PY" "$HERE/scripts/scenarios/memory_api.py"
}
runs() {  # runs migrate|api|ticker
  local w=$H/runswork; mkdir -p "$w/blob"
  local env=(PATH=/usr/bin:/bin HOME=$H/home LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY='*'
    TMPDIR=$H/tmp PYTHONPATH="$SCEN_RUNS_ROOT/src:$SCEN_RUNS_ROOT/sdk/python/src:$SCEN_CONTRACTS"
    RUNS__SERVICE__ENVIRONMENT=dev RUNS__SERVICE__HOST=127.0.0.1 RUNS__SERVICE__PORT=$SCEN_RUNS_PORT
    RUNS__SERVICE__WORKERS=1 RUNS__MEMORY__URL=http://127.0.0.1:$SCEN_MEM_PORT
    RUNS__DATABASE__URL="postgresql+psycopg://postgres@127.0.0.1:$SCEN_PG_PORT/$SCEN_DB_RUNS"
    RUNS__BLOB__ROOT="$w/blob" RUNS__TICKER__HEARTBEAT_FILE="$w/ticker.heartbeat"
    RUNS__OBSERVABILITY__LOG_JSON=false)
  cd "$SCEN_RUNS_ROOT" || exit 1
  case $1 in
    migrate) env -i "${env[@]}" "$SCEN_RUNS_PY" "$(dirname "$SCEN_RUNS_PY")/alembic" upgrade head ;;
    api) exec env -i "${env[@]}" "$SCEN_RUNS_PY" -m agent_runs ;;
    ticker) exec env -i "${env[@]}" "$SCEN_RUNS_PY" -m agent_runs.ticker ;;
  esac
}
start_one() {
  case $1 in
    memory) launch memory "$0" _memory ;;
    runs-api) launch runs-api "$0" _runs api ;;
    runs-ticker) launch runs-ticker "$0" _runs ticker ;;
  esac
}
stop_one() { alive "$1" && kill "$(cat "$H/run/$1.pid")"; for _ in $(seq 1 40); do alive "$1" || break; sleep 0.5; done; rm -f "$H/run/$1.pid"; }
start() {
  pg_isready -q -h 127.0.0.1 -p "$SCEN_PG_PORT" || { echo "no PostgreSQL on :$SCEN_PG_PORT"; exit 2; }
  redis-cli -p "$SCEN_REDIS_PORT" ping >/dev/null 2>&1 || {
    mkdir -p "$H/redis"; (cd "$H/redis" && launch redis redis-server --port "$SCEN_REDIS_PORT" --bind 127.0.0.1 --save "" --appendonly no); sleep 1; }
  up "$SCEN_MEM_PORT" || { start_one memory; for _ in $(seq 1 100); do up "$SCEN_MEM_PORT" && break; sleep 3; done; }
  up "$SCEN_RUNS_PORT" || { start_one runs-api; for _ in $(seq 1 30); do up "$SCEN_RUNS_PORT" && break; sleep 1; done; }
  alive runs-ticker || start_one runs-ticker
}
stop() { for s in runs-ticker runs-api memory; do stop_one $s; done; redis-cli -p "$SCEN_REDIS_PORT" shutdown nosave >/dev/null 2>&1; rm -f "$H/run/redis.pid"; }
status() { for s in memory runs-api runs-ticker redis; do alive $s && echo "$s pid $(cat "$H/run/$s.pid")" || echo "$s: not running"; done; }
reset() {
  stop
  pg_isready -q -h 127.0.0.1 -p "$SCEN_PG_PORT" || { echo "no PostgreSQL on :$SCEN_PG_PORT"; exit 2; }
  psqlc -c "drop database if exists $SCEN_DB_MEM with (force)" -c "create database $SCEN_DB_MEM" \
        -c "drop database if exists $SCEN_DB_RUNS with (force)" -c "create database $SCEN_DB_RUNS"
  rm -rf "$H/memwork" "$H/runswork"
  (umask 077; echo "export SCEN_BOOTSTRAP_KEY=scen-bootstrap-$(openssl rand -hex 24)" > "$H/keys.env"); . "$H/keys.env"
  memory migrate > "$H/logs/memory-migrate.log" 2>&1 || { echo "memory migration failed"; exit 1; }
  runs migrate > "$H/logs/runs-migrate.log" 2>&1 || { echo "agent-runs migration failed"; exit 1; }
  start
  env -i PATH=/usr/bin:/bin NO_PROXY='*' SCEN_HOME="$H" SCEN_MEM_PORT="$SCEN_MEM_PORT" SCEN_TENANT="$SCEN_TENANT" \
    SCEN_BOOTSTRAP_KEY="$SCEN_BOOTSTRAP_KEY" PYTHONPATH="$SCEN_MEM_ROOT/sdk/python/src:$SCEN_CONTRACTS" \
    "$SCEN_SVC_PY" "$HERE/scripts/scenarios/onboard.py" || exit 1
  status
}
case "${1:-status}" in
  _memory) memory ;;
  _runs) runs "$2" ;;
  start) start; status ;;
  stop) stop; status ;;
  status) status ;;
  reset) reset ;;
  stop-one) stop_one "$2" ;;
  start-one) start_one "$2" ;;
  drop) stop; psqlc -c "drop database if exists $SCEN_DB_MEM with (force)" -c "drop database if exists $SCEN_DB_RUNS with (force)"
        rm -rf "$H/memwork" "$H/runswork" "$H/redis" "$H/keys.env" ;;
  test) shift
    . "$H/keys.env"
    cd "$HERE" && env -i PATH="$PATH" HOME="$H/home" LANG=C.UTF-8 NO_PROXY='*' TMPDIR="$H/tmp" HF_HUB_OFFLINE=1 \
      PYTHONPATH="$HERE/src:$SCEN_MEM_ROOT/sdk/python/src:$SCEN_RUNS_ROOT/sdk/python/src:$SCEN_CONTRACTS:$SCEN_BIFROST" \
      MEMORY_URL=http://127.0.0.1:$SCEN_MEM_PORT RUNS_URL=http://127.0.0.1:$SCEN_RUNS_PORT \
      TRELLIS_API_KEY="$SCEN_KEY" TRELLIS_ADMIN_KEY="$SCEN_ADMIN_KEY" TRELLIS_API_KEY_OTHER="$SCEN_KEY2" \
      TRELLIS_MEMORY_OPENAPI="$SCEN_MEM_ROOT/docs/openapi.json" TRELLIS_RUNS_OPENAPI="$SCEN_RUNS_ROOT/docs/openapi.json" \
      SC_STACK="$0" SC="$H" "$SCEN_PY" -m pytest -m scenario tests/scenarios -p no:cacheprovider -rfEsxX "$@" ;;
esac
