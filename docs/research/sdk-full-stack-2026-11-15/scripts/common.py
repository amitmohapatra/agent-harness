# ruff: noqa: T201, PLC0415 - a research script: it reports with print, imports an SDK where a case needs it
"""Shared setup for the SDK full-stack proofs (real Haiku, the real Trellis stack).

Every Anthropic client goes through the meter (``meter_proxy.py`` of the 2026-11-15 Haiku
round, on :18477), which logs each response's provider usage: the spend is read from there.
The key is the caller's ``ANTHROPIC_API_KEY`` (set inside the command from the key file),
never printed or written.

``arm`` is one of ``bare`` (the SDK on its own), ``off`` (wrapped, middleware off, no
services: ``LocalRuns``, no memory), ``stack`` (wrapped, the private memory service and
agent-runs of ``scen_env.sh``), ``on`` (wrapped with the harness middleware, opt-in).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SDKFULL_DATA", HERE.parent / "data"))
METER_LOG = Path(os.environ.get("SDKFULL_METER", "/var/tmp/cons/sdkfull-tmp/meter.jsonl"))
MODEL = "claude-haiku-5-5"
METER = "http://127.0.0.1:18477"
os.environ["ANTHROPIC_BASE_URL"] = METER
PG = "postgresql://postgres@127.0.0.1:55957"
PRICE = {"in": 0.10, "cached": 0.01, "write": 0.125, "out": 0.50}  # benchmark.cost.prices


def key() -> str:
    got = os.environ.get("ANTHROPIC_API_KEY", "")
    if not got:
        sys.exit("ANTHROPIC_API_KEY is not set (read it from the key file inside the command)")
    return got


def mark() -> int:
    """The meter log's length now: :func:`spent` counts what came after."""
    return sum(1 for _ in METER_LOG.open()) if METER_LOG.exists() else 0


def _cost(u: dict[str, Any] | None) -> tuple[float, int, int]:
    if not u:
        return 0.0, 0, 0
    if "prompt_tokens" in u:
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        i, o = (u.get("prompt_tokens") or 0) - cached, u.get("completion_tokens") or 0
        return (i * PRICE["in"] + cached * PRICE["cached"] + o * PRICE["out"]) / 1e6, i + cached, o
    i, o = u.get("input_tokens") or 0, u.get("output_tokens") or 0
    cr, cw = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
    usd = (i * PRICE["in"] + cr * PRICE["cached"] + cw * PRICE["write"] + o * PRICE["out"]) / 1e6
    return usd, i + cr + cw, o


def spent(since: int) -> dict[str, Any]:
    lines = METER_LOG.read_text().splitlines()[since:] if METER_LOG.exists() else []
    usd = ti = to = n = 0
    models: set[str] = set()
    for line in lines:
        r = json.loads(line)
        if r["path"] == "/mcp":
            continue
        c, i, o = _cost(r.get("usage"))
        usd, ti, to, n = usd + c, ti + i, to + o, n + 1
        if r.get("model"):
            models.add(r["model"])
    return {"requests": n, "in": ti, "out": to, "usd": round(usd, 6), "models": sorted(models)}


# --------------------------------------------------------------------------- the stack
def stack_env() -> dict[str, str]:
    """The private stack's URLs and keys (``scen_env.sh`` + its ``keys.env``), never printed."""
    home = Path(os.environ.get("SCEN_HOME", "/var/tmp/cons/sdkfull-tmp/scen"))
    env: dict[str, str] = {}
    for line in (home / "keys.env").read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        env[name] = value
    return env


def harness(arm: str, **kw: Any) -> Any:
    from trellis import Harness, Settings
    from trellis.memory import MemoryClient

    if arm in ("bare", "off", "on"):
        return Harness(config=Settings(), **kw)
    env = stack_env()
    s = Settings(
        memory_url=f"http://127.0.0.1:{os.environ.get('SCEN_MEM_PORT', '18387')}",
        runs_url=f"http://127.0.0.1:{os.environ.get('SCEN_RUNS_PORT', '18397')}",
        api_key=env["SCEN_KEY"],
    )

    class Owned(Harness):
        async def aclose(self) -> None:
            await super().aclose()
            if self.memory is not None:
                await self.memory.client.aclose()

    assert s.memory_url is not None
    return Owned(config=s, memory=MemoryClient(s.memory_url, api_key=s.api_key, timeout=60.0), **kw)


def sql(db: str, query: str, *params: str) -> list[dict[str, Any]]:
    """Read-only rows of ``query`` in the stack's ``memory`` or ``runs`` database."""
    for i, value in enumerate(params, 1):
        query = query.replace(f":{i}", "'" + value.replace("'", "''") + "'")
    wrapped = f"select coalesce(json_agg(t), '[]'::json) from ({query}) t"
    out = subprocess.run(
        ["psql", f"{PG}/sdkfull_{db}", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", wrapped],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(out.stdout)


def _runs_api(path: str) -> Any:
    import httpx

    port = os.environ.get("SCEN_RUNS_PORT", "18397")
    got = httpx.get(
        f"http://127.0.0.1:{port}{path}", timeout=30, headers={"X-API-Key": stack_env()["SCEN_KEY"]}
    )
    got.raise_for_status()
    return got.json()


def records(run_id: str) -> dict[str, Any]:
    """What agent-runs (its API: an ended run's events are archived out of ``run_events``)
    and the memory service (its tables) hold for a run and its children."""
    rec = _runs_api(f"/v1/runs/{run_id}")
    run = {
        k: rec.get(k)
        for k in (
            "run_id",
            "status",
            "parent_run_id",
            "thread_id",
            "user_id",
            "agent_id",
            "attempt",
            "error",
        )
    }
    run["awaiting"] = rec.get("awaiting") is not None
    kids = sql(
        "runs",
        "select run_id, status, agent_id, error from agent_runs"
        " where parent_run_id = :1 order by created_at",
        run_id,
    )
    ids = [run_id, *(k["run_id"] for k in kids)]
    events: dict[str, dict[str, int]] = {}
    for rid in ids:
        counts: dict[str, int] = {}
        for e in _runs_api(f"/v1/runs/{rid}/events?limit=500"):
            t = e["event"]["type"]
            counts[t] = counts.get(t, 0) + 1
        events[rid] = counts
    messages = {
        rid: sql(
            "memory",
            "select role, left(content, 300) as content from messages"
            " where agent_run_id = :1 and deleted_at is null order by sequence",
            rid,
        )
        for rid in ids
    }
    tools = {
        rid: sql(
            "memory",
            "select tool_name, status, left(output_summary, 200) as output"
            " from tool_invocations where run_id = :1 order by occurred_at",
            rid,
        )
        for rid in ids
    }
    feedback = sql(
        "memory",
        "select target_kind, verdict, source from feedback where agent_run_id = :1",
        run_id,
    )
    return {
        "run": run,
        "children": kids,
        "events": events,
        "messages": messages,
        "tools": tools,
        "feedback": feedback,
    }


def memories(user: str) -> list[dict[str, Any]]:
    return sql(
        "memory",
        "select memory_type, left(content, 200) as content from memories"
        " where visibility_keys::text like :1 and deleted_at is null"
        " and superseded_by is null",
        f'%"user:sdkfull/{user}"%',
    )


# --------------------------------------------------------------------------- results
def uid() -> str:
    return uuid.uuid4().hex[:8]


async def case(
    sdk: str, feature: str, arm: str, body: Callable[[], Awaitable[dict[str, Any]]]
) -> dict[str, Any]:
    """Run one (feature, arm), save what it saw with its spend to ``data/<sdk>/``."""
    since, began = mark(), time.time()
    try:
        seen = await body()
        error = None
    except Exception as exc:  # the case's failure is data, not a crash of the batch
        seen, error = {}, "".join(traceback.format_exception_only(exc)).strip()[-2000:]
        seen["traceback"] = traceback.format_exc()[-4000:]
    out = {
        "sdk": sdk,
        "feature": feature,
        "arm": arm,
        "error": error,
        "seconds": round(time.time() - began, 1),
        "spend": spent(since),
        **seen,
    }
    folder = DATA / sdk
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1, default=str)
    secret = os.environ.get("ANTHROPIC_API_KEY")
    assert not secret or secret not in text, "the key reached a result"
    (folder / f"{feature}.{arm}.json").write_text(text)
    verdict = out.get("ok")
    print(f"{sdk:14} {feature:26} {arm:5} ok={verdict} err={bool(error)} ${out['spend']['usd']}")
    return out


def chosen(argv: list[str], every: dict[str, Any]) -> list[tuple[str, str]]:
    """``feature:arm`` pairs from the command line (``all``, ``feature``, ``feature:arm``)."""
    arms = ("bare", "off", "stack", "on")
    picks: list[tuple[str, str]] = []
    for a in argv or ["all"]:
        f, _, arm = a.partition(":")
        for name in every if f == "all" else [f]:
            for one in [arm] if arm else arms:
                if one in every[name]:
                    picks.append((name, one))
    return picks
