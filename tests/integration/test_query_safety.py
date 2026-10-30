"""The in-process engine's resource safety (``tools.query``): each query bounded in time, memory,
input and output; its database made for it and closed whatever happens — an error, a timeout,
a cancelled run; a query sees only its own run's saved results; parallel runs do not mix.
The soak (many runs, sequential and at once: memory and open files stable) is
``tests/performance/test_query_soak.py``."""

from __future__ import annotations

import asyncio
import gc
import itertools
import os
import sqlite3
import threading
import time
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness.journal import content_key
from trellis.harness.tools import query, results

ASKED = itertools.count()
ENDLESS = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"


def table(n: int, tag: str = "a") -> str:
    return "id,tag,v\n" + "\n".join(f"{i},{tag},{i % 97}" for i in range(n))


class Opened:
    """Every database the engine opens, to see each closed (its close recorded: the test never
    touches a database a worker thread may be using)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.made: list[Any] = []
        real = sqlite3.connect
        made = self.made

        class Tracked(sqlite3.Connection):
            closed = False

            def close(self) -> None:
                super().close()
                self.closed = True

        def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            db = real(*args, factory=Tracked, **kwargs)
            made.append(db)
            return db

        monkeypatch.setattr(query.sqlite3, "connect", connect)

    def all_closed(self) -> bool:
        return bool(self.made) and all(db.closed for db in self.made)


async def ask(harness: Harness, text: str, sql: str, **wrap: Any) -> str:
    """``sql`` over ``text`` kept by a function run, as ``query_result`` answers it."""

    async def work(input: str, agent: Runtime) -> str:
        path = results.keep(text, tool="rows")
        assert path is not None
        return await query.query({"file_path": path, "sql": sql})

    result = await harness.wrap(work, id=f"q-{next(ASKED)}", **wrap).run("q", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    return str(result.answer)


# --------------------------------------------------------------------------- limits
async def test_a_query_past_its_time_is_stopped_and_its_database_closed(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = Opened(monkeypatch)
    monkeypatch.setattr(query, "SECONDS", 0.3)
    began = time.monotonic()
    said = await ask(harness, table(10), ENDLESS)
    assert "took too long" in said and time.monotonic() - began < 3
    assert opened.all_closed()


async def test_a_query_over_25_million_joined_rows_holds_little_memory_and_stops_in_time(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A join of 5,000 rows with themselves, made distinct: its sort spills past
    ``TEMP_CACHE_BYTES`` to SQLite's temporary files (gone when it closes), and the time cap
    stops it."""
    opened = Opened(monkeypatch)
    monkeypatch.setattr(query, "SECONDS", 1.0)
    before = rss_mb()
    sql = "SELECT count(*) FROM (SELECT DISTINCT a.id, b.id FROM rows a JOIN rows b)"
    said = await ask(harness, table(5_000), sql)
    assert "took too long" in said and rss_mb() - before < 100
    assert opened.all_closed()


async def test_the_input_and_the_answer_are_bounded(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = await ask(harness, table(1_000), "SELECT id, tag, v, printf('%0500d', v) FROM rows")
    assert len(said) < query.OUTPUT_CHARS + 100 and said.endswith("aggregate more]")
    monkeypatch.setattr(query, "LOAD_CHARS", 1_000)
    assert "more than a query here loads" in await ask(harness, table(1_000), "SELECT 1")


# --------------------------------------------------------------------------- lifecycle
async def test_a_cancelled_run_interrupts_its_query_and_closes_it(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = Opened(monkeypatch)
    monkeypatch.setattr(query, "SECONDS", 30.0)
    began = time.monotonic()

    async def work(input: str, agent: Runtime) -> str:
        path = results.keep(table(10), tool="rows")
        assert path is not None
        return await query.query({"file_path": path, "sql": ENDLESS})

    ended = await harness.wrap(work, id="q-cancelled", run_timeout=0.5).run("q", user="u")
    assert ended.status is RunStatus.TIMEOUT  # the run's time ran out while it queried
    for _ in range(100):  # the worker thread stops at once, interrupted
        if opened.all_closed():
            break
        await asyncio.sleep(0.05)
    assert opened.all_closed() and time.monotonic() - began < 10
    # its slot is free again: the next query runs at once
    assert await ask(harness, table(10), "SELECT count(*) FROM rows") == "count(*)\n10"


async def test_a_failed_query_closes_its_database(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = Opened(monkeypatch)
    assert "no such column" in await ask(harness, table(10), "SELECT missing FROM rows")
    assert opened.all_closed()


# --------------------------------------------------------------------------- isolation
async def test_a_query_sees_only_its_own_runs_saved_results(harness: Harness) -> None:
    @tool(side_effects="read")
    def mine(tag: str) -> str:
        """Rows tagged ``tag``."""
        return "x" * (results.MAX_CHARS - 100) + "\n" + table(3_000, tag)

    def plan(tag: str, other: str) -> list[Any]:
        own = "x" * (results.MAX_CHARS - 100) + "\n" + table(3_000, tag)
        theirs = "x" * (results.MAX_CHARS - 100) + "\n" + table(3_000, other)
        path = f"/large_tool_results/{content_key('result', own)}"
        elsewhere = f"/large_tool_results/{content_key('result', theirs)}"
        sql = "SELECT group_concat(name) FROM sqlite_master WHERE type = 'table'"
        return [
            ("mine", {"tag": tag}),
            ("query_result", {"file_path": path, "sql": sql}),
            ("query_result", {"file_path": elsewhere, "sql": "SELECT 1"}),
            "done",
        ]

    a, b = ScriptedChat(plan("a", "b")), ScriptedChat(plan("b", "a"))
    agent_a = arm(harness.wrap(ReAct(system_prompt="s", model=a), id="iso-a"), [mine])
    agent_b = arm(harness.wrap(ReAct(system_prompt="s", model=b), id="iso-b"), [mine])
    await asyncio.gather(agent_a.run("a", user="ua"), agent_b.run("b", user="ub"))
    for model in (a, b):
        tables = model.requests[2]["messages"][-1]["content"]
        assert tables == "group_concat(name)\nmine", tables  # its own result: one table
        assert model.requests[3]["messages"][-1]["content"].startswith("Error: File ")


# --------------------------------------------------------------------------- concurrency
def rss_mb() -> float:
    with open("/proc/self/status") as status:
        line = next(x for x in status if x.startswith("VmRSS:"))
    return int(line.split()[1]) / 1024


async def test_parallel_runs_each_query_50000_rows_and_none_mixes_with_another(
    harness: Harness,
) -> None:
    runs = 8

    async def work(input: str, agent: Runtime) -> str:
        n = int(input)
        path = results.keep(table(50_000, f"t{n}"), tool="rows")
        assert path is not None
        sql = "SELECT tag, count(*), sum(v) FROM rows GROUP BY tag"
        return await query.query({"file_path": path, "sql": sql})

    agent = harness.wrap(work, id="parallel-50k")
    before = rss_mb()
    began = time.monotonic()
    done = await asyncio.gather(*(agent.run(str(n), user=f"u{n}") for n in range(runs)))
    seconds = time.monotonic() - began
    grew = rss_mb() - before
    total = sum(i % 97 for i in range(50_000))
    assert [d.answer for d in done] == [
        f"tag,count(*),sum(v)\nt{n},50000,{total}" for n in range(runs)
    ]
    print(f"\n{runs} runs at once, 50,000 rows each: {seconds:.2f} s, RSS +{grew:.0f} MB")
    assert grew < 600  # at most CONCURRENT databases at once


async def test_loading_past_its_time_or_waiting_past_a_turn_says_so(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(query, "LOAD_SECONDS", -1.0)
    assert "took too long" in await ask(harness, table(10), "SELECT 1")
    monkeypatch.setattr(query, "WAIT_SECONDS", 0.01)
    held = [query._RUNNING.acquire() for _ in range(query.CONCURRENT)]
    try:
        assert "waited" in await ask(harness, table(10), "SELECT 1")
    finally:
        for _ in held:
            query._RUNNING.release()


def test_a_call_cancelled_before_its_turn_does_not_run() -> None:
    call = query._Call()
    call.cancel()
    assert query._answer({"t": "a,b\n1,2"}, "t", "SELECT 1", (), call=call) == "Error: cancelled"


# --------------------------------------------------------------------------- leaks
def fds() -> int:
    return len(os.listdir("/proc/self/fd"))


async def test_many_queries_leave_memory_open_files_and_threads_as_they_were(
    harness: Harness,
) -> None:
    """50 runs one after another, then 10 at once, each loading 10,000 rows and querying them:
    resident memory and open files end where the warm-up left them (within a small tolerance),
    and the threads are at most the event loop's executor's own."""
    total = sum(i % 97 for i in range(10_000))

    async def work(input: str, agent: Runtime) -> str:
        path = results.keep(table(10_000, input), tool="rows")
        assert path is not None
        return await query.query({"file_path": path, "sql": "SELECT sum(v) FROM rows"})

    agent = harness.wrap(work, id="soak")
    for n in range(5):  # warm-up: the worker threads made, the allocator's arenas grown
        await agent.run(f"w{n}", user="u")
    gc.collect()
    rss, files, threads = rss_mb(), fds(), threading.active_count()
    for n in range(50):
        assert (await agent.run(f"s{n}", user="u")).answer == f"sum(v)\n{total}"
    done = await asyncio.gather(*(agent.run(f"c{n}", user=f"u{n}") for n in range(10)))
    assert all(d.answer == f"sum(v)\n{total}" for d in done)
    gc.collect()
    grew, opened, started = rss_mb() - rss, fds() - files, threading.active_count() - threads
    print(f"\n50 + 10 query runs: RSS {grew:+.1f} MB, open files {opened:+d}, threads {started:+d}")
    pool = min(32, (os.cpu_count() or 1) + 4)  # asyncio's default executor: kept, reused
    assert grew < 40 and opened <= 2 and started <= pool
