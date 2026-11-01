"""The in-process engine's resource safety (``tools.query``), on DuckDB and on the SQLite
fallback (each test taking ``engine``): each query bounded in time, memory, input and output;
its database made for it and closed whatever happens — an error, a timeout, a cancelled run; a
query sees only its own run's saved results, never a file, the network, an extension or a
setting; parallel runs do not mix; many runs leave memory, open files and threads as they
were."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import itertools
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
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
        self.made: list[bool] = []  # whether each is closed
        engine = query.ENGINE
        real = engine.opened
        made = self.made

        @contextlib.contextmanager
        def opened() -> Iterator[Any]:
            n = len(made)
            made.append(False)
            with real() as db:
                yield db
            made[n] = True

        monkeypatch.setattr(engine, "opened", opened)

    def all_closed(self) -> bool:
        return bool(self.made) and all(self.made)


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
    harness: Harness, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = Opened(monkeypatch)
    monkeypatch.setattr(query, "SECONDS", 0.3)
    began = time.monotonic()
    said = await ask(harness, table(10), ENDLESS)
    assert "took too long" in said and time.monotonic() - began < 3
    assert opened.all_closed()


async def test_a_query_over_25_million_joined_rows_holds_little_memory_and_stops_in_time(
    harness: Harness, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A join of 5,000 rows with themselves, made distinct: SQLite's sort spills past
    ``TEMP_CACHE_BYTES`` to its temporary files (gone when it closes) and the time cap stops
    it; DuckDB's is stopped by its ``memory_limit`` (never spilled) or by the time cap."""
    opened = Opened(monkeypatch)
    monkeypatch.setattr(query, "SECONDS", 1.0)
    before = rss_mb()
    sql = (
        "SELECT count(*) FROM (SELECT DISTINCT a.id || '-' || b.id || '-' || a.tag "
        "FROM rows a, rows b)"
    )
    said = await ask(harness, table(5_000), sql)
    grew = rss_mb() - before
    assert "took too long" in said or "memory" in said.lower(), said
    assert grew < {"sqlite": 100, "duckdb": query.DB_BYTES / 2**20 + 100}[engine]
    assert opened.all_closed()


async def test_the_input_and_the_answer_are_bounded(
    harness: Harness, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = await ask(harness, table(1_000), "SELECT id, tag, v, printf('%0500d', v) FROM rows")
    assert len(said) < query.OUTPUT_CHARS + 100 and said.endswith("aggregate more]")
    monkeypatch.setattr(query, "LOAD_CHARS", 1_000)
    assert "more than a query here loads" in await ask(harness, table(1_000), "SELECT 1")


# --------------------------------------------------------------------------- lifecycle
async def test_a_cancelled_run_interrupts_its_query_and_closes_it(
    harness: Harness, engine: str, monkeypatch: pytest.MonkeyPatch
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
    assert await ask(harness, table(10), "SELECT count(*) AS n FROM rows") == "n\n10"


async def test_a_failed_query_closes_its_database(
    harness: Harness, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = Opened(monkeypatch)
    said = await ask(harness, table(10), "SELECT missing FROM rows")
    assert said.startswith("Error: the query failed:") and "missing" in said
    assert opened.all_closed()


# --------------------------------------------------------------------------- isolation
async def test_a_query_sees_only_its_own_runs_saved_results(harness: Harness, engine: str) -> None:
    @tool(side_effects="read")
    def mine(tag: str) -> str:
        """Rows tagged ``tag``."""
        return "x" * (results.MAX_CHARS - 100) + "\n" + table(3_000, tag)

    def plan(tag: str, other: str) -> list[Any]:
        own = "x" * (results.MAX_CHARS - 100) + "\n" + table(3_000, tag)
        theirs = "x" * (results.MAX_CHARS - 100) + "\n" + table(3_000, other)
        path = f"/large_tool_results/{content_key('result', own)}"
        elsewhere = f"/large_tool_results/{content_key('result', theirs)}"
        sql = "SELECT group_concat(name) AS names FROM sqlite_master WHERE type = 'table'"
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
        assert tables == "names\nmine", tables  # its own result: one table
        assert model.requests[3]["messages"][-1]["content"].startswith("Error: File ")


# --------------------------------------------------------------------------- hostile queries
def probes(engine: str, folder: Path) -> dict[str, str]:
    """A query trying more than reading its own tables, as each engine's SQL spells it."""
    secret = folder / "secret.csv"
    if engine == "sqlite":
        return {
            "read another file": f"ATTACH '{folder}/secret.db' AS s",
            "write a file": f"ATTACH '{folder}/out.db' AS o",
            "load an extension": "SELECT load_extension('/nonexistent')",
            "change a setting": "PRAGMA writable_schema = 1",
            "write a table": "CREATE TABLE z AS SELECT 1",
            "delete rows": "DELETE FROM rows",
            "two statements": "SELECT 1; DELETE FROM rows",
        }
    return {
        "read another file": f"SELECT * FROM read_csv('{secret}')",
        "read a file as text": f"SELECT * FROM read_text('{secret}')",
        "list files": f"SELECT * FROM glob('{folder}/*')",
        "network": "SELECT * FROM read_csv('http://127.0.0.1:9/x.csv')",
        "write a file": f"COPY (SELECT 1) TO '{folder}/out.csv'",
        "load an extension": "LOAD httpfs",
        "install an extension": "INSTALL httpfs",
        "change a setting": "SET enable_external_access = true",
        "change a setting in a SELECT": "SELECT set_config('enable_external_access', 'true')",
        "write a table": "CREATE TABLE z AS SELECT 1 AS a",
        "delete rows": "DELETE FROM rows",
        "delete rows inside a SELECT": "SELECT * FROM query('DELETE FROM rows')",
        "attach a database": f"ATTACH '{folder}/out.db' AS o",
        "two statements": "SELECT 1; DELETE FROM rows",
    }


async def test_a_query_reads_only_its_own_tables(
    harness: Harness, engine: str, tmp_path: Path
) -> None:
    """No other file, no network, no extension, no setting, no write: each is refused, and the
    run's own table is whole after."""
    (tmp_path / "secret.csv").write_text("token\nhunter2\n")
    with contextlib.closing(sqlite3.connect(tmp_path / "secret.db")) as made:
        made.execute("CREATE TABLE secret (token TEXT)")
        made.execute("INSERT INTO secret VALUES ('hunter2')")
        made.commit()
    for probe, sql in probes(engine, tmp_path).items():
        said = await ask(harness, table(10), sql)
        assert said.startswith("Error: the query failed:") and "hunter2" not in said, (probe, said)
    assert not [p.name for p in os.scandir(tmp_path) if p.name.startswith("out")]
    assert await ask(harness, table(10), "SELECT count(*) AS n FROM rows") == "n\n10"


async def test_a_query_building_a_huge_value_is_stopped(harness: Harness, engine: str) -> None:
    """A query that builds ~200 MB of text in memory: SQLite stops it at its longest value
    (``VALUE_CHARS``), DuckDB at its ``memory_limit`` (``DB_BYTES``)."""
    concat = {"sqlite": "group_concat", "duckdb": "string_agg"}[engine]
    sql = f"SELECT length({concat}(a.tag || printf('%0200d', b.id), ',')) FROM rows a, rows b"
    before = rss_mb()
    said = await ask(harness, table(1_000, "x" * 1_000), sql)
    assert said.startswith("Error: the query failed:"), said[:200]
    assert rss_mb() - before < query.DB_BYTES / 2**20 + 100


# --------------------------------------------------------------------------- concurrency
def rss_mb() -> float:
    with open("/proc/self/status") as status:
        line = next(x for x in status if x.startswith("VmRSS:"))
    return int(line.split()[1]) / 1024


async def test_parallel_runs_each_query_50000_rows_and_none_mixes_with_another(
    harness: Harness,
    engine: str,
) -> None:
    runs = 8

    async def work(input: str, agent: Runtime) -> str:
        n = int(input)
        path = results.keep(table(50_000, f"t{n}"), tool="rows")
        assert path is not None
        sql = "SELECT tag, count(*) AS n, sum(v) AS s FROM rows GROUP BY tag"
        return await query.query({"file_path": path, "sql": sql})

    agent = harness.wrap(work, id="parallel-50k")
    before = rss_mb()
    began = time.monotonic()
    done = await asyncio.gather(*(agent.run(str(n), user=f"u{n}") for n in range(runs)))
    seconds = time.monotonic() - began
    grew = rss_mb() - before
    total = sum(i % 97 for i in range(50_000))
    assert [d.answer for d in done] == [f"tag,n,s\nt{n},50000,{total}" for n in range(runs)]
    print(f"\n{runs} runs at once, 50,000 rows each: {seconds:.2f} s, RSS +{grew:.0f} MB")
    assert grew < 600  # at most CONCURRENT databases at once


async def test_loading_past_its_time_or_waiting_past_a_turn_says_so(
    harness: Harness, engine: str, monkeypatch: pytest.MonkeyPatch
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


def test_a_call_cancelled_before_its_turn_does_not_run(engine: str) -> None:
    call = query._Call()
    call.cancel()
    assert query._answer({"t": "a,b\n1,2"}, "t", "SELECT 1", (), call=call) == "Error: cancelled"


# --------------------------------------------------------------------------- leaks
def fds() -> int:
    return len(os.listdir("/proc/self/fd"))


async def test_many_queries_leave_memory_open_files_and_threads_as_they_were(
    harness: Harness,
    engine: str,
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
