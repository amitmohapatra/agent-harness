"""One measurement in its own process (so its peak memory is its own): ``python -m
benchmark.query_engines.worker <mode> <json arguments>``, printing one JSON object.

Modes: ``truth`` (write the data and Python's answers for a size), ``measure`` (an engine's
load time, every question's correctness and query time, its peak memory), ``concurrent``
(8 calls at once, each loading and querying as the tool does), ``safety`` (what a query may
do beyond reading its own tables) and ``fidelity`` (how values come through: codes with a
leading zero, integers past 64 bits, empty cells, nested values)."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from benchmark import procstat
from benchmark.query_engines import engines, suite


def _texts(data: Path, fmt: str) -> dict[str, str]:
    return {
        "ledger": (data / f"ledger.{fmt}").read_text(),
        "inventory": (data / "inventory.csv").read_text(),
        "sales": (data / "sales.json").read_text(),
        "lead_times": (data / "lead_times.json").read_text(),
    }


def truth(args: dict[str, Any]) -> dict[str, Any]:
    data = Path(args["data"])
    paths = suite.write(data, args["rows"])
    answers = suite.truths(args["rows"])
    (data / "truth.json").write_text(json.dumps(answers))
    return {"bytes": {p.name: p.stat().st_size for p in paths.values()}}


def _quantiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    return {
        "p50": statistics.median(xs),
        "p95": xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))],
    }


def measure(args: dict[str, Any]) -> dict[str, Any]:
    engine = engines.ENGINES[args["engine"]]()
    data = Path(args["data"])
    texts = _texts(data, args["fmt"])
    want = json.loads((data / "truth.json").read_text())
    base = procstat.rss_mb() or 0.0
    loads = []
    db = None
    for n in range(args["load_reps"]):
        seconds, db = engines.timed(lambda: engine.load(texts, "ledger"))
        loads.append(seconds)
        if n < args["load_reps"] - 1:
            engine.close(db)
    loaded = procstat.rss_mb() or 0.0
    out: dict[str, Any] = {"load": _quantiles(loads), "questions": {}}
    for q in suite.QUESTIONS:
        found = _answer(engine, db, q, want[q.name])
        if found["sql"] is not None:
            times = []
            for _ in range(args["query_reps"]):
                seconds, _ = engines.timed(lambda sql=found["sql"]: engine.run(db, sql))
                times.append(seconds)
            found |= _quantiles(times)
        del found["sql"]
        out["questions"][q.name] = found
    engine.close(db)
    out["rss_loaded_mb"] = loaded - base
    out["peak_rss_mb"] = (procstat.peak_rss_mb() or 0.0) - base
    out["text_mb"] = sum(map(len, texts.values())) / 1e6
    return out


def _answer(engine: Any, db: Any, q: suite.Question, want: list[list[Any]]) -> dict[str, Any]:
    """Whether the harness's SQL answers right as it is, else the engine's dialect does."""
    tries = [("as-is", q.sql)]
    if engine.name in q.dialects:
        tries.append(("dialect", q.dialects[engine.name]))
    said = []
    for how, sql in tries:
        try:
            _, rows = engine.run(db, sql)
        except Exception as exc:
            said.append(f"{how}: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            continue
        got = [[engines.value(v) for v in r] for r in rows]
        if suite.same(got, want):
            return {"status": how, "sql": sql, "said": said}
        near = suite.same(got, want, rel=1e-9)
        said.append(f"{how}: wrong{' (within 1e-9)' if near else ''}: {str(got)[:160]}")
    return {
        "status": "wrong" if any("wrong" in s for s in said) else "error",
        "sql": None,
        "said": said,
    }


def concurrent(args: dict[str, Any]) -> dict[str, Any]:
    """``args["calls"]`` calls at once, each loading the texts and asking one question — the
    harness's own path for SQLite (``query._answer``: its 4-at-once limit and its 5 s)."""
    name = args["engine"]
    texts = _texts(Path(args["data"]), "csv")
    sql = next(q for q in suite.QUESTIONS if q.name == args["question"]).sql
    base = procstat.rss_mb() or 0.0
    gate = threading.Barrier(args["calls"])
    latencies: list[float] = []
    errors: list[str] = []
    if name == "sqlite":
        from trellis.harness.tools import query

        def call() -> None:
            gate.wait()
            start = time.perf_counter()
            said = query._answer(texts, "ledger", sql, (), call=query._Call())
            latencies.append(time.perf_counter() - start)
            if said.startswith("Error"):
                errors.append(said[:120])
    else:
        engine = engines.ENGINES[name]()

        def call() -> None:
            gate.wait()
            start = time.perf_counter()
            try:
                with engines.opened(engine, texts, "ledger") as db:
                    engine.run(db, sql)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:120])
            latencies.append(time.perf_counter() - start)

    threads = [threading.Thread(target=call) for _ in range(args["calls"])]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    return {
        "wall_s": wall,
        **_quantiles(latencies),
        "errors": errors,
        "peak_rss_mb": (procstat.peak_rss_mb() or 0.0) - base,
    }


# ------------------------------------------------------------------------------------ safety

ENDLESS = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"


def safety(args: dict[str, Any]) -> dict[str, Any]:
    """Each probe's outcome: ``blocked`` (refused before it touched anything), ``ALLOWED``
    (it did what it asked), or what happened."""
    name = args["engine"]
    folder = Path(tempfile.mkdtemp(prefix="probe-", dir=args["scratch"]))
    secret_csv = folder / "secret.csv"
    secret_csv.write_text("token\nhunter2\n")
    secret_db = folder / "secret.db"
    with contextlib.closing(sqlite3.connect(secret_db)) as made:
        made.execute("CREATE TABLE secret (token TEXT)")
        made.execute("INSERT INTO secret VALUES ('hunter2')")
        made.commit()
    texts = _texts(Path(args["data"]), "csv")
    engine = engines.ENGINES[name]()
    probes = _probes(name, folder)
    out: dict[str, Any] = {}
    db = engine.load(texts, "ledger")
    for probe, sql in probes.items():
        try:
            _, rows = engine.run(db, sql)
            out[probe] = f"ALLOWED: {str(rows)[:80]}"
        except Exception as exc:
            out[probe] = _refused(exc)
    out["file written"] = "ALLOWED" if any(folder.glob("out*")) else "no"
    out["timeout"] = _timeout(name, engine, db, texts)
    out["memory"] = _memory(name, engine, db)
    engine.close(db)
    return out


def _refused(exc: Exception) -> str:
    """A probe's error: a refusal, or a sign it ran (pandasql: a statement with no rows to read;
    a connection that was tried)."""
    said = f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
    if "'NoneType' object has no attribute 'columns'" in said:
        return "ALLOWED (it ran; no rows to read)"
    if "Connection" in said or "HTTP error" in said:
        return f"ALLOWED (the connection was tried: {said[:60]})"
    return f"blocked: {said}"


def _probes(name: str, folder: Path) -> dict[str, str]:
    if name in ("sqlite", "pandasql"):
        return {
            "read another file": f"ATTACH '{folder}/secret.db' AS s",
            "write a file": f"ATTACH '{folder}/out.db' AS o",
            "load an extension": "SELECT load_extension('/nonexistent')",
            "change a setting": "PRAGMA writable_schema = 1",
            "write a table": "CREATE TABLE z AS SELECT 1",
            "delete rows": "DELETE FROM ledger",
        }
    common = {
        "read another file": f"SELECT * FROM read_csv('{folder}/secret.csv')",
        "network": "SELECT * FROM read_csv('http://127.0.0.1:9/x.csv')",
        "write a table": "CREATE TABLE z AS SELECT 1 AS a",
        "delete rows": "DELETE FROM ledger",
    }
    if name == "duckdb":
        common |= {
            "write a file": f"COPY (SELECT 1) TO '{folder}/out.csv'",
            "load an extension": "LOAD httpfs",
            "install an extension": "INSTALL httpfs",
            "change a setting": "SET enable_external_access = true",
            "list files": "SELECT * FROM glob('/etc/*')",
        }
    return common


def _timeout(name: str, engine: Any, db: Any, texts: dict[str, str]) -> str:
    """How an endless query is stopped after 2 s (SQLite: the harness's own 5 s)."""
    start = time.perf_counter()
    if name == "sqlite":
        from trellis.harness.tools import query

        said = query._answer(texts, "ledger", ENDLESS, (), call=query._Call())
        return f"{said[:60]} after {time.perf_counter() - start:.1f} s"
    if name == "duckdb":
        timer = threading.Timer(2.0, db.interrupt)
        timer.start()
        try:
            engine.run(db, ENDLESS)
            return "ran to the end"
        except Exception as exc:
            return f"stopped ({type(exc).__name__}) after {time.perf_counter() - start:.1f} s"
        finally:
            timer.cancel()
    return "no way to interrupt a running query (not run)"


def _memory(name: str, engine: Any, db: Any) -> str:
    """What a query that builds a large value in memory does (about 1 GB of text)."""
    big = (
        "SELECT length(group_concat(sku || '{pad}', ',')) FROM ledger, "
        "(WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 200) "
        "SELECT x FROM c)"
    ).format(pad="x" * 90)
    if name == "duckdb":
        big = big.replace("group_concat", "string_agg")
    if name not in ("sqlite", "duckdb"):
        return "no memory limit setting (not run)"
    before = procstat.rss_mb() or 0.0
    start = time.perf_counter()
    try:
        _, rows = engine.run(db, big)
        said = f"ran: {rows}"
    except Exception as exc:
        said = f"stopped: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
    grew = (procstat.peak_rss_mb() or 0.0) - before
    return f"{said}; peak +{grew:.0f} MB, {time.perf_counter() - start:.1f} s"


# ---------------------------------------------------------------------------------- fidelity

FIDELITY_CSV = "code,big,price,note\n007,12345678901234567890,1.50,\n010,1,2,x\n"
FIDELITY_JSON = json.dumps(
    [
        {"code": "007", "big": 12345678901234567890, "nested": {"a": 1}, "flag": True},
        {"code": "010", "big": 1, "nested": {"a": 2}, "flag": False},
    ]
)


def fidelity(args: dict[str, Any]) -> dict[str, Any]:
    engine = engines.ENGINES[args["engine"]]()
    out: dict[str, Any] = {}
    for label, text, sql in (
        ("csv", FIDELITY_CSV, "SELECT code, big, price, note FROM result ORDER BY code"),
        ("json", FIDELITY_JSON, "SELECT code, big, nested, flag FROM result ORDER BY code"),
    ):
        try:
            with engines.opened(engine, {"t": text}, "t") as db:
                _, rows = engine.run(db, sql)
            out[label] = [[engines.value(v) for v in r] for r in rows]
        except Exception as exc:
            out[label] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
    return out


MODES = {
    "truth": truth,
    "measure": measure,
    "concurrent": concurrent,
    "safety": safety,
    "fidelity": fidelity,
}

if __name__ == "__main__":
    mode, raw = sys.argv[1], sys.argv[2]
    try:
        result: Any = MODES[mode](json.loads(raw))
    except ImportError as exc:
        result = {"not_installed": str(exc)}
    sys.stdout.write(json.dumps(result, default=str) + "\n")
    sys.stdout.flush()
    os._exit(0)
