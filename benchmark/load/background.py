"""Background agents under load (``python -m benchmark.load.background``): many open chats, many
running background tasks and users asking how they go, steering and stopping them — against
agent-runs (``RUNS_URL``, its key registry at ``MEMORY_URL``) with a scripted model, no
provider. Two arms on the same users and the same runs:

* ``A`` — background on: each user's chat has a lead ``ReAct`` with ``background(research)``;
  a turn starts the task (``start_async_task``), and the users' turns check it, list them,
  steer it and cancel it through the lead's tools;
* ``B`` — background off: the same research runs started as plain queued runs
  (``agent.start``), read (``runs.get``), listed (the chat's runs), steered (the run's inbox)
  and cancelled (``agent.cancel``) directly.

The research task is a ``ReAct`` whose model takes ``--pace`` seconds per call and calls a
tool ``--steps`` times; ``--workers`` worker processes work the tasks, ``--tasks`` at once.
``--chats`` idle conversations (a finished turn each) are seeded straight into agent-runs'
database (``--pg``), as many turns would leave them. During ``--seconds`` it measures the CPU
and memory of this process, the workers and ``--watch name=pid`` processes; the database's
statements per second by kind (``pg_stat_statements``, reset at the start); its connections;
each user operation's latency; and errors. Writes ``<out>.json``.

The stack (a private PostgreSQL, agent-runs, a key registry stub) is
``benchmark/load/background_stack.sh``; ``docs/evaluation.md#background-load`` has the runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from benchmark import procstat

#: Statement kinds, by what the SQL touches (the first that matches).
KINDS = (
    ("inbox", re.compile(r"run_messages", re.I)),
    ("events", re.compile(r"run_events", re.I)),
    ("claim", re.compile(r"SKIP LOCKED", re.I)),
    ("heartbeat", re.compile(r"UPDATE agent_runs SET .*lease_expires_at", re.I | re.S)),
    (
        "task list",
        re.compile(r"FROM agent_runs .*ORDER BY agent_runs\.created_at DESC", re.I | re.S),
    ),
    ("run read", re.compile(r"FROM agent_runs", re.I)),
)
#: What a user does on a running task, and how often.
OPERATIONS = ("status", "list", "steer", "cancel")
WEIGHTS = (0.4, 0.3, 0.25, 0.05)


class Scripted(BaseChatModel):
    """A model that answers after ``pace`` seconds. A research task: ``step`` ``steps`` times,
    then its answer. A lead: the command its user's message names (``start``, ``status <id>``,
    ``list``, ``steer <id> <text>``, ``cancel <id>``) as a tool call, then the tool's result."""

    pace: float = 0.0
    steps: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Scripted answers asynchronously")  # pragma: no cover

    async def _agenerate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        await asyncio.sleep(self.pace)
        last = messages[-1]
        if self.steps:  # a research task
            done = sum(isinstance(m, ToolMessage) for m in messages)
            call = ("step", {"n": done}) if done < self.steps else None
            text = f"done after {done} steps"
        elif isinstance(last, ToolMessage):
            call, text = None, str(last.content)[:300]
        else:
            call, text = _command(_human(messages)), "hello"
        if call is None:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        name, given = call
        tool_call = {"name": name, "args": given, "id": f"c{time.time_ns()}"}
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage("", tool_calls=[tool_call]))]
        )


def _human(messages: list[BaseMessage]) -> str:
    said = next(m for m in reversed(messages) if isinstance(m, HumanMessage)).content
    return said if isinstance(said, str) else json.dumps(said)


def _command(text: str) -> tuple[str, dict[str, Any]] | None:
    found = re.search(r"CMD (\w+) ?(\S*) ?(.*?)(?:\"|'|$)", text)
    if found is None:
        return None
    verb, task, rest = found.groups()
    return {
        "start": (
            "start_async_task",
            {"description": "research", "subagent_type": "research"},
        ),
        "status": ("check_async_task", {"task_id": task}),
        "list": ("list_async_tasks", {}),
        "steer": ("update_async_task", {"task_id": task, "message": rest or "go on"}),
        "cancel": ("cancel_async_task", {"task_id": task}),
    }.get(verb)


def step(n: int) -> str:
    """One step of research.

    Args:
        n: the step's number
    """
    return f"step {n} done"


def research(h: Any, pace: float, steps: int) -> Any:
    from trellis import ReAct, tool

    model = Scripted(pace=pace, steps=steps)
    target = ReAct(
        "You research.",
        model,
        tools=[tool(side_effects="read")(step)],
        max_steps=steps + 2,
    )
    return h.wrap(target, id="research", without={"memory"}, run_timeout=3600)


def lead(h: Any, child: Any, background_on: bool) -> Any:
    from trellis import ReAct, background

    tools = [background(child)] if background_on else []
    return h.wrap(ReAct("You lead.", Scripted(), tools=tools), id="lead", without={"memory"})


def harness() -> Any:
    from trellis import Harness, Settings

    config = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    return Harness(config=config, gateway=False)


# --------------------------------------------------------------------------- the workers
@dataclass(frozen=True)
class Served:
    concurrency: int
    seconds: float
    pace: float
    steps: int


async def _serve(served: Served) -> dict[str, Any]:
    began, cpu = time.monotonic(), procstat.cpu_seconds()
    rss: list[float] = []
    async with harness() as h:
        worker = h.worker([research(h, served.pace, served.steps)], concurrency=served.concurrency)
        running = asyncio.create_task(worker.run())
        while time.monotonic() - began < served.seconds:
            rss.append(procstat.rss_mb() or 0.0)
            await asyncio.sleep(5)
        worker.stop()
        worker.stop()
        await asyncio.gather(running, return_exceptions=True)
    return {
        "cpu_seconds": (procstat.cpu_seconds() or 0) - (cpu or 0),
        "rss_mb_max": max(rss or [0]),
    }


def serve(served: Served) -> dict[str, Any]:
    return asyncio.run(_serve(served))


# --------------------------------------------------------------------------- more drivers
@dataclass(frozen=True)
class Share:
    """What another driver process does: its users operate these tasks from ``start_at`` to
    ``end_at`` (wall clock), through a harness of its own."""

    arm: str
    direct: bool
    pace: float
    steps: int
    tasks: dict[str, tuple[str, str]]
    users: int
    think: float
    start_at: float
    end_at: float


async def _operate(share: Share) -> dict[str, Any]:
    async with harness() as h:
        child = research(h, share.pace, share.steps)
        users = Users(share.arm, h, child, lead(h, child, share.arm == "A"))
        users.direct, users.tasks = share.direct, dict(share.tasks)
        await asyncio.sleep(max(0.0, share.start_at - time.time()))
        cpu = procstat.cpu_seconds()
        until = time.monotonic() + share.end_at - time.time()
        await asyncio.gather(*(users.lane(until, share.think) for _ in range(share.users)))
        used = (procstat.cpu_seconds() or 0) - (cpu or 0)
    return {
        "latency": dict(users.latency),
        "errors": dict(users.errors),
        "cpu_seconds": used,
    }


def operate(share: Share) -> dict[str, Any]:
    """Another driver process's entry: its own event loop."""
    return asyncio.run(_operate(share))


# --------------------------------------------------------------------------- the database
def _sql(dsn: str, query: str) -> list[list[str]]:
    """``query``'s rows on ``dsn``, by ``psql`` (no driver needed here)."""
    out = subprocess.run(
        ["psql", dsn, "-AtqX", "-F", "\t", "-c", query],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def seed_chats(dsn: str, chats: int, users: int) -> None:
    """``chats`` idle conversations: one finished turn of the lead each, as a turn leaves it."""
    tenant = os.environ.get("BENCH_TENANT", "t1")
    _sql(dsn, "delete from agent_runs where run_id like 'seed_%'")
    _sql(
        dsn,
        "insert into agent_runs (run_id, tenant_id, agent_id, thread_id, user_id, status, input,"
        " output, run_metadata, attempt, worked_seconds, priority, created_at, updated_at)"
        f" select 'seed_'||g, '{tenant}', 'lead', 'idle'||g, 'u'||(g % {users}), 'SUCCESS',"
        " '\"hello\"', '\"hello\"', '{}', 1, 0, 0, now() - interval '1 hour',"
        f" now() - interval '1 hour' from generate_series(1, {chats}) g",
    )
    _sql(dsn, "analyze agent_runs")


def statements(dsn: str, *, reset: bool = False) -> dict[str, int]:
    """The statements run since the last reset, by kind (:data:`KINDS`)."""
    if reset:
        _sql(dsn, "select pg_stat_statements_reset()")
        return {}
    rows = _sql(
        dsn,
        "select regexp_replace(query, '\\s+', ' ', 'g'), calls from pg_stat_statements where dbid ="
        " (select oid from pg_database where datname = current_database())",
    )
    kinds: Counter[str] = Counter()
    for query, calls in rows:
        if "pg_stat" in query or query.strip().upper() in (
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
        ):
            continue
        kind = next((k for k, pattern in KINDS if pattern.search(query)), "other")
        kinds[kind] += int(calls)
    return dict(kinds)


def connections(dsn: str) -> int:
    return int(
        _sql(
            dsn,
            "select count(*) from pg_stat_activity where datname = current_database()",
        )[0][0]
    )


def rows_written(dsn: str) -> dict[str, int]:
    return {
        table: int(_sql(dsn, f"select count(*) from {table}")[0][0])
        for table in ("agent_runs", "run_events", "run_messages")
    }


# --------------------------------------------------------------------------- the users
class Users:
    """The users' chats and tasks, and what each operation took."""

    def __init__(self, arm: str, h: Any, child: Any, chief: Any) -> None:
        self.arm, self.h, self.child, self.lead = arm, h, child, chief
        self.tasks: dict[str, tuple[str, str]] = {}  # task id -> (user, chat)
        self.latency: defaultdict[str, list[float]] = defaultdict(list)
        self.errors: Counter[str] = Counter()
        self.direct = False

    async def turn(self, user: str, chat: str, command: str) -> str:
        result = await self.lead.run(f"CMD {command}", user=user, thread=chat)
        if result.error is not None:
            raise RuntimeError(result.error.message)
        return str(result.answer)

    async def start(self, n: int) -> None:
        user, chat = f"u{n}", f"chat{n}"
        if self.arm == "A":
            said = await self.turn(user, chat, "start")
            task = said.rsplit("task_id: ", 1)[1].strip()
        else:
            task = (await self.child.start("research", user=user, thread=chat)).id
        self.tasks[task] = (user, chat)

    async def operate(self, kind: str) -> None:
        task, (user, chat) = random.choice(list(self.tasks.items()))
        began = time.monotonic()
        try:
            if self.arm == "A" and self.direct:  # as a UI does: AG-UI's task routes
                await self.direct_operation(kind, task, user, chat)
            elif self.arm == "A":
                command = {
                    "status": f"status {task}",
                    "list": "list",
                    "steer": f"steer {task} also this",
                    "cancel": f"cancel {task}",
                }[kind]
                await self.turn(user, chat, command)
            elif kind == "status":
                await self.h.runs.get(task)
            elif kind == "list":  # what a status view shows: the user's runs, each whole
                found = self.h.runs.iterate(
                    agent_id="research", user_id=user, limit=20, max_pages=1
                )
                await asyncio.gather(*[self.h.runs.get(s.run_id) async for s in found])
            elif kind == "steer":
                from trellis.contracts import RunMessageKind

                await self.h.runs.send_message(
                    task,
                    RunMessageKind.INSTRUCTION,
                    message_id=f"m{time.time_ns()}",
                    text="also this",
                    sender=user,
                )
            else:
                await self.child.cancel(task)
        except Exception as exc:  # counted: a conflict on an ended task is not one
            if "already ended" in str(exc) or "409" in str(exc) or "CONFLICT" in str(exc).upper():
                return
            self.errors[f"{kind}: {type(exc).__name__}"] += 1
            return
        self.latency[kind].append(time.monotonic() - began)
        if kind == "cancel":
            self.tasks.pop(task, None)

    async def direct_operation(self, kind: str, task: str, user: str, chat: str) -> None:
        """What AG-UI's routes do for a user's task: its status and the user's tasks
        (``background.described``, ``users_tasks``), a message to its inbox, its cancel."""
        from trellis.contracts import RunMessageKind
        from trellis.harness.background import described, users_tasks

        if kind == "status":
            record = await self.h.runs.get(task)
            assert record is not None and record.user_id == user
            described(record)
        elif kind == "list":
            [described(r) for r in await users_tasks(self.lead, user)]
        elif kind == "steer":
            await self.h.runs.send_message(
                task,
                RunMessageKind.INSTRUCTION,
                message_id=f"m{time.time_ns()}",
                text="also this",
                sender=user,
            )
        else:
            await self.child.cancel(task)

    async def started(self, tasks: int) -> None:
        """``tasks`` tasks, one per user's chat, 50 starting at once."""
        gate = asyncio.Semaphore(50)

        async def start(n: int) -> None:
            async with gate:
                try:
                    await self.start(n)
                except Exception as exc:
                    self.errors[f"start: {type(exc).__name__}: {exc}"[:120]] += 1

        await asyncio.gather(*(start(n) for n in range(tasks)))

    async def wound_down(self) -> tuple[int, Counter[str]]:
        """How many tasks still ran, and how the others ended; each still going is cancelled."""
        running = 0
        ended: Counter[str] = Counter()
        for task in list(self.tasks):
            record = await self.h.runs.get(task)
            running += record is not None and record.status.value == "RUNNING"
            if record is not None and record.status.final:
                ended[record.status.value] += 1
            if record is not None and not record.status.final:
                try:
                    await self.child.cancel(task)
                except Exception as exc:  # ended meanwhile
                    if "409" not in str(exc) and "already" not in str(exc):
                        self.errors[f"cleanup: {type(exc).__name__}"] += 1
        return running, ended

    async def lane(self, until: float, think: float) -> None:
        while time.monotonic() < until and self.tasks:
            await self.operate(random.choices(OPERATIONS, WEIGHTS)[0])
            await asyncio.sleep(think)


async def drive(args: argparse.Namespace) -> dict[str, Any]:
    if args.pg and args.chats:
        seed_chats(args.pg, args.chats, args.users_total)
    plan = Served(
        args.tasks // max(1, args.workers) + 1,
        args.seconds + args.warmup + 30,
        args.pace,
        args.steps,
    )
    context = multiprocessing.get_context("spawn")
    report: dict[str, Any] = {
        "arm": args.arm,
        "chats": args.chats,
        "tasks": args.tasks,
        "users": args.users,
        "workers": args.workers,
    }
    with ProcessPoolExecutor(max(1, args.workers), mp_context=context) as pool:
        served = [pool.submit(serve, plan) for _ in range(args.workers)]
        async with harness() as h:
            child = research(h, args.pace, args.steps)
            # --workers 0: a worker in this process (the in-process run store)
            here = None if args.workers else h.worker([child], concurrency=plan.concurrency)
            working = None if here is None else asyncio.create_task(here.run())
            users = Users(args.arm, h, child, lead(h, child, args.arm == "A"))
            users.direct = args.ops == "direct"
            await users.started(args.tasks)
            await asyncio.sleep(args.warmup)  # the workers claim them
            mine = args.users // args.drivers
            others = _others(args, users, context, mine)
            if others:  # they start together, once spawned
                await asyncio.sleep(max(0.0, others[0] - time.time()))
            if args.pg:
                statements(args.pg, reset=True)
            cpu0 = {name: procstat.cpu_seconds(pid) for name, pid in args.watched.items()}
            mine0, began = procstat.cpu_seconds(), time.monotonic()
            until = began + args.seconds
            pool_use: list[int] = []

            async def sample() -> None:
                while time.monotonic() < until:
                    if args.pg:
                        pool_use.append(connections(args.pg))
                    await asyncio.sleep(5)

            await asyncio.gather(sample(), *(users.lane(until, args.think) for _ in range(mine)))
            took = time.monotonic() - began
            if others:
                report["drivers_cpu_pct"] = _merged(users, others[1], took)
            if args.pg:
                report.update(_database(args.pg, pool_use, took))
            report.update(_figures(args, users, cpu0, mine0, took))
            running, ended = await users.wound_down()
            report["running_at_end"] = running
            report["ended_before_cleanup"] = dict(ended)
            users.errors.update(
                {f"task {k}": v for k, v in ended.items() if k in ("ERROR", "TIMEOUT")}
            )
            report["errors"] = dict(users.errors)
            if args.pg:
                report["rows"] = rows_written(args.pg)
            if here is not None and working is not None:
                here.stop()
                here.stop()
                await asyncio.gather(working, return_exceptions=True)
        workers = [f.result() for f in served]
    report["workers_measured"] = [
        {
            "cpu_pct": round(100 * w["cpu_seconds"] / plan.seconds, 1),
            "rss_mb_max": round(w["rss_mb_max"], 1),
        }
        for w in workers
    ]
    return report


def _others(
    args: argparse.Namespace, users: Users, context: Any, mine: int
) -> tuple[float, Any] | None:
    """The other ``--drivers`` processes, started: when they begin, and their results."""
    if args.drivers < 2:
        return None
    start_at = time.time() + 10  # time to spawn and import
    per = (args.users - mine) // (args.drivers - 1)
    shares = [
        Share(
            args.arm,
            args.ops == "direct",
            args.pace,
            args.steps,
            dict(users.tasks),
            per,
            args.think,
            start_at,
            start_at + args.seconds,
        )
        for _ in range(args.drivers - 1)
    ]
    pool = ProcessPoolExecutor(len(shares), mp_context=context)
    futures = [pool.submit(operate, share) for share in shares]
    pool.shutdown(wait=False)
    return start_at, futures


def _merged(users: Users, futures: Any, took: float) -> list[float]:
    """The other drivers' latencies and errors added to this one's: each one's CPU, in %."""
    cpu = []
    for future in futures:
        got = future.result()
        for kind, values in got["latency"].items():
            users.latency[kind].extend(values)
        users.errors.update(got["errors"])
        cpu.append(round(100 * got["cpu_seconds"] / took, 1))
    return cpu


def _database(dsn: str, pool_use: list[int], took: float) -> dict[str, Any]:
    """The database's statements a second over the window, by kind, and its connections."""
    kinds = statements(dsn)
    return {
        "db_statements_per_s": {k: round(v / took, 1) for k, v in sorted(kinds.items())},
        "db_statements_per_s_total": round(sum(kinds.values()) / took, 1),
        "db_connections_max": max(pool_use or [0]),
    }


def _figures(
    args: argparse.Namespace,
    users: Users,
    cpu0: dict[str, float | None],
    mine0: float | None,
    took: float,
) -> dict[str, Any]:
    """Each process's CPU and memory over the window, and each operation's latency."""

    def cpu(pid: int | None, before: float | None) -> float:
        return round(100 * ((procstat.cpu_seconds(pid) or 0) - (before or 0)) / took, 1)

    return {
        "driver": {"cpu_pct": cpu(None, mine0), "rss_mb": procstat.rss_mb()},
        "watched": {
            name: {"cpu_pct": cpu(pid, cpu0[name]), "rss_mb": procstat.rss_mb(pid)}
            for name, pid in args.watched.items()
        },
        "latency_ms": {
            kind: {
                "n": len(v),
                "p50": _ms(procstat.percentile(v, 50)),
                "p95": _ms(procstat.percentile(v, 95)),
                "p99": _ms(procstat.percentile(v, 99)),
            }
            for kind, v in sorted(users.latency.items())
        },
    }


def _ms(value: float | None) -> float | None:
    return None if value is None else round(1000 * value, 1)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.load.background", description=__doc__
    )
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--chats", type=int, default=1000, help="idle conversations seeded")
    parser.add_argument("--tasks", type=int, default=100, help="research tasks running at once")
    parser.add_argument("--users", type=int, default=50, help="users operating at once")
    parser.add_argument(
        "--users-total", type=int, default=1000, help="the users owning the idle chats"
    )
    parser.add_argument(
        "--think", type=float, default=1.0, help="seconds between a user's operations"
    )
    parser.add_argument(
        "--ops",
        choices=("chat", "direct"),
        default="chat",
        help="arm A: the users ask through the lead's tools (a turn) or AG-UI's task routes",
    )
    parser.add_argument(
        "--drivers",
        type=int,
        default=1,
        help="processes the users operate from (a harness each)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes (0: one in this process)",
    )
    parser.add_argument(
        "--pace", type=float, default=0.5, help="the research model's seconds per call"
    )
    parser.add_argument("--steps", type=int, default=1000, help="the research task's tool calls")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--warmup", type=float, default=15.0)
    parser.add_argument(
        "--pg",
        default=os.environ.get("BENCH_PG"),
        help="agent-runs' database (psycopg DSN)",
    )
    parser.add_argument("--watch", action="append", default=[], help="name=pid")
    parser.add_argument("--out", default="build/bench-background")
    args = parser.parse_args(argv)
    args.watched = {n: int(p) for n, p in (w.split("=", 1) for w in args.watch)}
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    report = asyncio.run(drive(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(report, indent=1))
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
