"""The Claude Agent SDK bridges: the memory context as a ``UserPromptSubmit`` hook, and the
session kept in the run's journal as the SDK's ``SessionStore`` — so a paused run, or one whose
worker died, continues its session on another machine. The real SDK drives the scripted CLI
(``tests/support/fake_claude_cli.py``)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, InMemorySessionStore

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness.bridges.claude import JournalSessions, _said_local
from trellis.harness.journal import Journal
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
refunds: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return f"refunded {order}"


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    refunds.clear()
    _said_local.cache_clear()


def machine(tmp_path: Path, name: str, script: list[dict[str, Any]], **fields: Any) -> Any:
    """The options as one machine runs them: its own CLI sessions folder, the shared logs."""
    home = tmp_path / name
    home.mkdir(exist_ok=True)
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
            "FAKE_CLAUDE_SESSIONS": str(home),
            "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
        },
        **fields,
    )


def started_with(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "cli.json").read_text())


def built_ins(tmp_path: Path) -> list[dict[str, Any]]:
    log = tmp_path / "builtins.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


SCRIPT = [
    {"builtin": "Write", "args": {"file_path": "a.txt"}},
    {"tool": "refund", "args": {"order": "o1"}},
    {"text": "done: {last}"},
]


async def test_a_paused_session_continues_on_another_machine(tmp_path: Path) -> None:
    runs = LocalRuns()
    first = Harness(config=Settings(), runs=runs)
    second = Harness(config=Settings(), runs=runs)
    try:
        here = arm(first.wrap(machine(tmp_path, "one", SCRIPT), id="refunder"), [refund])
        there = arm(second.wrap(machine(tmp_path, "two", SCRIPT), id="refunder"), [refund])
        paused = await here.run("refund o1", user="ada")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        record = await runs.get(paused.run_id)
        assert record is not None
        journal = Journal.of(record.checkpoint)
        assert journal.session is not None and journal.transcripts[journal.session]
        done = await there.resume(paused.interrupt.interrupt_id, "approve", reviewer="ada")
    finally:
        await first.aclose()
        await second.aclose()
    assert done.status is RunStatus.SUCCESS and done.answer == "done: refunded o1", done.error
    started = started_with(tmp_path)
    assert started["resume"] == journal.session and started["materialised"] is True
    assert built_ins(tmp_path) == [{"tool": "Write", "args": {"file_path": "a.txt"}}]  # once
    assert refunds == ["o1"]


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def test_a_worker_that_died_leaves_no_side_effect_done_twice(tmp_path: Path) -> None:
    crashes = [Crash()]

    @tool(side_effects="read")
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    script = [
        {"builtin": "Write", "args": {"file_path": "a.txt"}},
        {"tool": "boom", "args": {}},
        {"text": "done: {last}"},
    ]
    runs = LocalRuns()
    first = Harness(config=Settings(), runs=runs)
    second = Harness(config=Settings(), runs=runs)
    try:
        dying = arm(first.wrap(machine(tmp_path, "one", script), id="r"), [boom])
        taking_over = arm(second.wrap(machine(tmp_path, "two", script), id="r"), [boom])
        handle = await dying.start("work", user="ada")
        worker = first.worker([dying])
        claimed = await runs.claim(worker.worker_id, [dying.id])
        assert claimed is not None
        job = Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=runs)
        # the worker dies inside the SDK's tool call: its task group carries the crash out
        with pytest.RaisesGroup(Crash, flatten_subgroups=True):
            await dying.execute(job)
        held, _ = runs._leases[handle.id]
        runs._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
        assert await second.worker([taking_over]).run_once()
        result = await handle.result(wait=10)
    finally:
        await first.aclose()
        await second.aclose()
    assert result.status is RunStatus.SUCCESS and result.answer == "done: fine", result.error
    assert built_ins(tmp_path) == [{"tool": "Write", "args": {"file_path": "a.txt"}}]


class DurableStore:
    """A session store standing for a durable one (a database's): not one kept in RAM."""

    def __init__(self) -> None:
        self.kept = InMemorySessionStore()

    async def append(self, key: Any, entries: list[Any]) -> None:
        await self.kept.append(key, entries)

    async def load(self, key: Any) -> list[Any] | None:
        return await self.kept.load(key)


def said(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "Claude's session is kept" in r.getMessage()]


async def test_options_with_their_own_store_keep_it(
    harness: Harness, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A store of the team's own is kept; one kept in RAM is said once, with the fix."""
    in_ram, durable = InMemorySessionStore(), DurableStore()
    ram = harness.wrap(machine(tmp_path, "one", [{"text": "ok"}], session_store=in_ram), id="ram")
    kept = harness.wrap(machine(tmp_path, "two", [{"text": "ok"}], session_store=durable), id="db")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for agent in (ram, ram, kept):
            assert (await agent.run("hi", user="ada")).status is RunStatus.SUCCESS
    [warning] = said(caplog)
    assert warning.__dict__["agent_id"] == "ram" and warning.__dict__["store"] == (
        "InMemorySessionStore"
    )
    assert "runs the query again from its prompt" in warning.getMessage()
    assert in_ram._store and durable.kept._store  # each team's store got its session


async def test_file_checkpointing_keeps_the_session_on_the_local_disk_and_says_so(
    harness: Harness, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    options = machine(tmp_path, "one", [{"text": "ok"}], enable_file_checkpointing=True)
    agent = harness.wrap(options, id="files")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for _ in range(2):
            assert (await agent.run("hi", user="ada")).status is RunStatus.SUCCESS
    [warning] = said(caplog)
    assert warning.__dict__["store"] == "enable_file_checkpointing"


async def test_a_memory_service_that_is_down_leaves_the_prompt_without_context(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    memory_service.fail.add("context")
    agent = memory_harness.wrap(machine(tmp_path, "one", [{"text": "[{context}]"}]), id="down")
    events = [e async for e in agent.stream("hi", user="ada")]
    assert events[-1].data["result"] == "[]"
    assert [e.data["code"] for e in events if e.data.get("name") == "warning"] == [
        "memory_unavailable"
    ]


# ---------------------------------------------------------------------- the store itself
async def test_the_journal_keeps_each_entry_once_by_its_uuid() -> None:
    journal = Journal()
    store = JournalSessions(journal)
    key = {"project_key": "here", "session_id": "s1"}
    assert await store.load(key) is None
    await store.append(key, [{"type": "user", "uuid": "u1"}, {"type": "title"}])
    await store.append(key, [{"type": "user", "uuid": "u1"}, {"type": "title"}])  # a retry
    await JournalSessions(journal).append(key, [{"type": "a", "uuid": "u1"}])  # another process
    elsewhere = {"project_key": "there", "session_id": "s1"}  # another machine's directory
    assert await store.load(elsewhere) == [
        {"type": "user", "uuid": "u1"},
        {"type": "title"},
        {"type": "title"},
    ]
    sub = {"project_key": "here", "session_id": "s1", "subpath": "subagents/agent-1"}
    await store.append(sub, [{"type": "user", "uuid": "u2"}])
    assert list(journal.transcripts) == ["s1", "s1/subagents/agent-1"]
