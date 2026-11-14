"""A conversation's Claude Code process is kept between its turns, so what Claude starts in the
background goes on while the next turns are answered (``bridges.claude_live``).

The CLI here is the scripted stand-in (``tests/support/fake_claude_cli.py``), driven by the
real SDK: process, control protocol, hooks and in-process MCP calls are the SDK's own, and the
frames the stand-in sends for background work are parsed by the SDK's own message parser. Its
rules for background work — a turn's result while a task runs, the session's own turn for a
finished task (no prompt frame, a result of ``task-notification`` origin), an interrupt that
stops the tasks while a hook's ``continue: false`` does not, a result's running cost and own
usage — are the bundled CLI's (2.1.283), as seen driving it against a scripted model on this
machine. No real model was called: what a real model does with them is a live run's to show.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    create_sdk_mcp_server,
)
from claude_agent_sdk import tool as sdk_tool
from claude_agent_sdk._internal.message_parser import parse_message

from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.memory import FakeMemoryService
from trellis import Budget, Deny, Harness, Hooks, Settings, current, tool
from trellis.contracts import (
    ConfigurationError,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStatus,
    ToolCall,
)
from trellis.harness import budget, holding
from trellis.harness.adapters.claude import (
    AT_LIMIT,
    BUILTIN_SIDE_EFFECTS,
    OUTLIVES_THE_QUERY,
    RESUMED_CALL,
    ClaudeAdapter,
)
from trellis.harness.agent import Agent
from trellis.harness.bridges import claude_live
from trellis.harness.bridges.claude_live import UNATTENDED, LiveSession
from trellis.harness.bridges.helpers import HELPER
from trellis.harness.events import RunEvents
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.identity import conversation_key
from trellis.harness.runs import LocalRuns
from trellis.harness.subagents import SUBAGENT

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
#: A subagent started in the background: its ``Agent`` call returns at once, task ``t1`` runs.
BACKGROUND = {"background": {"id": "t1", "agent": "researcher", "prompt": "research the market"}}
#: A shell command started in the background (``Bash`` with ``run_in_background``): task ``s1``.
SHELL = {"background": {"id": "s1", "shell": "make watch"}}
NOTIFIED = "<task-notification>the market is growing</task-notification>"
refunds: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return f"refunded {order}"


def options(tmp_path: Path, turns: list[list[dict[str, Any]]], **fields: Any) -> ClaudeAgentOptions:
    """Options whose "model" answers each turn of a session with the script of its number."""
    (tmp_path / "tasks").mkdir(exist_ok=True)
    env = {
        "FAKE_CLAUDE_TURNS": json.dumps(turns),
        "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
        "FAKE_CLAUDE_SESSIONS": str(tmp_path),
        "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
        "FAKE_CLAUDE_TASKS": str(tmp_path / "tasks"),
        **fields.pop("env", {}),
    }
    return ClaudeAgentOptions(cli_path=CLI, env=env, **fields)


def served(tmp_path: Path) -> list[dict[str, Any]]:
    """What the CLI recorded of each turn it was sent, in order: which process took it."""
    log = tmp_path / "cli.json.turns"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def built_ins(tmp_path: Path) -> list[dict[str, Any]]:
    log = tmp_path / "builtins.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def until(check: Callable[[], Any], seconds: float = 5.0) -> None:
    async with asyncio.timeout(seconds):
        while not check():  # noqa: ASYNC110 - what another process or the reader's task did
            await asyncio.sleep(0.02)


def live_of(agent: Agent, user: str = "u1", thread: str = "t1") -> LiveSession | None:
    adapter = agent.adapter
    assert isinstance(adapter, ClaudeAdapter)
    return adapter.live.get(conversation_key("default", user, thread))


def open_sessions(agent: Agent) -> int:
    adapter = agent.adapter
    assert isinstance(adapter, ClaudeAdapter)
    return len(adapter.live)


def warned(events: list[Any]) -> list[str]:
    return [e.data["message"] for e in events if e.data.get("code") == "claude_session"]


def turn_running(agent: Agent) -> str:
    """The one turn of a conversation under way (a background subagent's run, which is under
    way too, has a parent)."""
    [run_id] = [r for r, at in agent.running.items() if at.identity.parent_run is None]
    return run_id


async def child_of(harness: Harness, run_id: str, tenant: str | None = "default") -> RunRecord:
    """The one child run of ``run_id``."""
    [child] = [r async for r in harness.runs.iterate(parent_run_id=run_id, tenant=tenant)]
    record = await harness.runs.get(child.run_id, tenant=tenant)
    assert record is not None
    return record


async def when(harness: Harness, run_id: str, status: RunStatus, seconds: float = 5.0) -> RunRecord:
    """The run's record once it is ``status``."""
    async with asyncio.timeout(seconds):
        while True:
            record = await harness.runs.get(run_id, tenant="default")
            if record is not None and record.status is status:
                return record
            await asyncio.sleep(0.02)


async def calls(tmp_path: Path, step: dict[str, Any], task: str = "t1") -> dict[str, Any]:
    """Task ``task`` makes the call ``step`` now: what it was answered."""
    asked, answered = tmp_path / "tasks" / f"{task}.call", tmp_path / "tasks" / f"{task}.called"
    asked.write_text(json.dumps(step))
    await until(answered.exists)
    said: dict[str, Any] = json.loads(answered.read_text())
    answered.unlink()
    return said


async def followed(agent: Agent, run_id: str) -> tuple[list[RunEvent], asyncio.Task[None]]:
    """The run's events from now on, as they come."""
    seen: list[RunEvent] = []

    async def follow() -> None:
        async for event in agent.events(run_id):
            seen.append(event)

    following = asyncio.create_task(follow())
    await until(lambda: agent.watchers(run_id))
    return seen, following


def results(events: list[RunEvent]) -> list[str]:
    return [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT]


READS: dict[str, Any] = {"builtin": "Read", "args": {"file_path": "notes.txt"}}
WRITES: dict[str, Any] = {"builtin": "Write", "args": {"file_path": "notes.txt", "content": "x"}}


# --------------------------------------------------------------------------- one chat
async def test_a_background_task_goes_on_while_the_next_turns_are_answered_by_its_process(
    harness: Harness, tmp_path: Path
) -> None:
    """Turn 1 starts a slow subagent in the background and ends at once; turn 2 is answered
    while it runs, by the same process; after it finished — its notification and the turn the
    session ran for it came while nobody was asking — turn 3's answer is its own, and its
    "model" has the notification."""
    scripts = [
        [BACKGROUND, {"text": "started it"}],
        [{"text": "4"}],
        [{"text": "you asked: {heard}"}],
    ]
    agent = harness.wrap(options(tmp_path, scripts), id="chat")
    first = await agent.run("research the market", user="u1", thread="t1")
    assert first.status is RunStatus.SUCCESS and first.answer == "started it"
    live = live_of(agent)
    assert live is not None and live.working == {"t1": "research the market"}  # still running

    second = await agent.run("what is 2+2?", user="u1", thread="t1")
    assert second.status is RunStatus.SUCCESS and second.answer == "4"
    one, two = served(tmp_path)
    # the same process took the second prompt: none was started to resume the session
    assert two["pid"] == one["pid"] and two["resume"] is None and alive(one["pid"])
    assert live_of(agent) is live and open_sessions(agent) == 1 and list(live.working) == ["t1"]

    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await until(lambda: any(isinstance(m, ResultMessage) for m in live.kept))
    assert not live.working
    # what came with no turn to take it is kept: the task's notification, the session's own
    # turn for it and that turn's result — none of it anybody's answer
    [notification] = [m for m in live.kept if isinstance(m, TaskNotificationMessage)]
    assert notification.task_id == "t1" and notification.summary == "the market is growing"
    [injected] = [m for m in live.kept if isinstance(m, ResultMessage)]
    assert injected.origin == {"kind": "task-notification"}
    assert injected.result == "Noted: the market is growing"

    third = await agent.run("what did it find?", user="u1", thread="t1")
    asked = ["research the market", "what is 2+2?", NOTIFIED, "what did it find?"]
    assert third.status is RunStatus.SUCCESS
    assert third.answer == f"you asked: {' | '.join(asked)}"  # its own answer; it knows the task
    assert [turn["pid"] for turn in served(tmp_path)] == [one["pid"]] * 3


async def test_the_result_of_a_turn_the_session_injected_is_never_a_users_answer(
    harness: Harness, tmp_path: Path
) -> None:
    """A message that comes while the session runs its own turn (a finished task's
    notification) is sent once that turn ended — sent into it, the CLI could hand it to that
    turn, which has no result for it: that turn's text and result are not the user's, on the
    stream or in the answer."""
    slow = json.dumps([{"text": "Noted: {summary}"}, {"stall": 0.4}, {"text": "still noting"}])
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "my own answer"}]]
    target = options(tmp_path, scripts, env={"FAKE_CLAUDE_INJECTED": slow})
    agent = harness.wrap(target, id="injected")
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await until(lambda: live.injected)  # the session's own turn is under way
    events = [e async for e in agent.stream("and you?", user="u1", thread="t1")]
    assert events[-1].data["result"] == "my own answer"
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(said) == "my own answer"
    assert [m.result for m in live.kept if isinstance(m, ResultMessage)] == ["still noting"]


async def test_a_prompt_that_crosses_the_start_of_the_sessions_own_turn_still_gets_its_answer(
    harness: Harness, tmp_path: Path
) -> None:
    """The session's own turn has begun in the CLI but sent nothing yet when the user's
    prompt goes out: the CLI answers the prompt after that turn. The turn's result — not the
    user's, by its origin — is passed over, and the answer is the user's own. (What that turn
    said before its result cannot be told from the user's turn until then: it is on the
    stream. The window is the time the CLI takes to send that turn's first message.)"""
    slow = json.dumps([{"stall": 0.4}, {"text": "Noted: {summary}"}])
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "my own answer"}]]
    target = options(tmp_path, scripts, env={"FAKE_CLAUDE_INJECTED": slow})
    agent = harness.wrap(target, id="crossed")
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await until(lambda: not live.working)  # its notification came: the session's turn begins
    assert not live.injected
    result = await agent.run("and you?", user="u1", thread="t1")
    assert result.status is RunStatus.SUCCESS and result.answer == "my own answer"
    [injected] = [m for m in live.kept if isinstance(m, ResultMessage)]
    assert injected.result == "Noted: the market is growing"
    assert injected.origin == {"kind": "task-notification"}


def test_an_injected_turns_result_in_a_querys_messages_is_not_its_answer() -> None:
    """In a query's own process too, a background task that ends before the process does
    brings a turn of the session's own after the user's: the answer is the user's turn's."""

    def result(text: str, **more: Any) -> Any:
        frame = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 1,
            "session_id": "s1",
            "result": text,
            **more,
        }
        return parse_message(frame)

    messages = [
        result("mine"),
        result("Noted: done", origin={"kind": "task-notification"}),
    ]
    assert ClaudeAdapter().extract(None, messages).answer == "mine"
    stamped = [result("stamped", origin={"kind": "human"})]
    assert ClaudeAdapter().extract(None, stamped).answer == "stamped"


async def test_a_turn_a_live_session_serves_has_its_own_run_hooks_spending_and_governance(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process was started in the first turn; the second turn's tool calls are decided,
    run and recorded for the second turn's run, and it is charged what it spent itself."""
    counted: list[dict[str, Any]] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (counted.append(kw), account(run, **kw))
    )

    @tool(side_effects="read")
    def whoami() -> str:
        """The run this call is made in."""
        runtime = current()
        assert runtime is not None
        return runtime.run_id

    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | None:
            return Deny("no searching here") if call.tool == "Grep" else None

    scripts = [
        [{"text": "hello"}],
        [
            {"builtin": "Grep", "args": {"pattern": "x"}},
            {"text": "grep said: {last}"},
            {"builtin": "Read", "args": {"file_path": "notes.txt"}},
            {"tool": "whoami"},
            {"text": "{last}"},
        ],
    ]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = arm(harness.wrap(target, id="governed", hooks=[Guard()]), [whoami])
    first = await agent.run("hi", user="u1", thread="t1")
    events = [e async for e in agent.stream("who am I?", user="u1", thread="t1")]
    run_id = events[-1].run_id
    assert run_id != first.run_id and events[-1].data["result"] == run_id  # current() is its run
    assert len({turn["pid"] for turn in served(tmp_path)}) == 1
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert said[0] == "grep said: Grep was not run: no searching here"  # the run's hooks
    assert built_ins(tmp_path) == [{"tool": "Read", "args": {"file_path": "notes.txt"}}]
    results = [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert results == ["Read", "whoami"]  # recorded on the second turn's stream
    # one step, then five: each turn is charged its own, not the process's running cost
    assert [(c["input_tokens"], c["output_tokens"], c["cost"]) for c in counted] == [
        (10, 5, 0.5),
        (50, 25, 2.5),
    ]


async def test_what_a_background_run_spent_between_turns_is_on_it_and_not_on_the_next_turn(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result's cost is the process's running total: what it grew by between two turns —
    the subagent's own replies, the turn the session ran for its end — is charged to the
    subagent's run, with its replies' tokens, and the next turn is charged its own."""
    counted: list[tuple[str, int, float, int]] = []
    account = budget.account

    def count(run: Any, **kw: Any) -> None:
        tokens = kw["input_tokens"] + kw["output_tokens"]
        counted.append((run.run_id, tokens, kw["cost"], kw.get("calls", 1)))
        account(run, **kw)

    monkeypatch.setattr(budget, "account", count)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = harness.wrap(target, id="spent")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    seen, following = await followed(agent, child.run_id)
    assert await calls(tmp_path, READS) == {"result": "Read ran", "ok": True}
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await when(harness, child.run_id, RunStatus.SUCCESS)
    second = await agent.run("thanks", user="u1", thread="t1")
    await following
    assert counted == [
        (first.run_id, 30, 1.0, 1),  # two steps
        (child.run_id, 15, 0.0, 1),  # the subagent's reply that made its call
        (child.run_id, 15, 0.0, 1),  # its last reply
        # what both cost, known at the session's next result and shared by their tokens;
        # and the turn the session ran for the task's end
        (child.run_id, 0, 1.0, 0),
        (child.run_id, 0, 0.5, 0),
        (second.run_id, 15, 0.5, 1),  # one step of its own
    ]
    assert seen[-1].data["spent"] == {
        "tokens": 30,
        "cached_tokens": 0,
        "cost": 1.5,
        "model_calls": 2,
        "tool_calls": 0,
    }


async def test_the_options_own_budget_stays_a_querys_each_turn_in_a_process_of_its_own(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ClaudeAgentOptions(max_budget_usd=)`` is the CLI's cap on what one process spends
    (the bundled CLI, driven against a scripted model: in a process that serves several
    turns the turn that passes the cap ends ``error_max_budget_usd``, and so does every
    turn after it; a process that resumes a session starts counting anew). The SDK has no
    way to give a running process its cap back, so options with one are not kept between
    turns: each turn runs in a process of its own that resumes the session, as before — the
    cap a team set for a query stays a query's, and the conversation goes on. Each turn is
    charged its own cost, though a resumed session's result carries the session's."""
    costs: list[float] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (costs.append(kw["cost"]), account(run, **kw))
    )
    scripts = [[{"text": "you asked: {heard}"}]]
    target = options(
        tmp_path, scripts, model="m", max_budget_usd=1.2, env={"FAKE_CLAUDE_COST": "0.5"}
    )
    agent = harness.wrap(target, id="capped")
    for number in (1, 2, 3, 4):  # a step each, at 0.5: 2.0 in all, none above 1.2 by itself
        done = await agent.run(f"turn {number}", user="u1", thread="t1")
        assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer == "you asked: turn 1 | turn 2 | turn 3 | turn 4"
    one, *later = served(tmp_path)
    assert len({turn["pid"] for turn in (one, *later)}) == 4 and open_sessions(agent) == 0
    assert [turn["resume"] for turn in later] == [one["session"]] * 3
    assert costs == [0.5, 0.5, 0.5, 0.5]


async def test_a_live_session_that_resumes_a_session_charges_its_first_turn_its_own_cost(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that resumes a session reports the session's cost, the turns before it
    included: the turn it serves first is charged what the cost grew by since the
    conversation's last turn, in the process before."""
    costs: list[float] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (costs.append(kw["cost"]), account(run, **kw))
    )
    scripts = [[{"text": "ok"}, {"text": "ok"}], [{"text": "ok"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = harness.wrap(target, id="resumed")
    await agent.run("one", user="u1", thread="t1")
    adapter = agent.adapter
    assert isinstance(adapter, ClaudeAdapter)
    await adapter.live.close(conversation_key("default", "u1", "t1"), "a test closed it")
    await agent.run("two", user="u1", thread="t1")
    await agent.run("three", user="u1", thread="t1")
    one, two, three = served(tmp_path)
    assert two["pid"] != one["pid"] and two["resume"] == one["session"]
    assert three["pid"] == two["pid"]
    assert costs == [1.0, 0.5, 0.5]


async def _charged_for_a_paused_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, elsewhere: bool
) -> list[float]:
    """A conversation's first turn, then a second that pauses for an approval before any
    result, resumed on the same worker or (``elsewhere``) another sharing the run store:
    what the second turn's run is charged."""
    costs: list[tuple[str, float | None]] = []
    account = budget.account
    monkeypatch.setattr(
        budget,
        "account",
        lambda run, **kw: (costs.append((run.run_id, kw["cost"])), account(run, **kw)),
    )
    refunds.clear()
    scripts = [[{"text": "hi"}], [{"tool": "refund", "args": {"order": "o1"}}, {"text": "done"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    store = LocalRuns()
    one, two = Harness(config=Settings(), runs=store), Harness(config=Settings(), runs=store)
    try:
        first = arm(one.wrap(target, id="chat"), [refund])
        second = arm(two.wrap(target, id="chat"), [refund])
        await first.run("hi", user="u1", thread="t1")
        paused = await first.run("refund o1", user="u1", thread="t1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        resumer = second if elsewhere else first
        done = await resumer.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        assert refunds == ["o1"]
        return [cost or 0.0 for run_id, cost in costs if run_id == paused.run_id]
    finally:
        await one.aclose()
        await two.aclose()


async def test_a_turn_resumed_on_another_worker_is_charged_its_own_cost_not_the_earlier_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed session's result carries the session's running cost, the conversation's
    earlier turns included. A turn that paused before its first result is charged what the
    cost grew by since then wherever it is resumed: the cost already charged for the
    session is in its journal from the turn's start, not only in the memory of the worker
    that served the turn before."""
    (tmp_path / "same").mkdir()
    same = await _charged_for_a_paused_turn(tmp_path / "same", monkeypatch, elsewhere=False)
    (tmp_path / "elsewhere").mkdir()
    elsewhere = await _charged_for_a_paused_turn(
        tmp_path / "elsewhere", monkeypatch, elsewhere=True
    )
    assert sum(elsewhere) == pytest.approx(sum(same)) == pytest.approx(1.5), (same, elsewhere)


async def _spent_by_a_paused_run(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, thread: str | None
) -> tuple[int, int, float]:
    """A run that pauses for an approval after a step, then is approved: the tokens, model
    calls and cost it is charged."""
    counted: list[tuple[str, int, int, float]] = []
    account = budget.account

    def count(run: Any, **kw: Any) -> None:
        tokens = kw["input_tokens"] + kw["output_tokens"]
        counted.append((run.run_id, tokens, kw.get("calls", 1), kw["cost"] or 0.0))
        account(run, **kw)

    monkeypatch.setattr(budget, "account", count)
    refunds.clear()
    scripts = [[{"tool": "refund", "args": {"order": "o1"}}, {"text": "done"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = arm(harness.wrap(target, id=f"spent-{thread}"), [refund])
    paused = await agent.run("refund o1", user="u1", thread=thread)
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    mine = [(t, c, cost) for run_id, t, c, cost in counted if run_id == paused.run_id]
    return sum(t for t, _, _ in mine), sum(c for _, c, _ in mine), sum(c for _, _, c in mine)


async def test_a_live_turn_that_paused_is_charged_the_tokens_and_calls_of_its_paused_attempt(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attempt that paused made a model call, whose tokens are the run's: in a live
    session as in a process of its own (a run with no conversation) — though the result of
    the CLI's turn it left comes after the attempt stopped reading."""
    (tmp_path / "alone").mkdir()
    (tmp_path / "live").mkdir()
    alone = await _spent_by_a_paused_run(harness, tmp_path / "alone", monkeypatch, thread=None)
    live = await _spent_by_a_paused_run(harness, tmp_path / "live", monkeypatch, thread="t1")
    assert live[:2] == alone[:2] and live[2] == pytest.approx(alone[2]), (alone, live)


async def test_a_cancelled_live_turn_is_charged_its_own_spending_and_the_next_turn_its_own(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live turn cancelled mid-turn is interrupted: the result of the CLI's turn comes after
    the run stopped reading, and what it spent until then is the cancelled run's — not the
    conversation's next turn's, the next result the session charges."""
    costs: list[tuple[str, float | None]] = []
    account = budget.account
    monkeypatch.setattr(
        budget,
        "account",
        lambda run, **kw: (costs.append((run.run_id, kw["cost"])), account(run, **kw)),
    )
    reads = [{"builtin": "Read", "args": {"file_path": name}} for name in ("x", "y")]
    scripts = [[{"text": "a"}], [*reads, {"stall": 30}, {"text": "never"}], [{"text": "z"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = harness.wrap(target, id="cancelled")
    await agent.run("one", user="u1", thread="t1")
    slow = asyncio.create_task(agent.run("two", user="u1", thread="t1"))
    await until(lambda: len(built_ins(tmp_path)) == 2)  # its two reads ran
    await asyncio.sleep(0.5)  # and its third step began: it stalls now
    await agent.cancel(turn_running(agent))
    cancelled = await slow
    assert cancelled.status is RunStatus.CANCELLED
    third = await agent.run("three", user="u1", thread="t1")
    assert third.answer == "z"
    _, two, three = served(tmp_path)
    assert three["pid"] == two["pid"]  # the same live process served it
    charged = {run_id: sum(c or 0.0 for r, c in costs if r == run_id) for run_id, _ in costs}
    assert charged[third.run_id] == pytest.approx(0.5), costs  # its one step
    assert charged[cancelled.run_id] == pytest.approx(1.5), costs  # its three steps


# --------------------------------------------------------------------------- no live session
async def test_a_run_with_no_conversation_runs_in_a_process_of_its_own_as_before(
    harness: Harness, tmp_path: Path
) -> None:
    agent = harness.wrap(options(tmp_path, [[{"text": "ok"}]]), id="alone")
    result = await agent.run("hi", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "ok"
    [only] = served(tmp_path)
    assert open_sessions(agent) == 0
    await until(lambda: not alive(only["pid"]))  # its process ended with the query


async def launched(agent: Agent, events: list[RunEvent]) -> str:
    """The run whose ``Agent`` call has returned (a background subagent's launch)."""
    await until(lambda: "Agent" in results(events))
    return events[0].run_id


async def test_a_background_subagent_of_a_run_with_no_conversation_keeps_its_child_run(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run with no conversation has a process of its own, which the SDK keeps while a
    background subagent works in it (its ``query`` goes on past the turn's result until the
    task ended and the session ran its turn for it). The subagent's child run is open that
    long, not ended at its launch: its calls are recorded under it, it ends with what its
    task said, and the run says so (``helper_finished``). The run's answer is its own turn's;
    the process's running cost is charged once, not again with every result."""
    costs: list[float] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (costs.append(kw["cost"] or 0.0), account(run, **kw))
    )
    scripts = [[BACKGROUND, {"text": "started it"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = memory_harness.wrap(target, id="alone")
    events: list[RunEvent] = []

    async def run() -> None:
        events.extend([e async for e in agent.stream("research the market", user="u1")])

    seen, following = await followed_first(agent, events)
    running = asyncio.create_task(run())
    run_id = await launched(agent, seen)
    child = await child_of(memory_harness, run_id, None)
    assert child.status is RunStatus.RUNNING and open_sessions(agent) == 0
    assert await calls(tmp_path, READS) == {"result": "Read ran", "ok": True}
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await running
    await following
    assert events[-1].data["result"] == "started it"
    child = await child_of(memory_harness, run_id, None)
    assert (child.status, child.output) == (RunStatus.SUCCESS, "the market is growing")
    [ended] = [e.data for e in events if e.data.get("name") == "helper_finished"]
    assert (ended["run_id"], ended["status"]) == (child.run_id, "success")
    await memory_harness.writes.drain()
    made = {c.body["tool"]: c.scope["agent_run_id"] for c in memory_service.named("record_tool")}
    assert made == {"Agent": run_id, "Read": child.run_id}
    # five steps at 0.5 in all (two of the turn, the subagent's two, the session's own turn)
    assert sum(costs) == 2.5


async def followed_first(
    agent: Agent, events: list[RunEvent]
) -> tuple[list[RunEvent], asyncio.Task[None]]:
    """The events of the run about to start, as they come (``events`` gets them at its end)."""
    seen: list[RunEvent] = []

    async def follow() -> None:
        await until(lambda: agent.running)
        async for event in agent.events(next(iter(agent.running))):
            seen.append(event)

    return seen, asyncio.create_task(follow())


@pytest.mark.parametrize(
    ("ending", "why"),
    [
        ("failed", "no market data"),
        ("stop", "its work in the background was stopped"),
        ("quit", "its work in the background was cut off"),
    ],
)
async def test_a_background_subagent_of_a_run_with_no_conversation_that_did_not_finish_is_an_error(
    harness: Harness, tmp_path: Path, ending: str, why: str
) -> None:
    """Its task failed, was stopped in Claude Code, or was cut off with the process (which
    ended before the task did): the child run is an error saying which — never a success
    for work that was not done — while the run's own answer stands."""
    last = {"text": "started it", "quit": True} if ending == "quit" else {"text": "started it"}
    agent = harness.wrap(options(tmp_path, [[BACKGROUND, last]]), id="alone")
    events: list[RunEvent] = []

    async def run() -> None:
        events.extend([e async for e in agent.stream("research the market", user="u1")])

    seen, following = await followed_first(agent, events)
    running = asyncio.create_task(run())
    run_id = await launched(agent, seen)
    if ending != "quit":
        (tmp_path / "tasks" / f"t1.{ending}").write_text("no market data")
    await running
    await following
    assert events[-1].data["result"] == "started it"
    child = await child_of(harness, run_id)
    assert child.status is RunStatus.ERROR and child.error is not None
    assert why in child.error.message
    [ended] = [e.data for e in events if e.data.get("name") == "helper_finished"]
    assert (ended["run_id"], ended["status"]) == (child.run_id, "error")


async def test_today_what_claude_says_once_a_querys_background_task_ended_is_not_its_answer(
    harness: Harness, tmp_path: Path
) -> None:
    """Today: a run with no conversation is held until its background subagent's task ended
    and the session ran its turn for it (the SDK's ``query`` goes on that long). What Claude
    says in that turn — usually what the subagent found — is on the run's text stream, and
    the run's answer is still its own turn's result, said before the task ended. What a user
    needs: the answer of a run that waited for the task to hold what came of it."""
    scripts = [[BACKGROUND, {"text": "started it"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="alone")
    events: list[RunEvent] = []

    async def run() -> None:
        events.extend([e async for e in agent.stream("research the market", user="u1")])

    seen, following = await followed_first(agent, events)
    running = asyncio.create_task(run())
    await launched(agent, seen)
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await running
    await following
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert said == ["started it", "Noted: the market is growing"]
    assert events[-1].data["result"] == "started it"


async def test_a_background_subagents_run_has_its_turns_deadline_for_the_run_store_to_end_it(
    harness: Harness, tmp_path: Path
) -> None:
    """A subagent's child run is recorded with the deadline of the run that started it, as
    a sub-agent's run is: agent-runs ends a run that is not done by its deadline, running or
    not (its ticker's ``time_out_past_deadline``), so a background run a killed server left
    ``RUNNING`` is ended there, saying why, by then."""
    from datetime import UTC, datetime, timedelta

    by = datetime.now(UTC) + timedelta(hours=2)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="due")
    first = await agent.run("research the market", user="u1", thread="t1", deadline=by)
    child = await child_of(harness, first.run_id)
    assert child.status is RunStatus.RUNNING and child.deadline == by


async def test_a_background_subagents_run_is_held_in_the_run_store_while_its_session_lives(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session holds its background subagent's run in the run store (``holding``: a
    heartbeat every third of ``HOLD_SECONDS``, naming this harness) for as long as the run goes
    on, past the turn that started it — whose own run was held while its attempt ran, and is no
    more — so a server that dies without closing the session stops holding it
    (``test_killed_server``). The holds end with the run."""
    monkeypatch.setattr(holding, "HOLD_SECONDS", 1)
    beats: list[tuple[str, str, int]] = []
    beating = harness.runs.heartbeat

    async def heartbeat(run_id: str, worker_id: str, **given: Any) -> Any:
        beats.append((run_id, worker_id, given["lease_seconds"]))
        return await beating(run_id, worker_id, **given)

    monkeypatch.setattr(harness.runs, "heartbeat", heartbeat)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="held")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    assert (first.run_id, harness.holds.name, 1) in beats  # while its attempt ran
    beats.clear()
    await until(lambda: len(beats) >= 3)
    assert set(beats) == {(child.run_id, harness.holds.name, 1)}
    assert harness.holds.holds(child.run_id) and not harness.holds.holds(first.run_id)

    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await when(harness, child.run_id, RunStatus.SUCCESS)
    await until(lambda: not harness.holds.holds(child.run_id))
    held = len(beats)
    await asyncio.sleep(0.5)  # past a heartbeat's rhythm: no more
    assert len(beats) == held


async def test_with_no_live_session_the_next_turn_resumes_the_session_in_a_new_process(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session nothing works in closes when it has been idle (as a restart or another
    server leaves none): the next turn starts a process that resumes the conversation's
    session, as before — with what the closed one mirrored between the turns."""
    monkeypatch.setattr(claude_live, "IDLE_SECONDS", 0.2)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "you asked: {heard}"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="idle")
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    await asyncio.sleep(0.5)
    assert live_of(agent) is live and not live.closed  # work under way: not idle
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await until(lambda: open_sessions(agent) == 0)
    [one] = served(tmp_path)
    await until(lambda: not alive(one["pid"]))
    assert live.closed

    second = await agent.run("what came of it?", user="u1", thread="t1")
    one, two = served(tmp_path)
    assert two["pid"] != one["pid"] and two["resume"] == one["session"]
    asked = ["research the market", NOTIFIED, "what came of it?"]
    assert second.answer == f"you asked: {' | '.join(asked)}"


async def test_a_session_is_not_idle_while_its_own_turn_for_a_finished_task_is_due(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished background task's notification is followed by a turn Claude Code runs by
    itself, as soon as it gets to it (here half a second later, longer than the idle bound):
    the session is not idle meanwhile, nor while that turn runs — it closes once the turn
    ended, and the next turn's session holds what it said."""
    monkeypatch.setattr(claude_live, "IDLE_SECONDS", 0.2)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "you asked: {heard}"}]]
    target = options(tmp_path, scripts, env={"FAKE_CLAUDE_INJECTS_AFTER": "0.5"})
    agent = harness.wrap(target, id="due")
    await agent.run("research the market", user="u1", thread="t1")
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await until(lambda: open_sessions(agent) == 0)
    second = await agent.run("what came of it?", user="u1", thread="t1")
    asked = ["research the market", NOTIFIED, "what came of it?"]
    assert second.answer == f"you asked: {' | '.join(asked)}"


async def test_there_is_room_for_so_many_sessions_and_a_turn_beyond_runs_as_before(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background shell command, too, goes on in a live session. With every session at work
    and no room for another, a conversation's turn runs in a process of its own, where the
    command ends with the query — as it always did."""
    monkeypatch.setattr(claude_live, "KEPT", 1)
    scripts = [[SHELL, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts, allowed_tools=["Bash"]), id="room")
    await agent.run("watch the build", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None and live.working == {"s1": "make watch"}
    other = await agent.run("watch the build", user="u2", thread="t1")
    assert other.status is RunStatus.SUCCESS and other.answer == "started it"
    # the one session has work under way: it is not closed for another conversation's turn
    assert open_sessions(agent) == 1 and live_of(agent) is live and live_of(agent, "u2") is None
    one, two = served(tmp_path)
    assert alive(one["pid"])
    await until(lambda: not alive(two["pid"]))
    second = await agent.run("still building?", user="u1", thread="t1")
    assert second.answer == "ok" and served(tmp_path)[-1]["pid"] == one["pid"]
    assert list(live.working) == ["s1"]


async def test_an_idle_session_gives_its_place_to_another_conversation(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_live, "KEPT", 1)
    agent = harness.wrap(options(tmp_path, [[{"text": "ok"}]]), id="place")
    await agent.run("hi", user="u1", thread="t1")
    await agent.run("hi", user="u2", thread="t1")
    one, two = served(tmp_path)
    assert open_sessions(agent) == 1 and live_of(agent, "u2") is not None
    await until(lambda: not alive(one["pid"]))
    assert alive(two["pid"])


async def test_options_whose_parts_are_one_processes_own_run_in_a_process_of_their_own(
    harness: Harness, tmp_path: Path
) -> None:
    """A team's in-process MCP server runs its tools in the context of the process's first
    turn: with one, every turn has its own process, which resumes the session — as before."""

    @sdk_tool("find", "Find a customer.", {"name": str})
    async def find(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "Ada"}]}

    crm = create_sdk_mcp_server("crm", tools=[find])
    scripts = [[{"text": "noted"}], [{"text": "you told me: {heard}"}]]
    agent = harness.wrap(options(tmp_path, scripts, mcp_servers={"crm": crm}), id="own-server")
    await agent.run("the code is 4242", user="u1", thread="t1")
    assert open_sessions(agent) == 0
    second = await agent.run("what is it?", user="u1", thread="t1")
    assert second.answer == "you told me: the code is 4242 | what is it?"
    one, two = served(tmp_path)
    assert two["pid"] != one["pid"] and two["resume"] == one["session"]
    assert open_sessions(agent) == 0


async def test_a_session_the_options_keep_themselves_is_served_live_too(
    harness: Harness, tmp_path: Path
) -> None:
    """Options that keep the session on this machine (file checkpointing: no journal store)
    have a live session all the same."""
    scripts = [[{"text": "noted"}], [{"text": "you told me: {heard}"}]]
    target = options(tmp_path, scripts, enable_file_checkpointing=True)
    agent = harness.wrap(target, id="own-store")
    await agent.run("the code is 4242", user="u1", thread="t1")
    second = await agent.run("what is it?", user="u1", thread="t1")
    assert second.answer == "you told me: the code is 4242 | what is it?"
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"] and open_sessions(agent) == 1


async def test_at_its_turn_limit_a_conversations_turn_gets_its_last_answer_from_its_session(
    harness: Harness, tmp_path: Path
) -> None:
    """The one last answer at the turn limit is asked in the conversation's live session, as
    its next turn — the session and the background work the turn started there go on (a
    process of its own would be a second process on the session, and closing the live one
    first would stop that work): told the limit is reached, its tool calls are refused, and
    its answer is the run's."""
    reads = [{"builtin": "Read", "args": {"file_path": f"f{n}"}} for n in range(3)]
    target = options(tmp_path, [[BACKGROUND, *reads, {"text": "never"}]], max_turns=3)
    agent = harness.wrap(target, id="limited")
    result = await agent.run("research, then read", user="u1", thread="t1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == "What I found before the limit."
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"] and alive(one["pid"])  # the live session's own process
    child = await child_of(harness, result.run_id)
    live = live_of(agent)
    assert child.status is RunStatus.RUNNING, (child.status, child.error)
    assert live is not None and not live.closed and list(live.working) == ["t1"]
    assert open_sessions(agent) == 1


async def test_a_tool_call_in_the_last_answer_at_the_turn_limit_is_refused(
    harness: Harness, tmp_path: Path
) -> None:
    """The last answer's turn in a live session has the process's tools (a live process keeps
    the ones it started with): a call the model makes in it anyway is refused, not run, and
    the answer it gives then is the run's."""
    reads = [{"builtin": "Read", "args": {"file_path": f"f{n}"}} for n in range(3)]
    target = options(tmp_path, [[*reads, {"text": "never"}]], max_turns=2)
    target.env["FAKE_CLAUDE_AT_LIMIT_CALLS"] = json.dumps(
        {"builtin": "Write", "args": {"file_path": "notes.txt", "content": "x"}}
    )
    agent = harness.wrap(target, id="limited-calls")
    result = await agent.run("read both", user="u1", thread="t1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert [c["tool"] for c in built_ins(tmp_path)] == ["Read", "Read"]  # the Write did not run
    assert result.answer == AT_LIMIT


async def test_a_turn_after_the_hour_turned_is_served_by_the_same_session(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``schedule_task``'s description says the hour it is: a turn after the hour turned
    brings the same tools with another description. The session keeps serving the
    conversation — its process and the background work in it go on: what a later turn must
    bring the same is its tools and their schemas, not their wording of the moment."""
    from datetime import UTC, datetime

    from trellis.harness import snapshots

    clock = [datetime(2026, 11, 13, 10, 59, 50, tzinfo=UTC)]
    monkeypatch.setattr(snapshots, "now", lambda: clock[0])
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "still here"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="hourly")
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None and list(live.working) == ["t1"]
    clock[0] = datetime(2026, 11, 13, 11, 0, 10, tzinfo=UTC)  # the hour turned
    second = await agent.run("how is it going?", user="u1", thread="t1")
    assert second.status is RunStatus.SUCCESS and second.answer == "still here"
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"] and live_of(agent) is live and not live.closed
    assert list(live.working) == ["t1"]  # the background work goes on


async def test_each_turn_of_a_live_session_is_told_the_day_and_hour_it_is(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live process keeps the tools' wording it listed as it started — ``schedule_task``
    says the hour of the process's first turn — and Claude Code says the date once per
    session (and again only when the day changes). So each turn the session serves is told
    the day and the hour it is with its prompt: a turn 26 hours later reads its own, and
    the same process serves it."""
    from datetime import UTC, datetime

    from trellis.harness import snapshots

    clock = [datetime(2026, 11, 13, 10, 15, tzinfo=UTC)]
    monkeypatch.setattr(snapshots, "now", lambda: clock[0])
    scripts = [[{"text": "one"}], [{"text": "two"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="dated")
    await agent.run("remind me at 9", user="u1", thread="t1")
    clock[0] = datetime(2026, 11, 14, 12, 40, tzinfo=UTC)  # 26 hours later
    await agent.run("and at 9 again", user="u1", thread="t1")
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"]  # the same process
    assert "It is now Fri 2026-11-13, 10:00-10:59" in one["context"]
    assert "It is now Sat 2026-11-14, 12:00-12:59" in two["context"]
    assert "2026-11-13" not in two["context"]


async def test_a_session_another_worker_served_a_later_turn_of_is_not_served_stale(
    tmp_path: Path,
) -> None:
    """Worker 2 served turn 1 and keeps its process; worker 1 served turn 2 (worker 2's
    process never saw it). Turn 3 on worker 2 is not answered by that stale process, which
    holds the conversation as it was after turn 1: the process is ended and the turn runs as
    any turn another worker served the last of — a new session, the conversation's turns
    given as context (the run store's)."""
    scripts = [[{"text": "turn {heard}"}]]
    store = LocalRuns()
    one, two = Harness(config=Settings(), runs=store), Harness(config=Settings(), runs=store)
    try:
        first = one.wrap(options(tmp_path, scripts), id="chat")
        second = two.wrap(options(tmp_path, scripts), id="chat")
        await second.run("the code is 4242", user="u1", thread="t1")
        stale = live_of(second)
        assert stale is not None
        await first.run("the code is now 9999", user="u1", thread="t1")
        third = await second.run("what is the code?", user="u1", thread="t1")
        assert third.status is RunStatus.SUCCESS, third.error
        held, _, fresh = served(tmp_path)
        assert fresh["pid"] != held["pid"] and fresh["resume"] is None  # not the stale one
        assert fresh["session"] != held["session"]
        await until(lambda: not alive(held["pid"]))
        assert stale.closed and live_of(second) is not stale
        assert "the code is now 9999" in fresh["context"]  # turn 2, from the run store
    finally:
        await one.aclose()
        await two.aclose()


async def test_a_kept_session_whose_last_turn_the_run_store_no_longer_has_is_served(
    harness: Harness, tmp_path: Path
) -> None:
    """The last turn this process served is gone from the run store (pruned): nothing says
    another worker went past the session kept here, which serves the next turn."""
    scripts = [[{"text": "noted"}], [{"text": "you told me: {heard}"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="pruned")
    await agent.run("the code is 4242", user="u1", thread="t1")
    agent.turned[conversation_key("default", "u1", "t1")] = "run_pruned"
    again = await agent.run("what is it?", user="u1", thread="t1")
    assert again.answer == "you told me: the code is 4242 | what is it?"
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"]


async def test_a_session_whose_process_dies_fails_its_turn_and_the_next_turn_starts_another(
    harness: Harness, tmp_path: Path
) -> None:
    scripts = [[{"text": "noted"}], [{"stall": 30}, {"text": "never said"}], [{"text": "back"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="died")
    await agent.run("the code is 4242", user="u1", thread="t1")
    slow = asyncio.create_task(agent.run("a long one", user="u1", thread="t1"))
    live = live_of(agent)
    mirror = None if live is None else live.mirror
    assert mirror is not None
    # killed once its session holds the turn (its prompt mirrored to the harness's store): the
    # stand-in numbers a resumed session's turns by what its transcript holds
    await until(
        lambda: any(
            e.get("prompt") == "a long one" for kept in mirror.transcripts.values() for e in kept
        )
    )
    one, _ = served(tmp_path)
    os.kill(one["pid"], signal.SIGKILL)
    failed = await slow
    assert failed.status is RunStatus.ERROR
    await until(lambda: open_sessions(agent) == 0)
    again = await agent.run("are you there?", user="u1", thread="t1")
    assert again.status is RunStatus.SUCCESS and again.answer == "back"
    *_, three = served(tmp_path)
    assert three["pid"] != one["pid"] and three["resume"] == one["session"]


async def test_a_session_whose_process_ends_of_itself_is_gone_and_the_next_turn_starts_another(
    harness: Harness, tmp_path: Path
) -> None:
    scripts = [[{"text": "bye", "quit": True}], [{"text": "back"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="quit")
    first = await agent.run("the code is 4242", user="u1", thread="t1")
    assert first.status is RunStatus.SUCCESS and first.answer == "bye"
    await until(lambda: open_sessions(agent) == 0)
    again = await agent.run("are you there?", user="u1", thread="t1")
    assert again.status is RunStatus.SUCCESS and again.answer == "back"
    one, two = served(tmp_path)
    assert two["pid"] != one["pid"] and two["resume"] == one["session"]


async def test_a_session_that_owes_a_left_turns_result_for_too_long_is_replaced(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn left with work in the background is not interrupted (that would stop the work):
    it ends at its next tool call. While its result has not come the session cannot take the
    next prompt (the CLI could hand it to the turn under way); when it takes too long, the
    session is closed, the turn runs in a new process, and the run says what work that
    ended."""
    monkeypatch.setattr(claude_live, "SETTLE_SECONDS", 0.3)
    scripts = [
        [BACKGROUND, {"text": "started it"}],
        [{"stall": 30}, {"text": "never said"}],
        [{"text": "after"}],
    ]
    agent = harness.wrap(options(tmp_path, scripts), id="owing")
    await agent.run("research the market", user="u1", thread="t1")
    slow = asyncio.create_task(agent.run("a long one", user="u1", thread="t1"))
    await until(lambda: len(served(tmp_path)) == 2)
    await agent.cancel(turn_running(agent))
    assert (await slow).status is RunStatus.CANCELLED
    events = [e async for e in agent.stream("and now?", user="u1", thread="t1")]
    assert events[-1].data["result"] == "after"
    [said] = warned(events)
    assert "an earlier turn's result never came" in said and "t1 (research the market)" in said
    one, _, three = served(tmp_path)
    assert three["pid"] != one["pid"] and three["resume"] == one["session"]
    await until(lambda: not alive(one["pid"]))


async def test_a_turn_cancelled_while_its_session_starts_leaves_no_process(
    harness: Harness, tmp_path: Path
) -> None:
    pid = tmp_path / "pid"
    env = {"FAKE_CLAUDE_STARTS_AFTER": "5", "FAKE_CLAUDE_PID": str(pid)}
    agent = harness.wrap(options(tmp_path, [[{"text": "ok"}]], env=env), id="starting")
    slow = asyncio.create_task(agent.run("hi", user="u1", thread="t1"))
    await until(pid.exists)
    await until(lambda: pid.read_text() != "")
    [run_id] = agent.running
    await agent.cancel(run_id)
    assert (await slow).status is RunStatus.CANCELLED
    assert open_sessions(agent) == 0
    await until(lambda: not alive(int(pid.read_text())), seconds=20)


# --------------------------------------------------------------------------- a pause
async def test_a_pause_keeps_the_session_and_its_background_work_and_the_resume_goes_into_it(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [BACKGROUND, {"tool": "refund", "args": {"order": "o9"}}, {"text": "done"}]
    agent = arm(harness.wrap(options(tmp_path, [script]), id="paused"), [refund])
    paused = await agent.run("research, then refund o9", user="u1", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and refunds == []
    live = live_of(agent)
    [one] = served(tmp_path)
    assert live is not None and not live.closed and alive(one["pid"])
    await until(lambda: live.owed == 0)  # the CLI's turn ended at the call that waits
    assert list(live.working) == ["t1"]  # the approval did not end the task

    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "done"
    assert refunds == ["o9"]
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"] and two["resume"] is None  # sent into the live session
    assert two["prompt"] == RESUMED_CALL.format(tool="mcp__trellis__refund")
    assert live_of(agent) is live and list(live.working) == ["t1"]


async def test_background_work_that_ends_with_its_session_is_said_in_the_next_runs_events(
    harness: Harness, tmp_path: Path
) -> None:
    """A turn the live session cannot serve (its options are not the ones the process was
    started with) runs in a new process, which ends the old one and the task in it: the run
    says so."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="lost")
    await agent.run("research the market", user="u1", thread="t1")
    again = agent.stream("and now?", user="u1", thread="t1", framework_options={"max_turns": 7})
    events = [e async for e in again]
    assert events[-1].data["result"] == "ok"
    [said] = warned(events)
    assert "t1 (research the market)" in said and "was stopped" in said
    one, two = served(tmp_path)
    assert two["pid"] != one["pid"] and two["resume"] == one["session"]
    await until(lambda: not alive(one["pid"]))
    assert open_sessions(agent) == 1  # the new process is the conversation's session now


# --------------------------------------------------------------------------- a background run
async def test_a_background_subagents_calls_are_checked_and_recorded_for_its_own_run(
    harness: Harness, tmp_path: Path
) -> None:
    """A subagent at work in the background keeps the child run its ``Agent`` call started:
    the calls it makes are decided, run and recorded for that run — between two turns, and
    while a turn is answered too (they are not that turn's)."""

    @tool(side_effects="read")
    def whoami() -> str:
        """The run this call is made in."""
        runtime = current()
        assert runtime is not None
        return runtime.run_id

    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | None:
            return Deny("no searching here") if call.tool == "Grep" else None

    scripts = [
        [BACKGROUND, {"text": "started it"}],
        [{"stall": 0.8}, {"text": "done"}],
    ]
    agent = arm(harness.wrap(options(tmp_path, scripts), id="attended", hooks=[Guard()]), [whoami])
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    assert child.status is RunStatus.RUNNING and child.metadata[HELPER]["name"] == "researcher"
    seen, following = await followed(agent, child.run_id)
    live = live_of(agent)
    assert live is not None and live.turn is None  # no turn is being answered

    assert await calls(tmp_path, READS) == {"result": "Read ran", "ok": True}
    grep = {"builtin": "Grep", "args": {"pattern": "x"}}
    assert await calls(tmp_path, grep) == {
        "result": "Grep was not run: no searching here",  # the run's hooks
        "ok": False,
    }
    # a harness tool is the run's own too
    assert await calls(tmp_path, {"tool": "whoami"}) == {"result": child.run_id, "ok": True}
    assert built_ins(tmp_path) == [{"tool": "Read", "args": {"file_path": "notes.txt"}}]
    assert results(seen) == ["Read", "whoami"] and list(live.working) == ["t1"]

    async def during_the_turn() -> None:
        await until(lambda: len(served(tmp_path)) == 2)
        (tmp_path / "tasks" / "t1.call").write_text(json.dumps(READS))

    calling = asyncio.create_task(during_the_turn())
    events = [e async for e in agent.stream("how is it going?", user="u1", thread="t1")]
    await calling
    assert events[-1].data["result"] == "done"
    called = json.loads((tmp_path / "tasks" / "t1.called").read_text())
    assert called == {"result": "Read ran", "ok": True}
    assert results(events) == [] and results(seen) == ["Read", "whoami", "Read"]
    assert warned(events) == [] and not live.refused  # none was refused
    following.cancel()


async def test_a_background_subagents_run_stays_open_across_turns_and_ends_with_its_result(
    harness: Harness, tmp_path: Path
) -> None:
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "4"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="open")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    assert (child.status, child.parent_run_id) == (RunStatus.RUNNING, first.run_id)
    assert child.input == "research the market"
    second = await agent.run("what is 2+2?", user="u1", thread="t1")
    assert second.answer == "4"
    assert (await child_of(harness, first.run_id)).status is RunStatus.RUNNING
    assert child.run_id in agent.running
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    ended = await when(harness, child.run_id, RunStatus.SUCCESS)
    assert ended.output == "the market is growing"
    await until(lambda: child.run_id not in agent.running)
    live = live_of(agent)
    assert live is not None and not live.background and not live.closed


async def test_a_background_subagent_that_ends_while_its_turn_still_runs_says_so_on_its_stream(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn that started a background subagent is still being answered when its task
    ends: the child run's end is on that turn's task tree (``helper_finished``), as a
    subagent that ends with its call says it. Once the turn has ended there is no stream
    of it left to say so on (the child run's own stream has its end)."""
    # the task's run waits this long for the session's next result (what it cost) at most
    monkeypatch.setattr(claude_live, "SETTLE_SECONDS", 0.2)
    scripts = [[BACKGROUND, {"stall": 30}, {"text": "never said"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="told")
    events: list[RunEvent] = []

    async def run() -> None:
        events.extend([e async for e in agent.stream("research", user="u1", thread="t1")])

    seen, following = await followed_first(agent, events)
    running = asyncio.create_task(run())
    run_id = await launched(agent, seen)
    child = await child_of(harness, run_id)
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await when(harness, child.run_id, RunStatus.SUCCESS)
    await until(lambda: any(e.data.get("name") == "helper_finished" for e in seen))
    await agent.cancel(run_id)
    await running
    await following
    [ended] = [e.data for e in events if e.data.get("name") == "helper_finished"]
    assert (ended["run_id"], ended["agent_id"]) == (child.run_id, "researcher")
    assert (ended["status"], ended["framework"]) == ("success", "claude_agent_sdk")


@pytest.mark.parametrize(
    ("ending", "status", "why"),
    [
        ("failed", RunStatus.ERROR, "the source was down"),
        ("stop", RunStatus.CANCELLED, ""),  # stopped in Claude Code (its own ``TaskStop``)
    ],
)
async def test_a_background_subagents_run_ends_as_its_task_ended(
    harness: Harness, tmp_path: Path, ending: str, status: RunStatus, why: str
) -> None:
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id=f"ended-{ending}")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    (tmp_path / "tasks" / f"t1.{ending}").write_text(why)
    ended = await when(harness, child.run_id, status)
    assert (ended.error.message if ended.error is not None else "") == why
    assert not (tmp_path / "tasks" / "t1.stopped").exists()  # the harness stopped nothing
    after = await agent.run("and?", user="u1", thread="t1")
    assert after.answer == "ok" and len({turn["pid"] for turn in served(tmp_path)}) == 1


async def test_a_background_subagents_run_is_closed_as_lost_when_its_session_ends(
    harness: Harness, tmp_path: Path
) -> None:
    """The session's process died, or was replaced: the run says so, and is not left open."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="lost-run")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    [one] = served(tmp_path)
    os.kill(one["pid"], signal.SIGKILL)
    died = await when(harness, child.run_id, RunStatus.ERROR)
    assert died.error is not None
    assert "session was closed (its process ended)" in died.error.message
    assert "background subagent" in died.error.message

    again = await agent.run("research the market", user="u1", thread="t2")
    other = await child_of(harness, again.run_id)
    options_changed = {"max_turns": 7}
    await agent.run("and now?", user="u1", thread="t2", framework_options=options_changed)
    replaced = await when(harness, other.run_id, RunStatus.ERROR)
    assert replaced.error is not None
    assert "options are not the ones it was started with" in replaced.error.message


async def test_a_background_subagents_write_asks_a_person_in_the_next_turn_and_then_runs(
    tmp_path: Path,
) -> None:
    """A call the catalog asks a person about pauses the subagent's run as any run pauses,
    and waits in Claude Code (its process and the task stay). Nothing answers a child run's
    own question today (``Agent.resume`` refuses a run that has a parent), so it is asked in
    the conversation's next turn, before that turn's prompt is sent: the turn pauses with
    it, and its answer lets the call through."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "you asked: {heard}"}]]
    catalog = FakeCatalog({"Write": Rule(risk="write", approve_when="true")})
    async with with_governance(Harness(config=Settings()), Governance(catalog)) as h:
        agent = h.wrap(options(tmp_path, scripts, allowed_tools=["Write"]), id="asks")
        first = await agent.run("research the market", user="u1", thread="t1")
        child = await child_of(h, first.run_id)
        (tmp_path / "tasks" / "t1.call").write_text(json.dumps(WRITES))
        paused = await when(h, child.run_id, RunStatus.PAUSED)
        asked = paused.awaiting
        assert asked is not None and asked.tool_call is not None
        assert (asked.tool_call.tool, asked.tool_call.args) == ("Write", WRITES["args"])
        live = live_of(agent)
        [one] = served(tmp_path)
        assert live is not None and list(live.working) == ["t1"] and alive(one["pid"])
        assert built_ins(tmp_path) == [] and not (tmp_path / "tasks" / "t1.called").exists()
        with pytest.raises(ConfigurationError, match="answer the question its parent run"):
            await agent.resume(asked.interrupt_id, "approve", reviewer="u1")

        turn = await agent.run("how is it going?", user="u1", thread="t1")
        assert turn.status is RunStatus.PAUSED and turn.interrupt is not None
        named = (turn.interrupt.payload or {})[SUBAGENT]
        assert (named["run_id"], named["interrupt_id"]) == (child.run_id, asked.interrupt_id)
        assert turn.interrupt.tool_call == asked.tool_call
        assert len(served(tmp_path)) == 1  # its prompt was not sent yet

        done = await agent.resume(turn.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS
        assert done.answer == "you asked: research the market | how is it going?"
        await until((tmp_path / "tasks" / "t1.called").exists)
        called = json.loads((tmp_path / "tasks" / "t1.called").read_text())
        assert called == {"result": "Write ran", "ok": True}
        assert built_ins(tmp_path) == [{"tool": "Write", "args": WRITES["args"]}]  # once
        assert (await when(h, child.run_id, RunStatus.RUNNING)).attempt == 2
        (tmp_path / "tasks" / "t1.done").write_text("written")
        assert (await when(h, child.run_id, RunStatus.SUCCESS)).output == "written"
        _, two = served(tmp_path)
        assert two["pid"] == one["pid"] and two["prompt"] == "how is it going?"


async def test_a_rejected_call_of_a_background_subagent_is_told_to_it(
    harness: Harness, tmp_path: Path
) -> None:
    """Claude Code asks before a shell command that writes: the harness answers it for the
    subagent's run (``can_use_tool``), a person decides in the next turn, and the subagent
    reads the rejection."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="rejects")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    touch = {"builtin": "Bash", "args": {"command": "touch notes.txt"}}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(touch))
    await when(harness, child.run_id, RunStatus.PAUSED)
    turn = await agent.run("how is it going?", user="u1", thread="t1")
    assert turn.status is RunStatus.PAUSED and turn.interrupt is not None
    done = await agent.resume(
        turn.interrupt.interrupt_id, "reject", reviewer="u1", comment="not there"
    )
    assert done.answer == "ok"
    await until((tmp_path / "tasks" / "t1.called").exists)
    called = json.loads((tmp_path / "tasks" / "t1.called").read_text())
    assert called == {
        "result": "Bash was not run: the approver rejected it (not there)",
        "ok": False,
    }
    assert built_ins(tmp_path) == []
    assert (await when(harness, child.run_id, RunStatus.RUNNING)).attempt == 2


async def test_a_background_subagents_harness_tool_asks_a_person_and_runs_once(
    harness: Harness, tmp_path: Path
) -> None:
    """A harness tool the subagent calls is its run's too: one that asks first pauses that
    run, its call waits in Claude Code, and the approved call runs once."""
    refunds.clear()
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = arm(harness.wrap(options(tmp_path, scripts), id="bg-refund"), [refund])
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    step = {"tool": "refund", "args": {"order": "o9"}}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(step))
    paused = await when(harness, child.run_id, RunStatus.PAUSED)
    assert paused.awaiting is not None and paused.awaiting.tool_call is not None
    assert paused.awaiting.tool_call.tool == "refund" and refunds == []
    turn = await agent.run("how is it going?", user="u1", thread="t1")
    assert turn.status is RunStatus.PAUSED and turn.interrupt is not None
    done = await agent.resume(turn.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.answer == "ok"
    await until((tmp_path / "tasks" / "t1.called").exists)
    called = json.loads((tmp_path / "tasks" / "t1.called").read_text())
    assert called == {"result": "refunded o9", "ok": True} and refunds == ["o9"]


async def test_a_background_subagents_harness_tool_call_a_hook_rewrote_is_its_runs(
    harness: Harness, tmp_path: Path
) -> None:
    """A team's own ``PreToolUse`` hook that rewrites a background subagent's harness tool
    call (``updatedInput``): the call Claude Code makes, with the rewritten arguments, is
    still known as that subagent's, and runs in its run — not refused for want of one, nor
    run in a turn being answered meanwhile."""

    @tool(side_effects="read")
    def echo(x: str) -> str:
        """Say which run this call is made in, and what it was given."""
        runtime = current()
        assert runtime is not None
        return f"{runtime.run_id}:{x}"

    async def rewrite(given: Any, tool_use_id: Any, hook: Any) -> Any:
        if given.get("tool_name") != "mcp__trellis__echo":
            return {}
        decided = {"hookEventName": "PreToolUse", "permissionDecision": "allow"}
        return {"hookSpecificOutput": {**decided, "updatedInput": {"x": "rewritten"}}}

    from claude_agent_sdk import HookMatcher

    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    target = options(tmp_path, scripts, hooks={"PreToolUse": [HookMatcher(hooks=[rewrite])]})
    agent = arm(harness.wrap(target, id="bg-rewritten"), [echo])
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    said = await calls(tmp_path, {"tool": "echo", "args": {"x": "given"}})
    assert said == {"result": f"{child.run_id}:rewritten", "ok": True}


async def test_a_background_subagents_call_of_a_tool_its_own_run_was_not_offered(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    tmp_path: Path,
    narrowing: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hints narrow each run's tools by its own task: the process lists the tools the
    turn that started it was offered, and the background subagent's run may be offered
    others. The subagent is offered what the process lists, so its call of a tool its own
    run was not offered runs, in its run (``read_file`` and such are offered to any)."""

    def lookup(i: int) -> Any:
        def find(key: str) -> str:
            runtime = current()
            assert runtime is not None
            return f"t{i}:{key}:{runtime.run_id}"

        return tool(find, name=f"t{i}", side_effects="read")

    hinted = iter([["t0"]])  # the turn's, then the subagent's own (its task: "dig")
    monkeypatch.setattr(memory_service, "_candidates", lambda task, available: next(hinted, ["t1"]))
    dig = {"background": {"id": "t1", "agent": "researcher", "prompt": "dig"}}
    scripts = [[dig, {"text": "started it"}], [{"text": "ok"}]]
    agent = arm(memory_harness.wrap(options(tmp_path, scripts), id="bg-narrowed"),
                [lookup(i) for i in range(3)])  # fmt: skip
    first = await agent.run("research the market", user="u1", thread="t1")
    [listed] = served(tmp_path)
    assert "t0" in listed["tools"] and "t1" not in listed["tools"]
    child = await child_of(memory_harness, first.run_id, None)
    said = await calls(tmp_path, {"tool": "t0", "args": {"key": "k"}})
    assert said == {"result": f"t0:k:{child.run_id}", "ok": True}


async def test_a_background_subagents_call_of_a_tool_its_run_does_not_have_is_refused() -> None:
    """A tool the process lists that the subagent's run has not at all (another tool set)
    is refused, saying so: it never runs outside a run."""
    import contextvars
    from types import SimpleNamespace

    at = claude_live.Turn(SimpleNamespace(toolbox={}), None, None, contextvars.copy_context())
    answer = await claude_live._handler(at, "gone")({})
    assert answer["is_error"] and "gone is not a tool of this run" in answer["content"][0]["text"]


async def test_a_question_nobody_answers_ends_the_background_run_and_stops_its_task(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call waits in Claude Code's hook, which the CLI is told to wait for longer than the
    harness waits for a person: unanswered, the run is cancelled and the task stopped —
    never left to the CLI's own timeout."""
    monkeypatch.setattr(claude_live, "ANSWER_SECONDS", 0.2)
    monkeypatch.setattr(claude_live, "WAITS_SECONDS", 0.1)
    scripts = [[BACKGROUND, {"text": "started it"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="unanswered")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    touch = {"builtin": "Bash", "args": {"command": "touch notes.txt"}}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(touch))
    await when(harness, child.run_id, RunStatus.PAUSED)
    await when(harness, child.run_id, RunStatus.CANCELLED)
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    await until((tmp_path / "tasks" / "t1.called").exists)
    called = json.loads((tmp_path / "tasks" / "t1.called").read_text())
    assert called == {"result": UNATTENDED, "ok": False} and built_ins(tmp_path) == []
    live = live_of(agent)
    assert live is not None and not live.closed
    hooks = live.client.options.hooks
    assert {m.timeout for matchers in hooks.values() for m in matchers} == {
        claude_live.HOOK_SECONDS
    }


async def test_a_teams_own_hooks_keep_their_timeout_in_a_live_session(
    harness: Harness, tmp_path: Path
) -> None:
    """Only the harness's own hooks, which may wait for a person's answer about a background
    subagent's call, are given the long timeout in a live session: a team's own hook keeps
    what it set — none, Claude Code's own default, as in a process of its own; a time of its
    own, that time."""
    from claude_agent_sdk import HookMatcher

    async def own(given: Any, tool_use_id: Any, hook: Any) -> Any:
        return {}

    hooks = {
        "Stop": [HookMatcher(hooks=[own])],
        "PreToolUse": [HookMatcher(hooks=[own], timeout=7)],
    }
    agent = harness.wrap(options(tmp_path, [[{"text": "ok"}]], hooks=hooks), id="own-hooks")
    await agent.run("hi", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    given = live.client.options.hooks
    [stop] = given["Stop"]
    team, *ours = given["PreToolUse"]
    assert (stop.timeout, team.timeout) == (None, 7)
    assert ours and {m.timeout for m in ours} == {claude_live.HOOK_SECONDS}


async def test_cancelling_a_background_subagents_run_stops_that_subagent_only(
    harness: Harness, tmp_path: Path
) -> None:
    """``agent.cancel`` of the child run stops its task with the SDK's own ``stop_task`` (an
    interrupt would stop every task): the other subagent, its run and the session go on."""
    other = {"background": {"id": "t2", "agent": "writer", "prompt": "write it up"}}
    scripts = [[BACKGROUND, other, {"text": "started both"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="stops-one")
    first = await agent.run("research and write", user="u1", thread="t1")
    found = [r async for r in harness.runs.iterate(parent_run_id=first.run_id, tenant="default")]
    records = [await harness.runs.get(r.run_id, tenant="default") for r in found]
    by_task = {r.input: r.run_id for r in records if r is not None}
    assert set(by_task) == {"research the market", "write it up"}  # a run each
    research, writing = by_task["research the market"], by_task["write it up"]
    seen, following = await followed(agent, writing)

    cancelled = await agent.cancel(research, reason="not needed")
    assert cancelled.status is RunStatus.CANCELLED
    tasks = tmp_path / "tasks"
    await until((tasks / "t1.stopped").exists)
    live = live_of(agent)
    assert live is not None
    await until(lambda: list(live.working) == ["t2"])
    assert not (tasks / "t2.stopped").exists() and not live.closed
    # each subagent's calls are its own run's
    assert await calls(tmp_path, READS, task="t2") == {"result": "Read ran", "ok": True}
    assert results(seen) == ["Read"]
    assert (await agent.run("and?", user="u1", thread="t1")).answer == "ok"
    (tasks / "t2.done").write_text("written up")
    assert (await when(harness, writing, RunStatus.SUCCESS)).output == "written up"
    assert len({turn["pid"] for turn in served(tmp_path)}) == 1
    following.cancel()


async def test_cancelling_a_background_run_that_waits_for_a_person_stops_its_task(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paused run is cancelled at once, wherever it is cancelled from; the session, which
    looks every few seconds, then stops the task and answers the call that waited."""
    monkeypatch.setattr(claude_live, "WAITS_SECONDS", 0.1)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="cancel-paused")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    touch = {"builtin": "Bash", "args": {"command": "touch notes.txt"}}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(touch))
    await when(harness, child.run_id, RunStatus.PAUSED)
    assert (await agent.cancel(child.run_id)).status is RunStatus.CANCELLED
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    await until((tmp_path / "tasks" / "t1.called").exists)
    assert built_ins(tmp_path) == []
    after = await agent.run("and?", user="u1", thread="t1")  # nothing is asked any more
    assert after.status is RunStatus.SUCCESS and after.answer == "ok"


async def test_a_background_subagents_run_past_its_time_is_stopped_with_stop_task(
    harness: Harness, tmp_path: Path
) -> None:
    """The run's limit is its agent's ``run_timeout``, as for a run nothing else bounds: the
    turn that started it has ended."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="timed", run_timeout=3.0)
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    timed_out = await when(harness, child.run_id, RunStatus.TIMEOUT, seconds=15.0)
    assert timed_out.error is not None and timed_out.error.code == "run_timeout"
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    live = live_of(agent)
    assert live is not None and not live.closed
    await until(lambda: not live.working)
    assert (await agent.run("and?", user="u1", thread="t1")).answer == "ok"


async def test_a_background_subagents_run_past_its_budget_is_stopped_with_stop_task(
    harness: Harness, tmp_path: Path
) -> None:
    """Its replies' tokens count toward its run's budget (the agent's), between turns too."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    target = options(tmp_path, scripts)
    agent = harness.wrap(target, id="capped", budget=Budget(max_tokens=40))
    first = await agent.run("research the market", user="u1", thread="t1")  # 30 of its own
    child = await child_of(harness, first.run_id)
    for _ in range(2):  # 15 a reply
        assert await calls(tmp_path, READS) == {"result": "Read ran", "ok": True}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(READS))  # the third: past 40
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    partial = await when(harness, child.run_id, RunStatus.PARTIAL)
    assert partial.output == "stopped"


async def test_cancelling_the_turn_that_started_a_background_subagent_stops_it_too(
    harness: Harness, tmp_path: Path
) -> None:
    """A cancelled run's children are cancelled with it: the subagent it started and that
    still works is stopped — unlike a later turn's cancel, which leaves it."""
    scripts = [[BACKGROUND, {"stall": 2}, {"text": "never said"}], [{"text": "after"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="parent-cancelled")
    slow = asyncio.create_task(agent.run("research the market", user="u1", thread="t1"))
    await until(lambda: len(agent.running) == 2)
    live = live_of(agent)
    assert live is not None
    await until(lambda: all(found.turn is not None for found in live.background.values()))
    started = turn_running(agent)
    child = await child_of(harness, started)
    await agent.cancel(started)
    assert (await slow).status is RunStatus.CANCELLED
    await when(harness, child.run_id, RunStatus.CANCELLED)
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    after = await agent.run("and now?", user="u1", thread="t1")
    assert after.status is RunStatus.SUCCESS and after.answer == "after"


async def test_a_turn_cancelled_before_its_background_subagents_run_began_stops_it_too(
    harness: Harness, tmp_path: Path
) -> None:
    """The subagent's run was started but its first attempt had not begun to serve it (its
    hooks were still starting) when the turn that started it was cancelled: it does not go
    on unchecked — it ends ``CANCELLED`` and its task is stopped."""
    held = asyncio.Event()

    class Slow(Hooks):
        async def on_run_start(self, run: Any) -> None:
            if run.identity.parent_run is not None:  # the subagent's run
                await held.wait()

    scripts = [[BACKGROUND, {"stall": 2}, {"text": "never said"}], [{"text": "after"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="cancelled-early", hooks=[Slow()])
    slow = asyncio.create_task(agent.run("research the market", user="u1", thread="t1"))
    await until(lambda: len(agent.running) == 2)
    started = turn_running(agent)
    child = await child_of(harness, started)
    await agent.cancel(started)
    assert (await slow).status is RunStatus.CANCELLED
    held.set()
    await when(harness, child.run_id, RunStatus.CANCELLED)
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    await until(lambda: child.run_id not in agent.running)
    after = await agent.run("and now?", user="u1", thread="t1")
    assert after.status is RunStatus.SUCCESS and after.answer == "after"


async def test_a_background_run_cancelled_before_its_first_attempt_began_stops_its_task(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``agent.cancel`` finds no attempt of the run in this process yet (it is still being
    set up) and cancels its record: the attempt that then begins does not serve a run that
    has ended — the task is stopped."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="cancelled-unborn")
    held = asyncio.Event()
    bound = agent.run_memory

    async def slow(identity: Any, *args: Any, **kwargs: Any) -> Any:
        if identity.parent_run is not None:  # the subagent's run
            await held.wait()
        return await bound(identity, *args, **kwargs)

    monkeypatch.setattr(agent, "run_memory", slow)
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    assert child.run_id not in agent.running
    assert (await agent.cancel(child.run_id)).status is RunStatus.CANCELLED
    held.set()
    await until((tmp_path / "tasks" / "t1.stopped").exists)
    live = live_of(agent)
    assert live is not None
    await until(lambda: not live.background and child.run_id not in agent.running)
    assert (await child_of(harness, first.run_id)).status is RunStatus.CANCELLED
    assert (await agent.run("and?", user="u1", thread="t1")).answer == "ok"


async def test_a_turn_resumed_after_its_own_pause_leaves_its_background_run_open(
    harness: Harness, tmp_path: Path
) -> None:
    """The turn that started the subagent paused on a call of its own: the attempt that
    follows the answer does not end the subagent's run as one it left open."""
    refunds.clear()
    script = [BACKGROUND, {"tool": "refund", "args": {"order": "o9"}}, {"text": "done"}]
    agent = arm(harness.wrap(options(tmp_path, [script]), id="paused-parent"), [refund])
    paused = await agent.run("research, then refund o9", user="u1", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    child = await child_of(harness, paused.run_id)
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and refunds == ["o9"]
    assert (await child_of(harness, paused.run_id)).status is RunStatus.RUNNING
    assert await calls(tmp_path, READS) == {"result": "Read ran", "ok": True}
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    assert (await when(harness, child.run_id, RunStatus.SUCCESS)).output == (
        "the market is growing"
    )


async def test_a_turn_resumed_after_its_background_subagent_ended_goes_on_in_its_session(
    harness: Harness, tmp_path: Path
) -> None:
    """A background subagent the turn handed on has ended by the time the turn is resumed:
    nothing holds the session elsewhere, and the resumed attempt is sent into the live
    session, as any resume (no fork)."""
    refunds.clear()
    script = [BACKGROUND, {"tool": "refund", "args": {"order": "o9"}}, {"text": "done"}]
    agent = arm(harness.wrap(options(tmp_path, [script]), id="ended-bg"), [refund])
    paused = await agent.run("research, then refund o9", user="u1", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    child = await child_of(harness, paused.run_id)
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await when(harness, child.run_id, RunStatus.SUCCESS)
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and refunds == ["o9"]
    pids = {turn["pid"] for turn in served(tmp_path)}
    assert len(pids) == 1 and all("--fork-session" not in t["argv"] for t in served(tmp_path))


async def test_a_left_turns_result_is_read_past_what_came_before_it() -> None:
    """``LiveSession.outcome``: the result of a turn left and interrupted, past the messages
    that came before it; none when the process ended first (and the turn is no longer the
    one whose result is owed)."""
    import contextvars

    from claude_agent_sdk import AssistantMessage, TextBlock

    session = LiveSession("k", None, claude_live.LiveSessions())
    turn = claude_live.Turn(None, None, None, contextvars.copy_context(), stopped=True)
    said = AssistantMessage(content=[TextBlock("…")], model="m")
    ended = ResultMessage("success", 1, 1, False, 1, "s")
    for message in (said, ended):
        turn.queue.put_nowait(message)
    assert await session.outcome(turn) is ended
    session.left = turn
    turn.queue.put_nowait(None)  # the process ended
    assert await session.outcome(turn) is None and session.left is None
    turn.queue.put_nowait(None)  # and another's result is the one owed now
    session.left = other = claude_live.Turn(None, None, None, contextvars.copy_context())
    assert await session.outcome(turn) is None and session.left is other


async def test_a_background_run_wakes_only_the_requests_still_waiting_for_an_attempt() -> None:
    """A request that stopped waiting for its run's next attempt (cancelled with the
    session's close) is passed over when the run wakes those that wait."""
    from types import SimpleNamespace

    loop = asyncio.get_running_loop()
    gone, waiting = loop.create_future(), loop.create_future()
    gone.cancel()
    found: Any = SimpleNamespace(_waiting=[gone, waiting])
    claude_live.Background._woken(found)
    assert waiting.result() is None and found._waiting == []


async def test_a_turn_resumed_on_another_worker_leaves_the_first_its_background_subagent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn that started a background subagent (kept going by worker 1's live session)
    paused, and is resumed on worker 2, which shares the run store: the turn's own work goes
    on there, in a process of its own that forks the session (``fork_session``) — never a
    second process on the session worker 1's still holds — and the subagent stays with worker
    1, its run open: worker 2 does not end it as one the earlier attempt left open, whatever
    its own process knows (the run's journal says it was handed on)."""
    from trellis.harness.bridges import helpers

    refunds.clear()
    script = [BACKGROUND, {"tool": "refund", "args": {"order": "o9"}}, {"text": "done"}]
    target = options(tmp_path, [script])
    store = LocalRuns()
    one, two = Harness(config=Settings(), runs=store), Harness(config=Settings(), runs=store)
    try:
        first = arm(one.wrap(target, id="bg"), [refund])
        second = arm(two.wrap(target, id="bg"), [refund])
        paused = await first.run("research, then refund o9", user="u1", thread="t1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        live = live_of(first)
        assert live is not None and list(live.working) == ["t1"]  # at work on worker 1
        # worker 2 is another process: nothing worker 1 keeps in memory is known there
        monkeypatch.setattr(helpers, "_DETACHED", set(), raising=False)
        done = await second.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS and done.answer == "done", done.error
        assert refunds == ["o9"]
        held, resumed = served(tmp_path)
        assert resumed["pid"] != held["pid"] and alive(held["pid"])
        assert (
            resumed["started_resuming"] == held["session"] and "--fork-session" in resumed["argv"]
        )
        assert resumed["session"] != held["session"]  # a fork: worker 1's session is its own
        assert list(live.working) == ["t1"] and not live.closed  # still at work on worker 1
        child = await child_of(one, paused.run_id)
        assert child.status is RunStatus.RUNNING, (child.status, child.error)
        (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
        assert (await when(one, child.run_id, RunStatus.SUCCESS)).output == "the market is growing"
    finally:
        await one.aclose()
        await two.aclose()


async def test_the_main_agents_own_call_with_no_turn_is_still_refused(
    harness: Harness, tmp_path: Path
) -> None:
    """Only background work that has a run may act between turns: a call the main agent
    makes in the turn the session runs for a finished task has none, and is refused."""
    injected = json.dumps([READS, {"text": "Noted: {last}"}])
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    target = options(tmp_path, scripts, env={"FAKE_CLAUDE_INJECTED": injected})
    agent = harness.wrap(target, id="main-unattended")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await when(harness, child.run_id, RunStatus.SUCCESS)
    live = live_of(agent)
    assert live is not None
    await until(lambda: any(isinstance(m, ResultMessage) for m in live.kept))
    [noted] = [m.result for m in live.kept if isinstance(m, ResultMessage)]
    assert noted == f"Noted: {UNATTENDED}" and built_ins(tmp_path) == []
    events = [e async for e in agent.stream("and?", user="u1", thread="t1")]
    [said] = warned(events)
    assert "were refused" in said and "Read x1" in said


async def test_another_users_thread_of_the_same_name_sees_nothing_of_a_background_run(
    harness: Harness, tmp_path: Path
) -> None:
    """Ada's subagent waits for a person: Bob's thread of the same name has its own session
    and its own subagent's run, and is asked nothing of hers."""
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="two-users-bg")
    hers = await agent.run("research the market", user="ada", thread="t1")
    child = await child_of(harness, hers.run_id)
    touch = {"builtin": "Bash", "args": {"command": "touch notes.txt"}}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(touch))
    await when(harness, child.run_id, RunStatus.PAUSED)
    his = await agent.run("research the market", user="bob", thread="t1")
    assert his.status is RunStatus.SUCCESS and his.interrupt is None
    again = await agent.run("how is it going?", user="bob", thread="t1")
    assert again.status is RunStatus.SUCCESS and again.answer == "ok"  # asked nothing
    theirs, mine = live_of(agent, "ada"), live_of(agent, "bob")
    assert theirs is not None and mine is not None and theirs is not mine
    [her_run], [his_run] = theirs.background.values(), mine.background.values()
    other = await child_of(harness, his.run_id)
    assert (her_run.run_id, his_run.run_id) == (child.run_id, other.run_id)
    assert (child.user_id, other.user_id) == ("ada", "bob") and other.status is RunStatus.RUNNING
    asked = await agent.run("how is it going?", user="ada", thread="t1")
    assert asked.status is RunStatus.PAUSED  # hers to answer


async def test_today_the_thread_stream_lists_a_threads_turns_and_not_its_background_runs(
    harness: Harness, tmp_path: Path
) -> None:
    """What ``GET {path}/threads/{thread}/events`` replays and follows is the runs whose
    record names the thread: a conversation's turns. A subagent's child run is a helper's
    record, whose thread is its own (its run id): it is not on the stream, so its result
    reaches an open chat only through Claude's next answer. Delivery is not built here."""
    from trellis.harness.agui.threads import Threads

    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="streamed")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await when(harness, child.run_id, RunStatus.SUCCESS)
    second = await agent.run("and?", user="u1", thread="t1")
    listed = await Threads(agent)._replayed("default", "u1", "t1")
    assert [r.run_id for r in listed] == [first.run_id, second.run_id]
    assert child.thread_id == child.run_id and child.agent_id == agent.id


async def test_a_foreground_subagents_run_ends_with_its_call_in_a_live_session_too(
    harness: Harness, tmp_path: Path
) -> None:
    """A subagent the CLI does not run in the background is the turn's own helper, as in a
    query's own process: its child run ends with its ``Agent`` call's result."""
    steps = [{"builtin": "Read", "args": {"file_path": "notes.txt"}}, {"text": "found it"}]
    script = [{"agent": "researcher", "prompt": "look it up", "steps": steps}, {"text": "{last}"}]
    agent = harness.wrap(options(tmp_path, [script], allowed_tools=["Read"]), id="foreground")
    first = await agent.run("look it up", user="u1", thread="t1")
    assert first.answer == "found it"
    child = await child_of(harness, first.run_id)
    assert (child.status, child.output) == (RunStatus.SUCCESS, "found it")
    live = live_of(agent)
    assert live is not None and not live.background


# --------------------------------------------------------------------------- its life
async def test_a_session_with_work_is_kept_past_the_idle_bound_but_not_for_ever(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_live, "IDLE_SECONDS", 0.1)
    monkeypatch.setattr(claude_live, "WORKING_SECONDS", 0.8)
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="bounded")
    await agent.run("research the market", user="u1", thread="t1")
    [one] = served(tmp_path)
    await asyncio.sleep(0.4)
    assert open_sessions(agent) == 1 and alive(one["pid"])
    await until(lambda: open_sessions(agent) == 0)
    # Claude Code goes on with a subagent's work once its stdin closes (as the bundled CLI
    # does); the SDK's close stops it 5 s later
    await until(lambda: not alive(one["pid"]), seconds=15)
    events = [e async for e in agent.stream("and?", user="u1", thread="t1")]
    [said] = warned(events)
    assert "t1 (research the market)" in said  # the task that ended with it is not lost in silence


async def test_closing_the_harness_closes_every_session_and_its_process(tmp_path: Path) -> None:
    scripts = [[BACKGROUND, {"text": "started it"}]]
    async with Harness(config=Settings()) as harness:
        agent = harness.wrap(options(tmp_path, scripts), id="closing")
        await agent.run("research the market", user="u1", thread="t1")
        await agent.run("research the market", user="u2", thread="t1")
        sessions = [live_of(agent, "u1"), live_of(agent, "u2")]
        pids = [turn["pid"] for turn in served(tmp_path)]
        assert len(set(pids)) == 2 and all(alive(pid) for pid in pids)
    assert open_sessions(agent) == 0
    assert all(s is not None and s.closed and s.client._query is None for s in sessions)
    assert not any(alive(pid) for pid in pids)  # no process is left behind


async def test_closing_the_harness_ends_every_process_and_part_even_when_the_run_store_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A session's close ends the background run that waits for a person (cancelled in the
    run store) before it ends its CLI: a run store that fails then (agent-runs unreachable)
    still leaves no process behind, and the harness still closes every other part — the
    queued writes, the spans, the MCP sources, its clients — before it raises that error
    (the first; any other is logged)."""
    from trellis.harness import telemetry

    store = LocalRuns()
    harness = Harness(config=Settings(), runs=store)
    scripts = [[BACKGROUND, {"text": "started it"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="closing")
    first = await agent.run("research the market", user="u1", thread="t1")
    child = await child_of(harness, first.run_id)
    touch = {"builtin": "Bash", "args": {"command": "touch notes.txt"}}
    (tmp_path / "tasks" / "t1.call").write_text(json.dumps(touch))
    await when(harness, child.run_id, RunStatus.PAUSED)  # its call waits for a person
    [one] = served(tmp_path)

    async def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("agent-runs is unreachable")

    monkeypatch.setattr(store, "cancel", unreachable)
    closed: list[str] = []
    writes = harness.writes.aclose
    monkeypatch.setattr(harness.writes, "aclose", lambda: (closed.append("writes"), writes())[1])

    async def unexported() -> None:
        closed.append("spans")
        raise RuntimeError("the collector is gone")

    monkeypatch.setattr(telemetry, "flush", unexported)
    try:
        with pytest.raises(ConnectionError, match="unreachable"):
            await harness.aclose()
        assert closed == ["writes", "spans"]  # the rest was closed all the same
        assert "closing the harness: the collector is gone" in caplog.text
        await until(lambda: not alive(one["pid"]))
        assert open_sessions(agent) == 0
    finally:
        if alive(one["pid"]):
            os.kill(one["pid"], signal.SIGKILL)


async def test_another_users_thread_of_the_same_name_has_a_session_of_its_own(
    harness: Harness, tmp_path: Path
) -> None:
    scripts = [[{"text": "noted"}], [{"text": "you told me: {heard}"}]]
    # with no harness tool at all, too (no ``trellis`` server to relay)
    agent = harness.wrap(options(tmp_path, scripts), id="two-users", without={"schedules"})
    await agent.run("the code is 4242", user="ada", thread="t1")
    await agent.run("the code is 9999", user="bob", thread="t1")
    hers = await agent.run("what is it?", user="ada", thread="t1")
    his = await agent.run("what is it?", user="bob", thread="t1")
    assert hers.answer == "you told me: the code is 4242 | what is it?"
    assert his.answer == "you told me: the code is 9999 | what is it?"
    ada, bob, *_ = served(tmp_path)
    assert ada["pid"] != bob["pid"] and open_sessions(agent) == 2
    assert live_of(agent, "ada") is not live_of(agent, "bob")


async def test_cancelling_a_turn_ends_it_at_its_next_call_and_leaves_the_background_work(
    harness: Harness, tmp_path: Path
) -> None:
    """The SDK's interrupt stops the session's background work with the turn (the bundled
    CLI's own rule), so a cancelled turn is not interrupted while any is under way: its run
    ends at once, and the CLI's turn at its next tool call, which is refused with the hook's
    ``continue: false`` — the task goes on, and the next turn is answered by the session."""
    scripts = [
        [BACKGROUND, {"text": "started it"}],
        [
            {"stall": 0.6},
            {"builtin": "Read", "args": {"file_path": "notes.txt"}},
            {"text": "never said"},
        ],
        [{"text": "after"}],
    ]
    agent = harness.wrap(options(tmp_path, scripts), id="cancelled")
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    slow = asyncio.create_task(agent.run("a long one", user="u1", thread="t1"))
    await until(lambda: len(served(tmp_path)) == 2)  # the CLI has its prompt
    await agent.cancel(turn_running(agent), reason="changed my mind")
    result = await slow
    assert result.status is RunStatus.CANCELLED and live.owed == 1  # the run ended first
    await until(lambda: live.owed == 0)  # then the CLI's turn, at its call
    assert built_ins(tmp_path) == []  # which did not run
    one, _ = served(tmp_path)
    assert live_of(agent) is live and list(live.working) == ["t1"] and alive(one["pid"])

    events = [e async for e in agent.stream("and now?", user="u1", thread="t1")]
    assert events[-1].data["result"] == "after"
    assert [turn["pid"] for turn in served(tmp_path)] == [one["pid"]] * 3
    assert list(live.working) == ["t1"]
    [said] = warned(events)
    assert "Read x1" in said  # the call the left turn made is among those refused


async def test_cancelling_a_turn_with_nothing_in_the_background_interrupts_it(
    harness: Harness, tmp_path: Path
) -> None:
    scripts = [[{"text": "hello"}], [{"stall": 30}, {"text": "never said"}], [{"text": "after"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="interrupted")
    await agent.run("hi", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    slow = asyncio.create_task(agent.run("a long one", user="u1", thread="t1"))
    await until(lambda: len(served(tmp_path)) == 2)
    [run_id] = agent.running
    await agent.cancel(run_id)
    assert (await slow).status is RunStatus.CANCELLED
    await until(lambda: live.owed == 0)  # interrupted: its result came, long before its stall
    after = await agent.run("and now?", user="u1", thread="t1")
    assert after.status is RunStatus.SUCCESS and after.answer == "after"
    one, *_ = served(tmp_path)
    assert [turn["pid"] for turn in served(tmp_path)] == [one["pid"]] * 3


async def test_cancelling_a_turn_ends_its_call_under_way_before_the_run_ends(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn cancelled while a harness tool of it is under way (a helper's run, say): the
    call is cancelled with the turn, and what it emits as it ends — a helper's
    ``helper_finished`` — is on the run's stream before its ``RUN_FINISHED``, never after:
    as in a process of its own, whose query ends its calls as it is closed (the feature
    matrix's X10 and X16 on the AG-UI and A2A surfaces, whose runs are a conversation's)."""
    emitted: list[Any] = []
    listening = RunEvents.__init__

    def tapped(events: RunEvents, *args: Any, **kwargs: Any) -> None:
        listening(events, *args, **kwargs)
        events.listen(emitted.append)

    monkeypatch.setattr(RunEvents, "__init__", tapped)
    started = asyncio.Event()

    @tool(side_effects="read")
    async def hold() -> str:
        """Wait for the record to come in."""
        runtime = current()
        assert runtime is not None
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            await asyncio.sleep(0)  # a call that needs a moment to end, as a helper's run does
            runtime.events.custom("hold_ended")
        return "held"

    scripts = [[{"tool": "hold"}, {"text": "never said"}], [{"text": "after"}]]
    agent = arm(harness.wrap(options(tmp_path, scripts), id="held"), [hold])
    turn = asyncio.create_task(agent.run("Look it up.", user="u1", thread="t1"))
    await asyncio.wait_for(started.wait(), 10)
    [run_id] = agent.running
    await agent.cancel(run_id, reason="the user left")
    assert (await turn).status is RunStatus.CANCELLED
    await asyncio.sleep(0.2)  # whatever would come late has come
    mine = [e for e in emitted if e.run_id == run_id]
    names = [e.data.get("name") if e.type is RunEventType.CUSTOM else e.type.value for e in mine]
    assert names[-1] == "RUN_FINISHED" and names.count("RUN_FINISHED") == 1, names
    assert "hold_ended" in names  # the call ended with the turn: before the run's end
    after = await agent.run("and now?", user="u1", thread="t1")
    assert after.status is RunStatus.SUCCESS and after.answer == "after", after.error
    assert len({turn["pid"] for turn in served(tmp_path)}) == 1  # the session went on


# --------------------------------------------------------------------------- a busy conversation
@pytest.mark.parametrize("mode", ["queue", "interrupt", "steer"])
async def test_a_message_to_a_busy_conversation_whose_session_is_live(
    harness: Harness, tmp_path: Path, mode: str
) -> None:
    """What a message does when its conversation's turn is under way (``mode``, as
    ``docs/runs.md`` says it), in a conversation whose live session has background work: it
    waits its turn (``queue``), takes the turn from the working run (``interrupt``) or is
    given to it after its tool call (``steer``) — and in each case the session, its process
    and its background task stay."""
    started, release = asyncio.Event(), asyncio.Event()

    @tool(side_effects="read")
    async def hold() -> str:
        """Wait for the record to come in."""
        started.set()
        await release.wait()
        return "held"

    scripts = [
        [BACKGROUND, {"text": "started it"}],
        [{"tool": "hold"}, {"text": "looked it up"}],
        [{"text": "you asked: {heard}"}],
    ]
    agent = arm(harness.wrap(options(tmp_path, scripts), id=f"busy-{mode}"), [hold])
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None and list(live.working) == ["t1"]
    first = asyncio.create_task(agent.run("Look it up.", user="u1", thread="t1"))
    await asyncio.wait_for(started.wait(), 10)
    second = asyncio.create_task(agent.run("Also say hello.", user="u1", thread="t1", mode=mode))
    await asyncio.sleep(0.3)  # it arrives while the first works
    heard = "you asked: research the market | Look it up. | Also say hello."
    if mode == "interrupt":
        a = await asyncio.wait_for(first, 10)
        assert a.status is RunStatus.CANCELLED  # the newer message took over
        release.set()
        b = await asyncio.wait_for(second, 10)
        # the session went on from the cancelled turn: its "model" heard that question
        assert b.status is RunStatus.SUCCESS and b.answer == heard, b.error
    else:
        assert not second.done()  # it waits, or the first run took it
        release.set()
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 10)
        assert a.status is RunStatus.SUCCESS and a.answer == "looked it up", a.error
        assert b.status is RunStatus.SUCCESS and b.run_id != a.run_id, b.error
        if mode == "queue":
            assert b.answer == heard  # its own turn, after the first
        else:
            # given to the working turn after its tool call (the ``PostToolUse`` hook of the
            # turn being served, through the session's relay): that turn's answer is its own
            assert b.answer == a.answer and len(served(tmp_path)) == 2
            added = (tmp_path / "cli.json.added").read_text()
            assert "Also say hello." in added
    turns = served(tmp_path)
    assert len(turns) == (2 if mode == "steer" else 3)
    assert len({turn["pid"] for turn in turns}) == 1 and alive(turns[0]["pid"])  # one process
    assert live_of(agent) is live and not live.closed and list(live.working) == ["t1"]


# --------------------------------------------------------------------------- memory
#: What the memory service answers each turn of the chat below (``FakeMemoryService.context_text``).
CONTEXTS = ("FIRST: the user is Ada.", "SECOND: the user prefers email.", "THIRD: the user moved.")


def given(turn: dict[str, Any]) -> list[str]:
    """Which of :data:`CONTEXTS` the CLI recorded its ``UserPromptSubmit`` hooks adding."""
    return [text for text in CONTEXTS if text in turn["context"]]


async def test_each_turn_of_a_live_session_is_given_its_own_memory_context(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """The process registers its ``UserPromptSubmit`` hook once, in the first turn, whose
    context that hook closed over: the relay hands each prompt to the hook of the turn being
    served, so the second turn's prompt carries the second turn's context — not the first's —
    and the third's, after the session closed and a new process resumed it, its own."""
    scripts = [[{"text": "one: {context}"}], [{"text": "two: {context}"}], [{"text": "three"}]]
    agent = memory_harness.wrap(options(tmp_path, scripts), id="context")
    memory_service.context_text = CONTEXTS[0]
    first = await agent.run("who am I?", user="u1", thread="t1")
    memory_service.context_text = CONTEXTS[1]
    second = await agent.run("reach me how?", user="u1", thread="t1")
    assert first.status is RunStatus.SUCCESS and second.status is RunStatus.SUCCESS, second.error
    one, two = served(tmp_path)
    assert two["pid"] == one["pid"] and two["resume"] is None  # served by the live session
    assert (given(one), given(two)) == ([CONTEXTS[0]], [CONTEXTS[1]])
    assert CONTEXTS[1] in second.answer and CONTEXTS[0] not in second.answer  # what it read

    adapter = agent.adapter
    assert isinstance(adapter, ClaudeAdapter)
    # the conversation's key carries the tenant the memory service's key says
    key = conversation_key(str(memory_service.tenant), "u1", "t1")
    assert adapter.live.get(key) is not None
    await adapter.live.close(key, "it was idle")
    await until(lambda: not alive(one["pid"]))
    memory_service.context_text = CONTEXTS[2]
    third = await agent.run("where do I live?", user="u1", thread="t1")
    assert third.status is RunStatus.SUCCESS and third.answer == "three", third.error
    *_, three = served(tmp_path)
    assert three["pid"] != one["pid"] and three["resume"] == one["session"]  # the fallback
    assert given(three) == [CONTEXTS[2]]


async def test_with_memory_on_each_live_turn_asks_memory_first_and_is_saved_after(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Memory on, end to end: every turn of the conversation — the one that starts the
    process and the ones it serves live — asks the memory service for its context before it
    runs, with its own question, and its transcript is saved once it ended."""
    scripts = [[{"text": "noted"}], [{"text": "4"}], [{"text": "yes"}]]
    agent = memory_harness.wrap(options(tmp_path, scripts), id="remembered")
    questions = ["the code is 4242", "what is 2+2?", "are you sure?"]
    for number, question in enumerate(questions, start=1):
        done = await agent.run(question, user="u1", thread="t1")
        assert done.status is RunStatus.SUCCESS, done.error
        await memory_harness.writes.drain()
        # asked once more, before this turn's prompt reached the CLI; saved once more, after
        assert len(memory_service.named("context")) == number
        assert len(memory_service.named("messages")) == number
    assert len({turn["pid"] for turn in served(tmp_path)}) == 1 and open_sessions(agent) == 1
    asked = [call.body["query"] for call in memory_service.named("context")]
    assert asked == questions
    saved = [
        [(m["role"], m["content"]) for m in call.body["messages"]]
        for call in memory_service.named("messages")
    ]
    assert saved == [
        [("USER", "the code is 4242"), ("ASSISTANT", "noted")],
        [("USER", "what is 2+2?"), ("ASSISTANT", "4")],
        [("USER", "are you sure?"), ("ASSISTANT", "yes")],
    ]
    names = [call.name for call in memory_service.calls if call.name in ("context", "messages")]
    assert names == ["context", "messages"] * 3  # each turn: asked, then saved


# --------------------------------------------------------------------------- the tools
def test_the_task_tools_the_bundled_cli_exposes_have_a_declared_risk() -> None:
    """Named as the bundled CLI (2.1.283) lists them when asked (its ``init`` message), each
    by what it does: stopping a task is a write; tools that only list or look up, read-only by
    the CLI's own word, are reads. ``TaskOutput`` and ``Monitor`` are not tools it lists."""
    assert BUILTIN_SIDE_EFFECTS["TaskStop"] == "write"
    looked_up = ("ListAgents", "ListConnectors", "ReadNotifications", "ReportFindings")
    assert {BUILTIN_SIDE_EFFECTS[name] for name in looked_up} == {"read"}
    assert not {"TaskOutput", "Monitor"} & set(BUILTIN_SIDE_EFFECTS)


async def test_schedules_stay_disallowed_in_a_live_session(
    harness: Harness, tmp_path: Path
) -> None:
    """A live session outlives its turn, but not for good, and a turn a schedule fires has no
    run to check its calls or anyone to answer: Claude Code's schedules stay off, as in a
    query's own process."""
    agent = harness.wrap(options(tmp_path, [[{"text": "ok"}]]), id="no-cron-live")
    await agent.run("every Monday", user="u1", thread="t1")
    await agent.run("every Monday", user="u1")
    kept, alone = served(tmp_path)
    assert kept["pid"] != alone["pid"]
    for started in (kept, alone):
        assert set(OUTLIVES_THE_QUERY) <= set(started["disallowed_tools"].split(","))


def test_the_frames_the_stand_in_sends_are_the_sdks_task_messages() -> None:
    """The background frames of the stand-in CLI, parsed by the SDK's own parser."""
    started = parse_message(
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": "t1",
            "tool_use_id": "toolu_bg_t1",
            "description": "research the market",
            "task_type": "local_agent",
            "uuid": "u",
            "session_id": "s",
        }
    )
    assert isinstance(started, TaskStartedMessage) and started.task_type == "local_agent"
