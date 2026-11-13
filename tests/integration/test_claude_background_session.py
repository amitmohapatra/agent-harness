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

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from trellis import Deny, Harness, Hooks, Settings, current, tool
from trellis.contracts import RunEventType, RunStatus, ToolCall
from trellis.harness import budget
from trellis.harness.adapters.claude import (
    BUILTIN_SIDE_EFFECTS,
    OUTLIVES_THE_QUERY,
    RESUMED_CALL,
    ClaudeAdapter,
)
from trellis.harness.agent import Agent
from trellis.harness.bridges import claude_live
from trellis.harness.bridges.claude_live import UNATTENDED, LiveSession
from trellis.harness.identity import conversation_key

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


async def test_what_background_work_spent_between_turns_is_on_the_next_turn(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counted: list[dict[str, Any]] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (counted.append(kw), account(run, **kw))
    )
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    target = options(tmp_path, scripts, model="m", env={"FAKE_CLAUDE_COST": "0.5"})
    agent = harness.wrap(target, id="spent")
    await agent.run("research the market", user="u1", thread="t1")
    live = live_of(agent)
    assert live is not None
    (tmp_path / "tasks" / "t1.done").write_text("the market is growing")
    await until(lambda: any(isinstance(m, ResultMessage) for m in live.kept))
    await agent.run("thanks", user="u1", thread="t1")
    # two steps; then one of its own, at the cost of three: the task's own step and the
    # session's own turn for it were paid for between the turns
    assert [(c["input_tokens"], c["cost"]) for c in counted] == [(20, 1.0), (10, 1.5)]


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


async def test_at_its_turn_limit_a_conversations_turn_gets_its_last_answer_from_a_new_process(
    harness: Harness, tmp_path: Path
) -> None:
    """The one last answer at the turn limit is a query of other options (no tool, one
    turn): it runs in a process of its own, the live session closed first — two processes do
    not hold one session."""
    script = [
        {"builtin": "Read", "args": {"file_path": "a.txt"}},
        {"builtin": "Read", "args": {"file_path": "b.txt"}},
        {"text": "both read"},
    ]
    agent = harness.wrap(options(tmp_path, [script], max_turns=1), id="limited")
    result = await agent.run("read both", user="u1", thread="t1")
    assert result.status is RunStatus.SUCCESS and result.answer == "What I found before the limit."
    one, two = served(tmp_path)
    assert two["pid"] != one["pid"] and two["resume"] == one["session"]
    await until(lambda: not alive(one["pid"]) and not alive(two["pid"]))
    assert open_sessions(agent) == 0


async def test_a_session_whose_process_dies_fails_its_turn_and_the_next_turn_starts_another(
    harness: Harness, tmp_path: Path
) -> None:
    scripts = [[{"text": "noted"}], [{"stall": 30}, {"text": "never said"}], [{"text": "back"}]]
    agent = harness.wrap(options(tmp_path, scripts), id="died")
    await agent.run("the code is 4242", user="u1", thread="t1")
    slow = asyncio.create_task(agent.run("a long one", user="u1", thread="t1"))
    await until(lambda: len(served(tmp_path)) == 2)
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
    [run_id] = agent.running
    await agent.cancel(run_id)
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


async def test_a_tool_call_of_background_work_runs_only_while_a_run_can_check_it(
    harness: Harness, tmp_path: Path
) -> None:
    """Between two turns no run is under way: a call the background task makes then is
    refused, not run unchecked, and the next run says so. One it makes while a turn is being
    answered is decided and recorded for that turn's run."""
    scripts = [
        [BACKGROUND, {"text": "started it"}],
        [{"stall": 0.8}, {"text": "done"}],
    ]
    agent = harness.wrap(options(tmp_path, scripts), id="unattended")
    await agent.run("research the market", user="u1", thread="t1")
    tasks = tmp_path / "tasks"
    reads = json.dumps({"builtin": "Read", "args": {"file_path": "notes.txt"}})
    (tasks / "t1.call").write_text(reads)
    await until((tasks / "t1.called").exists)
    assert json.loads((tasks / "t1.called").read_text()) == {"result": UNATTENDED, "ok": False}
    assert built_ins(tmp_path) == []
    (tasks / "t1.called").unlink()
    live = live_of(agent)
    assert live is not None and list(live.working) == ["t1"]  # refused, the task goes on

    async def during_the_turn() -> None:
        await until(lambda: len(served(tmp_path)) == 2)
        (tasks / "t1.call").write_text(reads)

    calling = asyncio.create_task(during_the_turn())
    events = [e async for e in agent.stream("how is it going?", user="u1", thread="t1")]
    await calling
    assert events[-1].data["result"] == "done"
    assert json.loads((tasks / "t1.called").read_text()) == {"result": "Read ran", "ok": True}
    assert [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT] == ["Read"]
    [said] = warned(events)
    assert "were refused" in said and "Read x1" in said


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
    await until(lambda: not alive(one["pid"]))
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
    [run_id] = agent.running
    await agent.cancel(run_id, reason="changed my mind")
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
