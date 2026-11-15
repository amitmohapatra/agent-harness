"""A live Claude session's rules where no process is needed to show them
(``bridges.claude_live``): what a request gets when no turn can take it, what a turn is
charged of the process's running totals, and what a background subagent's run does with
what comes while no attempt of it is under way."""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher, PermissionResultDeny
from claude_agent_sdk._internal.message_parser import parse_message

from trellis.contracts import RunStatus
from trellis.harness.bounded import LRU
from trellis.harness.bridges import claude_live, live
from trellis.harness.bridges.claude import harness_matcher
from trellis.harness.bridges.claude_live import (
    GONE,
    GONE_SAID,
    LEFT,
    STOPPED,
    WORKFLOW_AGENT,
    Background,
    LiveSession,
)
from trellis.harness.bridges.helpers import Kind
from trellis.harness.bridges.live import NUDGE, OVER, UNATTENDED, Left, LiveSessions, Turn
from trellis.harness.holding import Holds
from trellis.harness.journal import Journal
from trellis.harness.runs import LocalRuns
from trellis.harness.tools.convert import claude as convert

REFUSED = {"content": [{"type": "text", "text": UNATTENDED}], "is_error": True}


def refusal(answer: Any) -> tuple[str, str]:
    said = answer["hookSpecificOutput"]
    return said["permissionDecision"], said["permissionDecisionReason"]


async def test_with_no_turn_to_take_it_a_tool_call_is_refused_and_counted() -> None:
    """Between two turns nothing can check a call: whichever way the CLI asks — the
    harness's ``PreToolUse`` hook, its permission check, a harness tool's call — the answer
    is no (a hook of the team's own runs then as Claude Code runs it: the integration
    test)."""

    async def never(given: Any, tool_use_id: str | None, context: Any) -> Any:
        raise AssertionError("a hook of no turn's")

    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    hooks = session._hooks(
        {"PreToolUse": [harness_matcher(never)], "Stop": [harness_matcher(never)]}
    )
    asked = await hooks["PreToolUse"][0].hooks[0]({"tool_name": "Bash"}, "toolu_1", None)
    assert refusal(asked) == ("deny", UNATTENDED)
    assert await hooks["Stop"][0].hooks[0]({}, None, None) == {}  # not a tool call: it passes
    denied = await session._can_use_tool("Bash", {"command": "rm -rf build"}, None)
    assert isinstance(denied, PermissionResultDeny) and denied.message == UNATTENDED
    assert await session._call("refund", {"order": "o1"}) == REFUSED
    assert session.refused == {"Bash": 2, "refund": 1}


async def test_the_main_agents_call_in_a_turn_that_was_left_ends_that_turn() -> None:
    """A left turn's result is still to come: the main agent's next call ends the turn — the
    hook's ``continue: false``, the permission check's ``interrupt``, which leave background
    work as it is. A subagent's call (the CLI names it) is refused and no more."""

    async def never(given: Any, tool_use_id: str | None, context: Any) -> Any:
        raise AssertionError("a hook of no turn's")

    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    session.owed = 1
    relay = session._hooks({"PreToolUse": [harness_matcher(never)]})["PreToolUse"][0].hooks[0]
    ended = await relay({"tool_name": "Bash"}, "toolu_1", None)
    assert refusal(ended) == ("deny", LEFT)
    assert ended["continue_"] is False and ended["stopReason"] == LEFT
    theirs = await relay({"tool_name": "Bash", "agent_id": "a1"}, "toolu_2", None)
    assert refusal(theirs) == ("deny", UNATTENDED) and "continue_" not in theirs
    denied = await session._can_use_tool("Bash", {}, SimpleNamespace(agent_id=None))
    assert (denied.message, denied.interrupt) == (LEFT, True)
    denied = await session._can_use_tool("Bash", {}, SimpleNamespace(agent_id="a1"))
    assert (denied.message, denied.interrupt) == (UNATTENDED, False)


async def test_a_request_under_way_when_its_turn_is_left_is_refused() -> None:
    """A turn that is left (cancelled, paused) takes its run with it: what the CLI had asked
    it and was still waiting for is cancelled, and answered as with no turn."""
    never = asyncio.Event()

    async def hook(given: Any, tool_use_id: str | None, context: Any) -> Any:
        await never.wait()

    async def can_use_tool(name: str, args: dict[str, Any], context: Any) -> Any:
        await never.wait()

    options = SimpleNamespace(
        hooks={"PreToolUse": [HookMatcher(hooks=[hook])]},
        can_use_tool=can_use_tool,
        session_store=None,
    )
    runtime: Any = SimpleNamespace(agent=SimpleNamespace(sessions=LRU()))
    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    turn = Turn(runtime, options, None, contextvars.copy_context())
    session._attach(turn)
    relay = session._hooks(options.hooks)["PreToolUse"][0].hooks[0]
    asked = asyncio.gather(
        relay({"tool_name": "Bash"}, "toolu_1", None),
        session._can_use_tool("Bash", {"command": "make"}, None),
    )
    await asyncio.sleep(0.01)
    assert len(turn.calls) == 2
    session._detach(turn)
    hooked, denied = await asked
    assert refusal(hooked) == ("deny", UNATTENDED)
    assert isinstance(denied, PermissionResultDeny) and denied.message == UNATTENDED
    await session.close()
    await session.close()  # closed once: a second time is no change
    assert session.closed and session._timer is None


def result(cost: float | None, **usage: Any) -> Any:
    """A result as the bundled CLI sends one (its usage holds more than token counts)."""
    return parse_message(
        {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 1,
            "session_id": "s1",
            "result": "ok",
            "total_cost_usd": cost,
            "usage": {**usage, "server_tool_use": {"web_search_requests": 0}},
        }
    )


def test_a_turn_is_charged_what_the_processs_running_cost_grew_by() -> None:
    """As the bundled CLI reports: a result's cost is its process's running total, its usage
    the turn's own."""
    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    first = session.charged(result(0.0012, input_tokens=200, output_tokens=40))
    assert (first.total_cost_usd, first.usage["input_tokens"]) == (0.0012, 200)
    second = session.charged(result(0.0016, input_tokens=100, output_tokens=20))
    assert round(second.total_cost_usd, 6) == 0.0004
    assert (second.usage["input_tokens"], second.usage["output_tokens"]) == (100, 20)
    # a total that did not grow is no running total: taken as the turn's own
    third = session.charged(result(0.001, input_tokens=40, output_tokens=5))
    assert third.total_cost_usd == 0.001
    assert session.charged(result(None, input_tokens=50, output_tokens=6)).total_cost_usd is None


# --------------------------------------------------------------------------- a background run
class Store:
    """The run store of a background run's agent (``agent.harness.runs``): the run's first
    read — its own attempt's — waits until ``broken`` is set and then fails, as a store that
    cannot be reached does; the others read ``records``."""

    def __init__(self) -> None:
        self.broken = asyncio.Event()
        self.records: dict[str, Any] = {}
        self.cancelled: list[tuple[str, str]] = []
        self.on_cancel: Any = None
        self._first = True

    async def get(self, run_id: str, tenant: str | None = None) -> Any:
        if self._first:
            self._first = False
            await self.broken.wait()
            raise ConnectionError("the run store cannot be reached")
        return self.records.get(run_id)

    async def heartbeat(self, run_id: str, worker_id: str, **given: Any) -> None:
        """A hold (``holding``): taken."""

    async def release(self, run_id: str, worker_id: str, **given: Any) -> None:
        """A hold let go of."""

    async def cancel(self, run_id: str, *, reason: str, tenant: str | None = None) -> None:
        self.cancelled.append((run_id, reason))
        if self.on_cancel is not None:
            self.on_cancel()


def runtime_of(agent: Any, run_id: str) -> Any:
    return SimpleNamespace(
        agent=agent,
        tenant="default",
        framework_options={},
        run_id=run_id,
        parent=None,
        budget=None,
        budget_spent=None,
        pending=None,
        cancelled=None,
        replay=SimpleNamespace(journal=Journal()),
    )


def turn_of(agent: Any, run_id: str) -> Turn:
    return Turn(
        runtime_of(agent, run_id), SimpleNamespace(model="m"), None, contextvars.copy_context()
    )


async def background() -> tuple[LiveSession, Background, Store]:
    """A session with one background subagent's run (``child``, task ``t1``), started by the
    turn ``turn``, whose own attempt has not begun (its record is still being read)."""
    store = Store()
    agent = SimpleNamespace(
        harness=SimpleNamespace(runs=store, holds=Holds(lambda: store)),
        running={},
        sessions=LRU(),
        adapter=SimpleNamespace(name="claude"),
    )
    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    session.turn = turn_of(agent, "turn")
    helper = SimpleNamespace(
        run_id="child", parent_run_id="turn", name="researcher", began=0.0, kind=None
    )
    found = Background(session, "t1", "toolu_bg_t1", helper)
    session.turn = None
    session.background["t1"] = found
    await asyncio.sleep(0)  # its attempt is reading the record
    return session, found, store


async def ended(session: LiveSession, found: Background, store: Store) -> None:
    """The store fails the run's read: the run ends (logged), the session closes."""
    store.broken.set()
    await asyncio.wait({found.task})
    assert found.over and "child" not in session.sessions.attended
    await session.close()


def reply(text: str, usage: dict[str, Any] | None, message_id: str | None) -> Any:
    from claude_agent_sdk import AssistantMessage, TextBlock

    return AssistantMessage(
        [TextBlock(text)], "m", parent_tool_use_id="toolu_bg_t1", usage=usage, message_id=message_id
    )


async def test_what_comes_while_no_attempt_of_a_background_run_is_under_way_waits_for_one() -> None:
    """The subagent's messages, what its replies used and its task's end, come while no
    attempt of its run is under way: each is kept, and the attempt that follows is served
    them in order, charged the replies (each counted once) and ends with the task's end."""
    session, found, store = await background()
    used = {"input_tokens": 10, "output_tokens": 2}
    first, again = reply("looking", used, "msg_1"), reply("looking", used, "msg_1")
    unnamed, quiet = reply("found it", used, None), reply("", None, None)
    for message in (first, again, unnamed, quiet):
        found.took(message)
    found.finished("completed", "")
    assert found.said == "found it" and found.unbilled == 24  # msg_1 once, the unnamed one
    assert len(found._owed) == 2 and found.turn is None

    store.records["child"] = SimpleNamespace(final=False)
    found._settled.set()  # what the task cost is known
    turn = turn_of(found.agent, "child")
    served = [m async for m in found.served(turn)]
    assert served[:4] == [first, again, unnamed, quiet]
    assert served[4].result == "found it" and not served[4].is_error
    spent = turn.runtime.replay.journal.spent
    assert (spent["tokens"], spent["model_calls"]) == (24, 2)
    await ended(session, found, store)


async def test_an_attempt_that_pauses_keeps_what_comes_after_and_ends_its_requests() -> None:
    """An attempt nudged with nothing new goes on; once its run pauses, what comes is kept
    for the attempt that follows, and the requests still under way end with the attempt."""
    session, found, store = await background()
    store.records["child"] = SimpleNamespace(final=False)
    turn = turn_of(found.agent, "child")
    attempt = found.served(turn)
    waiting = asyncio.ensure_future(anext(attempt))
    await asyncio.sleep(0)
    turn.queue.put_nowait(NUDGE)  # nothing new: no end of its task yet
    first = reply("one", None, None)
    turn.queue.put_nowait(first)
    assert await waiting is first
    never = asyncio.Event()
    asked = asyncio.ensure_future(turn.within(never.wait()))
    await asyncio.sleep(0)
    turn.runtime.pending = "paused"
    later = reply("two", None, None)
    turn.queue.put_nowait(later)
    assert [m async for m in attempt] == []  # it ended paused
    assert found._backlog == [later] and found.turn is None
    with pytest.raises(Left):
        await asked  # cancelled with the attempt
    await ended(session, found, store)


async def test_an_attempt_left_with_a_request_under_way_ends_it_first() -> None:
    """An attempt closed while it is served (its run cancelled) cancels the requests under
    way and waits for them before it ends."""
    session, found, store = await background()
    store.records["child"] = SimpleNamespace(final=False)
    turn = turn_of(found.agent, "child")
    attempt = found.served(turn)
    waiting = asyncio.ensure_future(anext(attempt))
    await asyncio.sleep(0)
    message = reply("one", None, None)
    turn.queue.put_nowait(message)
    assert await waiting is message
    never = asyncio.Event()
    asked = asyncio.ensure_future(turn.within(never.wait()))
    await asyncio.sleep(0)
    await attempt.aclose()
    assert found.turn is None and not turn.calls
    with pytest.raises(Left):
        await asked
    await ended(session, found, store)


async def test_a_background_run_that_cannot_go_on_says_why_and_asks_nothing_more() -> None:
    """A run given up (its session closed, its task stopped) fails its next attempt with the
    reason; a request whose attempt is left waits for the next, and once the run is over is
    answered as with no run — a hook, a harness tool's call — and refused."""
    session, found, store = await background()
    nobody: Any = SimpleNamespace()
    found.answered(nobody)  # nothing waits for an answer: no change
    found.cancel("the turn was cancelled")  # no attempt under way to cancel
    assert found.stopped == "the turn was cancelled"
    await found.abandoned("its session was closed")  # no attempt, no question: only said
    assert store.cancelled == []
    with pytest.raises(RuntimeError, match="its session was closed"):
        await anext(found.served(turn_of(found.agent, "child")))

    turn = turn_of(found.agent, "child")
    found.turn = turn
    never = asyncio.Event()

    async def make(at: Turn) -> Any:
        await never.wait()

    asked = asyncio.ensure_future(found.asked(make))
    await asyncio.sleep(0)
    found._detach(turn)  # the attempt ended: the request waits for the next
    await asyncio.sleep(0)
    assert not asked.done()
    found.over = True
    found._woken()
    assert await asked is OVER

    async def hook(given: Any, tool_use_id: str | None, context: Any) -> Any:
        raise AssertionError("a hook of a run that is over")

    relay = session._hooks({"PreToolUse": [harness_matcher(hook)]})["PreToolUse"][0].hooks[0]
    hooked = await relay({"tool_name": "Read", "agent_id": "t1"}, "toolu_1", None)
    assert refusal(hooked) == ("deny", UNATTENDED)
    session._calling["toolu_2"] = found

    async def served(ctx: Any) -> dict[str, Any]:
        return await session._call("refund", {"order": "o1"})

    naming = SimpleNamespace(meta={convert.CALL_META: "toolu_2"})
    assert await convert.named_call(naming, served) == REFUSED
    # a hook that asks (neither lets it through nor refuses it) says nothing of whose it is
    given = {"tool_name": "mcp__trellis__refund", "tool_use_id": "toolu_3"}
    session._noted(
        "PreToolUse", given, {"hookSpecificOutput": {"permissionDecision": "ask"}}, found
    )
    assert "toolu_3" not in session._calling
    await ended(session, found, store)


async def test_a_task_that_ends_while_its_run_waits_for_a_person_cancels_the_run() -> None:
    """The task a run waits about ends (stopped in Claude Code): the run is cancelled in the
    store, saying so, and stops waiting — also when the store's cancel ended the wait."""
    session, found, store = await background()
    loop = asyncio.get_running_loop()
    found._answer = loop.create_future()
    found.finished("killed", "")
    await asyncio.wait(session._aside)
    assert store.cancelled == [("child", STOPPED)] and found._answer.result() is None

    found._answer = waits = loop.create_future()
    store.on_cancel = lambda: waits.set_result(None)  # the run ended meanwhile
    await found.abandoned("nobody answered")
    assert waits.result() is None and store.cancelled[-1] == ("child", "nobody answered")
    await ended(session, found, store)


async def test_a_question_of_a_run_cancelled_where_it_waited_is_not_asked() -> None:
    """The conversation's next turn asks the questions its background runs wait on: one
    whose run is no longer paused (cancelled where it waited) is not asked."""
    session, found, store = await background()
    question: Any = SimpleNamespace(run_id="child", interrupt_id="i1")
    found.question = question
    store.records["child"] = SimpleNamespace(status=RunStatus.CANCELLED)
    await session.asks(runtime_of(found.agent, "turn2"))  # it would pause: it does not
    await ended(session, found, store)


async def test_the_share_of_a_turn_no_run_is_served_for_stays_for_the_next_turn() -> None:
    """A result of a turn the session ran by itself, no run served and no task's end to
    charge it to: each background run is charged its replies' share, and the turn's own
    share is not billed — the conversation's next turn is charged it."""
    session, found, store = await background()
    found.unbilled = 100
    charged = session.charged(
        result(0.003, input_tokens=150, output_tokens=50), own=False, taken=False
    )
    assert charged.total_cost_usd == 0.003  # the result as it came
    [(usage, share)] = found._owed  # no attempt under way: the next one's
    assert usage is None and round(share, 6) == 0.001
    assert round(session.cost.total, 6) == 0.001 and found.unbilled == 0
    await ended(session, found, store)


async def test_an_answer_its_session_is_not_here_to_take_ends_the_background_run() -> None:
    """``gone``: the answer is recorded on the run, which ends ``ERROR`` (a ``CANCEL`` ends it
    ``CANCELLED`` already), and the turn is told; a run no longer waiting on that question is
    left as it is, and nothing is said."""
    from trellis.contracts import Interrupt, InterruptDecision, InterruptResolution, RunStart

    runs, said = LocalRuns(), []
    runtime: Any = SimpleNamespace(
        agent=SimpleNamespace(
            harness=SimpleNamespace(runs=runs), adapter=SimpleNamespace(name="claude_agent_sdk")
        ),
        tenant="t",
        events=SimpleNamespace(warning=lambda code, message: said.append((code, message))),
    )

    async def waiting(run_id: str) -> Interrupt:
        await runs.start(RunStart(run_id=run_id, tenant_id="t", agent_id="a"))
        asked = Interrupt(tenant_id="t", run_id=run_id, question="refund?")
        await runs.pause(asked)
        return asked

    def answer(decision: InterruptDecision) -> InterruptResolution:
        return InterruptResolution(interrupt_id="turn.1", run_id="turn", decision=decision)

    async def gone(*given: Any) -> None:  # as the Claude adapter calls it
        await live.gone(*given, why=GONE, said=GONE_SAID, code="claude_session")

    asked = await waiting("run_approved")
    await gone(runtime, "run_approved", asked.interrupt_id, answer(InterruptDecision.APPROVE))
    ended = await runs.get("run_approved")
    assert ended is not None and ended.status is RunStatus.ERROR and ended.error is not None
    assert (ended.error.code, ended.error.message) == ("holder_lost", GONE)
    assert said == [("claude_session", GONE_SAID)]

    asked = await waiting("run_cancelled")
    await gone(runtime, "run_cancelled", asked.interrupt_id, answer(InterruptDecision.CANCEL))
    cancelled = await runs.get("run_cancelled")
    assert cancelled is not None and cancelled.status is RunStatus.CANCELLED

    said.clear()
    await gone(runtime, "run_approved", asked.interrupt_id, answer(InterruptDecision.APPROVE))
    assert said == []  # it waits on nothing any more


def test_the_harness_tools_a_live_session_lists_are_the_same_in_every_turn() -> None:
    """A live session's process keeps what its ``trellis`` server listed in the first turn
    (``adapters.claude._shape``: each tool's name and input schema, ``convert.listing``); a
    later turn is served by it only if it lists the same. The schedule tools, every parameter
    described, list alike each turn."""
    from trellis.harness.schedules import SPECS
    from trellis.harness.tools.base import Tool

    async def run(args: dict[str, Any]) -> Any:
        return None

    def listed() -> Any:
        return convert.listing(convert.convert([Tool(spec, run) for spec in SPECS]))

    first = listed()
    assert first == listed()
    [update] = [schema for name, schema in first if name == "update_schedule"]
    assert '"enabled": {"description": "false pauses it, true resumes it."' in update


async def test_a_workflow_agents_call_is_the_workflows_run_only_while_one_works() -> None:
    """A call named by a workflow agent's kind (``workflow-subagent``) is the run of the one
    workflow at work in the session — remembered by the agent's own id for its permission
    request, which names the agent alone — and no background run's while none works (a
    subagent's run is never taken for it)."""
    session, found, store = await background()
    assert await session._working("a1", WORKFLOW_AGENT) is None  # t1 is a subagent's
    found.kind = Kind.WORKFLOW
    assert await session._working("a1", WORKFLOW_AGENT) is found
    assert await session._working("a1") is found  # its permission request: the id alone
    session._forget(found)
    assert await session._working("a1") is None and not session.background
    session.background["t1"] = found
    await ended(session, found, store)


async def test_a_harness_tools_call_is_the_run_its_id_names_not_the_first_that_matches() -> None:
    """ARCH-16: two background runs make the same harness tool call with the same arguments.
    The request Claude Code makes names the call (its ``_meta``, the id its hook carried), so
    each is its own run's whichever arrives first — not the first noted that matches."""
    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    took: list[str] = []

    def run(label: str) -> Any:
        async def asked(make: Any) -> dict[str, Any]:
            took.append(label)
            return {"content": []}

        return SimpleNamespace(asked=asked)

    allow = {"hookSpecificOutput": {"permissionDecision": "allow"}}
    for ref, label in [("toolu_a", "a"), ("toolu_b", "b")]:
        call = {"tool_name": "mcp__trellis__refund", "tool_input": {"order": "o1"}}
        session._noted("PreToolUse", {**call, "tool_use_id": ref}, allow, run(label))

    async def served(ctx: Any) -> dict[str, Any]:
        return await session._call("refund", {"order": "o1"})

    for ref in ("toolu_b", "toolu_a"):
        await convert.named_call(SimpleNamespace(meta={convert.CALL_META: ref}), served)
    assert took == ["b", "a"]
    # a request that names no call is no background run's: the turn's own, or refused
    assert await session._call("refund", {"order": "o1"}) == REFUSED


# --------------------------------------------------------------------------- room (PERF-3)
def _rooms(monkeypatch: pytest.MonkeyPatch, cap: int) -> list[str]:
    """Sessions that start no process: the keys ``start`` was called for; the cap is ``cap``."""
    started: list[str] = []

    async def start(self: LiveSession, options: Any, transcripts: Any) -> None:
        started.append(self.key)
        await asyncio.sleep(0)  # the process starts: other turns' opens run meanwhile

    async def close(self: LiveSession) -> None:
        self.closed = True

    monkeypatch.setattr(LiveSession, "start", start)
    monkeypatch.setattr(LiveSession, "close", close)
    monkeypatch.setattr(live, "KEPT", cap)
    return started


async def test_first_turns_at_once_start_no_more_processes_than_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERF-3: a hundred conversations' first turns at once. The place is taken before the
    process starts, so no more than the cap are started or kept — over one agent's sessions
    and another's of the same server alike; the rest run without one (``None``)."""
    started = _rooms(monkeypatch, 16)
    one, two = LiveSessions(), LiveSessions()
    among = [one, two]
    opened = await asyncio.gather(
        *(
            claude_live.opened(
                one if n % 2 else two, f"c{n}", (), None, {}, owner=f"u{n}", among=among
            )
            for n in range(100)
        )
    )
    assert len(started) == 16 and len(one) + len(two) == 16
    assert sum(found is not None for found in opened) == 16
    for found in opened:  # each of them serves a turn: none is idle to give its place
        if found is not None:
            found.turn = SimpleNamespace()  # type: ignore[assignment]
    assert await claude_live.opened(one, "late", (), None, {}, owner="late", among=among) is None
    assert len(started) == 16


async def test_a_place_goes_from_who_holds_most_the_one_used_longest_ago(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERF-3, fairness: with every place taken, a newcomer takes the idle session of the
    user who holds the most — of theirs, the one used longest ago — so one user's many
    conversations cost that user a place, not another's only one. A session at work (a turn,
    background work) keeps its place."""
    _rooms(monkeypatch, 4)
    sessions = LiveSessions()
    among = [sessions]
    for key, owner in [("b1", "bob"), ("a1", "ada"), ("a2", "ada"), ("a3", "ada")]:
        assert (
            await claude_live.opened(sessions, key, (), None, {}, owner=owner, among=among)
            is not None
        )
    assert sessions.get("a1") is not None  # used again: a2 is ada's least recently used
    assert (
        await claude_live.opened(sessions, "c1", (), None, {}, owner="cy", among=among) is not None
    )
    assert sorted(s.key for s in among[0]) == ["a1", "a3", "b1", "c1"]
    held = sessions.get("a1")
    assert held is not None
    held.working["t1"] = "research"  # at work in the background: not given up
    assert (
        await claude_live.opened(sessions, "c2", (), None, {}, owner="cy", among=among) is not None
    )
    assert sorted(s.key for s in among[0]) == ["a1", "b1", "c1", "c2"]  # a3 went, not a1


async def test_one_user_holds_at_most_their_share_of_the_places(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One user's conversations, each at work in the background (none idle), take at most
    ``KEPT_PER_OWNER`` places, however many places are free: the next of theirs runs without
    a session, or takes their own idle one, and another user still finds a place."""
    _rooms(monkeypatch, 16)
    monkeypatch.setattr(live, "KEPT_PER_OWNER", 3)
    sessions = LiveSessions()
    among = [sessions]
    opened = await asyncio.gather(
        *(
            claude_live.opened(sessions, f"a{n}", (), None, {}, owner="ada", among=among)
            for n in range(5)
        )
    )
    mine = [found for found in opened if found is not None]
    assert len(mine) == 3 and len(sessions) == 3  # starting at once, still three
    for found in mine:
        found.working["t1"] = "research"  # at work: none of them gives its place
    assert await claude_live.opened(sessions, "a9", (), None, {}, owner="ada", among=among) is None
    bob = await claude_live.opened(sessions, "b1", (), None, {}, owner="bob", among=among)
    assert bob is not None  # another user's conversation finds a place
    mine[0].working.clear()  # idle now: ada's next takes ada's own place, not bob's
    assert await claude_live.opened(sessions, "a9", (), None, {}, owner="ada", among=among)
    assert sorted(s.key for s in sessions) == sorted([mine[1].key, mine[2].key, "a9", "b1"])


async def test_a_session_that_fails_to_start_gives_its_place_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rooms(monkeypatch, 1)

    async def broken(self: LiveSession, options: Any, transcripts: Any) -> None:
        raise RuntimeError("no CLI")

    sessions = LiveSessions()
    monkeypatch.setattr(LiveSession, "start", broken)
    with pytest.raises(RuntimeError, match="no CLI"):
        await claude_live.opened(sessions, "c1", (), None, {}, owner="u", among=[sessions])
    assert len(sessions) == 0 and not sessions.starting


async def test_a_lapse_lets_go_only_of_the_session_its_conversation_has() -> None:
    """A session's time up (``LiveSessions.lapse``) closes it while it is its conversation's;
    one the conversation no longer has — forgotten, a newer one in its place — is left to
    whoever forgot it, and the newer one stays."""
    sessions = LiveSessions()
    old = LiveSession("c1", (), sessions, owner="u")
    sessions.keep(old)
    sessions.forget(old, "its process ended")
    newer = LiveSession("c1", (), sessions, owner="u")
    sessions.keep(newer)
    await sessions.lapse(old, "it was idle")
    assert not old.closed and sessions.get("c1") is newer and not newer.closed
    await sessions.lapse(newer, "it was idle")
    assert newer.closed and sessions.get("c1") is None


async def test_closing_a_session_cancels_what_it_left_running_aside() -> None:
    """What a session started aside (a stop, an interrupt) ends with it."""
    session = LiveSession("conversation", (), LiveSessions(), owner="u")
    session._aside_run(asyncio.sleep(30))
    [aside] = list(session._aside)
    await session.close()
    await asyncio.sleep(0)
    assert aside.cancelled()
