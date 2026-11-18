"""An ACP agent's background work across a conversation's turns (``bridges.acp_live``), with the
scripted agent (``tests/support/fake_acp_agent.py``) standing in for Kimi Code: a sub-agent it
runs in the background (Kimi's ``Agent(run_in_background=true)``), a command it runs in the
background, the turn it runs of its own when one ends — all of which Kimi's ACP server shows
the client nothing of, and its log tells. ``tests/integration/test_kimi.py`` runs Kimi itself."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_acp import children, fake, recorded
from tests.support.arm import with_governance
from tests.support.catalog import FakeCatalog
from tests.support.invariants import descendants
from tests.support.kimi import kimi_home_of
from trellis import Harness, Settings
from trellis.contracts import RunStatus
from trellis.harness import budget
from trellis.harness.adapters.acp import ACPAdapter
from trellis.harness.agent import Agent
from trellis.harness.bridges import acp_live
from trellis.harness.bridges import live as shared
from trellis.harness.bridges.helpers import CUT_OFF
from trellis.harness.budget import Budget
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.identity import conversation_key

#: Claude Code and Kimi Code run their shells in a folder of this machine: the tests' deployment
#: allows it (TRELLIS_HOST_EXECUTION=allow, the harness fixtures).
pytestmark = pytest.mark.host_execution
#: the deployment of the tests that make their own harness
KIMI_HERE = Settings(host_execution="allow")

USER, THREAD = "u1", "chat"
#: The end of a turn of the scripted agent's (what follows is its next turn's).
END = {"stop": "end_turn"}
#: A sub-agent the agent runs in the background: its call answers at once with its task's id.
LOOK = {"description": "look", "prompt": "find tea", "run_in_background": True}


def started(tmp_path: Path) -> int:
    """How many agent processes were started."""
    return len(recorded(tmp_path).get("started", []))


def kept(agent: Agent) -> acp_live.ACPSession | None:
    adapter = agent.adapter
    assert isinstance(adapter, ACPAdapter)
    found = adapter.live.get(conversation_key("default", USER, THREAD))
    assert found is None or isinstance(found, acp_live.ACPSession)
    return found


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def until(check: Callable[[], Any], seconds: float = 10.0) -> None:
    """``check`` (a function, or a coroutine function) waited for until it holds."""
    async with asyncio.timeout(seconds):
        while True:
            held = check()
            if await held if asyncio.iscoroutine(held) else held:
                return
            await asyncio.sleep(0.05)


async def ended(h: Harness, run_id: str, seconds: float = 10.0) -> Any:
    """The run's record once it has ended."""

    async def final() -> bool:
        record = await h.runs.get(run_id)
        return record is not None and record.status.final

    await until(final, seconds)
    return await h.runs.get(run_id)


def warned(events: list[Any]) -> list[str]:
    return [e.data["message"] for e in events if e.data.get("code") == "acp_session"]


async def say(agent: Agent, text: str) -> Any:
    return await agent.run(text, user=USER, thread=THREAD)


# --------------------------------------------------------------------------- its work goes on
async def test_a_background_sub_agent_works_on_after_its_turn_and_its_run_ends_with_its_result(
    harness: Harness, tmp_path: Path
) -> None:
    """The turn that launched it ends at once; the sub-agent's child run stays ``RUNNING``
    (never ``SUCCESS`` with the launch's words) while the agent's process is kept, then ends
    with what the sub-agent said; the conversation's next turn is served by that process."""
    script = [
        {"agent": LOOK, "background": {"after": 1.0, "result": "found tea"}},
        {"text": "started it"},
        END,
        {"text": "anything else?"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg")
    first = await say(agent, "look for tea in the background")
    assert first.status is RunStatus.SUCCESS and first.answer == "started it", first.error
    [child] = await children(harness, first.run_id)
    assert child.status is RunStatus.RUNNING
    assert child.input == "find tea" and child.metadata["native_helper"]["name"] == "coder"
    session = kept(agent)
    assert session is not None and list(session.working)  # its task, under way
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.SUCCESS and done.output == "found tea"
    second = await say(agent, "what else?")
    assert second.status is RunStatus.SUCCESS and second.answer == "anything else?"
    assert started(tmp_path) == 1  # one process: the next turn was served by it
    assert [s["how"] for s in recorded(tmp_path)["sessions"]] == ["new"]
    # that turn ended with nothing at work in the background: its process ended with it
    assert kept(agent) is None and session.closed and not session.working


async def test_a_background_sub_agent_whose_task_fails_ends_its_run_failed(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"agent": LOOK, "background": {"after": 0.2, "fails": "the model refused"}},
        {"text": "started it"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-fails")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.ERROR and done.error is not None
    assert done.error.message == "the model refused"


async def test_with_no_conversation_background_work_lasts_as_long_as_the_attempt(
    harness: Harness, tmp_path: Path
) -> None:
    """A run of no conversation (no thread) has no next turn to keep the process for: it ends
    with the attempt, as ever. A background sub-agent that ended before the attempt did ends
    its run with its result; one still at work was cut off (``ERROR``), never ``SUCCESS``."""
    script = [
        {"agent": LOOK, "background": {"after": 0.1, "result": "quick"}},
        {"agent": {**LOOK, "prompt": "slow"}, "background": {"after": 30, "result": "slow"}},
        {"stall": 0.6},
        {"text": "both started"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-once")
    done = await agent.run("look twice", user=USER)
    assert done.status is RunStatus.SUCCESS, done.error
    quick, slow = sorted(await children(harness, done.run_id), key=lambda c: c.input != "find tea")
    assert (quick.status, quick.output) == (RunStatus.SUCCESS, "quick")
    assert slow.status is RunStatus.ERROR and slow.error is not None
    assert slow.error.message == CUT_OFF
    adapter = agent.adapter
    assert isinstance(adapter, ACPAdapter) and len(adapter.live) == 0


async def test_a_command_run_in_the_background_keeps_the_process_until_it_ends(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi's ``Bash(run_in_background=true)``: no child run (it is no sub-agent), but its
    process is kept while it works; once it ended and the session was idle long enough, the
    process ends."""
    monkeypatch.setattr(shared, "IDLE_SECONDS", 0.3)
    command = {"command": "make build", "run_in_background": True}
    script = [{"builtin": "Bash", "args": command, "background": {"after": 0.6}}, {"text": "ok"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-bash")
    done = await say(agent, "build it")
    assert done.status is RunStatus.SUCCESS and await children(harness, done.run_id) == []
    session = kept(agent)
    assert session is not None and session.process is not None
    pid = session.process.pid
    await until(lambda: session.closed)
    assert session.why == "it was idle" and kept(agent) is None
    await until(lambda: not alive(pid))


# --------------------------------------------------------------------------- what it asks
@pytest.mark.parametrize(
    ("after", "meanwhile"),
    [(0.3, []), (0, [{"stall": 0.4}])],
    ids=["after-the-turn", "while-the-turn-still-runs"],
)
async def test_a_background_sub_agents_call_is_governed_in_its_own_run(
    tmp_path: Path, after: float, meanwhile: list[dict[str, Any]]
) -> None:
    """BG-10. The sub-agent's call Kimi asks permission for is decided in the sub-agent's own
    run, whether it comes after the turn ended or while the turn that launched it still runs
    (before that turn took the launch from Kimi's log: the request takes it) — here governance
    asks a person, so that run pauses; the conversation's next turn asks the person (as a
    sub-agent's question is asked through its parent), and the answer lets the call run; the
    task ends, its run with it; then the turn's own prompt is sent."""
    catalog = FakeCatalog({"Write": Rule(risk="irreversible")})
    async with with_governance(Harness(config=KIMI_HERE), Governance(catalog)) as harness:
        ask = {"builtin": "Write", "args": {"path": "notes.md", "content": "x"}}
        script = [
            {"agent": LOOK, "background": {"after": after, "asks": ask, "result": "cleaned"}},
            *meanwhile,
            {"text": "started it"},
            END,
            {"text": "and now?"},
        ]
        agent = harness.wrap(fake(tmp_path, script), id="bg-asks")
        first = await say(agent, "clean up in the background")
        [child] = await children(harness, first.run_id)

        async def paused() -> bool:
            record = await harness.runs.get(child.run_id)
            return record is not None and record.status is RunStatus.PAUSED

        await until(paused)
        second = await say(agent, "how is it going?")
        assert second.status is RunStatus.PAUSED and second.interrupt is not None
        call = second.interrupt.tool_call
        assert call is not None and call.tool == "Write"
        named = (second.interrupt.payload or {})["subagent"]
        assert named["background"] is True and named["run_id"] == child.run_id
        done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="u1")
        assert done.status is RunStatus.SUCCESS and done.answer == "and now?", done.error
        assert recorded(tmp_path)["background"] == [
            {"task": recorded(tmp_path)["background"][0]["task"], "answer": "approve_once"}
        ]
        assert (await ended(harness, child.run_id)).status is RunStatus.SUCCESS


async def test_a_call_made_while_no_run_is_under_way_is_refused_and_the_next_run_told(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi's own turn for the task's end (ACP shows none of it) asks about a call: no run is
    under way to decide it, so it is refused, and the conversation's next run says so. That
    turn takes a while: the next prompt waits for it to end, and is answered as its own."""
    notified = {"builtin": "Bash", "args": {"command": "echo done"}}
    over = tmp_path / "the-first-run-is-over"
    task = {"gate": str(over), "result": "found", "notified": notified, "own_turn": 1.0}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}, END, {"text": "next"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-own")
    await say(agent, "look")
    over.touch()
    await until(lambda: "own" in recorded(tmp_path))
    events = [e async for e in agent.stream("and?", user=USER, thread=THREAD)]
    assert events[-1].error is None
    assert recorded(tmp_path)["own"][0]["answer"] == "reject"
    assert warned(events) == [
        "background work made tool calls while no run of the conversation was under way: "
        "they were refused, not run (Bash x1)"
    ]
    assert "queued" not in recorded(tmp_path)  # it waited for Kimi's own turn to end


async def test_a_prompt_kimi_queued_behind_its_own_turn_fails_the_turn_saying_so(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt sent while Kimi runs a turn of its own is queued behind it: Kimi answers it
    ``end_turn`` at once and runs it later, out of the client's sight. The harness waits for
    such a turn to end first; should one start just as the prompt goes, the turn fails saying
    so — never an empty answer taken for the prompt's."""

    async def ready(self: Any) -> bool:
        return True

    monkeypatch.setattr(acp_live.ACPSession, "settled", ready)
    task = {"after": 0.1, "result": "found", "own_turn": 3.0}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}, END, {"text": "next"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-queued")
    await say(agent, "look")
    session = kept(agent)
    assert session is not None
    await until(lambda: session.injected)
    second = await say(agent, "and?")
    assert second.status is RunStatus.ERROR and second.error is not None
    assert acp_live.QUEUED in second.error.message
    [queued] = recorded(tmp_path)["queued"]
    assert queued.endswith("and?")


# --------------------------------------------------------------------------- its spending
async def test_what_a_background_sub_agent_spends_is_its_runs(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sub-agent's model requests (its own log's ``usage.record``) are charged to its own
    run, not to the turn being served (nor to none)."""
    charged: list[tuple[str, int]] = []
    account = budget.account

    def counting(runtime: Any, **usage: Any) -> None:
        charged.append((runtime.run_id, usage["input_tokens"] + usage["output_tokens"]))
        account(runtime, **usage)

    monkeypatch.setattr(budget, "account", counting)
    task = {"after": 0.3, "usage": {"inputOther": 70, "output": 30}, "result": "found"}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-spends")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    await ended(harness, child.run_id)
    assert (child.run_id, 100) in charged
    assert not [n for run, n in charged if run == first.run_id]


# --------------------------------------------------------------------------- its end
async def test_work_that_ends_with_the_process_ends_its_run_and_the_next_turn_is_told(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session is kept at most ``live.WORKING_SECONDS`` after its last turn: the process
    ends then, its background sub-agent's run ends ``ERROR`` saying so, and the next turn —
    in a new process and a new session (the lapsed one's files are gone), given the earlier
    turns by the harness — is told what was stopped."""
    monkeypatch.setattr(shared, "WORKING_SECONDS", 0.5)
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        {"text": "started"},
        END,
        {"text": "next"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-lost")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.ERROR and done.error is not None
    assert done.error.message == acp_live.LOST.format(why="it was kept as long as a session is")
    events = [e async for e in agent.stream("and?", user=USER, thread=THREAD)]
    assert events[-1].error is None
    [said] = warned(events)
    assert said.startswith("Kimi Code's session was closed (it was kept as long as a session is)")
    assert [s["how"] for s in recorded(tmp_path)["sessions"]] == ["new", "new"]
    assert "look" in recorded(tmp_path)["prompts"][-1]  # the first turn, from the harness


def _team_home(tmp_path: Path) -> Path:
    """The team's Kimi home of :func:`fake`'s agent: its login and its config."""
    team = tmp_path / "home"
    (team / "credentials").mkdir(parents=True)
    (team / "config.toml").write_text('default_model = "k2"\n')
    return team


def _session_left(agent: Agent, session: acp_live.ACPSession) -> list[Path]:
    """What is left of ``session``'s own files in its user's Kimi home."""
    assert session.log is not None
    return list(session.log.home.glob(f"sessions/*/{session.session_id}"))


async def _no_child_left(before: set[int]) -> None:
    """No process this test started is left (the agent's, and what it started)."""
    await until(lambda: not set(descendants(os.getpid())) - before)


async def test_a_session_whose_hold_lapses_leaves_no_process_no_running_run_and_no_files(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KIMI-LIVE-1. The conversation's hold of its kept session lapses (kept as long as a
    session is): the process and what it started end, no run is left ``RUNNING``, the
    session's own files leave its user's Kimi home — its login and its config stay — and the
    conversation keeps no session for its next turn to continue."""
    monkeypatch.setattr(shared, "WORKING_SECONDS", 0.5)
    team = _team_home(tmp_path)
    script = [{"agent": LOOK, "background": {"after": 30, "result": "never"}}, {"text": "go"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-lapse")
    before = set(descendants(os.getpid()))
    first = await say(agent, "look")
    session = kept(agent)
    assert session is not None and _session_left(agent, session)
    home = session.log.home
    [child] = await children(harness, first.run_id)
    assert (await ended(harness, child.run_id)).status is RunStatus.ERROR
    await until(lambda: session.closed and not _session_left(agent, session))
    await _no_child_left(before)
    runs = [s async for s in harness.runs.iterate()]
    assert not [s for s in runs if s.status is RunStatus.RUNNING]
    assert (home / "config.toml").read_text().startswith((team / "config.toml").read_text())
    assert (home / "credentials").resolve() == team / "credentials"
    assert (home / "sessions").is_dir()  # only the session's own folder went
    assert agent.sessions.get(conversation_key("default", USER, THREAD)) is None


async def test_a_lapsing_session_leaves_a_newer_session_of_its_conversation_kept(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session the conversation keeps for its next turn is another one by then: that one
    stays kept; only the lapsed session's files go."""
    monkeypatch.setattr(shared, "WORKING_SECONDS", 0.5)
    script = [{"agent": LOOK, "background": {"after": 30, "result": "never"}}, {"text": "go"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-newer")
    await say(agent, "look")
    session = kept(agent)
    assert session is not None
    key = conversation_key("default", USER, THREAD)
    agent.sessions[key] = ("session_newer", {})
    await until(lambda: session.closed and not _session_left(agent, session))
    assert agent.sessions.get(key) == ("session_newer", {})


async def test_a_session_kept_when_the_harness_closes_leaves_no_process_and_no_files(
    tmp_path: Path,
) -> None:
    """KIMI-LIVE-2. ``Harness.aclose`` with a kept session: its process and what it started
    end, its background run is not left ``RUNNING``, and its files leave its user's Kimi home,
    whose login and config stay."""
    team = _team_home(tmp_path)
    script = [{"agent": LOOK, "background": {"after": 30, "result": "never"}}, {"text": "go"}]
    before = set(descendants(os.getpid()))
    async with Harness(config=KIMI_HERE) as harness:
        agent = harness.wrap(fake(tmp_path, script), id="bg-aclose")
        first = await say(agent, "look")
        [child] = await children(harness, first.run_id)
        session = kept(agent)
        assert session is not None and _session_left(agent, session)
    await _no_child_left(before)
    assert not _session_left(agent, session)
    record = await harness.runs.get(child.run_id)
    assert record is not None and record.status is RunStatus.ERROR
    home = session.log.home
    assert (home / "config.toml").read_text().startswith((team / "config.toml").read_text())
    assert (home / "credentials").resolve() == team / "credentials"


def _files_of(session_id: str) -> list[Path]:
    """The folder the session ``session_id`` keeps in its user's Kimi home."""
    return list(kimi_home_of(USER).glob(f"sessions/*/{session_id}"))


def _kept_for(agent: Agent, thread: str = THREAD) -> str:
    """The session the agent keeps for ``thread``'s next turn (``Agent.sessions``)."""
    found = agent.sessions.get(conversation_key("default", USER, thread))
    assert found is not None
    return found[0]


async def test_a_session_that_ended_with_nothing_at_work_keeps_its_files_while_it_is_kept(
    harness: Harness, tmp_path: Path
) -> None:
    """KIMI-LIVE-3. A turn that ends with nothing at work in the background ends its process;
    its session's files stay, and the conversation's next turn resumes it from them. Once the
    agent keeps that session no more — its place among the conversations' (``Agent.sessions``)
    given to another conversation's — its files leave its user's Kimi home; the other
    conversation's stay."""
    agent = harness.wrap(fake(tmp_path, [{"text": "one"}, END, {"text": "two"}]), id="idle")
    agent.sessions.maxsize = 1
    assert (await say(agent, "first")).answer == "one"
    first = _kept_for(agent)
    assert kept(agent) is None and _files_of(first)
    assert (await say(agent, "again")).answer == "two"
    assert [s["how"] for s in recorded(tmp_path)["sessions"]] == ["new", "resume"]
    assert _kept_for(agent) == first and _files_of(first)
    other = await agent.run("hello", user=USER, thread="other")
    assert other.status is RunStatus.SUCCESS, other.error
    assert not _files_of(first)
    assert _files_of(_kept_for(agent, "other"))
    assert (kimi_home_of(USER) / "config.toml").exists()  # only the session's own folder went


async def test_a_session_whose_process_is_kept_keeps_its_files_when_the_agent_lets_go_of_it(
    harness: Harness, tmp_path: Path
) -> None:
    """The agent lets go of the session of a conversation whose process is kept for its
    background work: the process still holds it, so its files stay (its own end removes them:
    ``ACPSession.released``)."""
    script = [{"agent": LOOK, "background": {"after": 30, "result": "never"}}, {"text": "go"}]
    agent = harness.wrap(fake(tmp_path, script), id="held")
    agent.sessions.maxsize = 1
    await say(agent, "look")
    session = kept(agent)
    assert session is not None and session.session_id is not None
    await agent.run("hello", user=USER, thread="other")
    assert agent.sessions.get(conversation_key("default", USER, THREAD)) is None
    assert not session.closed and _files_of(session.session_id)


async def test_a_session_kept_for_the_next_turn_leaves_its_files_when_the_harness_closes(
    tmp_path: Path,
) -> None:
    """KIMI-LIVE-4. ``Harness.aclose`` with a session kept for a conversation's next turn (its
    process ended with its turn): its files leave its user's Kimi home; its login and config
    stay."""
    async with Harness(config=KIMI_HERE) as harness:
        agent = harness.wrap(fake(tmp_path, [{"text": "one"}]), id="idle-aclose")
        await say(agent, "first")
        first = _kept_for(agent)
        assert _files_of(first)
    assert not _files_of(first)
    assert (kimi_home_of(USER) / "config.toml").exists()


async def test_a_paused_turns_session_keeps_its_files_when_the_harness_closes(
    tmp_path: Path,
) -> None:
    """A turn paused on a question keeps its session in its run's journal, which a resume
    continues on whichever server takes it: closing the harness leaves its files."""
    script = [{"question": "Which tea?", "options": ["green", "black"]}, {"text": "{last}"}]
    async with Harness(config=KIMI_HERE) as harness:
        agent = harness.wrap(fake(tmp_path, script), id="paused-aclose")
        paused = await say(agent, "pick a tea")
        assert paused.status is RunStatus.PAUSED, paused.error
        [created] = recorded(tmp_path)["sessions"]
    assert _files_of(created["id"])


async def test_a_process_that_dies_ends_its_background_runs(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        {"text": "started"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-dies")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    session = kept(agent)
    assert session is not None and session.process is not None
    session.process.kill()
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.ERROR and done.error is not None
    assert done.error.message == acp_live.LOST.format(why="its process ended")
    assert kept(agent) is None


async def test_closing_the_harness_ends_the_kept_process_and_its_background_runs(
    tmp_path: Path,
) -> None:
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        {"text": "started"},
    ]
    async with Harness(config=KIMI_HERE) as harness:
        agent = harness.wrap(fake(tmp_path, script), id="bg-closed")
        first = await say(agent, "look")
        [child] = await children(harness, first.run_id)
        session = kept(agent)
        assert session is not None and session.process is not None
        pid = session.process.pid
    assert not alive(pid) or session.process.returncode is not None
    record = await harness.runs.get(child.run_id)
    assert record is not None and record.status is RunStatus.ERROR


async def test_a_cancelled_turn_cancels_its_background_sub_agents_run(
    harness: Harness, tmp_path: Path
) -> None:
    """The turn that launched it is cancelled: the sub-agent's run is its child, cancelled with
    it (as a cancelled run's children are)."""
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        {"stall": 30},
        {"text": "never"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-cancel")
    events = agent.stream("look", user=USER, thread=THREAD)
    run_id = (await anext(events)).run_id
    rest = asyncio.ensure_future(_drained(events))

    async def launched() -> bool:
        return bool(await children(harness, run_id))

    await until(launched)
    [child] = await children(harness, run_id)
    await agent.cancel(run_id, reason="the user left")
    await rest
    assert (await harness.runs.get(run_id)).status is RunStatus.CANCELLED  # type: ignore[union-attr]
    assert (await ended(harness, child.run_id)).status is RunStatus.CANCELLED


async def _drained(events: Any) -> None:
    async for _ in events:
        pass


async def test_a_session_another_worker_still_holds_is_forked_not_resumed_twice(
    harness: Harness, tmp_path: Path
) -> None:
    """A turn paused after it launched a background sub-agent; its answer reaches another
    worker (here, another agent on the same run store: its process keeps nothing of the
    first's). The sub-agent's run, handed on to the first worker's session, has not ended:
    two processes do not hold one session, so the resumed turn forks it, and goes on there."""
    ask = {"builtin": "Edit", "args": {"path": "made.txt"}, "ask": True}  # the user's
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        ask,
        {"text": "done"},
    ]
    target = fake(tmp_path, script)
    first = harness.wrap(target, id="bg-worker")
    paused = await say(first, "look, then make it")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    # another worker, on the same run store: its process kept nothing
    worker = Harness(config=KIMI_HERE)
    worker.runs = harness.runs
    other = worker.wrap(target, id="bg-worker")
    done = await other.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    sessions = recorded(tmp_path)["sessions"]
    assert [s["how"] for s in sessions] == ["new", "fork"] and sessions[1]["from"] == sessions[0][
        "id"
    ]


# --------------------------------------------------------------------------- whose process
async def test_a_turn_with_other_settings_is_not_served_by_the_kept_process(
    harness: Harness, tmp_path: Path
) -> None:
    """A process keeps what it was started with (its environment, its tools...): a turn that
    brings other settings closes it — its background sub-agent's run ends saying why — and
    runs in a new process, resuming the session."""
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        {"text": "started"},
        END,
        {"text": "next"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-settings")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    options = {"env": {"SOME_SETTING": "changed"}}
    second = await agent.run("and?", user=USER, thread=THREAD, framework_options=options)
    assert second.status is RunStatus.SUCCESS and second.answer == "next", second.error
    done = await ended(harness, child.run_id)
    why = "this turn's settings are not the ones it was started with"
    assert done.error is not None and done.error.message == acp_live.LOST.format(why=why)
    assert [s["how"] for s in recorded(tmp_path)["sessions"]] == ["new", "resume"]


async def test_a_turn_that_continues_another_session_is_not_served_by_the_kept_process(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "never"}},
        {"text": "started"},
        END,
        {"text": "next"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-other-session")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    key = conversation_key("default", USER, THREAD)
    agent.sessions[key] = ("session_gone", {})  # what this process kept for the conversation
    second = await say(agent, "and?")
    assert second.status is RunStatus.SUCCESS, second.error
    done = await ended(harness, child.run_id)
    why = "this turn does not continue the session it holds"
    assert done.error is not None and done.error.message == acp_live.LOST.format(why=why)


async def test_a_kept_process_whose_own_turn_does_not_end_is_closed_for_the_next_turn(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next turn waits ``live.SETTLE_SECONDS`` for Kimi's own turn to end; past it the
    process is closed and the turn runs in a new one."""
    monkeypatch.setattr(shared, "SETTLE_SECONDS", 0.3)
    task = {"after": 0.1, "result": "found", "own_turn": 30}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}, END, {"text": "next"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-unsettled")
    await say(agent, "look")
    session = kept(agent)
    assert session is not None
    await until(lambda: session.injected)
    second = await say(agent, "and?")
    assert second.status is RunStatus.SUCCESS and second.answer == "next", second.error
    assert session.closed and session.why == "a turn of Kimi Code's own did not end"


async def test_an_answer_that_reaches_a_worker_without_the_session_ends_its_run(
    tmp_path: Path,
) -> None:
    """A background sub-agent's question, asked in a turn, is answered on another worker:
    its session — and its task — are not there to go on, so its run records the answer and
    ends, and the turn says so; the turn's own prompt goes on."""
    catalog = FakeCatalog({"Write": Rule(risk="irreversible")})
    async with with_governance(Harness(config=KIMI_HERE), Governance(catalog)) as harness:
        ask = {"builtin": "Write", "args": {"path": "notes.md", "content": "x"}}
        script = [
            {"agent": LOOK, "background": {"after": 0.2, "asks": ask, "result": "cleaned"}},
            {"text": "started it"},
            END,
            {"text": "and now?"},
        ]
        target = fake(tmp_path, script)
        agent = harness.wrap(target, id="bg-gone")
        first = await say(agent, "clean up")
        [child] = await children(harness, first.run_id)

        async def paused() -> bool:
            record = await harness.runs.get(child.run_id)
            return record is not None and record.status is RunStatus.PAUSED

        await until(paused)
        second = await say(agent, "how is it going?")
        assert second.interrupt is not None
        async with with_governance(Harness(config=KIMI_HERE), Governance(catalog)) as worker:
            worker.runs = harness.runs  # another worker, on the same run store
            other = worker.wrap(target, id="bg-gone")
            events: list[Any] = []

            async def follow() -> None:
                async for event in other.events(second.run_id):
                    events.append(event)

            following = asyncio.create_task(follow())
            await until(lambda: other.watchers(second.run_id))
            done = await other.resume(second.interrupt.interrupt_id, "approve", reviewer="u1")
            await following
        assert done.status is RunStatus.SUCCESS and done.answer == "and now?", done.error
        assert warned(events) == [acp_live.GONE_SAID]
        lost = await harness.runs.get(child.run_id)
        assert lost is not None and lost.status is RunStatus.ERROR and lost.error is not None
        assert lost.error.message == acp_live.GONE


# --------------------------------------------------------------------------- what it asks
async def test_a_background_sub_agents_harness_tool_call_is_made_in_its_own_run(
    harness: Harness, tmp_path: Path
) -> None:
    """Its harness tool's call (the MCP server's, which names no caller) is made in its run:
    Kimi asks permission for it first, which says whose the call is."""
    from tests.integration.test_acp import lookup
    from tests.support.arm import arm

    ask = {"tool": "lookup", "args": {"topic": "tea"}}
    script = [
        {"agent": LOOK, "background": {"after": 0.2, "asks": ask, "result": "looked"}},
        {"text": "started"},
    ]
    agent = arm(harness.wrap(fake(tmp_path, script), id="bg-tool"), [lookup])
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    assert (await ended(harness, child.run_id)).status is RunStatus.SUCCESS
    task = recorded(tmp_path)["background"][0]["task"]
    assert recorded(tmp_path)["background"] == [
        {"task": task, "answer": "approve_once"},
        {"task": task, "result": "facts about tea"},
    ]


async def test_a_background_sub_agents_call_a_person_rejects_is_refused(tmp_path: Path) -> None:
    catalog = FakeCatalog({"Write": Rule(risk="irreversible")})
    async with with_governance(Harness(config=KIMI_HERE), Governance(catalog)) as harness:
        ask = {"builtin": "Write", "args": {"path": "notes.md", "content": "x"}}
        script = [
            {"agent": LOOK, "background": {"after": 0.2, "asks": ask, "result": "left it"}},
            {"text": "started it"},
            END,
            {"text": "and now?"},
        ]
        agent = harness.wrap(fake(tmp_path, script), id="bg-rejected")
        first = await say(agent, "clean up")
        [child] = await children(harness, first.run_id)

        async def paused() -> bool:
            record = await harness.runs.get(child.run_id)
            return record is not None and record.status is RunStatus.PAUSED

        await until(paused)
        second = await say(agent, "how is it going?")
        assert second.interrupt is not None
        done = await agent.resume(second.interrupt.interrupt_id, "reject", reviewer="u1")
        assert done.status is RunStatus.SUCCESS, done.error
        assert recorded(tmp_path)["background"][0]["answer"] == "reject"
        assert (await ended(harness, child.run_id)).status is RunStatus.SUCCESS


# --------------------------------------------------------------------------- how it ends
async def test_a_background_task_stopped_in_kimi_cancels_its_run(
    harness: Harness, tmp_path: Path
) -> None:
    task = {"after": 0.2, "status": "killed"}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-killed")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.CANCELLED


async def test_a_background_sub_agent_past_its_runs_budget_ends_that_run(
    harness: Harness, tmp_path: Path
) -> None:
    """Its run has the agent's budget: past it, the run ends there (``PARTIAL``) — ACP has no
    way to stop the task, which works on, its requests refused."""
    task = {"after": 0.2, "usage": {"inputOther": 700, "output": 300}, "result": "found"}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-budget", budget=Budget(max_tokens=500))
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.PARTIAL


async def test_a_task_that_ended_before_its_launch_was_read_ends_its_run(
    harness: Harness, tmp_path: Path
) -> None:
    task = {"result": "quick", "first": True}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg-first")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    done = await ended(harness, child.run_id)
    assert (done.status, done.output) == (RunStatus.SUCCESS, "quick")


async def test_the_next_prompt_waits_for_each_of_kimis_own_turns(
    harness: Harness, tmp_path: Path
) -> None:
    """Two tasks end one after the other, the second while Kimi's own turn for the first is
    under way: the next prompt waits for both of Kimi's turns, and is answered as its own."""
    first_task = {"after": 0.1, "result": "a", "own_turn": 0.6}
    second_task = {"after": 0.3, "result": "b", "own_turn": 0.3}
    script = [
        {"agent": LOOK, "background": first_task},
        {"agent": {**LOOK, "prompt": "other"}, "background": second_task},
        {"text": "started"},
        END,
        {"text": "next"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-two")
    await say(agent, "look twice")
    session = kept(agent)
    assert session is not None
    await until(lambda: session.injected)
    second = await say(agent, "and?")
    assert second.status is RunStatus.SUCCESS and second.answer == "next", second.error
    assert "queued" not in recorded(tmp_path)


async def test_an_agent_that_cannot_be_started_fails_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    from trellis.harness.adapters.acp import ACPAgent

    target = ACPAgent(command=(str(tmp_path / "no-such-agent"),))
    done = await harness.wrap(target, id="missing").run("hi", user=USER)
    assert done.status is RunStatus.ERROR and done.error is not None
    assert "no-such-agent" in done.error.message


async def test_an_agent_the_harness_keeps_no_log_of_is_taken_at_its_word(
    harness: Harness, tmp_path: Path
) -> None:
    """With no log to read, a turn that ended with nothing said is no queued prompt: it is the
    agent's answer (none)."""
    agent = harness.wrap(fake(tmp_path, [], profile=None), id="no-log")
    done = await agent.run("hi", user=USER)
    assert done.status is RunStatus.SUCCESS and done.answer is None


async def test_a_harness_tool_call_no_run_can_make_is_refused(
    harness: Harness, tmp_path: Path
) -> None:
    """A harness tool's call that comes with no word of whose it is (no permission asked
    first) while no turn is served has no run to make it: refused, and the next run told."""
    from tests.integration.test_acp import lookup
    from tests.support.arm import arm

    task = {"after": 0.5, "calls": {"tool": "lookup", "args": {"topic": "tea"}}, "result": "x"}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}, END, {"text": "next"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="bg-unasked"), [lookup])
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    await ended(harness, child.run_id)
    assert recorded(tmp_path)["background"][0]["result"] == shared.UNATTENDED
    events = [e async for e in agent.stream("and?", user=USER, thread=THREAD)]
    assert warned(events) == [
        "background work made tool calls while no run of the conversation was under way: "
        "they were refused, not run (lookup x1)"
    ]


async def test_a_harness_tool_call_of_a_background_run_that_ended_is_refused(
    harness: Harness, tmp_path: Path
) -> None:
    """Its run ended (cancelled) between Kimi's permission request and the call it allowed:
    there is no run to make the call — refused."""
    from tests.integration.test_acp import lookup
    from tests.support.arm import arm

    ask = {"tool": "lookup", "args": {"topic": "tea"}}
    task = {"after": 0.2, "asks": ask, "call_after": 1.0, "result": "x"}
    script = [{"agent": LOOK, "background": task}, {"text": "started"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="bg-over"), [lookup])
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    await until(lambda: recorded(tmp_path).get("background"))
    await agent.cancel(child.run_id, reason="enough")
    await until(lambda: len(recorded(tmp_path)["background"]) == 2)
    assert recorded(tmp_path)["background"][1]["result"] == shared.UNATTENDED


async def test_with_no_log_of_the_agent_a_background_sub_agent_is_cut_off_with_its_attempt(
    harness: Harness, tmp_path: Path
) -> None:
    """An agent the harness keeps no log of (its profile names no usage log): nothing says
    when its background task ends, so its process is never kept for one and the sub-agent's
    run is cut off with the attempt (``ERROR``), never ``SUCCESS`` at its launch."""
    import dataclasses

    from trellis.harness.adapters.acp import KIMI

    profile = dataclasses.replace(KIMI, usage=None)
    script = [{"agent": LOOK, "background": {"after": 30, "result": "x"}}, {"text": "started"}]
    agent = harness.wrap(fake(tmp_path, script, profile=profile), id="bg-no-log")
    first = await say(agent, "look")
    [child] = await children(harness, first.run_id)
    assert child.status is RunStatus.ERROR and child.error is not None
    assert child.error.message == CUT_OFF
