"""One rule for a conversation's history across runs (``bridges.history``): the framework owns
it only when its thread store is durable; a thread kept in RAM leaves it to the memory service's
window — said once, with the fix — and with memory off nothing keeps it through a restart (said
too). Each framework's history is checked across a worker restart between turns: the harness
closed, a new one opened, the target built again, as a new worker process would."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, SQLiteSession
from claude_agent_sdk import ClaudeAgentOptions, InMemorySessionStore
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver

from tests.support.chat_model import ScriptedChatModel
from tests.support.durable import SharedSaver
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Harness, Settings
from trellis.contracts import RunStatus
from trellis.harness import bridges
from trellis.harness.bounded import SAID_ONCE, LRUSet
from trellis.harness.bridges import history
from trellis.harness.bridges.history import Policy, Thread, in_ram
from trellis.harness.clients.memory import SOURCE_SYSTEM
from trellis.harness.features import Feature
from trellis.harness.middleware import RunCheckpointer

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
FIRST, SECOND = "my order is QX-4471", "what is my order?"


@pytest.fixture(autouse=True)
def _forget_what_was_said(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridges, "_said", LRUSet(maxsize=SAID_ONCE))


def in_ram_said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.__dict__.get("bridge") == "in_ram"]


def windows(service: FakeMemoryService) -> list[bool]:
    return [c.body["window"] for c in service.named("context")]


async def restarted(
    service: FakeMemoryService | None,
    build: Callable[[], Any],
    *,
    framework_options: dict[str, Any] | None = None,
    without: frozenset[Feature] = frozenset(),
) -> list[Any]:
    """Turn one on a worker, the worker restarted, turn two on the next: each worker its own
    harness and the target built again; the memory service (when on) outlives both."""
    results = []
    for text in (FIRST, SECOND):
        memory = service.client() if service is not None else False
        async with Harness(config=Settings(), memory=memory) as h:
            agent = h.wrap(
                build(), id="keeper", framework_options=framework_options, without=without
            )
            results.append(await agent.run(text, user="ada", thread="t1"))
    assert [r.status for r in results] == [RunStatus.SUCCESS] * 2, [r.error for r in results]
    return results


def started(tmp_path: Path) -> list[dict[str, Any]]:
    """What each scripted Claude CLI was started with, in order."""
    return [json.loads(p.read_text()) for p in sorted(tmp_path.glob("cli-*.json"))]


def humans(messages: list[Any]) -> list[str]:
    return [str(m.content) for m in messages if m.type == "human"]


# ---------------------------------------------------------------------- the registry
def test_the_stores_kept_in_ram_are_named_and_any_other_is_durable(tmp_path: Path) -> None:
    class Mine(InMemorySaver):
        """A subclass keeps its parent's RAM."""

    assert in_ram(InMemorySaver()) and in_ram(MemorySaver()) and in_ram(Mine())
    assert in_ram(SQLiteSession("s")) and in_ram(InMemorySessionStore())
    on_disk = SQLiteSession("s", tmp_path / "s.db")
    assert not in_ram(on_disk)
    on_disk.close()
    assert not in_ram(SharedSaver("x")) and not in_ram(object())  # unknown: taken as durable


# ---------------------------------------------------------------------- LangGraph, Deep Agents
GRAPHS: dict[str, Callable[[Any, Any], Any]] = {
    "langgraph": lambda model, saver: create_agent(model, tools=[], checkpointer=saver),
    "deepagents": lambda model, saver: create_deep_agent(model=model, checkpointer=saver),
}


@pytest.mark.parametrize("framework", list(GRAPHS))
async def test_a_graph_on_an_in_ram_saver_leaves_the_history_to_memory(
    framework: str, memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedChatModel(turns=["noted", "QX-4471"])
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        await restarted(memory_service, lambda: GRAPHS[framework](model, InMemorySaver()))
    assert windows(memory_service) == [True, True]  # memory's window carries the turns
    assert humans(model.seen[-1]) == [SECOND]  # the saver's thread went with its worker
    assert memory_service.context_text in str(model.seen[-1])  # memory brought it back
    [said] = in_ram_said(caplog)
    assert "InMemorySaver keeps the thread in this process's memory only" in said
    assert "the memory service's window keeps the conversation;" in said
    assert "PostgresSaver" in said


@pytest.mark.parametrize("framework", list(GRAPHS))
async def test_a_graph_on_a_durable_saver_keeps_the_history_through_a_restart(
    framework: str, memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedChatModel(turns=["noted", "QX-4471"])
    name = f"db-{uuid.uuid4().hex}"
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        await restarted(memory_service, lambda: GRAPHS[framework](model, SharedSaver(name)))
    assert windows(memory_service) == [False, False]  # the framework's thread owns it
    assert humans(model.seen[-1]) == [FIRST, SECOND]  # read back on the next worker
    assert in_ram_said(caplog) == []


async def test_memory_off_and_an_in_ram_saver_say_the_history_will_not_survive(
    memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedChatModel(turns=["noted", "I do not know"])
    off: frozenset[Feature] = frozenset({"memory"})
    for service, without in ((None, frozenset[Feature]()), (memory_service, off)):
        bridges._said = LRUSet(maxsize=SAID_ONCE)
        caplog.clear()
        model.seen.clear()
        model.turns = ["noted", "I do not know"]
        with caplog.at_level(logging.WARNING, logger="trellis.harness"):
            await restarted(
                service,
                lambda: create_agent(model, tools=[], checkpointer=InMemorySaver()),
                without=without,
            )
        [said] = in_ram_said(caplog)
        assert "memory is off, so the history will not survive a restart" in said
        assert humans(model.seen[-1]) == [SECOND]  # lost, as said
    assert memory_service.calls == [] or not memory_service.named("context")


# ---------------------------------------------------------------------- OpenAI Agents
def users(input: Any) -> list[str]:
    if isinstance(input, str):
        return [input]
    return [i["content"] for i in input if isinstance(i, dict) and i.get("role") == "user"]


async def test_an_in_ram_session_leaves_the_history_to_memory(
    memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedModel(["noted", "QX-4471"])
    sessions: list[SQLiteSession] = []

    def build() -> Any:
        sessions.append(SQLiteSession("t1"))  # each worker's own, in RAM
        return Agent(name="s", model=model)

    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for text in (FIRST, SECOND):
            async with Harness(config=Settings(), memory=memory_service.client()) as h:
                agent = h.wrap(build(), id="keeper")
                options = {"session": sessions[-1]}
                result = await agent.run(text, user="ada", thread="t1", framework_options=options)
                assert result.status is RunStatus.SUCCESS
    assert windows(memory_service) == [True, True]
    assert users(model.inputs[-1]) == [SECOND]  # the RAM session went with its worker
    [said] = in_ram_said(caplog)
    assert "SQLiteSession keeps the thread in this process's memory only" in said
    for session in sessions:
        session.close()


async def test_a_durable_session_keeps_the_history_through_a_restart(
    memory_service: FakeMemoryService, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedModel(["noted", "QX-4471"])
    sessions: list[SQLiteSession] = []

    def build() -> Any:
        sessions.append(SQLiteSession("t1", tmp_path / "sessions.db"))
        return Agent(name="s", model=model)

    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for text in (FIRST, SECOND):
            async with Harness(config=Settings(), memory=memory_service.client()) as h:
                agent = h.wrap(build(), id="keeper")
                options = {"session": sessions[-1]}
                await agent.run(text, user="ada", thread="t1", framework_options=options)
    assert windows(memory_service) == [False, False]
    assert users(model.inputs[-1])[-2:] == [FIRST, SECOND]
    assert in_ram_said(caplog) == []
    for session in sessions:
        session.close()


async def test_the_memory_services_session_keeps_the_history_through_a_restart(
    memory_service: FakeMemoryService,
) -> None:
    model = ScriptedModel(["noted", "QX-4471"])
    await restarted(memory_service, lambda: Agent(name="s", model=model))
    assert windows(memory_service) == [False, False]  # the session (the thread) owns it
    assert users(model.inputs[-1]) == [FIRST, SECOND]


async def test_memory_off_the_sdk_keeps_no_history_and_nothing_is_said(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = ScriptedModel(["noted", "I do not know"])
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        await restarted(None, lambda: Agent(name="s", model=model))
    assert users(model.inputs[-1]) == [SECOND]
    assert in_ram_said(caplog) == []  # no store of its own: nothing kept in RAM to warn of


# ---------------------------------------------------------------------- Claude Agent SDK
async def test_claudes_history_across_runs_is_memorys_through_a_restart(
    memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    def build() -> Any:
        record = tmp_path / f"cli-{uuid.uuid4().hex[:6]}.json"
        return ClaudeAgentOptions(
            cli_path=CLI,
            env={
                "FAKE_CLAUDE_SCRIPT": json.dumps([{"text": "ok"}]),
                "FAKE_CLAUDE_RECORD": str(record),
                "FAKE_CLAUDE_SESSIONS": str(tmp_path / uuid.uuid4().hex),
            },
        )

    await restarted(memory_service, build)
    # each run is a new Claude session: memory's window carries the conversation, through the
    # prompt hook, on every worker
    assert windows(memory_service) == [True, True]
    assert [r["context"] for r in started(tmp_path)] == [memory_service.context_text] * 2


# ---------------------------------------------------------------------- the policy, both modes
@pytest.mark.parametrize(
    ("policy", "store", "bridged", "thread"),
    [
        (Policy.DURABLE_THREAD, SharedSaver("x"), True, Thread.OWN),
        (Policy.DURABLE_THREAD, None, True, Thread.BRIDGED),
        (Policy.DURABLE_THREAD, None, False, Thread.KEPT),
        (Policy.DURABLE_THREAD, InMemorySaver(), True, Thread.KEPT),
        (Policy.MEMORY_WINDOW, SharedSaver("x"), True, Thread.OWN),
        (Policy.MEMORY_WINDOW, None, True, Thread.RUN),
        (Policy.MEMORY_WINDOW, InMemorySaver(), False, Thread.RUN),
    ],
)
def test_one_policy_says_who_owns_the_history(
    monkeypatch: pytest.MonkeyPatch, policy: Policy, store: Any, bridged: bool, thread: Thread
) -> None:
    monkeypatch.setattr(history, "POLICY", policy)
    found = history.thread_for(store, bridged=bridged, agent_id="a", memory=True)
    assert found is thread
    assert history.owns(found) is (thread in (Thread.OWN, Thread.BRIDGED))


@pytest.fixture
def memory_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The policy flipped: memory's window owns the history across runs."""
    monkeypatch.setattr(history, "POLICY", Policy.MEMORY_WINDOW)


@pytest.mark.usefixtures("memory_window")
@pytest.mark.parametrize("saver", ["none", "in_ram"])
async def test_flipped_a_graph_without_a_durable_saver_runs_on_the_runs_own_thread(
    saver: str, memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedChatModel(turns=["noted", "QX-4471"])
    targets: list[Any] = []
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for text in (FIRST, SECOND):
            async with Harness(config=Settings(), memory=memory_service.client()) as h:
                given = create_agent(
                    model, tools=[], checkpointer=InMemorySaver() if saver == "in_ram" else None
                )
                agent = h.wrap(given, id="keeper")
                targets.append(agent.target)
                assert (await agent.run(text, user="ada", thread="t1")).status is RunStatus.SUCCESS
    assert all(isinstance(t.checkpointer, RunCheckpointer) for t in targets)
    assert windows(memory_service) == [True, True]  # memory's window owns the history
    if saver == "in_ram":
        [said] = in_ram_said(caplog)
        assert "the run's own thread (its journal) is used instead" in said


@pytest.mark.usefixtures("memory_window")
async def test_flipped_a_durable_saver_given_explicitly_keeps_the_history(
    memory_service: FakeMemoryService,
) -> None:
    model = ScriptedChatModel(turns=["noted", "QX-4471"])
    name = f"db-{uuid.uuid4().hex}"
    await restarted(
        memory_service, lambda: create_agent(model, tools=[], checkpointer=SharedSaver(name))
    )
    assert windows(memory_service) == [False, False]
    assert humans(model.seen[-1]) == [FIRST, SECOND]


@pytest.mark.usefixtures("memory_window")
async def test_flipped_an_openai_agent_runs_without_a_session_and_memory_keeps_the_history(
    memory_service: FakeMemoryService,
) -> None:
    model = ScriptedModel(["noted", "QX-4471"])
    in_ram_session = SQLiteSession("t1")
    await restarted(
        memory_service,
        lambda: Agent(name="s", model=model),
        framework_options={"session": in_ram_session},
    )
    assert windows(memory_service) == [True, True]
    assert await in_ram_session.get_items() == []  # not used: the run's own items only
    sent = [m for b in memory_service.named("messages") for m in b.body["messages"]]
    assert {m.get("source_system") for m in sent} == {SOURCE_SYSTEM}  # the harness's transcript
    in_ram_session.close()


@pytest.mark.usefixtures("memory_window")
async def test_flipped_claude_and_functions_keep_memorys_window(
    memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    async def function(input: Any, agent: Any) -> str:
        return "ok"

    await restarted(memory_service, lambda: function)
    assert windows(memory_service) == [True, True]
