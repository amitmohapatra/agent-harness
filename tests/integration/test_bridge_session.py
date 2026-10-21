"""The OpenAI Agents session bridge: an agent run without a session of its own keeps its
conversation in the memory service's thread (``bridges.session.MemorySession``), through the
SDK's own ``Session`` interface. The team's own session is kept; memory off leaves the SDK's
default (no session); a memory service that fails fails the run, as every memory call does."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from agents import Agent, SQLiteSession, function_tool

from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Harness, Settings, current, tool
from trellis.contracts import RunStatus
from trellis.harness.bridges.session import HISTORY_LIMIT, MemorySession
from trellis.harness.clients.memory import SOURCE_SYSTEM
from trellis.harness.runs import LocalRuns
from trellis.memory import MemoryClient
from trellis.runs import Job


def session_messages(service: FakeMemoryService, thread: str) -> list[dict[str, Any]]:
    return [
        m
        for m in service.threads.get(thread, [])
        if (m.get("custom_metadata") or {}).get("trellis_session") == "item"
    ]


def users(input: Any) -> list[str]:
    """The user messages a model was given."""
    return [i["content"] for i in input if isinstance(i, dict) and i.get("role") == "user"]


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return 7


async def test_the_conversation_is_the_memory_services_thread(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel([("stock", {"sku": "a"}), "7 of a", "you asked about a"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="sess", tools=[stock])
    first = await agent.run("how many a?", user="ada", thread="t1")
    assert first.status is RunStatus.SUCCESS and first.answer == "7 of a", first.error
    second = await agent.run("what did I ask?", user="ada", thread="t1")
    assert second.answer == "you asked about a"
    seen = model.inputs[-1]
    assert users(seen) == ["how many a?", "what did I ask?"]  # the earlier turn, from the thread
    kinds = [i.get("type") for i in seen if isinstance(i, dict)]
    assert "function_call" in kinds and "function_call_output" in kinds  # its tool call too
    # the memory context is in each input once, never kept in the thread
    contexts = [i for i in seen if isinstance(i, dict) and i.get("role") == "system"]
    assert contexts == [{"role": "system", "content": memory_service.context_text}]
    kept = session_messages(memory_service, "t1")
    assert [m["role"] for m in kept if m["kind"] == "VISIBLE"] == [
        "USER",
        "ASSISTANT",
        "USER",
        "ASSISTANT",
    ]
    assert {m["role"] for m in kept if m["kind"] == "INTERNAL"} == {"AGENT", "TOOL"}
    # the window is the session's, and the transcript is not written a second time
    assert [c.body["window"] for c in memory_service.named("context")] == [False, False]
    harness_written = [
        m
        for b in memory_service.named("messages")
        for m in b.body["messages"]
        if m.get("source_system") == SOURCE_SYSTEM
    ]
    assert harness_written == []


async def test_another_conversation_starts_empty(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="sess")
    await agent.run("first", user="ada", thread="t1")
    await agent.run("second", user="ada", thread="t2")
    assert users(model.inputs[-1]) == ["second"]


async def test_the_teams_own_session_is_kept(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    own = SQLiteSession("own")
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(
        Agent(name="s", model=model), id="own", framework_options={"session": own}
    )
    await agent.run("first", user="ada", thread="t1")
    await agent.run("second", user="ada", thread="t1")
    assert users(model.inputs[-1]) == ["first", "second"]  # from the team's session
    assert session_messages(memory_service, "t1") == []
    harness_written = [
        m["source_message_id"] for b in memory_service.named("messages") for m in b.body["messages"]
    ]
    assert harness_written and all(m.endswith(("user:0", "msg:1")) for m in harness_written)
    own.close()


async def test_memory_off_leaves_the_sdk_without_a_session(harness: Harness) -> None:
    model = ScriptedModel(["one", "two"])
    agent = harness.wrap(Agent(name="s", model=model), id="nomem")
    await agent.run("first", user="ada", thread="t1")
    await agent.run("second", user="ada", thread="t1")
    assert users(model.inputs[-1]) == ["second"]


@pytest.mark.parametrize("off", ["memory", "records", "memory_push"])
async def test_without_memory_the_sdk_has_no_session(
    memory_harness: Harness, memory_service: FakeMemoryService, off: str
) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="off", without={off})  # type: ignore[arg-type]
    await agent.run("first", user="ada", thread="t1")
    await agent.run("second", user="ada", thread="t1")
    assert users(model.inputs[-1]) == ["second"] or model.inputs[-1] == "second"
    assert session_messages(memory_service, "t1") == []


async def test_a_thread_that_cannot_be_read_fails_the_run(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.fail.add("thread_messages")
    agent = memory_harness.wrap(Agent(name="s", model=ScriptedModel(["one"])), id="down")
    result = await agent.run("first", user="ada", thread="t1")
    assert result.status is RunStatus.ERROR and result.error is not None


# ---------------------------------------------------------------------- pauses and restarts
async def test_an_sdk_approval_continues_the_attempts_items(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        return "sent"

    model = ScriptedModel([("send", {"order": "o9"}), "sent it", "you sent o9"])
    agent = memory_harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="ada", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="ada")
    assert done.status is RunStatus.SUCCESS and done.answer == "sent it", done.error
    await agent.run("what did I send?", user="ada", thread="t1")
    seen = model.inputs[-1]
    assert users(seen) == ["send o9", "what did I send?"]  # the paused run's input, once
    assert [i.get("output") for i in seen if i.get("type") == "function_call_output"] == ["sent"]


async def test_a_run_run_again_from_its_input_is_one_turn(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="read")
    async def confirm() -> str:
        """Ask the person."""
        run = current()
        assert run is not None
        return str(await run.ask("go?", options=["yes", "no"]))

    model = ScriptedModel([("confirm", {}), ("confirm", {}), "went", "you said go"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="asker", tools=[confirm])
    paused = await agent.run("go", user="ada", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert users(model.inputs[1]) == ["go"]  # the re-run reads none of its first attempt
    await agent.run("what did I say?", user="ada", thread="t1")
    seen = model.inputs[-1]
    assert users(seen) == ["go", "what did I say?"]
    assert len([i for i in seen if i.get("type") == "function_call"]) == 1


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def test_a_worker_that_died_mid_run_leaves_one_turn(
    memory_service: FakeMemoryService,
    clients: list[MemoryClient],
) -> None:
    crashes = [Crash()]

    @tool(side_effects="read")
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    model = ScriptedModel([("boom", {}), ("boom", {}), "done", "it was done"])
    runs = LocalRuns()
    first = Harness(config=Settings(), runs=runs, memory=memory_service.client())
    second = Harness(config=Settings(), runs=runs, memory=memory_service.client())
    try:
        dying = first.wrap(Agent(name="s", model=model), id="restart", tools=[boom])
        taking_over = second.wrap(Agent(name="s", model=model), id="restart", tools=[boom])
        handle = await dying.start("work", user="ada", thread="t1")
        worker = first.worker([dying])
        claimed = await runs.claim(worker.worker_id, [dying.id])
        assert claimed is not None
        with pytest.raises(Crash):
            await dying.execute(
                Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=runs)
            )
        held, _ = runs._leases[handle.run_id]
        runs._leases[handle.run_id] = (held, datetime.now(UTC) - timedelta(seconds=1))
        assert await second.worker([taking_over]).run_once()
        assert (await handle.result(timeout=5)).status is RunStatus.SUCCESS
        await taking_over.run("and then?", user="ada", thread="t1")
    finally:
        await first.aclose()
        await second.aclose()
    assert users(model.inputs[-1]) == ["work", "and then?"]


# ---------------------------------------------------------------------- the session's verbs
@pytest.fixture
async def clients() -> AsyncIterator[list[MemoryClient]]:
    """The memory clients a test made, closed when it ends."""
    made: list[MemoryClient] = []
    yield made
    for client in made:
        await client.aclose()


def session_of(
    service: FakeMemoryService,
    clients: list[MemoryClient],
    *,
    run: str = "run_1",
    attempt: int = 1,
    fresh: bool = True,
) -> MemorySession:
    client = service.client()
    clients.append(client)
    ctx = client.bind(tenant_id="acme", user_id="ada", thread_id="t1", agent_run_id=run)
    runtime = SimpleNamespace(
        run_memory=SimpleNamespace(ctx=ctx),
        context="the context",
        run_id=run,
        attempt=attempt,
        thread="t1",
    )
    return MemorySession(runtime, fresh=fresh)  # type: ignore[arg-type]


async def test_pop_and_clear_are_recorded_and_honoured(
    memory_service: FakeMemoryService, clients: list[MemoryClient]
) -> None:
    session = session_of(memory_service, clients)
    assert session.session_id == "t1"
    assert await session.pop_item() is None
    await session.clear_session()  # nothing to clear: nothing written
    assert memory_service.threads == {}
    await session.add_items(
        [
            {"role": "system", "content": "the context"},  # the harness's: never kept
            {"role": "user", "content": "a"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "b", "annotations": []}],
            },
            {"role": "user", "content": [{"type": "input_image", "image_url": "x"}]},
        ]
    )
    await session.add_items([{"role": "system", "content": "the context"}])
    items = await session.get_items()
    assert items[:2] == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert items[2]["content"][0]["type"] == "input_image"  # kept whole, as JSON
    assert await session.get_items(limit=1) == items[-1:]
    assert await session.get_items(limit=0) == []
    assert await session.pop_item() == items[-1]
    assert await session.get_items() == items[:2]
    await session.clear_session()
    assert await session.get_items() == []
    await session.add_items([{"role": "user", "content": "c"}])
    assert await session.get_items() == [{"role": "user", "content": "c"}]


async def test_attempts_of_one_run_are_read_as_one_turn(
    memory_service: FakeMemoryService, clients: list[MemoryClient]
) -> None:
    first = session_of(memory_service, clients, attempt=1)
    await first.add_items(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "half"}]
    )
    again = session_of(memory_service, clients, attempt=2, fresh=True)
    assert await again.get_items() == []  # it runs from its input again
    await again.add_items(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "all"}]
    )
    resumed = session_of(memory_service, clients, attempt=3, fresh=False)
    assert [i["content"] for i in await resumed.get_items()] == ["q", "all"]
    later = session_of(memory_service, clients, run="run_2")
    assert [i["content"] for i in await later.get_items()] == ["q", "all"]
    assert HISTORY_LIMIT == 500  # one page of the service


async def test_a_json_item_that_is_not_an_object_is_read_whole(
    memory_service: FakeMemoryService,
    clients: list[MemoryClient],
) -> None:
    session = session_of(memory_service, clients)
    await session.add_items([{"type": "reasoning", "summary": []}])
    memory_service.threads["t1"][-1]["content"] = "[1, 2]"
    assert await session.get_items() == [{"content": [1, 2]}]


async def test_messages_that_are_not_the_sessions_are_not_read(
    memory_service: FakeMemoryService,
    clients: list[MemoryClient],
) -> None:
    session = session_of(memory_service, clients)
    chat = session._ctx.client.bind(tenant_id="acme", user_id="ada", thread_id="t1")
    await chat.history.add([("USER", "said in a chat app")])
    await session.add_items([{"role": "user", "content": "x"}])
    assert await session.get_items() == [{"role": "user", "content": "x"}]


async def test_a_message_role_the_session_does_not_name_reads_as_the_users(
    memory_service: FakeMemoryService,
    clients: list[MemoryClient],
) -> None:
    session = session_of(memory_service, clients)
    await session.add_items([{"role": "user", "content": "x"}])
    memory_service.threads["t1"][-1]["role"] = "EVENT"
    assert await session.get_items() == [{"role": "user", "content": "x"}]
