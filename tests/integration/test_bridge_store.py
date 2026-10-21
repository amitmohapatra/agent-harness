"""The LangGraph store bridge: a graph compiled without a store is run with the memory service
as its long-term store (``bridges.store.MemoryStore``), through LangGraph's own ``BaseStore``
interface — what its nodes read with ``runtime.store``. A graph's own store is kept; memory off
(the deployment's, or the run's ``without=``) leaves LangGraph's default (no store); a memory
service that fails fails the call, as every memory call does."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, NotRequired, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime as GraphRuntime
from langgraph.store.base import BaseStore, GetOp, ListNamespacesOp, MatchCondition, PutOp
from langgraph.store.memory import InMemoryStore

from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings, current
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import bridges
from trellis.harness.bounded import SAID_ONCE, LRUSet
from trellis.harness.bridges.store import MAX_CHARS, MemoryStore
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

PREFS = ("prefs", "ada")


class State(TypedDict):
    topic: str
    seen: NotRequired[Any]


def graph_of(node: Any, *, store: BaseStore | None = None) -> Any:
    graph = StateGraph(State)
    graph.add_node("work", node)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile(store=store)


async def remembers(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
    """Store the topic, read it back, search for it."""
    store = runtime.store
    if store is None:
        return {"seen": "no store"}
    await store.aput(PREFS, "topic", {"text": state["topic"], "n": 1})
    item = await store.aget(PREFS, "topic")
    found = await store.asearch(("prefs",), query=state["topic"])
    assert item is not None
    return {"seen": {"item": item.value, "found": [f.key for f in found]}}


async def reads(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
    store = runtime.store
    assert store is not None
    item = await store.aget(PREFS, "topic")
    return {"seen": None if item is None else item.value}


@pytest.fixture(autouse=True)
def _forget_what_was_said(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridges, "_said", LRUSet(maxsize=SAID_ONCE))


async def test_a_graph_without_a_store_stores_in_the_memory_service(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = memory_harness.wrap(graph_of(remembers), id="prefs")
    result = await agent.run({"topic": "tides"}, user="ada", thread="t1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer is None  # a hand-built graph's answer is its state
    [stored] = [m for m in memory_service.memories.values() if m["current"]]
    assert stored["visibility"] == "USER" and stored["user"] == "ada"
    assert (
        '"key":"topic"' in stored["content"] and '"namespace":["prefs","ada"]' in stored["content"]
    )
    [call] = memory_service.named("remember")
    assert call.scope["agent_run_id"] == result.run_id  # in the run's scope


async def test_what_one_conversation_stored_another_reads(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    seen: list[Any] = []

    async def keep(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
        out = (
            await remembers(state, runtime)
            if state["topic"] != "read"
            else await reads(state, runtime)
        )
        seen.append(out.get("seen"))
        return out

    agent = memory_harness.wrap(graph_of(keep), id="prefs")
    await agent.run({"topic": "tides"}, user="ada", thread="t1")
    await agent.run({"topic": "read"}, user="ada", thread="t2")
    assert seen[0] == {"item": {"text": "tides", "n": 1}, "found": ["topic"]}
    assert seen[1] == {"text": "tides", "n": 1}
    other = await agent.run({"topic": "read"}, user="bob", thread="t3")  # another person's
    assert other.status is RunStatus.SUCCESS and seen[2] is None


async def test_a_graphs_own_store_is_kept_and_said_once(
    memory_harness: Harness, memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    own = InMemoryStore()
    agent = memory_harness.wrap(graph_of(remembers, store=own), id="own")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        for thread in ("t1", "t2"):
            assert (await agent.run({"topic": "tides"}, user="ada", thread=thread)).status is (
                RunStatus.SUCCESS
            )
    said = [r for r in caplog.records if "store of its own" in r.getMessage()]
    assert len(said) == 1
    assert not memory_service.named("remember")  # the memory service is not behind it
    item = await own.aget(PREFS, "topic")
    assert item is not None and item.value == {"text": "tides", "n": 1}


async def test_memory_off_leaves_the_graph_without_a_store(harness: Harness) -> None:
    seen: list[Any] = []

    async def look(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
        seen.append(runtime.store)
        return {}

    agent = harness.wrap(graph_of(look), id="nomem")
    assert (await agent.run({"topic": "tides"}, user="ada")).status is RunStatus.SUCCESS
    assert seen == [None]  # LangGraph's default: no store


@pytest.mark.parametrize("off", [{"memory"}, {"memory_pull"}])
async def test_without_memory_the_graph_has_no_store(
    memory_harness: Harness, memory_service: FakeMemoryService, off: set[str]
) -> None:
    seen: list[Any] = []

    async def look(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
        seen.append(runtime.store)
        return {}

    agent = memory_harness.wrap(graph_of(look), id="off", without=off)  # type: ignore[arg-type]
    await agent.run({"topic": "x"}, user="ada")
    one = memory_harness.wrap(graph_of(look), id="run-off")
    await one.run({"topic": "x"}, user="ada", without=off)  # type: ignore[arg-type]
    assert seen == [None, None]


async def test_a_memory_service_that_fails_fails_the_graphs_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.fail.add("remember")
    agent = memory_harness.wrap(graph_of(remembers), id="down")
    result = await agent.run({"topic": "tides"}, user="ada")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "503" in result.error.message or "unavailable" in result.error.message.lower()
    assert not memory_service.memories  # nothing kept anywhere else


# ---------------------------------------------------------------------- the store's verbs
@pytest.fixture
async def store(memory_service: FakeMemoryService) -> AsyncIterator[MemoryStore]:
    client = memory_service.client()
    yield MemoryStore(client.bind(tenant_id="acme", user_id="ada"), asyncio.get_running_loop())
    await client.aclose()


async def test_a_put_of_a_key_held_supersedes_it_and_none_forgets_it(
    store: MemoryStore, memory_service: FakeMemoryService
) -> None:
    await store.aput(PREFS, "k", {"v": 1})
    await store.aput(PREFS, "k", {"v": 1})  # the same: nothing written
    assert len(memory_service.named("remember")) == 1
    await store.aput(PREFS, "k", {"v": 2})
    [superseded] = memory_service.named("supersede")
    item = await store.aget(PREFS, "k")
    assert item is not None and item.value == {"v": 2}
    assert item.created_at <= item.updated_at
    assert superseded.path["memory_id"] in memory_service.memories
    await store.adelete(PREFS, "k")
    assert await store.aget(PREFS, "k") is None
    await store.adelete(PREFS, "gone")  # nothing held: nothing to forget
    assert len(memory_service.named("forget")) == 1


async def test_two_keys_with_one_value_are_two_items(store: MemoryStore) -> None:
    await store.aput(PREFS, "a", {"v": 1})
    await store.aput(PREFS, "b", {"v": 1})
    assert [i.key for i in await store.asearch(PREFS)] == ["b", "a"]  # newest first


async def test_a_search_filters_ranks_and_pages(
    store: MemoryStore, memory_service: FakeMemoryService
) -> None:
    await store.aput(("docs", "x"), "one", {"text": "tides and moons", "n": 1})
    await store.aput(("docs", "y"), "two", {"text": "tides", "n": 5})
    await store.aput(("other",), "three", {"text": "tides", "n": 9})
    await store.memory.remember("not an item")
    under = await store.asearch(("docs",))
    assert sorted(i.key for i in under) == ["one", "two"] and all(i.score is None for i in under)
    assert [i.key for i in await store.asearch(("docs",), filter={"n": 5})] == ["two"]
    assert [i.key for i in await store.asearch(("docs",), filter={"n": {"$gt": 1}})] == ["two"]
    assert [i.key for i in await store.asearch((), filter={"n": {"$lte": 5, "$ne": 1}})] == ["two"]
    assert await store.asearch((), filter={"missing": {"$lt": 3}}) == []
    assert await store.asearch((), filter={"n": {"$gte": 10}}) == []
    ranked = await store.asearch(("docs",), query="tides moons")
    assert [i.key for i in ranked] == ["one", "two"]
    assert ranked[0].score is not None and ranked[1].score is not None
    assert ranked[0].score > ranked[1].score
    # the service's order (the fake: most words shared, then newest): two, one under docs
    assert [i.key for i in await store.asearch(("docs",), query="tides", offset=1)] == ["one"]
    assert [i.key for i in await store.asearch((), query="tides", limit=1)] == ["three"]
    assert await store.asearch(("docs",), query="nothing") == []
    assert await store.asearch(("docs",), query="tides", filter={"n": 2}) == []


async def test_namespaces_are_listed_matched_and_cut(store: MemoryStore) -> None:
    await store.aput(("a", "x", "1"), "k", {"v": 1})
    await store.aput(("a", "y", "1"), "k", {"v": 1})
    await store.aput(("b", "x"), "k", {"v": 1})
    assert await store.alist_namespaces() == [("a", "x", "1"), ("a", "y", "1"), ("b", "x")]
    assert await store.alist_namespaces(prefix=("a",)) == [("a", "x", "1"), ("a", "y", "1")]
    assert await store.alist_namespaces(suffix=("x",)) == [("b", "x")]
    assert await store.alist_namespaces(prefix=("*", "x")) == [("a", "x", "1"), ("b", "x")]
    assert await store.alist_namespaces(max_depth=1) == [("a",), ("b",)]
    assert await store.alist_namespaces(limit=1, offset=1) == [("a", "y", "1")]
    assert await store.alist_namespaces(prefix=("a", "x", "1", "deeper")) == []
    [everything] = await store.abatch(
        [ListNamespacesOp(match_conditions=(MatchCondition("prefix", ()),))]
    )
    assert isinstance(everything, list) and len(everything) == 3


async def test_what_the_store_cannot_hold_is_refused(store: MemoryStore) -> None:
    with pytest.raises(ConfigurationError, match="at most 8000"):
        await store.aput(PREFS, "big", {"text": "x" * MAX_CHARS})
    with pytest.raises(ConfigurationError, match="no index= or ttl="):
        await store.abatch([PutOp(PREFS, "k", {"v": 1}, index=["text"])])
    with pytest.raises(ConfigurationError, match="no index= or ttl="):
        await store.abatch([PutOp(PREFS, "k", {"v": 1}, ttl=5.0)])
    with pytest.raises(ConfigurationError, match="takes no"):
        await store.abatch([object()])  # type: ignore[list-item]
    assert await store.abatch([PutOp(PREFS, "k", {"v": 1}, index=False)]) == [None]


async def test_a_memory_that_is_not_an_item_is_not_read(
    store: MemoryStore, memory_service: FakeMemoryService
) -> None:
    scope = store.memory
    unmarked = '{"namespace": ["prefs"], "key": "k", "value": {}}'  # someone else's JSON
    texts = ("{not json", '{"langgraph_store": 2}', "[1]", '{"langgraph_store": 1, "key": 3}')
    for text in (*texts, unmarked):
        await scope.remember(text)
    assert await store.asearch(()) == []
    assert await store.asearch((), query="langgraph_store key") == []


async def test_a_sync_call_from_a_worker_thread_reaches_the_service(store: MemoryStore) -> None:
    def in_a_thread() -> Any:
        store.put(PREFS, "k", {"v": 3})
        return store.get(PREFS, "k")

    item = await asyncio.to_thread(in_a_thread)
    assert item is not None and item.value == {"v": 3}
    with pytest.raises(ConfigurationError, match="async methods"):
        store.batch([GetOp(PREFS, "k")])  # on the loop itself it would wait for itself


async def test_a_sync_node_uses_the_store_through_the_loop(memory_harness: Harness) -> None:
    def sync_node(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
        assert runtime.store is not None
        runtime.store.put(PREFS, "sync", {"from": "a thread"})
        item = runtime.store.get(PREFS, "sync")
        return {"seen": None if item is None else item.value}

    agent = memory_harness.wrap(graph_of(sync_node), id="sync")
    result = await agent.run({"topic": "x"}, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error


# ---------------------------------------------------------------------- pauses and restarts
async def test_what_was_stored_before_a_pause_is_there_after_it(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def ask_between(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
        store = runtime.store
        assert store is not None
        await store.aput(PREFS, "draft", {"text": state["topic"]})
        run = current()
        assert run is not None
        answer = await run.ask("ship it?", options=["yes", "no"])
        item = await store.aget(PREFS, "draft")
        assert item is not None
        return {"seen": {"answer": answer, "draft": item.value}}

    agent = memory_harness.wrap(graph_of(ask_between), id="pause")
    paused = await agent.run({"topic": "tides"}, user="ada", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    # the re-run put the same item again: the service holds it once
    assert len([m for m in memory_service.memories.values() if m["current"]]) == 1
    assert len(memory_service.named("remember")) == 1


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def test_a_worker_that_dies_after_a_put_leaves_one_item(
    memory_service: FakeMemoryService,
) -> None:
    crashes = [Crash()]

    async def put_then_die(state: State, runtime: GraphRuntime[None]) -> dict[str, Any]:
        store = runtime.store
        assert store is not None
        await store.aput(PREFS, "once", {"text": "kept"})
        if crashes:
            raise crashes.pop()
        return {"seen": "done"}

    runs = LocalRuns()
    first = Harness(config=Settings(), runs=runs, memory=memory_service.client())
    second = Harness(config=Settings(), runs=runs, memory=memory_service.client())
    try:
        dying = first.wrap(graph_of(put_then_die), id="restart")
        taking_over = second.wrap(graph_of(put_then_die), id="restart")
        handle = await dying.start({"topic": "x"}, user="ada", thread="t")
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
        result = await handle.result(timeout=5)
        assert result.status is RunStatus.SUCCESS, result.error
    finally:
        await first.aclose()
        await second.aclose()
    assert len([m for m in memory_service.memories.values() if m["current"]]) == 1


def test_what_the_bridges_said_once_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridges, "_said", LRUSet(maxsize=2))
    said = [bridges.said(k, "x", extra={}) for k in ("a", "b", "a", "c", "b", "a")]
    assert said == [True, True, False, True, False, True]  # a forgotten past the bound
