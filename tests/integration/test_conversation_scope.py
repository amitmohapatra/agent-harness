"""A conversation is its tenant's, its user's and its thread's (``identity.conversation_key``):
a thread id is the caller's name for a chat, so two users naming theirs alike never share what
a conversation keeps — the memory service's thread (its window, the OpenAI Agents session read
from it), a graph's own checkpointer thread, the turn a busy conversation waits for, the run
store's ``concurrency_key``, the trace's session — whatever the adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support.adapters import BUILDERS
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Harness, Runtime, Settings
from trellis.contracts import RunStatus
from trellis.harness.harness import LOCAL_TENANT
from trellis.harness.identity import CONVERSATION_PREFIX, Identity, conversation_key

THREAD = "chat-1"  # the same name, given by both users


def test_the_key_is_the_tenants_the_users_and_the_threads() -> None:
    ada = conversation_key("acme", "ada", THREAD)
    assert ada == conversation_key("acme", "ada", THREAD)  # the same wherever it is made
    assert ada.startswith(CONVERSATION_PREFIX) and len(ada) == len(CONVERSATION_PREFIX) + 32
    others = {
        conversation_key("acme", "bob", THREAD),
        conversation_key("globex", "ada", THREAD),
        conversation_key("acme", "ada", "chat-2"),
        conversation_key("acme", "ada:chat", "1"),  # no separator to collide on
        conversation_key("acme:ada", "chat", "1"),
    }
    assert ada not in others and len(others) == 5


def test_a_run_without_a_conversation_is_its_own() -> None:
    alone = Identity(tenant="acme", user="ada", agent_id="a", run_id="run_1")
    assert alone.conversation == "run_1"
    own = alone.model_copy(update={"thread": "run_1"})  # a run's record names itself
    assert own.conversation == "run_1" and own.scope()["thread_id"] == "run_1"
    chat = alone.model_copy(update={"thread": THREAD})
    assert chat.conversation == conversation_key("acme", "ada", THREAD)
    assert chat.scope()["thread_id"] == chat.conversation  # the memory service's thread
    child = chat.model_copy(update={"run_id": "run_2", "parent_run": "run_1"})
    assert child.parent_scope()["thread_id"] == chat.conversation  # the parent's conversation


@pytest.mark.parametrize("framework", sorted(BUILDERS))
async def test_two_users_naming_a_thread_alike_never_share_its_memory(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path, framework: str
) -> None:
    target, _ = await BUILDERS[framework](memory_harness, [], tmp_path, [])
    agent = memory_harness.wrap(target, id=f"scope-{framework}")
    ada = await agent.run("ada's secret is tides", user="ada", thread=THREAD)
    bob = await agent.run("what is the secret?", user="bob", thread=THREAD)
    assert ada.status is RunStatus.SUCCESS, ada.error
    assert bob.status is RunStatus.SUCCESS, bob.error
    await memory_harness.writes.drain()
    keys = {user: conversation_key("acme", user, THREAD) for user in ("ada", "bob")}
    # each user's turns in a memory thread of their own; none under the bare thread id
    assert THREAD not in memory_service.threads
    said = {user: [m["content"] for m in memory_service.threads[k]] for user, k in keys.items()}
    assert "ada's secret is tides" in said["ada"] and "what is the secret?" in said["bob"]
    assert not any("tides" in text for text in said["bob"])
    # and each run's context was asked in its own user's thread
    asked = {c.scope["agent_run_id"]: c.scope["thread_id"] for c in memory_service.named("context")}
    assert asked == {ada.run_id: keys["ada"], bob.run_id: keys["bob"]}


async def test_a_graphs_own_checkpointer_keeps_each_users_conversation_apart(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=["noted", "I know of no secret"])
    graph = create_agent(model, tools=[], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="kept")
    await agent.run("ada's secret is tides", user="ada", thread=THREAD)
    await agent.run("what is the secret?", user="bob", thread=THREAD)
    told_bob = [str(m.content) for m in model.seen[1]]
    assert told_bob == ["what is the secret?"]  # ada's thread is not bob's
    for user, said in (("ada", "ada's secret is tides"), ("bob", "what is the secret?")):
        key = await harness.conversation(user=user, thread=THREAD)
        assert key == conversation_key(LOCAL_TENANT, user, THREAD)
        state = await graph.aget_state({"configurable": {"thread_id": key}})
        assert state.values["messages"][0].content == said


async def test_the_openai_agents_session_reads_only_its_users_thread(
    memory_harness: Harness,
) -> None:
    model = ScriptedModel(["noted", "I know of no secret", "tides"])
    agent = memory_harness.wrap(OpenAIAgent(name="s", model=model), id="sessions")
    await agent.run("ada's secret is tides", user="ada", thread=THREAD)
    await memory_harness.writes.drain()
    await agent.run("what is the secret?", user="bob", thread=THREAD)
    await memory_harness.writes.drain()
    await agent.run("what is my secret?", user="ada", thread=THREAD)

    def told(turn: int) -> list[str]:
        chat = ("user", "assistant")
        return [
            i["content"]
            for i in model.inputs[turn]
            if isinstance(i, dict) and i.get("role") in chat
        ]

    assert told(1) == ["what is the secret?"]  # bob's session holds none of ada's turns
    assert told(2) == ["ada's secret is tides", "noted", "what is my secret?"]  # ada's does


async def test_two_users_naming_a_thread_alike_do_not_wait_for_each_other(
    harness: Harness,
) -> None:
    started: dict[str, asyncio.Event] = {"ada": asyncio.Event(), "bob": asyncio.Event()}

    async def both(input: Any, agent: Runtime) -> str:
        started[agent.user].set()
        await asyncio.wait_for(started["bob" if agent.user == "ada" else "ada"].wait(), 5)
        return "ok"

    agent = harness.wrap(both, id="busy")
    runs = [agent.run("hi", user=user, thread=THREAD) for user in ("ada", "bob")]
    results = await asyncio.wait_for(asyncio.gather(*runs), 10)  # held one after the other: never
    assert [r.status for r in results] == [RunStatus.SUCCESS] * 2


async def test_a_queued_run_waits_only_for_its_own_users_conversation(harness: Harness) -> None:
    async def answer(input: Any, agent: Runtime) -> str:
        return "ok"

    agent = harness.wrap(answer, id="queued")
    tenant = await harness._tenant()

    async def key(**given: Any) -> str | None:
        handle = await agent.start("hi", **given)
        return (await handle.status()).concurrency_key

    ada, bob = await key(user="ada", thread=THREAD), await key(user="bob", thread=THREAD)
    assert ada == f"thread:{conversation_key(tenant, 'ada', THREAD)}"
    assert bob == f"thread:{conversation_key(tenant, 'bob', THREAD)}" and bob != ada
    assert await key(user="ada", thread=THREAD) == ada  # the same conversation still waits
    assert await key(user="ada") is None  # no conversation: nothing to wait for
    assert await key(user="ada", thread=THREAD, concurrency_key="mine") == "mine"


async def test_a_document_added_to_a_thread_is_in_its_users_conversation(
    memory_service: FakeMemoryService,
) -> None:
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        await h.add_document(b"tides", user="ada", thread=THREAD, wait=None)
    [added] = memory_service.named("add_document")
    assert added.scope["thread_id"] == conversation_key("acme", "ada", THREAD)


async def test_the_traces_session_is_the_conversation(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    async def answer(input: Any, agent: Runtime) -> str:
        return "ok"

    agent = harness.wrap(answer, id="traced")
    for user in ("ada", "bob"):
        await agent.run("hi", user=user, thread=THREAD)
    sessions = [
        (s.attributes or {}).get("session.id")
        for s in spans.get_finished_spans()
        if s.name.startswith("invoke_agent")
    ]
    assert sessions == [conversation_key(LOCAL_TENANT, user, THREAD) for user in ("ada", "bob")]
