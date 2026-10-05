"""Sub-agents: ``agent.as_tool()`` makes any wrapped agent a tool of any other; each call is a
child run — named by its parent's call, inheriting its tenant, user, thread, time and trace —
that answers, fails, or pauses its parent with its question (the parent's resume answers it);
after a crash the parent's next attempt continues it from the journal its progress saved;
cancelling the parent cancels it."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from agents import function_tool
from langchain.agents import create_agent
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, Runtime, tool
from trellis.contracts import ConfigurationError, RunRecord, RunStatus
from trellis.harness import telemetry
from trellis.harness.runs import LocalRuns
from trellis.harness.subagents import SUBAGENT

done: list[str] = []


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up."""
    done.append(f"lookup {topic}")
    return f"facts about {topic}"


@tool(side_effects="write")
def file(note: str) -> str:
    """File a note."""
    done.append(f"file {note}")
    return f"filed {note}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


async def children(h: Harness, parent_run_id: str) -> list[RunRecord]:
    found = [s async for s in h.runs.iterate(parent_run_id=parent_run_id)]
    records = [await h.runs.get(s.run_id) for s in found]
    return sorted((r for r in records if r is not None), key=lambda r: r.agent_id)


def researcher(h: Harness, name: str = "research") -> Any:
    async def research(input: str, agent: Runtime) -> str:
        """Research a topic and report what was found."""
        return str(await agent.tools.call("lookup", topic=input))

    return h.wrap(research, id=name, tools=[lookup], version="r-2")


async def test_a_child_run_of_any_framework_answers_its_parents_call(harness: Harness) -> None:
    graph = create_agent(
        ScriptedChatModel(turns=[("lookup", {"topic": "graphs"}), "graphs are graphs"]),
        tools=await harness.tools(lookup, framework="langgraph"),
    )
    openai = OpenAIAgent(
        name="o", model=ScriptedModel([("lookup", {"topic": "sdks"}), "sdks are sdks"])
    )
    react = ReAct(system="r", model=ScriptedChat([("lookup", {"topic": "loops"}), "loops"]))
    kids = [
        researcher(harness),
        harness.wrap(graph, id="graph"),
        harness.wrap(openai, id="openai", tools=[lookup]),
        harness.wrap(react, id="react", tools=[lookup]),
    ]
    model = ScriptedChat(
        [
            [("research", {"message": "trees"}), ("graph", {"message": "g"})],
            [("openai", {"message": "o"}), ("react", {"message": "r"})],
            "all four answered",
        ]
    )
    parent = harness.wrap(
        ReAct(system="p", model=model), id="lead", tools=[k.as_tool() for k in kids]
    )
    result = await parent.run("research", user="ada", thread="t1")
    assert result.status is RunStatus.SUCCESS, result.error
    told = [m["content"] for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert told == ["facts about trees", "graphs are graphs"]
    told = [m["content"] for m in model.requests[2]["messages"] if m["role"] == "tool"]
    assert told[2:] == ["sdks are sdks", "loops"]
    records = await children(harness, result.run_id)
    assert [r.agent_id for r in records] == ["graph", "openai", "react", "research"]
    for record in records:
        assert record.status is RunStatus.SUCCESS
        assert (record.user_id, record.thread_id, record.tenant_id) == ("ada", "t1", "default")
    assert records[3].agent_version == "r-2" and records[3].input == "trees"
    assert {
        t["function"]["name"]: t["function"]["description"] for t in model.requests[0]["tools"]
    }["research"] == "Research a topic and report what was found."


async def test_children_that_only_read_run_at_once(harness: Harness) -> None:
    together = asyncio.Barrier(2)

    async def waits(input: str, agent: Runtime) -> str:
        await asyncio.wait_for(together.wait(), 1)  # only when both run at once
        return str(await agent.tools.call("lookup", topic=input))

    first = harness.wrap(waits, id="first", tools=[lookup])
    second = harness.wrap(waits, id="second", tools=[lookup])
    model = ScriptedChat([[("first", {"message": "a"}), ("second", {"message": "b"})], "both"])
    tools = [first.as_tool(), second.as_tool()]
    parent = harness.wrap(ReAct(system="p", model=model), id="pair", tools=tools)
    assert (await parent.run("x", user="u")).answer == "both"


async def test_what_a_child_does_decides_whether_its_tool_reads(harness: Harness) -> None:
    async def nothing(input: str, agent: Runtime) -> str:
        return input

    async def tool_of(agent: Any, **kwargs: Any) -> Any:
        [made] = await agent.as_tool(**kwargs).resolve()
        return made.spec

    reads = harness.wrap(nothing, id="reads", tools=[lookup])
    writes = harness.wrap(nothing, id="writes", tools=[lookup, file])
    assert (await tool_of(reads)).side_effects == "read"
    assert (await tool_of(writes)).side_effects == "write"
    assert (await tool_of(writes, side_effects="read", name="w")).side_effects == "read"
    assert (await tool_of(harness.wrap(nothing, id="bare"))).description.startswith(
        "Ask the bare agent to do a task"
    )

    @function_tool
    def own(x: str) -> str:
        """The SDK's own tool: the harness does not run it."""
        return x

    model = ScriptedModel(["x"])
    sdk = harness.wrap(OpenAIAgent(name="o", model=model, tools=[own]), id="sdk")
    assert (await tool_of(sdk)).side_effects == "write"
    graph = create_agent(ScriptedChatModel(turns=["x"]), tools=[own_langchain])
    assert (await tool_of(harness.wrap(graph, id="graph"))).side_effects == "write"
    from claude_agent_sdk import ClaudeAgentOptions

    claude = harness.wrap(ClaudeAgentOptions(), id="claude")
    assert (await tool_of(claude)).side_effects == "write"


def own_langchain(x: str) -> str:
    """A graph's own tool, not built with h.tools."""
    return x


async def test_a_child_that_asks_pauses_its_parent_and_the_answer_reaches_it(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    asked: list[str] = []

    @tool(side_effects="irreversible")
    def publish(text: str) -> str:
        """Publish a text."""
        done.append(f"publish {text}")
        return f"published {text}"

    async def editor(input: str, agent: Runtime) -> str:
        asked.append(input)
        tone = await agent.ask("Which tone?", options=["warm", "dry"])
        return str(await agent.tools.call("publish", text=f"{input} ({tone})"))

    child = memory_harness.wrap(editor, id="editor", tools=[publish])
    model = ScriptedChat([("editor", {"message": "news"}), "published"])
    parent = memory_harness.wrap(ReAct(system="p", model=model), id="desk", tools=[child.as_tool()])
    paused = await parent.run("publish the news", user="ada")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    question = paused.interrupt
    assert question.run_id == paused.run_id and question.question == "Which tone?"
    assert question.options == ["warm", "dry"] and question.assignee == "user:ada"
    [kid] = await children(memory_harness, paused.run_id)
    assert kid.status is RunStatus.PAUSED and kid.awaiting is not None
    assert question.payload == {
        SUBAGENT: {
            "agent_id": "editor",
            "run_id": kid.run_id,
            "interrupt_id": kid.awaiting.interrupt_id,
        }
    }
    # the person sees the question once, on the parent; the child is answered through it
    assert [s.run_id for s in await memory_harness.inbox("user:ada")] == [paused.run_id]
    with pytest.raises(ConfigurationError, match="is a sub-agent's run: answer the question"):
        await child.resume(kid.awaiting.interrupt_id, "answer", answer="dry", reviewer="ed")
    approval = await parent.resume(question.interrupt_id, "answer", answer="dry", reviewer="ed")
    assert approval.status is RunStatus.PAUSED and approval.interrupt is not None
    assert approval.interrupt.tool_call is not None
    assert approval.interrupt.tool_call.args == {"text": "news (dry)"}
    finished = await parent.resume(approval.interrupt.interrupt_id, "approve", reviewer="ed")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "published"
    assert done == ["publish news (dry)"] and asked == ["news", "news", "news"]
    [kid] = await children(memory_harness, paused.run_id)
    assert kid.status is RunStatus.SUCCESS and kid.attempt == 3
    await memory_harness.writes.drain()
    # the approval is the child's feedback, where the call was made, once
    [feedback] = [f for f in memory_service.named("feedback") if f.body.get("verdict") == "approve"]
    assert feedback.body["agent_run_id"] == kid.run_id


async def test_a_child_that_fails_is_an_error_its_parents_model_reads(harness: Harness) -> None:
    async def broken(input: str, agent: Runtime) -> str:
        raise ValueError("no data")

    child = harness.wrap(broken, id="broken")
    model = ScriptedChat([("broken", {"message": "x"}), ("broken", {"message": "x"}), "gave up"])
    parent = harness.wrap(ReAct(system="p", model=model), id="boss", tools=[child.as_tool()])
    result = await parent.run("x", user="u")
    assert result.answer == "gave up"
    told = model.requests[1]["messages"][-1]["content"]
    assert told == "broken failed: the broken agent's run ended ERROR: no data"


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def test_a_worker_killed_inside_a_child_repeats_none_of_its_side_effects(
    harness: Harness,
) -> None:
    crashes = [Crash()]

    @tool(side_effects="read")
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    async def filer(input: str, agent: Runtime) -> str:
        filed = await agent.tools.call("file", note=input)
        await agent.tools.call("boom")
        return str(filed)

    child = harness.wrap(filer, id="filer", tools=[file, boom])
    model = ScriptedChat([("filer", {"message": "n1"}), "filed n1"])
    parent = harness.wrap(ReAct(system="p", model=model), id="clerk", tools=[child.as_tool()])
    store = harness.runs
    assert isinstance(store, LocalRuns)
    handle = await parent.start("file n1", user="u")
    worker = harness.worker([parent])
    claimed = await store.claim(worker.worker_id, [parent.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await parent._claimed(claimed.run, worker.worker_id, lease_seconds=60)
    held, _ = store._leases[handle.run_id]
    store._leases[handle.run_id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    [kid] = await children(harness, handle.run_id)
    assert kid.status is RunStatus.RUNNING  # cut by the crash
    saved = (await handle.status()).checkpoint
    assert saved is not None and kid.run_id in saved["children"]
    assert await harness.worker([parent]).run_once()
    result = await handle.result(timeout=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "filed n1", result.error
    assert done == ["file n1"]  # filed once, before the crash
    [kid] = await children(harness, handle.run_id)
    assert kid.status is RunStatus.SUCCESS and kid.output == "filed n1"
    assert len(model.requests) == 2  # the parent's first step was replayed


async def test_cancelling_a_parent_cancels_its_children(harness: Harness) -> None:
    started = asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    async def asks(input: str, agent: Runtime) -> str:
        return str(await agent.ask("Go on?"))

    running = harness.wrap(slow, id="slow")
    model = ScriptedChat([("slow", {"message": "x"})])
    parent = harness.wrap(ReAct(system="p", model=model), id="p1", tools=[running.as_tool()])
    task = asyncio.create_task(parent.run("x", user="u"))
    await asyncio.wait_for(started.wait(), 5)
    [record] = [s async for s in harness.runs.iterate(agent_id="p1")]
    await parent.cancel(record.run_id, reason="stop")
    assert (await task).status is RunStatus.CANCELLED
    [kid] = await children(harness, record.run_id)
    assert kid.status is RunStatus.CANCELLED

    waiting = harness.wrap(asks, id="asks")
    for reason in ("cancel", "resume"):
        model = ScriptedChat([("asks", {"message": "x"})])
        tools = [waiting.as_tool()]
        parent = harness.wrap(ReAct(system="p", model=model), id=f"p-{reason}", tools=tools)
        paused = await parent.run("x", user="u")
        assert paused.interrupt is not None
        if reason == "cancel":
            await parent.cancel(paused.run_id, reason="no longer needed")
        else:
            ended = await parent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="r")
            assert ended.status is RunStatus.CANCELLED
        [kid] = await children(harness, paused.run_id)
        assert kid.status is RunStatus.CANCELLED  # nobody is left waiting on it


async def test_a_child_has_what_is_left_of_its_parents_time_and_its_deadline(
    harness: Harness,
) -> None:
    async def slow(input: str, agent: Runtime) -> str:
        await asyncio.sleep(30)
        return "never"

    child = harness.wrap(slow, id="slow")
    model = ScriptedChat([("slow", {"message": "x"})])
    parent = harness.wrap(ReAct(system="p", model=model), id="timed", tools=[child.as_tool()])
    deadline = datetime.now(UTC) + timedelta(hours=1)
    result = await parent.run("x", user="u", timeout=0.3, deadline=deadline)
    assert result.status is RunStatus.TIMEOUT
    [kid] = await children(harness, result.run_id)
    assert kid.timeout_seconds is not None and kid.timeout_seconds <= 0.3
    assert kid.deadline == deadline
    assert kid.final  # it stopped with its parent


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


async def test_a_childs_spans_are_in_its_parents_trace_under_the_call(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    model = ScriptedChat([("research", {"message": "x"}), "done"])
    parent = harness.wrap(
        ReAct(system="p", model=model), id="lead", tools=[researcher(harness).as_tool()]
    )
    result = await parent.run("x", user="u")
    by_name = {s.name: s for s in spans.get_finished_spans()}
    lead, research = by_name["invoke_agent lead"], by_name["invoke_agent research"]
    call = by_name["execute_tool research"]
    assert lead.context is not None and research.context is not None and call.context is not None
    assert (
        research.context.trace_id == lead.context.trace_id == telemetry.trace_id_of(result.run_id)
    )
    assert research.parent is not None and research.parent.span_id == call.context.span_id
    attributes = dict(research.attributes or {})
    assert attributes["trellis.parent_run_id"] == result.run_id
    assert "langfuse.trace.name" not in attributes


@pytest.mark.parametrize("framework", ["langgraph", "openai_agents", "function"])
async def test_any_framework_calls_a_child_and_answers_its_question(
    harness: Harness, framework: str
) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        """Ask which colour, then paint."""
        colour = await agent.ask("Which colour?", options=["red", "blue"])
        return f"painted {input} {colour}"

    child = harness.wrap(asks, id="painter")
    call = ("painter", {"message": "the door"})
    if framework == "langgraph":
        model = ScriptedChatModel(turns=[call, call, "done"])
        tools = await harness.tools(child.as_tool(), framework="langgraph")
        parent = harness.wrap(create_agent(model, tools=tools), id="graph-parent")
    elif framework == "openai_agents":
        sdk = OpenAIAgent(name="p", model=ScriptedModel([call, call, "done"]))
        parent = harness.wrap(sdk, id="sdk-parent", tools=[child.as_tool()])
    else:

        async def delegates(input: str, agent: Runtime) -> Any:
            return await agent.tools.call("painter", message="the door")

        parent = harness.wrap(delegates, id="fn-parent", tools=[child.as_tool()])
    paused = await parent.run("paint", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question == "Which colour?"
    result = await parent.resume(
        paused.interrupt.interrupt_id, "answer", answer="blue", reviewer="r"
    )
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer in ("done", "painted the door blue")
    [kid] = await children(harness, paused.run_id)
    assert kid.status is RunStatus.SUCCESS and kid.output == "painted the door blue"
