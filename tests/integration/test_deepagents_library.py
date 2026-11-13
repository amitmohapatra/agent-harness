"""Deep Agents' own features, each run through the harness as the library gives it: the
general-purpose sub-agent, a compiled graph as a sub-agent, a sub-agent's structured answer, a
sub-agent that forks the conversation; the file tools on a store, on a composite of backends,
and the shell of a local backend; file permissions; memory files (``AGENTS.md``) and skills
read from the backend; a large result moved to a file; its summarization, and the tool that
compacts on request; a rubric; a harness profile; a structured answer; the run's context. Each
test says what the library does and what the harness still does around it. (``write_todos``
and the files in the graph's state: ``test_react_long_tasks``; ``interrupt_on``:
``test_hitl_middleware``; async sub-agents: ``test_react_async_subagent``.)"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents import (
    FilesystemPermission,
    HarnessProfile,
    RubricMiddleware,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.middleware import create_summarization_tool_middleware
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from trellis import Harness
from trellis.contracts import RunEventType, RunStatus

ran: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


def results(events: list[Any]) -> list[tuple[str, str]]:
    """The tool calls a stream reported, in order: each tool's name and how it ended."""
    return [
        (e.data["tool"], e.data["status"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


@lc_tool
def stock(sku: str) -> str:
    """Units of a SKU in stock."""
    ran.append(f"stock {sku}")
    return f"{sku}: 7"


stock.metadata = {"side_effects": "read"}


async def state_of(harness: Harness, graph: Any, thread: str, user: str = "u") -> dict[str, Any]:
    key = await harness.conversation(user=user, thread=thread)
    return (await graph.aget_state({"configurable": {"thread_id": key}})).values


# ------------------------------------------------------------------------------- sub-agents
async def test_the_general_purpose_sub_agent_works_with_the_main_agents_tools(
    harness: Harness,
) -> None:
    """With no sub-agent named, ``task`` has Deep Agents' ``general-purpose`` one, built with
    the main agent's model and tools: its call of a tool is the run's, and the helper is a
    child run."""
    model = ScriptedChatModel(
        turns=[
            ("task", {"description": "count a", "subagent_type": "general-purpose"}),
            ("stock", {"sku": "a"}),  # the sub-agent's
            "a: 7",  # the sub-agent's answer
            "7 units of a",
        ]
    )
    graph = create_deep_agent(model=model, tools=[stock], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="deep")
    events = [e async for e in agent.stream("how many a?", user="u", thread="t")]
    assert events[-1].data["result"] == "7 units of a"
    assert ("stock", "ok") in results(events) and ("task", "ok") in results(events)
    run_id = events[-1].run_id
    kids = [r async for r in harness.runs.iterate(parent_run_id=run_id)]
    assert [k.status for k in kids] == [RunStatus.SUCCESS]


class Verdict(BaseModel):
    ok: bool
    why: str


async def test_a_compiled_graph_and_a_structured_sub_agent_answer_through_task(
    harness: Harness,
) -> None:
    """A ``CompiledSubAgent`` (a graph of your own) and a sub-agent with a ``response_format``:
    ``task`` gives the main agent the compiled graph's last message, and the structured
    answer as JSON."""
    inner = create_agent(
        ScriptedChatModel(turns=[("stock", {"sku": "a"}), "a is fine"]),
        tools=[stock],
        name="inspector",
    )
    judge = ScriptedChatModel(turns=[("Verdict", {"ok": True, "why": "7 left"})])
    model = ScriptedChatModel(
        turns=[
            ("task", {"description": "inspect a", "subagent_type": "inspector"}),
            ("task", {"description": "judge a", "subagent_type": "judge"}),
            "inspected and judged",
        ]
    )
    graph = create_deep_agent(
        model=model,
        subagents=[
            {"name": "inspector", "description": "Inspects a SKU.", "runnable": inner},
            {
                "name": "judge",
                "description": "Judges a SKU.",
                "system_prompt": "Judge it.",
                "model": judge,
                "tools": [],
                "response_format": ToolStrategy(Verdict),
            },
        ],
        checkpointer=InMemorySaver(),
    )
    agent = harness.wrap(graph, id="deep")
    result = await agent.run("inspect and judge a", user="u", thread="t")
    assert result.answer == "inspected and judged" and ran == ["stock a"]
    read = [str(m.content) for m in model.seen[-1] if m.type == "tool"]
    assert read[0] == "a is fine"
    assert '"ok":true' in read[1].replace(" ", "") and "7 left" in read[1]


async def test_a_sub_agent_that_forks_the_conversation_reads_it_and_answers_through_task(
    harness: Harness,
) -> None:
    """``mode="fork"``: the sub-agent goes on from the main agent's conversation instead of
    starting from its task alone. It is a child run of the run like any ``task``."""
    model = ScriptedChatModel(
        turns=[
            ("task", {"description": "say what was asked", "subagent_type": "echo"}),
            "you asked for the tides",  # the fork's
            "done",
        ]
    )
    graph = create_deep_agent(
        model=model,
        subagents=[{"name": "echo", "description": "Goes on from here.", "mode": "fork"}],
        checkpointer=InMemorySaver(),
    )
    result = await harness.wrap(graph, id="deep").run("the tides, please", user="u", thread="t")
    assert result.status is RunStatus.SUCCESS and result.answer == "done"
    forked = [str(m.content) for m in model.seen[1] if m.type == "human"]
    assert forked[0] == "the tides, please" and "say what was asked" in forked[-1]
    assert model.seen[-1][-1].content == "you asked for the tides"
    kids = [r async for r in harness.runs.iterate(parent_run_id=result.run_id)]
    assert [k.status for k in kids] == [RunStatus.SUCCESS]


# -------------------------------------------------------------------------------- backends
async def test_files_on_a_store_of_the_graphs_own_are_kept_past_the_thread(
    harness: Harness,
) -> None:
    """``StoreBackend`` keeps files in the graph's store: written on one thread, read on
    another. The file tools change what outlives the run, so each call is the run's: governed
    (a read runs, a write of the framework's own files is announced), on its events."""
    store = InMemoryStore()

    def graph(model: Any) -> Any:
        return create_deep_agent(
            model=model,
            backend=StoreBackend(namespace=lambda runtime: ("files",)),
            store=store,
            checkpointer=InMemorySaver(),
        )

    write = ("write_file", {"file_path": "/plan.md", "content": "ship friday"})
    writer = harness.wrap(graph(ScriptedChatModel(turns=[write, "kept"])), id="writer")
    events = [e async for e in writer.stream("keep the plan", user="u", thread="one")]
    assert results(events) == [("write_file", "ok")]
    assert [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"] == ["write_file"]
    model = ScriptedChatModel(turns=[("read_file", {"file_path": "/plan.md"}), "friday"])
    reader = harness.wrap(graph(model), id="reader")
    events = [e async for e in reader.stream("when?", user="u", thread="two")]
    assert results(events) == [("read_file", "ok")]
    assert "ship friday" in str(model.seen[-1][-1].content)


async def test_files_on_a_store_go_to_the_memory_service_when_the_graph_has_no_store(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A Deep Agent compiled without a store is run with the memory service as its store: a
    ``StoreBackend`` file is one of the user's memories, and its calls are recorded as the
    run's."""
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/plan.md", "content": "ship friday"}),
            ("read_file", {"file_path": "/plan.md"}),
            "friday",
        ]
    )
    graph = create_deep_agent(
        model=model,
        backend=StoreBackend(namespace=lambda runtime: ("files",)),
        checkpointer=InMemorySaver(),
    )
    done = await memory_harness.wrap(graph, id="deep").run("keep the plan", user="u", thread="t")
    assert done.status is RunStatus.SUCCESS and done.answer == "friday"
    assert "ship friday" in str(model.seen[-1][-1].content)
    assert memory_service.named("remember")
    await memory_harness.writes.drain()
    assert results_named(memory_service) == ["write_file", "read_file"]


async def test_a_composite_backend_routes_each_path_to_its_backend(
    harness: Harness, tmp_path: Path
) -> None:
    """``CompositeBackend``: ``/kept/`` on a disk, the rest in the graph's state. The graph
    keeps its ``files`` channel, so its file tools stay the graph's state tools (not the run's
    tool calls), and each path lands where its route says."""
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/kept/": FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)},
    )
    model = ScriptedChatModel(
        turns=[
            [
                ("write_file", {"file_path": "/scratch.md", "content": "draft"}),
                ("write_file", {"file_path": "/kept/final.md", "content": "final"}),
            ],
            "written",
        ]
    )
    graph = create_deep_agent(model=model, backend=backend, checkpointer=InMemorySaver())
    events = [e async for e in harness.wrap(graph, id="deep").stream("w", user="u", thread="t")]
    assert events[-1].data["result"] == "written" and results(events) == []
    assert (tmp_path / "final.md").read_text() == "final"
    files = (await state_of(harness, graph, "t"))["files"]
    assert set(files) == {"/scratch.md"}


async def test_a_local_shell_backends_execute_runs_the_command_as_a_write_of_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    """``LocalShellBackend`` gives ``execute`` a real shell: the library runs the command in
    its folder, and the call is the run's — a write, announced and on its events."""
    backend = LocalShellBackend(root_dir=str(tmp_path), virtual_mode=True)
    model = ScriptedChatModel(turns=[("execute", {"command": "echo tides > t.txt"}), "done"])
    graph = create_deep_agent(model=model, backend=backend, checkpointer=InMemorySaver())
    events = [e async for e in harness.wrap(graph, id="deep").stream("w", user="u", thread="t")]
    assert results(events) == [("execute", "ok")] and events[-1].data["result"] == "done"
    assert [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"] == ["execute"]
    assert (tmp_path / "t.txt").read_text().strip() == "tides"


async def test_execute_without_a_sandbox_backend_is_the_librarys_own_refusal(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=[("execute", {"command": "ls"}), "no shell"])
    graph = create_deep_agent(model=model, checkpointer=InMemorySaver())
    done = await harness.wrap(graph, id="deep").run("run ls", user="u", thread="t")
    assert done.answer == "no shell"
    assert "not available" in str(model.seen[-1][-1].content).lower()


# ----------------------------------------------------------------------------- permissions
async def test_a_denied_path_is_the_librarys_refusal_and_an_interrupt_path_asks(
    harness: Harness,
) -> None:
    """``permissions``: a write under a ``deny`` rule is refused by the library, as an error
    the model reads; one under an ``interrupt`` rule pauses for a person (LangChain's
    ``HumanInTheLoopMiddleware``), an approval the harness answers."""
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/secrets/key.txt", "content": "k"}),
            ("write_file", {"file_path": "/review/plan.md", "content": "p"}),
            "one refused, one written",
        ]
    )
    graph = create_deep_agent(
        model=model,
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            FilesystemPermission(operations=["write"], paths=["/review/**"], mode="interrupt"),
        ],
        checkpointer=InMemorySaver(),
    )
    agent = harness.wrap(graph, id="deep")
    paused = await agent.run("write both", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.args["file_path"] == "/review/plan.md"
    refused = model.seen[1][-1]
    assert isinstance(refused, ToolMessage) and refused.status == "error"
    assert "permission denied" in str(refused.content).lower()
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == "one refused, one written"
    assert set((await state_of(harness, graph, "t"))["files"]) == {"/review/plan.md"}


# ------------------------------------------------------------------- memory files and skills
async def test_memory_files_and_skills_are_read_from_the_backend_into_the_prompt(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """``memory=[...]`` (``AGENTS.md``) and ``skills=[...]`` are Deep Agents' own: read from
    the backend into the system prompt, beside the memory service's context, which still comes
    with the question."""
    (tmp_path / "AGENTS.md").write_text("Always answer in metric units.\n")
    skill = tmp_path / "skills" / "tides"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: tides\ndescription: How to read a tide table.\n---\n\n# Tides\nRead the table.\n"
    )
    model = ScriptedChatModel(turns=["in metres"])
    graph = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        memory=["/AGENTS.md"],
        skills=["/skills/"],
    )
    result = await memory_harness.wrap(graph, id="deep").run("how high?", user="u")
    assert result.answer == "in metres"
    system = str(model.seen[0][0].content)
    assert "Always answer in metric units." in system
    assert "tides" in system and "How to read a tide table." in system
    asked = model.seen[0][-1]
    assert asked.type == "human" and memory_service.context_text in str(asked.content)
    assert not results_named(memory_service)  # reading them is no tool call of the run


def results_named(service: FakeMemoryService) -> list[str]:
    return [c.body["tool"] for c in service.named("record_tool")]


# ---------------------------------------------------------------------------- large results
async def test_a_large_result_of_the_agents_own_tool_is_moved_to_a_file_it_reads(
    harness: Harness,
) -> None:
    """Deep Agents moves a tool result past its size limit to ``/large_tool_results/`` and
    shows the model a preview; the model reads the rest with ``read_file``."""

    @lc_tool
    def export(sku: str) -> str:
        """Export a SKU's history."""
        return "\n".join(f"{sku} row {n}" for n in range(12_000))

    export.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(turns=[("export", {"sku": "a"}), "exported"])
    graph = create_deep_agent(model=model, tools=[export], checkpointer=InMemorySaver())
    result = await harness.wrap(graph, id="deep").run("export a", user="u", thread="t")
    assert result.answer == "exported"
    shown = str(model.seen[-1][-1].content)
    assert len(shown) < 20_000 and "/large_tool_results/" in shown
    files = (await state_of(harness, graph, "t"))["files"]
    assert any(path.startswith("/large_tool_results/") for path in files)


# ---------------------------------------------------------------- structured answer, context
class Shop(BaseModel):
    region: str


async def test_a_deep_agents_structured_answer_and_context_are_the_librarys(
    harness: Harness,
) -> None:
    seen: list[str] = []

    @lc_tool
    def region(runtime: ToolRuntime[Shop, Any]) -> str:
        """The shop's region."""
        seen.append(runtime.context.region)
        return runtime.context.region

    region.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(turns=[("region", {}), ("Verdict", {"ok": True, "why": "eu"})])
    graph = create_deep_agent(
        model=model,
        tools=[region],
        response_format=ToolStrategy(Verdict),
        context_schema=Shop,
    )
    result = await harness.wrap(graph, id="deep").run(
        "where?", user="u", framework_options={"context": Shop(region="eu")}
    )
    assert result.status is RunStatus.SUCCESS and seen == ["eu"]
    assert result.answer == Verdict(ok=True, why="eu")


# ------------------------------------------------------- summarization, a rubric, a profile
async def test_deep_agents_summarizes_a_long_conversation_and_keeps_its_history_to_read(
    harness: Harness,
) -> None:
    """Past its trigger Deep Agents has its model summarize the older messages, sends the
    summary in their place and writes what it took out to ``/conversation_history/`` on its
    backend — as it does without the harness."""
    model = ScriptedChatModel(turns=["They spoke of tides.", "and now: the moon"])
    graph = create_deep_agent(model=model, checkpointer=InMemorySaver())
    long = "tides " * 60_000
    said: list[dict[str, str]] = []
    for n in range(4):
        said += [
            {"role": "user", "content": f"{n} {long}"},
            {"role": "assistant", "content": f"noted {n}"},
        ]
    given = {"messages": [*said, {"role": "user", "content": "and now?"}]}
    result = await harness.wrap(graph, id="deep").run(given, user="u", thread="t")
    assert result.status is RunStatus.SUCCESS and result.answer == "and now: the moon"
    assert len(model.seen) == 2 and "They spoke of tides." in str(model.seen[1][1].content)
    assert len(model.seen[1]) < len(given["messages"])
    files = (await state_of(harness, graph, "t"))["files"]
    assert [path.rpartition("/")[0] for path in files] == ["/conversation_history"]


async def test_the_compaction_tool_asks_as_any_tool_that_says_nothing_then_runs(
    harness: Harness,
) -> None:
    """``compact_conversation`` (Deep Agents' ``SummarizationToolMiddleware``) declares no side
    effects and is not one of the state tools the harness knows by name: a call asks a person
    first, then the library's tool answers."""
    model = ScriptedChatModel(turns=[("compact_conversation", {}), "nothing to compact"])
    compacts = create_summarization_tool_middleware(model, StateBackend())
    graph = create_deep_agent(model=model, middleware=[compacts], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="deep")
    paused = await agent.run("compact", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "compact_conversation"
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == "nothing to compact"
    assert "Nothing to compact yet" in str(model.seen[-1][-1].content)


async def test_a_rubric_has_the_answer_graded_and_revised(harness: Harness) -> None:
    """``RubricMiddleware``: given a ``rubric`` in the input, a grader (a model of its own)
    judges the answer, and the agent is asked to revise it until the grader is satisfied. The
    run's answer is the revised one."""

    def verdict(passed: bool) -> tuple[str, dict[str, Any]]:
        criterion = {"name": "Says the unit", "passed": passed, **({} if passed else {"gap": "m?"})}
        result = "satisfied" if passed else "needs_revision"
        return "GraderResponse", {"result": result, "explanation": "units", "criteria": [criterion]}

    grader = ScriptedChatModel(turns=[verdict(False), verdict(True)])
    model = ScriptedChatModel(turns=["7", "7 metres"])
    graph = create_deep_agent(model=model, middleware=[RubricMiddleware(model=grader)])
    given = {"messages": [{"role": "user", "content": "how high?"}], "rubric": "Says the unit"}
    result = await harness.wrap(graph, id="deep").run(given, user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "7 metres"
    assert len(grader.seen) == 2 and len(model.seen) == 2


class Profiled(ScriptedChatModel):
    """A scripted model of a provider of its own (LangChain names a provider after its model's
    class: ``profiled``), for a harness profile no other test's model takes."""


async def test_a_harness_profile_takes_tools_away_and_adds_to_the_prompt(harness: Harness) -> None:
    """Deep Agents' ``HarnessProfile`` for a provider: the tools it excludes are not offered,
    and its suffix ends the system prompt (before what the harness adds to it)."""
    register_harness_profile(
        "profiled",
        HarnessProfile(
            excluded_tools=frozenset({"execute", "grep"}), system_prompt_suffix="Answer briefly."
        ),
    )
    model = Profiled(turns=["hi"])
    result = await harness.wrap(create_deep_agent(model=model), id="deep").run("hi", user="u")
    assert result.answer == "hi"
    offered = set(model.requests[0]["tools"])
    assert offered >= {"ls", "read_file", "task"} and not offered & {"execute", "grep"}
    assert "Answer briefly." in str(model.seen[0][0].content)
