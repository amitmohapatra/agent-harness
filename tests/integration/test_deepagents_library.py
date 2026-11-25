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
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.runlog import LoggedRuns
from trellis import Harness, Settings, middleware
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness.adapters.langgraph import graph_tools
from trellis.harness.middleware import RunCheckpointer

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


async def test_a_sub_agents_calls_are_on_its_parents_log_between_its_start_and_end() -> None:
    """SUB-HELPER-LOG (docs/subagents.md, Limits): a framework's own helper is a step of its
    parent's run — its tool calls are on the parent's log in agent-runs, between
    ``helper_started`` and ``helper_finished``, as Deep Agents streams them in the run that
    spawned it; its child run logs no event of its own (each call is logged once)."""
    model = ScriptedChatModel(
        turns=[
            ("task", {"description": "count a", "subagent_type": "general-purpose"}),
            ("stock", {"sku": "a"}),
            "a: 7",
            "7 units of a",
        ]
    )
    runs = LoggedRuns()
    async with Harness(config=Settings(), runs=runs) as h:
        graph = create_deep_agent(model=model, tools=[stock], checkpointer=InMemorySaver())
        result = await h.wrap(graph, id="deep").run("how many a?", user="u", thread="t")
        [kid] = [r async for r in h.runs.iterate(parent_run_id=result.run_id)]
    steps = [
        e.data.get("name") or e.data.get("tool")
        for e in runs.logged(result.run_id)
        if e.type in (RunEventType.CUSTOM, RunEventType.TOOL_CALL_RESULT)
    ]
    assert steps == ["helper_started", "stock", "helper_finished", "task"]
    assert runs.logged(kid.run_id) == []


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
@pytest.mark.usefixtures(
    "writes_unasked"
)  # not about the write's approval (it asks: test_approval_rule)
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


@pytest.mark.usefixtures(
    "writes_unasked"
)  # not about the write's approval (it asks: test_approval_rule)
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


def routed(tmp_path: Path) -> CompositeBackend:
    """``/kept/`` on a disk, the rest in the graph's state."""
    return CompositeBackend(
        default=StateBackend(),
        routes={"/kept/": FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)},
    )


@pytest.mark.usefixtures(
    "writes_unasked"
)  # not about the write's approval (it asks: test_approval_rule)
async def test_a_composite_backend_routes_each_path_to_its_backend(
    harness: Harness, tmp_path: Path
) -> None:
    """SEC: ``CompositeBackend``, ``/kept/`` on a disk and the rest in the graph's state. Each
    path lands where its route says, and each call is what its path makes it: a file in the
    graph's state is the graph's scratch (no call of the run), a file on the disk is a write
    of the run — governed, announced and recorded — though the graph has a ``files`` channel."""
    model = ScriptedChatModel(
        turns=[
            [
                ("write_file", {"file_path": "/scratch.md", "content": "draft"}),
                ("write_file", {"file_path": "/kept/final.md", "content": "final"}),
            ],
            [
                ("read_file", {"file_path": "/scratch.md"}),
                ("read_file", {"file_path": "/kept/final.md"}),
            ],
            "written",
        ]
    )
    graph = create_deep_agent(model=model, backend=routed(tmp_path), checkpointer=InMemorySaver())
    events = [e async for e in harness.wrap(graph, id="deep").stream("w", user="u", thread="t")]
    assert events[-1].data["result"] == "written"
    assert results(events) == [("write_file", "ok"), ("read_file", "ok")]
    said = [e.data for e in events if e.type is RunEventType.TOOL_CALL_ARGS]
    assert all("/kept/final.md" in str(d) for d in said) and len(said) == 2
    assert [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"] == ["write_file"]
    assert (tmp_path / "final.md").read_text() == "final"
    files = (await state_of(harness, graph, "t"))["files"]
    assert set(files) == {"/scratch.md"}


@pytest.mark.usefixtures(
    "writes_unasked"
)  # not about the write's approval (it asks: test_approval_rule)
@pytest.mark.parametrize(
    ("call", "governed"),
    [
        (("ls", {"path": "/"}), True),  # lists the disk route's folder too
        (("glob", {"pattern": "**/*.md", "path": "/"}), True),
        (("grep", {"pattern": "final"}), True),  # no path: every route is searched
        (("glob", {"pattern": "*.md", "path": "/kept/"}), True),
        (("ls", {"path": "/notes/"}), False),  # a folder of the state alone
        (
            ("edit_file", {"file_path": "/kept/final.md", "old_string": "a", "new_string": "b"}),
            True,
        ),
        (("edit_file", {"file_path": "/s.md", "old_string": "a", "new_string": "b"}), False),
        # a path Deep Agents refuses cannot be placed: governed, it fails closed
        (("read_file", {"file_path": "../kept/final.md"}), True),
    ],
)
async def test_a_composite_backends_call_is_governed_when_it_reaches_past_the_state(
    harness: Harness, tmp_path: Path, call: tuple[str, dict[str, Any]], governed: bool
) -> None:
    """SEC, the boundary: a search or a listing from a folder that holds a route outside the
    state reads that route too, so it is the run's call; one within the state alone is not."""
    (tmp_path / "final.md").write_text("a final")
    model = ScriptedChatModel(turns=[call, "done"])
    graph = create_deep_agent(model=model, backend=routed(tmp_path), checkpointer=InMemorySaver())
    events = [e async for e in harness.wrap(graph, id="deep").stream("w", user="u", thread="t")]
    assert events[-1].data["result"] == "done"
    assert [tool for tool, _ in results(events)] == ([call[0]] if governed else [])


async def test_a_composite_backends_write_past_the_state_waits_for_its_approval(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """SEC: the catalog's rule for a file tool holds on a composite's disk route as on a disk
    alone — a write there that needs a person's approval does not run before it is given, and
    never once rejected; a write to the state route is the graph's scratch, never asked."""
    memory_service.catalog = {"write_file": {"side_effects": "irreversible"}}
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/scratch.md", "content": "draft"}),
            ("write_file", {"file_path": "/kept/final.md", "content": "final"}),
            "not written",
        ]
    )
    graph = create_deep_agent(model=model, backend=routed(tmp_path), checkpointer=InMemorySaver())
    agent = memory_harness.wrap(graph, id="deep")
    paused = await agent.run("w", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    asked = paused.interrupt.tool_call
    assert asked is not None and asked.args["file_path"] == "/kept/final.md"
    assert not (tmp_path / "final.md").exists()
    done = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="u")
    assert done.status is RunStatus.SUCCESS and not (tmp_path / "final.md").exists()
    assert set((await state_of(memory_harness, graph, "t"))["files"]) == {"/scratch.md"}


@lc_tool
def todo_count(runtime: ToolRuntime[Any, Any]) -> str:
    """How many todos the plan in the graph's state holds."""
    return f"{len(runtime.state.get('todos') or [])} todos"


todo_count.metadata = {"side_effects": "read"}


@pytest.mark.parametrize("kept", ["no thread", "the run"])
async def test_a_conversations_next_turn_has_the_files_and_todos_of_its_last(
    harness: Harness, kept: str
) -> None:
    """CONV: a Deep Agents graph with no thread of its own (compiled without a checkpointer,
    or checkpointing into its run) keeps its state files and its todos across the turns of
    one conversation, as a thread of the graph's own would: a file one turn wrote is read by
    the next, one a turn deleted stays deleted, and another conversation has none of them."""
    turns: list[Any] = [
        [
            ("write_file", {"file_path": "/notes.md", "content": "tides at six"}),
            ("write_file", {"file_path": "/draft.md", "content": "a draft"}),
            ("write_todos", {"todos": [{"content": "check tides", "status": "pending"}]}),
        ],
        "noted",
        [
            ("read_file", {"file_path": "/notes.md"}),
            ("delete", {"file_path": "/draft.md"}),
            ("todo_count", {}),
        ],
        "read",
        [("read_file", {"file_path": "/notes.md"}), ("read_file", {"file_path": "/draft.md"})],
        "again",
        ("read_file", {"file_path": "/notes.md"}),
        "elsewhere",
    ]
    model = ScriptedChatModel(turns=turns)
    saver = RunCheckpointer() if kept == "the run" else None
    graph = create_deep_agent(
        model=model, tools=[todo_count], middleware=[TodoListMiddleware()], checkpointer=saver
    )
    agent = harness.wrap(graph, id="deep")

    def read(n: int) -> list[str]:
        """What the model's ``n``-th call read of its tools, the latest results last."""
        return [str(m.content) for m in model.seen[n] if isinstance(m, ToolMessage)]

    for said in ("note it", "read it", "read both"):
        assert (await agent.run(said, user="u", thread="t")).status is RunStatus.SUCCESS
    assert "tides at six" in read(3)[-3]  # the second turn read the first turn's file
    assert read(3)[-1] == "1 todos"  # and has its todos, which LangChain omits from an input
    third = read(5)
    assert "tides at six" in third[-2] and "not found" in third[-1]  # deleted stays deleted
    assert (await agent.run("read it", user="u", thread="other")).answer == "elsewhere"
    assert "not found" in read(7)[-1]  # another conversation's files are its own


@pytest.mark.parametrize("kept", ["no thread", "the run"])
async def test_files_and_todos_reach_a_turn_on_a_worker_that_kept_none(
    memory_harness: Harness, kept: str
) -> None:
    """CONV: memory keeps the conversation, and the file store the files and todos its last turn
    ended with (``pipeline._keep_carried``): a turn served by another worker — this process's
    own copy gone, as after a restart — reads the file the earlier turn wrote and has its
    todos, as a thread of the graph's own would give them. Another conversation has none."""
    from trellis.harness.tools.results import Conversations

    turns: list[Any] = [
        [
            ("write_file", {"file_path": "/notes.md", "content": "tides at six"}),
            ("write_todos", {"todos": [{"content": "check tides", "status": "pending"}]}),
        ],
        "noted",
        [("read_file", {"file_path": "/notes.md"}), ("todo_count", {})],
        "read",
        ("read_file", {"file_path": "/notes.md"}),
        "elsewhere",
    ]
    model = ScriptedChatModel(turns=turns)
    saver = RunCheckpointer() if kept == "the run" else None
    graph = create_deep_agent(
        model=model, tools=[todo_count], middleware=[TodoListMiddleware()], checkpointer=saver
    )
    agent = memory_harness.wrap(graph, id="deep-moved")
    assert (await agent.run("note it", user="u", thread="t")).status is RunStatus.SUCCESS
    agent.results = Conversations()  # another worker: nothing of the conversation in memory
    assert (await agent.run("read it", user="u", thread="t")).status is RunStatus.SUCCESS
    read = [str(m.content) for m in model.seen[3] if isinstance(m, ToolMessage)]
    assert "tides at six" in read[-2] and read[-1] == "1 todos"
    agent.results = Conversations()
    assert (await agent.run("read it", user="u", thread="other")).answer == "elsewhere"
    assert "not found" in [str(m.content) for m in model.seen[5] if isinstance(m, ToolMessage)][-1]


async def test_a_file_past_the_bound_reaches_another_worker_whole_and_goes_when_deleted(
    memory_harness: Harness,
) -> None:
    """CONV: a file larger than one turn's file in the store holds (``CARRIED_BYTES``) is kept
    whole, as an object of its own (``pipeline._fitted``): a turn on a worker that kept none
    reads its last line; once a turn deletes it, its object leaves the store."""
    from trellis.harness.pipeline import CARRIED_BYTES, CARRIED_FOLDER, CARRIED_OBJECTS
    from trellis.harness.tools.results import Conversations

    lines = [f"row {n:05d}: " + "x" * 40 for n in range(7_000)] + ["END-OF-BIG"]
    big = "\n".join(lines)
    assert len(big) > CARRIED_BYTES
    turns: list[Any] = [
        [
            ("write_file", {"file_path": "/big.csv", "content": big}),
            ("write_todos", {"todos": [{"content": "check rows", "status": "pending"}]}),
        ],
        "kept",
        [
            ("read_file", {"file_path": "/big.csv", "offset": 6_995, "limit": 10}),
            ("todo_count", {}),
        ],
        "read",
        ("delete", {"file_path": "/big.csv"}),
        "deleted",
    ]
    model = ScriptedChatModel(turns=turns)
    graph = create_deep_agent(model=model, tools=[todo_count], middleware=[TodoListMiddleware()])
    agent = memory_harness.wrap(graph, id="deep-big")

    async def objects() -> list[str]:
        listed = await memory_harness.files.files(CARRIED_FOLDER)
        return [n for where, n, size in listed if where.endswith(f"/{CARRIED_OBJECTS}") and size]

    assert (await agent.run("keep it", user="u", thread="t")).status is RunStatus.SUCCESS
    assert len(await objects()) == 1  # the file, whole, as an object of its own
    agent.results = Conversations()  # another worker: nothing of the conversation in memory
    assert (await agent.run("read it", user="u", thread="t")).status is RunStatus.SUCCESS
    read = [str(m.content) for m in model.seen[3] if isinstance(m, ToolMessage)]
    assert "END-OF-BIG" in read[-2] and read[-1] == "1 todos"
    agent.results = Conversations()
    assert (await agent.run("delete it", user="u", thread="t")).status is RunStatus.SUCCESS
    assert await objects() == []  # deleted: nothing names its object any more


#: Window sizes from small to the model's whole, and unknown (``None``).
WINDOWS = [8_000, 16_000, 32_000, 128_000, None]


@pytest.mark.parametrize("window", WINDOWS)
async def test_deep_agents_planning_is_its_own_beside_the_harness_middleware(
    harness: Harness, window: int | None
) -> None:
    """PLAN: Deep Agents' planning is LangChain's ``TodoListMiddleware``, which a team adds.
    Beside ``middleware()`` it is the library's own at every window, small, large or unknown:
    ``write_todos`` offered from the first call with its own guidance, the plan in the graph's
    state. The harness's ``Planning`` stays out of a Deep Agents graph, so one without it has no
    ``write_todos``, as Deep Agents alone; ``create_agent`` keeps the harness's planning."""
    todos = [{"content": "check tides", "status": "in_progress"}]
    model = ScriptedChatModel(turns=[("write_todos", {"todos": todos}), "planned"])
    stack = [TodoListMiddleware(), *middleware(context_window=window)]
    graph = create_deep_agent(model=model, middleware=stack, checkpointer=InMemorySaver())
    result = await harness.wrap(graph, id="deep").run("plan it", user="u", thread="t")
    assert result.answer == "planned"
    assert all("write_todos" in request["tools"] for request in model.requests)
    assert "write_todos" in str(model.seen[0][0].content)  # the library's own guidance
    assert (await state_of(harness, graph, "t"))["todos"] == todos
    alone = ScriptedChatModel(turns=["hello"])
    bare = create_deep_agent(model=alone, middleware=middleware(context_window=window))
    assert (await harness.wrap(bare, id="bare").run("hi", user="u")).answer == "hello"
    assert "write_todos" not in alone.requests[0]["tools"]
    held = create_agent(alone, tools=[], middleware=middleware(context_window=window))
    assert "write_todos" in {t.name for t in graph_tools(held)}  # held for a long task


@lc_tool
def tides(day: str) -> str:
    """The tide table of a day."""
    return f"{day}: " + "high water at six, low water at noon. " * 400


tides.metadata = {"side_effects": "read"}


@pytest.mark.parametrize("kept", ["a small window", "no thread"])
async def test_the_todo_list_stays_in_view_when_the_turns_that_wrote_it_do_not(
    harness: Harness, kept: str
) -> None:
    """PLAN: the todo list is the graph's state, and the model reads it in the ``write_todos``
    call that wrote it. When no such call is in a request — the older turns' calls left out to
    fit a small window, or the earlier turns given as text (a graph with no thread of its own)
    — the list as last written is shown after the instructions: a later turn knows its plan."""
    todos = [
        {"content": "check monday tides", "status": "completed"},
        {"content": "check sunday tides", "status": "pending"},
    ]
    turns: list[Any] = [("write_todos", {"todos": todos}), "planned"]
    small = kept == "a small window"
    asks = [f"read tides {n}" for n in range(8)] if small else ["what is pending?"]
    for n in range(len(asks)):
        turns += [("tides", {"day": f"a{n}"}), ("tides", {"day": f"b{n}"})] if small else []
        turns.append(f"read {n}")
    model = ScriptedChatModel(turns=turns)
    stack = [TodoListMiddleware(), *middleware(context_window=8_000 if small else None)]
    saver = InMemorySaver() if small else None
    graph = create_deep_agent(model=model, tools=[tides], middleware=stack, checkpointer=saver)
    agent = harness.wrap(graph, id="deep")
    for said in ["plan it", *asks]:
        assert (await agent.run(said, user="u", thread="t")).status is RunStatus.SUCCESS
    unseen = [
        seen
        for seen in model.seen[2:]
        if not any(isinstance(m, AIMessage) and m.tool_calls[:1] for m in seen[:3])
    ]  # the call that wrote the list is not in these requests
    assert len(unseen) >= (8 if small else 1)
    for seen in unseen:  # said at the start of the turn's question, the instructions as ever
        shown = [m.text for m in seen if isinstance(m, HumanMessage)][-1]
        assert shown.startswith("Your todo list (write_todos), as you last wrote it:")
        assert "- [completed] check monday tides\n- [pending] check sunday tides" in shown
        assert "check sunday" not in seen[0].text
    assert "Your todo list" not in model.seen[1][-3].text  # the call shows it: said once


async def test_a_local_shell_backend_runs_as_the_library_runs_it_governed_at_each_call(
    harness: Harness, tmp_path: Path
) -> None:
    """``LocalShellBackend`` gives ``execute`` a real shell on this machine: the library's own,
    run as it is under the harness — as Claude Code's and Kimi Code's shells are —, each
    command the run's own call (announced and recorded, never asked), and refused at
    ``h.wrap`` only where the deployment says ``TRELLIS_HOST_EXECUTION=deny``."""
    backend = LocalShellBackend(root_dir=str(tmp_path), virtual_mode=True)
    model = ScriptedChatModel(turns=[("execute", {"command": "echo tides > t.txt"}), "done"])
    graph = create_deep_agent(model=model, backend=backend, checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="deep")
    events = [e async for e in agent.stream("write it", user="u", thread="t")]
    assert (tmp_path / "t.txt").read_text() == "tides\n"
    called = [e.data for e in events if e.data.get("tool") == "execute"]
    assert called, "the command is a call of the run"
    assert not [e for e in events if e.type.value == "interrupt"]  # announced, not asked
    denying = Harness(config=Settings(host_execution="deny"), memory=False)
    with pytest.raises(ConfigurationError, match="TRELLIS_HOST_EXECUTION=deny"):
        denying.wrap(graph, id="deep")


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


async def test_a_large_result_of_the_agents_own_tool_is_read_to_its_last_row(
    harness: Harness,
) -> None:
    """The moved result is within reach to its last row: ``read_file`` from an offset past the
    preview reads the lines there."""

    @lc_tool
    def export(sku: str) -> str:
        """Export a SKU's history."""
        return "\n".join(f"{sku} row {n}" for n in range(12_000))

    export.metadata = {"side_effects": "read"}
    page = {"file_path": "/large_tool_results/call_1_0", "offset": 11_990, "limit": 10}
    model = ScriptedChatModel(turns=[("export", {"sku": "a"}), ("read_file", page), "read"])
    graph = create_deep_agent(model=model, tools=[export], checkpointer=InMemorySaver())
    result = await harness.wrap(graph, id="deep").run("export a", user="u", thread="t")
    assert result.answer == "read", result.error
    assert "/large_tool_results/call_1_0" in str(model.seen[1][-1].content)
    read = str(model.seen[-1][-1].content)
    assert "a row 11999" in read and "a row 11990" in read and "a row 11989" not in read, read


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


async def test_the_compaction_tool_is_the_graphs_own_and_asks_nobody(harness: Harness) -> None:
    """``compact_conversation`` (Deep Agents' ``SummarizationToolMiddleware``) declares no side
    effects itself; the harness knows it by name as it knows Deep Agents' other tools
    (``middleware.declare_files``): the graph's own state, a read, so the library's tool
    answers with nobody asked."""
    model = ScriptedChatModel(turns=[("compact_conversation", {}), "nothing to compact"])
    compacts = create_summarization_tool_middleware(model, StateBackend())
    graph = create_deep_agent(model=model, middleware=[compacts], checkpointer=InMemorySaver())
    done = await harness.wrap(graph, id="deep").run("compact", user="u", thread="t")
    assert done.status is RunStatus.SUCCESS and done.answer == "nothing to compact"
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
