"""Deep Agents' machinery in ``ReAct`` (``trellis.harness.automatic``): LangChain's
``TodoListMiddleware`` (``planning``), Deep Agents' scratch files on the journal and its
general-purpose helper (``files``). ``ReAct`` asked for a Deep Agents capability (a backend:
here the journal's) offers them from its first step. ``ReAct`` alone holds them until the run
shows a long task — its conversation has outgrown half the window, where old tool results start
being cleared — on a known window with room for them; a shorter task is offered none of them,
whatever its wording or the calls it made, nor a model whose window is not known."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness import middleware, react
from trellis.harness.automatic import LONG_AT, SCRATCH, TASK, JournalBackend, long_task
from trellis.harness.middleware import SCRATCH_NOTE
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

LONG_TOOLS = {"write_todos", TASK, *SCRATCH}
#: A window with room for the long-task tools.
ROOM = 128_000
#: The least window they are offered in (documented: 32,000 tokens or more).
MIN_WINDOW = 32_000
STEPS = "Do this step by step: "
done: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    return 7


@tool(side_effects="irreversible")
def publish(text: str) -> str:
    """Publish a text: a person approves it first.

    Args:
        text: What to publish.
    """
    done.append(f"publish {text}")
    return f"published {text}"


def offered(model: ScriptedChatModel, n: int) -> set[str]:
    return set(model.requests[n].get("tools", []))


def result_of(messages: list[BaseMessage], call: str | None = None) -> str:
    found = [m for m in messages if isinstance(m, ToolMessage)]
    if call is not None:
        found = [m for m in found if m.tool_call_id == call]
    return str(found[-1].content)


# --------------------------------------------------------------------------- triggers
#: A text that alone outgrows half of the smallest window with room (about 17,000 tokens).
LONG_TEXT = "stock report line. " * 3_600


@tool(side_effects="read")
def report() -> str:
    """The whole stock report."""
    return LONG_TEXT


async def test_react_alone_is_offered_them_once_its_conversation_outgrows_half_the_window(
    harness: Harness,
) -> None:
    """No Deep Agents capability asked for: the tools are held while the task is an ordinary
    one — four model calls made, a fifth: the calls made do not switch — and offered from the
    call whose conversation is past half the window, for the rest of the run."""
    turns: list[Any] = [("stock", {"sku": f"a-{n}"}) for n in range(4)]
    model = ScriptedChatModel(turns=[*turns, ("report", {}), ("stock", {"sku": "a-9"}), "7 each"])
    agent = arm(
        harness.wrap(ReAct("s", model, context_window=MIN_WINDOW), id="auto"), [stock, report]
    )
    assert (await agent.run("how many a-0 .. a-3?", user="u")).answer == "7 each"
    assert all(not offered(model, n) & LONG_TOOLS for n in range(5))
    assert offered(model, 5) >= LONG_TOOLS and offered(model, 6) >= LONG_TOOLS
    # where old results start being cleared: the request is edited anyway
    assert LONG_AT == react.CLEAR_AT
    # the instructions stay the same from the first call on (a prompt cache's prefix holds):
    # the plan's guidance is the tool's own description
    assert model.seen[5][0].content == model.seen[0][0].content


@pytest.mark.parametrize(
    "asked",
    [
        STEPS + "count a-1.",
        "Make a plan to count a-1.",
        "Count:\n1. a-1\n2. a-2\n3. a-3",
    ],
)
async def test_the_wording_of_a_request_does_not_switch(harness: Harness, asked: str) -> None:
    """A request that says "step by step", "a plan" or lists items is offered what any other
    is: on the long set no request said so, and the one benchmark request that did ("pick the
    plan with...") asked for no steps."""
    model = ScriptedChatModel(turns=["7"])
    await arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="words"), [stock]).run(
        asked, user="u"
    )
    assert not offered(model, 0) & LONG_TOOLS


@pytest.mark.parametrize("window", [ROOM, MIN_WINDOW, None], ids=["room", "least", "unknown"])
async def test_react_asked_for_a_backend_is_deep_agents_from_its_first_step(
    harness: Harness, window: int | None
) -> None:
    """A Deep Agents capability asked for: its tools, planning and ``read_file`` from the first
    call — the developer's choice — on any window not known to be too small for them."""
    model = ScriptedChatModel(turns=["7"])
    target = ReAct("s", model, backend=JournalBackend(), context_window=window)
    await arm(harness.wrap(target, id=f"asked-{window}"), [stock]).run("count", user="u")
    assert offered(model, 0) >= LONG_TOOLS | {"read_file"}


async def test_a_window_known_to_be_small_has_no_room_for_them(harness: Harness) -> None:
    """Together they cost about 2,600 tokens a call: under ``MIN_WINDOW`` they are held, asked
    for or not, even once the conversation is past half of it."""
    model = ScriptedChatModel(turns=[("stock", {"sku": "a-1"}), "7"])
    target = ReAct("s", model, backend=JournalBackend(), context_window=MIN_WINDOW - 1)
    await arm(harness.wrap(target, id="small"), [stock]).run("x " * 40_000, user="u")
    assert not offered(model, 0) & LONG_TOOLS and not offered(model, 1) & LONG_TOOLS


@pytest.mark.parametrize("window", [MIN_WINDOW - 1, None], ids=["small", "unknown"])
async def test_react_alone_on_a_small_or_unknown_window_is_offered_none(
    harness: Harness, window: int | None
) -> None:
    model = ScriptedChatModel(turns=[("report", {}), "7"])
    target = ReAct("s", model, context_window=window)
    await arm(harness.wrap(target, id=f"none-{window}"), [stock, report]).run("count", user="u")
    assert not offered(model, 0) & LONG_TOOLS and not offered(model, 1) & LONG_TOOLS


# --------------------------------------------------------------------------- planning
async def test_a_plan_is_written_and_read_back(harness: Harness) -> None:
    todos = [{"content": "count a-1", "status": "in_progress"}]
    model = ScriptedChatModel(
        turns=[("write_todos", {"todos": todos}), ("write_todos", {"todos": []}), "planned"]
    )
    agent = arm(
        harness.wrap(ReAct("s", model, backend=JournalBackend(), context_window=ROOM), id="plan"),
        [stock],
    )
    assert (await agent.run(STEPS + "count a-1", user="u")).answer == "planned"
    assert "count a-1" in result_of(model.seen[1])
    assert "Updated todo list" in result_of(model.seen[2])  # an empty plan is a plan too


# --------------------------------------------------------------------------- files
async def test_scratch_files_are_written_edited_listed_and_read(harness: Harness) -> None:
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/notes.md", "content": "a-1: ?\ncafé ☃ \x00"}),
            ("edit_file", {"file_path": "/notes.md", "old_string": "?", "new_string": "7"}),
            ("ls", {"path": "/"}),
            ("read_file", {"file_path": "/notes.md"}),
            ("edit_file", {"file_path": "/missing.md", "old_string": "a", "new_string": "b"}),
            ("write_file", {"file_path": "../escape.md", "content": "x"}),
            "noted",
        ]
    )
    agent = arm(
        harness.wrap(ReAct("s", model, backend=JournalBackend(), context_window=ROOM), id="files"),
        [stock],
    )
    assert (await agent.run(STEPS + "note a-1", user="u")).answer == "noted"
    assert "/notes.md" in result_of(model.seen[1])
    assert "/notes.md" in result_of(model.seen[3])
    read = result_of(model.seen[4])
    assert "a-1: 7" in read and "café ☃" in read
    assert "not found" in result_of(model.seen[5])
    assert "traversal" in result_of(model.seen[6]).lower()


async def test_a_scratch_file_written_says_the_user_cannot_open_it(harness: Harness) -> None:
    """A file a write puts in the graph's own state is the run's scratch copy: the model is
    told so where its path is, not to hand it over (break-and-fix round 1: asked for a chart, a
    model saved an SVG at /tmp/... and told the user to open it)."""
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/chart.svg", "content": "<svg/>"}),
            ("edit_file", {"file_path": "/chart.svg", "old_string": "/", "new_string": " /"}),
            ("read_file", {"file_path": "/chart.svg"}),
            "done",
        ]
    )
    agent = arm(
        harness.wrap(ReAct("s", model, backend=JournalBackend(), context_window=ROOM), id="svg"),
        [stock],
    )
    assert (await agent.run(STEPS + "chart it", user="u")).answer == "done"
    assert SCRATCH_NOTE in result_of(model.seen[1])  # the run's first write says it
    assert SCRATCH_NOTE not in result_of(model.seen[2])  # once a run: the history keeps it
    assert SCRATCH_NOTE not in result_of(model.seen[3])  # a read says nothing of the kind


async def test_a_deep_agents_state_write_says_it_and_a_failed_one_does_not(
    harness: Harness,
) -> None:
    """Deep Agents' own state backend answers a write with a Command (the files and the
    message): its message says it; a write that failed says nothing, and the next one does."""
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "../escape.md", "content": "x"}),
            ("write_file", {"file_path": "/chart.svg", "content": "<svg/>"}),
            "done",
        ]
    )
    agent = harness.wrap(create_deep_agent(model=model, tools=[]), id="da-svg")
    assert (await agent.run("chart it", user="u")).answer == "done"
    assert SCRATCH_NOTE not in result_of(model.seen[1])  # refused: nothing written
    assert SCRATCH_NOTE in result_of(model.seen[2])


def test_a_writes_command_says_it_on_its_message_and_a_failed_one_is_left() -> None:
    """Deep Agents' state backend answers a write with a Command (the files and the message)."""
    said = ToolMessage("Updated file /a.md", tool_call_id="c1")
    noted = middleware._scratch_noted(Command(update={"files": {}, "messages": [said]}))
    assert isinstance(noted, Command) and isinstance(noted.update, dict)
    assert noted.update["files"] == {}
    assert noted.update["messages"][0].content.endswith(SCRATCH_NOTE)
    failed = Command(update={"messages": [ToolMessage("no", tool_call_id="c2", status="error")]})
    assert middleware._scratch_noted(failed) is failed
    files = Command(update={"files": {"/a.md": "x"}})
    assert middleware._scratch_noted(files) is files


async def test_files_are_kept_across_a_pause(harness: Harness) -> None:
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/draft.md", "content": "the draft"}),
            ("publish", {"text": "the draft"}),
            ("read_file", {"file_path": "/draft.md"}),
            "published",
        ]
    )
    agent = arm(
        harness.wrap(ReAct("s", model, backend=JournalBackend(), context_window=ROOM), id="pause"),
        [publish],
    )
    paused = await agent.run(STEPS + "publish", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert result.answer == "published" and done == ["publish the draft"]
    assert "the draft" in result_of(model.seen[-1])


async def test_a_file_or_a_runs_files_too_large_are_refused() -> None:
    backend = JournalBackend(max_file=10, max_run=15)
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/a.md", "content": "x" * 11}),
            ("write_file", {"file_path": "/a.md", "content": "x" * 10}),
            ("write_file", {"file_path": "/b.md", "content": "y" * 6}),
            ("edit_file", {"file_path": "/a.md", "old_string": "x", "new_string": "xx"}),
            ("write_file", {"file_path": "/large_tool_results/r", "content": "z" * 50}),
            "ok",
        ]
    )
    graph = create_deep_agent(model, backend=backend)
    out = await graph.ainvoke({"messages": [HumanMessage(STEPS + "write")]})
    said = [str(m.content) for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "holds at most 10" in said[0]
    assert "Error" not in said[1]
    assert "a run's files hold at most 15" in said[2]
    assert "holds at most 10" in said[3]  # the edit would make it 11
    assert "Error" not in said[4]  # what the harness saves itself is not capped
    assert out["files"]["/a.md"]["content"] in ("x" * 10, ["x" * 10])


async def test_the_journal_holds_a_file_of_100_000_characters_and_a_run_of_1_000_000() -> None:
    from deepagents.backends.utils import create_file_data

    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/big.md", "content": "x" * 100_001}),
            ("write_file", {"file_path": "/f10.md", "content": "y"}),
            ("write_file", {"file_path": "/f0.md", "content": "z" * 100_000}),  # in its place
            "ok",
        ]
    )
    graph = create_deep_agent(model, backend=JournalBackend())
    # ten files of 100,000 characters: the run's 1,000,000 exactly
    files = {f"/f{n}.md": create_file_data("x" * 100_000) for n in range(10)}
    given: Any = {"messages": [HumanMessage(STEPS + "write")], "files": files}
    out = await graph.ainvoke(given)
    said = [str(m.content) for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "/big.md would hold 100001 characters; a file holds at most 100000" in said[0]
    assert "would hold 1000001 characters; a run's files hold at most 1000000" in said[1]
    assert "Error" not in said[2]  # a file written over is counted once, at its new size
    assert out["files"]["/f0.md"]["content"] in ("z" * 100_000, ["z" * 100_000])


def test_a_task_is_long_once_its_conversation_is_past_half_the_window() -> None:
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages.utils import count_tokens_approximately

    messages: list[Any] = [HumanMessage("stock report line. " * 200)]
    request = ModelRequest(model=ScriptedChatModel(turns=[]), messages=messages)
    half = count_tokens_approximately(messages)
    assert not long_task(request, 2 * half)  # at half: not past it
    assert long_task(request, 2 * half - 2)


async def test_a_thousand_files_list(harness: Harness) -> None:
    writes = [("write_file", {"file_path": f"/f{n:04}.md", "content": "x"}) for n in range(1000)]
    model = ScriptedChatModel(turns=[writes, ("ls", {"path": "/"}), "listed"])
    agent = arm(
        harness.wrap(
            ReAct("s", model, backend=JournalBackend(), max_steps=4, context_window=ROOM), id="many"
        ),
        [stock],
    )
    assert (await agent.run(STEPS + "write", user="u")).answer == "listed"
    listing = result_of(model.seen[2])
    assert "/f0000.md" in listing


async def test_files_written_before_a_crash_are_read_on_another_worker(
    harness: Harness,
) -> None:
    crashes = [Crash()]

    @tool(side_effects="read")
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/kept.md", "content": "kept"}),
            ("boom", {}),
            ("boom", {}),
            ("read_file", {"file_path": "/kept.md"}),
            "kept it",
        ]
    )
    agent = arm(
        harness.wrap(ReAct("s", model, backend=JournalBackend(), context_window=ROOM), id="crash"),
        [boom],
    )
    store = harness.runs
    assert isinstance(store, LocalRuns)
    handle = await agent.start(STEPS + "keep", user="u")
    worker = harness.worker([agent])
    claimed = await store.claim(worker.worker_id, [agent.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await agent.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await harness.worker([agent]).run_once()
    result = await handle.result(wait=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "kept it", result.error
    assert "kept" in result_of(model.seen[-1])


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


async def test_an_edit_replacing_nothing_is_the_backends_own_answer() -> None:
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/a.md", "content": "abc"}),
            ("edit_file", {"file_path": "/a.md", "old_string": "", "new_string": "x"}),
            "ok",
        ]
    )
    graph = create_deep_agent(model, backend=JournalBackend())
    out = await graph.ainvoke({"messages": [HumanMessage(STEPS + "write")]})
    assert out["files"]["/a.md"]["content"] in ("abc", ["abc"])
