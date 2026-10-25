"""What ``ReAct`` turns on for a long task (``trellis.harness.automatic``): LangChain's
``TodoListMiddleware`` (``planning``), Deep Agents' scratch files on the journal
(``files``) — offered from the model's :data:`LONG_AFTER`-th call or when the request asks for
steps, on a known window with room for them, each off with ``without=``; a short task is
offered none of them, nor a model whose window is not known."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.automatic import (
    LONG_AFTER,
    MIN_WINDOW,
    SCRATCH,
    JournalBackend,
    JournalFiles,
    ScratchFiles,
)
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

LONG_TOOLS = {"write_todos", *SCRATCH}
#: A window with room for the long-task tools.
ROOM = 128_000
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
async def test_the_long_task_tools_come_with_the_models_long_after_th_call(
    harness: Harness,
) -> None:
    turns: list[Any] = [("stock", {"sku": f"a-{n}"}) for n in range(LONG_AFTER + 1)]
    model = ScriptedChatModel(turns=[*turns, "7 each"])
    agent = arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="long"), [stock])
    assert (await agent.run("how many a-0 .. a-3?", user="u")).answer == "7 each"
    # threshold - 1, at, + 1: none before, all from the LONG_AFTER-th call on
    assert not offered(model, LONG_AFTER - 1) & LONG_TOOLS
    assert offered(model, LONG_AFTER) >= LONG_TOOLS
    assert offered(model, LONG_AFTER + 1) >= LONG_TOOLS
    assert all(not offered(model, n) & LONG_TOOLS for n in range(LONG_AFTER))
    prompt = str(model.seen[LONG_AFTER][0].content)
    assert "write_todos" in prompt and "write_todos" not in str(model.seen[0][0].content)


@pytest.mark.parametrize(
    ("asked", "long"),
    [
        (STEPS + "count a-1.", True),
        ("Make a plan to count a-1.", True),
        ("Count:\n1. a-1\n2. a-2\n3. a-3", True),
        ("Count:\n1. a-1\n2. a-2", False),  # two items: not yet steps
        ("How many a-1? First check, then answer.", False),
    ],
)
async def test_a_request_asking_for_steps_gets_them_from_the_first_call(
    harness: Harness, asked: str, long: bool
) -> None:
    model = ScriptedChatModel(turns=["7"])
    await arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="steps"), [stock]).run(
        asked, user="u"
    )
    assert (offered(model, 0) >= LONG_TOOLS) is long
    assert bool(offered(model, 0) & LONG_TOOLS) is long


async def test_a_small_window_has_no_room_for_them(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["7"])
    target = ReAct("s", model, context_window=MIN_WINDOW - 1)
    await arm(harness.wrap(target, id="small"), [stock]).run(STEPS + "count", user="u")
    assert not offered(model, 0) & LONG_TOOLS
    model = ScriptedChatModel(turns=["7"])
    target = ReAct("s", model, context_window=MIN_WINDOW)
    await arm(harness.wrap(target, id="room"), [stock]).run(STEPS + "count", user="u")
    assert offered(model, 0) >= LONG_TOOLS


async def test_a_window_not_known_has_none_of_them(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["7"])
    await arm(harness.wrap(ReAct("s", model), id="unknown"), [stock]).run(STEPS + "count", user="u")
    assert not offered(model, 0) & LONG_TOOLS


# --------------------------------------------------------------------------- planning
async def test_a_plan_is_written_and_read_back(harness: Harness) -> None:
    todos = [{"content": "count a-1", "status": "in_progress"}]
    model = ScriptedChatModel(
        turns=[("write_todos", {"todos": todos}), ("write_todos", {"todos": []}), "planned"]
    )
    agent = arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="plan"), [stock])
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
    agent = arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="files"), [stock])
    assert (await agent.run(STEPS + "note a-1", user="u")).answer == "noted"
    assert "/notes.md" in result_of(model.seen[1])
    assert "/notes.md" in result_of(model.seen[3])
    read = result_of(model.seen[4])
    assert "a-1: 7" in read and "café ☃" in read
    assert "not found" in result_of(model.seen[5])
    assert "traversal" in result_of(model.seen[6]).lower()


async def test_files_are_kept_across_a_pause(harness: Harness) -> None:
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/draft.md", "content": "the draft"}),
            ("publish", {"text": "the draft"}),
            ("read_file", {"file_path": "/draft.md"}),
            "published",
        ]
    )
    agent = arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="pause"), [publish])
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
    middleware: list[Any] = [JournalFiles(backend, tools=["read_file", *sorted(SCRATCH)])]
    graph = create_agent(model, middleware=[*middleware, ScratchFiles()])
    out = await graph.ainvoke({"messages": [HumanMessage(STEPS + "write")]})
    said = [str(m.content) for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "holds at most 10" in said[0]
    assert "Error" not in said[1]
    assert "a run's files hold at most 15" in said[2]
    assert "holds at most 10" in said[3]  # the edit would make it 11
    assert "Error" not in said[4]  # what the harness saves itself is not capped
    assert out["files"]["/a.md"]["content"] in ("x" * 10, ["x" * 10])


async def test_a_thousand_files_list(harness: Harness) -> None:
    writes = [("write_file", {"file_path": f"/f{n:04}.md", "content": "x"}) for n in range(1000)]
    model = ScriptedChatModel(turns=[writes, ("ls", {"path": "/"}), "listed"])
    agent = arm(
        harness.wrap(ReAct("s", model, max_steps=4, context_window=ROOM), id="many"), [stock]
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
    agent = arm(harness.wrap(ReAct("s", model, context_window=ROOM), id="crash"), [boom])
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
    middleware: list[Any] = [JournalFiles(JournalBackend(), tools=["read_file", *sorted(SCRATCH)])]
    graph = create_agent(model, middleware=[*middleware, ScratchFiles()])
    out = await graph.ainvoke({"messages": [HumanMessage(STEPS + "write")]})
    assert out["files"]["/a.md"]["content"] in ("abc", ["abc"])
