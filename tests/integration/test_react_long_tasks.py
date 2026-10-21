"""What ``ReAct`` turns on for a long task (``trellis.harness.automatic``): LangChain's
``TodoListMiddleware`` (``planning``), Deep Agents' scratch files on the journal
(``files``) and Deep Agents' ``task`` helper as a child run (``subagent_task``) — offered from
the model's :data:`LONG_AFTER`-th call or when the request asks for steps, on a window with
room for them, each off with ``without=``; a short task is offered none of them."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct, tool
from trellis.contracts import RunRecord, RunStatus
from trellis.harness import automatic
from trellis.harness.automatic import (
    LONG_AFTER,
    MAX_HELPERS,
    MIN_WINDOW,
    SCRATCH,
    JournalBackend,
    JournalFiles,
    ScratchFiles,
)
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

LONG_TOOLS = {"write_todos", "task", *SCRATCH}
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


_lock = threading.Lock()


class Routed(ScriptedChatModel):
    """One model for a parent and its helpers, which run at once: each run answers from the
    script of its first user message (``scripts``; the parent's under ``""``)."""

    scripts: dict[str, list[Any]]

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        asked = _of(messages)
        key = next((k for k in self.scripts if k and asked.startswith(k)), "")
        with _lock:
            self.seen.append(list(messages))
            self.requests.append(kw)
            turn = self.scripts[key].pop(0)
            n = len(self.seen)
        if isinstance(turn, str):
            message = AIMessage(content=turn)
        else:
            calls = turn if isinstance(turn, list) else [turn]
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call_{n}_{i}"}
                    for i, (name, args) in enumerate(calls)
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def task(description: str) -> tuple[str, dict[str, Any]]:
    return ("task", {"description": description, "subagent_type": automatic.HELPER})


# --------------------------------------------------------------------------- triggers
async def test_the_long_task_tools_come_with_the_models_long_after_th_call(
    harness: Harness,
) -> None:
    turns: list[Any] = [("stock", {"sku": f"a-{n}"}) for n in range(LONG_AFTER + 1)]
    model = ScriptedChatModel(turns=[*turns, "7 each"])
    agent = harness.wrap(ReAct("s", model), id="long", tools=[stock])
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
    await harness.wrap(ReAct("s", model), id="steps", tools=[stock]).run(asked, user="u")
    assert (offered(model, 0) >= LONG_TOOLS) is long
    assert bool(offered(model, 0) & LONG_TOOLS) is long


async def test_a_small_window_has_no_room_for_them(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["7"])
    target = ReAct("s", model, context_window=MIN_WINDOW - 1)
    await harness.wrap(target, id="small", tools=[stock]).run(STEPS + "count", user="u")
    assert not offered(model, 0) & LONG_TOOLS
    model = ScriptedChatModel(turns=["7"])
    target = ReAct("s", model, context_window=MIN_WINDOW)
    await harness.wrap(target, id="room", tools=[stock]).run(STEPS + "count", user="u")
    assert offered(model, 0) >= LONG_TOOLS


@pytest.mark.parametrize("feature", ["planning", "files", "subagent_task"])
async def test_without_a_feature_its_tools_are_neither_offered_nor_run(
    harness: Harness, feature: str
) -> None:
    gone = {"planning": {"write_todos"}, "files": SCRATCH, "subagent_task": {"task"}}[feature]
    calls = {
        "planning": ("write_todos", {"todos": []}),
        "files": ("ls", {"path": "/"}),
        "subagent_task": task("x"),
    }
    model = ScriptedChatModel(turns=[calls[feature], "done"])
    agent = harness.wrap(ReAct("s", model), id=f"no-{feature}", tools=[stock])
    result = await agent.run(STEPS + "x", user="u", without={feature})  # type: ignore[arg-type]
    assert result.answer == "done"
    assert offered(model, 0) & LONG_TOOLS == LONG_TOOLS - gone  # exactly its own removed
    name = calls[feature][0]
    assert result_of(model.seen[1]) == automatic.NOT_AVAILABLE.format(tool=name)


# --------------------------------------------------------------------------- planning
async def test_a_plan_is_written_and_read_back(harness: Harness) -> None:
    todos = [{"content": "count a-1", "status": "in_progress"}]
    model = ScriptedChatModel(
        turns=[("write_todos", {"todos": todos}), ("write_todos", {"todos": []}), "planned"]
    )
    agent = harness.wrap(ReAct("s", model), id="plan", tools=[stock])
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
    agent = harness.wrap(ReAct("s", model), id="files", tools=[stock])
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
    agent = harness.wrap(ReAct("s", model), id="pause", tools=[publish])
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
    agent = harness.wrap(ReAct("s", model, max_steps=4), id="many", tools=[stock])
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
    agent = harness.wrap(ReAct("s", model), id="crash", tools=[boom])
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
    held, _ = store._leases[handle.run_id]
    store._leases[handle.run_id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await harness.worker([agent]).run_once()
    result = await handle.result(timeout=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "kept it", result.error
    assert "kept" in result_of(model.seen[-1])


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


# --------------------------------------------------------------------------- helpers
async def kids(h: Harness, parent_run_id: str) -> list[RunRecord]:
    found = [s async for s in h.runs.iterate(parent_run_id=parent_run_id)]
    records = [await h.runs.get(s.run_id) for s in found]
    return [r for r in records if r is not None]


async def test_a_helper_is_a_child_run_that_answers_its_task(harness: Harness) -> None:
    model = Routed(
        turns=[],
        scripts={
            "": [task("count a-1"), "a-1: 7, said the helper"],
            "count a-1": [("stock", {"sku": "a-1"}), "7"],
        },
    )
    agent = harness.wrap(ReAct("s", model), id="boss", tools=[stock])
    result = await agent.run(STEPS + "count a-1", user="u")
    assert result.answer == "a-1: 7, said the helper"
    [child] = await kids(harness, result.run_id)
    assert child.agent_id == "boss" and child.status is RunStatus.SUCCESS
    assert child.output == "7" and child.parent_run_id == result.run_id
    assert result_of(model.seen[-1]) == "7"
    # the helper is offered no helper (one level only)
    helper_calls = [
        r for r, seen in zip(model.requests, model.seen, strict=True) if _of(seen) == "count a-1"
    ]
    assert helper_calls and all("task" not in r.get("tools", []) for r in helper_calls)


def _of(messages: list[BaseMessage]) -> str:
    return next(str(m.content) for m in messages if isinstance(m, HumanMessage))


async def test_a_helper_calling_task_is_told_it_is_not_available(harness: Harness) -> None:
    model = Routed(
        turns=[],
        scripts={"": [task("go deeper"), "done"], "go deeper": [task("deeper"), "cannot"]},
    )
    agent = harness.wrap(ReAct("s", model), id="deep", tools=[stock])
    assert (await agent.run(STEPS + "x", user="u")).answer == "done"
    told = [result_of(seen) for seen in model.seen if _of(seen) == "go deeper" and len(seen) > 2]
    assert told == [automatic.NOT_AVAILABLE.format(tool="task")]


async def test_helpers_run_at_once_at_most_max_helpers(harness: Harness) -> None:
    running = 0
    most = 0
    both = asyncio.Event()

    @tool(side_effects="read")
    async def slow(part: str) -> str:
        """A slow read.

        Args:
            part: Which part.
        """
        nonlocal running, most
        running += 1
        most = max(most, running)
        if running >= 2:
            both.set()
        await asyncio.wait_for(both.wait(), 5)
        await asyncio.sleep(0.05)
        running -= 1
        return f"part {part}"

    parts = [str(n) for n in range(MAX_HELPERS + 1)]
    scripts: dict[str, list[Any]] = {"": [[task(f"part {p}") for p in parts], "all parts"]}
    for p in parts:
        scripts[f"part {p}"] = [("slow", {"part": p}), f"done {p}"]
    model = Routed(turns=[], scripts=scripts)
    agent = harness.wrap(ReAct("s", model), id="many-helpers", tools=[slow])
    result = await agent.run(STEPS + "all parts", user="u")
    assert result.answer == "all parts", result.error
    assert 2 <= most <= MAX_HELPERS  # at once, and the fifth waited for a slot
    assert len(await kids(harness, result.run_id)) == MAX_HELPERS + 1


async def test_a_helper_asked_to_write_pauses_the_parent_for_approval(harness: Harness) -> None:
    model = Routed(
        turns=[],
        scripts={
            "": [task("publish it"), "the helper published it"],
            "publish it": [("publish", {"text": "it"}), "published it"],
        },
    )
    agent = harness.wrap(ReAct("s", model), id="writer", tools=[publish])
    paused = await agent.run(STEPS + "publish", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "publish"
    assert done == []
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert result.answer == "the helper published it" and done == ["publish it"]


async def test_cancelling_the_parent_cancels_its_helper(harness: Harness) -> None:
    started = asyncio.Event()

    @tool(side_effects="read")
    async def wait(x: str) -> str:
        """Waits a long time.

        Args:
            x: Anything.
        """
        started.set()
        await asyncio.sleep(30)
        return "never"

    model = Routed(
        turns=[],
        scripts={"": [task("wait for it")], "wait for it": [("wait", {"x": "1"})]},
    )
    agent = harness.wrap(ReAct("s", model), id="cancel-helper", tools=[wait])
    running = asyncio.create_task(agent.run(STEPS + "wait", user="u"))
    await asyncio.wait_for(started.wait(), 5)
    summaries = [s async for s in harness.runs.iterate(agent_id="cancel-helper")]
    records = [await harness.runs.get(s.run_id) for s in summaries]
    [record] = [r for r in records if r is not None and r.parent_run_id is None]
    await agent.cancel(record.run_id, reason="stop")
    assert (await running).status is RunStatus.CANCELLED
    [child] = await kids(harness, record.run_id)
    assert child.status is RunStatus.CANCELLED


async def test_a_helper_cut_by_a_crash_continues_from_its_journal(harness: Harness) -> None:
    crashes = [Crash()]

    @tool(side_effects="read")
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    @tool(side_effects="write")
    def note(text: str) -> str:
        """Write a note.

        Args:
            text: The note.
        """
        done.append(f"note {text}")
        return f"noted {text}"

    model = Routed(
        turns=[],
        scripts={
            "": [task("note then check"), "helper done"],
            "note then check": [("note", {"text": "n1"}), ("boom", {}), "noted and checked"],
        },
    )
    agent = harness.wrap(ReAct("s", model), id="crash-helper", tools=[note, boom])
    store = harness.runs
    assert isinstance(store, LocalRuns)
    handle = await agent.start(STEPS + "note", user="u")
    worker = harness.worker([agent])
    claimed = await store.claim(worker.worker_id, [agent.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await agent.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.run_id]
    store._leases[handle.run_id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    # the script goes on where the crash cut it: the helper asks for boom again
    model.scripts["note then check"] = [("boom", {}), "noted and checked"]
    model.scripts[""] = ["helper done"]
    assert await harness.worker([agent]).run_once()
    result = await handle.result(timeout=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "helper done", result.error
    assert done == ["note n1"]  # the helper's write ran once


async def test_a_helper_that_fails_is_an_error_the_parent_reads(harness: Harness) -> None:
    model = Routed(
        turns=[],
        scripts={"": [task("fail"), "it failed"], "fail": []},  # the helper's model has nothing
    )
    agent = harness.wrap(ReAct("s", model), id="fails", tools=[stock])
    result = await agent.run(STEPS + "x", user="u")
    assert result.answer == "it failed"
    assert "task failed" in result_of(model.seen[-1])


async def test_outside_a_run_the_helper_is_not_available() -> None:
    from trellis.harness.automatic import Helpers

    model = ScriptedChatModel(turns=[task("x"), "no helper"])
    graph = create_agent(model, middleware=[Helpers(JournalBackend())])
    out = await graph.ainvoke({"messages": [HumanMessage(STEPS + "x")]})
    assert result_of(out["messages"]) == automatic.NOT_AVAILABLE.format(tool="task")


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
