"""``bridges.helpers``: a native helper's run record, started once and ended once."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from trellis.contracts import RunStart, RunStatus
from trellis.harness.bridges.helpers import HELPER, NOT_CONTINUED, Helpers, Kind, text_of
from trellis.harness.identity import Identity
from trellis.harness.runs import LocalRuns


class GraphInterrupt(Exception):
    """A pause signal."""


def runtime(runs: LocalRuns, attempt: int = 1) -> Any:
    identity = Identity(tenant="t", user="u", agent_id="desk", run_id="run_p")
    return SimpleNamespace(
        agent=SimpleNamespace(harness=SimpleNamespace(runs=runs, memory=None)),
        identity=identity,
        run_id="run_p",
        tenant="t",
        agent_id="desk",
        user="u",
        attempt=attempt,
        run_memory=None,
        pending=None,
    )


async def statuses(runs: LocalRuns) -> list[RunStatus]:
    return [c.status async for c in runs.iterate(parent_run_id="run_p")]


async def test_a_helper_started_again_is_the_same_record_and_ends_once() -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    await helpers.end("k", output="fine")
    await helpers.end("k", output="again")  # ended already: no change
    again = Helpers(runtime(runs, attempt=2), "langgraph")
    await again.start("k2", kind=Kind.TASK, name="checker", task="check o1")  # the same record
    await again.end("k2", output="fine")  # ended before: left as it is
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.status is RunStatus.SUCCESS and record.output == "fine"
    assert record.metadata[HELPER] == {"framework": "langgraph", "kind": "task", "name": "checker"}


async def test_the_same_task_twice_in_an_attempt_is_two_helpers() -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    for key in ("a", "b"):
        await helpers.start(key, kind=Kind.TASK, name="checker", task="check o1")
    await helpers.closed(None)
    assert await statuses(runs) == [RunStatus.SUCCESS, RunStatus.SUCCESS]


@pytest.mark.parametrize("error", [GraphInterrupt(), KeyboardInterrupt()], ids=["pause", "crash"])
async def test_a_pause_or_a_crash_leaves_the_helper_to_the_next_attempt(
    error: BaseException,
) -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    await helpers.closed(error)
    assert await statuses(runs) == [RunStatus.RUNNING]


async def test_a_failed_run_ends_its_open_helpers_as_errors() -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    await helpers.closed(ValueError("boom"))
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.error is not None
    assert record.error.message == "the run ended first: ValueError"


async def test_a_helper_an_earlier_attempt_left_open_and_the_next_did_not_take_up_ends() -> None:
    runs = LocalRuns()
    first = Helpers(runtime(runs), "claude_agent_sdk")
    await first.start("k", kind=Kind.SUBAGENT, name="checker", task="check o1")
    await first.closed(GraphInterrupt())  # paused with it open
    await runs.start(  # a sub-agent of the harness's own: it ends itself
        RunStart(
            run_id="run_own",
            tenant_id="t",
            agent_id="x",
            thread_id="x",
            user_id="u",
            parent_run_id="run_p",
            input="x",
        )
    )
    await Helpers(runtime(runs, attempt=2), "claude_agent_sdk").closed(None)
    found = {c.run_id: c.status async for c in runs.iterate(parent_run_id="run_p")}
    assert found.pop("run_own") is RunStatus.RUNNING
    [(run_id, status)] = found.items()
    record = await runs.get(run_id)
    assert status is RunStatus.ERROR and record is not None and record.error is not None
    assert record.error.message == NOT_CONTINUED


def test_a_helpers_answer_is_its_text() -> None:
    assert text_of("fine") == "fine"
    assert text_of(AIMessage(content="said")) == "said"
    assert text_of(Command(update={"messages": [ToolMessage("done", tool_call_id="c")]})) == "done"
    assert text_of(SimpleNamespace(content="plain")) == "plain"
    assert text_of(SimpleNamespace(content=[1])) == "namespace(content=[1])"


# ---------------------------------------------------------------------- the adapters' hooks
class Recorded:
    """Helpers as the hooks drive them: what started and ended."""

    def __init__(self) -> None:
        self.runtime = SimpleNamespace(task="refund o1")
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def start(self, key: str, **given: Any) -> None:
        self.calls.append(("start", key, given))

    async def end(self, key: str, **given: Any) -> None:
        self.calls.append(("end", key, given))


async def test_a_claude_subagents_call_starts_it_and_its_result_ends_it() -> None:
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from trellis.harness.adapters.claude import GENERAL_PURPOSE, SUBAGENT_FAILED, _subagents

    helpers = Recorded()
    called = AssistantMessage(
        content=[
            TextBlock(text="checking"),
            ToolUseBlock(id="t1", name="Agent", input={"description": "check o1"}),
            ToolUseBlock(id="t2", name="Read", input={}),
        ],
        model="m",
    )
    await _subagents(helpers, called)  # type: ignore[arg-type]
    await _subagents(helpers, AssistantMessage(content=[], model="m", parent_tool_use_id="t1"))  # type: ignore[arg-type]
    answered = UserMessage(
        content=[
            TextBlock(text="note"),
            ToolResultBlock(tool_use_id="t1", content="fine"),
            ToolResultBlock(tool_use_id="t2", content=None, is_error=True),
        ]
    )
    await _subagents(helpers, answered)  # type: ignore[arg-type]
    await _subagents(helpers, UserMessage(content="a prompt"))  # type: ignore[arg-type]
    assert helpers.calls == [
        ("start", "t1", {"kind": Kind.SUBAGENT, "name": GENERAL_PURPOSE, "task": "check o1"}),
        ("end", "t1", {"output": "fine"}),
        ("end", "t2", {"error": SUBAGENT_FAILED}),
    ]


def test_the_options_own_max_budget_is_the_sdks_cap() -> None:
    from trellis.harness.adapters.claude import _left

    assert _left(SimpleNamespace(), SimpleNamespace(max_budget_usd=2)) == 2.0  # type: ignore[arg-type]


def test_the_runs_callbacks_go_beside_the_ones_it_was_given() -> None:
    from langchain_core.callbacks import AsyncCallbackManager, StdOutCallbackHandler

    from trellis.harness.adapters.langgraph import _with_handler

    ours, theirs = object(), StdOutCallbackHandler()
    assert _with_handler(None, ours) == [ours]
    assert _with_handler([theirs], ours) == [theirs, ours]
    given = AsyncCallbackManager(handlers=[theirs])
    manager = _with_handler(given, StdOutCallbackHandler())
    assert len(manager.handlers) == 2 and given.handlers == [theirs]  # theirs: unchanged


async def test_a_reply_without_a_message_counts_nothing() -> None:
    from langchain_core.outputs import Generation, LLMResult

    from trellis.harness.hooks.langgraph import GraphCallbacks

    spent: dict[str, Any] = {}
    run = SimpleNamespace(replay=SimpleNamespace(journal=SimpleNamespace(spent=spent)), parent=None)
    run.agent = SimpleNamespace(budget=None)
    callbacks = GraphCallbacks(run, Helpers(run, "langgraph"))  # type: ignore[arg-type]
    await callbacks.on_llm_end(LLMResult(generations=[[], [Generation(text="plain")]]))
    assert spent == {}


async def test_a_handoff_back_to_the_starting_agent_is_no_helper() -> None:
    from trellis.harness.hooks.openai_agents import ModelHooks

    helpers = Recorded()
    hooks = ModelHooks(helpers=helpers, starting="desk")  # type: ignore[arg-type]
    desk, billing = SimpleNamespace(name="desk"), SimpleNamespace(name="billing")
    await hooks.on_handoff(None, desk, billing)
    await hooks.on_handoff(None, billing, desk)
    await hooks.on_agent_end(None, desk, ["not text"])
    assert [(c[0], c[1]) for c in helpers.calls] == [
        ("end", "desk"),
        ("start", "billing"),
        ("end", "billing"),
        ("end", "desk"),
    ]
    plain = ModelHooks()  # code that runs the SDK itself: no helpers
    await plain.on_handoff(None, desk, billing)
    await plain.on_agent_end(None, billing, "done")
