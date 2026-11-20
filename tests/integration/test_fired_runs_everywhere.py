"""A fired run on every adapter (``BUILDERS``, and ``ReAct`` on Deep Agents): what a schedule set
in chat leaves behind, before any change to schedules — the checks the schedule work rests on.

* audit: the schedule is made by a recorded tool call of the chat's run, and its fire is a run
  of its own: the schedule's user, in the chat's thread, its tool calls recorded;
* memory: the fired run's report is written to the chat's conversation, as a chat turn's is;
* conversation: the fired run reads the chat it was set in (what the user asked there).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from deepagents.backends import StateBackend

from tests.support.adapters import BUILDERS, SYSTEM, Built, _saw
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.planned import FINAL, Call, PlannedChat, PlannedChatModel, _Planned
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.runs import LocalRuns

THREAD = "chat-fired"
ASKED = "Watch SKU-1 for me and tell me when it is under 5"
WATCH = {"task": "check the stock of SKU-1", "when": "0 9 * * *", "condition": "SKU-1 under 5"}


@tool(side_effects="read")
def stock(sku: str) -> str:
    """Units of a SKU in stock now.

    Args:
        sku: The SKU, such as SKU-1.
    """
    return f"{sku}: 3 left"


async def _react_deep(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
    stall: float = 0.0,
) -> Built:
    """``ReAct`` on Deep Agents: a backend given (its file tools, paging, summarization)."""
    model = PlannedChat(plan, stall=stall)
    _saw(seen, model)
    return ReAct(system_prompt=system, model=model, backend=StateBackend()), tools


FRAMEWORKS = {**BUILDERS, "react_deep": _react_deep}


def _replanned(target: Any, seen: list[Any], plan: list[Call], new: list[Call]) -> None:
    """The model of ``target`` makes ``new`` from its next run on: each test model keeps its
    plan where it is (a function's is the list it was given; a CLI's, its script)."""
    model: Any = seen[0] if seen else None
    # a planned model counts the tool results it reads, the chat's ones it is given back among
    # them: its next calls follow those
    if isinstance(model, PlannedChatModel):
        model.plan = [*model.plan, *new]
    elif isinstance(getattr(model, "script", None), _Planned) or hasattr(model, "plan"):
        held = model.script if isinstance(getattr(model, "script", None), _Planned) else model
        held.plan = [*held.plan, *new]
    elif isinstance(target, ClaudeAgentOptions):
        script = [{"tool": n, "args": a} for n, a in new] + [{"text": FINAL}]
        target.env["FAKE_CLAUDE_SCRIPT"] = json.dumps(script)
    plan[:] = new  # the function target follows the list it was built with


def _read(model: Any) -> str:
    """Everything a run's model was sent, as one text (a CLI's: the file it wrote)."""
    if isinstance(model, Path):
        return model.read_text() if model.exists() else ""
    said = getattr(model, "said", None)
    return (
        str(said()) if callable(said) else json.dumps(getattr(model, "requests", []), default=str)
    )


@pytest.mark.parametrize("framework", sorted(FRAMEWORKS))
async def test_a_fired_run_is_audited_remembered_and_reads_its_chat(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path, framework: str
) -> None:
    h, seen = memory_harness, []
    plan: list[Call] = [("schedule_task", WATCH)]
    target, tools = await FRAMEWORKS[framework](h, [stock], tmp_path, plan, seen=seen)
    agent = arm(h.wrap(target, id="desk"), tools)
    said = await agent.run(ASKED, user="ada", thread=THREAD)
    assert said.status is RunStatus.SUCCESS, said.error
    runs = h.runs
    assert isinstance(runs, LocalRuns)
    [schedule] = runs._schedules.values()
    await h.writes.drain()
    # audit: the schedule was made by a tool call the chat's run recorded
    made = [c for c in memory_service.named("record_tool") if "schedule_task" in str(c.body)]
    assert made, [c.body for c in memory_service.named("record_tool")]
    memory_service.calls.clear()

    _replanned(target, seen, plan, [("stock", {"sku": "SKU-1"})])
    fired = await runs.schedules.fire(schedule.schedule_id)
    assert await h.worker([agent]).run_once() is True
    record = await runs.get(fired.run_id)
    assert record is not None
    # audit: the fire is a run of its own, for the schedule's user, in the chat's thread
    assert record.status is RunStatus.SUCCESS, record.error
    assert (record.on_behalf_of, record.thread_id) == ("ada", THREAD)
    # the schedule names the run its fire queued
    after = await runs.schedules.get(schedule.schedule_id)
    assert after is not None and after.last_run_id == fired.run_id
    await h.writes.drain()
    # audit: its tool calls are recorded as any run's
    assert [c for c in memory_service.named("record_tool") if "SKU-1" in str(c.body)], record.output
    # memory: its report is written to the chat's conversation, as a chat turn's is
    written = memory_service.named("messages")
    assert written, "the fired run's report is not in the conversation"
    assert all(c.scope.get("thread_id") for c in written)
    # conversation: the fired run read the chat it was set in
    if seen:  # a function target has no model: what it reads is its input, checked elsewhere
        assert ASKED in _read(seen[0]), "the fired run did not read the chat it was set in"


async def test_a_failed_fired_run_leaves_its_error_in_the_chat_for_explain(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A question like "why did my stock alert fail?" is answered as any "why" is, from the
    run's own record in the chat (``explain``): a fired run that fails writes its turn to the
    chat's conversation, and that turn says it failed and why, in the error's own words."""
    h, chats = memory_harness, []

    async def desk(input: Any, agent: Any) -> str:
        chats.append(input)
        if len(chats) == 1:
            await agent.tools.call("schedule_task", **WATCH)
            return "I will check it every day at 9:00."
        raise RuntimeError("Gmail: sign-in expired")

    agent = h.wrap(desk, id="desk")
    said = await agent.run(ASKED, user="ada", thread=THREAD)
    assert said.status is RunStatus.SUCCESS, said.error
    runs = h.runs
    assert isinstance(runs, LocalRuns)
    [schedule] = runs._schedules.values()
    await h.writes.drain()
    memory_service.calls.clear()

    fired = await runs.schedules.fire(schedule.schedule_id)
    await h.worker([agent]).run_once()
    record = await runs.get(fired.run_id)
    assert record is not None and record.status is RunStatus.ERROR
    await h.writes.drain()
    written = json.dumps([c.body for c in memory_service.named("messages")], default=str)
    assert "Gmail: sign-in expired" in written, written[:3000]
