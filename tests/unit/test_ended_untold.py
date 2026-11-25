"""What a turn is told of the background runs its conversation's earlier turns handed on
(``Journal.in_background``): ``live.ended_untold`` and what the Claude adapter makes of it
(BG-11), against the local run store."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from trellis.contracts import AgentError, RunStart, RunStatus
from trellis.harness.adapters.claude import ClaudeAdapter
from trellis.harness.bridges import live
from trellis.harness.bridges.claude_live import SHARED
from trellis.harness.bridges.helpers import HELPER
from trellis.harness.bridges.live import Kept, LiveSessions
from trellis.harness.journal import Journal
from trellis.harness.runs import LocalRuns


async def _run(runs: LocalRuns, run_id: str, name: str = "researcher") -> None:
    await runs.start(
        RunStart(
            run_id=run_id,
            tenant_id="default",
            agent_id="chat",
            parent_run_id="run_turn1",
            input="research the market",
            metadata={HELPER: {"framework": "claude_agent_sdk", "kind": "subagent", "name": name}},
        )
    )


def _runtime(runs: LocalRuns, in_background: list[str], run_id: str = "run_turn3") -> Any:
    journal = Journal(in_background=in_background)
    agent = SimpleNamespace(harness=SimpleNamespace(runs=runs))
    return SimpleNamespace(
        agent=agent, tenant="default", replay=SimpleNamespace(journal=journal), run_id=run_id
    )


async def test_each_run_that_ended_untold_is_told_once_and_those_at_work_stay() -> None:
    runs = LocalRuns()
    for run_id in ("run_here", "run_working", "run_told", "run_done"):
        await _run(runs, run_id)
    await runs.finish("run_told", RunStatus.SUCCESS, output="seen")
    await runs.finish("run_done", RunStatus.SUCCESS, output="the market is growing")
    sessions = LiveSessions()
    sessions.attended["run_here"] = cast("Kept", object())  # at work in a live session here
    runtime = _runtime(runs, ["run_here", "run_working", "run_told", "run_gone", "run_done"])

    ended = await live.ended_untold(runtime, sessions, {"run_told"})

    assert [r.run_id for r in ended] == ["run_done"]
    # at work, here or elsewhere, they stay; told by the CLI, or no longer in the store, gone
    assert runtime.replay.journal.in_background == ["run_here", "run_working"]


async def test_a_turn_is_told_what_ended_at_the_head_of_its_prompt_once_and_again_on_a_retry() -> (
    None
):
    """Deep Agents' notification, what a failure says to do next being Claude's (it has no
    ``check_async_task``); a run the workflows shared is no task to tell; an attempt that
    sends the prompt again (the session not resumed) tells it again."""
    runs = LocalRuns()
    await _run(runs, "run_failed")
    await _run(runs, "run_shared", name=SHARED)
    error = AgentError(code="TaskFailed", message="the source was down")
    await runs.finish("run_failed", RunStatus.ERROR, error=error)
    await runs.finish("run_shared", RunStatus.SUCCESS, output="calls")
    adapter = ClaudeAdapter()
    runtime = _runtime(runs, ["run_failed", "run_shared"])

    prompt = await adapter._with_ended("and?", runtime, None)

    assert prompt.endswith("\n\nand?") and prompt.count("<task-notification>") == 1
    assert "<agent-name>researcher</agent-name>" in prompt
    assert "<status>error</status>" in prompt and "the source was down" in prompt
    assert "check_async_task" not in prompt and "workflows" not in prompt
    assert runtime.replay.journal.in_background == []
    assert await adapter._with_ended("and?", runtime, None) == prompt  # the same attempt again


async def test_a_turn_with_nothing_untold_keeps_its_prompt() -> None:
    adapter = ClaudeAdapter()
    runs = LocalRuns()
    await _run(runs, "run_seen")
    await runs.finish("run_seen", RunStatus.SUCCESS, output="seen")
    assert await adapter._with_ended("hi", _runtime(runs, [], run_id="run_a"), None) == "hi"
    serving = SimpleNamespace(told={"run_seen"})  # the live session the CLI told of it
    runtime = _runtime(runs, ["run_seen"], run_id="run_b")
    assert await adapter._with_ended("hi", runtime, serving) == "hi"  # type: ignore[arg-type]
    assert runtime.replay.journal.in_background == []
