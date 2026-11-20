"""Text in any script, an emoji and a right-to-left word among it, goes through every adapter
as it is: a tool's result read by the model, and the answer the run returns, streams and
keeps — Claude's CLI and Kimi Code over their JSON lines too."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.planned import FINAL
from trellis import Harness, tool
from trellis.contracts import RunEventType

#: What the tool answers: several scripts, combining marks, an emoji outside the BMP.
TEXT = "在庫あり ✓ — ¡sí! Ünïcödé 🚀👍🏽 שלום مرحبا é"


@tool(side_effects="read")
def note(sku: str) -> str:
    """A SKU's note."""
    return TEXT


@pytest.mark.host_execution
@pytest.mark.parametrize("framework", sorted(BUILDERS))
async def test_unicode_reaches_the_model_and_is_the_answer_as_it_is(
    harness: Harness, tmp_path: Path, framework: str
) -> None:
    target, tools = await BUILDERS[framework](harness, [note], tmp_path, [("note", {"sku": "ä"})])
    agent = arm(harness.wrap(target, id=f"unicode-{framework}", without={"schedules"}), tools)
    events = [e async for e in agent.stream("the note of ä?", user="u")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED, finished
    expected = FINAL.replace("{last}", TEXT)
    assert finished.data["result"] == expected, finished.data
    argued = [e.data for e in events if e.type is RunEventType.TOOL_CALL_ARGS]
    assert "ä" in str(argued), argued  # the call's arguments as the model made them
    streamed = "".join(
        e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT
    )
    assert streamed in ("", expected), streamed  # streamed, it is the answer
    record = await harness.runs.get(finished.run_id)
    assert record is not None and record.output == expected
