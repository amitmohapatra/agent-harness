"""The run's record goes with the turn it writes to memory (``pipeline._with_facts``): each tool
result its call's status, seconds and sources, the turn's last message the run's status,
seconds and approvals, under ``custom_metadata["run_record"]`` — what the memory service's
steps list of the turn shows."""

from __future__ import annotations

import json

import pytest

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Citation, Harness, ReAct, tool
from trellis.harness.history import RUN_RECORD, Said, Speaker, recorded, said_of

NASDAQ = Citation(name="Nasdaq Last Sale feed", date="2026-10-27")
#: How deep the memory service lets custom metadata nest (its ``METADATA_MAX_DEPTH``).
METADATA_DEPTH = 2


def _depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max(map(_depth, value.values()), default=0)
    if isinstance(value, list):
        return 1 + max(map(_depth, value), default=0)
    return 0


@tool(side_effects="read", response_format="content_and_artifact")
def stock_quote(symbol: str) -> tuple[str, Citation]:
    """The latest closing quote of a stock symbol.

    Args:
        symbol: the ticker, e.g. ACME
    """
    return json.dumps({"symbol": symbol, "close": 123.45}), NASDAQ


@pytest.mark.asyncio
async def test_the_turn_is_written_with_its_run_record(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("stock_quote", {"symbol": "ACME"}), "123.45"])
    agent = arm(memory_harness.wrap(ReAct("Quotes.", model), id="q"), [stock_quote])
    await agent.run("What did ACME close at?", user="u1", thread="t1")
    written = [m for c in memory_service.named("messages") for m in c.body["messages"]]
    facts = [m.get("custom_metadata", {}).get(RUN_RECORD) for m in written]
    [result] = [f for m, f in zip(written, facts, strict=True) if m["role"] == "TOOL"]
    assert result["status"] == "ok" and result["seconds"] >= 0
    assert result["sources"] == "Nasdaq Last Sale feed, 2026-10-27"
    run = facts[-1]
    assert written[-1]["role"] == "ASSISTANT" and written[-1]["content"] == "123.45"
    assert run["status"] == "SUCCESS" and run["seconds"] > 0 and run["approvals"] == "none"
    assert facts[0] is None  # the question carries nothing
    assert all(_depth(m.get("custom_metadata", {})) <= METADATA_DEPTH for m in written)


def test_the_facts_survive_the_write_spool() -> None:
    said = Said(Speaker.TOOL, "123.45", tool_call_id="c1", facts={"status": "OK"})
    assert recorded(said)["custom_metadata"] == {RUN_RECORD: {"status": "OK"}}
    assert said_of(recorded(said)) == said
    assert "custom_metadata" not in recorded(Said(Speaker.USER, "hi"))
