"""Manual compaction, opt-in: Deep Agents' ``SummarizationToolMiddleware`` (its
``compact_conversation`` tool) given to ``ReAct(middleware=)`` — the tool is the graph's own
machinery, a read: it asks nobody, and compacts the conversation as Deep Agents does."""

from __future__ import annotations

import json
from typing import Any

from deepagents.middleware.summarization import create_summarization_tool_middleware

from tests.support.arm import arm
from tests.support.models import Script, ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.automatic import JournalBackend

COMPACT = "compact_conversation"


@tool(side_effects="read")
def report(n: int) -> str:
    """A long report."""
    return f"report {n}: " + "figures " * 3000


class Counting(Script):
    """A scripted endpoint that reports each request's size (a provider's ``prompt_tokens``):
    Deep Agents reads how full the window is from the usage a reply reports."""

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        made = super().reply(body)
        made["usage"]["prompt_tokens"] = len(json.dumps(body["messages"])) // 4
        made["usage"]["total_tokens"] = made["usage"]["prompt_tokens"] + 5
        return made


def compacting(harness: Harness, turns: list[Any], **profile: int) -> tuple[Any, ScriptedChat]:
    model = ScriptedChat(script=Counting(turns), profile=profile or None)
    backend = JournalBackend()
    graph = ReAct(
        "You analyse reports.",
        model,
        backend=backend,
        middleware=[create_summarization_tool_middleware(model, backend)],
    )
    return arm(harness.wrap(graph, id="analyst"), [report]), model


def offered(request: dict[str, Any]) -> set[str]:
    return {t["function"]["name"] for t in request.get("tools", [])}


async def test_the_compact_tool_is_offered_and_asks_nobody(harness: Harness) -> None:
    agent, model = compacting(harness, [(COMPACT, {}), "nothing to compact yet"])
    events = [e async for e in agent.stream("hello", user="u")]
    assert COMPACT in offered(model.requests[0])
    assert not [e for e in events if e.type is RunEventType.INTERRUPT]
    done = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
    assert done.data["result"] == "nothing to compact yet"
    told = next(m for m in model.requests[1]["messages"] if m["role"] == "tool")
    assert "compact" in str(told["content"]).lower()  # Deep Agents' own word: too early


async def test_a_long_conversation_is_compacted_on_request(harness: Harness) -> None:
    turns: list[Any] = [
        ("report", {"n": 1}),
        ("report", {"n": 2}),
        ("report", {"n": 3}),
        (COMPACT, {}),
        "A summary: three reports of figures.",  # the summarizer's call
        "All three hold figures.",
    ]
    agent, model = compacting(harness, turns, max_input_tokens=32_000)
    result = await agent.run("compare the three reports", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "All three hold figures."
    last = str(model.requests[-1]["messages"])
    assert "A summary: three reports of figures." in last  # the summary stands for the history
    assert "compare the three reports" in last  # the run's question is kept ahead of it
