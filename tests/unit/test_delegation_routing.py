"""When ``ReAct`` offers ``delegate`` (``automatic.Delegation``), read without a model call: a
wide request — independent items of one kind — on a large model; never steps in order, a
request to change something, a narrow one, a small model or window, or outside a run."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from trellis.harness.automatic import (
    MIN_WINDOW,
    Delegation,
    model_tier,
    wide,
)
from trellis.harness.delegate import DELEGATE, tool


@pytest.mark.parametrize(
    "text",
    [
        "Compare the revenue of Apple, Microsoft, Google and Amazon.",
        "For each of these, find the CEO: Apple, Microsoft, Google, Amazon",
        "Summarize every one of a.pdf, b.pdf, c.pdf, d.pdf",
        "Prices of:\n- AAPL\n- MSFT\n- GOOG\n- AMZN",
        "Prices of:\n1. AAPL\n2. MSFT\n3. GOOG\n4. AMZN\n5. NVDA",
        "Compare Ünïcode Ltd, Ça Va SA, Ōsaka KK; and Zürich AG",
    ],
)
def test_a_wide_request_is_wide(text: str) -> None:
    assert wide(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "What is Apple's revenue?",
        "Compare Apple and Microsoft.",  # two items
        "Compare Apple, Microsoft and Google.",  # three
        "Apple, Microsoft, Google and Amazon are big.",  # four, no wide cue
        "Look up Apple, then Microsoft, then Google, then Amazon.",  # steps in order
        "Step by step, compare Apple, Microsoft, Google and Amazon.",
        "Prices of:\n- AAPL\n- MSFT\n- GOOG\n- AMZN\nthen buy the cheapest",
        "Update the records of each: Apple, Microsoft, Google and Amazon.",  # a change
        "Send each of Apple, Microsoft, Google and Amazon a note.",
        "Prices of:\n- AAPL\n- MSFT\n- GOOG",  # a list of three
    ],
)
def test_a_narrow_sequential_or_changing_request_is_not(text: str) -> None:
    assert not wide(text)


@pytest.mark.parametrize(
    ("name", "window", "tier"),
    [
        ("gpt-4o", 128_000, "large"),
        ("claude-sonnet-4", 200_000, "large"),
        ("scripted", MIN_WINDOW, "large"),
        ("qwen3-30b-a3b", 128_000, "large"),  # a mixture of experts: its total
        ("llama-3.1-70b-instruct", 128_000, "large"),
        (None, 128_000, "large"),
        ("granite-3.3-2b-instruct", 128_000, "small"),
        ("ibm/granite-4.0-h-small", 128_000, "small"),  # the family, whatever its size
        ("qwen2.5-7b", 128_000, "small"),
        ("llama-3.1-8B", 128_000, "small"),
        ("phi-3.5-mini-3.8b", 128_000, "small"),
        ("gpt-4o", MIN_WINDOW - 1, "small"),
        ("gpt-4o", None, "small"),  # the window not known
    ],
)
def test_the_model_tier_from_its_name_and_window(
    name: str | None, window: int | None, tier: str
) -> None:
    assert model_tier(name, window) == tier


def _request(*messages: Any) -> ModelRequest[Any]:
    return ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=list(messages),
        tools=[tool()],
        state={"messages": list(messages)},  # type: ignore[typeddict-item]
    )


async def test_outside_a_run_or_on_a_small_model_delegate_is_not_offered() -> None:
    sent: list[set[str]] = []

    async def handler(request: ModelRequest[Any]) -> Any:
        sent.append({t.name for t in request.tools})  # type: ignore[union-attr]
        return AIMessage("ok")

    request = _request(HumanMessage("Compare Apple, Microsoft, Google and Amazon."))
    await Delegation(tier="large").awrap_model_call(request, handler)  # no run
    await Delegation(tier="small").awrap_model_call(request, handler)
    assert sent == [set(), set()]


def test_the_tool_names_its_subagents_and_only_reads() -> None:
    made = tool(["research", "audit", "research"])
    assert made.name == DELEGATE
    assert (made.metadata or {})["side_effects"] == "read"
    schema: Any = made.args_schema
    job = schema["properties"]["jobs"]["items"]
    assert job["properties"]["subagent_type"]["enum"] == ["audit", "research"]
    assert "either `subagent_type`" in made.description and "audit, research" in made.description
    assert "subagent_type" not in tool().args_schema["properties"]["jobs"]["items"]["properties"]  # type: ignore[index]
