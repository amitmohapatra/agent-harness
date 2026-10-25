"""``FitWindow``: a model call never fails because its request does not fit the window — the
optional guidance dropped, the tools narrowed (the used ones kept, the hinted ones next, then by
the words they share with the request), their descriptions compacted, older results cleared and
cut, older turns left out — each only as far as needed, said on the run and counted; a request
that fits is the same object; one whose user message alone cannot fit fails saying so."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct, middleware, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness import middleware as ours
from trellis.harness.events import WARNING
from trellis.harness.middleware import (
    FIT_CLEARED,
    FITTED,
    TOOL_GUIDANCE,
    ContextTooLarge,
    FitWindow,
    RunCheckpointer,
    _size,
)

LONG = "Looks something up in one of the many systems this company runs. " * 6


def lc(name: str, text: str = LONG) -> Any:
    def run(query: str) -> str:
        return query

    return StructuredTool.from_function(
        func=run, name=name, description=f"{name}: {text}", args_schema=None
    )


def request(
    messages: list[Any], tools: list[Any], system: str = "You help.", **settings: Any
) -> ModelRequest[Any]:
    model = ScriptedChatModel(turns=[])
    return ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content=system),
        tools=tools,
        tool_choice=None,
        response_format=None,
        state={"messages": messages},
        runtime=None,  # type: ignore[arg-type]
        model_settings=settings,
    )


async def sent(fit: FitWindow, given: ModelRequest[Any]) -> ModelRequest[Any]:
    seen: list[ModelRequest[Any]] = []

    async def handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(r)
        return ModelResponse(result=[AIMessage(content="ok")])

    await fit.awrap_model_call(given, handler)
    return seen[0]


async def test_a_request_that_fits_is_sent_as_it_is() -> None:
    given = request([HumanMessage("hi")], [lc("a")])
    assert await sent(FitWindow(100_000), given) is given
    assert await sent(FitWindow(), given) is given  # no window known: it steps aside
    with pytest.raises(ConfigurationError, match="a window is a number of tokens"):
        FitWindow(0)


async def test_the_guidance_goes_first_and_only_as_far_as_needed() -> None:
    system = f"You help.\n\n{TOOL_GUIDANCE}\nThese tools change data: a."
    given = request([HumanMessage("hi")], [lc("a")], system=system)
    window = int((_size(given) - 100) / 0.95)
    before = FITTED["guidance"]
    out = await sent(FitWindow(window), given)
    assert out.system_message is not None and out.system_message.text == "You help."
    assert out.tools == given.tools  # no further than needed
    assert FITTED["guidance"] == before + 1


async def test_the_tools_are_narrowed_to_the_most_relevant_then_compacted() -> None:
    tools = [lc(f"tool_{n}") for n in range(40)] + [lc("refund_order", "Refunds an order.")]
    given = request([HumanMessage("please refund order o-7")], tools)
    window = int(_size(request([HumanMessage("please refund order o-7")], tools[:12])) / 0.95)
    out = await sent(FitWindow(window), given)
    names = [t["function"]["name"] if isinstance(t, dict) else t.name for t in out.tools]
    assert "refund_order" in names  # it shares the request's words
    assert len(names) < len(tools)
    assert _size(out) <= int(window * 0.95)
    tiny = int(_size(request([HumanMessage("please refund order o-7")], tools[:3])) / 0.95)
    compact = await sent(FitWindow(tiny), given)
    described = [t["function"]["description"] for t in compact.tools if isinstance(t, dict)]
    assert described and all(len(d) <= ours.FIT_DESCRIPTION for d in described)


async def test_older_results_are_cleared_cut_and_older_turns_left_out() -> None:
    big = "x" * 40_000
    messages = [
        SystemMessage("remembered"),
        HumanMessage("first"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "a", "args": {}}]),
        ToolMessage(big, tool_call_id="c1"),
        AIMessage(content="done"),
        HumanMessage("second"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "a", "args": {}}]),
        ToolMessage(big, tool_call_id="c2"),
    ]
    out = await sent(FitWindow(6000), request(messages, [lc("a")]))
    results = [m.text for m in out.messages if isinstance(m, ToolMessage)]
    assert FIT_CLEARED in results or len(results) == 1  # the older one cleared or left out
    assert "characters cut to fit the window" in results[-1]
    assert _size(out) <= int(6000 * 0.95)
    kept = await sent(FitWindow(6000, turns=False), request(messages, [lc("a")]))
    assert kept.messages[1].text == "first"  # a summarization after it has the older turns


async def test_a_user_message_that_cannot_fit_fails_naming_the_window() -> None:
    given = request([HumanMessage("y" * 40_000)], [lc("a")], max_tokens=500)
    with pytest.raises(ContextTooLarge, match="the user's message alone"):
        await sent(FitWindow(4000), given)


async def test_react_on_a_small_window_with_many_tools_runs_and_says_what_it_dropped(
    harness: Harness,
) -> None:
    many = []
    for n in range(60):

        def look(query: str, n: int = n) -> str:
            return f"found {query}"

        many.append(tool(look, name=f"look_{n}", description=LONG, side_effects="read"))
    model = ScriptedChatModel(turns=[("look_3", {"query": "a"}), "found it"])
    target = ReAct("s", model, context_window=4096, tool_guidance=True)
    agent = arm(harness.wrap(target, id="crowded"), many)
    events = [e async for e in agent.stream("look a up with look_3", user="u")]
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED][-1]
    assert finished.data.get("result") == "found it", finished.data
    said = [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == WARNING
    ]
    assert any(d.get("code") == "context_fit" for d in said), said
    assert len(model.requests[0]["tools"]) < 60
    assert "look_3" in model.requests[1]["tools"]  # a tool the run used stays offered


async def test_middleware_on_create_agent_fits_with_its_window(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["ok"])
    graph = create_agent(
        model,
        tools=[lc(f"t{n}") for n in range(60)],
        middleware=middleware(context_window=4096),
        checkpointer=RunCheckpointer(),
    )
    result = await harness.wrap(graph, id="fits").run("hello", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert len(model.requests[0]["tools"]) < 60


MEMORY = "\n".join(
    [
        "## Profile",
        "### standing",
        "- Answer in French.",
        "## Learned skills for this task",
        "- refund: lookup_order then refund_order (worked 90% of 10 runs)",
        "## Most relevant",
        "- [m2] 2026-01-02 u: the order o-7 was damaged",
        "## Memories",
        "- [m1] 2026-01-01 u: likes tea",
        "- [m2] (above)",
        "- [m3] 2026-01-03 u: never ship to the old address",
        "- [m4] 2026-01-04 u: " + "lived in Paris for a while " * 20,
        "- [m5] 2026-01-05 u: " + "talked about the weather at length " * 20,
    ]
)


def remembered(**settings: Any) -> ModelRequest[Any]:
    context = SystemMessage(MEMORY, id=ours.CONTEXT_MESSAGE_ID)
    return request([context, HumanMessage("refund o-7")], [], **settings)


async def test_memory_is_trimmed_last_its_lowest_ranked_items_first_never_rules_or_the_plan() -> (
    None
):
    given = remembered()
    whole = _size(given)
    fits_without_m5 = int((whole - 130) / 0.95)
    before = FITTED["memory"]
    out = await sent(FitWindow(fits_without_m5), given)
    text = out.messages[0].text
    assert "[m5]" not in text and "[m4]" in text  # the last first, only as far as needed
    assert FITTED["memory"] == before + 1
    unranked = "\n".join(
        line for line in MEMORY.split("\n") if not any(f"[m{n}]" in line for n in (1, 4, 5))
    )
    without = request(
        [SystemMessage(unranked, id=ours.CONTEXT_MESSAGE_ID), HumanMessage("refund o-7")],
        [],
    )
    out = await sent(FitWindow(int(_size(without) / 0.95) + 2), given)
    left = out.messages[0].text
    for kept in (
        "Answer in French.",
        "refund: lookup_order",
        "never ship",
        "[m2] 2026",
        "[m2] (above)",
    ):
        assert kept in left  # standing instructions, the plan, a rule, the best-ranked: kept
    assert not any(f"[m{n}]" in left for n in (1, 4, 5))
    least = int(_size(request([HumanMessage("refund o-7")], [])) / 0.95) + 5
    bare = (await sent(FitWindow(least), given)).messages[0].text  # every item that may go: gone
    assert not any(f"[m{n}]" in bare for n in (1, 2, 4, 5)) and "never ship" in bare
    pinned = SystemMessage("## Profile\n- Answer in French." * 40, id=ours.CONTEXT_MESSAGE_ID)
    only = request([pinned, HumanMessage("refund o-7")], [])
    assert (await sent(FitWindow(least), only)).messages[0] is pinned  # nothing it may drop


async def test_memory_stays_whole_when_a_summarization_after_it_makes_it_fit() -> None:
    older = [HumanMessage("earlier " * 400), AIMessage(content="ok " * 400)]
    context = SystemMessage(MEMORY, id=ours.CONTEXT_MESSAGE_ID)
    given = request([context, *older, HumanMessage("refund o-7")], [])
    window = int((_size(given) - 200) / 0.95)
    out = await sent(FitWindow(window, turns=False), given)
    assert out.messages[0].text == MEMORY  # the older turns are the summarization's to shrink
    assert len(out.messages) == len(given.messages)


def test_the_tools_keep_the_order_they_were_first_offered_in() -> None:
    class Run:  # a run, as the order is kept per run
        pass

    hooks, run = ours.ModelHooks(), Run()
    a, b, c, d = (lc(n, "x.") for n in "abcd")

    def order(*tools: Any) -> list[str]:
        out = hooks._in_order(request([HumanMessage("hi")], list(tools)), run)  # type: ignore[arg-type]
        return [ours._named(t) for t in out.tools]

    assert order(c, a) == ["a", "c"]  # the first call's by name
    assert order(d, a, c, b) == ["a", "c", "b", "d"]  # those offered later after them
    assert order(b, a) == ["a", "b"]  # narrowed (a request over the window), its order kept
    assert order(c, b, a, d) == ["a", "c", "b", "d"]  # back in their places
    given = request([HumanMessage("hi")], [a])
    assert hooks._in_order(given, run) is given and hooks._in_order(given, None) is given  # type: ignore[arg-type]
