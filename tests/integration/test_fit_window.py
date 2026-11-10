"""``FitWindow``: a model call never fails because its request does not fit the window — the
tools narrowed (the used ones kept, the hinted ones next, then by the words they share with the
request), their descriptions compacted, older results cleared and cut, older turns left out —
each only as far as needed, said on the run and counted; a request that fits is the same object;
one whose user message alone cannot fit fails saying so."""

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
    FITTED,
    ContextTooLarge,
    FitWindow,
    RunCheckpointer,
    _size,
)
from trellis.harness.tools.results import PREFIX as READ_PREFIX

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
    with pytest.raises(TypeError, match=r"FitWindow\(optional=\) was removed: .*tool guidance"):
        FitWindow(optional=("a text",))  # pyright: ignore[reportCallIssue]


async def test_the_tools_are_narrowed_to_the_most_relevant_then_compacted(narrowing: None) -> None:
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
    # one clearing pass (LangChain's ClearToolUsesEdit, before it): FitWindow only cuts
    assert all("cleared" not in r for r in results)
    assert "characters cut to fit the window" in results[-1]
    assert not hasattr(ours, "FIT_CLEARED") and not hasattr(ours, "read_result")
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
    target = ReAct("s", model, context_window=4096)
    agent = arm(harness.wrap(target, id="crowded", without={"schedules"}), many)
    events = [e async for e in agent.stream("look a up with look_3", user="u")]
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED][-1]
    assert finished.data.get("result") == "found it", finished.data
    said = [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == WARNING
    ]
    assert any(d.get("code") == "context_fit" for d in said), said
    assert len(model.requests[0]["tools"]) < 60
    assert "look_3" in model.requests[1]["tools"]  # a tool the run used stays offered
    # the same tools call after call (a prompt cache keeps its prefix)
    assert model.requests[1]["tools"] == model.requests[0]["tools"]


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


async def test_a_request_the_server_refuses_as_too_large_is_fitted_by_its_count() -> None:
    from langchain_core.exceptions import ContextOverflowError

    from trellis.contracts import ModelError

    tools = [lc(f"tool_{n}") for n in range(30)]
    given = request([HumanMessage("please refund order o-7")], tools)
    window = int(_size(given) / 0.95) + 50  # fits by the estimate
    seen: list[ModelRequest[Any]] = []
    refusals = [
        ModelError(
            f"the model call failed (400): request ({_size(given) * 2} tokens) exceeds the "
            "available context size",
            source="langgraph",
        )
    ]

    async def handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(r)
        if refusals:
            raise refusals.pop()
        return ModelResponse(result=[AIMessage(content="ok")])

    fit, before = FitWindow(window), FITTED["refit"]
    model = given.model
    await fit.awrap_model_call(given, handler)
    assert seen[0] is given and _size(seen[1]) <= int(window * 0.95) // 2  # the server's count
    assert FITTED["refit"] == before + 1 and ours._RATIOS.get(ours._model_name(model) or "")
    seen.clear()
    await fit.awrap_model_call(given, handler)  # that model's later calls: fitted at once
    assert len(seen) == 1 and len(seen[0].tools) < len(tools)
    ours._RATIOS._data.clear()

    async def refused(r: ModelRequest[Any]) -> ModelResponse[Any]:
        raise ContextOverflowError("prompt is too long")

    with pytest.raises(ContextOverflowError):  # once only: a second refusal is the call's
        await fit.awrap_model_call(given, refused)
    assert ours._RATIOS.get(ours._model_name(model) or "") is None  # no count said: kept
    ours._RATIOS._data.clear()

    async def broken(r: ModelRequest[Any]) -> ModelResponse[Any]:
        raise ModelError("the model call failed (500): boom", source="langgraph")

    with pytest.raises(ModelError, match="boom"):  # anything else is not a refusal
        await fit.awrap_model_call(given, broken)


async def test_the_ratio_is_learned_from_what_replies_say_their_requests_were() -> None:
    from langchain_core.messages.ai import UsageMetadata

    ours._RATIOS._data.clear()
    tools = [lc(f"tool_{n}") for n in range(30)]
    given = request([HumanMessage("please refund order o-7")], tools)
    counted = _size(given)
    window = int((counted - 40) / 0.95)  # over the window by the estimate
    seen: list[ModelRequest[Any]] = []
    said: list[int] = [int(counted * 0.5), int(counted * 0.9)]
    hooks = ours.ModelHooks()  # where the model is called: what it counted is learned there

    async def replied(r: ModelRequest[Any]) -> ModelResponse[Any]:
        usage = UsageMetadata(input_tokens=said.pop(0), output_tokens=1, total_tokens=1)
        return ModelResponse(result=[AIMessage(content="ok", usage_metadata=usage)])

    async def handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(r)
        return await hooks.awrap_model_call(r, replied)

    fit = FitWindow(window)
    await fit.awrap_model_call(given, handler)
    assert len(seen[0].tools) < len(tools)  # nothing learned yet: fitted by the estimate
    name = ours._model_name(given.model) or ""
    first = ours._RATIOS.get(name)
    assert first is not None and first >= ours.FIT_RATIOS[0]  # within the bounds
    await fit.awrap_model_call(given, handler)
    assert seen[1] is given  # fits by the model's own count: sent as it is
    assert ours._RATIOS.get(name) == pytest.approx(
        (first + 0.9 * counted / _size(seen[1])) / 2, 0.05
    )

    async def silent(r: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(result=[])

    before = ours._RATIOS.get(name)
    await fit.awrap_model_call(given, silent)  # a reply that says nothing teaches nothing
    assert ours._RATIOS.get(name) == before
    ours._RATIOS._data.clear()

    async def summarized(r: ModelRequest[Any]) -> ModelResponse[Any]:
        return await handler(r.override(tools=r.tools[:1]))  # a summarization after it

    said.append(_size(given.override(tools=tools[:1])))  # the model counts as estimated
    await FitWindow(window * 10).awrap_model_call(given, summarized)
    assert ours._RATIOS.get(name) == pytest.approx(1.0, 0.05)  # of the request it was sent


def test_the_longest_results_are_cut_first_and_only_as_far_as_needed() -> None:
    calls = [
        AIMessage(content="", tool_calls=[{"id": f"c{n}", "name": "a", "args": {}}]) for n in (1, 2)
    ]
    messages = [
        HumanMessage("go"),
        calls[0],
        ToolMessage("y" * 4_000, tool_call_id="c1"),
        calls[1],
        ToolMessage("x" * 40_000, tool_call_id="c2"),
    ]
    given = request(messages, [])
    out = ours._cut(given, _size(given) - 2_000)
    assert out.messages[2].text == "y" * 4_000  # the shorter one is left whole
    assert "characters cut to fit the window" in out.messages[4].text
    small = request([HumanMessage("go"), calls[0], ToolMessage("z" * 50, tool_call_id="c1")], [])
    assert ours._cut(small, 1) is small  # too short to cut


async def test_tools_dropped_to_fit_come_back_once_the_older_turns_are_left_out() -> None:
    """A long chat on a small window: the tools are narrowed (to none, past ``FIT_KEEP``) before
    the older turns are left out, and leaving them out frees the room; the tools that fit then
    are offered again — a request with no tool left the model nothing to fetch with (a 6,000-token
    window, Haiku: every turn after the fourth offered no tool, the model saying it had none)."""
    older = []
    for n in range(20):
        older += [HumanMessage(f"question {n}: {LONG}"), AIMessage(content=f"answer {n}: {LONG}")]
    messages = [*older, HumanMessage("How many footwear SKUs are priced above £100?")]
    tools = [lc(name) for name in ("sku_catalog", "sales_summary", "stock_level", "store_list")]
    out = await sent(FitWindow(3000), request(messages, tools))
    assert [m.text for m in out.messages] == [messages[-1].text]  # the older turns left out
    assert [t["function"]["name"] if isinstance(t, dict) else t.name for t in out.tools] == [
        "sku_catalog", "sales_summary", "stock_level", "store_list",
    ]  # fmt: skip
    assert _size(out) <= int(3000 * 0.95)


async def test_a_refusal_after_a_summarization_cuts_the_latest_turns_results() -> None:
    """With a summarization after it (Deep Agents'), the results are its to offload — but it
    keeps a ``read_file`` page whole, and on a small window that page and its summary did not
    fit (its own check refused the request, saying no size): fitted again, the latest turn's
    longest results are cut too, the older turns' left for the summary, every message where
    it was (the summary's cut point counts them)."""
    from langchain_core.exceptions import ContextOverflowError

    calls = [
        AIMessage(content="", tool_calls=[{"id": f"c{n}", "name": "a", "args": {}}]) for n in (1, 2)
    ]
    messages = [
        HumanMessage("the first question"),
        calls[0],
        ToolMessage("y" * 8_000, tool_call_id="c1"),
        AIMessage(content="its answer"),
        HumanMessage("the run's question"),
        calls[1],
        ToolMessage("x" * 8_000, tool_call_id="c2"),
    ]
    given = request(messages, [])
    seen: list[ModelRequest[Any]] = []

    async def handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(r)
        if len(seen) == 1:
            raise ContextOverflowError("Context remains above the input budget after compaction")
        return ModelResponse(result=[AIMessage(content="ok")])

    window = int((_size(ours._latest(given)) + 50) / 0.95)  # the latest turn alone fits
    await FitWindow(window, turns=False).awrap_model_call(given, handler)
    assert seen[0].messages == messages  # the results were the summarization's
    again = seen[1].messages
    assert len(again) == len(messages) and again[2].text == "y" * 8_000  # older: as they were
    assert "characters cut to fit the window" in again[6].text


def test_the_readers_of_a_kept_result_stay_offered_when_the_tools_are_narrowed() -> None:
    """A result the run keeps is previewed with the call that reads it (``query_result``,
    ``read_file``): narrowing the tools never drops those — a small window left the model a
    preview saying to call ``query_result`` and no ``query_result``, so it paged 250 rows with
    ``read_file`` and could not count them."""
    from trellis.harness.tools.query import QUERY_RESULT
    from trellis.harness.tools.results import READ_FILE

    tools = [lc(f"tool_{n}") for n in range(12)] + [lc(QUERY_RESULT), lc(READ_FILE)]
    given = request([HumanMessage("how many over 100?")], tools)
    kept: Any = {"messages": given.messages, "files": {f"{READ_PREFIX}/r1": {}}}  # Deep Agents'
    given = given.override(state=kept)
    out = ours._narrowed(given, 1, None, least=0)
    assert {ours._named(t) for t in out.tools} == {QUERY_RESULT, READ_FILE}
    bare = ours._narrowed(given.override(state={"messages": given.messages}), 1, None, least=0)
    assert QUERY_RESULT not in {ours._named(t) for t in bare.tools}  # nothing kept to read
