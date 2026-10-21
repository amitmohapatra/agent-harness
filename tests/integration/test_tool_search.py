"""Tool search (``tool_search``): a catalogue of ``TOOL_SEARCH_MIN`` tools or more that the
tool hints did not narrow is narrowed to the tools the task is about — the best matches of a
search over every tool's definition, the tools the task names, Code Mode's and the skills'
tools — for every framework that narrows per call or per run; a ``ReAct`` also narrows tool
definitions that crowd its window, and searches again at every step."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent

from benchmark.cost import catalog
from tests.integration.test_react_context import Summarizing
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus, ToolSpec
from trellis.harness.automatic import TOOLS_SHARE
from trellis.harness.tools import search
from trellis.harness.tools.base import Tool
from trellis.harness.tools.search import TOOL_SEARCH_MIN, TOP_K


def tools(n: int, *, prefix: str = "report") -> list[Any]:
    def make(i: int) -> Any:
        def report(store: str) -> str:
            """The report of one store.

            Args:
                store: The store's id, such as S12.
            """
            return f"{prefix} {i} of {store}: nothing unusual"

        return tool(
            report,
            name=f"{prefix}_{i}",
            description=f"Quarterly {prefix} number {i} of one store.",
            side_effects="read",
        )

    return [make(i) for i in range(n)]


@tool(side_effects="read")
def lost_sales(store: str) -> str:
    """The sales a store lost to stock-outs, by style.

    Args:
        store: The store's id, such as S12.
    """
    return "ST-101: 38 units; see also the warehouse transfer backlog"


@tool(side_effects="read")
def transfer_backlog(warehouse: str) -> str:
    """Transfers waiting at a warehouse.

    Args:
        warehouse: The warehouse's id.
    """
    return "3 transfers waiting"


def names(request: dict[str, Any]) -> set[str]:
    return set(request.get("tools", []))


# --------------------------------------------------------------------------- the threshold
@pytest.mark.parametrize(
    ("count", "narrowed"), [(TOOL_SEARCH_MIN - 1, False), (TOOL_SEARCH_MIN, True)]
)
async def test_a_catalogue_is_narrowed_from_tool_search_min_tools(
    harness: Harness, count: int, narrowed: bool
) -> None:
    model = ScriptedChatModel(turns=["none lost"])
    catalogue = [*tools(count - 1), lost_sales]
    agent = harness.wrap(ReAct("s", model), id=f"n{count}", tools=catalogue)
    await agent.run("Which styles lost sales at S12?", user="u")
    offered = names(model.requests[0])
    if narrowed:
        assert "lost_sales" in offered and len(offered) <= TOP_K
    else:
        assert len(offered) == count


async def test_without_tool_search_every_tool_is_offered(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["none lost"])
    agent = harness.wrap(ReAct("s", model), id="all", tools=[*tools(40), lost_sales])
    await agent.run("Which styles lost sales at S12?", user="u", without={"tool_search"})
    assert len(names(model.requests[0])) == 41


async def test_five_hundred_tools_narrowed_to_the_ones_each_task_needs(harness: Harness) -> None:
    for taskset in catalog.task_sets():
        for n, task in enumerate(taskset.tasks):
            model = ScriptedChatModel(turns=["ok"])
            agent = harness.wrap(
                ReAct(taskset.system, model),
                id=f"five-hundred-{taskset.name.replace(' ', '-')}-{n}",
                tools=catalog.catalogue(taskset.tools, 500),
            )
            asked = (
                task["input"] if isinstance(task["input"], str) else task["input"][-1]["content"]
            )
            await agent.run(asked, user="u")
            offered = names(model.requests[0])
            assert set(task["metadata"]["needed_tools"]) <= offered, task["metadata"]["task"]
            assert len(offered) <= TOP_K + len(search.named(asked, offered))


async def test_a_tool_the_task_names_is_never_dropped(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["ok"])
    agent = harness.wrap(ReAct("s", model), id="named", tools=[*tools(30), lost_sales])
    await agent.run("Use report_27 for store S12, and tell me the lost sales.", user="u")
    assert {"report_27", "lost_sales"} <= names(model.requests[0])


async def test_a_tool_needed_mid_run_is_found_by_the_next_steps_search(harness: Harness) -> None:
    model = ScriptedChatModel(
        turns=[("lost_sales", {"store": "S12"}), ("transfer_backlog", {"warehouse": "W1"}), "ok"]
    )
    catalogue = [*tools(30), lost_sales, transfer_backlog]
    agent = harness.wrap(ReAct("s", model), id="again", tools=catalogue)
    result = await agent.run("Which styles lost sales at S12?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert "transfer_backlog" not in names(model.requests[0])
    # the result mentioned the transfer backlog: the next step is offered its tool
    assert "transfer_backlog" in names(model.requests[1])
    assert names(model.requests[0]) <= names(model.requests[1])  # only grows


async def test_the_tool_hints_narrow_first(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.candidates = ["report_3"]
    model = ScriptedChatModel(turns=["ok"])
    agent = memory_harness.wrap(ReAct("s", model), id="hinted", tools=[*tools(30), lost_sales])
    await agent.run("Which styles lost sales at S12?", user="u")
    assert "lost_sales" not in names(model.requests[0])  # the hints said report_3
    assert "report_3" in names(model.requests[0])


async def test_with_memory_but_no_candidates_the_search_narrows(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.candidates = []
    model = ScriptedChatModel(turns=["ok"])
    agent = memory_harness.wrap(ReAct("s", model), id="unhinted", tools=[*tools(30), lost_sales])
    await agent.run("Which styles lost sales at S12?", user="u")
    own = names(model.requests[0]) - set(MEMORY_TOOLS)
    assert "lost_sales" in own and len(own) <= TOP_K + 1


async def test_a_create_agent_graph_built_with_h_tools_is_narrowed_too(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["ok"])
    graph = create_agent(
        model, tools=await harness.tools(*tools(30), lost_sales, framework="langgraph")
    )
    await harness.wrap(graph, id="graph").run("Which styles lost sales at S12?", user="u")
    offered = names(model.requests[0])
    assert "lost_sales" in offered and len(offered) <= TOP_K


async def test_a_function_target_calls_any_tool_by_name(harness: Harness) -> None:
    from trellis import Runtime

    async def fn(input: str, agent: Runtime) -> str:
        assert agent.offered is None  # a function narrows nothing: it calls tools by name
        return str(await agent.tools.call("report_25", store="S1"))

    agent = harness.wrap(fn, id="fn", tools=tools(30))
    assert "report 25" in str((await agent.run("lost sales?", user="u")).answer)


# --------------------------------------------------------------------------- the window
def wide(n: int) -> list[Any]:
    """Tools with large definitions: a long description each."""

    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            """Look a key up.

            Args:
                key: The key.
            """
            return "found"

        words = " ".join(f"detail{i}x{j}" for j in range(150))
        return tool(
            lookup, name=f"lookup_{i}", description=f"Lookup {i}: {words}", side_effects="read"
        )

    return [make(i) for i in range(n)]


@pytest.mark.parametrize(("window", "narrowed"), [(4_000, True), (128_000, False)])
async def test_definitions_crowding_the_window_are_narrowed_whatever_their_number(
    harness: Harness, window: int, narrowed: bool
) -> None:
    catalogue = [*wide(10), lost_sales]
    tokens = search.definition_tokens(t.spec for t in catalogue)
    assert (tokens > TOOLS_SHARE * window) is narrowed
    model = ScriptedChatModel(turns=["ok"])
    agent = harness.wrap(ReAct("s", model, context_window=window), id=f"w{window}", tools=catalogue)
    await agent.run("Which styles lost sales at S12?", user="u")
    offered = names(model.requests[0])
    assert ("lost_sales" in offered) and ((len(offered) < 11) is narrowed)


async def test_definitions_that_fill_a_small_window_do_not_stop_a_long_run(
    harness: Harness,
) -> None:
    """Fifty tools on an 8k window, a four-step run: the definitions are narrowed before the
    summarization guard counts them, every call is offered the few tools the task is about,
    and the run completes (on the local model the benchmark measured, the guard — unable to
    shrink tool definitions — stopped such runs)."""

    @tool(side_effects="read")
    def sales(store: str) -> str:
        """The sales a store lost to stock-outs.

        Args:
            store: The store's id.
        """
        return "lost " + "x" * 3000

    steps: list[Any] = [("sales", {"store": f"S{i}"}) for i in range(4)]
    model = ScriptedChat(script=Summarizing([*steps, "done"]))
    target = ReAct("s", model, context_window=8_192)
    agent = harness.wrap(target, id="small", tools=[*wide(49), sales])
    result = await agent.run("Lost sales at S0, S1, S2 and S3?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    offered = [len(r.get("tools", [])) for r in model.requests if r.get("tools")]
    assert offered and max(offered) <= TOP_K + 2  # the search's, and the read tools


# --------------------------------------------------------------------------- the search
def spec(name: str, description: str = "", **params: str) -> ToolSpec:
    properties = {p: {"type": "string", "description": d} for p, d in params.items()}
    return ToolSpec(name=name, description=description, input_schema={"properties": properties})


def test_a_tool_is_found_by_its_parameters_description_alone() -> None:
    specs = [
        spec("q1", "Runs a query.", ref="The invoice number printed on the bill"),
        spec("q2", "Runs a query.", ref="The order id"),
    ]
    assert search.search("find the bill with invoice number 42", specs)[0] == "q1"


def test_a_tool_with_no_description_is_found_by_its_name() -> None:
    specs = [spec("get_weather"), spec("get_stock", "Units in stock.")]
    assert search.search("what is the weather in Oslo", specs) == ["get_weather"]


def test_the_same_tool_on_two_servers_is_two_names_both_found() -> None:
    specs = [
        spec("wiki1-read_page", "Read a wiki page."),
        spec("wiki2-read_page", "Read a wiki page."),
        spec("crm-get_deal", "Get a deal."),
    ]
    assert set(search.search("read the wiki page", specs, 2)) == {
        "wiki1-read_page",
        "wiki2-read_page",
    }


def test_code_mode_and_skills_tools_are_always_kept() -> None:
    async def run(args: dict[str, Any]) -> Any:
        return None

    kept = [
        Tool(spec("execute_tool_code", "Run a script."), run, feature="code_mode"),
        Tool(spec("load_skill", "Load a skill."), run, feature="skills"),
    ]
    others = [Tool(spec(f"r{i}", f"Report {i}."), run) for i in range(30)]
    offered = search.offered_for("report 3", [*others, *kept])
    assert {"execute_tool_code", "load_skill", "r3"} <= set(offered)


def test_the_terms_and_the_estimate() -> None:
    assert search.terms("Expense_claims for the policies, booking") == [
        "expense",
        "claim",
        "policy",
        "book",
    ]
    assert search.stem("bus") == "bus" and search.stem("tickets") == "ticket"
    assert search.ranked("anything", []) == []
    assert search.search("zzz", [spec("a", "b")]) == []
    assert search.definition_tokens([spec("a", "b" * 400)]) > 100
    assert search.named("use report_7, not report_77x", ["report_7", "report_77"]) == ["report_7"]


def test_a_parameter_schema_that_is_true_still_names_the_parameter() -> None:
    specs = [ToolSpec(name="t", description="", input_schema={"properties": {"invoice": True}})]
    assert search.search("invoice", specs) == ["t"]


def test_a_steps_search_reads_the_latest_answers_back_to_the_models_last_words() -> None:
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from trellis.harness.automatic import _latest

    history = [
        SystemMessage("context"),
        HumanMessage("asked"),
        AIMessage("thinking aloud"),
        ToolMessage("result one", tool_call_id="1"),
        ToolMessage("result two", tool_call_id="2"),
    ]
    request: Any = SimpleNamespace(state={"messages": history})
    assert _latest(request) == "result two result one thinking aloud"
    request = SimpleNamespace(state={"messages": [SystemMessage("only context")]})
    assert _latest(request) == "only context"
