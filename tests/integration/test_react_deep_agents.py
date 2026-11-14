"""``ReAct`` on Deep Agents (``create_deep_agent``) and the harness's part on its own
(``trellis.middleware()``): Deep Agents' sub-agents as ``ReAct`` gives them (their model names
routed, the loop's guards, their ``interrupt_on`` as the catalog's), the tools and sub-agents each
parameter takes, the store and checkpointer a wrapped agent refuses, the tools of a backend
outside the graph's state governed, the graph's machinery left out of the run's tool set, and
``middleware()`` on a ``create_agent`` and a ``create_deep_agent`` graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.tools import tool as lc_tool

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, a2a, middleware, sandbox, stored, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.adapters.langgraph import bound_tools, graph_tools
from trellis.harness.automatic import JournalBackend
from trellis.harness.gateways import routed
from trellis.harness.middleware import KEPT_BY_RUN, MACHINERY, OUTSIDE_STATE, RunCheckpointer
from trellis.harness.react import PLACED, _native, defaults_off


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU.
    """
    return 7


@lc_tool
def wire(amount: int) -> str:
    """Wire money.

    Args:
        amount: How much.
    """
    return f"wired {amount}"


def names(model: ScriptedChatModel, n: int = 0) -> set[str]:
    return set(model.requests[n].get("tools", []))


def marked(graph: Any, key: str) -> dict[str, Any]:
    return {t.name: (t.metadata or {}).get(key) for t in graph_tools(graph)}


# --------------------------------------------------------------------------- middleware()
def test_middleware_is_the_harness_part_in_its_order() -> None:
    found = [m.name for m in middleware(context_window=64_000)]
    assert found == [
        "TodoListMiddleware",
        "HarnessTools",
        "GovernTools",
        "StallGuard",
        "ModelCallLimitMiddleware",
        "Budgeted",
        "ContextEditingMiddleware",
        "ReadTools",
        "FitWindow",
        "ModelHooks",
        "AnthropicPromptCachingMiddleware",  # PromptCache, in Deep Agents' own one's place
    ]
    with pytest.raises(
        TypeError, match=r"middleware\(capabilities=\) was removed: .*system_prompt"
    ):
        middleware(capabilities={"upload": False})  # pyright: ignore[reportCallIssue]
    for name, said in (("tool_guidance", "tool guidance"), ("answer_check", "answer check")):
        given: dict[str, Any] = {name: True}
        with pytest.raises(TypeError, match=rf"middleware\({name}=\) was removed: the {said}"):
            middleware(**given)


def test_middleware_settings_mean_what_reacts_do() -> None:
    stack = {
        m.name: m
        for m in middleware(max_steps=4, max_repeats=5, call_timeout=2.0, context_window=64_000)
    }
    # LangChain's model call limit, per invocation here (the graph's thread may outlive a run):
    # max_steps calls, then one that answers (LangChain's own limit, one later, the backstop)
    limit = stack["ModelCallLimitMiddleware"]
    assert (limit.max_steps, limit.run_limit, limit.thread_limit) == (4, 5, None)
    assert stack["StallGuard"].max_repeats == 5
    assert stack["ModelHooks"].timeout == 2.0
    pinned = {m.name: m for m in middleware(system_prompt=stored("triage"))}
    assert pinned["ModelHooks"].prompt == stored("triage")


@pytest.mark.parametrize(
    ("given", "said"),
    [
        ({"max_steps": 0}, "max_steps is a number of steps, 1 or more"),
        ({"max_repeats": 1}, "max_repeats is a number of steps, 2 or more"),
        ({"call_timeout": 0}, "call_timeout is a number of seconds over 0"),
        ({"context_window": -1}, "context_window is a number of tokens over 0"),
        ({"system_prompt": "a text"}, "a stored prompt"),
    ],
)
def test_middleware_refuses_what_is_not_a_setting(given: dict[str, Any], said: str) -> None:
    with pytest.raises(ConfigurationError, match=said):
        middleware(**given)


async def test_middleware_on_create_agent_gives_its_runs_the_harness_tools(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "done"])
    graph = create_agent(model, middleware=middleware(max_steps=2), checkpointer=RunCheckpointer())
    agent = arm(harness.wrap(graph, id="plain"), [stock])
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    assert "stock" in names(model, 0)  # the run's tool, offered per call
    assert len(model.requests) == 2  # the step and the answer, within max_steps


async def test_middleware_on_create_deep_agent_runs_and_resumes_in_place(
    harness: Harness,
) -> None:
    @tool(side_effects="irreversible")
    def publish(text: str) -> str:
        """Publish a text.

        Args:
            text: What.
        """
        return f"published {text}"

    model = ScriptedChatModel(turns=[("publish", {"text": "x"}), "published"])
    graph = create_deep_agent(model, middleware=middleware(), checkpointer=RunCheckpointer())
    agent = arm(harness.wrap(graph, id="deep"), [publish])
    paused = await agent.run("publish x", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "published" and len(model.seen) == 2  # not asked again


async def test_a_graph_wrapped_without_the_middleware_is_said_once(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, "trellis.harness")
    graph = create_agent(ScriptedChatModel(turns=[]), name="bare-graph")
    harness.wrap(graph, id="bare")
    harness.wrap(graph, id="bare-again")
    said = [r for r in caplog.records if "without the harness's middleware" in r.getMessage()]
    assert len(said) == 1 and "bare-graph" in said[0].getMessage()
    caplog.clear()
    harness.wrap(ReAct("s", ScriptedChatModel(turns=[])), id="react")
    assert not [r for r in caplog.records if "without the harness's" in r.getMessage()]


# --------------------------------------------------------------------------- tools, sub-agents
def test_a_sandbox_is_a_backend_and_an_agent_a_sub_agent(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[])
    for given in (sandbox(FakeSandboxes()), sandbox(FakeSandboxes()).tools()):
        with pytest.raises(ConfigurationError, match=r"ReAct\(backend=sandbox\(\)\), not tools="):
            ReAct("s", model, tools=[given])
    with pytest.raises(ConfigurationError, match="an agent is a sub-agent"):
        ReAct("s", model, tools=[a2a("http://agent.test")])
    helper = harness.wrap(ReAct("s", model), id="helper")
    with pytest.raises(ConfigurationError, match="an agent is a sub-agent"):
        ReAct("s", model, tools=[helper])
    graph = ReAct("s", model, subagents=[helper, a2a("http://agent.test")])
    assert len(graph.trellis_tools) == 2  # each a tool of the run, its call a child run


def test_a_declarative_sub_agent_is_given_the_loops_guards_and_a_routed_model() -> None:
    mine = ModelCallLimitMiddleware(run_limit=2)
    cached = AnthropicPromptCachingMiddleware(ttl="1h")
    spec = _native(
        {
            "name": "payer",
            "description": "pays",
            "model": "gpt-4o-mini",
            "middleware": [mine, cached],
            "tools": [wire, stock_lc],
            "interrupt_on": {"wire": True, "stock_lc": False, "write_file": True},
        },
        call_timeout=5.0,
        max_steps=9,
        max_repeats=3,
    )
    assert routed(spec["model"])
    assert [m.name for m in spec["middleware"]] == [
        "ModelCallLimitMiddleware",
        "StallGuard",
        "ModelHooks",
        "AnthropicPromptCachingMiddleware",  # in Deep Agents' own one's place
    ]
    assert spec["middleware"][0] is mine  # its own kept
    assert spec["middleware"][2].timeout == 5.0
    assert spec["middleware"][3].native is cached  # its own caching, for a routed model too
    by_name = {t.name: t for t in spec["tools"]}
    assert by_name["wire"].metadata["side_effects"] == "irreversible"  # the catalog's ask
    assert "side_effects" not in (by_name["stock_lc"].metadata or {})
    assert spec["interrupt_on"] == {"write_file": True}  # not its tool: Deep Agents' gate
    plain = _native(
        {"name": "p", "description": "d", "interrupt_on": {"x": False}},
        call_timeout=None,
        max_steps=1,
        max_repeats=2,
    )
    assert "interrupt_on" not in plain and "tools" not in plain
    compiled = {"name": "c", "description": "d", "runnable": object()}
    assert _native(dict(compiled), call_timeout=None, max_steps=1, max_repeats=2) == compiled


@lc_tool
def stock_lc(sku: str) -> int:
    """Units in stock.

    Args:
        sku: The SKU.
    """
    return 7


def test_a_function_named_in_interrupt_on_becomes_a_tool_that_asks() -> None:
    def refund(order: str) -> str:
        """Refund an order.

        Args:
            order: The order.
        """
        return order

    spec = _native(
        {"name": "r", "description": "d", "tools": [refund], "interrupt_on": {"refund": True}},
        call_timeout=None,
        max_steps=1,
        max_repeats=2,
    )
    assert spec["tools"][0].name == "refund"
    assert spec["tools"][0].metadata["side_effects"] == "irreversible"


async def test_a_sub_agents_tool_named_in_its_interrupt_on_asks_a_person(
    harness: Harness,
) -> None:
    helper = ScriptedChatModel(turns=[("wire", {"amount": 5}), "wired 5"])
    payer = {"name": "payer", "description": "Pays.", "model": helper, "tools": [wire]}
    payer["interrupt_on"] = {"wire": True}
    model = ScriptedChatModel(
        turns=[("task", {"description": "pay 5", "subagent_type": "payer"}), "paid"]
    )
    agent = harness.wrap(ReAct("s", model, subagents=[payer]), id="pays")
    paused = await agent.run("pay 5", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    assert "wire" in paused.interrupt.question
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "paid", done.error
    assert "wired 5" in str(model.seen[-1][-1].content)


def test_a_compiled_graph_is_a_sub_agent_with_its_name_and_description() -> None:
    inner = ReAct("inner", ScriptedChatModel(turns=[]))
    graph = ReAct(
        "s",
        ScriptedChatModel(turns=[]),
        subagents=[{"name": "inner", "description": "Helps.", "runnable": inner}],
    )
    task = next(t for t in bound_tools(graph) if t.name == "task")
    assert not (task.metadata or {}).get(MACHINERY)  # an agent of its own: not machinery


async def test_on_your_own_graph_a_cleared_result_is_read_back(harness: Harness) -> None:
    """``create_agent`` with ``middleware(context_window=)``: once an older result was cleared
    from the request, ``read_file`` is offered (the graph has none of its own) and reads it
    back by its call's path — a result that is no table, so nothing else offered a reader."""

    @tool(side_effects="read")
    def page(n: int) -> str:
        """A page of a book."""
        return f"page {n} " + "w" * 2000

    turns: list[Any] = [("page", {"n": n}) for n in range(1, 7)]
    model = ScriptedChatModel(turns=[*turns, "done"])
    target = create_agent(
        model, middleware=middleware(context_window=6000), checkpointer=RunCheckpointer()
    )
    result = await arm(harness.wrap(target, id="own-cleared"), [page]).run("read", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    offered = ["read_file" in set(r.get("tools", [])) for r in model.requests]
    assert offered[0] is False and offered[-1] is True


# --------------------------------------------------------------------------- backends
def test_without_a_deep_agents_capability_react_holds_deep_agents_machinery() -> None:
    """No backend, sub-agent of its own, skills, memory files or permissions: Deep Agents'
    graph all the same, its file tools and general-purpose helper there for a long task
    (``automatic.long_task``) — with a large result kept by the run, as on ``create_agent``
    with ``middleware()`` (Deep Agents' ``read_file`` is marked so), where a graph asked for a
    backend leaves it to Deep Agents."""
    graph = ReAct("s", ScriptedChatModel(turns=[]))
    assert {t.name for t in graph_tools(graph)} >= {"write_todos", "task", "write_file"}
    assert marked(graph, KEPT_BY_RUN)["read_file"]
    asked = ReAct("s", ScriptedChatModel(turns=[]), backend=JournalBackend())
    assert not marked(asked, KEPT_BY_RUN)["read_file"]


def test_on_the_graphs_state_the_file_tools_are_its_own_and_execute_is_machinery() -> None:
    graph = ReAct("s", ScriptedChatModel(turns=[]), backend=JournalBackend())
    outside = marked(graph, OUTSIDE_STATE)
    assert not any(outside.values())
    machinery = marked(graph, MACHINERY)
    assert machinery["execute"] and machinery["task"]  # runs nothing; the general helper
    stated = ReAct("s", ScriptedChatModel(turns=[]), backend=StateBackend())
    assert not any(marked(stated, OUTSIDE_STATE).values())


async def test_on_a_disk_the_file_tools_are_governed_and_offered_from_the_start(
    harness: Harness, tmp_path: Path
) -> None:
    (tmp_path / "notes.md").write_text("seven")
    model = ScriptedChatModel(turns=[("read_file", {"file_path": "/notes.md"}), "seven"])
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    graph = ReAct("s", model, backend=backend)
    outside = marked(graph, OUTSIDE_STATE)
    assert outside["read_file"] and outside["write_file"]
    sides = {t.name: (t.metadata or {}).get("side_effects") for t in graph_tools(graph)}
    assert sides["read_file"] == "read" and sides["edit_file"] == "write"
    reads = {"ls", "read_file", "glob", "grep"}  # the reads read; the others write
    governed = [n for n, mark in outside.items() if mark]
    assert set(governed) == {"ls", "read_file", "glob", "grep", "write_file", "edit_file", "delete"}
    assert all(sides[n] == ("read" if n in reads else "write") for n in governed)
    assert outside["delete"] and sides["delete"] == "write"
    agent = harness.wrap(graph, id="disk")
    result = await agent.run("what do the notes say?", user="u")
    assert result.answer == "seven", result.error
    assert {"read_file", "ls", "write_file"} <= names(model, 0)  # the agent's from the start
    assert "seven" in str(model.seen[1][-1].content)
    native = {s.name for s in await agent.adapter.native(graph)}
    assert {"read_file", "write_file"} <= native and "execute" not in native


# --------------------------------------------------------------------------- measuring
def test_a_measurement_turns_a_deep_agents_default_off_by_its_name() -> None:
    with defaults_off(PLACED, "SummarizationMiddleware", "StallGuard"):
        graph = ReAct("s", ScriptedChatModel(turns=[]))
    nodes = set(graph.nodes)
    assert not any("Summarization" in n for n in nodes)
    assert "StallGuard" not in str(graph.nodes)


async def test_memory_tools_reach_a_deep_agent_given_the_middleware(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChatModel(turns=["hi"])
    graph = create_deep_agent(model, middleware=middleware(), checkpointer=RunCheckpointer())
    result = await memory_harness.wrap(graph, id="remembering").run("hi", user="u")
    assert result.answer == "hi", result.error
    assert any(name.startswith("memory_") for name in names(model, 0))


async def test_a_sub_agent_that_names_no_tools_has_the_runs(harness: Harness) -> None:
    """Deep Agents gives a sub-agent that names no tools the main agent's. ``ReAct``'s tools are
    the run's, offered per model call (``HarnessTools``), so its sub-agents had none of them:
    told to look a figure up, a helper answered that it had no such tool (after searching the
    files for one) and the parent made every call itself — 20 helpers, 69 model calls, the
    tools called by the parent. A sub-agent that names its own tools keeps exactly those."""
    executed: list[str] = []

    @tool(side_effects="read")
    def stock(sku: str) -> int:
        """Units of a SKU in stock."""
        executed.append(f"stock:{sku}")
        return 7

    helper = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7 units of a"])
    counter = {"name": "counter", "description": "Counts stock.", "model": helper}
    model = ScriptedChatModel(
        turns=[("task", {"description": "count a", "subagent_type": "counter"}), "done"]
    )
    agent = harness.wrap(ReAct("s", model, tools=[stock], subagents=[counter]), id="counts")
    result = await agent.run("how many a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    assert "stock" in helper.requests[0]["tools"]  # offered the run's tool
    assert executed == ["stock:a"]  # and it ran, governed as the run's
    assert "7 units of a" in str(model.seen[-1][-1].content)
    # the general-purpose helper, the same model: its request is offered the run's tools too
    executed.clear()
    one = ScriptedChatModel(
        turns=[
            ("task", {"description": "count b", "subagent_type": "general-purpose"}),
            ("stock", {"sku": "b"}),
            "7 units of b",
            "done",
        ]
    )
    graph = ReAct("s", one, tools=[stock], backend=JournalBackend())
    general = harness.wrap(graph, id="counts-general")
    again = await general.run("how many b?", user="u")
    assert again.interrupt is None  # handing the task on asks no one (test_react_helpers.py)
    assert again.status is RunStatus.SUCCESS and again.answer == "done", again.error
    assert executed == ["stock:b"] and "stock" in one.requests[1]["tools"]
    # its own tools named: those, and no other
    own = ScriptedChatModel(turns=["nothing to do"])
    payer = {"name": "payer", "description": "Pays.", "model": own, "tools": [wire]}
    parent = ScriptedChatModel(
        turns=[("task", {"description": "pay", "subagent_type": "payer"}), "done"]
    )
    await harness.wrap(ReAct("s", parent, tools=[stock], subagents=[payer]), id="pays-own").run(
        "pay", user="u"
    )
    assert "wire" in own.requests[0]["tools"] and "stock" not in own.requests[0]["tools"]
