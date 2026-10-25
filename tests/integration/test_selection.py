"""``without=``: everything configured is on, and one switch turns parts of it off — for every
run of an agent (``h.wrap``) or for one run (``agent.run``/``stream``/``start``, added to the
agent's, kept with the run across a pause, a worker and its sub-agents' runs)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import pytest
from pydantic import Field

from tests.support.adapters import BUILDERS
from tests.support.arm import arm, native_tools
from tests.support.memory import FakeMemoryService
from tests.support.planned import Call, PlannedChatModel
from tests.unit.test_toolbox import Def
from tests.unit.test_toolbox import Gateway as StubGateway
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import middleware
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.evals import EvalCase, EvalScore
from trellis.harness.features import features
from trellis.harness.middleware import ModelHooks

#: The calls through which a run reaches the memory service.
MEMORY_CALLS: Final = {"context", "agent_tools", "messages", "record_tool", "feedback", "verify"}


@tool(side_effects="read")
def stock(sku: str) -> str:
    """Units in stock."""
    return f"{sku}: 7"


def reached(service: FakeMemoryService) -> set[str]:
    return {c.name for c in service.calls} & MEMORY_CALLS


def test_without_names_features_and_refuses_anything_else() -> None:
    memory = {"memory_push", "memory_pull", "records", "hints", "tool_search"}
    assert features(["memory", "judges"]) == {*memory, "judges"}
    assert features(["mcp"]) == {"mcp", "code_mode"}
    assert features(["tool_search"]) == {"hints", "tool_search"}
    every = "memory, tool_search, mcp, code_mode, grounding, judges"
    with pytest.raises(
        ConfigurationError, match=f"names no feature 'memroy': the features are {every}"
    ):
        features(["memroy"])


async def test_an_unknown_feature_is_refused_on_wrap_and_on_run(harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return input

    with pytest.raises(ConfigurationError, match="names no feature 'judge'"):
        harness.wrap(fn, id="a", without={"judge"})  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="names no feature 'record'"):
        await harness.wrap(fn, id="b").run("x", user="u", without={"record"})  # type: ignore[arg-type]


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_run_without_memory_reaches_no_memory_on_every_adapter(
    framework: str, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    plan: list[Call] = [("stock", {"sku": "A-1"})]
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        target, tools = await BUILDERS[framework](h, [stock], tmp_path, plan)
        agent = arm(h.wrap(target, id=f"stock-{framework}"), tools)
        memory_service.calls.clear()  # a graph's memory tools were listed when it was built
        alone = await agent.run("How many A-1?", user="ada", without={"memory"})
        await h.writes.drain()
        assert alone.status is RunStatus.SUCCESS and alone.answer == "Done. A-1: 7"
        assert reached(memory_service) == set()
        record = await h.runs.get(alone.run_id)
        assert record is not None and record.metadata == {
            "without": ["hints", "memory_pull", "memory_push", "records", "tool_search"]
        }
        remembered = await agent.run("How many A-1?", user="ada")  # everything on again
        await h.writes.drain()
    assert {"context", "messages", "record_tool"} <= reached(memory_service)
    assert remembered.status is RunStatus.SUCCESS


async def test_each_feature_turns_off_its_part(
    memory_service: FakeMemoryService, sampled: Any
) -> None:
    seen: list[dict[str, Any]] = []

    async def fn(input: str, agent: Runtime) -> str:
        seen.append({"context": agent.context, "tools": sorted(agent.toolbox)})
        return "fine"

    judged: list[str] = []

    async def judge(case: EvalCase) -> EvalScore | None:
        judged.append(case.output)
        return None

    defs = [Def(f"wiki-t{i}", "wiki") for i in range(20)]  # read only: Code Mode
    gateway = cast("Gateway", StubGateway([*defs, Def("erp-pay", "erp", annotations=None)]))
    sampled(1.0)
    settings = Settings()
    async with Harness(config=settings, memory=memory_service.client(), gateway=gateway) as h:
        agent = arm(h.wrap(fn, id="all", judges=[judge]), [stock])

        async def run(*without: str) -> tuple[dict[str, Any], set[str]]:
            memory_service.calls.clear()
            judged.clear()
            await agent.run("How many A-1?", user="ada", without=without)  # type: ignore[arg-type]
            await h.writes.drain()
            return seen[-1], {c.name for c in memory_service.calls}

        everything, calls = await run()
        assert everything["context"] and "memory_search" in everything["tools"]
        assert "execute_tool_code" in everything["tools"] and "erp-pay" in everything["tools"]
        assert {"context", "messages", "verify"} <= calls
        assert judged == ["fine"]
        alone, calls = await run("memory")
        assert alone["context"] is None and "memory_search" not in alone["tools"]
        assert not {"context", "agent_tools", "messages", "record_tool", "feedback"} & calls
        searched, calls = await run("tool_search")
        assert searched["context"] and "memory_search" in searched["tools"]
        [asked] = memory_service.named("context")
        assert asked.body["tools"]["hints"] is False
        _, calls = await run("grounding")
        assert "verify" not in calls
        await run("judges")
        assert judged == []
        scripted, _ = await run("code_mode")
        assert "execute_tool_code" not in scripted["tools"] and "wiki-t0" in scripted["tools"]
        bare, _ = await run("mcp")
        assert "stock" in bare["tools"] and "memory_search" in bare["tools"]
        assert not [
            n
            for n in bare["tools"]
            if n.startswith(("wiki", "erp", "execute", "list_tool", "read_tool", "get_tool"))
        ]


async def test_without_tool_search_the_context_asks_for_no_tool_hints(
    memory_service: FakeMemoryService,
) -> None:
    """Without tool search the agent still asks with its tools (for the skills it learned), but no
    hints come back and every tool is offered."""
    many = [tool(lambda sku: sku, name=f"t{i}", side_effects="read") for i in range(5)]

    async def fn(input: str, agent: Runtime) -> list[str]:
        return sorted(n for n in agent.toolbox if agent.offers(n))

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(fn, id="hinted"), many)
        await agent.run("q", user="u")
        [hinted] = memory_service.named("context")
        assert hinted.body["tools"] == {"available": [f"t{i}" for i in range(5)], "k": 8}
        memory_service.calls.clear()
        offered = await arm(h.wrap(fn, id="unhinted", without={"tool_search"}), many).run(
            "q", user="u"
        )
        [unhinted] = memory_service.named("context")
        assert unhinted.body["tools"]["hints"] is False
        assert {f"t{i}" for i in range(5)} <= set(offered.answer), "every tool offered"


async def test_without_mcp_the_gateway_is_not_asked_for_the_keys_tools(
    memory_service: FakeMemoryService,
) -> None:
    """``without={"mcp"}``: the key's MCP tools are neither listed nor published to the
    catalog — for a wrapped agent, and for a graph's tools built so (``h.tools``)."""

    async def fn(input: str, agent: Runtime) -> list[str]:
        return sorted(agent.toolbox)

    stub = StubGateway([Def("wiki-search", "wiki")])
    async with Harness(
        config=Settings(), memory=memory_service.client(), gateway=cast("Gateway", stub)
    ) as h:
        agent = arm(h.wrap(fn, id="local", without={"mcp"}), [stock])
        local = (await agent.run("q", user="u")).answer
        assert "stock" in local and "wiki-search" not in local
        built = await native_tools(h, stock, framework="langgraph", without={"mcp", "memory"})
        assert [t.name for t in built] == ["stock"]
        await h.writes.drain()
        assert stub.listed == 0
        published = json.dumps([c.body for c in memory_service.named("put_catalog")])
        assert "stock" in published and "wiki-search" not in published
        assert "wiki-search" in (await h.wrap(fn, id="all").run("q", user="u")).answer
        assert stub.listed == 1


class Offering(PlannedChatModel):
    """A planned model that keeps the tools it was offered at each call."""

    offered: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Offering:  # type: ignore[override]
        self.offered.append(sorted(getattr(t, "name", "") for t in tools))
        return self


async def test_a_graph_with_the_middleware_is_offered_only_the_tools_its_run_uses(
    memory_service: FakeMemoryService,
) -> None:
    """A graph's tools are bound when it is built; the harness's middleware hides from its
    model those of a part turned off afterwards — the agent's ``without=``, or a run's."""
    from langchain.agents import create_agent

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        tools = await native_tools(h, stock, framework="langgraph")
        bound = sorted(t.name for t in tools)
        assert "memory_search" in bound and "stock" in bound
        model = Offering(plan=[("stock", {"sku": "A-1"})])
        graph = create_agent(model, tools=tools, middleware=[ModelHooks()])
        everything = h.wrap(graph, id="everything")
        assert (await everything.run("A-1?", user="u")).answer == "Done. A-1: 7"
        assert model.offered[-1] == bound
        await everything.run("A-1?", user="u", without={"memory"})
        assert model.offered[-1] == ["stock"]
        pulled = h.wrap(graph, id="unpulled", without={"memory"})
        assert (await pulled.run("A-1?", user="u")).answer == "Done. A-1: 7"
        assert model.offered[-1] == ["stock"]
        request = cast("Any", Tools([]))
        assert middleware._offered(request, None) is request  # no run: as is


class Tools:
    """A model request as the middleware reads it, outside any run."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools
        self.messages: list[Any] = []


async def test_a_runs_own_without_adds_to_the_agents_and_holds_across_a_resume(
    memory_service: FakeMemoryService,
) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        return f"{input}: {await agent.ask('Go on?')}"

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = h.wrap(asks, id="asks", without={"judges"})
        paused = await agent.run("q", user="u", without={"memory"})
        assert paused.interrupt is not None
        memory_service.calls.clear()
        done = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u"
        )
        await h.writes.drain()
    assert done.answer == "q: yes"
    assert not {"context", "messages", "feedback"} & {c.name for c in memory_service.calls}


async def test_a_queued_run_and_its_sub_agents_keep_its_without(
    memory_service: FakeMemoryService,
) -> None:
    async def child(input: str, agent: Runtime) -> str:
        return f"child did {input}"

    async def parent(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("helper", message=input)

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        helper = h.wrap(child, id="helper")
        boss = arm(h.wrap(parent, id="boss"), [helper])
        handle = await boss.start("x", user="u", without={"memory"})
        memory_service.calls.clear()
        assert await h.worker([boss]).run_once()
        done = await handle.result(wait=5)
        await h.writes.drain()
        assert done.answer == "child did x"
        [summary] = [r async for r in h.runs.iterate(parent_run_id=handle.id)]
        child_run = await h.runs.get(summary.run_id)
        assert child_run is not None
        assert child_run.metadata == {
            "without": ["hints", "memory_pull", "memory_push", "records", "tool_search"],
            "agent_group": handle.id,  # the delegation tree, named by its top run
        }
    assert reached(memory_service) == set()
