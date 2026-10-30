"""The gateway's stored prompts through a wrapped agent: ``ReAct(system_prompt=stored(...))``
pinned for the run on every model call (selected through Bifrost, or rendered when it has
variables to fill), a graph's through the harness's middleware, and ``h.prompt`` for any
framework — inside a run pinned and journaled, so a resumed run reads what it started with."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import langchain_openai
import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from tests.support.gateway import URL, FakeGateway, completion
from trellis import Harness, ReAct, Runtime, Settings, stored, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness import fresh
from trellis.harness.clients import bifrost
from trellis.harness.middleware import ModelHooks
from trellis.harness.prompts import Rendered

SYSTEM = [{"role": "system", "content": "You triage."}]
GREET = [{"role": "system", "content": "You help the {{team}} team."}]


@pytest.fixture
def fake() -> FakeGateway:
    return FakeGateway(prompts={"triage": [SYSTEM, SYSTEM], "greet": [GREET]})


@pytest.fixture
async def h(fake: FakeGateway, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Harness]:
    fake.serve_models(monkeypatch)
    settings = Settings(llm_gateway="bifrost", llm_base_url=URL, llm_api_key="vk")
    async with Harness(config=settings, gateway=fake.gateway()) as made:
        yield made


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: now[0])
    return now


def newer(fake: FakeGateway, clock: list[float]) -> None:
    """A newer version of ``triage`` committed, and read once the kept one is stale."""
    fake.prompts["triage"].append(SYSTEM)
    clock[0] += bifrost.REPOSITORY_TTL_SECONDS + 1


def events_of(found: list[Any], name: str) -> list[dict[str, Any]]:
    return [
        e.data
        for e in found
        if e.type is RunEventType.CUSTOM and (e.data or {}).get("name") == name
    ]


@tool(side_effects="irreversible")
def close_ticket(ticket: str) -> str:
    """Close a ticket."""
    return f"closed {ticket}"


async def test_every_model_call_selects_the_prompt_pinned_for_the_run(
    h: Harness, fake: FakeGateway, spans: InMemorySpanExporter, clock: list[float]
) -> None:
    fake.chat.turns += [("close_ticket", {"ticket": "T-1"}), "closed it"]
    target = ReAct(system_prompt=stored("triage"), model="local/small", tools=[close_ticket])
    agent = h.wrap(target, id="triage")
    paused = await agent.run("close T-1", user="ada")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    newer(fake, clock)  # a newer version, read after the TTL...
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "closed it"
    # ...but the run keeps the version it started with, on every call
    assert [r.headers["x-bf-prompt-version"] for r in fake.completions] == ["2", "2"]
    assert {r.headers["x-bf-prompt-id"] for r in fake.completions} == {"p-triage"}
    chats = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    assert [(s.attributes or {})["trellis.prompt.version"] for s in chats] == [2, 2]
    assert {(s.attributes or {})["trellis.prompt.name"] for s in chats} == {"triage"}
    fresh_run = await agent.run("hello", user="ada")  # a new run pins the newer version
    assert fresh_run.status is RunStatus.ERROR  # (the script has nothing more to say)
    assert fake.completions[-1].headers["x-bf-prompt-version"] == "3"


async def test_a_prompt_with_variables_is_rendered_into_the_instructions(
    h: Harness, fake: FakeGateway
) -> None:
    fake.chat.turns += [("look", {}), "hello"]  # two model calls: one pin

    @tool(side_effects="read")
    def look() -> str:
        """Look."""
        return "seen"

    target = ReAct(system_prompt=stored("greet", team="EU"), model="local/small", tools=[look])
    agent = h.wrap(target, id="g")
    events = [e async for e in agent.stream("hi", user="ada")]
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    for sent in fake.completions:  # filled here: nothing for the gateway to select
        assert "x-bf-prompt-id" not in sent.headers
        assert completion(sent)["messages"][0] == {
            "role": "system",
            "content": "You help the EU team.",
        }
    assert events_of(events, "prompt") == [
        {"name": "prompt", "prompt": "greet", "version": "1", "source": "Bifrost"}
    ]


def test_a_stored_prompt_is_a_name_or_name_at_version() -> None:
    for ref in ("triage@", "@1"):
        with pytest.raises(ConfigurationError, match="is not a name"):
            stored(ref)


@pytest.mark.parametrize(
    ("prompt", "problem"),
    [
        (stored("triage@latest"), "a stored prompt's version is a number from 1, not 'latest'"),
        (stored("greet"), "the prompt greet@1 needs team: pass them"),
        (stored("nope"), "no committed prompt named 'nope'"),
    ],
)
async def test_a_run_whose_prompt_cannot_be_given_fails_saying_why(
    h: Harness, prompt: Any, problem: str
) -> None:
    agent = h.wrap(ReAct(system_prompt=prompt, model="local/small"), id="t")
    result = await agent.run("hi", user="ada")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert problem in result.error.message


async def test_a_stored_prompt_needs_the_gateway() -> None:
    async with Harness(config=Settings()) as bare:
        with pytest.raises(ConfigurationError, match="set LLM_GATEWAY=bifrost"):
            await bare.prompt("triage")


async def test_h_prompt_renders_any_frameworks_prompt_and_a_run_pins_it(
    h: Harness, fake: FakeGateway, clock: list[float]
) -> None:
    assert await h.prompt("triage@1") == Rendered("You triage.", SYSTEM)
    assert (await h.prompt(stored("greet", team="EU"))).text == "You help the EU team."
    assert (await h.prompt("greet", team="ops")).text == "You help the ops team."
    seen: list[str] = []

    async def triage(question: str, agent: Runtime) -> str:
        before = await h.prompt("triage")  # pinned for the run: journaled
        await agent.ask("Go on?")
        after = await h.prompt("triage")
        seen.append(before.text)
        return f"{before == after}"

    agent = h.wrap(triage, id="triage")
    events = [e async for e in agent.stream("x", user="ada")]
    paused = await h.runs.get(events[-1].run_id)
    assert paused is not None and paused.awaiting is not None
    assert [e["version"] for e in events_of(events, "prompt")] == ["2"]
    newer(fake, clock)  # 3: committed while the run waits
    done = await agent.resume(paused.awaiting.interrupt_id, "answer", answer="y", reviewer="a")
    assert done.answer == "True" and seen == ["You triage."]


async def test_a_graphs_model_gets_the_prompt_through_the_harness_middleware(
    h: Harness, fake: FakeGateway, spans: InMemorySpanExporter, clock: list[float]
) -> None:
    model = langchain_openai.ChatOpenAI(model="local/small", base_url=URL, api_key=SecretStr("vk"))
    graph = create_agent(model, middleware=[ModelHooks(prompt=stored("triage"))])
    agent = h.wrap(graph, id="graph")
    fake.chat.turns.append("triaged")
    newer(fake, clock)
    assert (await agent.run("triage this", user="ada")).answer == "triaged"
    [sent] = fake.completions
    # a model of yours is not the harness's to route: the prompt is its instructions
    assert completion(sent)["messages"][0] == {"role": "system", "content": "You triage."}
    chats = [dict(s.attributes or {}) for s in spans.get_finished_spans() if "chat" in s.name]
    assert [c["trellis.prompt.version"] for c in chats] == [3]


async def test_a_stored_prompt_outside_a_run_is_refused_by_the_middleware(h: Harness) -> None:
    model = langchain_openai.ChatOpenAI(model="local/small", base_url=URL, api_key=SecretStr("vk"))
    graph = create_agent(model, middleware=[ModelHooks(prompt=stored("triage"))])
    with pytest.raises(ConfigurationError, match="pinned for a harness run"):
        await graph.ainvoke({"messages": [HumanMessage("hi")]})
