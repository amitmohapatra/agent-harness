"""The gateway's repositories and bundles through a wrapped agent: a stored prompt on every
model call of a ``ReAct`` (pinned for the run) and of the LLM judge, skills disclosed in the
context and read through two journaled tools, Virtual MCPs as the agent's MCP tools, and the
headers a framework's own model client is given."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import langchain_openai
import pytest
from agents import Agent as OpenAIAgent
from agents import OpenAIChatCompletionsModel
from claude_agent_sdk import ClaudeAgentOptions
from langchain.agents import create_agent
from openai import AsyncOpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from tests.support.adapters import CLI
from tests.support.gateway import URL, FakeGateway, SkillVersions
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, skills, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness import fresh, telemetry
from trellis.harness.clients import bifrost
from trellis.harness.evals import EvalCase, EvalServices, judge, llm_judge
from trellis.harness.middleware import ModelHooks
from trellis.harness.prompts import CUSTOM_HEADERS, ResolvedPrompt, selected_env
from trellis.harness.skills import LOAD_SKILL, READ_SKILL_FILE, SECTION

SYSTEM = [{"role": "system", "content": "You triage."}]
SQL = SkillVersions(
    {
        "1.0.0": ("Reviews SQL, the old way.", "Old rules.", {"rules.md": "old"}),
        "1.1.0": ("Reviews SQL.", "Read rules.md first.", {"rules.md": "No SELECT *."}),
    },
    served="1.1.0",
)
REFUNDS = SkillVersions({"2.0.0": ("Handles refunds.", "Refund within 30 days.", {})}, "2.0.0")


@pytest.fixture
def fake() -> FakeGateway:
    skills = copy.deepcopy({"sql": SQL, "refunds": REFUNDS})
    return FakeGateway(prompts={"triage": [SYSTEM, SYSTEM]}, skills=skills)


@pytest.fixture
async def h(fake: FakeGateway, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Harness]:
    fake.serve_models(monkeypatch)
    settings = Settings(bifrost_url=URL, bifrost_virtual_key="vk")
    async with Harness(config=settings, gateway=fake.gateway()) as made:
        yield made


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


@tool(side_effects="irreversible")
def close_ticket(ticket: str) -> str:
    """Close a ticket."""
    return f"closed {ticket}"


# --------------------------------------------------------------------------- prompts
async def test_every_model_call_selects_the_prompt_pinned_for_the_run(
    h: Harness, fake: FakeGateway, spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: clock[0])
    fake.chat.turns += [("close_ticket", {"ticket": "T-1"}), "closed it"]
    agent = h.wrap(
        ReAct(system="Tickets.", model="local/small", prompt="triage"),
        id="triage",
        tools=[close_ticket],
    )
    paused = await agent.run("close T-1", user="ada")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    fake.prompts["triage"].append(SYSTEM)  # a newer version, read after the TTL...
    clock[0] += bifrost.REPOSITORY_TTL_SECONDS + 1
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


def test_a_prompt_is_a_name_or_name_at_version() -> None:
    with pytest.raises(ConfigurationError, match="is not a name"):
        ReAct(system="s", model=ScriptedChat([]), prompt="triage@")
    with pytest.raises(ConfigurationError, match="is not a name"):
        llm_judge("Polite.", prompt="@1")
    with pytest.raises(ConfigurationError, match="fills the variables of a prompt"):
        ReAct(system="s", model=ScriptedChat([]), prompt_vars={"x": 1})


@pytest.mark.parametrize(
    ("prompt", "variables", "problem"),
    [
        ("triage", {"x": 1}, "no prompt_vars="),
        ("triage@latest", None, "a stored prompt's version is a number from 1, not 'latest'"),
    ],
)
async def test_a_stored_prompt_takes_no_vars_and_a_version_number(
    h: Harness, prompt: str, variables: dict[str, Any] | None, problem: str
) -> None:
    target = ReAct(system="s", model="local/small", prompt=prompt, prompt_vars=variables)
    result = await h.wrap(target, id="triage").run("hi", user="ada")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert problem in result.error.message


async def test_a_run_whose_prompt_is_not_there_fails_saying_so(h: Harness) -> None:
    agent = h.wrap(ReAct(system="s", model="local/small", prompt="nope"), id="nope")
    result = await agent.run("hi", user="ada")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "no committed prompt named 'nope'" in result.error.message


async def test_the_judge_selects_its_prompt(h: Harness, fake: FakeGateway) -> None:
    fake.chat.turns.append('{"score": 1, "reasoning": "polite"}')
    services = EvalServices(judge_gateway=h.gateway, judge_model="local/judge")
    case = EvalCase(input="hi", output="Hello!", run_id="run_1")
    scores, failed = await judge(case, [llm_judge("Polite.", prompt="triage@1")], services=services)
    assert failed == {} and scores[0].value == 1
    assert fake.completions[-1].headers["x-bf-prompt-version"] == "1"


async def test_a_judge_prompt_of_the_gateway_needs_a_model_name(h: Harness) -> None:
    services = EvalServices(judge_model=object(), judge_gateway=h.gateway)
    case = EvalCase(input="hi", output="Hello!", run_id="run_1")
    _, failed = await judge(case, [llm_judge("Polite.", prompt="triage")], services=services)
    assert "needs a Bifrost model name" in failed["llm_judge"]


async def test_a_frameworks_model_client_gets_the_deny_all_scope_and_the_prompt(
    h: Harness,
) -> None:
    assert await h.model_headers(prompt="triage@1") == {
        "x-bf-mcp-include-clients": "",
        "x-bf-mcp-include-tools": "",
        "x-bf-prompt-id": "p-triage",
        "x-bf-prompt-version": "1",
    }
    async with Harness(config=Settings()) as bare:
        assert await bare.model_headers() == {
            "x-bf-mcp-include-clients": "",
            "x-bf-mcp-include-tools": "",
        }
        with pytest.raises(ConfigurationError, match="BIFROST_URL"):
            await bare.model_headers(prompt="triage")


def newer(fake: FakeGateway, clock: list[float]) -> None:
    """A newer version of ``triage`` committed, and read once the kept one is stale."""
    fake.prompts["triage"].append(SYSTEM)
    clock[0] += bifrost.REPOSITORY_TTL_SECONDS + 1


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: now[0])
    return now


async def test_an_openai_client_selects_the_version_its_run_pinned_on_every_request(
    h: Harness, fake: FakeGateway, spans: InMemorySpanExporter, clock: list[float]
) -> None:
    headers = await h.model_headers(prompt="triage")  # version 2, the latest now
    client = AsyncOpenAI(
        base_url=URL,
        api_key="vk",
        default_headers=headers,  # kept, and read at each request
        http_client=cast(Any, httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))),
    )
    model = OpenAIChatCompletionsModel(model="local/small", openai_client=client)
    target = OpenAIAgent(name="triage", instructions="Tickets.", model=model)
    agent = h.wrap(target, id="triage", tools=[close_ticket])
    fake.chat.turns += [("close_ticket", {"ticket": "T-1"}), "closed it"]
    newer(fake, clock)  # 3: committed after the client was built, before the run
    events = [e async for e in agent.stream("close T-1", user="ada")]
    paused = await h.runs.get(events[-1].run_id)
    assert paused is not None
    assert paused.awaiting is not None
    newer(fake, clock)  # 4: committed while the run waits
    done = await agent.resume(paused.awaiting.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "closed it"
    # the run's start's version on every request, the resume's too
    assert [r.headers["x-bf-prompt-version"] for r in fake.completions] == ["3", "3"]
    assert {r.headers["x-bf-prompt-id"] for r in fake.completions} == {"p-triage"}
    assert events_of(events, "prompt") == [
        {"name": "prompt", "prompt": "triage", "version": "3", "source": "Bifrost"}
    ]
    runs = [dict(s.attributes or {}) for s in spans.get_finished_spans()]
    said = [a for a in runs if a.get("trellis.run_id") == done.run_id]
    assert {a.get("trellis.prompt.version") for a in said} >= {3}
    assert dict(headers)["x-bf-prompt-version"] == "2"  # outside a run: as resolved then
    assert len(headers) == 4 and "ModelHeaders('triage'" in repr(headers)
    await agent.run("hello", user="ada")  # a new run pins the newest
    assert fake.completions[-1].headers["x-bf-prompt-version"] == "4"


async def test_a_langchain_model_selects_it_through_the_harness_middleware(
    h: Harness, fake: FakeGateway, spans: InMemorySpanExporter, clock: list[float]
) -> None:
    headers = await h.model_headers(prompt="triage")
    model = langchain_openai.ChatOpenAI(
        model="local/small", base_url=URL, api_key=SecretStr("vk"), default_headers=headers
    )  # copied when built: version 2
    agent = h.wrap(create_agent(model, middleware=[ModelHooks()]), id="graph")
    fake.chat.turns.append("triaged")
    newer(fake, clock)
    assert (await agent.run("triage this", user="ada")).answer == "triaged"
    [sent] = fake.completions
    assert sent.headers["x-bf-prompt-version"] == "3"  # each call: the run's version
    chats = [dict(s.attributes or {}) for s in spans.get_finished_spans() if "chat" in s.name]
    assert [c["trellis.prompt.version"] for c in chats] == [3]


async def test_model_headers_made_in_a_run_or_by_another_harness(
    h: Harness, fake: FakeGateway, clock: list[float]
) -> None:
    async with Harness(config=Settings(), gateway=fake.gateway()) as other:
        theirs = await other.model_headers(prompt="triage")
        seen: list[dict[str, str]] = []

        async def triage(question: str, agent: Runtime) -> str:
            seen.append(dict(await h.model_headers(prompt="triage")))  # pinned now
            seen.append(dict(theirs))  # not this harness's: as resolved then
            return "ok"

        agent = h.wrap(triage, id="triage")
        newer(fake, clock)
        assert (await agent.run("x", user="ada")).answer == "ok"
        assert [s["x-bf-prompt-version"] for s in seen] == ["3", "2"]
        # handed out now: the next run pins it at its start; one it cannot pin is a warning
        del fake.prompts["triage"]
        clock[0] += bifrost.REPOSITORY_TTL_SECONDS + 1
        events = [e async for e in agent.stream("x", user="ada")]
        [warning] = events_of(events, "warning")
        assert warning["code"] == "prompt_unavailable" and "'triage'" in warning["message"]
        assert events_of(events, "prompt") == []


async def test_the_claude_cli_selects_the_version_its_run_pinned(
    h: Harness, fake: FakeGateway, clock: list[float], tmp_path: Path
) -> None:
    headers = await h.model_headers(prompt="triage")
    lines = "\n".join(f"{name}: {value}" for name, value in headers.items())
    record = tmp_path / "cli.json"
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps([{"text": "ok"}]),
        "FAKE_CLAUDE_RECORD": str(record),
        "ANTHROPIC_CUSTOM_HEADERS": f"x-team: eu\n{lines}",
    }
    agent = h.wrap(ClaudeAgentOptions(cli_path=CLI, env=env), id="claude")
    newer(fake, clock)
    assert (await agent.run("x", user="ada")).answer == "ok"
    sent = json.loads(record.read_text())["custom_headers"].splitlines()
    assert sent[0] == "x-team: eu" and sent[-2:] == [
        "x-bf-prompt-id: p-triage",
        "x-bf-prompt-version: 3",
    ]
    assert lines.endswith("x-bf-prompt-version: 2")  # the options' own: unchanged


def test_only_custom_headers_that_select_a_pinned_prompt_change() -> None:
    pinned = ResolvedPrompt("triage", "3", "Bifrost", selection="p-triage")
    runtime = cast(Runtime, SimpleNamespace(prompts={"k": pinned}))
    assert selected_env({}, runtime) is None
    assert selected_env({CUSTOM_HEADERS: "x-bf-prompt-id: p-other"}, runtime) is None
    assert selected_env({CUSTOM_HEADERS: "X-Bf-Prompt-Id: p-triage"}, runtime) == {
        CUSTOM_HEADERS: "x-bf-prompt-id: p-triage\nx-bf-prompt-version: 3"
    }


# --------------------------------------------------------------------------- skills
def events_of(found: list[Any], name: str) -> list[dict[str, Any]]:
    return [
        e.data
        for e in found
        if e.type is RunEventType.CUSTOM and (e.data or {}).get("name") == name
    ]


async def test_skills_are_disclosed_pinned_loaded_and_read_by_any_target(
    h: Harness, spans: InMemorySpanExporter
) -> None:
    seen: dict[str, Any] = {}

    async def analyst(question: str, agent: Runtime) -> str:
        seen["context"] = agent.context
        seen["loaded"] = await agent.tools.call(LOAD_SKILL, name="sql")
        seen["file"] = await agent.tools.call(READ_SKILL_FILE, name="sql", path="rules.md")
        return "reviewed"

    agent = h.wrap(analyst, id="analyst", skills=["sql", "refunds@2.0.0"])
    events = [e async for e in agent.stream("review this query", user="ada")]
    assert seen["context"] == "\n".join(
        [SECTION, "- sql: Reviews SQL.", "- refunds: Handles refunds."]
    )
    assert seen["loaded"] == (
        "# sql (version 1.1.0)\n\nRead rules.md first.\n\nFiles (read_skill_file):\n- rules.md"
    )
    assert seen["file"] == "No SELECT *."
    assert events_of(events, "skills") == [
        {"name": "skills", "versions": {"sql": "1.1.0", "refunds": "2.0.0"}}
    ]
    [root] = [s for s in spans.get_finished_spans() if s.name.startswith("invoke_agent")]
    assert (root.attributes or {})["trellis.skills"] == "sql@1.1.0,refunds@2.0.0"
    called = [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert called == [LOAD_SKILL, READ_SKILL_FILE]  # through the bridge, journaled


async def test_a_resumed_run_keeps_the_versions_it_started_with(
    h: Harness, fake: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: clock[0])
    loaded: list[str] = []

    async def analyst(question: str, agent: Runtime) -> str:
        loaded.append(await agent.tools.call(LOAD_SKILL, name="sql"))
        await agent.ask("Go on?")
        return await agent.tools.call(READ_SKILL_FILE, name="sql", path="rules.md")

    agent = h.wrap(analyst, id="analyst", skills=["sql"])
    paused = await agent.run("review", user="ada")
    assert paused.interrupt is not None
    fake.skills["sql"].served = "1.0.0"  # rolled back while the run waited
    clock[0] += bifrost.REPOSITORY_TTL_SECONDS + 1
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="ada")
    assert loaded[0] == loaded[-1]  # the body is the pinned version's, read from the journal
    # the gateway serves files of 1.0.0 now: 1.1.0's file is refused, saying why
    assert "uses version 1.1.0" in str(done.answer) and "serves (1.0.0)" in str(done.answer)


async def test_a_skill_the_gateway_cannot_give_is_a_warning_and_the_run_goes_on(
    h: Harness, fake: FakeGateway
) -> None:
    async def analyst(question: str, agent: Runtime) -> str:
        return f"{agent.context}|{await agent.tools.call(LOAD_SKILL, name='ghost')}"

    agent = h.wrap(analyst, id="analyst", skills=["ghost", "sql"])
    events = [e async for e in agent.stream("review", user="ada")]
    [warning] = [w for w in events_of(events, "warning") if w["code"] == "skills_unavailable"]
    assert "no skill named 'ghost'" in warning["message"]
    answer = next(e for e in events if e.type is RunEventType.RUN_FINISHED).data["result"]
    assert answer.startswith(SECTION) and "- sql: Reviews SQL." in answer
    assert "no skill 'ghost' in this run (its skills: sql)" in answer
    fake.down = ("/api/skills",)  # unreachable: a run that never read a skill has none
    async with Harness(config=Settings(bifrost_url=URL), gateway=fake.gateway()) as other:
        found = [
            e async for e in other.wrap(analyst, id="a2", skills=["sql"]).stream("r", user="u")
        ]
    assert [w["code"] for w in events_of(found, "warning")] == ["skills_unavailable"]
    assert "None|" in next(e for e in found if e.type is RunEventType.RUN_FINISHED).data["result"]


async def test_a_skill_file_must_be_one_the_skill_lists(h: Harness) -> None:
    async def analyst(question: str, agent: Runtime) -> str:
        return str(await agent.tools.call(READ_SKILL_FILE, name="sql", path="x.md"))

    result = await h.wrap(analyst, id="analyst", skills=["sql"]).run("r", user="u")
    assert result.status is RunStatus.SUCCESS
    assert "has no file 'x.md'" in str(result.answer)


async def test_a_file_of_a_skill_deleted_since_the_run_began_is_refused(
    h: Harness, fake: FakeGateway
) -> None:
    async def analyst(question: str, agent: Runtime) -> str:
        del fake.skills["sql"]
        return str(await agent.tools.call(READ_SKILL_FILE, name="sql", path="rules.md"))

    result = await h.wrap(analyst, id="analyst", skills=["sql"]).run("r", user="u")
    assert "serves files only of the version it serves (none)" in str(result.answer)


async def test_a_react_model_reads_the_skills_section_and_is_offered_the_tools(
    h: Harness, fake: FakeGateway
) -> None:
    fake.chat.turns += [(LOAD_SKILL, {"name": "refunds"}), "Refund within 30 days."]
    agent = h.wrap(ReAct(system="Support.", model="local/small"), id="support", skills=["refunds"])
    result = await agent.run("can I get a refund?", user="ada")
    assert result.answer == "Refund within 30 days."
    first = fake.chat.requests[0]
    # the pushed context (the skills section) is the system message after the instructions
    assert first["messages"][1]["content"].endswith("- refunds: Handles refunds.")
    offered = {t["function"]["name"]: t["function"] for t in first["tools"]}
    assert [n for n in offered if n in (LOAD_SKILL, READ_SKILL_FILE)] == [
        LOAD_SKILL,
        READ_SKILL_FILE,
    ]
    assert offered[LOAD_SKILL]["parameters"]["properties"]["name"]["enum"] == ["refunds"]


def test_skills_are_named_once_each() -> None:
    with pytest.raises(ConfigurationError, match="each skill once"):
        skills("sql", "sql@1.0.0")
    with pytest.raises(ConfigurationError, match="each skill once"):
        skills()


async def test_skills_need_the_gateway() -> None:
    async def analyst(question: str, agent: Runtime) -> str:
        return "x"

    async with Harness(config=Settings()) as bare:
        result = await bare.wrap(analyst, id="a", skills=["sql"]).run("r", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "BIFROST_URL" in result.error.message


async def test_skills_are_read_inside_a_run_only() -> None:
    [load, _] = await skills("sql").resolve()
    with pytest.raises(Exception, match="inside a Harness run"):
        await load.run({"name": "sql"})


# --------------------------------------------------------------------------- Virtual MCPs
async def test_an_agent_has_the_tools_of_its_virtual_mcps(h: Harness, fake: FakeGateway) -> None:
    fake.bundles = {"": ["erp-pay", "crm-get"], "finance": ["erp-pay"]}

    async def payer(question: str, agent: Runtime) -> Any:
        return [sorted(agent.toolbox), await agent.tools.call("erp-pay", amount=3)]

    names, paid = (await h.wrap(payer, id="payer", mcp=["finance"]).run("pay", user="u")).answer
    assert names == ["erp-pay"]
    assert paid == {"name": "erp-pay", "arguments": {"amount": 3}}
    assert fake.asked("/mcp/finance") == 2  # listed, then run, through the bundle
    everything = await h.wrap(payer, id="all").run("pay", user="u")
    assert everything.answer[0] == ["crm-get", "erp-pay"]  # no mcp=: the key's whole reach


def test_a_graph_takes_skills_and_virtual_mcps_through_h_tools(h: Harness) -> None:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    graph = create_agent(GenericFakeChatModel(messages=iter([])), tools=[])
    for given in ({"skills": ["sql"]}, {"mcp": ["finance"]}):
        with pytest.raises(ConfigurationError, match="skills as skills"):
            h.wrap(graph, id="g", **given)  # type: ignore[arg-type]


async def test_a_graphs_toolbox_is_its_h_tools_calls(h: Harness, fake: FakeGateway) -> None:
    fake.bundles = {"": ["erp-pay"], "finance": ["erp-pay"], "audit": ["log-read"]}
    finance = await h.tools(skills("sql"), framework="langgraph", mcp=["finance"])
    audit = await h.tools(framework="langgraph", mcp=["audit", "finance"])
    assert [t.name for t in finance] == [LOAD_SKILL, READ_SKILL_FILE, "erp-pay"]
    sources, mcp = h.built_for([*finance, *audit])
    assert mcp == ["finance", "audit"] and [type(s).__name__ for s in sources] == ["Skills"]
    whole = await h.tools(framework="langgraph")
    assert h.built_for([*finance, *whole])[1] is None  # one call reaches everything the key does
