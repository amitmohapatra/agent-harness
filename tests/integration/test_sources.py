"""Prompts and skills from every source, the same way through every way of running an agent:
each adapter (Way 1, the adapters table of ``test_reliability``), ``ReAct`` and code that is
not wrapped (Way 2); pinned for a run, so a resumed run reads what it started with even after
the source changed; a name no source has, and Langfuse down, said plainly."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support import langfuse as lf
from tests.support.adapters import BUILDERS
from tests.support.models import ScriptedChat
from tests.support.planned import FINAL, PlannedChat
from trellis import Harness, ReAct, Runtime, Settings, skills, tool
from trellis.contracts import RunEvent, RunEventType, RunStatus
from trellis.harness import fresh, telemetry
from trellis.harness.evals import EvalCase, EvalServices, judge, llm_judge
from trellis.harness.prompts import Prompt, PromptSources
from trellis.harness.repository import TTL_SECONDS
from trellis.harness.skills import (
    LOAD_SKILL,
    READ_SKILL_FILE,
    SECTION,
    Skill,
    SkillSources,
    skills_dir,
)

TONE = Skill("tone", "How we write.", "Short sentences.", files={"words.md": "Use: refund."})
TRIAGE = "---\nversion: 2\ndescription: Triage tickets.\n---\nTriage for {{team}}.\n"
SQL = "---\nname: sql\ndescription: Reviews SQL.\nversion: 1.0.0\n---\nRead rules.md first.\n"


@pytest.fixture
def folders(tmp_path: Path) -> Path:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "triage.md").write_text(TRIAGE)
    (tmp_path / "skills" / "sql").mkdir(parents=True)
    (tmp_path / "skills" / "sql" / "SKILL.md").write_text(SQL)
    (tmp_path / "skills" / "sql" / "rules.md").write_text("No SELECT *.")
    return tmp_path


@pytest.fixture
async def h(folders: Path) -> AsyncIterator[Harness]:
    settings = Settings(prompts_dir=str(folders / "prompts"), skills_dir=str(folders / "skills"))
    skills_ = [TONE, skills_dir(folders / "skills")]
    async with Harness(config=settings, skills=skills_) as made:
        yield made


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


@tool(side_effects="read")
def ping() -> str:
    """Check the line."""
    return "pong"


@tool(side_effects="irreversible")
def close_ticket(ticket: str) -> str:
    """Close a ticket."""
    return f"closed {ticket}"


def customs(events: list[RunEvent], name: str) -> list[dict[str, Any]]:
    return [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and (e.data or {}).get("name") == name
    ]


def said(model: Any) -> str:
    """Everything a framework's model was told (the Claude CLI: what it was started with)."""
    if isinstance(model, Path):
        return json.dumps(json.loads(model.read_text()))
    return model.said()


# --------------------------------------------------------------------------- every adapter
@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_skills_of_every_source_through_every_adapter(
    h: Harness, framework: str, tmp_path: Path
) -> None:
    plan = [(LOAD_SKILL, {"name": "tone"}), (READ_SKILL_FILE, {"name": "sql", "path": "rules.md"})]
    seen: list[Any] = []
    target, tools = await BUILDERS[framework](h, [skills("sql", "tone")], tmp_path, plan, seen=seen)
    agent = h.wrap(target, id=f"skilled-{framework}", tools=tools)
    events = [e async for e in agent.stream("review my query", user="ada")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.outcome is not None
    assert finished.outcome.value == "success", finished
    assert str(finished.data["result"]).endswith("No SELECT *.")  # a folder skill's file
    started = [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert started == [LOAD_SKILL, READ_SKILL_FILE]  # through the bridge
    results = [e.data["output"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert "Short sentences." in str(results[0])  # a code skill's body
    assert customs(events, "skills") == [
        {"name": "skills", "versions": {"sql": "1.0.0", "tone": "1"}}
    ]
    for model in seen:  # one section, both sources, in what the model read
        told = said(model)
        assert SECTION.splitlines()[0] in told and "- sql: Reviews SQL." in told
        assert "- tone: How we write." in told


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_prompt_from_a_folder_is_any_frameworks_instructions(
    h: Harness, framework: str, tmp_path: Path
) -> None:
    seen: list[Any] = []
    plan = [("ping", {})]
    target: Any
    if framework == "react":  # the target names it: rendered into its instructions
        model = PlannedChat(plan)
        seen.append(model)
        prompted = ReAct(
            system="Be brief.", model=model, prompt="triage", prompt_vars={"team": "EU"}
        )
        target, tools = prompted, [ping]
    elif framework == "function":  # read inside the run: pinned in its journal

        async def read(question: str, agent: Runtime) -> str:
            return await h.prompt("triage", team="EU")

        target, tools = read, []
    else:  # the framework's own instructions, read once when it is built
        system = await h.prompt("triage", team="EU")
        target, tools = await BUILDERS[framework](
            h, [ping], tmp_path, plan, system=system, seen=seen
        )
    events = [
        e async for e in h.wrap(target, id=f"p-{framework}", tools=tools).stream("hi", user="u")
    ]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    for model in seen:
        assert "Triage for EU." in said(model)
    if framework in ("react", "function"):  # resolved inside the run: said there
        assert customs(events, "prompt") == [
            {
                "name": "prompt",
                "prompt": "triage",
                "version": "2",
                "source": f"prompts_dir({h.settings.prompts_dir})",
            }
        ]
    if framework == "react":
        system_message = seen[0].requests[0]["messages"][0]["content"]
        assert system_message.startswith("Triage for EU.\n\nBe brief.")
    if framework == "function":
        assert finished.data["result"] == "Triage for EU."


# --------------------------------------------------------------------------- replay
async def test_a_resumed_react_reads_the_prompt_it_started_with(
    h: Harness, folders: Path, spans: InMemorySpanExporter
) -> None:
    model = PlannedChat([("close_ticket", {"ticket": "T-1"})])
    agent = h.wrap(
        ReAct(system="", model=model, prompt="triage", prompt_vars={"team": "EU"}),
        id="triage",
        tools=[close_ticket],
    )
    paused = await agent.run("close T-1", user="ada")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    (folders / "prompts" / "triage.md").write_text("---\nversion: 3\n---\nNew rules for {{team}}.")
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == FINAL.replace("{last}", "closed T-1")
    systems = [r["messages"][0]["content"] for r in model.requests]
    assert systems == ["Triage for EU.", "Triage for EU."]  # the journal's text, not the file's
    chats = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    assert {(s.attributes or {})["trellis.prompt.version"] for s in chats} == {"2"}
    assert {(s.attributes or {})["trellis.prompt.source"] for s in chats} == {
        f"prompts_dir({folders / 'prompts'})"
    }
    await agent.run("hello", user="ada")  # a new run reads the folder now
    assert model.requests[-1]["messages"][0]["content"] == "New rules for EU."


async def test_a_resumed_run_keeps_its_skills_and_prompt_whatever_the_folders_hold(
    h: Harness, folders: Path
) -> None:
    seen: list[str] = []

    async def analyst(question: str, agent: Runtime) -> str:
        seen.append(await h.prompt("triage", team="EU"))
        seen.append(await agent.tools.call(LOAD_SKILL, name="sql"))
        await agent.ask("Go on?")
        seen.append(str(agent.context))
        return await agent.tools.call(READ_SKILL_FILE, name="sql", path="rules.md")

    agent = h.wrap(analyst, id="analyst", skills=["sql"])
    paused = await agent.run("review", user="ada")
    assert paused.interrupt is not None
    (folders / "prompts" / "triage.md").write_text("Changed.")
    (folders / "skills" / "sql" / "SKILL.md").write_text(
        "---\nname: sql\ndescription: Changed.\nversion: 2.0.0\n---\nNew body.\n"
    )
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="ada")
    assert seen[0] == seen[2] == "Triage for EU."  # the prompt, from the journal
    assert seen[1] == seen[3] and "Read rules.md first." in seen[3]  # the skill's old body
    assert "- sql: Reviews SQL." in seen[4]  # its old description, in the pushed context
    # the folder holds another version now: its file is refused, saying why
    assert "this run uses version 1.0.0" in str(done.answer) and "has 2.0.0 now" in str(done.answer)


# --------------------------------------------------------------------------- the order and failures
async def test_a_name_no_source_has_fails_the_run_naming_the_sources_tried(
    h: Harness, folders: Path
) -> None:
    agent = h.wrap(ReAct(system="s", model=ScriptedChat([]), prompt="ghost"), id="ghost")
    result = await agent.run("hi", user="ada")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == (
        f"no prompt 'ghost' in any source (prompts_dir({folders / 'prompts'}): it has no ghost.md)"
    )

    async def analyst(question: str, agent: Runtime) -> str:
        return str(agent.context)

    events = [
        e async for e in h.wrap(analyst, id="a", skills=["ghost", "sql"]).stream("r", user="u")
    ]
    [warning] = customs(events, "warning")
    assert warning["code"] == "skills_unavailable"
    assert warning["message"].startswith("skill ghost: no skill 'ghost' in any source (code: ")
    assert "it has no ghost/SKILL.md" in warning["message"]  # and the run goes on


async def test_a_prompt_given_in_code_and_a_chat_prompt_with_examples() -> None:
    examples = Prompt(
        "examples",
        [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Sky?"},
            {"role": "assistant", "content": "Blue."},
        ],
    )
    model = ScriptedChat(["Green."])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct(system="", model=model, prompt=examples), id="one-word")
        assert (await agent.run("Grass?", user="u")).answer == "Green."
        assert await h.prompt_messages(examples) == list(examples.text)  # type: ignore[arg-type]
    assert [m["content"] for m in model.requests[0]["messages"]] == [
        "Answer in one word.",
        "Sky?",
        "Blue.",
        "Grass?",
    ]


@respx.mock
async def test_langfuse_prompts_through_a_run_and_its_last_copy_while_it_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: clock[0])
    store = lf.LangfusePrompts(prompts={"triage": ["Langfuse triage for {{team}}."]})
    route = respx.get(url__startswith=lf.HOST + lf.PATH).mock(side_effect=store.handle)
    settings = Settings(
        langfuse_host=lf.HOST, langfuse_public_key=lf.PUBLIC, langfuse_secret_key=lf.SECRET
    )
    model = ScriptedChat(["one", "two", "three"])
    async with Harness(config=settings) as h:
        agent = h.wrap(
            ReAct(system="", model=model, prompt="triage", prompt_vars={"team": "EU"}), id="t"
        )
        assert (await agent.run("a", user="u")).answer == "one"
        clock[0] += TTL_SECONDS + 1
        route.mock(side_effect=httpx.ConnectError("down"))
        assert (await agent.run("b", user="u")).answer == "two"  # the last good copy
        failed = await h.wrap(ReAct(system="", model=model, prompt="other"), id="o").run(
            "c", user="u"
        )
    assert [r["messages"][0]["content"] for r in model.requests] == [
        "Langfuse triage for EU.",
        "Langfuse triage for EU.",
    ]
    assert failed.status is RunStatus.ERROR and failed.error is not None
    assert "never read before: ConnectError: down" in failed.error.message
    assert failed.error.retryable


async def test_the_judge_reads_a_prompt_from_any_source(h: Harness) -> None:
    model = ScriptedChat(['{"score": 1, "reasoning": "fine"}', '{"score": 0.5}'])
    services = EvalServices(judge_model=model, prompts=h.prompts)
    case = EvalCase(input="hi", output="Hello!", run_id="run_1")
    strict = Prompt("strict", "Be strict.")
    scores, failed = await judge(case, [llm_judge("Polite.", prompt=strict)], services=services)
    assert failed == {} and scores[0].value == 1
    assert model.requests[0]["messages"][0] == {"role": "system", "content": "Be strict."}
    bare = EvalServices(judge_model=model)  # no sources: the judge's gateway only, and none
    _, failed = await judge(case, [llm_judge("Polite.", prompt="strict")], services=bare)
    assert "no prompt source for 'strict'" in failed["llm_judge"]


# --------------------------------------------------------------------------- blocks and selection
async def test_sources_given_as_blocks_are_used_as_they_are(folders: Path) -> None:
    settings = Settings(prompts_dir=str(folders / "prompts"), skills_dir=str(folders / "skills"))
    async with Harness(config=settings) as from_env:
        assert from_env.prompts.labels == [f"prompts_dir({folders / 'prompts'})"]
        assert from_env.skills.labels == [f"skills_dir({folders / 'skills'})"]
    chain = PromptSources([Prompt("triage", "Given.")])
    async with Harness(config=settings, prompts=chain, skills=SkillSources([TONE])) as given:
        assert given.prompts is chain and given.evals.prompts is chain
        assert given.skills.labels == ["code"]  # the environment's folder is not asked
        assert await given.prompt("triage") == "Given."
    async with Harness(config=settings, prompts=[], skills=[]) as none:
        assert none.prompts.labels == [] and none.skills.labels == []


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_without_skills_no_source_is_asked_and_nothing_is_offered(
    h: Harness, framework: str, tmp_path: Path
) -> None:
    seen: list[Any] = []
    target, tools = await BUILDERS[framework](
        h, [skills("sql", "tone"), ping], tmp_path, [("ping", {})], seen=seen
    )
    agent = h.wrap(target, id=f"plain-{framework}", tools=tools, without={"skills"})
    events = [e async for e in agent.stream("review", user="ada")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert customs(events, "skills") == [] and customs(events, "warning") == []
    for model in seen:
        told = said(model)
        assert "## Skills" not in told and LOAD_SKILL not in told
