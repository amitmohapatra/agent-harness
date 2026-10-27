"""Skill sources in harness runs, through each framework's own skills slot: ``ReAct`` and a Deep
Agent (Deep Agents' skills middleware, the skills a route of the backend — with a sandbox
too), Claude (a local plugin written per run), the OpenAI Agents SDK (its ``Skills``
capability) and an engine with none (``list_skills``/``load_skill``). Per run: the run's scope,
the versions pinned in its journal (a resume uses them after an edit), an edit from the next
run, a deleted skill and a source that is down failing soft, two tenants and two users apart."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.planned import PlannedModel
from tests.support.sandbox import FakeSandboxes
from tests.support.skills import SkillStore
from trellis import Harness, ReAct, Runtime, Settings, deep_agent, sandbox, skills, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.skills import SKILLS_ROOT, UNAVAILABLE, combine

SKILL = f"{SKILLS_ROOT}sql-review/SKILL.md"


def store_with(body: str = "Check every JOIN.", **kwargs: Any) -> SkillStore:
    store = SkillStore()
    store.put("sql-review", body, description="Reviews SQL.", **kwargs)
    return store


def read_skill(path: str = SKILL) -> ScriptedChatModel:
    """A model that reads ``path`` with Deep Agents' ``read_file``, then answers."""
    return ScriptedChatModel(turns=[("read_file", {"file_path": path, "limit": 1000}), "read"])


def tool_said(model: ScriptedChatModel) -> str:
    """What the model was told by the tool calls it made, all of it."""
    return "\n".join(
        str(m.content) for call in model.seen for m in call if type(m).__name__ == "ToolMessage"
    )


def events_of(events: list[Any], name: str) -> list[dict[str, Any]]:
    return [e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == name]


def warnings_of(events: list[Any]) -> list[str]:
    return [
        str(e.data.get("message"))
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("code") == UNAVAILABLE
    ]


# --------------------------------------------------------------------------- ReAct
async def test_react_reads_a_skill_through_deep_agents_own_skills_slot(harness: Harness) -> None:
    store = store_with()
    model = read_skill()
    agent = harness.wrap(ReAct("You review SQL.", model, skills=combine(store)), id="reviewer")
    events = [e async for e in agent.stream("review this", user="ada")]
    assert events[-1].data.get("result") == "read"
    said = tool_said(model)
    assert "Check every JOIN." in said and 'description: "Reviews SQL."' in said
    system = str(model.seen[0][0].content)
    assert "sql-review" in system and "Reviews SQL." in system  # Deep Agents' own listing
    assert [e.get("versions") for e in events_of(events, "skills")] == [{"sql-review": "t1"}]


async def test_an_edit_applies_from_the_next_run_and_a_resume_keeps_the_pinned_version(
    harness: Harness,
) -> None:
    @tool(side_effects="irreversible")
    def publish(text: str) -> str:
        """Publish a text."""
        return f"published {text}"

    store = store_with("Version one.")
    model = ScriptedChatModel(
        turns=[
            ("read_file", {"file_path": SKILL, "limit": 1000}),
            ("publish", {"text": "x"}),
            ("read_file", {"file_path": SKILL, "limit": 1000}),
            "done",
        ]
    )
    agent = arm(harness.wrap(ReAct("s", model, skills=combine(store)), id="pinned"), [publish])
    paused = await agent.run("go", user="ada")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    store.put("sql-review", "Version two.")  # edited while the run waits
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.answer == "done", done.error
    said = tool_said(model)
    assert said.count("Version one.") >= 2 and "Version two." not in said  # the run's version
    model.turns = [("read_file", {"file_path": SKILL, "limit": 1000}), "again"]
    model.seen.clear()
    assert (await agent.run("go", user="ada")).answer == "again"
    assert "Version two." in tool_said(model)  # the next run reads the edit


async def test_a_skill_deleted_mid_run_fails_soft_and_is_gone_from_the_next(
    harness: Harness,
) -> None:
    store = store_with()
    store.put("tone", "Be brief.", description="How we write.")
    model = ScriptedChatModel(
        turns=[("ls", {"path": SKILLS_ROOT}), ("read_file", {"file_path": SKILL}), "soft"]
    )

    src = combine(store)
    agent = harness.wrap(ReAct("s", model, skills=src), id="deleted")

    original = store.load

    async def load_after_delete(name: str, version: str, scope: Any) -> Any:
        store.kept.pop(("sql-review", "tenant", "default"), None)
        return await original(name, version, scope)

    store.load = load_after_delete  # type: ignore[method-assign]
    events = [e async for e in agent.stream("go", user="ada")]
    assert events[-1].data.get("result") == "soft"  # the run went on
    assert any("sql-review" in w and "cannot be read" in w for w in warnings_of(events))
    store.load = original  # type: ignore[method-assign]
    model.turns = [("ls", {"path": SKILLS_ROOT}), "listed"]
    model.seen.clear()
    assert (await agent.run("go", user="ada")).answer == "listed"
    listing = tool_said(model)
    assert "tone" in listing and "sql-review" not in listing


async def test_a_source_that_is_down_stands_at_its_last_listing_and_never_fails_a_run(
    harness: Harness,
) -> None:
    store = store_with()
    model = read_skill()
    agent = harness.wrap(ReAct("s", model, skills=combine(store)), id="down")
    assert (await agent.run("go", user="ada")).answer == "read"
    store.down = True
    model.turns, model.seen = [("ls", {"path": SKILLS_ROOT}), "still"], []
    events = [e async for e in agent.stream("go", user="ada")]
    assert events[-1].data.get("result") == "still"
    assert "sql-review" in tool_said(model)  # the last listing stands
    assert any("could not be read" in w and "last listing stands" in w for w in warnings_of(events))
    never = SkillStore("never")
    never.down = True
    fresh = harness.wrap(ReAct("s", ScriptedChatModel(turns=["ok"]), skills=never), id="never")
    assert (await fresh.run("go", user="ada")).answer == "ok"  # nothing listed, no failure


async def test_two_tenants_and_two_users_each_get_their_own(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.tenant = None  # a platform key: each run names its tenant
    harness = Harness(config=Settings(), memory=memory_service.client())
    store = SkillStore()
    store.put("sql-review", "Acme's rules.", owner="acme")
    store.put("sql-review", "Globex's rules.", owner="globex")
    store.put("sql-review", "Ada's own rules.", level="user", owner="ada")
    src = combine(store)
    seen: dict[str, str] = {}

    async def reader(name: str, tenant: str, user: str) -> None:
        model = read_skill()
        agent = harness.wrap(ReAct("s", model, skills=src), id=f"r-{name}")
        result = await agent.run("go", user=user, tenant=tenant)
        assert result.answer == "read", result.error
        seen[name] = tool_said(model)

    await asyncio.gather(
        reader("acme-bob", "acme", "bob"),
        reader("globex-bob", "globex", "bob"),
        reader("acme-ada", "acme", "ada"),
    )
    assert "Acme's rules." in seen["acme-bob"]
    assert "Globex's rules." in seen["globex-bob"] and "Acme" not in seen["globex-bob"]
    assert "Ada's own rules." in seen["acme-ada"]  # her override wins over the tenant's
    await harness.aclose()


async def test_react_with_a_sandbox_and_skills_works_in_both(harness: Harness) -> None:
    provider = FakeSandboxes()
    model = ScriptedChatModel(
        turns=[
            ("read_file", {"file_path": SKILL, "limit": 1000}),
            ("execute", {"command": "echo ran"}),
            "both",
        ]
    )
    target = ReAct("s", model, backend=sandbox(provider), skills=combine(store_with()))
    result = await harness.wrap(target, id="both").run("go", user="ada")
    assert result.answer == "both", result.error
    assert "Check every JOIN." in tool_said(model) and provider.commands == ["echo ran"]


# --------------------------------------------------------------------------- Deep Agents
async def test_a_deep_agent_of_your_own_takes_a_sandbox_and_skills_in_one_backend(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    model = ScriptedChatModel(
        turns=[
            ("read_file", {"file_path": SKILL, "limit": 1000}),
            ("execute", {"command": "echo ran"}),
            "both",
        ]
    )
    src = skills.combine(store_with())
    graph = create_deep_agent(model=model, **deep_agent(sandbox(provider), src))
    result = await harness.wrap(graph, id="deep").run("go", user="ada")
    assert result.answer == "both", result.error
    assert "Check every JOIN." in tool_said(model) and provider.commands == ["echo ran"]
    only = create_deep_agent(model=read_skill(), **src.deep_agent())
    assert (await harness.wrap(only, id="deep-skills").run("go", user="ada")).answer


# --------------------------------------------------------------------------- Claude
async def test_claude_reads_the_runs_skills_as_a_local_plugin_written_for_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    store = store_with(files={"rules.md": "No SELECT *."})
    src = combine(store)
    options, _ = await BUILDERS["claude_agent_sdk"](harness, [], tmp_path, [])
    from dataclasses import replace

    options = replace(options, **src.claude())
    record = Path(options.env["FAKE_CLAUDE_RECORD"])
    result = await harness.wrap(options, id="claude").run("go", user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    [(folder, files)] = json.loads(await asyncio.to_thread(record.read_text))["plugins"].items()
    assert json.loads(files[".claude-plugin/plugin.json"])["name"] == "trellis"
    assert "Check every JOIN." in files["skills/sql-review/SKILL.md"]
    assert files["skills/sql-review/rules.md"] == "No SELECT *."
    assert not await asyncio.to_thread(Path(folder).exists)  # removed when the query ended
    assert options.skills == "all"


# --------------------------------------------------------------------------- OpenAI
async def test_an_openai_sandbox_agent_loads_a_skill_into_its_sandbox(harness: Harness) -> None:
    from agents import RunConfig
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Capabilities
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    src = combine(store_with(files={"rules.md": "No SELECT *."}))
    plan = [
        ("load_skill", {"skill_name": "sql-review"}),
        ("exec_command", {"cmd": "cat .agents/sql-review/SKILL.md .agents/sql-review/rules.md"}),
    ]
    target = SandboxAgent(
        name="box",
        instructions="Work.",
        model=PlannedModel(plan),
        capabilities=[*Capabilities.default(), src.openai()],
    )
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    agent = harness.wrap(target, id="oai", framework_options={"run_config": config})
    result = await agent.run("review", user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    assert "Check every JOIN." in str(result.answer) and "No SELECT *." in str(result.answer)


# --------------------------------------------------------------------------- no skills slot
async def test_calls_at_once_pin_the_run_once(harness: Harness) -> None:
    class Slow(SkillStore):
        async def list(self, scope: Any) -> Any:
            await asyncio.sleep(0.01)  # the others ask meanwhile
            return await super().list(scope)

    store = Slow()
    store.put("sql-review", "Check every JOIN.", description="Reviews SQL.")
    src = combine(store)

    async def work(input: str, agent: Runtime) -> list[list[str]]:
        found = await asyncio.gather(*(src.pinned() for _ in range(3)))
        return [list(f) for f in found]

    events = [e async for e in harness.wrap(work, id="once").stream("go", user="ada")]
    assert events[-1].data.get("result") == [["sql-review"]] * 3
    assert len(events_of(events, "skills")) == 1 and store.lists == 1


async def test_an_engine_with_no_skills_slot_lists_and_loads_them_as_tools(
    harness: Harness,
) -> None:
    src = combine(store_with(files={"rules.md": "No SELECT *."}))

    async def work(input: str, agent: Runtime) -> list[str]:
        await agent.tools.use(src.tools())
        listed = await agent.tools.call("list_skills")
        loaded = await agent.tools.call("load_skill", name="sql-review")
        file = await agent.tools.call("load_skill", name="sql-review", path="rules.md")
        missing = await agent.tools.call("load_skill", name="nope")
        return [listed, loaded, file, missing]

    result = await harness.wrap(work, id="fn").run("go", user="ada")
    listed, loaded, file, missing = result.answer
    assert listed == "- sql-review: Reviews SQL."
    assert "Check every JOIN." in loaded and "- rules.md" in loaded
    assert file == "No SELECT *." and "no skill 'nope'" in missing


@pytest.mark.parametrize("framework", ["function", "langgraph", "openai_agents"])
async def test_the_skill_tools_work_where_the_harness_runs_the_tools(
    harness: Harness, tmp_path: Path, framework: str
) -> None:
    src = combine(store_with())
    plan = [("load_skill", {"name": "sql-review"})]
    target, tools = await BUILDERS[framework](harness, [src.tools()], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"tools-{framework}"), tools)
    events = [e async for e in agent.stream("go", user="ada")]
    outputs = [str(e.data["output"]) for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert any("Check every JOIN." in o for o in outputs), outputs
