"""Code by default: an agent the harness wraps gets a code tool in a sandbox of the run's own
with nothing to write (``sandbox.code_for``) — ``ReAct`` as Deep Agents' ``execute`` on its
backend (``CodeBackend``), every other agent as ``sandbox_exec`` — wherever an isolated sandbox
can be made; none where none can (said once), ``without=["code"]``, under ``deny``, or for an
agent that brings its own. The sandbox is made at the run's first command, never before."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, CodeInterpreterTool
from langchain_core.messages import ToolMessage

from tests.support.chat_model import ScriptedChatModel
from tests.support.openai_model import ScriptedModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, Runtime, Settings, middleware, sandbox
from trellis.contracts import RunStatus
from trellis.harness import sandbox as sandbox_module
from trellis.harness.adapters.acp import kimi
from trellis.harness.sandbox import CODE, EXEC, HostSandbox, SandboxSource
from trellis.harness.sandbox.backend import NO_CODE, CodeBackend


@pytest.fixture
def boxes() -> FakeSandboxes:
    """The sandbox the harness gives an agent by itself, on a fake provider."""
    provider = FakeSandboxes()
    sandbox_module.use_default(sandbox(provider))
    return provider


def deny() -> Harness:
    return Harness(config=Settings(host_execution="deny"), memory=False)


def graph(model: Any, **given: Any) -> Any:
    """``create_agent`` with the harness's middleware: it takes the run's tools per call."""
    from langchain.agents import create_agent

    return create_agent(model, tools=[], middleware=middleware(), **given)


def offered(model: ScriptedChatModel, call: int = 0) -> dict[str, str]:
    """The tools a model call was offered, each with its description."""
    return model.described[call]


def results(model: ScriptedChatModel) -> list[str]:
    return [str(m.content) for m in model.seen[-1] if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- none can be made
def test_where_no_isolated_sandbox_can_be_made_nothing_is_added_said_once(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sandbox_module, "_DEFAULT", {})  # decided again, here
    monkeypatch.setattr(sandbox_module, "docker_answers", lambda: False)
    monkeypatch.setattr(HostSandbox, "cannot_isolate", lambda self: "no root for the layer")
    sandbox_module._said_once.cache_clear()

    async def work(input: str, agent: Runtime) -> str:
        return "done"

    with caplog.at_level(logging.WARNING, logger="trellis.sandbox"):
        made = [
            harness.wrap(graph(ScriptedChatModel(turns=["hi"])), id="graph"),
            harness.wrap(work, id="function"),
            harness.wrap(ReAct("s", ScriptedChatModel(turns=["hi"])), id="react"),
        ]
    assert [a.code for a in made] == [None, None, None]
    assert all(a._code_tools == [] for a in made)
    said = [r.getMessage() for r in caplog.records if "no root for the layer" in r.getMessage()]
    assert len(said) == 1 and "no code tool" in said[0]
    assert sandbox_module.default_sandbox() is None  # never an unconfined shell


def test_the_default_is_dockers_else_the_hosts_isolation_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox_module, "_DEFAULT", {})
    monkeypatch.setattr(sandbox_module, "docker_answers", lambda: True)
    found = sandbox_module.default_sandbox()
    assert isinstance(found, SandboxSource)
    assert isinstance(found.provider, sandbox_module.SharedDockerSandbox)
    assert sandbox_module.default_sandbox() is found  # decided once
    monkeypatch.setattr(sandbox_module, "_DEFAULT", {})
    monkeypatch.setattr(sandbox_module, "docker_answers", lambda: False)
    monkeypatch.setattr(HostSandbox, "cannot_isolate", lambda self: None)
    found = sandbox_module.default_sandbox()
    assert found is not None and isinstance(found.provider, HostSandbox)


# --------------------------------------------------------------------------- each family
async def test_react_runs_code_in_its_backend_made_at_the_first_command(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    model = ScriptedChatModel(turns=[("execute", {"command": "echo hi"}), "it printed hi"])
    agent = harness.wrap(ReAct("You analyse data.", model), id="analyst")
    assert agent.code is not None and agent.codes == "default" and agent._code_tools == []
    assert boxes.calls == [] and boxes.probes == []  # nothing made at wrap
    done = await agent.run("say hi in the shell", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert offered(model)["execute"] == CODE
    assert EXEC not in offered(model)  # one code tool: Deep Agents' own
    assert any("hi" in r for r in results(model))
    [made] = boxes.made()
    assert ("delete", made) in boxes.calls and boxes.probes == []


async def test_a_run_that_never_runs_code_makes_no_sandbox(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    model = ScriptedChatModel(turns=["hello"])
    done = await harness.wrap(ReAct("s", model), id="idle").run("hi", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert "execute" in offered(model)
    assert boxes.calls == [] and boxes.probes == [] and boxes.commands == []


async def test_react_on_deep_agents_runs_code_in_its_backend_too(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    helper = {"name": "helper", "description": "Looks things up.", "system_prompt": "Help."}
    model = ScriptedChatModel(turns=[("execute", {"command": "echo deep"}), "done"])
    done = await harness.wrap(ReAct("s", model, subagents=[helper]), id="deep").run(
        "go", user="ada"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    assert offered(model)["execute"] == CODE
    assert any("deep" in r for r in results(model)) and boxes.made()


async def test_a_middleware_graph_is_offered_sandbox_exec(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    model = ScriptedChatModel(turns=[(EXEC, {"command": "echo graph"}), "done"])
    agent = harness.wrap(graph(model), id="graph")
    assert agent.codes == "none" and agent.code is not None
    done = await agent.run("go", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert offered(model)[EXEC] == CODE
    assert any("graph" in r for r in results(model)) and boxes.made()


async def test_a_deep_agent_of_your_own_is_offered_sandbox_exec(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    from deepagents import create_deep_agent

    model = ScriptedChatModel(turns=[(EXEC, {"command": "echo own"}), "done"])
    built = create_deep_agent(model=model, tools=[], middleware=middleware())
    agent = harness.wrap(built, id="deep-own")
    assert agent.codes == "none"  # its execute runs nothing: its state's files
    done = await agent.run("go", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert offered(model)[EXEC] == CODE and "execute" not in offered(model)
    assert boxes.made()


async def test_a_graph_built_by_hand_binds_it_with_the_harness_tools(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    from langchain.agents import create_agent

    bound = await harness.harness_tools()
    assert EXEC in {t.name for t in bound}
    model = ScriptedChatModel(turns=[(EXEC, {"command": "echo hand"}), "done"])
    agent = harness.wrap(create_agent(model, tools=bound), id="by-hand")
    done = await agent.run("go", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert any("hand" in r for r in results(model)) and boxes.made()
    assert ("delete", boxes.made()[0]) in boxes.calls  # its run's end finds it


async def test_an_openai_agent_is_offered_sandbox_exec(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    model = ScriptedModel([(EXEC, {"command": "echo openai"}), "done"])
    agent = harness.wrap(Agent(name="analyst", model=model), id="openai")
    done = await agent.run("go", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert EXEC in model.tools[0]
    assert any("openai" in str(o) for o in model.outputs()) and boxes.made()


async def test_a_function_calls_it_by_name(harness: Harness, boxes: FakeSandboxes) -> None:
    async def work(input: str, agent: Runtime) -> str:
        ran = await agent.tools.call(EXEC, command="echo fn")
        return str(ran["stdout"])

    done = await harness.wrap(work, id="fn").run("go", user="ada")
    assert done.answer == "fn", done.error
    assert boxes.made()


def test_claude_code_and_kimi_keep_their_own_shells(
    harness: Harness, boxes: FakeSandboxes, tmp_path: Path
) -> None:
    from claude_agent_sdk import ClaudeAgentOptions

    claude = harness.wrap(ClaudeAgentOptions(cwd=str(tmp_path)), id="claude")
    kimi_agent = harness.wrap(kimi(sandbox=sandbox(FakeSandboxes())), id="kimi")
    assert (claude.code, claude.codes) == (None, "own")
    assert (kimi_agent.code, kimi_agent.codes) == (None, "own")


# --------------------------------------------------------------------------- opting out
async def test_a_backend_given_wins(harness: Harness, boxes: FakeSandboxes) -> None:
    theirs = FakeSandboxes()
    model = ScriptedChatModel(turns=[("execute", {"command": "echo theirs"}), "done"])
    agent = harness.wrap(ReAct("s", model, backend=sandbox(theirs)), id="theirs")
    assert agent.code is None and agent.codes == "own"
    done = await agent.run("go", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert theirs.made() and boxes.calls == []


async def test_a_sandbox_among_its_tools_wins(harness: Harness, boxes: FakeSandboxes) -> None:
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE

    theirs = FakeSandboxes()
    model = ScriptedChatModel(turns=[(EXEC, {"command": "echo theirs"}), "done"])
    built = graph(model)
    setattr(built, TOOLS_ATTRIBUTE, [sandbox(theirs).tools()])
    agent = harness.wrap(built, id="declared")
    assert agent.code is None
    done = await agent.run("go", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    assert theirs.made() and boxes.calls == []


async def test_a_sandbox_a_function_adds_wins(harness: Harness, boxes: FakeSandboxes) -> None:
    theirs = FakeSandboxes()

    async def work(input: str, agent: Runtime) -> str:
        await agent.tools.use(sandbox(theirs).tools())
        return str((await agent.tools.call(EXEC, command="echo theirs"))["stdout"])

    done = await harness.wrap(work, id="adds").run("go", user="ada")
    assert done.answer == "theirs", done.error
    assert theirs.made() and boxes.calls == []


def test_an_openai_agent_with_its_own_code_gets_none(
    harness: Harness, boxes: FakeSandboxes
) -> None:
    hosted = CodeInterpreterTool(
        tool_config={"type": "code_interpreter", "container": {"type": "auto"}}
    )
    agent = harness.wrap(Agent(name="a", model=ScriptedModel(["x"]), tools=[hosted]), id="ci")
    assert (agent.code, agent.codes) == (None, "own")


async def test_without_code_offers_none(harness: Harness, boxes: FakeSandboxes) -> None:
    model = ScriptedChatModel(turns=["no code"])
    agent = harness.wrap(ReAct("s", model), id="off", without=["code"])
    assert agent.code is None
    assert (await agent.run("hi", user="ada")).status is RunStatus.SUCCESS
    assert "execute" not in offered(model)
    plain = ScriptedChatModel(turns=["no code"])
    other = harness.wrap(graph(plain), id="off-graph", without=["code"])
    assert other.code is None and other._code_tools == []
    assert (await other.run("hi", user="ada")).status is RunStatus.SUCCESS
    assert EXEC not in offered(plain)


async def test_a_run_without_code_is_offered_none(harness: Harness, boxes: FakeSandboxes) -> None:
    model = ScriptedChatModel(turns=["no code", "no code"])
    agent = harness.wrap(ReAct("s", model), id="run-off")
    assert (await agent.run("hi", user="ada", without=["code"])).status is RunStatus.SUCCESS
    assert "execute" not in offered(model)
    plain = ScriptedChatModel(turns=["no code"])
    other = harness.wrap(graph(plain), id="run-off-graph")
    assert (await other.run("hi", user="ada", without=["code"])).status is RunStatus.SUCCESS
    assert EXEC not in offered(plain)


async def test_deny_adds_nothing(boxes: FakeSandboxes) -> None:
    h = deny()
    model = ScriptedChatModel(turns=["no code"])
    react = h.wrap(ReAct("s", model), id="react")
    plain = h.wrap(graph(ScriptedChatModel(turns=["x"])), id="graph")
    assert (react.code, plain.code, plain._code_tools) == (None, None, [])
    assert (await react.run("hi", user="ada")).status is RunStatus.SUCCESS
    assert "execute" not in offered(model)
    assert EXEC not in {t.name for t in await h.harness_tools()}
    assert boxes.calls == []


async def test_a_command_in_a_run_without_code_is_refused() -> None:
    assert (await CodeBackend().aexecute("echo hi")).output == NO_CODE  # outside a run too
