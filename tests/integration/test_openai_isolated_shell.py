"""OpenAI Agents' shell tools on this machine run through the harness's isolation layer
(``tools.convert.openai_local.isolated``): a ``ShellTool`` with a local executor and a
``LocalShellTool`` — each command run in the run's workspace as its uid, its result what the
SDK's own executors return (a ``ShellResult``; the command's text), never by the tool's own
executor. Each test tries what a run's command must not reach, or checks a timeout, the
clean-up and a machine that cannot isolate. Run as root, or with a password-less sudo."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Any

import pytest
from agents import Agent, LocalShellTool, ShellTool
from openai.types.responses import ResponseFunctionShellToolCall
from openai.types.responses.response_function_shell_tool_call import Action
from openai.types.responses.response_output_item import LocalShellCall, LocalShellCallAction

from tests.integration.test_openai_agents_features import shell_call, shell_said
from tests.support.isolation import fresh_root, processes_of, root_helper
from tests.support.openai_model import Items, ScriptedModel
from trellis import Harness
from trellis.contracts import RunStatus
from trellis.harness.sandbox.base import RUN
from trellis.harness.sandbox.isolation import HostSandbox, use_host

SECRET = "sk-service-openai-0123456789"
called: list[str] = []


def own_executor(request: Any) -> str:
    called.append("the tool's own executor")
    return "ran on the host"


@pytest.fixture
def helper() -> tuple[str, ...]:
    return root_helper()


@pytest.fixture
def root(helper: tuple[str, ...]) -> Iterator[str]:
    yield from fresh_root(helper)


@pytest.fixture
def layer(helper: tuple[str, ...], root: str) -> Iterator[HostSandbox]:
    base = 1_700_000_000 + secrets.randbelow(100_000) * 1_000
    made = HostSandbox(root, helper=helper, uid_base=base, tenants=1)
    replaced = use_host(made)
    called.clear()
    try:
        yield made
    finally:
        use_host(replaced)


def outputs(model: ScriptedModel) -> list[dict[str, Any]]:
    return [
        part
        for item in model.inputs[-1]
        if isinstance(item, dict) and item.get("type") == "shell_call_output"
        for part in item["output"]
    ]


async def test_a_shell_tools_commands_run_in_the_runs_workspace_and_the_sdk_reads_its_result(
    harness: Harness, layer: HostSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRELLIS_TEST_SECRET", SECRET)
    model = ScriptedModel([Items(shell_call("id -u; pwd; env; echo err >&2; exit 4")), "done"])
    agent = harness.wrap(
        Agent(name="ops", model=model, tools=[ShellTool(executor=own_executor)]), id="ops"
    )
    result = await agent.run("go", user="u1")
    assert result.status is RunStatus.SUCCESS and called == []
    [said] = outputs(model)
    uid, home, *env = said["stdout"].splitlines()
    assert int(uid) >= layer.isolation.uid_base and home.startswith(layer.isolation.root)
    assert SECRET not in said["stdout"] and said["stderr"] == "err\n"
    assert said["outcome"] == {"type": "exit", "exit_code": 4}
    assert await layer.labelled({RUN: result.run_id}) == [] and processes_of(int(uid)) == []


async def test_a_command_past_its_time_is_killed_and_said_timed_out(
    harness: Harness, layer: HostSandbox
) -> None:
    call = ResponseFunctionShellToolCall(
        id="sh_t",
        call_id="call_t",
        status="completed",
        type="shell_call",
        action=Action(commands=["sleep 30"], max_output_length=None, timeout_ms=500),
    )
    model = ScriptedModel([Items(call), "done"])
    agent = harness.wrap(
        Agent(name="ops", model=model, tools=[ShellTool(executor=own_executor)]), id="t"
    )
    assert (await agent.run("go", user="u1")).status is RunStatus.SUCCESS
    [said] = outputs(model)
    assert said["outcome"]["type"] == "timeout" and called == []


async def test_a_local_shell_tools_command_runs_in_the_workspace_with_its_own_env(
    harness: Harness, layer: HostSandbox
) -> None:
    call = LocalShellCall(
        id="ls_1",
        call_id="call_ls",
        status="completed",
        type="local_shell_call",
        action=LocalShellCallAction(
            type="exec", command=["sh", "-c", "id -u; echo $GREETING"], env={"GREETING": "hi"}
        ),
    )
    model = ScriptedModel([Items(call), "done"])
    agent = harness.wrap(
        Agent(name="ops", model=model, tools=[LocalShellTool(executor=own_executor)]), id="ls"
    )
    assert (await agent.run("go", user="u1")).status is RunStatus.SUCCESS
    said = next(
        str(item["output"])
        for item in model.inputs[-1]
        if isinstance(item, dict) and item.get("type") == "local_shell_call_output"
    )
    uid, greeting = said.split()
    assert int(uid) >= layer.isolation.uid_base and greeting == "hi" and called == []


async def test_where_commands_cannot_be_isolated_a_shell_command_is_refused(
    harness: Harness, root: str
) -> None:
    replaced = use_host(HostSandbox(root, helper=("false",)))
    model = ScriptedModel([Items(shell_call("id -u")), "done"])
    try:
        agent = harness.wrap(
            Agent(name="ops", model=model, tools=[ShellTool(executor=own_executor)]), id="t"
        )
        await agent.run("go", user="u1")
    finally:
        use_host(replaced)
    [said] = shell_said(model)  # the SDK tells the model why its executor failed
    assert called == [] and "commands cannot be isolated on this machine" in said
