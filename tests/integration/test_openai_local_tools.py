"""The OpenAI Agents SDK's tools that run on this machine, each call through the bridge
(``tools.convert.openai_local``): the cases the feature tests leave — an edited local shell
command, a shell's structured result, an approved custom call, a patch's other operations, a
computer made per run, an edited or rejected computer action — and the seam itself
(``through``) driven by a function target: a call replayed from the journal on a resume, a
pause inside the code it runs, a call outside a run."""

from __future__ import annotations

from typing import Any

import pytest
from agents import (
    Agent,
    ApplyPatchTool,
    ComputerProvider,
    ComputerTool,
    CustomTool,
    LocalShellTool,
    ShellCommandOutput,
    ShellResult,
    ShellTool,
)
from openai.types.responses import (
    ResponseApplyPatchToolCall,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
)
from openai.types.responses.response_output_item import LocalShellCall, LocalShellCallAction

from tests.integration.test_openai_agents_features import shell_call, shell_said, twice
from tests.support.computer import Screen
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import Items, ScriptedModel
from trellis import Harness, Runtime
from trellis.contracts import RunStatus
from trellis.harness.tools import bridge
from trellis.harness.tools.convert import openai_local
from trellis.harness.tools.convert.openai_local import call_args, through

done: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


def recorded(memory_service: FakeMemoryService) -> list[tuple[str, str]]:
    return [(c.body["tool"], c.body["status"]) for c in memory_service.named("record_tool")]


async def test_an_edited_local_shell_command_runs_as_edited(harness: Harness) -> None:
    def execute(request: Any) -> str:
        done.append(" ".join(request.data.action.command))
        return "ok"

    call = LocalShellCall(
        id="ls_1",
        call_id="call_ls",
        status="completed",
        type="local_shell_call",
        action=LocalShellCallAction(type="exec", command=["rm", "-rf", "/"], env={}),
    )
    model = ScriptedModel([*twice(call), "done"])
    agent = harness.wrap(
        Agent(name="ops", model=model, tools=[LocalShellTool(executor=execute)]), id="edit"
    )
    paused = await agent.run("clean", user="u")
    assert paused.interrupt is not None
    edited = {"action": {"type": "exec", "command": ["rm", "-rf", "build"], "env": {}}}
    result = await agent.resume(paused.interrupt.interrupt_id, "edit", answer=edited, reviewer="r")
    assert result.status is RunStatus.SUCCESS and done == ["rm -rf build"]


async def test_a_shells_structured_result_is_the_sdks_and_its_record_is_its_data(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def structured(request: Any) -> ShellResult:
        return ShellResult(output=[ShellCommandOutput(stdout="a.txt", command="ls")])

    memory_service.catalog = {"shell": {"side_effects": "read"}}
    model = ScriptedModel([Items(shell_call("ls")), "listed"])
    agent = memory_harness.wrap(
        Agent(name="ops", model=model, tools=[ShellTool(executor=structured)]), id="structured"
    )
    assert (await agent.run("list", user="u")).status is RunStatus.SUCCESS
    assert shell_said(model) == ["a.txt"]  # as the SDK renders its own result
    await memory_harness.writes.drain()
    [record] = memory_service.named("record_tool")
    assert record.body["status"] == "ok" and "a.txt" in str(record.body["output"])


async def test_an_approved_custom_call_runs_once_with_its_input(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    class Rows:
        def __str__(self) -> str:
            return "3 rows"

    def invoke(context: Any, given: str) -> Any:
        done.append(given)
        return Rows()

    call = ResponseCustomToolCall(
        id="ct_1", call_id="call_ct", type="custom_tool_call", name="sql", input="SELECT 1"
    )
    model = ScriptedModel([*twice(call), "ran it"])
    tool = CustomTool(name="sql", description="Run SQL.", on_invoke_tool=invoke)
    agent = memory_harness.wrap(Agent(name="dba", model=model, tools=[tool]), id="custom")
    paused = await agent.run("select", user="u")
    assert paused.interrupt is not None and done == []
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert result.status is RunStatus.SUCCESS and done == ["SELECT 1"]
    await memory_harness.writes.drain()
    assert recorded(memory_service) == [("sql", "ok")]


async def test_a_patchs_create_and_update_operations_are_each_a_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    class Editor:
        def create_file(self, operation: Any) -> str:
            done.append(f"create {operation.path}")
            return "created"

        def update_file(self, operation: Any) -> str:
            done.append(f"update {operation.path}")
            return "updated"

        def delete_file(self, operation: Any) -> str:
            raise AssertionError("not asked to delete")

    def patch(call_id: str, operation: dict[str, Any]) -> Any:
        return ResponseApplyPatchToolCall.model_validate(
            {
                "id": f"ap_{call_id}",
                "call_id": call_id,
                "status": "completed",
                "type": "apply_patch_call",
                "operation": operation,
            }
        )

    memory_service.catalog = {"apply_patch": {"side_effects": "read"}}
    created = {"type": "create_file", "path": "new.py", "diff": "+x = 1\n"}
    updated = {"type": "update_file", "path": "old.py", "diff": "-y\n+z\n"}
    model = ScriptedModel([Items(patch("c1", created)), Items(patch("c2", updated)), "patched"])
    agent = memory_harness.wrap(
        Agent(name="dev", model=model, tools=[ApplyPatchTool(editor=Editor())]), id="patch"
    )
    assert (await agent.run("patch", user="u")).status is RunStatus.SUCCESS
    assert done == ["create new.py", "update old.py"]
    await memory_harness.writes.drain()
    assert recorded(memory_service) == [("apply_patch", "ok")] * 2


def click(call_id: str) -> Any:
    return ResponseComputerToolCall.model_validate(
        {
            "id": f"cu_{call_id}",
            "call_id": call_id,
            "status": "completed",
            "type": "computer_call",
            "pending_safety_checks": [],
            "action": {"type": "click", "x": 1, "y": 2, "button": "left"},
        }
    )


@pytest.mark.parametrize("made", ["provider", "per_run"])
async def test_a_computer_made_per_run_acts_through_the_bridge(
    memory_harness: Harness, memory_service: FakeMemoryService, made: str
) -> None:
    """A ``ComputerTool`` given a ``ComputerProvider``, or a function that makes its computer
    for each run: the computer it makes is the one whose actions are governed and recorded."""
    screen = Screen()

    async def create(**_: Any) -> Screen:
        return screen

    computer: Any = ComputerProvider(create=create) if made == "provider" else create
    memory_service.catalog = {"computer_use_preview": {"side_effects": "read"}}
    model = ScriptedModel([Items(click("c1")), "clicked"])
    agent = memory_harness.wrap(
        Agent(name="ui", model=model, tools=[ComputerTool(computer=computer)]), id=f"ui-{made}"
    )
    assert (await agent.run("click", user="u")).status is RunStatus.SUCCESS
    assert screen.done == ["click 1 2", "screenshot"]
    await memory_harness.writes.drain()
    assert len(recorded(memory_service)) == 2


@pytest.mark.parametrize("decision", ["edit", "reject"])
async def test_an_edited_computer_action_runs_as_edited_a_rejected_one_is_skipped(
    harness: Harness, decision: str
) -> None:
    screen = Screen()
    model = ScriptedModel([*twice(click("c1")), Items(click("c1")), "done"])
    agent = harness.wrap(
        Agent(name="ui", model=model, tools=[ComputerTool(computer=screen)]), id=f"ui-{decision}"
    )
    paused = await agent.run("click", user="u")
    assert paused.interrupt is not None and screen.done == []
    answer = {"action": {"type": "double_click", "x": 5, "y": 6}} if decision == "edit" else None
    then = await agent.resume(paused.interrupt.interrupt_id, decision, answer=answer, reviewer="r")
    # the screenshot after the action is a call of its own: asked about next
    assert then.interrupt is not None and then.interrupt.tool_call is not None
    assert then.interrupt.tool_call.args == {"action": {"type": "screenshot"}}
    then = await agent.resume(then.interrupt.interrupt_id, "approve", reviewer="r")
    assert then.status is RunStatus.SUCCESS
    assert screen.done == (
        ["double_click 5 6", "screenshot"] if decision == "edit" else ["screenshot"]
    )


# --------------------------------------------------------------------------- the seam itself
async def test_through_outside_a_run_runs_the_call() -> None:
    assert await through("shell", {"action": {}}, lambda args: "ran") == "ran"


async def test_a_call_made_before_a_pause_is_replayed_from_the_journal(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``through`` in a function target: the first call runs; the second asks; the resume
    runs the function again — the first call's result is the journal's, the second runs."""
    memory_service.catalog = {"look": {"side_effects": "read"}}

    async def work(input: str, agent: Runtime) -> str:
        first = await through("look", {"what": "a"}, lambda args: done.append("look") or "seen")
        second = await through("send", {"to": "b"}, lambda args: done.append("send") or "sent")
        return f"{first}, {second}"

    agent = memory_harness.wrap(work, id="seam")
    paused = await agent.run("go", user="u")
    assert paused.interrupt is not None and done == ["look"]
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert result.answer == "seen, sent" and done == ["look", "send"]


async def test_a_pause_inside_a_call_and_an_error_of_the_bridge_are_raised(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call whose code asks a person pauses the run (the pause kept for the SDK's run to be
    stopped by, once); what the bridge raises otherwise is raised as it is."""

    async def asks(input: str, agent: Runtime) -> str:
        async def run(args: dict[str, Any]) -> str:
            return str(await agent.ask("Which one?", options=["a", "b"]))

        said = await through("pick", {}, run)
        assert openai_local.paused_inside(agent) is None  # not paused again on the resume
        return said

    agent = harness.wrap(asks, id="asks")
    from tests.support.arm import declaring

    declaring(harness, pick="read")
    paused = await agent.run("go", user="u")
    assert paused.interrupt is not None
    result = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="a", reviewer="r")
    assert result.answer == "a"

    async def broken(*_: Any, **__: Any) -> Any:
        raise ValueError("the bridge broke")

    monkeypatch.setattr(bridge, "call", broken)

    async def calls(input: str, agent: Runtime) -> str:
        return str(await through("pick", {}, lambda args: "never"))

    failed = await harness.wrap(calls, id="broken").run("go", user="u")
    assert failed.status is RunStatus.ERROR and failed.error is not None
    assert "the bridge broke" in failed.error.message


def test_what_a_call_does_leaves_out_which_call_it_is() -> None:
    assert call_args({"type": "x", "id": "1", "call_id": "c", "status": "s", "a": None}) == {}
    assert call_args({"type": "x", "action": {"commands": ["ls"], "timeout_ms": None}}) == {
        "action": {"commands": ["ls"]}
    }
    assert call_args(shell_call("ls")) == {"action": {"commands": ["ls"]}}
