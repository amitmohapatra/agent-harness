"""``docker_sandbox()``: the run's sandbox as a Deep Agents backend — a command, a file written,
a file read, in the run's sandbox (made at its first call, deleted at its end), bounded by
``call_timeout`` and the run's time left; Deep Agents' ``execute`` governed and recorded; the
synchronous operations and a call outside a run refused."""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes, PausingSandboxes
from trellis import Harness, ReAct, Runtime, docker_sandbox, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolError
from trellis.harness.sandbox import SandboxSource
from trellis.harness.sandbox.backend import ASYNC_ONLY, SandboxBackend, _error_of


async def test_a_command_and_files_work_in_the_runs_sandbox_deleted_at_its_end(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    backend = docker_sandbox(provider, call_timeout=30)
    seen: dict[str, Any] = {}

    async def work(input: str, agent: Runtime) -> str:
        seen["id"] = backend.id
        ran = await backend.aexecute("echo hi")
        wrote = await backend.aupload_files([("notes.txt", b"kept"), ("ok.txt", b"x")])
        read = await backend.adownload_files(["notes.txt", "missing.txt"])
        seen["ran"], seen["wrote"], seen["read"] = ran, wrote, read
        assert backend.source in agent.added  # its end finds it
        return "done"

    result = await harness.wrap(work, id="boxed").run("go", user="u")
    assert result.answer == "done", result.error
    assert seen["id"].startswith("trellis-")
    assert seen["ran"].output == "hi" and seen["ran"].exit_code == 0
    assert [w.error for w in seen["wrote"]] == [None, None]
    found, missing = seen["read"]
    assert found.content == b"kept" and found.error is None
    assert missing.content is None and missing.error == "file_not_found"
    assert provider.made() and ("delete", provider.made()[0]) in provider.calls


async def test_a_write_that_fails_is_said_per_file(harness: Harness) -> None:
    backend = docker_sandbox(FakeSandboxes())

    async def work(input: str, agent: Runtime) -> str:
        box = await backend.source.opened(agent)

        async def refuse(path: str, data: bytes) -> None:
            raise ToolError(f"permission denied: {path}", source="tools")

        box.write = refuse  # type: ignore[method-assign]
        agent.sandbox = box
        [said] = await backend.aupload_files([("/etc/x", b"x")])
        return str(said.error)

    assert (await harness.wrap(work, id="refused").run("go", user="u")).answer == (
        "permission_denied"
    )


@pytest.mark.parametrize(
    ("said", "named"),
    [
        ("there is no file a in the sandbox", "file_not_found"),
        ("could not find it", "file_not_found"),
        ("no such path", "file_not_found"),
        ("a is a directory", "is_directory"),
        ("access denied", "permission_denied"),
        ("bad name", "invalid_path"),
    ],
)
def test_a_sandbox_error_is_named_as_deep_agents_does(said: str, named: str) -> None:
    assert _error_of(ToolError(said, source="tools")) == named


async def test_the_command_is_bounded_by_the_least_of_its_limits(harness: Harness) -> None:
    provider = FakeSandboxes()
    backend = docker_sandbox(provider, call_timeout=0.05)

    async def work(input: str, agent: Runtime) -> str:
        try:
            await backend.aexecute("sleep 5", timeout=600)
        except TimeoutError:
            return "killed"
        return "ran"

    assert (await harness.wrap(work, id="slow").run("go", user="u")).answer == "killed"
    assert provider.killed == ["sleep 5"]


async def test_outside_a_run_and_synchronously_it_refuses() -> None:
    backend = docker_sandbox(FakeSandboxes())
    assert backend.id == "trellis-sandbox"
    with pytest.raises(ConfigurationError, match="inside a harness run"):
        await backend.aexecute("echo hi")
    for call in (
        lambda: backend.execute("echo hi"),
        lambda: backend.upload_files([]),
        lambda: backend.download_files([]),
    ):
        with pytest.raises(ConfigurationError, match=re.escape(ASYNC_ONLY)):
            call()
    with pytest.raises(ConfigurationError, match="call_timeout"):
        SandboxBackend(backend.source, call_timeout=0)
    assert isinstance(backend.source, SandboxSource)


async def test_react_runs_its_commands_in_the_sandbox_governed_and_recorded(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    model = ScriptedChatModel(turns=[("execute", {"command": "echo 42"}), "42"])
    agent = harness.wrap(ReAct("s", model, backend=docker_sandbox(provider)), id="coder")
    events = [e async for e in agent.stream("what is 6*7?", user="u")]
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED][-1]
    assert finished.data.get("result") == "42"
    called = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [c["tool"] for c in called] == ["execute"]  # through the bridge: governed, recorded
    assert "42" in str(called[0]["output"])
    assert "execute" in set(model.requests[0]["tools"])  # the agent's from the start
    assert provider.commands == ["echo 42"]
    assert ("delete", provider.made()[0]) in provider.calls  # the run's end deleted it
    result = await agent.run("again", user="u")
    assert result.status is RunStatus.ERROR  # the script ran out: a fresh run, a fresh box


async def test_reacts_sandbox_is_paused_with_its_run_and_attached_by_the_resume(
    harness: Harness,
) -> None:
    @tool(side_effects="irreversible")
    def publish(text: str) -> str:
        """Publish a text.

        Args:
            text: What.
        """
        return f"published {text}"

    provider = PausingSandboxes()
    model = ScriptedChatModel(
        turns=[
            ("execute", {"command": "echo drafted"}),
            ("publish", {"text": "the draft"}),
            ("execute", {"command": "echo again"}),
            "published",
        ]
    )
    graph = ReAct("s", model, backend=docker_sandbox(provider))
    agent = arm(harness.wrap(graph, id="publisher"), [publish])
    paused = await agent.run("draft and publish", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    made = provider.made()[0]
    assert ("snapshot", made) in provider.calls and ("pause", made) in provider.calls
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "published", done.error
    assert ("attach", made) in provider.calls  # the same sandbox, not a new one
    assert provider.commands == ["echo drafted", "echo again"]
    assert provider.made() == [made] and ("delete", made) in provider.calls
