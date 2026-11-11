"""``sandbox()`` as a Deep Agents backend — ``ReAct(backend=sandbox())``,
``create_deep_agent(**sandbox().deep_agent())`` — a command, a file written, a file read, in the
run's sandbox (made at its first call, deleted at its end), bounded by ``call_timeout`` and the
run's time left; Deep Agents' ``execute`` governed and recorded; the synchronous operations and
a call outside a run refused."""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes, PausingSandboxes
from trellis import Harness, ReAct, Runtime, sandbox, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolError
from trellis.harness.sandbox import SandboxSource, SandboxSpec
from trellis.harness.sandbox.backend import ASYNC_ONLY, SandboxBackend, _error_of


def backend_of(*args: Any, **kwargs: Any) -> SandboxBackend:
    """``sandbox(...)`` as ``create_deep_agent`` takes it."""
    [(name, backend)] = sandbox(*args, **kwargs).deep_agent().items()
    assert name == "backend" and isinstance(backend, SandboxBackend)
    return backend


async def test_a_command_and_files_work_in_the_runs_sandbox_deleted_at_its_end(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    backend = backend_of(provider, call_timeout=30)
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
    backend = backend_of(FakeSandboxes())

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
    backend = backend_of(provider, call_timeout=0.05)

    async def work(input: str, agent: Runtime) -> str:
        try:
            await backend.aexecute("sleep 5", timeout=600)
        except TimeoutError:
            return "killed"
        return "ran"

    assert (await harness.wrap(work, id="slow").run("go", user="u")).answer == "killed"
    assert provider.killed == ["sleep 5"]


async def test_outside_a_run_and_synchronously_it_refuses() -> None:
    backend = backend_of(FakeSandboxes())
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
    assert backend.call_timeout == backend.source.call_timeout == 120


async def test_a_deep_agent_of_your_own_works_in_the_runs_sandbox(harness: Harness) -> None:
    from deepagents import create_deep_agent

    provider = FakeSandboxes()
    model = ScriptedChatModel(turns=[("execute", {"command": "echo 42"}), "42"])
    graph = create_deep_agent(model=model, **sandbox(provider, call_timeout=9).deep_agent())
    result = await harness.wrap(graph, id="own").run("what is 6*7?", user="u")
    assert result.answer == "42", result.error
    assert provider.commands == ["echo 42"]
    assert ("delete", provider.made()[0]) in provider.calls  # the run's end deleted it


async def test_react_runs_its_commands_in_the_sandbox_governed_and_recorded(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    model = ScriptedChatModel(turns=[("execute", {"command": "echo 42"}), "42"])
    agent = harness.wrap(ReAct("s", model, backend=sandbox(provider)), id="coder")
    events = [e async for e in agent.stream("what is 6*7?", user="u")]
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED][-1]
    assert finished.data.get("result") == "42"
    called = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [c["tool"] for c in called] == ["execute"]  # through the bridge: governed, recorded
    assert "42" in str(called[0]["output"])
    assert "execute" in set(model.requests[0]["tools"])  # the agent's from the start
    assert (
        "Files you write in outputs/ (its working directory's) are given to the user"
        in (model.described[0]["execute"])
    )
    assert model.described[0]["execute"].endswith(
        "Its environment: Python 3.12.7; installed: numpy 2.1.0, openpyxl 3.1.5. It has no "
        "network: nothing else can be installed."
    )
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
    graph = ReAct("s", model, backend=sandbox(provider))
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


async def test_a_result_the_run_keeps_is_a_file_in_its_sandbox_for_code(harness: Harness) -> None:
    """``ReAct(backend=sandbox())``: a result the harness keeps (one cut to fit the window)
    is written into the sandbox at the path the model was given, once per attempt, before a
    command runs there — code reads every row of it."""
    from trellis.harness.tools import results

    provider = FakeSandboxes()
    backend = backend_of(provider)
    seen: list[str] = []

    async def work(input: str, agent: Runtime) -> str:
        path = results.keep("sku,stock\nSKU-1,4")
        assert path is not None
        seen.append((await backend.aexecute(f"cat {path}")).output)
        seen.append((await backend.aexecute(f"cat {path}")).output)  # written once: still there
        return "done"

    assert (await harness.wrap(work, id="kept-in-box").run("go", user="u")).answer == "done"
    assert seen == ["sku,stock\nSKU-1,4"] * 2


async def test_execute_keeps_its_own_description_when_the_card_cannot_be_read(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    made = provider.create

    async def no_probe(ref: Any, spec: Any) -> Any:
        if "trellis.probe" in ref.labels:
            raise ToolError("the image is gone", source="tools")
        return await made(ref, spec)

    provider.create = no_probe  # type: ignore[method-assign]
    model = ScriptedChatModel(turns=[("execute", {"command": "echo 42"}), "42"])
    agent = harness.wrap(ReAct("s", model, backend=sandbox(provider)), id="coder")
    assert (await agent.run("6*7?", user="u")).answer == "42"
    assert "Its environment" not in model.described[0]["execute"]
    assert model.described[0]["execute"].endswith("are given to the user when you finish.")


async def test_executes_output_ends_with_what_the_sandbox_knows_about_why(
    harness: Harness,
) -> None:
    backend = backend_of(FakeSandboxes(), spec=SandboxSpec(memory=64))
    seen: dict[str, Any] = {}

    async def work(input: str, agent: Runtime) -> str:
        seen["oom"] = await backend.aexecute("exits 137 ")
        seen["ok"] = await backend.aexecute("echo hi")
        return "done"

    assert (await harness.wrap(work, id="hinted").run("go", user="u")).answer == "done"
    assert seen["oom"].exit_code == 137
    assert seen["oom"].output.startswith(
        "Killed (exit 137): the command went past the sandbox's memory limit (64 MiB)"
    )
    assert seen["ok"].output == "hi"


async def test_a_command_naming_outputs_is_told_what_the_user_gets(harness: Harness) -> None:
    """Only a file written directly in ``outputs/`` of the working directory is the user's. A
    command that names an ``outputs`` folder is told at once, with its result, what that folder
    holds and where it is: a model that copied its file to ``/outputs/`` at the root kept
    nothing and answered that the file was saved (the priced pilot's bf2-mixed-04). A command
    that names none, and leaves no file beside the folder, is told nothing."""
    provider = FakeSandboxes()
    model = ScriptedChatModel(
        turns=[
            ("execute", {"command": "cp /large_tool_results/abc /outputs/report.csv"}),
            ("execute", {"command": "echo 42"}),
            "saved",
        ]
    )
    agent = harness.wrap(ReAct("s", model, backend=sandbox(provider)), id="saver")
    assert (await agent.run("save the report", user="u")).answer == "saved"
    missed, plain = [str(m.content) for m in model.seen[-1] if m.type == "tool"]
    assert (
        "No file is in /workspace/outputs/ yet: only a file written directly there is given "
        "to the user." in missed
    ), missed
    assert "outputs" not in plain  # told only for the one naming it

    async def work(input: str, runtime: Runtime) -> Any:
        await runtime.tools.use(sandbox(provider).tools())
        first = await runtime.tools.call("sandbox_exec", command="cp a /outputs/a.csv")
        await runtime.tools.call("sandbox_write", path="outputs/report.csv", content="n\n1\n")
        return [first, await runtime.tools.call("sandbox_exec", command="cat outputs/report.csv")]

    done = await harness.wrap(work, id="saver-tools").run("go", user="u")
    assert done.status is RunStatus.SUCCESS, done.error
    first, then = done.answer
    assert first["outputs"].startswith("No file is in /workspace/outputs/ yet"), first
    assert then["outputs"] == "In /workspace/outputs/ for the user: report.csv (4 bytes).", then


async def test_a_file_written_outside_the_kept_folder_is_known_at_once_not_to_be_the_users(
    harness: Harness,
) -> None:
    """A file for the user written with ``write_file`` to an ``outputs`` that is not the kept
    folder (the root's: the live chart case), or left by a command beside the kept folder (a
    chart saved in the working directory), is said with that call's result, with the kept
    folder's whole path — once per file; a file the sandbox started with, or one the model
    was told of when it wrote it by its path, is not one a command left."""
    provider = FakeSandboxes()
    model = ScriptedChatModel(
        turns=[
            ("write_file", {"file_path": "/outputs/chart.svg", "content": "<svg/>"}),
            ("write_file", {"file_path": "/workspace/plot.py", "content": "pass"}),
            ("execute", {"command": "touch chart.png"}),
            ("execute", {"command": "echo 42"}),
            ("write_file", {"file_path": "/workspace/outputs/chart.svg", "content": "<svg/>"}),
            "saved",
        ]
    )
    made = sandbox(provider, spec=SandboxSpec(files={"given.csv": "n\n1\n"}))
    agent = harness.wrap(ReAct("s", model, backend=made), id="charter")
    assert (await agent.run("chart it", user="u")).answer == "saved"
    root, own, beside, plain, kept = [str(m.content) for m in model.seen[-1] if m.type == "tool"]
    assert own.endswith(  # whichever path: outside the kept folder is not the user's
        "Not in /workspace/outputs/, so not given to the user: /workspace/plot.py."
    ), own
    assert root.endswith(
        "Not in /workspace/outputs/, so not given to the user: /outputs/chart.svg."
    ), root
    assert "Not in /workspace/outputs/, so not given to the user: chart.png." in beside, beside
    assert "outputs" not in plain, plain  # said once, and never of the file it started with
    assert kept.endswith("In /workspace/outputs/ for the user: chart.svg (6 bytes)."), kept

    async def work(input: str, runtime: Runtime) -> Any:
        await runtime.tools.use(sandbox(provider).tools())
        wrote = await runtime.tools.call("sandbox_write", path="/outputs/a.svg", content="x")
        ran = await runtime.tools.call("sandbox_exec", command="touch b.png")
        return [wrote, ran, await runtime.tools.call("sandbox_exec", command="echo 1")]

    done = await harness.wrap(work, id="charter-tools").run("go", user="u")
    assert done.status is RunStatus.SUCCESS, done.error
    wrote, ran, then = done.answer
    assert wrote.endswith(
        "Not in /workspace/outputs/, so not given to the user: /outputs/a.svg."
    ), wrote
    assert ran["outputs"] == "Not in /workspace/outputs/, so not given to the user: b.png.", ran
    assert "outputs" not in then, then


#: Every tool that can write a file in the run's sandbox, and the call that writes ``path``.
WRITERS = {
    "write_file": lambda path: ("write_file", {"file_path": path, "content": "x"}),
    "execute": lambda path: ("execute", {"command": f"touch {path}"}),
    "sandbox_write": lambda path: ("sandbox_write", {"path": path, "content": "x"}),
    "sandbox_exec": lambda path: ("sandbox_exec", {"command": f"touch {path}"}),
}
#: Where a file is written, and whether the user is given it: only directly in the kept folder.
PLACES = {
    "/workspace/outputs/chart.svg": True,
    "/outputs/chart.svg": False,  # another outputs: the root's
    "/workspace/chart.svg": False,  # beside the kept folder
}


@pytest.mark.parametrize("path", list(PLACES))
@pytest.mark.parametrize("writer", list(WRITERS))
async def test_whichever_tool_wrote_a_file_the_model_knows_whether_the_user_has_it(
    harness: Harness, writer: str, path: str
) -> None:
    """Every tool that writes a file in the sandbox (Deep Agents' ``write_file`` and
    ``execute`` on ``ReAct(backend=sandbox())``; ``sandbox_write`` and ``sandbox_exec`` of
    ``sandbox().tools()``), to a path inside the kept folder and to two outside it: its result
    says at once whether the user is given the file, with the kept folder's whole path."""
    provider, call = FakeSandboxes(), WRITERS[writer](path)
    if writer in ("write_file", "execute"):
        model = ScriptedChatModel(turns=[call, "done"])
        agent = harness.wrap(ReAct("s", model, backend=sandbox(provider)), id="writer")
        assert (await agent.run("write it", user="u")).answer == "done"
        said = str(model.seen[-1][-1].content)
    else:

        async def work(input: str, runtime: Runtime) -> Any:
            await runtime.tools.use(sandbox(provider).tools())
            return await runtime.tools.call(call[0], **call[1])

        done = await harness.wrap(work, id="writer-tools").run("go", user="u")
        assert done.status is RunStatus.SUCCESS, done.error
        said = str(done.answer)
    given = "In /workspace/outputs/ for the user: chart.svg" in said
    not_given = (
        "Not in /workspace/outputs/, so not given to the user" in said
        or "No file is in /workspace/outputs/ yet" in said
    )
    assert (given, not_given) == (PLACES[path], not PLACES[path]), said
