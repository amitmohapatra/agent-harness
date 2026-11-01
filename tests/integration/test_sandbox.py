"""The run's sandbox, whichever framework calls its tools: ``sandbox_exec``, ``sandbox_read`` and
``sandbox_write`` go through the bridge (governed, journaled, recorded, redacted, at most their
timeout); the sandbox is made at the run's first call, named after the run and recorded before
any call uses it; a later attempt — after a pause, after a crash — attaches to it, makes it again
from the snapshot of the last pause while that still holds, and never replaces it blindly; the
run's end deletes it, and the reaper deletes what a dead process left."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, cast

import pytest

import trellis
from tests.integration.test_progress import Crash, crash_once
from tests.support.adapters import BUILDERS
from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.models import ScriptedChat
from tests.support.planned import Call
from tests.support.sandbox import FakeSandboxes, PausingSandboxes
from trellis import Harness, ReAct, Runtime, Settings
from trellis.contracts import (
    AgentError,
    ConfigurationError,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStatus,
    ToolError,
)
from trellis.harness.governance import Governance, governed
from trellis.harness.governance.catalog import Rule
from trellis.harness.runs import LocalRuns
from trellis.harness.sandbox import (
    READ_BYTES,
    REAP_SECONDS,
    DockerSandbox,
    SandboxRef,
    SandboxSpec,
    ended,
    paused,
    reap,
    ref_of,
    sandbox,
)
from trellis.harness.sandbox.base import RUN, TENANT
from trellis.harness.sandbox.docker import DEFAULT_IMAGE

PLAN: Final[list[Call]] = [
    ("sandbox_write", {"path": "notes.txt", "content": "hello"}),
    ("sandbox_exec", {"command": "echo hi"}),
    ("sandbox_read", {"path": "notes.txt"}),
]
UNKNOWN: Final = "it may or may not have taken effect: check before calling it again"


def custom(events: list[RunEvent], name: str) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in e.data.items() if k != "name"}
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == name
    ]


def results(events: list[RunEvent]) -> list[tuple[str, Any]]:
    return [
        (e.data["tool"], e.data["output"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_every_adapter_works_in_its_runs_sandbox_governed_and_deleted_at_the_end(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    provider = FakeSandboxes()
    target, tools = await BUILDERS[framework](harness, [sandbox(provider).tools()], tmp_path, PLAN)
    agent = arm(harness.wrap(target, id=f"coder-{framework}"), tools)
    events = [e async for e in agent.stream("take notes", user="u")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert finished.data["result"] == "Done. hello"  # the model read the file
    assert results(events) == [
        ("sandbox_write", "wrote 5 bytes to notes.txt"),
        ("sandbox_exec", {"exit_code": 0, "stdout": "hi", "stderr": ""}),
        ("sandbox_read", "hello"),
    ]
    # a write and a command are announced; a read runs
    assert [n["tool"] for n in custom(events, "tool_notice")] == ["sandbox_write", "sandbox_exec"]
    name = f"trellis-{finished.run_id}"
    assert custom(events, "sandbox") == [{"action": "created", "sandbox": name}]
    assert provider.calls == [("create", name), ("delete", name)]
    assert provider.commands == ["echo hi"] and provider.boxes == {}


async def test_a_function_targets_own_sandbox_is_its_runs_and_deleted_at_the_end(
    harness: Harness,
) -> None:
    """A function target's code adds the sandbox (``agent.tools.use``): its run's, paused and
    deleted as a builder's would be."""
    provider = PausingSandboxes()
    box = sandbox(provider)

    async def work(input: str, agent: Runtime) -> Any:
        assert await agent.tools.use(box.tools()) == [
            "sandbox_exec",
            "sandbox_read",
            "sandbox_write",
        ]
        assert await agent.tools.use(box.tools()) == [
            "sandbox_exec",
            "sandbox_read",
            "sandbox_write",
        ]
        current = trellis.current()
        assert current is not None and len(current.added) == 1  # joined once, however often
        await agent.tools.call("sandbox_write", path="a.txt", content="hi")
        await agent.ask("Go on?")
        return await agent.tools.call("sandbox_read", path="a.txt")

    agent = harness.wrap(work, id="own-box")
    paused = await agent.run("x", user="u")
    assert paused.interrupt is not None
    name = f"trellis-{paused.run_id}"
    assert ("pause", name) in provider.calls  # snapshotted and paused with the run
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="y", reviewer="u")
    assert done.answer == "hi" and provider.calls[-1] == ("delete", name)


async def test_a_sandbox_is_given_as_its_tools_the_same_source_every_time(
    harness: Harness,
) -> None:
    box = sandbox(FakeSandboxes())
    assert box.tools() is box.tools() and box.tools().source is box

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.use(box)

    result = await harness.wrap(work, id="bare").run("x", user="u")
    assert result.error is not None and "sandbox().tools()" in result.error.message


async def test_the_first_calls_made_at_once_make_one_sandbox_named_and_labelled_after_the_run(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    spec = SandboxSpec(files={"a.txt": "A", "b.txt": b"B"})
    reads = [("sandbox_read", {"path": "a.txt"}), ("sandbox_read", {"path": "b.txt"})]
    model = ScriptedChat([reads, "read both"])
    agent = arm(
        harness.wrap(ReAct(system_prompt="s", model=model), id="reader"),
        [sandbox(provider, spec=spec).tools()],
    )
    result = await agent.run("read", user="u")
    assert result.answer == "read both"
    assert provider.made() == [f"trellis-{result.run_id}"]  # one, however many came at once
    told = [m["content"] for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert told == ["A", "B"]  # the files it was made with
    assert provider.boxes == {}


async def test_a_pause_snapshots_and_pauses_the_sandbox_and_the_resume_attaches_to_it(
    harness: Harness,
) -> None:
    provider = PausingSandboxes()

    async def work(input: str, agent: Runtime) -> list[Any]:
        first = await agent.tools.call("sandbox_exec", command="echo one")
        await agent.ask("Go on?")
        second = await agent.tools.call("sandbox_exec", command="echo two")
        return [first["stdout"], second["stdout"]]

    agent = arm(harness.wrap(work, id="worker"), [sandbox(provider).tools()])
    waiting = await agent.run("go", user="u")
    assert waiting.status is RunStatus.PAUSED and waiting.interrupt is not None
    name = f"trellis-{waiting.run_id}"
    record = await harness.runs.get(waiting.run_id, tenant="default")
    assert record is not None and record.checkpoint is not None
    assert record.checkpoint["sandbox"] == {
        "provider": "fake",
        "id": name,
        "labels": {RUN: waiting.run_id, TENANT: "default", "trellis.agent_id": "worker"},
        "snapshot": f"{name}@0",
    }
    assert provider.boxes[name].paused
    done = await agent.resume(waiting.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    assert done.answer == ["one", "two"]
    assert provider.commands == ["echo one", "echo two"]  # the first replayed, not run again
    assert provider.calls == [
        ("create", name),
        ("snapshot", name),
        ("pause", name),
        ("attach", name),
        ("delete", name),
    ]
    assert provider.snapshots == {}  # deleted with it


async def test_a_sandbox_lost_while_its_run_waited_is_made_again_from_its_snapshot(
    harness: Harness,
) -> None:
    provider = PausingSandboxes()

    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_write", path="a.txt", content="kept")
        await agent.ask("Go on?")
        return await agent.tools.call("sandbox_read", path="a.txt")

    agent = arm(harness.wrap(work, id="restorer"), [sandbox(provider).tools()])
    waiting = await agent.run("go", user="u")
    assert waiting.interrupt is not None
    provider.boxes.clear()  # the provider lost it (expired, a host restarted)
    done = await agent.resume(waiting.interrupt.interrupt_id, "answer", answer="y", reviewer="u")
    assert done.answer == "kept"
    name = f"trellis-{waiting.run_id}"
    assert [call for call, _ in provider.calls] == [
        "create",
        "snapshot",
        "pause",
        "attach",
        "restore",
        "delete",
    ]
    assert provider.calls[-1] == ("delete", name)


async def test_a_sandbox_lost_after_a_call_changed_it_is_lost_never_replaced(
    harness: Harness,
) -> None:
    provider = PausingSandboxes()
    crashes = [Crash()]

    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_write", path="a.txt", content="1")
        await agent.ask("Go on?")
        await agent.tools.call("sandbox_write", path="b.txt", content="2")  # past the snapshot
        if crashes:
            raise crashes.pop()
        return await agent.tools.call("sandbox_read", path="b.txt")

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = arm(harness.wrap(work, id="careful"), [sandbox(provider).tools()])
    handle = await agent.start("go", user="u")
    assert await harness.worker([agent]).run_once()
    paused_record = await handle.status()
    assert paused_record.awaiting is not None
    await agent.resume(paused_record.awaiting.interrupt_id, "answer", answer="yes", reviewer="u")
    await crash_once(store, agent, handle)
    provider.boxes.clear()
    assert await harness.worker([agent]).run_once()
    done = await handle.result(wait=5)
    name = f"trellis-{handle.id}"
    assert done.answer == f"sandbox_read failed: the sandbox {name} is gone"
    assert ("restore", name) not in provider.calls and provider.made() == [name]


async def test_a_crash_right_after_the_sandbox_was_made_adopts_it_in_the_next_attempt(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    provider.crash, provider.crash_after_create = Crash(), True

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_read", path="seed.txt")

    store = harness.runs
    assert isinstance(store, LocalRuns)
    seeded = sandbox(provider, spec=SandboxSpec(files={"seed.txt": "seeded"}))
    agent = arm(harness.wrap(work, id="adopter"), [seeded.tools()])
    handle = await agent.start("go", user="u")
    await crash_once(store, agent, handle)  # made, but not yet recorded
    assert (await handle.status()).checkpoint is None
    assert await harness.worker([agent]).run_once()
    done = await handle.result(wait=5)
    name = f"trellis-{handle.id}"
    assert done.answer == "seeded"
    assert provider.calls == [("create", name), ("adopt", name), ("delete", name)]


async def test_a_crash_replays_the_commands_that_ran_and_attaches_for_the_next(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    crashes = [Crash()]

    async def work(input: str, agent: Runtime) -> list[str]:
        said = [(await agent.tools.call("sandbox_exec", command="echo one"))["stdout"]]
        if crashes:
            raise crashes.pop()
        said.append((await agent.tools.call("sandbox_exec", command="echo two"))["stdout"])
        return said

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = arm(harness.wrap(work, id="replayer"), [sandbox(provider).tools()])
    handle = await agent.start("go", user="u")
    await crash_once(store, agent, handle)
    record = await handle.status()
    assert record.checkpoint is not None and record.checkpoint["sandbox"]["id"] == (
        f"trellis-{handle.id}"
    )
    assert await harness.worker([agent]).run_once()
    assert (await handle.result(wait=5)).answer == ["one", "two"]
    assert provider.commands == ["echo one", "echo two"]
    assert [call for call, _ in provider.calls] == ["create", "attach", "delete"]


async def test_a_command_running_when_its_worker_died_is_of_unknown_effect_never_run_again(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    provider.crash = Crash()

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="crash")

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = arm(harness.wrap(work, id="crasher"), [sandbox(provider).tools()])
    handle = await agent.start("go", user="u")
    await crash_once(store, agent, handle)
    assert await harness.worker([agent]).run_once()
    done = await handle.result(wait=5)
    assert done.answer == f"sandbox_exec was interrupted by a crash; {UNKNOWN}"
    assert provider.commands == ["crash"]


async def test_a_command_out_of_time_is_killed_and_its_effect_is_unknown(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="sleep 5")

    agent = arm(harness.wrap(work, id="slow"), [sandbox(provider, call_timeout=0.05).tools()])
    result = await agent.run("go", user="u")
    assert result.answer == f"sandbox_exec timed out after 0.05s; {UNKNOWN}"
    assert provider.killed == ["sleep 5"]


async def test_the_catalog_governs_the_sandbox_tools_like_any_other() -> None:
    governance = Governance(FakeCatalog({"sandbox_exec": Rule(risk="irreversible")}))
    provider = FakeSandboxes()

    async def work(input: str, agent: Runtime) -> Any:
        return (await agent.tools.call("sandbox_exec", command="echo approved"))["stdout"]

    async with with_governance(Harness(config=Settings()), governance) as h:
        agent = arm(h.wrap(work, id="governed"), [sandbox(provider).tools()])
        waiting = await agent.run("go", user="u")
        assert waiting.interrupt is not None and waiting.interrupt.tool_call is not None
        assert waiting.interrupt.tool_call.tool == "sandbox_exec" and provider.calls == []
        done = await agent.resume(waiting.interrupt.interrupt_id, "approve", reviewer="cfo")
    assert done.answer == "approved"


async def test_what_a_command_prints_reaches_the_stream_and_memory_as_it_is(
    memory_harness: Harness, memory_service: Any
) -> None:
    secret = "sk-live0123456789abcdef"
    provider = FakeSandboxes()
    keyed = sandbox(provider, spec=SandboxSpec(files={"key.txt": secret}))

    async def work(input: str, agent: Runtime) -> Any:
        return (await agent.tools.call("sandbox_exec", command="cat key.txt"))["stdout"]

    agent = arm(memory_harness.wrap(work, id="leaky"), [keyed.tools()])
    events = [e async for e in agent.stream("go", user="u")]
    assert events[-1].data["result"] == secret  # the model and the tool had it as it is
    [(_, shown)] = results(events)
    assert shown["stdout"] == secret  # an on_output hook masks what a user sees
    await memory_harness.writes.drain()
    [recorded] = memory_service.named("record_tool")
    assert recorded.body["output"]["stdout"] == secret


async def test_a_run_that_fails_times_out_or_is_cancelled_deletes_its_sandbox(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    started = asyncio.Event()

    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_exec", command="echo made")
        if input == "fail":
            raise ValueError("broken")
        started.set()
        await asyncio.sleep(5)

    agent = arm(harness.wrap(work, id="ender"), [sandbox(provider).tools()])
    failed = await agent.run("fail", user="u")
    late = await agent.run("wait", user="u", run_timeout=0.2)
    assert (failed.status, late.status) == (RunStatus.ERROR, RunStatus.TIMEOUT)
    started.clear()
    running = asyncio.create_task(agent.run("wait", user="u"))
    await started.wait()
    [run_id] = list(agent.running)
    await agent.cancel(run_id, reason="not needed")
    assert (await running).status is RunStatus.CANCELLED
    stream = agent.stream("wait", user="u")
    async for event in stream:
        if event.type is RunEventType.TOOL_CALL_RESULT:
            break
    await stream.aclose()  # the caller went away: the run ends CANCELLED
    await asyncio.sleep(0.05)
    asked = arm(harness.wrap(_asking, id="asker"), [sandbox(provider).tools()])
    for decided in ("cancel", "resume"):
        waiting = await asked.run("x", user="u")  # paused: no attempt ends it when cancelled
        assert waiting.interrupt is not None
        if decided == "cancel":
            await asked.cancel(waiting.run_id, reason="not needed")
        else:
            await asked.resume(waiting.interrupt.interrupt_id, "cancel", reviewer="u")
    # one a worker elsewhere runs is that worker's to stop (at its next heartbeat), and so is
    # its sandbox
    handle = await asked.start("x", user="u")
    assert await harness.runs.claim("w-elsewhere", [asked.id]) is not None
    assert (await asked.cancel(handle.id)).status is RunStatus.RUNNING
    made, deleted = provider.made(), [n for c, n in provider.calls if c == "delete"]
    assert len(made) == 6 and deleted == made and provider.boxes == {}


async def test_a_queued_run_agent_runs_again_goes_on_in_its_sandbox() -> None:
    provider = FakeSandboxes()
    busy = [ToolError("the service is busy", retryable=True)]

    async def work(input: str, agent: Runtime) -> Any:
        said = (await agent.tools.call("sandbox_exec", command="echo made"))["stdout"]
        if busy:
            raise busy.pop()
        return said

    async with Harness(config=Settings(), runs=Retrying()) as h:
        agent = arm(h.wrap(work, id="retried"), [sandbox(provider).tools()])
        handle = await agent.start("go", user="u")
        assert await h.worker([agent]).run_once()
        await h.writes.drain()  # the reaper too: the run is queued again, not ended
        assert (await handle.status()).status is RunStatus.QUEUED
        assert [call for call, _ in provider.calls] == ["create"]
        assert await h.worker([agent]).run_once()
        assert (await handle.result(wait=5)).answer == "made"
    assert [call for call, _ in provider.calls] == ["create", "delete"]
    assert provider.commands == ["echo made"]  # replayed in the next attempt


class Retrying(LocalRuns):
    """Queues a run that failed with an error that may pass again, as agent-runs does."""

    async def finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: AgentError | None = None,
        worker_id: str | None = None,
        **fields: Any,
    ) -> RunRecord:
        if error is not None and error.retryable and worker_id is not None:
            return await self.release(run_id, worker_id, tenant=fields.get("tenant"))
        return await super().finish(run_id, status, error=error, worker_id=worker_id, **fields)


async def test_the_reaper_deletes_what_ended_runs_left_and_nothing_else(
    harness: Harness,
) -> None:
    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="echo hi")

    finished = await arm(harness.wrap(work, id="done"), [sandbox(FakeSandboxes()).tools()]).run(
        "x", user="u"
    )
    provider = FakeSandboxes()
    waiting = await arm(harness.wrap(_asking, id="asking"), [sandbox(provider).tools()]).run(
        "x", user="u"
    )
    left = SandboxRef(provider="fake", id="left", labels={RUN: finished.run_id, TENANT: "default"})
    unknown = SandboxRef(provider="fake", id="unknown", labels={RUN: "run_gone", TENANT: "default"})
    other = SandboxRef(provider="fake", id="other", labels={RUN: finished.run_id, TENANT: "t2"})
    for ref in (left, unknown, other):
        await provider.create(ref, SandboxSpec())
    deleted = await reap(provider, harness.runs, tenant="default")
    assert [r.id for r in deleted] == ["left"]  # a paused run's, an unknown run's: left alone
    assert sorted(provider.boxes) == sorted(["unknown", "other", f"trellis-{waiting.run_id}"])
    # a process reaps in the background when it makes a sandbox: once, then every REAP_SECONDS
    again = FakeSandboxes()
    source = sandbox(again)
    late = arm(harness.wrap(work, id="late"), [source.tools()])
    for _ in range(2):
        await again.create(left, SandboxSpec())
        await late.run("x", user="u")
        await harness.writes.drain()
    assert sorted(again.boxes) == ["left"]  # reaped the first time, not the second
    source._reaped["default"] -= REAP_SECONDS
    await late.run("x", user="u")
    await harness.writes.drain()
    assert again.boxes == {}


async def _asking(input: str, agent: Runtime) -> Any:
    await agent.tools.call("sandbox_exec", command="echo hi")
    return await agent.ask("Go on?")


def test_without_a_provider_a_sandbox_is_dockers_of_the_image_given() -> None:
    chosen = sandbox(image="registry.test/img:1").provider_for()
    assert isinstance(chosen, DockerSandbox) and chosen.image == "registry.test/img:1"
    assert cast(DockerSandbox, sandbox().provider_for()).image == DEFAULT_IMAGE
    given = FakeSandboxes()
    assert sandbox(given).provider_for() is given
    with pytest.raises(ConfigurationError, match="sandbox_exec: a timeout is a number of seconds"):
        sandbox(call_timeout=0)


async def test_a_large_file_is_read_in_parts_and_a_sandbox_of_another_provider_is_lost(
    harness: Harness,
) -> None:
    provider = FakeSandboxes()
    source = sandbox(provider, spec=SandboxSpec(files={"big.txt": "x" * (READ_BYTES + 1)}))

    async def reads(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_read", path="big.txt")

    big = await arm(harness.wrap(reads, id="big"), [source.tools()]).run("x", user="u")
    assert big.answer == (
        f"sandbox_read failed: big.txt is {READ_BYTES + 1} bytes, more than the {READ_BYTES} "
        "read at once: read a part of it with sandbox_exec (head -c, tail -c, sed -n)"
    )
    agent = arm(harness.wrap(_asking, id="moved"), [source.tools()])
    waiting = await agent.run("x", user="u")
    assert waiting.interrupt is not None
    source.provider = Renamed()  # the deployment changed its provider while the run waited

    async def again(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="echo later")

    agent.target = again  # the same run continues in code that calls the sandbox again
    done = await agent.resume(waiting.interrupt.interrupt_id, "answer", answer="y", reviewer="u")
    assert done.answer == (
        f"sandbox_exec failed: the run's sandbox trellis-{waiting.run_id} is a fake sandbox, "
        "and its tools make renamed sandboxes now"
    )


class Renamed(FakeSandboxes):
    name = "renamed"


async def test_a_sandbox_that_cannot_be_paused_is_a_warning_and_one_no_source_holds_is_left(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    provider = Unpausable()
    plain = harness.wrap(_asking, id="plain")  # an agent without the sandbox's source

    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_exec", command="echo hi")
        await paused(dataclasses.replace(agent, agent=plain))  # nothing: not its source
        await ended(plain, agent.replay.journal.sandbox, agent.run_id)
        return await agent.ask("Go on?")

    agent = arm(harness.wrap(work, id="stubborn"), [sandbox(provider).tools()])
    with caplog.at_level(logging.INFO, logger="trellis.sandbox"):
        events = [e async for e in agent.stream("x", user="u")]
        run_id = events[-1].run_id
        name = f"trellis-{run_id}"
        [warning] = custom(events, "warning")
        assert warning == {
            "code": "sandbox",
            "message": f"the sandbox {name} was not snapshotted and paused: the daemon is gone",
        }
        assert f"the sandbox {name} of run {run_id} is left to the reaper" in caplog.text
        assert provider.boxes  # still there
        await agent.resume(events[-1].data["interrupt"]["interrupt_id"], "cancel", reviewer="u")
    assert f"the sandbox {name} of run {run_id} was deleted" in caplog.text
    assert provider.boxes == {}


class Unpausable(PausingSandboxes):
    async def pause(self, ref: SandboxRef) -> None:
        raise ConnectionError("the daemon is gone")


async def test_a_sandbox_whose_delete_fails_is_left_to_the_reaper(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    provider = Undeletable()

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="echo hi")

    agent = arm(harness.wrap(work, id="sticky"), [sandbox(provider).tools()])
    with caplog.at_level(logging.WARNING, logger="trellis.sandbox"):
        result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS
    assert f"the sandbox trellis-{result.run_id} of run {result.run_id} was not deleted" in (
        caplog.text
    )


class Undeletable(FakeSandboxes):
    async def delete(self, ref: SandboxRef) -> None:
        raise ConnectionError("the daemon is gone")


def test_a_run_id_no_provider_takes_as_a_name_is_named_by_its_digest() -> None:
    def runtime(run_id: str) -> Runtime:
        return cast(Runtime, SimpleNamespace(run_id=run_id, tenant="t", agent_id="a"))

    assert ref_of(runtime("run_1"), "fake").id == "trellis-run_1"
    for odd in ("task:42/7", "run_" + "x" * 60):
        name = ref_of(runtime(odd), "fake").id
        assert name.startswith("trellis-") and len(name) == len("trellis-") + 32
    assert ref_of(runtime("task:42/7"), "fake").labels == {
        RUN: "task:42/7",
        TENANT: "t",
        "trellis.agent_id": "a",
    }


async def test_way_2_a_provider_and_governed_without_a_harness() -> None:
    provider = FakeSandboxes()
    box = await provider.create(SandboxRef(provider="fake", id="mine"), SandboxSpec())

    async def sandbox_exec(command: str) -> dict[str, Any]:
        return (await box.exec(command, timeout=10, env={"TOKEN": "t"})).model_dump()

    run = governed(sandbox_exec, Governance(), on_ask=lambda decision: True)
    assert await run(command="echo hi") == {"exit_code": 0, "stdout": "hi TOKEN=t", "stderr": ""}
    await provider.delete(box.ref)
    assert provider.boxes == {}


MISSING: Final = "exits 1 ModuleNotFoundError: No module named 'polars'"


@pytest.mark.parametrize(
    ("spec", "command", "hint"),
    [
        (
            SandboxSpec(memory=256),
            "exits 137 ",
            "Killed (exit 137): the command went past the sandbox's memory limit (256 MiB), or "
            "was killed. Work on the data in parts (read it in chunks, keep only the columns "
            "needed) rather than running it again as it is.",
        ),
        (
            SandboxSpec(),
            "exits 1 MemoryError",
            "Out of memory: the command went past the sandbox's memory limit (the provider's). "
            "Work on the data in parts (read it in chunks, keep only the columns needed) rather "
            "than running it again as it is.",
        ),
        (
            SandboxSpec(),
            MISSING,
            "The module polars is not installed in the sandbox, and the sandbox has no network "
            "to install it: use the standard library (csv, json, statistics, sqlite3) or a "
            "module that is installed.",
        ),
        (
            SandboxSpec(network="open"),
            MISSING,
            "The module polars is not installed in the sandbox: install it (pip install), then "
            "run the command again.",
        ),
        (SandboxSpec(), "exits 2 usage: no such option", None),
        (SandboxSpec(), "echo fine", None),
    ],
)
async def test_a_failed_command_says_what_the_sandbox_knows_about_why(
    harness: Harness, spec: SandboxSpec, command: str, hint: str | None
) -> None:
    """Facts the command's own output lacks: the memory limit it went past, whether a missing
    module can be installed there."""

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command=command)

    agent = arm(harness.wrap(work, id="hints"), [sandbox(FakeSandboxes(), spec=spec).tools()])
    result = await agent.run("x", user="u")
    assert result.answer.get("hint") == hint, result.error
