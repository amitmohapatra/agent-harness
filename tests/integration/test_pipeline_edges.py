"""The pipeline's less travelled paths, through the public API: a cancel that reaches the run,
a framework that swallows a pause, answers a run record cannot hold as JSON, Code Mode calls
read back from the gateway's log, a stream that is closed or breaks, and the calls that need
memory when the deployment has none."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.memory import FakeMemoryService
from trellis import Harness, Runtime, Settings
from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    RunEventType,
    RunRecord,
    RunStatus,
    ToolSpec,
)
from trellis.harness import telemetry
from trellis.harness.runs import LocalRuns
from trellis.harness.runtime import Paused
from trellis.harness.tools.base import Tool
from trellis.runs import ConflictError, LeaseLostError, RunsError


class CancelInTheRun(LocalRuns):
    """A run store that hands a cancel to the run instead of ending it itself."""

    async def resume(
        self, resolution: InterruptResolution, *, tenant: str | None = None
    ) -> RunRecord:
        if resolution.decision is not InterruptDecision.CANCEL:
            return await super().resume(resolution, tenant=tenant)
        record = self._require(resolution.run_id, tenant)
        return self._move(
            record,
            RunStatus.RUNNING,
            awaiting=None,
            last_resolution=resolution,
            attempt=record.attempt + 1,
        )


async def test_a_cancel_that_reaches_the_run_ends_it_cancelled() -> None:
    async def asks(input: str, agent: Runtime) -> str:
        return await agent.ask("Go on?")

    async with Harness(config=Settings()) as h:
        h.runs = CancelInTheRun()
        agent = h.wrap(asks, id="asks")
        paused = await agent.run("q", user="u")
        assert paused.interrupt is not None
        cancelled = await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="lee")
        assert cancelled.status is RunStatus.CANCELLED
        record = await h.runs.get(paused.run_id)
        assert record is not None and record.status is RunStatus.CANCELLED


async def test_a_framework_that_swallows_the_pause_is_still_paused_on_the_first_question(
    harness: Harness,
) -> None:
    async def swallows(input: str, agent: Runtime) -> str:
        with contextlib.suppress(Paused):  # a framework catching everything
            await agent.ask("First?")
        await agent.ask("Second?")
        return "never"

    result = await harness.wrap(swallows, id="swallows").run("q", user="u")
    assert result.status is RunStatus.PAUSED and result.interrupt is not None
    assert result.interrupt.question == "First?"


async def test_an_answer_that_is_not_json_is_kept_as_its_text(harness: Harness) -> None:
    async def odd(input: str, agent: Runtime) -> Any:
        return {1, 2}

    result = await harness.wrap(odd, id="odd").run("q", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == {1, 2}
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.output == "{1, 2}"


async def test_an_empty_answer_records_only_the_question(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def silent(input: str, agent: Runtime) -> str:
        return ""

    await memory_harness.wrap(silent, id="silent").run("anything?", user="u")
    await memory_harness.writes.drain()
    [batch] = memory_service.named("messages")
    assert [(m["role"], m["content"]) for m in batch.body["messages"]] == [("USER", "anything?")]


# --------------------------------------------------------------------------- Code Mode


class Script:
    """A Code Mode meta-tool: what a script run through the gateway looks like to the bridge."""

    async def resolve(self) -> list[Tool]:
        spec = ToolSpec(name="executeToolCode", description="run a script", side_effects="read")

        async def run(args: dict[str, Any]) -> Any:
            return "printed"

        return [Tool(spec, run, code_mode=True)]


class LoggingGateway:
    """The gateway's MCP log, holding the nested calls one script made."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def tools(self) -> list[Any]:
        return []

    async def code_mode_calls(self, run_id: str, since: Any) -> list[Any]:
        self.asked.append(run_id)
        return [
            SimpleNamespace(
                name="wiki-search", arguments={"q": "x"}, result="found", error=None, latency_ms=4
            ),
            SimpleNamespace(
                name="wiki-read", arguments="not an object", result=None, error="boom", latency_ms=1
            ),
        ]

    async def aclose(self) -> None:
        return None


async def test_the_calls_a_script_made_are_recorded_from_the_gateways_log(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def scripted(input: str, agent: Runtime) -> str:
        return await agent.tools.call("executeToolCode", code="print(wiki.search(q='x'))")

    gateway = LoggingGateway()
    memory_harness.gateway = gateway  # type: ignore[assignment]
    result = await memory_harness.wrap(scripted, id="coder", tools=[Script()]).run("q", user="u")
    await memory_harness.writes.drain()
    assert result.answer == "printed" and gateway.asked == [result.run_id]
    recorded = {c.body["tool"]: c.body for c in memory_service.named("record_tool")}
    assert recorded["wiki-search"]["status"] == "ok"
    assert recorded["wiki-search"]["args"] == {"q": "x"}
    assert recorded["wiki-read"]["status"] == "error"
    assert recorded["wiki-read"]["args"] == {}
    assert recorded["wiki-read"]["error_class"] == "MCPToolError"


async def test_without_a_gateway_a_script_has_no_log_to_read(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def scripted(input: str, agent: Runtime) -> str:
        return await agent.tools.call("executeToolCode", code="print(1)")

    await memory_harness.wrap(scripted, id="coder", tools=[Script()]).run("q", user="u")
    await memory_harness.writes.drain()
    assert [c.body["tool"] for c in memory_service.named("record_tool")] == ["executeToolCode"]


# --------------------------------------------------------------------------- streams


async def test_closing_a_stream_early_cancels_the_run(harness: Harness) -> None:
    async def forever(input: str, agent: Runtime) -> str:
        await asyncio.Event().wait()
        return "never"

    stream = harness.wrap(forever, id="forever").stream("q", user="u")
    first = await anext(stream)
    assert first.type is RunEventType.RUN_STARTED
    await stream.aclose()
    await asyncio.sleep(0.01)  # the cancelled run settles
    record = await harness.runs.get(first.run_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def test_a_stream_whose_run_the_store_refuses_raises(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    async def refused(*args: Any, **kwargs: Any) -> Any:
        raise RunsError("agent-runs refused the finish")

    monkeypatch.setattr(harness.runs, "finish", refused)
    stream = harness.wrap(echo, id="echo").stream("q", user="u")
    with pytest.raises(RunsError, match="refused the finish"):
        [e async for e in stream]


# --------------------------------------------------------------------------- refusals


async def test_a_resume_names_a_run_of_this_agent(harness: Harness) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        return await agent.ask("Go on?")

    mine = harness.wrap(asks, id="mine")
    other = harness.wrap(asks, id="other")
    paused = await mine.run("q", user="u")
    assert paused.interrupt is not None
    with pytest.raises(ConfigurationError, match="no run run_missing of agent mine"):
        await mine.resume("run_missing.1.1", "answer", answer="x", reviewer="u")
    with pytest.raises(ConfigurationError, match="of agent other"):
        await other.resume(paused.interrupt.interrupt_id, "answer", answer="x", reviewer="u")


async def test_a_handle_on_a_run_that_does_not_exist_says_so(harness: Harness) -> None:
    from trellis import RunHandle

    async def echo(input: str, agent: Runtime) -> str:
        return input

    handle = RunHandle(harness.wrap(echo, id="echo"), "run_missing", tenant="default")
    with pytest.raises(ConfigurationError, match="no run run_missing"):
        await handle.status()


@pytest.mark.parametrize(
    ("use", "message"),
    [
        ("memory", "memory is off in this deployment"),
        ("hints", "tool hints come from the memory service"),
        ("unknown tool", "no tool 'nope' in this run"),
    ],
)
async def test_what_a_run_cannot_reach_fails_the_run_saying_why(
    harness: Harness, use: str, message: str
) -> None:
    async def reaches(input: str, agent: Runtime) -> Any:
        if use == "memory":
            return agent.memory
        if use == "hints":
            return await agent.tools.hints("anything")
        return await agent.tools.call("nope")

    result = await harness.wrap(reaches, id="reaches").run("q", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert message in result.error.message


async def test_documents_need_memory(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="memory is off"):
        await harness.add_document(b"text", user="u")


async def test_a_memory_tool_outside_a_run_is_refused(memory_harness: Harness) -> None:
    assert memory_harness.memory is not None
    tools = await memory_harness.memory_tools(memory_harness.memory.scoped("acme"))
    search = next(t for t in tools if t.name == "memory_search")
    with pytest.raises(ConfigurationError, match="memory_search needs a run with memory on"):
        await search.run({"query": "x"})


async def test_feedback_with_memory_off_is_only_a_score(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def echo(input: str, agent: Runtime) -> str:
        return input

    scored: list[tuple[Any, ...]] = []
    monkeypatch.setattr(telemetry, "score_span", lambda *args, **kw: scored.append((*args, kw)))
    result = await harness.wrap(echo, id="echo").run("q", user="u")
    assert await harness.feedback(result.run_id, "edit", correction={"fixed": True}) is None
    trace = telemetry.trace_hex(result.run_id)
    run = {"run_id": result.run_id}
    assert scored == [(trace, "feedback", 0.5, "{'fixed': True}", run)]


async def test_a_sampled_answer_with_no_checkable_claim_gets_no_score(
    memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness.clients.memory import Memory

    memory_service.claims = 0
    memory_service.unsupported = 0
    scored: list[Any] = []
    monkeypatch.setattr(telemetry, "score_span", lambda *args, **kw: scored.append(args))
    settings = Settings(memory_url="http://m", api_key="test", grounding_sample=1.0)
    async with Harness(config=settings) as h:
        h.memory = Memory("http://m", None, client=memory_service.client())

        async def fn(input: str, agent: Runtime) -> str:
            return "hello"

        await h.wrap(fn, id="judged").run("hi", user="u")
        await h.writes.drain()
    assert len(memory_service.named("verify")) == 1
    assert scored == []


async def test_the_toolbox_for_openai_agents_is_function_tools(harness: Harness) -> None:
    from agents import FunctionTool

    from trellis import tool

    @tool(side_effects="read")
    def lookup(sku: str) -> int:
        """Units of a SKU."""
        return 1

    [native] = await harness.tools(lookup, framework="openai_agents")
    assert isinstance(native, FunctionTool) and native.name == "lookup"
    assert harness.built_for([native]) == ([], None)  # only LangGraph tools name their toolbox


async def test_a_finish_already_recorded_is_not_a_failed_run(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The finish reached the store, its answer did not, and the store refuses the retried
    finish (an agent-runs that is not idempotent on endings): the run is read back, and
    since it already ended as written, the run is reported as it ended — not failed, not
    executed again."""

    async def echo(input: str, agent: Runtime) -> str:
        return input

    finish = harness.runs.finish

    async def recorded_then_refused(run_id: str, status: RunStatus, **kwargs: Any) -> RunRecord:
        await finish(run_id, status, **kwargs)
        raise ConflictError(f"run {run_id} already ended", status=409, code="CONFLICT")

    monkeypatch.setattr(harness.runs, "finish", recorded_then_refused)
    agent = harness.wrap(echo, id="echo")
    with caplog.at_level("INFO", logger="trellis.run"):
        result = await agent.run("q", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "q"
    assert "already recorded as SUCCESS" in caplog.text


async def test_a_pause_already_recorded_is_still_the_pause(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        return await agent.ask("Go on?")

    pause = harness.runs.pause

    async def recorded_then_refused(interrupt: Any, **kwargs: Any) -> RunRecord:
        await pause(interrupt, **kwargs)
        raise ConflictError("not RUNNING", status=409, code="CONFLICT")

    monkeypatch.setattr(harness.runs, "pause", recorded_then_refused)
    result = await harness.wrap(asks, id="asks").run("q", user="u")
    assert result.status is RunStatus.PAUSED and result.interrupt is not None


async def test_a_conflict_on_a_finish_that_did_not_happen_raises(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store refused the ending and the run is not what was written: that is a real
    refusal, raised."""

    async def echo(input: str, agent: Runtime) -> str:
        return input

    async def refused(run_id: str, status: RunStatus, **kwargs: Any) -> RunRecord:
        raise ConflictError("RUNNING cannot become SUCCESS", status=409, code="CONFLICT")

    monkeypatch.setattr(harness.runs, "finish", refused)
    with pytest.raises(ConflictError):
        await harness.wrap(echo, id="echo").run("q", user="u")

    async def gone(run_id: str, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(harness.runs, "get", gone)
    with pytest.raises(ConflictError):
        await harness.wrap(echo, id="echo2").run("q", user="u")


async def test_a_lost_lease_on_the_write_is_never_read_as_a_conflict(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LeaseLostError`` is not a ``ConflictError``: even when the run already is what was
    written, a write refused for a lost lease is raised, not reconciled by a read."""

    async def echo(input: str, agent: Runtime) -> str:
        return input

    finish = harness.runs.finish
    read: list[str] = []
    get = harness.runs.get

    async def recorded_then_lost(run_id: str, status: RunStatus, **kwargs: Any) -> RunRecord:
        await finish(run_id, status, **kwargs)
        raise LeaseLostError("w no longer holds the run", status=409, code="LEASE_LOST")

    async def reading(run_id: str, **kwargs: Any) -> RunRecord | None:
        read.append(run_id)
        return await get(run_id, **kwargs)

    monkeypatch.setattr(harness.runs, "finish", recorded_then_lost)
    monkeypatch.setattr(harness.runs, "get", reading)
    with pytest.raises(LeaseLostError):
        await harness.wrap(echo, id="echo").run("q", user="u")
    assert read == []


async def test_a_failed_run_keeps_the_retryability_its_error_says(harness: Harness) -> None:
    """The harness hands the exception to AgentError.of as it is (no retryable= of its own),
    so an SDK error that says it may pass — or may not — is recorded so (contracts X6)."""
    from trellis.memory.errors import DependencyUnavailableError, ValidationError

    async def flaky(input: str, agent: Runtime) -> str:
        if input == "down":
            raise DependencyUnavailableError("memory is restarting", retryable=True)
        raise ValidationError("bad request", retryable=False)

    agent = harness.wrap(flaky, id="flaky")
    down = await agent.run("down", user="u")
    assert down.status is RunStatus.ERROR and down.error is not None
    assert down.error.retryable is True and down.error.source == "function"
    refused = await agent.run("bad", user="u")
    assert refused.error is not None and refused.error.retryable is False
    record = await harness.runs.get(down.run_id)
    assert record is not None and record.error is not None and record.error.retryable is True
