"""``Agent``: a framework target with the harness attached — what ``Harness.wrap`` returns."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from trellis.contracts import (
    AgentEvalEvent,
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStart,
    RunStatus,
    Schedule,
    ScheduleSpec,
    ToolCall,
    ToolOutcome,
    ToolStatus,
    new_id,
    safe_id,
)
from trellis.harness import pipeline
from trellis.harness.adapters import detect
from trellis.harness.clients.memory import RunMemory
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal, content_key
from trellis.harness.result import Result
from trellis.harness.runtime import Runtime, run_of
from trellis.harness.telemetry import metrics
from trellis.harness.tools.base import Tool
from trellis.harness.tools.policy import Policy, Rule
from trellis.harness.tools.sources import as_source
from trellis.memory.models import ContextBundle

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

MemoryMode = Literal["off", "read", "read_write"]
MEMORY_MODES: Final = frozenset({"off", "read", "read_write"})
#: How long a resolved tool list is reused before its sources are listed again.
TOOLS_TTL_SECONDS: Final = 300.0
#: How often ``RunHandle.result`` looks at a queued run.
POLL_SECONDS: Final = 0.5


class Agent:
    """Run it, stream it, queue it, resume it, schedule it, serve it."""

    def __init__(
        self,
        harness: Harness,
        target: Any,
        *,
        id: str,
        tools: Sequence[Any] = (),
        memory: MemoryMode = "off",
        approve: Mapping[str, Rule] | None = None,
        tool_hints: bool = False,
    ) -> None:
        if memory not in MEMORY_MODES:
            raise ConfigurationError(
                f"memory must be one of {sorted(MEMORY_MODES)}, not {memory!r}"
            )
        if memory != "off" and harness.memory is None:
            raise ConfigurationError(f"agent {id!r} has memory={memory!r} but MEMORY_URL is unset")
        if tool_hints and memory == "off":
            raise ConfigurationError("tool_hints needs memory='read' or 'read_write'")
        self.harness = harness
        self.target = target
        self.id = safe_id(id)
        self.adapter = detect(target)
        if tools and self.adapter.fixed_tools:
            raise ConfigurationError(
                f"a {self.adapter.name} target binds its tools when it is built: pass "
                f"await h.tools(..., framework='langgraph') to the graph instead of tools="
            )
        self.sources = [as_source(t) for t in tools]
        self.memory_mode: MemoryMode = memory
        self.policy = Policy(approve)
        self.tool_hints = tool_hints
        self._tools: list[Tool] | None = None
        self._tools_at = 0.0

    # ------------------------------------------------------------------ running
    async def run(
        self, input: Any, *, user: str, thread: str | None = None, tenant: str | None = None
    ) -> Result:
        """Run to its end (or its first pause) and return how it ended."""
        identity = await self._opened(input, user=user, thread=thread, tenant=tenant)
        return await pipeline.attempt(self, identity, input)

    async def stream(
        self, input: Any, *, user: str, thread: str | None = None, tenant: str | None = None
    ) -> AsyncGenerator[RunEvent]:
        """The run's events as they happen, ending with ``RUN_FINISHED``. Closing the stream
        early cancels the run."""
        identity = await self._opened(input, user=user, thread=thread, tenant=tenant)
        async for event in self._events(
            lambda listen: pipeline.attempt(self, identity, input, listener=listen, streaming=True)
        ):
            yield event

    async def start(
        self, input: Any, *, user: str, thread: str | None = None, tenant: str | None = None
    ) -> RunHandle:
        """Queue the run for a worker (``h.worker([...]).run()``); it outlives this process."""
        try:
            json.dumps(input)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("a queued run's input must be JSON") from exc
        start = self._start(input, user=user, thread=thread, tenant=tenant)
        await self.harness.runs.queued(start)
        return RunHandle(self, start.run_id)

    async def resume(
        self,
        interrupt_id: str,
        decision: InterruptDecision | str,
        *,
        answer: Any = None,
        reviewer: str,
    ) -> Result:
        """Answer the interrupt a run is paused on. A run started in process continues here;
        a run that came from the queue goes back to it (``QUEUED``) and a worker continues it.
        ``answer`` is the answer to a question, or the edited arguments of an ``EDIT``."""
        record, resolution = await self._resolution(interrupt_id, decision, answer, reviewer)
        return await self._continue(record, resolution)

    async def schedule(
        self,
        cron: str,
        input: Any,
        *,
        on_behalf_of: str,
        tz: str = "UTC",
        tenant: str | None = None,
    ) -> Schedule:
        """Queue a run of this agent on ``cron`` (evaluated in ``tz``), acting for
        ``on_behalf_of``. Workers run them. Idempotent: the same agent, person, cadence and
        input are one schedule, so a redeploy updates it rather than adding another."""
        spec = ScheduleSpec(
            tenant_id=tenant or self.harness.settings.tenant,
            agent_id=self.id,
            name=schedule_name(self.id, on_behalf_of, cron, input),
            cadence=cron,
            timezone=tz,
            on_behalf_of=on_behalf_of,
            input=input,
        )
        return await self.harness.runs.schedule(spec)

    # ------------------------------------------------------------------ surfaces
    def serve_chat(self, app: Any, *, path: str = "/agui", identity: Any = None) -> None:
        """Mount the AG-UI routes for this agent on a FastAPI ``app``."""
        from trellis.harness.surfaces.agui import mount  # noqa: PLC0415 - optional extra

        mount(app, self, path=path, identity=identity)

    def serve_a2a(self, app: Any, url: str, *, identity: Any = None) -> None:
        """Publish this agent over A2A at ``url`` (its card and JSON-RPC routes on ``app``)."""
        from trellis.harness.surfaces.a2a import mount  # noqa: PLC0415 - optional extra

        mount(app, self, url=url, identity=identity)

    # ------------------------------------------------------------------ used by surfaces
    async def _resolution(
        self, interrupt_id: str, decision: InterruptDecision | str, answer: Any, reviewer: str
    ) -> tuple[RunRecord, InterruptResolution]:
        run_id = run_of(interrupt_id)
        record = await self.harness.runs.get(run_id)
        if record is None or record.agent_id != self.id:
            raise ConfigurationError(f"no run {run_id} of agent {self.id}")
        if record.status is not RunStatus.PAUSED or record.awaiting is None:
            raise ConfigurationError(f"run {run_id} is {record.status.value}, not paused")
        if record.awaiting.interrupt_id != interrupt_id:
            raise ConfigurationError(
                f"run {run_id} waits on {record.awaiting.interrupt_id}, not {interrupt_id}"
            )
        chosen = InterruptDecision(str(decision).upper())
        edited = chosen is InterruptDecision.EDIT
        resolution = InterruptResolution(
            interrupt_id=interrupt_id,
            run_id=run_id,
            decision=chosen,
            answer=None if edited else answer,
            payload=answer if edited else None,
            reviewer=reviewer,
        )
        return record, resolution

    async def _continue(
        self,
        record: RunRecord,
        resolution: InterruptResolution,
        listener: Callable[[RunEvent], None] | None = None,
    ) -> Result:
        runs = self.harness.runs
        assert record.awaiting is not None
        identity = self._identity_of(record)
        feedback = resolution.to_feedback(record.awaiting, identity.context())
        if feedback is not None and self.memory_mode == "read_write":
            run_memory = self.run_memory(identity)
            if run_memory is not None:
                self.harness.writes.submit("memory.feedback", lambda: run_memory.feedback(feedback))
        resumed = await runs.resumed(resolution)
        if resumed.status is not RunStatus.RUNNING:
            # cancelled, or back on the queue for a worker (a run that came from the queue)
            return Result(run_id=record.run_id, status=resumed.status)
        return await pipeline.attempt(
            self,
            identity,
            record.input,
            number=resumed.attempt,
            journal=Journal.of(record.checkpoint),
            resolution=resolution,
            listener=listener,
            streaming=listener is not None,
        )

    async def _claimed(self, record: RunRecord, worker_id: str) -> Result:
        """A worker's run: fresh from the queue, or continuing after a resolution."""
        return await pipeline.attempt(
            self,
            self._identity_of(record),
            record.input,
            number=record.attempt,
            journal=Journal.of(record.checkpoint),
            resolution=record.last_resolution,
            worker_id=worker_id,
        )

    async def _events(
        self, execute: Callable[[Callable[[RunEvent], None]], Any]
    ) -> AsyncIterator[RunEvent]:
        """Run ``execute(listener)`` in a task and yield what it emits up to its
        ``RUN_FINISHED``. Closing the iterator early cancels the task."""
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        task = asyncio.create_task(execute(queue.put_nowait))
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while (event := await queue.get()) is not None:
                yield event
                if event.type is RunEventType.RUN_FINISHED:
                    break
            await task  # surface an unexpected failure of the harness itself
        finally:
            if not task.done():
                task.cancel()

    # ------------------------------------------------------------------ used by the pipeline
    def run_memory(self, identity: Identity) -> RunMemory | None:
        memory = self.harness.memory
        if self.memory_mode == "off" or memory is None:
            return None
        self.harness.registered(memory, identity)
        return memory.bind(identity)

    async def tools_for(self, runtime: Runtime) -> list[Tool]:
        """The agent's own tools (resolved once per TTL) and the memory pull tools."""
        now = time.monotonic()
        if self._tools is None or now - self._tools_at > TOOLS_TTL_SECONDS:
            self._tools = await self.harness.resolve(self.sources, tenant=runtime.tenant)
            self._tools_at = now
        tools = list(self._tools)
        if runtime.run_memory is not None and not self.adapter.fixed_tools:
            tools.extend(
                await self.harness.memory_tools(runtime.run_memory, self.memory_mode == "read")
            )
        return tools

    async def push(self, runtime: Runtime, tool_names: list[str]) -> ContextBundle | None:
        """The memory context for this run, in the runtime; a failure is a warning."""
        if runtime.run_memory is None or not runtime.task:
            return None
        try:
            bundle = await runtime.run_memory.context(
                runtime.task, tools=tool_names if self.tool_hints else None
            )
        except Exception as exc:
            runtime.events.warning("memory_unavailable", f"no memory context: {exc}")
            return None
        text = bundle.rendered
        if self.adapter.keeps_conversation(self.target):
            text = without_conversation(bundle)
        runtime.context = text or None
        runtime.events.emit(RunEventType.CONTEXT_LOADED, data={"chars": len(text)})
        return bundle

    def record_tool(self, runtime: Runtime, call: ToolCall, outcome: ToolOutcome) -> None:
        memory = runtime.run_memory
        if memory is not None and self.memory_mode == "read_write" and call.tool not in _pull(self):
            self.harness.writes.submit(
                "memory.record_tool",
                lambda: memory.record_tool(call, outcome),
                events=runtime.events,
            )

    def recorded_run(self, runtime: Runtime, messages: Sequence[tuple[str, str]]) -> None:
        """The attempt's transcript, whether the run succeeded, paused or failed."""
        memory = runtime.run_memory
        if memory is not None and self.memory_mode == "read_write" and messages:
            run_id, attempt = runtime.run_id, runtime.attempt
            self.harness.writes.submit(
                "memory.transcript",
                lambda: memory.record_messages(messages, run_id, attempt),
                events=runtime.events,
            )

    def recorded_outcome(self, runtime: Runtime, *, success: bool, note: str | None) -> None:
        """How the run ended, as its outcome — unless the agent recorded its own
        (``record_outcome``), which the harness never overwrites."""
        memory = runtime.run_memory
        if memory is not None and self.memory_mode == "read_write" and not runtime.outcome_recorded:
            self.harness.writes.submit(
                "memory.outcome",
                lambda: memory.outcome(success=success, note=note),
                events=runtime.events,
            )

    def judged(
        self, runtime: Runtime, question: str, answer: Any, bundle: ContextBundle | None
    ) -> None:
        """The sampled online judge, in the background: grounded against the context the run
        was given, its verdict a feedback record on the answer and a metric."""
        judge = self.harness.judge
        if not isinstance(answer, str) or not answer or not judge.admits(self.id, runtime.run_id):
            return
        memory = runtime.run_memory
        event = AgentEvalEvent(
            agent_id=self.id,
            agent_run_id=runtime.run_id,
            tenant_id=runtime.tenant,
            trace_id=runtime.identity.context().trace_id,
        )

        async def work() -> None:
            verdict = await judge.verdict(
                event,
                question=question,
                answer=answer,
                verifier=memory,
                bundle=bundle,
                evidence=bundle.rendered if bundle is not None else "",
            )
            if verdict is None:
                return
            metrics.judged(self.id, verdict.score, verdict.method.value)
            if memory is not None and self.memory_mode == "read_write":
                await memory.feedback(verdict.as_feedback(event))

        self.harness.writes.submit("judge", work, events=runtime.events)

    def imported_code_mode_calls(self, runtime: Runtime, *, delay: float) -> None:
        memory, gateway = runtime.run_memory, self.harness.gateway
        if memory is None or gateway is None or self.memory_mode != "read_write":
            return
        run_id, since, task = runtime.run_id, runtime.started_at or datetime.now(UTC), runtime.task

        async def work() -> None:
            for entry in await gateway.code_mode_calls(run_id, since):
                call = ToolCall(
                    tool=entry.name,
                    args=entry.arguments if isinstance(entry.arguments, dict) else {},
                    task=task,
                )
                outcome = ToolOutcome(
                    tool=entry.name,
                    status=ToolStatus.OK if entry.error is None else ToolStatus.ERROR,
                    output=entry.result,
                    latency_ms=entry.latency_ms,
                    error_class="MCPToolError" if entry.error else None,
                )
                await memory.record_tool(call, outcome)

        self.harness.writes.submit(
            "memory.code_mode_calls", work, events=runtime.events, delay=delay
        )

    # ------------------------------------------------------------------ internals
    async def _opened(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None,
        tenant: str | None,
        run_id: str | None = None,
    ) -> Identity:
        """Record an in-process run as started; its identity. (Surfaces pass their own
        ``run_id`` when the protocol names the run.)"""
        start = self._start(
            input, user=user, thread=thread, tenant=tenant, run_id=run_id, record_input=True
        )
        await self.harness.runs.started(start)
        return self._identity_of(RunRecord.from_start(start))

    def _start(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None,
        tenant: str | None,
        run_id: str | None = None,
        record_input: bool = False,
    ) -> RunStart:
        """``record_input`` keeps an in-process run's input as JSON (it may be any object);
        a queued run's input already is."""
        if not user:
            raise ConfigurationError("a run is for somebody: pass user=")
        run_id = run_id or new_id("run_")
        return RunStart(
            run_id=run_id,
            tenant_id=tenant or self.harness.settings.tenant,
            agent_id=self.id,
            thread_id=thread or run_id,  # a run with no conversation is its own thread
            user_id=user,
            input=pipeline.jsonable(input) if record_input else input,
        )

    @staticmethod
    def _identity_of(record: RunRecord) -> Identity:
        return Identity(
            tenant=record.tenant_id,
            user=record.user_id or record.on_behalf_of or "system",
            agent_id=record.agent_id,
            run_id=record.run_id,
            # a scheduled run has no thread: its transcript is its own
            thread=record.thread_id or record.run_id,
            workspace=record.workspace_id,
        )


def schedule_name(agent_id: str, on_behalf_of: str, cron: str, input: Any) -> str:
    """A schedule's name, the same for the same agent, person, cadence and input."""
    digest = content_key("schedule", agent_id, on_behalf_of, cron, pipeline.jsonable(input))
    return f"{agent_id}:{digest}"


#: How the memory service heads the recent conversation in a rendered bundle.
CONVERSATION_HEADING: Final = "## Recent conversation\n"


def without_conversation(bundle: ContextBundle) -> str:
    """The rendered context without its recent-conversation section (the service renders
    sections joined by blank lines), for a target that holds the thread's messages itself."""
    window = bundle.conversation.rendered
    if not window:
        return bundle.rendered
    section = f"{CONVERSATION_HEADING}{window}"
    parts = bundle.rendered.split(f"\n\n{section}", 1)
    if len(parts) == 1:
        parts = bundle.rendered.split(section, 1)
    return "".join(parts).strip()


def _pull(agent: Agent) -> frozenset[str]:
    """The memory service's own tools are recorded by the service, not again by the harness."""
    listed = agent.harness.memory.listed if agent.harness.memory is not None else None
    return frozenset(t.name for t in listed or ())


class RunHandle:
    """A queued run: its id, where it is, and a way to wait for it."""

    def __init__(self, agent: Agent, run_id: str) -> None:
        self.agent = agent
        self.run_id = run_id

    async def status(self) -> RunRecord:
        record = await self.agent.harness.runs.get(self.run_id)
        if record is None:
            raise ConfigurationError(f"no run {self.run_id}")
        return record

    async def result(self, *, timeout: float | None = None) -> Result:  # noqa: ASYNC109
        """Wait until the run pauses or ends (a worker must be running it)."""
        async with asyncio.timeout(timeout):
            while True:
                record = await self.status()
                if record.status is RunStatus.PAUSED or record.final:
                    return Result(
                        run_id=record.run_id,
                        status=record.status,
                        answer=record.output,
                        interrupt=record.awaiting,
                        error=record.error,
                    )
                await asyncio.sleep(POLL_SECONDS)


__all__ = ["Agent", "MemoryMode", "RunHandle"]
