"""``Agent``: a framework target with the harness attached — what ``Harness.wrap`` returns."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import (
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
from trellis.harness.adapters.base import context_window
from trellis.harness.adapters.langgraph import bound_tools, hitl_response, is_hitl
from trellis.harness.adapters.react import ReAct
from trellis.harness.clients.memory import RunMemory, context_budget
from trellis.harness.evals import (
    EvalCase,
    EvalServices,
    Evaluator,
    grounding_score,
    judge,
    name_of,
    sampled,
)
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal
from trellis.harness.result import Result
from trellis.harness.runtime import Runtime, run_of
from trellis.harness.telemetry import output, retrieval_span
from trellis.harness.tools.base import Tool
from trellis.harness.tools.sources import as_source
from trellis.harness.tools.toolbox import Toolbox
from trellis.memory.models import PromptContext

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

#: From this many tools, the tool hints are asked for and narrow what the model is offered.
TOOL_HINTS_MIN: Final = 5
#: How often ``RunHandle.result`` looks at a queued run.
POLL_SECONDS: Final = 0.5
#: What the model is told when memory has nothing for the question (``evidence_status``
#: INSUFFICIENT) or only part of it (INCOMPLETE): answer "I don't know" for what depends on
#: the user's history, instead of a confident guess the memory never held.
ABSTAIN_NOTES: Final = {
    "INSUFFICIENT": (
        "## Memory\nMemory holds nothing about this question. If the answer depends on "
        "something the user told you before, say you do not know it; do not guess."
    ),
    "INCOMPLETE": (
        "## Memory\nMemory covers only part of this question, or someone else. Say what "
        "you do not know rather than fill the gap."
    ),
}


class Agent:
    """Run it, stream it, queue it, resume it, schedule it, serve it."""

    def __init__(self, harness: Harness, target: Any, *, id: str, tools: Sequence[Any] = ()):
        self.harness = harness
        self.target = target
        self.id = safe_id(id)
        self.adapter = detect(target)
        #: the pushed context's token budget: a share of the model's window when it is known
        self.context_budget = context_budget(context_window(target))
        if tools and self.adapter.fixed_tools:
            raise ConfigurationError(
                f"a {self.adapter.name} target binds its tools when it is built: pass "
                f"await h.tools(..., framework='langgraph') to the graph instead of tools="
            )
        self.sources = [as_source(t) for t in tools]
        if self.adapter.fixed_tools:
            self.sources = harness.built_for(bound_tools(target))
        #: the toolbox per tenant, kept fresh
        self._toolboxes: dict[str, Toolbox] = {}

    @functools.cached_property
    def evals(self) -> EvalServices:
        """What this agent's runs are evaluated with: the harness's services, the judge falling
        back to a ``ReAct`` target's own model when ``TRELLIS_JUDGE_MODEL`` is unset."""
        model = self.target.model if isinstance(self.target, ReAct) else None
        return dataclasses.replace(self.harness.evals, fallback_model=model)

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
        start = await self._start(input, user=user, thread=thread, tenant=tenant)
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
        input are one schedule (agent-runs answers the existing one), so a redeploy adds none.
        Pause or resume it in agent-runs (``PATCH /v1/schedules/{id} {"enabled": …}``)."""
        spec = ScheduleSpec(
            tenant_id=await self.harness.tenant(tenant),
            agent_id=self.id,
            name=f"{self.id} for {on_behalf_of}",
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
        awaited = record.awaiting.payload
        if chosen is not InterruptDecision.CANCEL and is_hitl(awaited):
            hitl_response(awaited or {}, resolution)  # a decision the calls do not allow raises
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
        # The run store first: a decision is feedback only once it took effect. A resume
        # the store refuses (answered already, a stale interrupt) raises here, before
        # anything is sent, so approval patterns never learn from a decision that never was.
        resumed = await runs.resumed(resolution)
        run_memory = await self.run_memory(identity)
        if feedback is not None and run_memory is not None and await self.harness.writes_memory():
            await self.harness.writes.submit(
                "memory.feedback",
                lambda: run_memory.feedback(feedback),
                record=run_memory.record("feedback", record=feedback.model_dump(mode="json")),
            )
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

    async def _claimed(
        self, record: RunRecord, worker_id: str, *, lease_seconds: float | None = None
    ) -> Result:
        """A worker's run: fresh from the queue, continuing after a resolution, or after a
        worker died (its checkpoint is the progress it saved)."""
        return await pipeline.attempt(
            self,
            self._identity_of(record),
            record.input,
            number=record.attempt,
            journal=Journal.of(record.checkpoint),
            resolution=record.last_resolution,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
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
    async def run_memory(self, identity: Identity) -> RunMemory | None:
        memory = self.harness.memory
        if memory is None:
            return None
        await self.harness.registered(memory, identity)
        return memory.bind(identity)

    async def tools_for(self, runtime: Runtime) -> list[Tool]:
        """The toolbox (kept fresh per tenant: ``tools/toolbox.py``) and the memory pull
        tools — none, with a warning, when the memory service cannot list them."""
        box = self._toolboxes.get(runtime.tenant)
        if box is None:
            box = self._toolboxes[runtime.tenant] = self.harness.toolbox(
                self.sources, tenant=runtime.tenant
            )
        tools = await box.tools()
        if runtime.run_memory is not None and not self.adapter.fixed_tools:
            try:
                tools.extend(await self.harness.memory_tools(runtime.run_memory))
            except Exception as exc:
                runtime.events.warning("memory_unavailable", f"no memory tools: {exc}")
        return tools

    async def push(self, runtime: Runtime) -> PromptContext | None:
        """The memory context for this run, in the runtime — with the tools section once the
        toolbox is large enough, whose candidates narrow the tools the model is offered. A
        failure is a warning."""
        memory = runtime.run_memory
        if memory is None or not runtime.task:
            return None
        own = runtime.tool_names()
        hinted = len(own) >= TOOL_HINTS_MIN
        with retrieval_span(runtime.task) as span:
            try:
                pushed = await memory.context(
                    runtime.task,
                    tools=own if hinted else None,
                    window=not self.adapter.keeps_conversation(self.target),
                    budget=self.context_budget,
                )
            except Exception as exc:
                runtime.events.warning("memory_unavailable", f"no memory context: {exc}")
                return None
            output(span, pushed.rendered)
        # candidates the model is offered; no candidates at all narrows nothing
        candidates = [n for n in pushed.tool_names if n in runtime.toolbox]
        if hinted and candidates and self.adapter.narrows != "none":
            runtime.offered = set(candidates)
        status = pushed.evidence_status
        note = ABSTAIN_NOTES.get(status)
        rendered = "\n\n".join(part for part in (pushed.rendered, note) if part)
        runtime.context = rendered or None
        runtime.events.emit(
            RunEventType.CONTEXT_LOADED,
            data={"chars": len(rendered), "evidence_status": status},
        )
        return pushed

    async def record_tool(self, runtime: Runtime, call: ToolCall, outcome: ToolOutcome) -> None:
        memory = runtime.run_memory
        if memory is not None and runtime.writes_memory and call.tool not in _pull(self):
            await self.harness.writes.submit(
                "memory.record_tool",
                lambda: memory.record_tool(call, outcome),
                events=runtime.events,
                record=memory.record(
                    "record_tool",
                    call=call.model_dump(mode="json"),
                    outcome=outcome.model_dump(mode="json"),
                ),
            )

    async def recorded_run(self, runtime: Runtime, messages: Sequence[tuple[str, str]]) -> None:
        """The attempt's transcript, whether the run succeeded, paused or failed."""
        memory = runtime.run_memory
        if memory is not None and runtime.writes_memory and messages:
            run_id, attempt = runtime.run_id, runtime.attempt
            await self.harness.writes.submit(
                "memory.transcript",
                lambda: memory.record_messages(messages, run_id, attempt),
                events=runtime.events,
                record=memory.record(
                    "record_messages",
                    messages=[list(m) for m in messages],
                    run_id=run_id,
                    attempt=attempt,
                ),
            )

    async def recorded_outcome(self, runtime: Runtime, status: RunStatus, note: str | None) -> None:
        """How the run ended, as the run's ``system`` feedback: the lowest-ranked voice on
        its outcome (the judge's and a person's override it in the memory service)."""
        memory = runtime.run_memory
        verdict = OUTCOME_VERDICTS.get(status)
        if memory is None or not runtime.writes_memory or verdict is None:
            return
        feedback = {
            "verdict": verdict,
            "source": "system",
            "comment": note,
            "key": f"{runtime.run_id}:outcome",
        }
        await self.harness.writes.submit(
            "memory.outcome",
            lambda: memory.run_feedback(**feedback),
            events=runtime.events,
            record=memory.record("run_feedback", **feedback),
        )

    async def grounded(self, runtime: Runtime, answer: Any, pushed: PromptContext | None) -> None:
        """On a sampled run, the answer checked against the context it was given (the memory
        service's ``/v1/verify``, which records it as the run's ``judge`` feedback), and the
        score put on the run's trace."""
        memory = runtime.run_memory
        if (
            memory is None
            or pushed is None
            or not isinstance(answer, str)
            or not answer
            or not sampled(runtime.run_id, self.harness.settings.grounding_sample)
        ):
            return
        bundle_id, run_id = pushed.bundle_id, runtime.run_id

        async def work() -> None:
            score = await grounding_score(memory.ctx, answer, bundle_id)
            if score is not None:
                await self.harness.score(run_id, "grounding", score, key=f"{run_id}:grounding")

        await self.harness.writes.submit("memory.verify", work, events=runtime.events)

    async def judged(self, runtime: Runtime, answer: Any, pushed: PromptContext | None) -> None:
        """On a sampled run (``TRELLIS_JUDGE_SAMPLE``), each of the harness's online judges in
        the background — never on the request path — as ``judge(case, [it], services=self.evals)``,
        its score on the run's trace. A judge that fails is a warning."""
        judges = self.harness.judges
        rate = self.harness.judge_sample
        if not judges or not isinstance(answer, str) or not answer:
            return
        if not sampled(f"{runtime.run_id}:judges", rate):
            return
        memory = runtime.run_memory
        case = EvalCase(
            input=runtime.task,
            output=answer,
            run_id=runtime.run_id,
            bundle_id=pushed.bundle_id if pushed is not None else None,
            context=runtime.context,
            memory=memory.ctx if memory is not None else None,
        )
        services, events = self.evals, runtime.events
        for evaluator in judges:
            name = name_of(evaluator)

            async def work(evaluator: Evaluator = evaluator, name: str = name) -> None:
                _, failed = await judge(case, [evaluator], services=services)
                if failed:
                    events.warning("judge_failed", f"judge {name}: {failed[name]}")

            await self.harness.writes.submit(f"judge.{name}", work, events=events)

    async def imported_code_mode_calls(self, runtime: Runtime) -> None:
        memory, gateway = runtime.run_memory, self.harness.gateway
        if memory is None or gateway is None or not runtime.writes_memory:
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

        await self.harness.writes.submit("memory.code_mode_calls", work, events=runtime.events)

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
        start = await self._start(
            input, user=user, thread=thread, tenant=tenant, run_id=run_id, record_input=True
        )
        await self.harness.runs.started(start)
        return self._identity_of(RunRecord.from_start(start))

    async def _start(
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
            tenant_id=await self.harness.tenant(tenant),
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


#: The run's ``system`` feedback verdict for how it ended (a cancelled run says nothing
#: about the agent).
OUTCOME_VERDICTS: Final = {RunStatus.SUCCESS: "confirm", RunStatus.ERROR: "reject"}


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


__all__ = ["Agent", "RunHandle"]
