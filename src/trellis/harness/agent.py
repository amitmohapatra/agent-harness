"""``Agent``: a framework target with the harness attached — what ``Harness.wrap`` returns."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import (
    ConfigurationError,
    Interrupt,
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
from trellis.harness import pipeline, skills
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
from trellis.harness.redaction import DEFAULT as REDACTOR
from trellis.harness.result import Result
from trellis.harness.runtime import Runtime, reason_of, run_of
from trellis.harness.skills import Skills
from trellis.harness.subagents import SubAgent, asked_by, cancel_children
from trellis.harness.telemetry import output, retrieval_span, trace_hex
from trellis.harness.tools.base import SideEffects, Tool, arguments_problem
from trellis.harness.tools.sources import as_source
from trellis.harness.tools.toolbox import Toolbox
from trellis.memory.models import PromptContext
from trellis.runs.answers import answer_problem

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.run")

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
    """Run it, stream it, queue it, resume it, cancel it, schedule it, serve it."""

    def __init__(
        self,
        harness: Harness,
        target: Any,
        *,
        id: str,
        tools: Sequence[Any] = (),
        version: str | None = None,
        mcp: Sequence[str] | None = None,
        skills: Sequence[str] = (),
    ):
        self.harness = harness
        self.target = target
        self.id = safe_id(id)
        #: the version of the agent's code: recorded with every run it starts, on its spans
        self.version = version or harness.settings.agent_version
        self.adapter = detect(target)
        #: the pushed context's token budget: a share of the model's window when it is known
        self.context_budget = context_budget(context_window(target))
        given = [n for n, v in (("tools", tools), ("mcp", mcp), ("skills", skills)) if v]
        if given and self.adapter.fixed_tools:
            raise ConfigurationError(
                f"a {self.adapter.name} target binds its tools when it is built: pass "
                f"await h.tools(..., framework='langgraph') to the graph instead of "
                f"{'/'.join(f'{n}=' for n in given)} (skills as skills(...), mcp= to h.tools)"
            )
        self.sources = [as_source(t) for t in tools]
        if skills:
            self.sources.append(Skills(skills))
        #: the Virtual MCPs whose tools the agent has (``None``: everything its key allows)
        self.mcp = None if mcp is None else list(mcp)
        if self.adapter.fixed_tools:
            self.sources, self.mcp = harness.built_for(bound_tools(target))
        #: the toolbox per tenant, kept fresh
        self._toolboxes: dict[str, Toolbox] = {}
        #: the runs an attempt of which runs in this process now, by id
        self.running: dict[str, Runtime] = {}

    @functools.cached_property
    def evals(self) -> EvalServices:
        """What this agent's runs are evaluated with: the harness's services, the judge falling
        back to a ``ReAct`` target's own model when ``TRELLIS_JUDGE_MODEL`` is unset."""
        model = self.target.model if isinstance(self.target, ReAct) else None
        return dataclasses.replace(self.harness.evals, fallback_model=model)

    # ------------------------------------------------------------------ running
    async def run(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None = None,
        tenant: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
        deadline: datetime | None = None,
    ) -> Result:
        """Run to its end (or its first pause) and return how it ended. ``timeout``: the most
        working time the run may take, in seconds (not counting a pause), and ``deadline``
        when it must have ended; past either it ends ``TIMEOUT``."""
        identity = await self._opened(
            input, user=user, thread=thread, tenant=tenant, timeout=timeout, deadline=deadline
        )
        budget = pipeline.Budget.of(timeout=timeout, deadline=deadline)
        return await pipeline.attempt(self, identity, input, budget=budget)

    async def stream(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None = None,
        tenant: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
        deadline: datetime | None = None,
    ) -> AsyncGenerator[RunEvent]:
        """The run's events as they happen, ending with ``RUN_FINISHED``. Closing the stream
        early cancels the run. ``timeout`` and ``deadline`` as for :meth:`run`."""
        identity = await self._opened(
            input, user=user, thread=thread, tenant=tenant, timeout=timeout, deadline=deadline
        )
        budget = pipeline.Budget.of(timeout=timeout, deadline=deadline)
        async for event in self._events(
            lambda listen: pipeline.attempt(
                self, identity, input, listener=listen, streaming=True, budget=budget
            )
        ):
            yield event

    async def start(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None = None,
        tenant: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
        deadline: datetime | None = None,
    ) -> RunHandle:
        """Queue the run for a worker (``h.worker([...]).run()``); it outlives this process.
        ``timeout`` and ``deadline`` as for :meth:`run`: the working time counts across every
        worker that runs it, a crash included."""
        try:
            json.dumps(input)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("a queued run's input must be JSON") from exc
        start = await self._start(
            input, user=user, thread=thread, tenant=tenant, timeout=timeout, deadline=deadline
        )
        await self.harness.runs.start(start, queue=True)
        return RunHandle(self, start.run_id, tenant=start.tenant_id)

    async def cancel(
        self, run_id: str, *, reason: str | None = None, tenant: str | None = None
    ) -> RunRecord:
        """Cancel a run, whatever it is doing, keeping ``reason`` with it: a queued or paused
        run is ``CANCELLED`` at once; one running in this process stops now and ends
        ``CANCELLED``; one a worker elsewhere runs is stopped by that worker at its next
        heartbeat (agent-runs). Its sub-agents' runs that have not ended are cancelled with
        it. A run that already ended raises ``ConflictError``. The run's record is returned.
        ``tenant`` is the run's, named by a platform key only."""
        tenant = await self.harness.tenant(tenant)
        runs = self.harness.runs
        record = await runs.cancel(run_id, reason=reason, tenant=tenant)
        runtime = self.running.get(run_id)
        if runtime is not None:
            runtime.cancelled = reason or "cancelled"
            assert runtime.running_in is not None
            runtime.running_in.cancel()
            await asyncio.wait({runtime.running_in})
            found = await runs.get(run_id, tenant=tenant)
            assert found is not None
            record = found
        await cancel_children(runs, run_id, reason=reason, tenant=tenant)
        return record

    async def resume(
        self,
        interrupt_id: str,
        decision: InterruptDecision | str,
        *,
        answer: Any = None,
        reviewer: str,
        tenant: str | None = None,
    ) -> Result:
        """Answer the interrupt a run is paused on. A run started in process continues here;
        a run that came from the queue goes back to it (``QUEUED``) and a worker continues it.
        ``answer`` is the answer to a question, or the edited arguments of an ``EDIT``;
        ``tenant`` is the run's, named by a platform key only. A question a sub-agent asked is
        answered here, on its parent's run: the answer goes on to the sub-agent's run."""
        record, resolution = await self._resolution(
            interrupt_id, decision, answer, reviewer, tenant=await self.harness.tenant(tenant)
        )
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
        Pause or resume it with ``trellis.runs.RunsClient``: ``schedules.update(id,
        ScheduleUpdate(enabled=…))``."""
        spec = ScheduleSpec(
            tenant_id=await self.harness.tenant(tenant),
            agent_id=self.id,
            name=f"{self.id} for {on_behalf_of}",
            cadence=cron,
            timezone=tz,
            on_behalf_of=on_behalf_of,
            input=input,
        )
        return await self.harness.runs.schedules.create(spec)

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        side_effects: SideEffects | None = None,
    ) -> SubAgent:
        """This agent as a tool other agents call, any framework (``tools=[agent.as_tool()]``,
        ``h.tools(agent.as_tool(), framework=...)``): each call is a child run of this agent
        (``trellis.harness.subagents``). ``name`` (else the agent's id), ``description`` (else
        the function's docstring, or a generic one) and ``side_effects`` (else ``read`` when
        every tool it declares only reads, ``write`` otherwise) are the tool's."""
        return SubAgent(self, name=name, description=description, side_effects=side_effects)

    # ------------------------------------------------------------------ surfaces
    def serve_chat(self, app: Any, *, path: str = "/agui", identity: Any = None) -> None:
        """Mount the AG-UI routes for this agent on a FastAPI ``app``."""
        from trellis.harness.agui import mount  # noqa: PLC0415 - optional extra

        mount(app, self, path=path, identity=identity)

    def serve_a2a(self, app: Any, url: str, *, identity: Any = None) -> None:
        """Publish this agent over A2A at ``url`` (its card and JSON-RPC routes on ``app``)."""
        from trellis.harness.a2a.server import mount  # noqa: PLC0415 - optional extra

        mount(app, self, url=url, identity=identity)

    # ------------------------------------------------------------------ used by surfaces
    async def _resolution(
        self,
        interrupt_id: str,
        decision: InterruptDecision | str,
        answer: Any,
        reviewer: str,
        *,
        tenant: str,
    ) -> tuple[RunRecord, InterruptResolution]:
        run_id = run_of(interrupt_id)
        record = await self.harness.runs.get(run_id, tenant=tenant)
        if record is None or record.agent_id != self.id:
            raise ConfigurationError(f"no run {run_id} of agent {self.id}")
        if record.status is not RunStatus.PAUSED or record.awaiting is None:
            raise ConfigurationError(f"run {run_id} is {record.status.value}, not paused")
        if record.parent_run_id is not None:
            raise ConfigurationError(
                f"run {run_id} is a sub-agent's run: answer the question its parent run "
                f"{record.parent_run_id} waits on"
            )
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
        problem = answer_problem(record.awaiting, resolution) or await self._edit_problem(
            record.awaiting, resolution, tenant=tenant
        )
        if problem:
            raise ConfigurationError(f"not an answer to {interrupt_id}: {problem}")
        awaited = record.awaiting.payload
        if chosen is not InterruptDecision.CANCEL and is_hitl(awaited):
            hitl_response(awaited or {}, resolution)  # a decision the calls do not allow raises
        return record, resolution

    async def _edit_problem(
        self, awaiting: Interrupt, resolution: InterruptResolution, *, tenant: str
    ) -> str | None:
        """Why the edited arguments of a tool call do not fit the tool's input schema
        (``arguments_problem``). Only for a tool the agent's toolbox lists now: one it cannot
        list (an MCP server that is down) is left to the call, where the tool checks its own
        arguments when it runs; a call LangChain's middleware holds is the middleware's
        (``hitl_response``)."""
        call = awaiting.tool_call
        if (
            resolution.decision is not InterruptDecision.EDIT
            or call is None
            or is_hitl(awaiting.payload)
        ):
            return None
        try:
            tools = await self._toolbox(tenant).tools()
        except Exception as exc:
            log.info("the edit of %s is not checked: no toolbox (%s)", call.tool, exc)
            return None
        found = next((t for t in tools if t.name == call.tool), None)
        if found is None:
            return None
        problem = arguments_problem(found.spec.input_schema or {}, resolution.payload or {})
        return f"the edited arguments do not fit {call.tool}: {problem}" if problem else None

    async def _continue(
        self,
        record: RunRecord,
        resolution: InterruptResolution,
        listener: Callable[[RunEvent], None] | None = None,
    ) -> Result:
        journal, resumed = await self._resumed(record, resolution)
        if resumed.status is RunStatus.CANCELLED:  # its children wait on nobody now
            await cancel_children(
                self.harness.runs,
                record.run_id,
                reason=reason_of(resolution),
                tenant=record.tenant_id,
            )
        if resumed.status is not RunStatus.RUNNING:
            # cancelled, or back on the queue for a worker (a run that came from the queue)
            return Result(run_id=record.run_id, status=resumed.status)
        return await pipeline.attempt(
            self,
            self._identity_of(record),
            record.input,
            number=resumed.attempt,
            journal=journal,
            resolution=resolution,
            listener=listener,
            streaming=listener is not None,
            budget=_budget(record),
            started_on=record.agent_version,
        )

    async def _resumed(
        self, record: RunRecord, resolution: InterruptResolution
    ) -> tuple[Journal, RunRecord]:
        """The paused run answered in the run store, with the journal its next attempt reads;
        a decision about a tool call becomes feedback (a sub-agent's question is fed back
        where it was asked: on the sub-agent's run)."""
        runs = self.harness.runs
        assert record.awaiting is not None
        identity = self._identity_of(record)
        feedback = (
            None
            if asked_by(record.awaiting) is not None
            else resolution.to_feedback(record.awaiting, identity.context())
        )
        # read before the resume: a journal that cannot be read leaves the run waiting
        journal = await Journal.read(record.checkpoint, runs.artifacts, tenant=record.tenant_id)
        # The run store first: a decision is feedback only once it took effect. A resume
        # the store refuses (answered already, a stale interrupt) raises here, before
        # anything is sent, so approval patterns never learn from a decision that never was.
        resumed = await runs.resume(resolution, tenant=record.tenant_id)
        run_memory = await self.run_memory(identity)
        if feedback is not None and run_memory is not None and await self.harness.writes_memory():
            await self.harness.writes.submit(
                "memory.feedback",
                lambda: run_memory.feedback(feedback),
                record=run_memory.record("feedback", record=feedback.model_dump(mode="json")),
            )
        return journal, resumed

    async def _claimed(
        self,
        record: RunRecord,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
        remaining: float | None = None,
    ) -> Result:
        """A worker's run: fresh from the queue, continuing after a resolution, or after a
        worker died (its checkpoint is the progress it saved). ``remaining`` is the working
        time agent-runs says the run has left (its lease's)."""
        artifacts = self.harness.runs.artifacts
        return await pipeline.attempt(
            self,
            self._identity_of(record),
            record.input,
            number=record.attempt,
            journal=await Journal.read(record.checkpoint, artifacts, tenant=record.tenant_id),
            resolution=record.last_resolution,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            budget=_budget(record, remaining),
            started_on=record.agent_version,
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
        tools = await self._toolbox(runtime.tenant).tools()
        if runtime.run_memory is not None and not self.adapter.fixed_tools:
            try:
                tools.extend(await self.harness.memory_tools(runtime.run_memory))
            except Exception as exc:
                runtime.events.warning("memory_unavailable", f"no memory tools: {exc}")
        return tools

    def _toolbox(self, tenant: str) -> Toolbox:
        box = self._toolboxes.get(tenant)
        if box is None:
            box = self._toolboxes[tenant] = self.harness.toolbox(
                self.sources, tenant=tenant, mcp=self.mcp
            )
        return box

    async def push(self, runtime: Runtime) -> PromptContext | None:
        """What is pushed into the framework's input, in the runtime's context: the memory
        context (:meth:`remembered`), then the section of the skills the run pinned
        (``skills.pin``). The memory context, when there is one, is returned."""
        pushed = await self.remembered(runtime)
        await skills.pin(runtime, self.sources)
        return pushed

    async def remembered(self, runtime: Runtime) -> PromptContext | None:
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
            call, outcome = _redacted(call, outcome)
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
        bundle_id, run_id, traced = pushed.bundle_id, runtime.run_id, runtime.trace_run

        async def work() -> None:
            score = await grounding_score(memory.ctx, answer, bundle_id)
            if score is not None:
                key = f"{run_id}:grounding"
                await self.harness.score(run_id, "grounding", score, key=key, trace=traced)

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
            trace_id=trace_hex(runtime.trace_run),
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
                await memory.record_tool(*_redacted(call, outcome))

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
        timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
        deadline: datetime | None = None,
    ) -> Identity:
        """Record an in-process run as started; its identity. (Surfaces pass their own
        ``run_id`` when the protocol names the run.)"""
        start = await self._start(
            input,
            user=user,
            thread=thread,
            tenant=tenant,
            run_id=run_id,
            record_input=True,
            timeout=timeout,
            deadline=deadline,
        )
        await self.harness.runs.start(start)
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
        timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
        deadline: datetime | None = None,
        parent: str | None = None,
    ) -> RunStart:
        """``record_input`` keeps an in-process run's input as JSON (it may be any object);
        a queued run's input already is. The run's time limit, deadline, the agent's version
        and the run it is a sub-agent's run of (``parent``) go with it when there are any."""
        if not user:
            raise ConfigurationError("a run is for somebody: pass user=")
        run_id = run_id or new_id("run_")
        given = {
            "timeout_seconds": timeout,
            "deadline": deadline,
            "agent_version": self.version,
            "parent_run_id": parent,
        }
        return RunStart(
            run_id=run_id,
            tenant_id=await self.harness.tenant(tenant),
            agent_id=self.id,
            thread_id=thread or run_id,  # a run with no conversation is its own thread
            user_id=user,
            input=pipeline.jsonable(input) if record_input else input,
            **{name: value for name, value in given.items() if value is not None},
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


def _budget(record: RunRecord, remaining: float | None = None) -> pipeline.Budget | None:
    """What is left of a run's time, from its record: the working-time limit less the time
    it already worked (agent-runs keeps it across attempts, a crash included) — or what its
    lease says is ``remaining`` — and the deadline."""
    return pipeline.Budget.of(
        timeout=record.timeout_seconds,
        worked=record.worked_seconds,
        deadline=record.deadline,
        remaining=remaining,
    )


def _redacted(call: ToolCall, outcome: ToolOutcome) -> tuple[ToolCall, ToolOutcome]:
    """A tool call as the memory service's tool records get it: its arguments and output
    redacted (``redaction.py``), as everything leaving the process is; the model and the tool
    had them as they are."""
    return (
        call.model_copy(update={"args": REDACTOR.redact_input(call.args)}),
        outcome.model_copy(update={"output": REDACTOR.redact_output(outcome.output)}),
    )


def _pull(agent: Agent) -> frozenset[str]:
    """The memory service's own tools are recorded by the service, not again by the harness."""
    listed = agent.harness.memory.listed if agent.harness.memory is not None else None
    return frozenset(t.name for t in listed or ())


class RunHandle:
    """A queued run: its id, where it is, and a way to wait for it."""

    def __init__(self, agent: Agent, run_id: str, *, tenant: str) -> None:
        self.agent = agent
        self.run_id = run_id
        self.tenant = tenant

    async def status(self) -> RunRecord:
        record = await self.agent.harness.runs.get(self.run_id, tenant=self.tenant)
        if record is None:
            raise ConfigurationError(f"no run {self.run_id}")
        return record

    async def cancel(self, *, reason: str | None = None) -> RunRecord:
        """Cancel the run, whatever it is doing (``Agent.cancel``)."""
        return await self.agent.cancel(self.run_id, reason=reason, tenant=self.tenant)

    async def result(self, *, timeout: float | None = None) -> Result:  # noqa: ASYNC109
        """Wait until the run pauses or ends (a worker must be running it)."""
        async with asyncio.timeout(timeout):
            while True:
                record = await self.status()
                if record.status is RunStatus.PAUSED or record.final:
                    return Result.of(record)
                await asyncio.sleep(POLL_SECONDS)


__all__ = ["Agent", "RunHandle"]
