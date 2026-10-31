"""``Agent``: a framework target with the harness attached — what ``Harness.wrap`` returns."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError as PydanticError

from trellis.contracts import (
    ConfigurationError,
    ErrorCategory,
    HarnessError,
    Interrupt,
    InterruptDecision,
    InterruptRemember,
    InterruptResolution,
    RunEvent,
    RunEventType,
    RunOutcome,
    RunRecord,
    RunStart,
    RunStatus,
    Schedule,
    ScheduleSpec,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
    new_id,
    safe_id,
)
from trellis.harness import (
    compat,
    conversation,
    history,
    pipeline,
    sandbox,
    snapshots,
    telemetry,
)
from trellis.harness.adapters import detect
from trellis.harness.adapters.base import (
    FRAMEWORK_OPTIONS,
    context_window,
    declared_tools,
    merged,
    model_name,
)
from trellis.harness.adapters.langgraph import checkpointing, hitl_response, is_hitl
from trellis.harness.bounded import LRU
from trellis.harness.bridges import helpers
from trellis.harness.budget import BUDGET, Budget
from trellis.harness.clients.memory import (
    CITED_MAX,
    DOCUMENTS,
    PUSHED_BUNDLE,
    PUSHED_MEMORIES,
    RunMemory,
    Window,
    context_budget,
    documents,
)
from trellis.harness.conversation import STEERED_INTO, Mode, Turns
from trellis.harness.evals import (
    GROUNDING_SAMPLE,
    EvalCase,
    EvalServices,
    Evaluator,
    grounding,
    grounding_score,
    judge,
    name_of,
    sample_of,
    sampled,
)
from trellis.harness.features import COVERS, WITHOUT, Feature, features, run_without
from trellis.harness.history import Said, from_window, recorded
from trellis.harness.hooks import Chain, Hooks
from trellis.harness.identity import AGENT_GROUP, Identity, conversation_key
from trellis.harness.journal import Journal
from trellis.harness.result import Result
from trellis.harness.runlog import event_log
from trellis.harness.runtime import Runtime, current, reason_of, run_of
from trellis.harness.subagents import asked_by, cancel_children, child_run
from trellis.harness.telemetry import metrics, output, retrieval_span, trace_hex
from trellis.harness.tools.base import Source, Tool, arguments_problem, max_age
from trellis.harness.tools.sources import as_source
from trellis.harness.tools.toolbox import Toolbox
from trellis.memory.models import PromptContext
from trellis.runs import Job, RateLimitedError
from trellis.runs.answers import answer_problem

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.run")

#: Everything a run does with the memory service: without all of it, a run has no memory scope.
MEMORY: Final = frozenset(COVERS["memory"])
#: Whether this trellis-contracts' schedules carry a queue order (``ScheduleSpec.priority``,
#: ``concurrency_key``: 0.6.1 on).
SCHEDULES_QUEUE: Final = {"priority", "concurrency_key"} <= set(ScheduleSpec.model_fields)
#: How often ``RunHandle.result`` looks at a queued run.
POLL_SECONDS: Final = 0.5
#: The best tool hint's confidence below which the hints narrow nothing: every tool is offered
#: (the window fits them) with ``tool_search``. A match scores about 0.6 at its best (the
#: service scores relative to the best match); nothing matching scores 0 to 0.1.
NARROW_FLOOR: Final = 0.5
#: How a run's memory context reaches an agent whose input has no place for it, by framework
#: (``Agent.unreached``): what the warning names as the fix.
UNREACHED_FIX: Final = {
    "function": "read agent.context (the runtime your function is given; "
    "trellis.current().context in the code it calls), or pass the input as a list of "
    "messages, which gets the context as its leading system message",
    "langgraph": "give the graph's input its messages (a 'messages' state key: "
    "{'messages': [...]}), which get the context as a system message, or read "
    "trellis.current().context in a node",
}
#: How long a turn's transcript may take to write before the run returns, where memory keeps
#: the conversation; past it the write is queued (``writes``) and the run returns.
TURN_WRITE_SECONDS: Final = 5.0
#: How many conversations' last framework sessions an agent keeps in this process
#: (``Agent.sessions``; a session's transcripts can be large): the least recently used goes,
#: and its next turn starts a new one.
SESSIONS_KEPT: Final = 128
#: What the model is told when memory has nothing for the question (``evidence_status``
#: INSUFFICIENT) or only part of it (INCOMPLETE): for what depends on the user's history, an
#: honest "I don't know" instead of a confident guess the memory never held — one that names
#: the detail it does not have and offers to look it up or to be told it, so the user knows
#: what is missing and what to do next. Not when a hinted tool answers the question (its
#: confidence at least :data:`NARROW_FLOOR`): told "do not guess", a weak model searched
#: memory in loops instead of calling it.
ABSTAIN_NOTES: Final = {
    "INSUFFICIENT": (
        "## Memory\nMemory holds nothing about this question. If the answer depends on "
        "something the user told you before, do not guess: say which detail you do not "
        "have, and offer to look it up or ask the user for it."
    ),
    "INCOMPLETE": (
        "## Memory\nMemory covers only part of this question, or someone else. Answer the "
        "part it covers; for the rest, do not fill the gap: say which detail you do not "
        "have, and offer to look it up or ask the user for it."
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
        version: str | None = None,
        description: str | None = None,
        hooks: Sequence[Hooks] = (),
        judges: Sequence[Evaluator] = (),
        budget: Budget | None = None,
        run_timeout: float | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
    ):
        if run_timeout is not None and run_timeout <= 0:
            raise ConfigurationError(f"{id}: run_timeout is a number of seconds over 0")
        if any(isinstance(j, grounding) for j in judges):
            raise ConfigurationError(
                f"{id}: grounding() is not one of h.wrap's judges: every wrapped run's answer is "
                "already checked against its memory context, on a sampled share of runs "
                '(without={"grounding"} turns it off); grounding() is for h.evaluate and judge()'
            )
        self.harness = harness
        self.target = target
        self.id = safe_id(id)
        #: the version of the agent's code: recorded with every run it starts, on its spans
        #: (else the OpenTelemetry resource's ``service.version``)
        self.version = version or telemetry.agent_version()
        #: what the agent does, as other agents and the A2A card read it (else its function's
        #: docstring)
        self.description = description
        #: the most working time one of its runs may take, in seconds, when the run names
        #: none (``None``: no limit)
        self.timeout = run_timeout
        #: what the harness does not do for any of its runs (``h.wrap(without=)``)
        self.without = features(without)
        #: the most each of its runs may spend (``budget.py``; ``None``: no cap)
        self.budget = budget
        #: the hooks around its runs, model calls and tool calls
        self.hooks = Chain(list(hooks))
        #: the online judges every sampled successful run of it is scored by
        #: (``llm_judge(sample=)``: the share of runs each scores)
        self.judges: list[Evaluator] = list(judges)
        self.adapter = detect(target)
        if self.adapter.name == "langgraph":  # a middleware() graph: its checkpoint in the run
            self.target = checkpointing(target)
        # a framework outside the range this release was tested with: said once, never refused
        compat.check(self.adapter.name)
        if self.adapter.name == "openai_agents":  # its SDK's own tracing: left as set, said
            from trellis.harness.adapters.openai_agents import sdk_tracing_said  # noqa: PLC0415

            sdk_tracing_said()
        for gated in sorted(gated_tools(target)):
            log.warning(
                "%s: %s is approved in the framework's own gate: governance counts that "
                "approval, so each call is asked once; approval rules belong in the tool "
                "catalog (risk, approve_when), the one place every framework reads",
                id,
                gated,
            )
        #: the framework's own options for every run's call (``h.wrap(framework_options=)``):
        #: a run's own go over them
        self.framework_options = dict(framework_options or {})
        self.adapter.check_options(self.framework_options)
        #: the pushed context's token budget: a share of the model's window when it is known
        self.context_budget = context_budget(context_window(target))
        #: the name of the agent's model when its builder says it (``ReAct``): what the
        #: memory service is told with the endpoint, with its profile
        self.model = model_name(target)
        #: the tools the harness runs for the agent: what its builder declares
        #: (``ReAct(tools=, subagents=)``); besides them every run has the MCP tools the
        #: virtual key allows and the memory tools (``tools_for``)
        self.sources: list[Source] = [as_source(t) for t in declared_tools(target)]
        #: the toolbox per tenant, kept fresh
        self._toolboxes: LRU[str, Toolbox] = LRU()
        #: the framework's own tools last published to each tenant's catalog, by name
        self._published_native: LRU[str, tuple[str, ...]] = LRU()
        #: the runs an attempt of which runs in this process now, by id
        self.running: dict[str, Runtime] = {}
        #: one run of a conversation at a time in this process (``pipeline.attempt``)
        self.turns = Turns()
        #: the session its framework kept for each conversation's last turn in this process
        #: (a Claude Agent SDK session and its transcripts, an ACP agent's session): the next
        #: turn continues it (``pipeline``, ADR 0005), the most recent kept
        self.sessions: LRU[str, tuple[str, dict[str, list[dict[str, Any]]]]] = LRU(SESSIONS_KEPT)
        #: who follows a run's events in this process (``events``), by run id
        self._watching: dict[str, list[Callable[[RunEvent | None], None]]] = {}
        #: whether a run's memory context reaching nothing was said (``unreached``: once)
        self._unreached_said = False

    @property
    def evals(self) -> EvalServices:
        """What this agent's runs are evaluated with: the harness's services (the judge is the
        gateway's judge alias, never the agent's own model)."""
        return self.harness.evals

    # ------------------------------------------------------------------ running
    async def run(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None = None,
        tenant: str | None = None,
        run_timeout: float | None = None,
        deadline: datetime | None = None,
        budget: Budget | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        mode: Mode | str = Mode.QUEUE,
    ) -> Result:
        """Run to its end (or its first pause) and return how it ended. ``run_timeout``: the
        most working time the run may take, in seconds (not counting a pause; else the
        agent's, ``h.wrap(run_timeout=)``), and ``deadline`` when it must have ended; past
        either it ends ``TIMEOUT``. ``budget``: the most the run may spend (else the agent's,
        ``h.wrap(budget=)``). ``without``: what the harness does not do for this run, besides
        what the agent's ``without`` says (``trellis.harness.features``). ``framework_options``: the
        framework's own options for this run's call, over the agent's
        (``h.wrap(framework_options=)``; each framework's page says where they go). The run
        keeps the JSON ones with its record, for its later attempts (a resume); a value that
        is not JSON (an object) is this call's only. ``document_ids``: the documents
        (``h.add_document``'s ``document_id``) the run's memory context draws its passages
        from, none else (unset: all the person's); kept with the run, its sub-agents' runs
        included. ``mode``: what the run does when another run of the same conversation
        (``thread``) works in this process (``conversation.Mode``): ``"queue"`` waits for it
        to end (the default), ``"interrupt"`` cancels it and continues the conversation,
        ``"steer"`` gives the message to it at its framework's next step — its answer is this
        run's — where the target can take one (else it waits, as ``"queue"``).

        Called inside another run (a tool of it, a node, a function target's code), the run is
        that run's child, as a sub-agent's is (``trellis.harness.subagents.child_run``): its
        record names its parent, its memory scope is derived from the parent's, it takes the
        parent's tenant, thread, documents and what the parent is without, and never works
        past the parent's time; its journal is kept in the parent's (a crash continues it, a
        re-run finds it again), a question it asks pauses the parent, whose resume answers it,
        and cancelling the parent cancels it."""
        parent = current()
        if parent is not None:
            result = await child_run(
                self,
                parent,
                input,
                user=user,
                thread=thread,
                tenant=tenant,
                timeout=run_timeout,
                deadline=deadline,
                budget=budget,
                without=without,
                framework_options=framework_options,
                document_ids=document_ids,
            )
            return self.hooks.answer(result)
        mode = Mode(mode)
        record = await self._opened(
            input,
            user=user,
            thread=thread,
            tenant=tenant,
            timeout=run_timeout,
            deadline=deadline,
            budget=budget,
            without=without,
            framework_options=framework_options,
            document_ids=document_ids,
            mode=mode,
        )
        return self.hooks.answer(await pipeline.attempt(self, record, input, mode=mode))

    async def stream(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None = None,
        tenant: str | None = None,
        run_timeout: float | None = None,
        deadline: datetime | None = None,
        budget: Budget | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        mode: Mode | str = Mode.QUEUE,
    ) -> AsyncGenerator[RunEvent]:
        """The run's events as they happen, ending with ``RUN_FINISHED``. Closing the stream
        early cancels the run. ``run_timeout``, ``deadline``, ``budget``, ``without``,
        ``framework_options``, ``document_ids`` and ``mode`` as for :meth:`run` (a message
        steered into the working run streams that run's answer, under its id). Streamed
        inside another run, it is that run's child (its record names the parent; tenant,
        thread, documents and what the parent is without carried over), whose events are the
        stream's: a
        question it asks ends the stream paused, for the caller to answer (``run`` pauses the
        parent instead)."""
        record = await self._opened(
            input,
            user=user,
            thread=thread,
            tenant=tenant,
            timeout=run_timeout,
            deadline=deadline,
            budget=budget,
            without=without,
            framework_options=framework_options,
            document_ids=document_ids,
            parent=current(),
            mode=Mode(mode),
        )
        async for event in self._events(
            lambda listen: pipeline.attempt(self, record, input, listener=listen, mode=Mode(mode))
        ):
            yield self.hooks.event(event)

    async def start(
        self,
        input: Any,
        *,
        user: str,
        thread: str | None = None,
        tenant: str | None = None,
        run_timeout: float | None = None,
        deadline: datetime | None = None,
        budget: Budget | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        priority: int = 0,
        concurrency_key: str | None = None,
    ) -> RunHandle:
        """Queue the run for a worker (``h.worker([...]).run()``); it outlives this process.
        ``run_timeout``, ``deadline``, ``budget``, ``without``, ``framework_options`` and
        ``document_ids`` as for :meth:`run`: the
        working time counts across every worker that runs it, a crash included; the framework
        options are kept with the run, so they are JSON (an object goes on
        ``h.wrap(framework_options=)``). ``priority`` (-1000 to 1000):
        a higher one is claimed first among the tenant's queued runs. ``concurrency_key``:
        runs sharing it run one at a time (the run store's limit) — by default the
        conversation's (``thread:<conversation>``, when ``thread`` is given: the key of the
        tenant's, the user's and the thread's, ``identity.conversation_key``), so a second
        message to a busy conversation waits for the first run instead of running beside it,
        and another user's conversation named alike never waits for it."""
        try:
            json.dumps(input)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("a queued run's input must be JSON") from exc
        _kept_as_json(framework_options)
        tenant = await self.harness._tenant(tenant)
        conversation = conversation_key(tenant, user, thread) if thread else None
        concurrency_key = concurrency_key or (conversation and f"thread:{conversation}")
        start = await self._start(
            input,
            user=user,
            thread=thread,
            tenant=tenant,
            timeout=run_timeout,
            deadline=deadline,
            budget=budget,
            without=without,
            framework_options=framework_options,
            priority=priority,
            concurrency_key=concurrency_key,
            document_ids=document_ids,
            parent=current(),
        )
        await admitted(self.harness.runs.start(start, queue=True), "start")
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
        tenant = await self.harness._tenant(tenant)
        runs = self.harness.runs
        runtime = self.running.get(run_id)
        # read first: its ending clears the checkpoint, which names the run's sandbox
        held = await runs.get(run_id, tenant=tenant) if runtime is None else None
        record = await runs.cancel(run_id, reason=reason, tenant=tenant)
        if runtime is not None:
            runtime.cancelled = reason or "cancelled"
            assert runtime.running_in is not None
            runtime.running_in.cancel()
            await asyncio.wait({runtime.running_in})
            found = await runs.get(run_id, tenant=tenant)
            assert found is not None
            record = found
        elif held is not None and record.status is RunStatus.CANCELLED:  # no attempt ends it
            # (a journal kept as a run artifact is not read for it: the reaper deletes it)
            await sandbox.ended(self, (held.checkpoint or {}).get("sandbox"), run_id)
            self._unwatched(run_id)
        await cancel_children(runs, run_id, reason=reason, tenant=tenant)
        return record

    async def resume(
        self,
        interrupt_id: str,
        decision: InterruptDecision | str,
        *,
        answer: Any = None,
        reviewer: str,
        comment: str | None = None,
        remember: InterruptRemember = "once",
        tenant: str | None = None,
    ) -> Result:
        """Answer the interrupt a run is paused on (``interrupt_id``, or the run's id: what it
        waits on now). A run started in process continues here; a run that came from the queue
        goes back to it (``QUEUED``) and a worker continues it. ``answer`` is the answer to a
        question (a result from outside the run included: what an ``ask`` in a tool returns
        to it), or the edited arguments of an ``EDIT``. ``reviewer`` is who decided;
        ``comment`` their remark, kept with the decision (the run record, the ``decision``
        event, the span, the feedback); ``remember="run"`` with an ``approve`` of a tool call
        approves that tool's later calls in this run without asking (never another run's).
        ``tenant`` is the run's, named by a platform key only. A question a sub-agent asked is
        answered here, on its parent's run: the answer goes on to the sub-agent's run."""
        record, resolution = await self._resolution(
            interrupt_id,
            decision,
            answer,
            reviewer,
            tenant=await self.harness._tenant(tenant),
            comment=comment,
            remember=remember,
        )
        return self.hooks.answer(await self._continue(record, resolution))

    async def schedule(
        self,
        cron: str,
        *,
        user: str,
        input: Any,
        tz: str = "UTC",
        tenant: str | None = None,
        run_timeout: float | None = None,
        budget: Budget | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        priority: int = 0,
        concurrency_key: str | None = None,
    ) -> Schedule:
        """Queue a run of this agent on ``cron`` (evaluated in ``tz``), for ``user`` with
        ``input``. Workers run them. Idempotent: the same agent, person, cadence and input are
        one schedule (agent-runs answers the existing one), so a redeploy adds none. Each
        fired run records the agent's ``version`` and takes what :meth:`start` takes for one
        run: ``run_timeout`` (else the agent's), ``budget``, ``without``, ``framework_options``
        (JSON),
        ``document_ids``, ``priority`` and ``concurrency_key`` — kept with the schedule (its
        ``metadata``, under the keys a started run keeps them under) and copied into every run
        it fires. Pause or resume it with ``trellis.runs.RunsClient``: ``schedules.update(id,
        ScheduleUpdate(enabled=…))``."""
        _kept_as_json(framework_options)
        queued = {"priority": priority or None, "concurrency_key": concurrency_key}
        ordered = {name: value for name, value in queued.items() if value is not None}
        if ordered and not SCHEDULES_QUEUE:
            raise ConfigurationError(
                f"schedule({', '.join(f'{n}=' for n in ordered)}) needs trellis-contracts "
                "0.6.1 or later: this one's ScheduleSpec has no priority or concurrency_key"
            )
        spec = ScheduleSpec(
            tenant_id=await self.harness._tenant(tenant),
            agent_id=self.id,
            name=f"{self.id} for {user}",
            cadence=cron,
            timezone=tz,
            on_behalf_of=user,
            input=input,
            timeout_seconds=self.timeout if run_timeout is None else run_timeout,
            agent_version=self.version,
            metadata=self._metadata(features(without), framework_options, document_ids, budget),
            **ordered,
        )
        return await admitted(self.harness.runs.schedules.create(spec), "schedule")

    # ------------------------------------------------------------------ surfaces
    def serve_chat(self, app: Any, *, path: str = "/agui", identity: Any = None) -> None:
        """Mount the AG-UI routes for this agent on a FastAPI ``app``. The caller is the user
        the trusted ``X-Trellis-Identity`` header names (as ``serve_a2a`` reads it; a request
        without it is ``401``), else ``identity(request) -> user`` when given."""
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
        comment: str | None = None,
        remember: InterruptRemember = "once",
    ) -> tuple[RunRecord, InterruptResolution]:
        run_id = run_of(interrupt_id)
        record = await self.harness.runs.get(run_id, tenant=tenant)
        if record is None or record.agent_id != self.id:
            raise ConfigurationError(f"no run {run_id} of agent {self.id}")
        if record.status is not RunStatus.PAUSED or record.awaiting is None:
            raise ConfigurationError(f"run {run_id} is {record.status.value}, not paused")
        if interrupt_id == run_id:  # the run's id: what it waits on now
            interrupt_id = record.awaiting.interrupt_id
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
        try:
            resolution = InterruptResolution(
                interrupt_id=interrupt_id,
                run_id=run_id,
                decision=chosen,
                answer=None if edited else answer,
                payload=answer if edited else None,
                reviewer=reviewer,
                comment=comment,
                remember=remember,
            )
        except PydanticError as exc:
            reasons = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
            raise ConfigurationError(f"not an answer to {interrupt_id}: {reasons}") from exc
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
            await sandbox.ended(self, journal.sandbox, record.run_id)
            self._unwatched(record.run_id)
        if resumed.status is not RunStatus.RUNNING:
            # cancelled, or back on the queue for a worker (a run that came from the queue)
            return Result(run_id=record.run_id, status=resumed.status)
        return await pipeline.attempt(
            self, resumed, record.input, journal=journal, resolution=resolution, listener=listener
        )

    async def _resumed(
        self, record: RunRecord, resolution: InterruptResolution
    ) -> tuple[Journal, RunRecord]:
        """The paused run answered in the run store, with the journal its next attempt reads;
        a decision about a tool call becomes feedback (a sub-agent's question is fed back
        where it was asked: on the sub-agent's run)."""
        runs = self.harness.runs
        assert record.awaiting is not None
        identity = Identity.of(record)
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
        resumed = await admitted(runs.resume(resolution, tenant=record.tenant_id), "resume")
        without = run_without(record)
        run_memory = await self.run_memory(identity, without)
        if (
            feedback is not None
            and run_memory is not None
            and "records" not in without | self.without
        ):
            await self.harness.writes.submit(
                "memory.feedback",
                lambda: run_memory.feedback(feedback),
                record=run_memory.record("feedback", record=feedback.model_dump(mode="json")),
            )
        return journal, resumed

    async def execute(self, job: Job) -> Result:
        """The next attempt of a run of this agent a worker claimed (``job``: its record, the
        worker holding its lease, the lease's length and the working time it says is left) —
        fresh from the queue, continuing after a person's answer, or after a worker died (its
        checkpoint is the progress it saved) — with its journal, governance, memory and
        limits. Any worker hands it the runs it claims: ``h.worker``, a
        ``trellis.runs.Worker(runs, agent.execute, [agent.id])``, or a loop of your own around
        ``runs.claim``."""
        record = job.record
        if record.agent_id != self.id:
            raise ConfigurationError(f"run {record.run_id} is {record.agent_id}'s, not {self.id}'s")
        queued = record.created_at if record.attempt == 1 else None
        if record.last_resolution is not None:  # queued again when a person answered
            queued = record.last_resolution.resolved_at
        if queued is not None:
            metrics.queue_waited(self.id, (datetime.now(UTC) - queued).total_seconds())
        artifacts = self.harness.runs.artifacts
        return await pipeline.attempt(
            self,
            record,
            record.input,
            journal=await Journal.read(record.checkpoint, artifacts, tenant=record.tenant_id),
            resolution=record.last_resolution,
            job=job,
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

    async def events(
        self, run_id: str, *, after: int = 0, tenant: str | None = None
    ) -> AsyncIterator[RunEvent]:
        """A run's events, until it ends (a paused run's stay open: its next attempt's
        follow). With agent-runs (``RUNS_URL``) every event of the run past position ``after``
        in its log, then each as it is appended — wherever the run is executed (a worker, a
        replica). In process (no ``RUNS_URL``), the events of its attempts run in this process
        from now on (``after`` is not used: nothing is kept). ``tenant`` is the run's, named by
        a platform key only."""
        tenant = await self.harness._tenant(tenant)
        store = event_log(self.harness.runs)
        if store is not None:
            async for entry in store.stream_events(run_id, after=after, tenant=tenant):
                yield self.hooks.event(entry.event)
            return
        record = await self.harness.runs.get(run_id, tenant=tenant)
        if record is None or record.agent_id != self.id:
            raise ConfigurationError(f"no run {run_id} of agent {self.id}")
        if record.final:
            return
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        watchers = self._watching.setdefault(run_id, [])
        watchers.append(queue.put_nowait)
        runtime = self.running.get(run_id)
        if runtime is not None:  # the attempt running now
            runtime.events.listen(queue.put_nowait)
        try:
            while (event := await queue.get()) is not None:
                yield self.hooks.event(event)
                if event.type is RunEventType.RUN_FINISHED and event.outcome not in (
                    RunOutcome.INTERRUPT,
                    None,
                ):
                    return
        finally:
            watchers.remove(queue.put_nowait)
            if not watchers:
                del self._watching[run_id]

    def watchers(self, run_id: str) -> list[Callable[[RunEvent], None]]:
        """Who follows ``run_id``'s events in this process (``events``)."""
        return list(self._watching.get(run_id, ()))

    def _unwatched(self, run_id: str) -> None:
        """The run ended with no attempt to say so (cancelled while it waited): whoever
        follows its events here is done."""
        for watcher in self._watching.get(run_id, ()):
            watcher(None)

    # ------------------------------------------------------------------ used by the pipeline
    async def run_memory(
        self, identity: Identity, without: frozenset[Feature] = frozenset()
    ) -> RunMemory | None:
        """The memory service in the run's scope — none with memory off, for the deployment or
        for the run (``without=memory``: no push, no pull, no records)."""
        memory = self.harness.memory
        if memory is None or without | self.without >= MEMORY:
            return None
        await self.harness._registered(memory, identity, self.model)
        return memory.bind(identity)

    async def tools_for(self, runtime: Runtime) -> list[Tool]:
        """The toolbox (kept fresh per tenant: ``tools/toolbox.py``) and the memory pull
        tools — none, with a warning, when the memory service cannot list them — less the
        tools of what the run is without (``without=``: its MCP tools...)."""
        box = self._toolbox(runtime.tenant)
        tools = await box.tools(code_mode=runtime.uses("code_mode"))
        if (
            runtime.run_memory is not None
            and not self.adapter.fixed_tools
            and runtime.uses("memory_pull")
        ):
            try:
                tools.extend(await self.harness._pull_tools(runtime.run_memory))
            except Exception as exc:
                runtime.events.warning("memory_unavailable", f"no memory tools: {exc}")
        return [t for t in tools if t.feature is None or runtime.uses(t.feature)]

    async def native_tools(self, runtime: Runtime) -> dict[str, ToolSpec]:
        """The tools the framework gives the model itself (the adapter's ``native``), less the
        run's harness tools — with them the run's tool set — published to the tenant's catalog
        (an administrator's risk and rules govern them as any tool's), once per set of names
        per tenant. One that cannot be read is a warning: the run goes on with its harness
        tools."""
        try:
            specs = await self.adapter.native(self.target)
        except Exception as exc:
            runtime.events.warning("native_tools", f"the framework's own tools: {exc}")
            return {}
        found = {s.name: s for s in specs if s.name not in runtime.toolbox}
        names = tuple(found)
        if found and self._published_native.get(runtime.tenant) != names:
            governance = self.harness._governance(runtime.tenant)
            await governance.publish(list(found.values()))
            self._published_native[runtime.tenant] = names
        return found

    def _toolbox(self, tenant: str) -> Toolbox:
        box = self._toolboxes.get(tenant)
        if box is None:
            box = self._toolboxes[tenant] = self.harness._toolbox(
                self.sources, tenant=tenant, without=self.without, agent_id=self.id
            )
        return box

    async def push(self, runtime: Runtime) -> PromptContext | None:
        """What is pushed into the framework's input, in the runtime's context: the memory
        context (:meth:`remembered`), returned when there is one — or, with memory off, the
        conversation's earlier turns (:meth:`recalled`)."""
        if runtime.run_memory is None:
            await self.recalled(runtime)
            return None
        return await self.remembered(runtime)

    async def recalled(self, runtime: Runtime) -> None:
        """With memory off, the conversation's earlier turns from the run store
        (``conversation.earlier``: each one's question and answer) — only where nothing else
        keeps them (ADR 0005): not where the framework keeps its own thread (a checkpointer,
        a session of the run's, a conversation on OpenAI's servers) or continues the session
        it kept for the conversation's last turn (``Runtime.continued``). A target that takes
        messages (``Adapter.history_in == "messages"``) gets them as its own (a function as
        ``agent.history``); the others in the context's text."""
        store = self.adapter.thread_store(self.target, runtime.framework_options)
        if runtime.continued or (store is not None and store is not history.MEMORY_THREAD):
            return
        try:
            earlier = await conversation.earlier(runtime)
        except Exception as exc:
            runtime.events.warning("earlier_turns", f"the conversation's earlier turns: {exc}")
            return
        if not earlier:
            return
        if self.adapter.history_in == "messages":
            runtime.history = earlier
        else:
            runtime.context = conversation.rendered(earlier)

    async def remembered(self, runtime: Runtime) -> PromptContext | None:
        """The memory context for this run, in the runtime. Asked with the run's tool set —
        the harness's tools and the ones its framework gives the model itself
        (``Runtime.native``: an OpenAI Agents agent's own, hosted, MCP and handoff tools,
        Claude's built-ins and MCP tools, a graph's tool nodes', a Deep Agent's sub-agents') —
        it carries the skills the agent learned for the task, and - from five tools, the
        service's threshold - the tools that fit, whose candidates narrow the tools the model is
        offered (the ``hints`` feature), the framework's own among them where the framework
        lets them be narrowed. The tenant's first hints wait for its tools to be indexed
        (``Toolbox.indexed``); hints below :data:`NARROW_FLOOR` narrow nothing; the tools
        behind Code Mode are asked about too, and a hint naming one keeps the meta-tools that
        reach it; a hinted tool keeps its chain (``Toolbox.related``: the tools its output
        feeds, and those whose output it needs). A failure is a warning."""
        memory = runtime.run_memory
        if memory is None or not runtime.task or not runtime.uses("memory_push"):
            return None
        own = runtime.tool_set()
        hints = runtime.uses("hints")
        box = self._toolbox(runtime.tenant)
        behind = box.hidden(code_mode=runtime.uses("code_mode")) if hints else []
        if hints:
            await box.indexed()
        window = self.window(runtime)
        with retrieval_span(runtime.task) as span:
            try:
                pushed = await memory.context(
                    runtime.task,
                    tools=[*own, *behind] or None,
                    window=window,
                    hints=hints,
                    budget=self.context_budget,
                    document_ids=runtime.document_ids,
                )
            except Exception as exc:
                runtime.events.warning("memory_unavailable", f"no memory context: {exc}")
                return None
            output(span, pushed.rendered)
        stale: list[str] = []
        if window is Window.MESSAGES:
            runtime.history, stale = snapshots.out_of_date(
                from_window(pushed.conversation.messages if pushed.conversation else []),
                lambda name: _max_age(runtime, name),
            )
        # candidates the model is offered — the hints, and the tools the context names (the
        # plan of a skill learned for the task lists its steps); no hint, or none confident
        # enough, narrows nothing
        candidates = [n for n in pushed.tool_names if n in runtime.toolbox or n in runtime.native]
        best = max((t.confidence for t in pushed.tools or ()), default=0.0)
        if hints and candidates and best >= NARROW_FLOOR and self.adapter.narrows != "none":
            offered = set(candidates) | named_in(pushed.rendered, runtime.tool_set())
            # a candidate's chain stays whole: the tools its output feeds, and those giving
            # what it needs (``tools.links``, from the tools' schemas)
            chain = box.related(offered, code_mode=runtime.uses("code_mode"))
            offered |= chain & set(runtime.toolbox)
            if set(pushed.tool_names) & set(behind):  # reached through the meta-tools
                offered |= {n for n, t in runtime.toolbox.items() if t.feature == "code_mode"}
            runtime.offered = offered
        if stale and runtime.offered is not None:
            runtime.offered |= set(stale)  # a tool whose earlier values are out of date: again
        status = pushed.evidence_status
        runtime.evidence = status
        runtime.pushed = {
            "bundle_id": pushed.bundle_id,
            "memory_ids": pushed.memory_ids[:CITED_MAX],
        }
        # a question a hinted tool answers is no question memory must answer
        note = None if hints and best >= NARROW_FLOOR else ABSTAIN_NOTES.get(status)
        parts = (pushed.rendered, note, snapshots.note(stale))
        rendered = "\n\n".join(part for part in parts if part)
        runtime.context = rendered or None
        runtime.events.emit(
            RunEventType.CONTEXT_LOADED,
            data={"chars": len(rendered), "evidence_status": status},
        )
        return pushed

    def unreached(self, runtime: Runtime) -> None:
        """The memory context pushed for ``runtime`` reached nothing: the adapter found no
        place for it in the input (a function's text, a graph's input with no messages) and
        the run's own code never read it (``agent.context``). Said once per agent, naming the
        fix (:data:`UNREACHED_FIX`); the context is left as it is, never put elsewhere."""
        if self._unreached_said or runtime.context_read:
            return
        self._unreached_said = True
        message = (
            f"{self.id}: the memory context pushed for its runs reaches neither its input nor "
            f"its code, so the agent answers without it: {UNREACHED_FIX[self.adapter.name]}"
        )
        log.warning("%s", message)
        runtime.events.warning("memory_context", message)

    def window(self, runtime: Runtime) -> Window:
        """How this run's context carries the thread's recent messages: not at all where the
        framework owns the thread (a durable store, ``history.owner``) or continues the session
        it kept for the conversation's last turn (``Runtime.continued``: that session replays
        the turns, ADR 0005 §4); as the framework's own
        messages where memory owns it, the framework keeps no store and replays messages; in
        the context's text otherwise (a store kept in RAM replays the thread itself while the
        process lives: the window is never given twice)."""
        if runtime.continued:
            return Window.NONE  # the continued session holds the turns
        store = self.adapter.thread_store(self.target, runtime.framework_options)
        if self.owner(runtime, store) is history.Owner.FRAMEWORK or store is history.MEMORY_THREAD:
            return Window.NONE  # the framework's store holds it, or reads memory's thread itself
        if store is None and self.adapter.replays_history:
            return Window.MESSAGES
        return Window.RENDERED

    def owner(self, runtime: Runtime, store: Any = ...) -> history.Owner:
        """Who owns the run's conversation (``history.owner``) — of ``store`` when it was read."""
        if store is ...:
            store = self.adapter.thread_store(self.target, runtime.framework_options)
        return history.owner(store, agent_id=self.id)

    async def record_tool(self, runtime: Runtime, call: ToolCall, outcome: ToolOutcome) -> None:
        """A tool call, recorded in the memory service under the run — or, made while a helper
        the framework spawned works (``bridges.helpers.inside``), under the helper's run."""
        helper = helpers.inside(runtime)
        memory = helper.memory if helper is not None else runtime.run_memory
        if memory is not None and runtime.uses("records") and call.tool not in _pull(self):
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

    async def recorded_run(self, runtime: Runtime, messages: Sequence[Said]) -> None:
        """The attempt's transcript, whether the run succeeded, paused or failed. Where memory
        owns the conversation (``history.owner``), it is written before the run returns, so
        the next turn's window holds it wherever that turn runs; a write that fails there is
        queued like every other (:mod:`trellis.harness.writes`: retried, spooled)."""
        memory = runtime.run_memory
        if memory is None or not runtime.uses("records") or not messages:
            return
        # each tool result says which tool returned it and when: a later turn replays it so
        messages = snapshots.stamped(messages, runtime.fetched)
        run_id, attempt = runtime.run_id, runtime.attempt
        if self.owner(runtime) is history.Owner.MEMORY:
            try:
                async with asyncio.timeout(TURN_WRITE_SECONDS):
                    await memory.record_messages(messages, run_id, attempt)
                return
            except Exception as exc:
                log.warning(
                    "the turn could not be written before the run returned; it is queued",
                    extra={"run_id": run_id, "error_class": type(exc).__name__},
                )
        await self.harness.writes.submit(
            "memory.transcript",
            lambda: memory.record_messages(messages, run_id, attempt),
            events=runtime.events,
            record=memory.record(
                "record_messages",
                messages=[recorded(m) for m in messages],
                run_id=run_id,
                attempt=attempt,
            ),
        )

    async def recorded_outcome(self, runtime: Runtime, status: RunStatus, note: str | None) -> None:
        """How the run ended, as the run's ``system`` feedback: the lowest-ranked voice on
        its outcome (the judge's and a person's override it in the memory service). Its
        ``confirm`` is unverified, so the service keeps it as the outcome only: it moves no
        memory's confidence, learned skill or tool statistic. It keeps what the run's context
        showed (``Runtime.pushed``, under :data:`PUSHED_BUNDLE` and :data:`PUSHED_MEMORIES`):
        the memories a person's verdict on the run cites when no grounding verdict did
        (``h.feedback``)."""
        memory = runtime.run_memory
        verdict = OUTCOME_VERDICTS.get(status)
        if memory is None or not runtime.uses("records") or verdict is None:
            return
        feedback = {
            "verdict": verdict,
            "source": "system",
            "comment": note,
            "key": f"{runtime.run_id}:outcome",
            # what its context showed, which a person's verdict on the run cites
            "metadata": {
                PUSHED_BUNDLE: runtime.pushed["bundle_id"],
                PUSHED_MEMORIES: runtime.pushed["memory_ids"],
            }
            if runtime.pushed
            else None,
        }
        await self.harness.writes.submit(
            "memory.outcome",
            lambda: memory.run_feedback(**feedback),
            events=runtime.events,
            record=memory.record("run_feedback", **feedback),
        )

    async def grounded(self, runtime: Runtime, answer: Any, pushed: PromptContext | None) -> None:
        """On a sampled run, the answer (:func:`graded`) checked against the context it was
        given (the memory service's ``/v1/verify``, which records it as the run's ``judge``
        feedback), and the score put on the run's trace. Only such a verified run, and a
        person's reviewed verdict (``h.feedback``), teach the memory service: a run's own
        ``confirm`` (:meth:`recorded_outcome`) is its outcome only."""
        memory = runtime.run_memory
        text = graded(answer)
        if (
            memory is None
            or pushed is None
            or not text
            or not runtime.uses("grounding")
            or not sampled(runtime.run_id, GROUNDING_SAMPLE)
        ):
            return
        bundle_id, run_id, traced = pushed.bundle_id, runtime.run_id, runtime.trace_run

        async def work() -> None:
            score = await grounding_score(memory.ctx, text, bundle_id)
            if score is not None:
                key = f"{run_id}:grounding"
                await self.harness._score(run_id, "grounding", score, key=key, trace=traced)

        await self.harness.writes.submit("memory.verify", work, events=runtime.events)

    async def judged(self, runtime: Runtime, answer: Any, pushed: PromptContext | None) -> None:
        """Each of the agent's online judges (``h.wrap(judges=)``) on the share of runs it
        samples (``llm_judge(sample=)``, chosen by the run id), in the background — never on the
        request path — as ``judge(case, [it], services=self.evals)``, its score on the run's
        trace; the case's output is the answer as :func:`graded`. A judge that fails is a
        warning."""
        text = graded(answer)
        if not text or not runtime.uses("judges"):
            return
        chosen = [
            e for e in self.judges if sampled(f"{runtime.run_id}:judge:{name_of(e)}", sample_of(e))
        ]
        if not chosen:
            return
        memory = runtime.run_memory
        case = EvalCase(
            input=runtime.task,
            output=text,
            run_id=runtime.run_id,
            trace_id=trace_hex(runtime.trace_run),
            bundle_id=pushed.bundle_id if pushed is not None else None,
            context=runtime.pushed_context,
            memory=memory.ctx if memory is not None else None,
            trajectory=list(runtime.replay.journal.trajectory),
        )
        services, events = self.evals, runtime.events
        dims = {**telemetry.attribution(runtime), "purpose": "judge"}
        for evaluator in chosen:
            name = name_of(evaluator)

            async def work(evaluator: Evaluator = evaluator, name: str = name) -> None:
                with telemetry.attributed(dims):  # the judge's calls: whose run they judge
                    _, failed = await judge(case, [evaluator], services=services)
                if failed:
                    events.warning("judge_failed", f"judge {name}: {failed[name]}")

            await self.harness.writes.submit(f"judge.{name}", work, events=events)

    async def imported_code_mode_calls(self, runtime: Runtime) -> None:
        """The calls the run's Code Mode scripts made, read back from the gateway's log and
        recorded in the order they ran, each at the step of the script that made it
        (:func:`script_step`): a learned procedure takes its steps in that order, a tool the
        run called after the script after them."""
        memory, gateway = runtime.run_memory, self.harness.gateway
        if memory is None or gateway is None or not runtime.uses("records"):
            return
        run_id, since, task = runtime.run_id, runtime.started_at or datetime.now(UTC), runtime.task
        scripts = list(runtime.code_mode_scripts)

        async def work() -> None:
            logged = await gateway.code_mode_calls(run_id, since)
            for entry in sorted(logged, key=lambda e: (e.timestamp, e.id)):
                call = ToolCall(
                    tool=entry.name,
                    args=entry.arguments if isinstance(entry.arguments, dict) else {},
                    task=task,
                    step=script_step(scripts, entry.timestamp),
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
        timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
        deadline: datetime | None = None,
        budget: Budget | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
        document_ids: Sequence[str] | None = None,
        parent: Runtime | None = None,
        mode: Mode = Mode.QUEUE,
    ) -> RunRecord:
        """Record an in-process run as started; its record, which its first attempt starts
        from (``pipeline.attempt``) — with its framework options as given, where the stored
        record keeps their JSON. (Surfaces pass their own ``run_id`` when the protocol names
        the run.)"""
        start = await self._start(
            input,
            user=user,
            thread=thread,
            tenant=tenant,
            run_id=run_id,
            record_input=True,
            timeout=timeout,
            deadline=deadline,
            budget=budget,
            without=without,
            framework_options=framework_options,
            document_ids=document_ids,
            parent=parent,
        )
        live = None
        if mode is Mode.STEER and parent is None and thread:
            live = self.turns.live(conversation_key(start.tenant_id, user, thread))
        if live is not None and live.steers:  # its answer may be this run's (``earlier``)
            given = {**start.metadata, STEERED_INTO: live.run_id}
            start = start.model_copy(update={"metadata": given})
        await admitted(self.harness.runs.start(start), "start")
        record = RunRecord.from_start(start)
        if not framework_options:
            return record
        given = {**record.metadata, FRAMEWORK_OPTIONS: dict(framework_options)}
        return record.model_copy(update={"metadata": given})

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
        budget: Budget | None = None,
        parent: Runtime | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
        priority: int = 0,
        concurrency_key: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RunStart:
        """``record_input`` keeps an in-process run's input as JSON (it may be any object);
        a queued run's input already is. The run's time limit (else the agent's), deadline,
        its budget, the agent's version, what it is ``without``, its framework options that are
        JSON, the
        documents its context draws on (``document_ids``), and how a queued run waits its turn
        (``priority``, ``concurrency_key``) go with it when there are any. A run started
        inside another (``parent``: a sub-agent's, a nested ``run``/``stream``/``start``) is
        its child: its record names the parent and the agent group of the
        tree, and it takes the parent's tenant, thread, documents, deadline (the earlier of
        both) and what the parent is without, unless it names its own (``without`` adds to the
        parent's)."""
        if not user:
            raise ConfigurationError("a run is for somebody: pass user=")
        run_id = run_id or new_id("run_")
        turned_off = features(without)
        if parent is not None:
            tenant = tenant or parent.tenant
            thread = thread or parent.thread
            turned_off |= parent.without
            document_ids = parent.document_ids if document_ids is None else document_ids
            ends = [d for d in (deadline, parent.deadline) if d is not None]
            deadline = min(ends) if ends else None
        metadata = self._metadata(turned_off, framework_options, document_ids, budget)
        if parent is not None:
            metadata[AGENT_GROUP] = parent.identity.own_group
        given = {
            "timeout_seconds": self.timeout if timeout is None else timeout,
            "deadline": deadline,
            "agent_version": self.version,
            "parent_run_id": None if parent is None else parent.run_id,
            "priority": priority or None,
            "concurrency_key": concurrency_key,
        }
        return RunStart(
            run_id=run_id,
            tenant_id=await self.harness._tenant(tenant),
            agent_id=self.id,
            thread_id=thread or run_id,  # a run with no conversation is its own thread
            user_id=user,
            input=pipeline.jsonable(input) if record_input else input,
            metadata=metadata,
            **{name: value for name, value in given.items() if value is not None},
        )

    def _metadata(
        self,
        without: frozenset[Feature],
        framework_options: Mapping[str, Any] | None,
        document_ids: Sequence[str] | None = None,
        budget: Budget | None = None,
    ) -> dict[str, Any]:
        """What a run keeps of its own choices (``RunStart.metadata``; a schedule's, copied
        into each run it fires): the parts it is ``without``, its framework options — checked
        against the adapter, over the agent's — that are JSON, the documents its context
        draws on, and its own budget."""
        own = dict(framework_options or {})
        if own:
            self.adapter.check_options(merged(self.framework_options, own))
        metadata: dict[str, Any] = {WITHOUT: sorted(without)} if without else {}
        kept = {key: value for key, value in own.items() if _json(value)}
        if kept:
            metadata[FRAMEWORK_OPTIONS] = kept
        if document_ids is not None:
            metadata[DOCUMENTS] = documents(document_ids)
        if budget is not None:
            metadata[BUDGET] = budget.kept()
        return metadata


class Throttled(HarnessError):
    """agent-runs refused a call with ``429`` (the tenant's rate limit) and its SDK already
    waited the ``Retry-After`` it asked for, as often as it retries: try again after
    ``retry_after`` seconds (``details["retry_after"]``). Not retried by the harness — one
    retry layer, the SDK's."""

    code = "RUNS_RATE_LIMITED"
    category = ErrorCategory.RATE_LIMIT
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None) -> None:
        super().__init__(message, details={"retry_after": retry_after}, source="agent-runs")
        self.retry_after = retry_after


async def admitted[T](call: Awaitable[T], operation: str) -> T:
    """``call`` to the run store; a ``429`` it still answers after the SDK's retries is a
    :class:`Throttled`, counted (``trellis.runs.rate_limited``)."""
    try:
        return await call
    except RateLimitedError as exc:
        metrics.rate_limited(operation)
        after = f" after {exc.retry_after:g}s" if exc.retry_after is not None else " later"
        raise Throttled(
            f"agent-runs is rate limiting this tenant ({operation}): try again{after}",
            retry_after=exc.retry_after,
        ) from exc


#: The run's ``system`` feedback verdict for how it ended (a cancelled run says nothing
#: about the agent).
OUTCOME_VERDICTS: Final = {RunStatus.SUCCESS: "confirm", RunStatus.ERROR: "reject"}


def script_step(scripts: Sequence[tuple[datetime, int | None]], at: datetime) -> int | None:
    """The step of the Code Mode script a call logged ``at`` ran in: of ``scripts`` (when each
    started, its step; in the order they started; at least one), the last one started by
    then, else the first (the gateway's clock a little behind the harness's)."""
    started = [step for when, step in scripts if when <= at]
    return started[-1] if started else scripts[0][1]


def graded(answer: Any) -> str:
    """An answer as the grounding check and the online judges grade it: its text, or — a
    structured answer (a pydantic model, a dict, a list) — its JSON; empty for no answer."""
    if answer is None or isinstance(answer, str):
        return answer or ""
    return json.dumps(pipeline.jsonable(answer), default=str)


def named_in(text: str, names: Collection[str]) -> set[str]:
    """The tool ``names`` that ``text`` names, as whole words (a learned skill's steps in the
    memory context, a hint's line): one pass over the text, whatever the number of tools."""
    found = set(re.findall(r"[A-Za-z0-9_.:-]+", text or ""))
    return {name for name in names if name in found}


def gated_tools(target: Any) -> set[str]:
    """The tools a target's own approval gate holds: LangChain's
    ``HumanInTheLoopMiddleware`` (Deep Agents' ``interrupt_on``) in a compiled graph, an
    OpenAI Agents tool with ``needs_approval``."""
    gated: set[str] = set()
    for node in (getattr(target, "nodes", None) or {}).values():
        owner = getattr(getattr(getattr(node, "bound", None), "func", None), "__self__", None)
        gates = getattr(owner, "interrupt_on", None)
        if type(owner).__name__ == "HumanInTheLoopMiddleware" and isinstance(gates, dict):
            gated |= {name for name, config in gates.items() if config}
    for tool in getattr(target, "tools", None) or []:
        if getattr(tool, "needs_approval", False) and isinstance(getattr(tool, "name", None), str):
            gated.add(tool.name)
    return gated


def _kept_as_json(framework_options: Mapping[str, Any] | None) -> None:
    """A queued or scheduled run keeps its framework options with it: each is JSON."""
    objects = [key for key, value in (framework_options or {}).items() if not _json(value)]
    if objects:
        raise ConfigurationError(
            f"a queued run keeps its framework_options as JSON, and "
            f"{', '.join(map(repr, objects))} is not: give it on h.wrap(framework_options=), "
            "which every worker's agent has"
        )


def _json(value: Any) -> bool:
    """Whether ``value`` is JSON (what a run's record keeps of it)."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _max_age(runtime: Runtime, name: str) -> float | None:
    """How long a value of the run's tool ``name`` stays current, as it declared it (``None``:
    it did not say, or it is not the run's tool)."""
    found = runtime.toolbox.get(name)
    spec = found.spec if found is not None else runtime.native.get(name)
    return None if spec is None else max_age(spec)


def _pull(agent: Agent) -> frozenset[str]:
    """The memory service's own tools are recorded by the service, not again by the harness."""
    listed = agent.harness.memory.listed if agent.harness.memory is not None else None
    return frozenset(t.name for t in listed or ())


class RunHandle:
    """A queued run: its ``id``, where it is (:meth:`status`), and a way to wait for it
    (:meth:`result`). Cancel it with ``agent.cancel(handle.id)``."""

    def __init__(self, agent: Agent, run_id: str, *, tenant: str) -> None:
        self._agent = agent
        self.id = run_id
        self._tenant = tenant

    async def status(self) -> RunRecord:
        record = await self._agent.harness.runs.get(self.id, tenant=self._tenant)
        if record is None:
            raise ConfigurationError(f"no run {self.id}")
        return record

    async def result(self, *, wait: float | None = None) -> Result:
        """Wait until the run pauses or ends (a worker must be running it), at most ``wait``
        seconds (``None``: as long as it takes; past it, ``TimeoutError``). A poll the run
        store refuses for the tenant's request budget (429) waits what it says, then polls
        again."""
        async with asyncio.timeout(wait):
            while True:
                try:
                    record = await self.status()
                except RateLimitedError as exc:
                    await asyncio.sleep(exc.retry_after or POLL_SECONDS * 4)
                    continue
                if record.status is RunStatus.PAUSED or record.final:
                    return self._agent.hooks.answer(Result.of(record))
                await asyncio.sleep(POLL_SECONDS)


__all__ = ["Agent", "RunHandle", "Throttled"]
