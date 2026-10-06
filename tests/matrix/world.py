"""One cell's world: the deployment a selection describes (which blocks are on), the agent
under test built for its adapter and way, a run driven in its mode, and what everything saw.

* **Blocks.** The memory service, the gateway, the online judges, the grounding sample,
  tracing, the agent's default time limit and its version are on exactly when the selection
  says so (``World.switched``). The fakes are the suite's own (``tests/support``): every memory
  request is checked against the memory service's OpenAPI document; the run store is the
  in-process one, bounding checkpoints as agent-runs does.
* **Targets.** ``tests.support.adapters.BUILDERS``: each adapter follows a plan of tool calls
  with a scripted model and answers ``Done. <last result>``.
* **Modes.** :meth:`World.go` starts the run in the cell's mode, answers each pause with the
  scenario's ``answer`` (the same way a person would on that surface) and returns the
  :class:`Outcome` — the run's record from the store, its events (every event the harness
  emitted for it, tapped at the source, in every mode), the questions it asked.
* **Off leaves no trace.** :meth:`World.verify` checks, for every switch, that an off switch
  left nothing (no calls, records, spans, events) and an on switch did its part, for every run
  the scenario made; and that each run's events are well formed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
from a2a.client import A2ACardResolver, Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.types import CancelTaskRequest, Role, SendMessageRequest, StreamResponse
from fastapi import FastAPI
from langchain.agents import create_agent
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.matrix.dimensions import SWITCHES
from tests.matrix.model import Feature, Selection
from tests.matrix.models import ModelLog
from tests.support.adapters import BUILDERS
from tests.support.catalog import FakeCatalog
from tests.support.gateway import URL as GATEWAY_URL
from tests.support.gateway import FakeGateway
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from tests.support.planned import Call, PlannedChatModel
from trellis import Harness, Settings, tool
from trellis.contracts import (
    Interrupt,
    InterruptReason,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStatus,
    ToolCall,
    new_id,
)
from trellis.harness import pipeline, telemetry
from trellis.harness.a2a.identity import EXTENSION_URI
from trellis.harness.agent import Agent
from trellis.harness.agui.sse import decode
from trellis.harness.evals import EvalCase, EvalScore
from trellis.harness.events import RunEvents
from trellis.harness.features import features
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.hooks import Hooks
from trellis.harness.identity import identity_headers
from trellis.harness.journal import MAX_CHECKPOINT_BYTES
from trellis.harness.result import Result
from trellis.harness.runs import LocalRuns
from trellis.harness.runtime import Runtime
from trellis.harness.tools import base
from trellis.runs import Lease, PayloadTooLargeError
from trellis.runs import Worker as RunsWorker

USER: Final = "ada"
#: The agent's version and default time limit when those switches are on.
VERSION: Final = "v-matrix"
AGENT_TIMEOUT: Final = 60.0
#: The MCP tool every key with the gateway on reaches (``<server>-<tool>``).
KEY_TOOL: Final = "kb-search"
#: The approval ``elsewhere`` asks first, so every scenario pauses once and is continued by
#: another process.
CONFIRM: Final[Call] = ("confirm", {"step": "start"})
A2A_URL: Final = "http://a2a.matrix/agents/under-test"
#: How long a run may take in a cell before the cell fails (nothing here waits on purpose).
RUN_LIMIT_SECONDS: Final = 60.0
#: A run that pauses more often than this is looping.
MAX_PAUSES: Final = 6

Answer = tuple[str, Any]
Answering = Callable[[Interrupt], Answer]
#: Builds a target of one's own: (the harness, the tools, the plan) -> (target, tools to wrap)
Target = Callable[[Harness, list[Any], list[Any]], Awaitable[tuple[Any, list[Any]]]]


def approve_all(interrupt: Interrupt) -> Answer:
    """A person approving every call (and answering a question with ``yes``)."""
    if interrupt.reason is InterruptReason.APPROVAL:
        return "approve", None
    return "answer", "yes"


@tool(side_effects="irreversible")
def confirm(step: str) -> str:
    """Confirm a step before going on."""
    return f"confirmed {step}"


class MatrixGateway(FakeGateway):
    """The suite's fake gateway, also running the key's own MCP tools
    (``/v1/mcp/tool/execute``) and listing tools with an object schema, as MCP servers do."""

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/mcp/tool/execute":
            self.requests.append(request)
            call = json.loads(request.content)
            text = json.dumps(
                {
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
            )
            return httpx.Response(
                200, json={"role": "tool", "content": text, "tool_call_id": call.get("id")}
            )
        return await super().handle(request)

    def _rpc(self, slug: str, message: dict[str, Any]) -> httpx.Response:
        if message["method"] != "tools/list":
            return super()._rpc(slug, message)
        names = self.bundles.get(slug, [])
        tools = [{"name": n, "inputSchema": {"type": "object"}} for n in names]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}})


class Recorder(Hooks):
    """The hooks switch: a hook that only notes what it saw (on: every run and tool call)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    async def on_run_start(self, run: Runtime) -> None:
        self.seen.append(("start", run.run_id))

    async def on_run_end(self, run: Runtime, result: Result) -> None:
        self.seen.append(("end", run.run_id))

    async def before_tool(self, call: ToolCall) -> None:
        self.seen.append(("tool", call.tool))


class MatrixRuns(LocalRuns):
    """The in-process run store, refusing a checkpoint over agent-runs' bound as agent-runs
    does (so a journal larger than one goes to an artifact, as in production)."""

    async def heartbeat(
        self, run_id: str, worker_id: str, *, checkpoint: dict[str, Any] | None = None, **kw: Any
    ) -> Lease:
        _bounded(checkpoint)
        return await super().heartbeat(run_id, worker_id, checkpoint=checkpoint, **kw)

    async def pause(self, interrupt: Interrupt, **kw: Any) -> RunRecord:
        _bounded(kw.get("checkpoint"))
        return await super().pause(interrupt, **kw)


def _bounded(checkpoint: dict[str, Any] | None) -> None:
    if checkpoint is not None and len(json.dumps(checkpoint)) > MAX_CHECKPOINT_BYTES:
        raise PayloadTooLargeError("checkpoint too large", code="PAYLOAD_TOO_LARGE", status=413)


@dataclass
class Outcome:
    """How one run went: its record as the store has it at the end, its events (all
    attempts), the questions it asked on the way, and what its surface said."""

    run_id: str
    record: RunRecord
    events: list[RunEvent]
    pauses: list[Interrupt]
    surface: list[Any] = field(default_factory=list)
    #: the parts the run's own ``without=`` turned off
    without: frozenset[str] = frozenset()

    @property
    def status(self) -> RunStatus:
        return self.record.status

    @property
    def answer(self) -> Any:
        return self.record.output

    @property
    def text(self) -> str:
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)

    def succeeded(self) -> Outcome:
        assert self.status is RunStatus.SUCCESS, (self.status, self.record.error)
        return self

    def results(self, tool_name: str) -> list[dict[str, Any]]:
        """What each call of ``tool_name`` came to, as the run's events say."""
        return [
            e.data
            for e in self.events
            if e.type is RunEventType.TOOL_CALL_RESULT and e.data.get("tool") == tool_name
        ]

    def called(self) -> list[str]:
        return [
            e.data["tool"]
            for e in self.events
            if e.type is RunEventType.TOOL_CALL_START and "tool" in e.data
        ]

    def custom(self, name: str) -> list[dict[str, Any]]:
        return [
            e.data
            for e in self.events
            if e.type is RunEventType.CUSTOM and (e.data or {}).get("name") == name
        ]


@dataclass
class _Handle:
    """A run being driven: its id once known, and the task driving its current attempt."""

    agent: Agent
    process: int
    task: asyncio.Task[Any] | None = None
    run_id: str | None = None
    surface: list[Any] = field(default_factory=list)
    http: httpx.AsyncClient | None = None
    client: Client | None = None
    context_id: str | None = None
    thread: str | None = None


class World:
    """One cell (see the module)."""

    def __init__(
        self,
        *,
        feature: Feature,
        adapter: str,
        way: str,
        mode: str,
        selection: Selection,
        tmp: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.feature, self.adapter, self.way, self.mode = feature, adapter, way, mode
        self.selection = selection
        self.switched = selection.on
        self.tmp = tmp
        self.monkeypatch = monkeypatch
        #: whether the feature under test is on in this selection
        self.on = feature.needs <= self.switched or (
            way == "with_blocks" and feature.id in ON_WITH_BLOCKS
        )
        self.store = MatrixRuns()
        #: the parts the selection turns off while their service is on (``without=``)
        self.without = frozenset(
            sw.without
            for sw in SWITCHES
            if sw.without and sw.requires <= self.switched and sw.id not in self.switched
        )
        self.recorder = Recorder()
        self.memory_service = FakeMemoryService(candidates=[])  # no narrowing unless asked
        self.fake_gateway = MatrixGateway(bundles={"": [KEY_TOOL]})
        #: the team's own approval rules (``with_blocks``: its governance block)
        self.team_catalog = FakeCatalog()
        self.judged: list[EvalCase] = []
        self.tap: list[RunEvent] = []
        self.outcomes: list[Outcome] = []
        self.harnesses: list[Harness] = []
        #: other deployments a scenario made (a remote A2A agent's), closed with the world
        self.others: list[Harness] = []
        self.overrides: dict[str, Any] = {}
        self._agents = 0
        self._run_options: dict[str, Any] = {}
        self._current: _Handle | None = None
        self.exporter: InMemorySpanExporter | None = None
        self._tap(monkeypatch)
        #: what every scripted model was sent and offered
        self.models = ModelLog(monkeypatch)
        #: the tools each Claude run was given (its CLI lists them itself)
        self.claude_offered: list[list[str]] = []
        converted = pipeline.convert

        def converting(tool_format: Any, tools: Any) -> Any:
            if tool_format == "claude":
                self.claude_offered.append([t.name for t in tools])
            return converted(tool_format, tools)

        monkeypatch.setattr(pipeline, "convert", converting)
        monkeypatch.setattr(base, "RETRY_BACKOFF_SECONDS", 0.001)
        if "tracing" in self.switched:
            self.exporter = InMemorySpanExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(self.exporter))
            monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("matrix"))

    def _tap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every event the harness emits, for every run, in every mode."""
        original = RunEvents.__init__
        tap = self.tap

        def listening(events: RunEvents, *args: Any, **kwargs: Any) -> None:
            original(events, *args, **kwargs)
            events.listen(tap.append)

        monkeypatch.setattr(RunEvents, "__init__", listening)

    # ------------------------------------------------------------------ the deployment
    async def judge(self, case: EvalCase) -> EvalScore:
        """The online judge (on with ``judges``): it remembers what it was given."""
        self.judged.append(case)
        return EvalScore("matrix", 1.0)

    def harness(self, process: int = 0) -> Harness:
        """The deployment as process ``process`` (0, or 1 for ``elsewhere``) has it: the
        blocks the selection turns on, sharing one run store and one of each service."""
        while len(self.harnesses) <= process:
            self.harnesses.append(self._made())
        return self.harnesses[process]

    def _made(self) -> Harness:
        on = self.switched
        settings = Settings(
            grounding_sample=1.0 if "grounding" in on else 0.0,
            judge_sample=1.0 if "judges" in on else None,
            bifrost_virtual_key="vk" if "gateway" in on else None,
            bifrost_url=GATEWAY_URL if "gateway" in on else None,
        )
        memory = self.memory_service.client() if "memory" in on else False
        gateway = self.fake_gateway.gateway() if "gateway" in on else False
        governance: Governance | None = None
        if self.way == "with_blocks":
            governance = Governance(self.team_catalog)
        return Harness(
            config=settings,
            runs=self.store,
            memory=memory,
            gateway=gateway,
            governance=governance,
            judges=[self.judge] if "judges" in on else (),
            hooks=[self.recorder] if "hooks" in on and self.way == "with_blocks" else (),
        )

    def rules(self, rules: dict[str, Rule]) -> None:
        """Approval rules for the tools: the memory service's catalog (Way 1), the team's own
        governance block (the with_blocks way)."""
        self.memory_service.catalog = {
            name: {"side_effects": rule.risk, "approve_when": rule.approve_when}
            for name, rule in rules.items()
        }
        self.team_catalog.rules = dict(rules)

    async def aclose(self) -> None:
        for h in [*self.harnesses, *self.others]:
            await h.aclose()

    def contract_violations(self) -> list[str]:
        """What the memory service's document refused, taken (``tests/conftest.py`` would fail
        the test at teardown, where no xfail reaches)."""
        found = list(self.memory_service.violations)
        self.memory_service.violations.clear()
        return found

    async def drain(self) -> None:
        """Every background write delivered (memory records, judges, grounding)."""
        for h in self.harnesses:
            await h.writes.drain()

    # ------------------------------------------------------------------ the agent
    async def agent(
        self,
        tools: Sequence[Any],
        plan: Sequence[Call],
        *,
        process: int = 0,
        name: str | None = None,
        target: Target | None = None,
        mcp: Sequence[str] | None = None,
        parent: str | None = None,
        **wrap: Any,
    ) -> Agent:
        """The agent under test in ``process``: the adapter's target following ``plan``
        (or one ``target`` builds; ``parent`` names another adapter to build it with), wrapped
        with the switched blocks. ``name`` reuses an id (another process's copy of the same
        agent)."""
        h = self.harness(process)
        adapter = parent or self.adapter
        if target is not None:
            built, own = await target(h, list(tools), list(plan))
        else:
            built, own = await self._built(adapter, h, list(tools), list(plan), mcp)
        if adapter not in FIXED and mcp is not None:
            wrap["mcp"] = list(mcp)
        for key in ("timeout", "version"):
            if key in wrap:
                self.overrides[key] = wrap[key]
        if "version" in self.switched:
            wrap.setdefault("version", VERSION)
        if "agent_timeout" in self.switched:
            wrap.setdefault("timeout", AGENT_TIMEOUT)
        if self.without:
            wrap["without"] = {*self.without, *wrap.get("without", ())}
        if "hooks" in self.switched and self.way != "with_blocks":
            wrap["hooks"] = [*wrap.get("hooks", ()), self.recorder]
        if name is None:
            self._agents += 1
            name = f"{self.feature.id.lower()}.{self.adapter}.{self._agents}"
        elif name in h.agents:
            return h.agents[name]  # this process wrapped it already
        return h.wrap(built, id=name, tools=own, **wrap)

    async def _built(
        self,
        adapter: str,
        h: Harness,
        tools: list[Any],
        plan: list[Call],
        mcp: Sequence[str] | None,
    ) -> tuple[Any, list[Any]]:
        adapter = "claude_agent_sdk" if adapter == "claude" else adapter
        if adapter in FIXED and mcp is not None:
            native = await h.tools(*tools, framework=adapter, mcp=list(mcp))  # type: ignore[arg-type]
            model = PlannedChatModel(plan=plan)
            if adapter == "langgraph":
                return create_agent(model, tools=native), []
            from deepagents import create_deep_agent

            return create_deep_agent(model=model, tools=native), []
        return await BUILDERS[adapter](h, tools, self.tmp, plan)

    # ------------------------------------------------------------------ driving a run
    async def go(
        self,
        tools: Sequence[Any],
        plan: Sequence[Call],
        *,
        input: str = "do the task",
        answer: Answering = approve_all,
        mcp: Sequence[str] | None = None,
        target: Target | None = None,
        parent: str | None = None,
        during: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        run: dict[str, Any] | None = None,
        **wrap: Any,
    ) -> Outcome:
        """Run the agent under test once, in the cell's mode, to its end: each pause answered
        with ``answer`` (by another process, in ``elsewhere``). ``during`` is called with the
        run's id while its first attempt is under way (to cancel it). ``run`` holds the run's
        own options (``timeout=``, ``without=``: ``run``, ``stream`` and ``start`` take them)."""
        tools, plan = list(tools), list(plan)
        if self.mode == "elsewhere":
            tools, plan = [*tools, confirm], [CONFIRM, *plan]
        agent = await self.agent(tools, plan, mcp=mcp, target=target, parent=parent, **wrap)
        self._run_options = dict(run or {})
        if "timeout" in self._run_options:
            self.overrides["timeout"] = self._run_options["timeout"]
        handle = self._current = await self._start(agent, input)
        watching = None
        if during is not None:
            watching = asyncio.create_task(during(await self._run_id(handle)))
        record = await self._settled(handle)
        pauses: list[Interrupt] = []
        while record.status is RunStatus.PAUSED:
            assert record.awaiting is not None
            if len(pauses) >= MAX_PAUSES:
                raise AssertionError(f"run {record.run_id} keeps pausing: {pauses}")
            interrupt = record.awaiting
            pauses.append(interrupt)
            injected = self.mode == "elsewhere" and len(pauses) == 1
            decision, value = ("approve", None) if injected else answer(interrupt)
            if self.mode == "elsewhere":  # the other process answers and continues it
                process = 1 - handle.process
                other = await self.agent(
                    tools,
                    plan,
                    process=process,
                    name=agent.id,
                    mcp=mcp,
                    target=target,
                    parent=parent,
                    **wrap,
                )
                handle = self._current = _Handle(other, process, run_id=handle.run_id)
            await self._resume(handle, interrupt, decision, value)
            record = await self._settled(handle)
        if watching is not None:
            async with asyncio.timeout(RUN_LIMIT_SECONDS):
                await watching
        await self.drain()
        run_id = record.run_id
        events = [e for e in self.tap if e.run_id == run_id]
        if injected_pause(self.mode, pauses):
            pauses = pauses[1:]
        outcome = Outcome(run_id, record, events, pauses, handle.surface)
        outcome.without = features(self._run_options.get("without", ()))
        self.outcomes.append(outcome)
        return outcome

    async def _run_id(self, handle: _Handle) -> str:
        async with asyncio.timeout(RUN_LIMIT_SECONDS):
            while handle.run_id is None:
                started = [
                    e
                    for e in self.tap
                    if e.type is RunEventType.RUN_STARTED
                    and (e.data or {}).get("agent_id") == handle.agent.id
                ]
                if started:
                    handle.run_id = started[-1].run_id
                    break
                await asyncio.sleep(0.001)
        return handle.run_id

    async def _settled(self, handle: _Handle) -> RunRecord:
        """The run's record once its current attempt is over (paused or ended)."""
        assert handle.task is not None
        async with asyncio.timeout(RUN_LIMIT_SECONDS):
            done = await handle.task
        if self.mode == "agui":
            response: httpx.Response = done
            assert response.status_code == 200, response.text
            handle.surface.extend(decode(response.text))
        elif self.mode == "a2a":
            handle.surface.extend(done)
            handle.run_id = handle.run_id or _task_id(done)
            handle.context_id = handle.context_id or _context_id(done)
        elif self.mode == "stream":
            handle.surface.extend(done)
            if done and handle.run_id is None:
                handle.run_id = done[-1].run_id
        elif self.mode == "run":
            handle.run_id = handle.run_id or done.run_id
        elif self.mode == "schedule" and handle.run_id is None:
            handle.run_id = self._scheduled_run()
        assert handle.run_id is not None, "the run never started"
        record = await self.store.get(handle.run_id)
        assert record is not None
        if record.status in (RunStatus.QUEUED, RunStatus.RUNNING):
            raise AssertionError(f"run {record.run_id} is still {record.status.value}")
        return record

    def _scheduled_run(self) -> str | None:
        fired = [r for r in self.store._runs.values() if r.metadata.get("schedule_id")]
        return fired[-1].run_id if fired else None

    async def _start(self, agent: Agent, input: str) -> _Handle:
        handle = _Handle(agent, 0)
        mode = self.mode
        if mode == "run":
            handle.task = asyncio.create_task(agent.run(input, user=USER, **self._run_options))
        elif mode == "stream":
            handle.task = asyncio.create_task(_streamed(agent, input, self._run_options))
        elif mode in ("worker", "elsewhere"):
            started = await agent.start(input, user=USER, **self._run_options)
            handle.run_id = started.run_id
            handle.task = asyncio.create_task(self._work(agent))
        elif mode == "schedule":
            schedule = await agent.schedule("0 0 1 1 *", input, on_behalf_of=USER)
            self.store._schedules[schedule.schedule_id] = schedule.model_copy(
                update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
            )
            handle.task = asyncio.create_task(self._work(agent))
        elif mode == "agui":
            app = FastAPI()
            agent.serve_chat(app, identity=lambda request: USER)
            handle.http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agui.matrix"
            )
            handle.run_id = new_id("run_")
            handle.thread = f"thread-{handle.run_id}"
            body = {
                "threadId": handle.thread,
                "runId": handle.run_id,
                "messages": [{"id": "m1", "role": "user", "content": input}],
            }
            handle.task = asyncio.create_task(handle.http.post("/agui/run", json=body))
        elif mode == "a2a":
            app = FastAPI()
            agent.serve_a2a(app, A2A_URL)
            handle.http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://a2a.matrix"
            )
            card = await A2ACardResolver(handle.http, A2A_URL).get_agent_card()
            handle.client = ClientFactory(ClientConfig(httpx_client=handle.http)).create(card)
            tenant = await agent.harness.tenant()
            handle.task = asyncio.create_task(_sent(handle.client, tenant, new_text_part(input)))
        else:  # pragma: no cover - the modes are fixed
            raise AssertionError(mode)
        return handle

    async def _work(self, agent: Agent) -> None:
        """The process's worker: claims and executes until nothing is queued
        (``with_blocks``: the team's own ``trellis.runs.Worker`` around
        ``agent.execute``)."""
        h = agent.harness
        if self.way == "with_blocks":
            loop = RunsWorker(self.store, agent.execute, [agent.id])
            while await loop.run_once():
                pass
            return
        worker = h.worker([agent])
        while await worker.run_once():
            pass

    async def _resume(
        self, handle: _Handle, interrupt: Interrupt, decision: str, value: Any
    ) -> None:
        agent, mode = handle.agent, self.mode
        if mode in ("run", "stream"):
            handle.task = asyncio.create_task(
                agent.resume(interrupt.interrupt_id, decision, answer=value, reviewer=USER)
            )
            if mode == "stream":
                resumed = handle.task
                handle.task = asyncio.create_task(_as_list(resumed))
        elif mode in ("worker", "elsewhere", "schedule"):
            queued = await agent.resume(
                interrupt.interrupt_id, decision, answer=value, reviewer=USER
            )
            assert queued.status in (RunStatus.QUEUED, RunStatus.CANCELLED), queued
            handle.task = asyncio.create_task(self._work(agent))
        elif mode == "agui":
            assert handle.http is not None
            payload: Any = value
            if decision == "approve" and value is None:
                payload = True
            entry = {"interruptId": interrupt.interrupt_id, "decision": decision.upper()}
            entry["payload"] = payload
            body = {"threadId": handle.thread, "messages": [], "resume": [entry]}
            handle.task = asyncio.create_task(handle.http.post("/agui/run", json=body))
        elif mode == "a2a":
            assert handle.client is not None and handle.run_id is not None
            tenant = await agent.harness.tenant()
            part = new_data_part({"decision": decision, "answer": value})
            handle.task = asyncio.create_task(
                _sent(
                    handle.client,
                    tenant,
                    part,
                    task_id=handle.run_id,
                    context_id=handle.context_id,
                )
            )

    async def cancel(self, run_id: str, reason: str) -> None:
        """Cancel the running run the way the cell's surface does: ``agent.cancel`` in
        process, A2A's ``tasks/cancel``, AG-UI's cancel route (it has none yet: G29)."""
        handle = self._current
        assert handle is not None
        if self.mode == "a2a":
            assert handle.client is not None
            tenant = await handle.agent.harness.tenant()
            headers = {HTTP_EXTENSION_HEADER: EXTENSION_URI, **identity_headers(tenant, USER)}
            context = ClientCallContext(service_parameters=headers)
            await handle.client.cancel_task(CancelTaskRequest(id=run_id), context=context)
            return
        if self.mode == "agui":
            assert handle.http is not None
            response = await handle.http.post(f"/agui/runs/{run_id}/cancel", json={})
            assert response.status_code < 300, f"AG-UI cancel: {response.status_code}"
            return
        await handle.agent.cancel(run_id, reason=reason)

    # ------------------------------------------------------------------ what was seen
    def claude_records(self) -> list[dict[str, Any]]:
        """What the scripted Claude CLI was started with, each time (its system prompt, the
        tools it may call, the prompt)."""
        return [json.loads(p.read_text()) for p in sorted(self.tmp.glob("cli-*.json"))]

    def said(self) -> str:
        """Everything the models under test were sent."""
        claude = json.dumps(self.claude_records())
        return self.models.said() + "\n" + claude

    def spans(self) -> list[ReadableSpan]:
        return list(self.exporter.get_finished_spans()) if self.exporter is not None else []

    def verify(self) -> None:
        """Off leaves no trace; on did its part; every run's events are well formed."""
        if self.way == "way2":
            return
        for outcome in self.outcomes:
            _well_formed(outcome)
            self._timeout_and_version(outcome.record)
            self._run_switches(outcome)
        self._memory()
        self._services()

    def _run_switches(self, outcome: Outcome) -> None:
        """One run: memory's parts, the gateway's tools, the judges, tracing and hooks as
        switched (the run's own ``without=`` turning more off)."""
        on, names = self.switched - outcome.without, set(outcome.called())
        loaded = [e for e in outcome.events if e.type is RunEventType.CONTEXT_LOADED]
        if {"memory", "memory_push"} <= on:
            assert loaded, f"memory push on: no context was pushed into {outcome.run_id}"
        else:
            assert not loaded, "memory push off: a context was pushed"
        if not {"memory", "memory_pull"} <= on:
            pulled = names & set(MEMORY_TOOLS)
            assert not pulled, f"memory pull off: memory tools were called: {pulled}"
        if not {"gateway", "mcp"} <= on:
            assert KEY_TOOL not in names, "the key's MCP tools off: one was called"
        succeeded = outcome.status is RunStatus.SUCCESS
        judged = [c for c in self.judged if c.run_id == outcome.run_id]
        if "judges" in on and succeeded and outcome.answer not in (None, ""):
            assert len(judged) == 1, f"judges on: run {outcome.run_id} judged {len(judged)}x"
        else:
            assert not judged, f"run {outcome.run_id} was judged though it should not be"
        if "tracing" in on:
            attempts = {e.attempt for e in outcome.events}
            spans = [
                s
                for s in self.spans()
                if s.name.startswith("invoke_agent")
                and (s.attributes or {}).get("langfuse.trace.metadata.run_id") == outcome.run_id
            ]
            assert len(spans) >= len(attempts), f"tracing on: {len(spans)} agent spans"
        hooked = {kind for kind, ref in self.recorder.seen if ref == outcome.run_id}
        if "hooks" in on:
            assert hooked == {"start", "end"}, f"hooks on: run {outcome.run_id} saw {hooked}"

    def _memory(self) -> None:
        """Memory on: what each part does, as switched; off: not a call."""
        on = self.switched
        if "memory" not in on:
            assert not self.memory_service.calls, "memory off: the memory service was called"
            return
        verified = len(self.memory_service.named("verify"))
        if "grounding" not in on:
            assert not verified, "grounding off: a run was verified"
        else:
            grounded = [
                o
                for o in self.outcomes
                if o.status is RunStatus.SUCCESS
                and any(e.type is RunEventType.CONTEXT_LOADED for e in o.events)
                and isinstance(o.answer, str)
                and o.answer
            ]
            assert verified >= len(grounded), f"grounding on: {verified} verified"
        recording = [o for o in self.outcomes if "records" not in o.without]
        recorded = self.memory_service.named("messages")
        if "records" in on and recording:
            assert recorded, "memory records on: no transcript was recorded"
        elif "records" not in on:
            stored = recorded + self.memory_service.named("record_tool")
            assert not stored, "memory records off: a transcript or a tool call was recorded"

    def _services(self) -> None:
        """The gateway, the judges, tracing and hooks: nothing when off."""
        on = self.switched
        listed = [r for r in self.fake_gateway.requests if r.url.path == "/mcp"]
        if "gateway" not in on:
            assert not self.fake_gateway.requests, "gateway off: the gateway was called"
        elif "mcp" not in on and listed:
            raise OffButCalled("the key's MCP tools off: they were listed")
        elif self.outcomes and self.feature.id not in NO_KEY_LISTING:
            assert listed, "gateway on: the key's MCP tools were never listed"
        if "judges" not in on:
            assert not self.judged, "judges off: a judge ran"
        if "tracing" not in on:
            assert not telemetry._tracer.start_span("probe").is_recording()
        if "hooks" not in on:
            assert not self.recorder.seen, "hooks off: a hook ran"

    def _timeout_and_version(self, record: RunRecord) -> None:
        """The agent's version and time limit are recorded with each run it starts,
        scheduled ones included."""
        on = self.switched
        if "version" not in self.overrides:
            expected = VERSION if "version" in on else None
            assert record.agent_version == expected, ("version", record.agent_version)
        if "timeout" not in self.overrides and record.parent_run_id is None:
            limit = AGENT_TIMEOUT if "agent_timeout" in on else None
            assert record.timeout_seconds == limit, ("timeout", record.timeout_seconds)


#: The adapters whose tools are bound when the target is built (``h.tools``).
FIXED: Final = frozenset({"langgraph", "deepagents"})
#: Features on with the team's blocks whatever the selection (its own governance's rules).
ON_WITH_BLOCKS: Final = frozenset({"F33"})
#: Features whose runs never list the key's MCP tools (only their Virtual MCPs).
NO_KEY_LISTING: Final = frozenset({"F18"})


def injected_pause(mode: str, pauses: list[Interrupt]) -> bool:
    return (
        mode == "elsewhere"
        and bool(pauses)
        and pauses[0].tool_call is not None
        and (pauses[0].tool_call.tool == CONFIRM[0])
    )


class UnclosedToolCall(AssertionError):
    """A tool call started on the event stream never ended there (BUG-2)."""


class OffButCalled(AssertionError):
    """A part turned off (``without=``) still called its service (BUG-9)."""


class OffButOffered(AssertionError):
    """A part turned off (``without=``) still had its tools offered to the model (BUG-10)."""


class NoEnding(AssertionError):
    """An attempt's events have no RUN_FINISHED (BUG-7)."""


class NotTimedOut(AssertionError):
    """A run past its time limit ended otherwise than TIMEOUT (BUG-7)."""


class MemoryContract(AssertionError):
    """What the harness sent the memory service broke its OpenAPI document (BUG-3)."""


class EventAfterEnd(AssertionError):
    """An attempt emitted events after its RUN_FINISHED (BUG-1)."""


def _well_formed(outcome: Outcome) -> None:
    """Each attempt's events: numbered from 0 without a gap, one ``RUN_STARTED`` first and one
    ``RUN_FINISHED`` last, every tool call started and finished."""
    by_attempt: dict[int, list[RunEvent]] = {}
    for event in outcome.events:
        by_attempt.setdefault(event.attempt, []).append(event)
    for attempt, events in by_attempt.items():
        assert [e.sequence for e in events] == list(range(len(events))), (attempt, events)
        assert events[0].type is RunEventType.RUN_STARTED, (attempt, events[0])
        finished = [n for n, e in enumerate(events) if e.type is RunEventType.RUN_FINISHED]
        if not finished:
            raise NoEnding((attempt, "no RUN_FINISHED"))
        if finished[0] != len(events) - 1:
            raise EventAfterEnd((attempt, events[finished[0] + 1 :]))
        started = [e.tool_call_id for e in events if e.type is RunEventType.TOOL_CALL_START]
        ended = [e.tool_call_id for e in events if e.type is RunEventType.TOOL_CALL_END]
        if sorted(map(str, started)) != sorted(map(str, ended)):
            raise UnclosedToolCall((attempt, started, ended))


async def _streamed(agent: Agent, input: str, options: dict[str, Any]) -> list[RunEvent]:
    return [e async for e in agent.stream(input, user=USER, **options)]


async def _as_list(task: Awaitable[Any]) -> list[Any]:
    await task
    return []


async def _sent(
    client: Client,
    tenant: str,
    part: Any,
    *,
    task_id: str = "",
    context_id: str | None = None,
) -> list[StreamResponse]:
    message = new_message([part], context_id=context_id, role=Role.ROLE_USER)
    if task_id:
        message.task_id = task_id
    headers = {HTTP_EXTENSION_HEADER: EXTENSION_URI, **identity_headers(tenant, USER)}
    context = ClientCallContext(service_parameters=headers)
    return [
        r async for r in client.send_message(SendMessageRequest(message=message), context=context)
    ]


def _task_id(responses: Sequence[StreamResponse]) -> str | None:
    for response in responses:
        which = response.WhichOneof("payload")
        if which == "task":
            return response.task.id
        if which == "status_update":
            return response.status_update.task_id
    return None


def _context_id(responses: Sequence[StreamResponse]) -> str | None:
    for response in responses:
        if response.WhichOneof("payload") == "task":
            return response.task.context_id
    return None
