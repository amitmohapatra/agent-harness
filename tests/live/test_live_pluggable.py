"""Way 2 against the real services: a team's own LangGraph graph (``tests/live/team.py``: a
planned model, LangGraph's checkpointer and ``interrupt()``) using Trellis's blocks with no
harness agent anywhere in the code under test —

* ``trellis.memory``: the memory service's context in the prompt, the turn recorded, feedback;
* ``trellis.harness.governance``: tools published, an administrator's ``approve_when`` in the
  catalog making a ``governed`` tool ask, through LangGraph's ``interrupt()``;
* ``trellis.runs``: the run started, paused in agent-runs while the graph waits, found in the
  inbox, resumed from the graph's checkpointer and finished; the decision told to governance;
* agent-runs' webhooks: signed ``run.paused`` / ``run.finished`` deliveries a local receiver
  verifies (agent-runs takes plain ``http`` in its ``dev`` environment only, which the local
  stack runs in: ``RUNS__SERVICE__ENVIRONMENT=dev``; elsewhere a receiver is ``https``);
* a schedule fired now, claimed and run by ``trellis.runs.Worker`` with a plain handler;
* ``trellis.harness.evals``: ``evaluate`` over the graph as a callable and ``judge`` on-line —
  without Langfuse (scores are spans), and with the suite's local Langfuse stand-in, which
  records what reaches Langfuse's API (no real Langfuse here);
* ``trellis.harness.a2a.remote``: the graph calling an agent served over A2A (the remote side
  is a wrapped agent: serving is Way 1), the remote question answered through ``interrupt()``.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Final

import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt

from tests.live.conftest import live_harness, needs_memory, needs_runs
from tests.live.support import StubLangfuse, eventually, free_port, seed, serving
from tests.live.team import Team, graph
from tests.support.planned import PlannedChatModel
from trellis import Runtime
from trellis.contracts import (
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStatus,
    ScheduleSpec,
    ToolSpec,
    new_id,
)
from trellis.harness import telemetry
from trellis.harness.a2a import remote
from trellis.harness.evals import (
    EvalCase,
    EvalOutput,
    EvalScore,
    EvalServices,
    contains,
    evaluate,
    grounding,
    judge,
)
from trellis.harness.governance import governed
from trellis.harness.journal import content_key
from trellis.runs import Job, WebhookCreated, WebhookEvent, Worker, parse_delivery
from trellis.runs.webhooks import (
    DELIVERY_HEADER,
    EVENT_HEADER,
    SIGNATURE_HEADER,
    verify_signature,
)

pytestmark = [pytest.mark.live, needs_memory, needs_runs]

#: Each test waits on both services; the webhook one also on the ticker's sweep (every 5 s).
TIMEOUT_SECONDS: Final = 180
#: The administrator's rule every test that gates a purchase order sets.
RULE: Final = "qty > 100"
#: Where the receiver's path starts: a subscription left by a session that died is found by it.
HOOK_PATH: Final = "/trellis-live-hooks/"


@pytest.fixture
async def team() -> AsyncIterator[Team]:
    team = await Team.open(f"live-plain-{uuid.uuid4().hex[:8]}")
    try:
        yield team
    finally:
        await team.aclose()


@dataclass
class Case:
    """One test's names, unique so the services' state it reads back is its own, and a ledger
    each real execution of the purchase-order tool appends to."""

    suffix: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ordered: list[tuple[str, int]] = field(default_factory=list)

    @property
    def user(self) -> str:
        return f"live-plain-{self.suffix}"

    @property
    def thread(self) -> str:
        return f"live-plain-thread-{self.suffix}"

    @property
    def tool(self) -> str:
        return f"create_po_{self.suffix}"

    @property
    def sku(self) -> str:
        return f"A-{self.suffix}"

    def create_po(self, sku: str, qty: int) -> str:
        """The team's own tool: places a purchase order (a real side effect: the ledger)."""
        self.ordered.append((sku, qty))
        return f"PO for {qty} x {sku}"


async def gate(team: Team, case: Case) -> None:
    """Publish the purchase-order tool and set an administrator's rule on it in the catalog
    (the memory service's catalog API, as the other live tests set rules)."""
    spec = ToolSpec(
        name=case.tool,
        description="Create a purchase order.",
        input_schema={
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
        },
        side_effects="write",
    )
    await team.governance.publish([spec])
    catalog = team.scope(case.user).advanced.tools
    await catalog.put_catalog([{"name": case.tool, "side_effects": "write", "approve_when": RULE}])
    [entry] = await catalog.catalog(names=[case.tool])
    assert entry.approve_when == RULE


def purchasing(team: Team, case: Case, run_id: str, qty: int) -> tuple[Any, PlannedChatModel]:
    """The graph for one purchase-order run, and its model (which plans the one call)."""
    tools = {case.tool: governed(case.create_po, team.governance, name=case.tool, on_ask=team.ask)}
    model = PlannedChatModel(
        plan=[(case.tool, {"sku": case.sku, "qty": qty})], final="Done. {last}"
    )
    memory = team.scope(case.user, case.thread, run_id)
    return graph(memory, model, tools, team.checkpointer), model


# --------------------------------------------------------------------------- (a) memory
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_the_graph_gets_memory_context_records_its_turn_and_takes_feedback(
    team: Team,
) -> None:
    case = Case()
    supplier = f"SUP-{case.suffix}"
    await seed(
        team.scope(case.user),
        f"{case.user} buys steel only from Acme Steel, supplier id {supplier}.",
        visibility="USER",
    )
    question = "Who do I buy steel from?"
    record = await team.start(question, user=case.user, thread=case.thread)
    answer = f"You buy steel from Acme Steel ({supplier})."
    model = PlannedChatModel(plan=[], final=answer)
    memory = team.scope(case.user, case.thread, record.run_id)

    done = await team.advance(graph(memory, model, {}, team.checkpointer), record)

    assert done.status is RunStatus.SUCCESS and done.output == answer
    assert supplier in model.said()  # the memory service's context reached the prompt
    history = await team.scope(case.user, case.thread).history()
    assert [(m.role, m.content) for m in history][-2:] == [
        ("USER", question),
        ("ASSISTANT", answer),
    ]
    given = await memory.feedback(
        "run", record.run_id, "confirm", reviewer=case.user, comment="the right supplier"
    )
    listed = (await memory.feedback.page_for("run", record.run_id)).items
    assert given.feedback_id in [f.feedback_id for f in listed]


# --------------------------------------------------------------------------- (b) governance
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_an_administrators_rule_makes_a_governed_tool_ask_through_interrupt(
    team: Team,
) -> None:
    case = Case()
    await gate(team, case)

    small = await team.governance.check(case.tool, {"sku": case.sku, "qty": 10})
    assert small.announces and small.rule == RULE
    big = await team.governance.check(case.tool, {"sku": case.sku, "qty": 500})
    assert big.asks and big.question == f"Approve {case.tool}? {RULE}."

    # under the rule the graph runs straight through
    app, _ = purchasing(team, case, new_id("run_"), 10)
    config: Any = {"configurable": {"thread_id": f"small-{case.suffix}"}}
    out = await app.ainvoke({"messages": [HumanMessage("Order 10")]}, config)
    assert out["answer"] == f"Done. PO for 10 x {case.sku}" and team.asked == []

    # over it the graph stops in LangGraph's interrupt(), its state in the checkpointer
    app, _ = purchasing(team, case, new_id("run_"), 500)
    config = {"configurable": {"thread_id": f"big-{case.suffix}"}}
    out = await app.ainvoke({"messages": [HumanMessage("Order 500")]}, config)
    [waiting] = out["__interrupt__"]
    assert waiting.value["question"] == big.question and waiting.value["args"]["qty"] == 500
    assert case.ordered == [(case.sku, 10)]  # the big order was not placed
    assert (await app.aget_state(config)).next == ("act",)

    # the person rejects it: the tool never runs, and the model reads why
    out = await app.ainvoke(Command(resume=False), config)
    assert case.ordered == [(case.sku, 10)]
    assert out["answer"] == f"Done. {case.tool} was not run: the approver rejected it"


# --------------------------------------------------------------------------- (c) runs
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_paused_graph_waits_in_the_inbox_and_resumes_from_its_checkpointer(
    team: Team,
) -> None:
    case = Case()
    await gate(team, case)
    buyers = f"role:live-buyers-{case.suffix}"
    record = await team.start("Order 500 units", user=case.user, thread=case.thread)
    app, _ = purchasing(team, case, record.run_id, 500)

    paused = await team.advance(app, record, assignee=buyers)

    assert paused.status is RunStatus.PAUSED and case.ordered == []
    inbox = [r async for r in team.runs.iterate(status=RunStatus.PAUSED, assignee=buyers)]
    [waiting] = inbox
    asked = waiting.awaiting
    assert waiting.run_id == record.run_id and asked is not None
    assert asked.reason is InterruptReason.APPROVAL and asked.question == team.asked[0].question
    assert asked.tool_call is not None and asked.tool_call.args == {"sku": case.sku, "qty": 500}

    resolution = InterruptResolution(
        interrupt_id=asked.interrupt_id,
        run_id=record.run_id,
        decision=InterruptDecision.APPROVE,
        reviewer="live-lee",
    )
    resumed = await team.runs.resume(resolution)
    assert resumed.status is RunStatus.RUNNING and resumed.attempt == 2  # not a queued run
    done = await team.advance(app, resumed, resume=True)  # from LangGraph's checkpointer

    assert done.status is RunStatus.SUCCESS and done.output == f"Done. PO for 500 x {case.sku}"
    assert case.ordered == [(case.sku, 500)]
    assert [r async for r in team.runs.iterate(status=RunStatus.PAUSED, assignee=buyers)] == []
    [entry] = (await team.runs.resolutions(record.run_id)).items
    assert entry.resolution.decision is InterruptDecision.APPROVE

    # the person's verdict, told to governance: the feedback approval suggestions learn from
    await team.governance.decided(
        team.asked[0], "approve", reviewer="live-lee", run_id=record.run_id, user=case.user
    )
    memory = team.scope(case.user, run_id=record.run_id)
    target = content_key("call", case.tool, {"sku": case.sku, "qty": 500})
    learned = (await memory.feedback.page_for("tool_call", target)).items
    assert [(f.verdict, f.source, f.reviewer) for f in learned] == [
        ("approve", "interrupt", "live-lee")
    ]


# --------------------------------------------------------------------------- (d) webhooks
@dataclass
class Receiver:
    """A team's webhook endpoint: every POST received, its headers and its exact body."""

    url: str
    received: list[tuple[dict[str, str], bytes]] = field(default_factory=list)


@pytest.fixture
def receiver() -> Iterator[Receiver]:
    got: list[tuple[dict[str, str], bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            got.append(({k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield Receiver(f"http://127.0.0.1:{port}{HOOK_PATH}{uuid.uuid4().hex[:8]}", got)
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
async def subscription(team: Team, receiver: Receiver) -> AsyncIterator[WebhookCreated]:
    """The receiver subscribed to the tenant's pauses and endings (agent-runs keeps at most 20
    per tenant: one a dead session left is removed first)."""
    stale = await team.runs.webhooks.list(limit=100)
    for hook in stale.items:
        if HOOK_PATH in hook.url:
            await team.runs.webhooks.delete(hook.webhook_id)
    created = await team.runs.webhooks.create(
        receiver.url, [WebhookEvent.PAUSED, WebhookEvent.FINISHED]
    )
    try:
        yield created
    finally:
        await team.runs.webhooks.delete(created.webhook_id)


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_pauses_and_endings_are_signed_webhook_deliveries(
    team: Team, receiver: Receiver, subscription: WebhookCreated
) -> None:
    case = Case()
    await gate(team, case)
    record = await team.start("Order 500 units", user=case.user, thread=case.thread)
    app, _ = purchasing(team, case, record.run_id, 500)
    paused = await team.advance(app, record)
    assert paused.awaiting is not None
    resumed = await team.runs.resume(
        InterruptResolution(
            interrupt_id=paused.awaiting.interrupt_id,
            run_id=record.run_id,
            decision=InterruptDecision.REJECT,
            reviewer="live-lee",
        )
    )
    done = await team.advance(app, resumed, resume=False)
    assert done.status is RunStatus.SUCCESS and case.ordered == []

    def deliveries() -> dict[str, tuple[dict[str, str], bytes]]:
        """This run's deliveries by event (the subscription hears the whole tenant)."""
        mine = {}
        for headers, body in list(receiver.received):
            delivery = parse_delivery(body)
            if delivery.data.run.run_id == record.run_id:
                mine[delivery.type.value] = (headers, body)
        return mine

    async def both() -> bool:
        return set(deliveries()) == {"run.paused", "run.finished"}

    # the ticker sends the outbox every 5 s
    assert await eventually(both, within=90), receiver.received
    for event, (headers, body) in deliveries().items():
        assert verify_signature(subscription.secret, headers[SIGNATURE_HEADER.lower()], body)
        delivery = parse_delivery(body)
        assert headers[EVENT_HEADER.lower()] == event
        assert headers[DELIVERY_HEADER.lower()] == delivery.event_id
        assert delivery.tenant_id == team.tenant and delivery.data.run.agent_id == team.agent_id
        # a body changed on the way, or another secret, does not verify
        assert not verify_signature(
            subscription.secret, headers[SIGNATURE_HEADER.lower()], body + b" "
        )
        assert not verify_signature("not-the-secret", headers[SIGNATURE_HEADER.lower()], body)
    paused_body = parse_delivery(deliveries()["run.paused"][1])
    finished_body = parse_delivery(deliveries()["run.finished"][1])
    assert paused_body.data.run.status is RunStatus.PAUSED
    assert finished_body.data.run.status is RunStatus.SUCCESS


# --------------------------------------------------------------------------- (e) schedules
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_fired_schedule_is_claimed_and_run_by_a_plain_worker(team: Team) -> None:
    case = Case()
    stock = f"check_stock_{case.suffix}"
    await seed(
        team.scope(case.user),
        f"{case.user} wants the stock digest for SKU {case.sku}.",
        visibility="USER",
    )

    def check_stock(sku: str) -> str:
        return f"42 units of {sku}"

    tools = {
        stock: governed(
            check_stock, team.governance, name=stock, side_effects="read", on_ask=team.ask
        )
    }
    seen: list[str] = []

    async def handle(job: Job) -> None:
        """The team's handler: its graph on the claimed run's input, as the schedule's user."""
        run = job.record
        assert run.on_behalf_of is not None
        user = run.on_behalf_of.removeprefix("user:")
        model = PlannedChatModel(plan=[(stock, {"sku": case.sku})], final="Digest: {last}")
        memory = team.scope(user, f"digest-{run.run_id}", run.run_id)
        app = graph(memory, model, tools, team.checkpointer)
        out = await app.ainvoke(
            {"messages": [HumanMessage(run.input["question"])]},
            {"configurable": {"thread_id": run.run_id}},
        )
        seen.append(model.said())
        await job.finish(RunStatus.SUCCESS, output=out["answer"])

    schedule = await team.runs.schedules.create(
        ScheduleSpec(
            tenant_id=team.tenant,
            agent_id=team.agent_id,
            name=f"live stock digest {case.suffix}",
            cadence="manual",
            on_behalf_of=f"user:{case.user}",
            input={"question": "What is my stock digest?"},
        )
    )
    try:
        worker = Worker(team.runs, handle, [team.agent_id], concurrency=1, lease_seconds=30)
        fired = await team.runs.schedules.fire(schedule.schedule_id)
        assert await eventually(worker.run_once, within=30)  # claimed and run to its end
        run = await team.runs.get(fired.run_id)
        assert run is not None and run.status is RunStatus.SUCCESS
        assert run.output == f"Digest: 42 units of {case.sku}"
        assert run.metadata["schedule_id"] == schedule.schedule_id
        assert case.sku in seen[0] and team.asked == []  # memory's context; a read never asks

        # the same handler under the worker's own loop: the next fire is picked up by it
        loop = asyncio.create_task(worker.run())
        try:
            await asyncio.sleep(1.1)  # another tick: a fire for another second is another run
            again = await team.runs.schedules.fire(schedule.schedule_id)
            assert again.run_id != fired.run_id

            async def finished() -> bool:
                current = await team.runs.get(again.run_id)
                return current is not None and current.status is RunStatus.SUCCESS

            assert await eventually(finished, within=60)
        finally:
            worker.stop()
            await loop
    finally:
        await team.runs.schedules.delete(schedule.schedule_id)


# --------------------------------------------------------------------------- (f) evaluation
@dataclass(frozen=True)
class cites_supplier:
    """A deterministic evaluator: whether the answer names the supplier id."""

    name: str = "cites_supplier"

    async def __call__(self, case: EvalCase) -> EvalScore:
        return EvalScore(self.name, "SUP-40" in str(case.output))


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_evaluate_and_judge_score_the_graph_as_a_plain_callable(team: Team) -> None:
    case = Case()
    # unique per run: the memory SDK's default idempotency key leaves the user out
    await seed(
        team.scope(case.user),
        f"The Berlin office reorders steel from Acme Steel, supplier id SUP-40 ({case.suffix}).",
        visibility="USER",
    )
    answer = "Acme Steel supplies steel to the Berlin office; its supplier id is SUP-40."

    async def ask(question: str) -> EvalOutput:
        """The graph as evaluation calls it: its answer, and the memory context it was given."""
        run_id = new_id("run_")
        memory = team.scope(case.user, f"eval-{run_id}", run_id)
        app = graph(memory, PlannedChatModel(plan=[], final=answer), {}, team.checkpointer)
        out = await app.ainvoke(
            {"messages": [HumanMessage(question)]}, {"configurable": {"thread_id": run_id}}
        )
        return EvalOutput(out["answer"], bundle_id=out["bundle_id"], memory=memory)

    dataset = [
        {"input": "Who supplies steel to the Berlin office?", "expected": "Acme Steel"},
        {"input": "What is Acme Steel's supplier id?", "expected": "SUP-40"},
    ]
    evaluators = [contains(), grounding(), cites_supplier()]

    # without Langfuse: the scores are the report's and spans', nothing is posted
    async with EvalServices() as bare:
        report = await evaluate(
            ask, dataset, evaluators, services=bare, user=case.user, run_name=f"live-{case.suffix}"
        )
    assert [i.status for i in report.items] == ["success", "success"], report
    assert report.summary["contains"].mean == 1.0
    assert report.summary["cites_supplier"].mean == 1.0
    grounded = report.summary["grounding"]
    assert grounded.count == 2 and grounded.mean is not None and grounded.mean > 0.5, report
    assert all(i.trace_url is None for i in report.items) and report.dataset_run_url is None

    # with Langfuse (the suite's local stand-in: a real Langfuse is not part of this stack)
    with StubLangfuse() as langfuse:
        async with EvalServices.from_env(langfuse.environ()) as services:
            assert services.langfuse is not None
            report = await evaluate(ask, dataset, evaluators, services=services, user=case.user)
            # on-line, from plain code: one answer judged, its scores on its own trace
            given = await ask("Who supplies steel to the Berlin office?")
            online = EvalCase(
                input="Who supplies steel to the Berlin office?",
                output=given.answer,
                run_id=new_id("run_"),
                bundle_id=given.bundle_id,
                memory=given.memory,
            )
            scores, failed = await judge(online, [grounding(), cites_supplier()], services=services)
    assert failed == {} and {s.name for s in scores} == {"grounding", "cites_supplier"}
    posted = {(s["name"], s["traceId"]) for s in langfuse.posted("/api/public/scores")}
    traces = {telemetry.trace_hex(i.run_id or "") for i in report.items}
    assert {(n, t) for n in ("contains", "grounding", "cites_supplier") for t in traces} <= posted
    online_trace = telemetry.trace_hex(online.run_id or "")
    assert {("grounding", online_trace), ("cites_supplier", online_trace)} <= posted
    assert all(i.trace_url and langfuse.url in i.trace_url for i in report.items)


# --------------------------------------------------------------------------- (g) A2A
async def planner(input: Any, agent: Runtime) -> str:
    region = await agent.ask("Which region?", options=["eu", "us"])
    return f"deploying {input} to {region}"


@pytest.fixture
def planner_url() -> Iterator[str]:
    """The remote side: an agent served over A2A by a harness (serving is Way 1)."""
    port = free_port()
    url = f"http://127.0.0.1:{port}/a2a"
    app = FastAPI()
    live_harness().wrap(planner, id=f"live-planner-{uuid.uuid4().hex[:6]}").serve_a2a(app, url)
    with serving(app, port):
        yield url


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_the_graph_calls_a_remote_a2a_agent_and_answers_its_question(
    team: Team, planner_url: str
) -> None:
    case = Case()
    questions: list[str] = []

    def answer_from_a_person(question: str) -> Any:
        """The remote question becomes the graph's own interrupt()."""
        questions.append(question)
        return interrupt({"question": question})

    async with remote(
        planner_url,
        tenant=team.tenant,
        user=case.user,
        thread=case.thread,
        on_input=answer_from_a_person,
    ) as deployer:
        assert deployer.spec.source == "a2a"
        model = PlannedChatModel(plan=[(deployer.spec.name, {"message": "the shop"})])
        memory = team.scope(case.user, case.thread, new_id("run_"))
        app = graph(memory, model, {deployer.spec.name: deployer}, team.checkpointer)
        config: Any = {"configurable": {"thread_id": case.thread}}
        out = await app.ainvoke({"messages": [HumanMessage("Deploy the shop")]}, config)
        [waiting] = out["__interrupt__"]
        assert waiting.value == {"question": "Which region?"}
        # the person answers: the graph runs the node again, the remote agent asks again
        # (its first task was cancelled when the graph stopped) and gets the answer
        out = await app.ainvoke(Command(resume="eu"), config)
    assert out["answer"] == "Done. deploying the shop to eu"
    assert questions == ["Which region?", "Which region?"]
