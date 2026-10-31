"""The matrix's adversarial rows: what tries to break the harness itself, on every adapter —
two people on one thread id, messages sent while a run works (queued, interrupting, steering),
a burst of runs at once, a slow run store, malformed tool calls, a model looping on one call,
and helpers: twenty-four at once under the admission limit, nested three deep, failing or
timing out, cancelled with their parent, asking for an approval, steered around.

Each row is a :class:`~tests.matrix.model.Feature` (ids ``X..``, audit ``ADV``) on the same
cells as the others; a row runs only in the modes where its point is (the surfaces are the
other rows' concern), and a failure this suite found and that is not fixed yet is a ``Bug`` in
``generate.KNOWN`` — and an entry of ``docs/research/IMPROVEMENT_BACKLOG.md`` (Adversarial
suite) — not a weakened assertion. The model-facing half (what the model must be told and
answers) is ``benchmark.cost.adversarial``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

from tests.matrix.kit import Desk
from tests.matrix.model import ADAPTERS, NA, Feature, Gap
from tests.matrix.world import USER, World
from tests.support.arm import arm
from trellis import Runtime, current, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.admission import CEILING

#: How long the scenarios wait for a run to reach its tool, at most (a guard, not a measure).
STARTED_SECONDS: Final = 30.0


def _only(*modes: str, reason: str) -> dict[str, Any]:
    from tests.matrix.model import MODES

    return {m: NA(reason) for m in MODES if m not in modes}


#: Why a row that drives runs itself runs in the ``run`` mode only.
DIRECT: Final = "the scenario drives several runs of one agent itself (agent.run): the modes are the other rows'"
#: The adapters with no model between the calls (a function target calls its tools itself).
NO_MODEL: Final = NA("a function target has no model: your code calls the tools itself")


def _whoami() -> Any:
    @tool(side_effects="read")
    async def whoami() -> str:
        """Who is asking, and in which thread."""
        runtime = current()
        assert runtime is not None
        return f"user {runtime.user} thread {runtime.thread}"

    return whoami


# --------------------------------------------------------------------------- conversations
async def two_people_one_thread(w: World) -> None:
    """Two people naming one thread alike: each run reads only its person's conversation (a
    leak shows as the second person's model answering from the first one's tool result)."""
    agent = await w.agent([_whoami()], [("whoami", {})])
    ana = await agent.run("who am I?", user="ana", thread="shared")
    ben = await agent.run("who am I?", user="ben", thread="shared")
    again = await agent.run("who am I?", user="ana", thread="shared")
    assert ana.status is RunStatus.SUCCESS and ana.answer == "Done. user ana thread shared", ana
    assert ben.status is RunStatus.SUCCESS, (ben.status, ben.error)
    assert ben.answer == "Done. user ben thread shared", f"ben read ana's turn: {ben.answer!r}"
    assert again.status is RunStatus.SUCCESS and "user ana" in str(again.answer), again.answer


async def busy_conversation(w: World) -> None:
    """A second message while the first works: ``queue`` waits for it, ``interrupt`` cancels
    it and continues, ``steer`` reaches its model at the next step (where the framework takes
    one) — each sent the moment the first run's tool started."""
    d = Desk()
    agent = await w.agent([d.wait()], [("wait", {"seconds": 1})])

    async def second(thread: str, mode: str, text: str) -> tuple[Any, Any]:
        d.started.clear()
        first = asyncio.create_task(agent.run("first message", user=USER, thread=thread))
        await asyncio.wait_for(d.started.wait(), STARTED_SECONDS)
        later = await agent.run(text, user=USER, thread=thread, mode=mode)
        return await first, later

    queued_first, queued = await second("q", "queue", "second message")
    assert queued_first.status is RunStatus.SUCCESS, (queued_first.status, queued_first.error)
    assert queued.status is RunStatus.SUCCESS, (queued.status, queued.error)
    one = await w.store.get(queued_first.run_id)
    two = await w.store.get(queued.run_id)
    assert one is not None and two is not None
    assert two.updated_at >= one.updated_at, (
        "the queued message finished before the one it waited on"
    )

    cut, interrupting = await second("i", "interrupt", "stop, do this instead")
    assert cut.status is RunStatus.CANCELLED, (cut.status, cut.error)
    assert interrupting.status is RunStatus.SUCCESS, (interrupting.status, interrupting.error)

    mark = len(w.models.calls)
    steered_first, steering = await second("s", "steer", "and mind the north")
    assert steered_first.status is RunStatus.SUCCESS, (steered_first.status, steered_first.error)
    assert steering.status is RunStatus.SUCCESS, (steering.status, steering.error)
    if agent.adapter.steers(agent.target) and w.adapter not in ("claude", "kimi"):
        said = "\n".join(s for s, _ in w.models.calls[mark:])
        assert "and mind the north" in said, "the steered message never reached the model"
        assert steering.answer == steered_first.answer, (steering.answer, steered_first.answer)


async def burst(w: World) -> None:
    """Twenty runs of one agent at once, each its own thread: every one ends well with its own
    answer (nothing crosses between runs)."""
    agent = await w.agent([_whoami()], [("whoami", {})])
    results = await asyncio.gather(
        *(agent.run("who am I?", user=USER, thread=f"b{k}") for k in range(BURST)),
        return_exceptions=True,
    )
    raised = [r for r in results if isinstance(r, BaseException)]
    assert not raised, raised[:3]
    ended = [r for r in results if not isinstance(r, BaseException)]
    for k, result in enumerate(ended):
        assert result.status is RunStatus.SUCCESS, (k, result.status, result.error)
        assert result.answer == f"Done. user {USER} thread b{k}", (k, result.answer)


BURST: Final = 20


async def slow_store(w: World) -> None:
    """Every call to the run store 30 ms late (a run service under load): the run still ends
    well, its answer and record whole."""
    store = w.store
    for name in ("get", "heartbeat", "pause", "append_events", "complete", "fail", "claim"):
        original = getattr(store, name, None)
        if original is None:
            continue

        async def late(*a: Any, _original: Any = original, **k: Any) -> Any:
            await asyncio.sleep(0.03)
            return await _original(*a, **k)

        w.monkeypatch.setattr(store, name, late)
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "tides"})])).succeeded()
    assert o.answer == "Done. facts about tides", o.answer
    assert o.record.output == o.answer or o.record.status is RunStatus.SUCCESS


# --------------------------------------------------------------------------- the model misbehaves
async def malformed_calls(w: World) -> None:
    """The model calls a tool that does not exist, then one with an argument of the wrong type:
    each is answered with what is wrong (nothing runs), and the run goes on to its answer."""
    d = Desk()
    mark = len(w.models.calls)
    o = await w.go(
        [d.lookup()],
        [
            ("lookup_all", {"topic": "tides"}),
            ("lookup", {"topic": 5}),
            ("lookup", {"topic": "tides"}),
        ],
    )
    assert o.status is RunStatus.SUCCESS, (o.status, o.record.error)
    assert o.answer == "Done. facts about tides", o.answer
    assert d.ran("lookup") == [{"topic": "tides"}], d.ran("lookup")
    if w.adapter not in ("claude", "kimi"):
        said = "\n".join(s for s, _ in w.models.calls[mark:])
        assert "lookup_all" in said and ("not" in said or "error" in said.lower()), said[-800:]


#: How many times the looping model repeats its call (the run must stop it well before).
LOOPS: Final = 30


async def looping(w: World) -> None:
    """The model makes one call over and over: the harness stops it well before the model
    does, and the run ends with an answer (not an error with none)."""
    d = Desk()
    o = await w.go([d.lookup()], [("lookup", {"topic": "tides"})] * LOOPS)
    ran = len(d.ran("lookup"))
    assert ran < LOOPS, f"nothing stopped a model repeating one call ({ran} calls ran)"
    assert o.status in (RunStatus.SUCCESS, RunStatus.PARTIAL), (o.status, o.record.error)
    assert o.text, "the run ended with no answer"


# --------------------------------------------------------------------------- helpers
class Gauge:
    """How many helpers work at once, at most."""

    def __init__(self) -> None:
        self.now = 0
        self.most = 0

    async def held(self, seconds: float) -> None:
        self.now += 1
        self.most = max(self.most, self.now)
        try:
            await asyncio.sleep(seconds)
        finally:
            self.now -= 1


def _helper(w: World, name: str, body: Any, tools: list[Any] | None = None, **wrap: Any) -> Any:
    """A function agent ``name`` wrapped in the cell's deployment, as a tool of another run."""
    h = w.harness(0)
    found = h.agents.get(name)
    return found or arm(h.wrap(body, id=name, **wrap), tools or [])


def _kids(w: World, run_id: str) -> list[Any]:
    return [r for r in w.store._runs.values() if r.parent_run_id == run_id]


#: How many helpers one turn fans out to.
FAN_OUT: Final = 24


async def fan_out(w: World) -> None:
    """One turn calls a helper 24 times at once: every helper is a child run that answers, and
    no more than the admission limit work at the same time."""
    gauge = Gauge()

    async def count(input: str, agent: Runtime) -> str:
        """Count one bin and say what was counted."""
        await gauge.held(0.02)
        return f"counted {input}"

    helper = _helper(w, "count", count)
    step = [("count", {"message": f"bin {k}"}) for k in range(FAN_OUT)]
    if w.adapter in ("claude", "kimi"):
        o = await w.go([helper], step)  # its CLI makes one call at a time
    else:
        from tests.matrix.features import _parallel_target

        o = await w.go([helper], [step], target=_parallel_target(w))  # type: ignore[list-item]
    o.succeeded()
    kids = _kids(w, o.run_id)
    assert len(kids) == FAN_OUT, f"{len(kids)} child runs of {FAN_OUT}"
    assert all(k.status is RunStatus.SUCCESS for k in kids), [k.status for k in kids]
    assert gauge.most <= CEILING, f"{gauge.most} helpers at once, past the ceiling {CEILING}"
    assert o.answer.startswith("Done. counted bin "), o.answer


async def nested(w: World) -> None:
    """A helper that calls a helper that calls a tool: three runs, each the child of the one
    above, all ended well, the answer passed up whole."""
    d = Desk()

    async def leaf(input: str, agent: Runtime) -> str:
        """Look a topic up."""
        return str(await agent.tools.call("lookup", topic=input))

    async def middle(input: str, agent: Runtime) -> str:
        """Ask the leaf helper."""
        return str(await agent.tools.call("leaf", message=input))

    inner = _helper(w, "leaf", leaf, [d.lookup()])
    outer = _helper(w, "middle", middle, [inner])
    o = (await w.go([outer], [("middle", {"message": "tides"})])).succeeded()
    assert o.answer == "Done. facts about tides", o.answer
    [mid] = _kids(w, o.run_id)
    [low] = _kids(w, mid.run_id)
    assert (mid.agent_id, low.agent_id) == ("middle", "leaf"), (mid.agent_id, low.agent_id)
    assert mid.status is RunStatus.SUCCESS and low.status is RunStatus.SUCCESS


async def failing_helpers(w: World) -> None:
    """One helper fails (a 503), another outlives its own time limit: the parent's model reads
    both as errors and the parent still answers; the children end ERROR and TIMEOUT."""

    async def broken(input: str, agent: Runtime) -> str:
        """Ask the forecast service."""
        raise RuntimeError("forecast service: 503 Service Unavailable")

    async def sleepy(input: str, agent: Runtime) -> str:
        """Scan the warehouse (slow)."""
        await asyncio.sleep(5)
        return "scanned"

    one = _helper(w, "broken", broken)
    two = _helper(w, "sleepy", sleepy, run_timeout=0.3)
    o = await w.go([one, two], [("broken", {"message": "a"}), ("sleepy", {"message": "b"})])
    assert o.status is RunStatus.SUCCESS, (o.status, o.record.error)
    assert o.text, "no answer"
    kids = {k.agent_id: k.status for k in _kids(w, o.run_id)}
    assert kids.get("broken") is RunStatus.ERROR, kids
    assert kids.get("sleepy") is RunStatus.TIMEOUT, kids


async def cancel_children(w: World) -> None:
    """Cancelling a run whose helper's helper is working cancels all three."""
    d = Desk()

    async def leaf(input: str, agent: Runtime) -> str:
        """Hold."""
        return str(await agent.tools.call("hold"))

    async def middle(input: str, agent: Runtime) -> str:
        """Ask the leaf."""
        return str(await agent.tools.call("leaf2", message=input))

    inner = _helper(w, "leaf2", leaf, [d.hold()])
    outer = _helper(w, "middle2", middle, [inner])

    async def cancelling(run_id: str) -> None:
        await asyncio.wait_for(d.started.wait(), STARTED_SECONDS)
        await w.cancel(run_id, "the user left")

    o = await w.go([outer], [("middle2", {"message": "x"})], during=cancelling)
    assert o.status is RunStatus.CANCELLED, (o.status, o.record.error)
    [mid] = _kids(w, o.run_id)
    [low] = _kids(w, mid.run_id)
    assert mid.status is RunStatus.CANCELLED and low.status is RunStatus.CANCELLED, (
        mid.status,
        low.status,
    )
    assert d.ran("hold") == []


async def approval_in_child(w: World) -> None:
    """A helper's irreversible call asks: the parent pauses on the helper's question, the
    approval resumes both, and the call runs once."""
    d = Desk()

    async def refunder(input: str, agent: Runtime) -> str:
        """Refund an order."""
        return str(await agent.tools.call("refund", order=input))

    helper = _helper(w, "refunder", refunder, [d.refund()])
    o = (await w.go([helper], [("refunder", {"message": "A1"})])).succeeded()
    assert d.ran("refund") == [{"order": "A1"}], d.ran("refund")
    assert o.pauses, "the helper's irreversible call was never asked about"
    assert o.answer == "Done. refunded A1", o.answer


async def steer_while_helping(w: World) -> None:
    """A message steered into a run while its helper works reaches the run (or waits its turn
    where the framework takes none), and both end well."""
    gauge = Gauge()
    entered = asyncio.Event()

    async def slowhelp(input: str, agent: Runtime) -> str:
        """A helper that takes a moment."""
        entered.set()
        await gauge.held(0.5)
        return "helped"

    helper = _helper(w, "slowhelp", slowhelp)
    steered: list[Any] = []

    async def steering(run_id: str) -> None:
        await asyncio.wait_for(entered.wait(), STARTED_SECONDS)
        handle = w._current
        assert handle is not None
        steered.append(
            await handle.agent.run("also the north", user=USER, thread="t", mode="steer")
        )

    o = await w.go([helper], [("slowhelp", {"message": "x"})], during=steering, run={"thread": "t"})
    o.succeeded()
    [later] = steered
    assert later.status is RunStatus.SUCCESS, (later.status, later.error)
    agent = w._current.agent if w._current else None
    if agent is not None and agent.adapter.steers(agent.target) and w.adapter not in CLIS:
        assert "also the north" in w.models.said(), "the steered message never reached the model"


#: The adapters whose model is a CLI's (what it was sent is the CLI's own record).
CLIS: Final = ("claude", "kimi", "function")


# --------------------------------------------------------------------------- scenario jobs
#: The scenarios a job compares, and what each one's helper reports (margin per £ of cash).
SCENARIOS: Final = {"keep": 0.866, "cut": 0.9093, "add": 0.8664}


def _scenarios(
    w: World,
    counted: dict[str, int],
    *,
    process: int = 0,
    hold: float = 0.05,
    crash: list[BaseException] | None = None,
) -> Any:
    """The helper that works out one scenario (a child run per scenario): it counts its runs,
    and the first ``crash`` it is given kills the worker at the last scenario."""

    async def scenario(input: str, agent: Runtime) -> str:
        """Work out one assortment scenario's margin per pound of cash tied up."""
        if crash and input == list(SCENARIOS)[-1]:
            raise crash.pop()
        await asyncio.sleep(hold)
        counted[input] = counted.get(input, 0) + 1
        return f"{input}: {SCENARIOS[input]} per £"

    h = w.harness(process)
    found = h.agents.get("scenario")
    return found or arm(h.wrap(scenario, id="scenario"), [])


def _apply_plan(applied: list[str]) -> Any:
    @tool(side_effects="irreversible")
    async def apply_plan(scenario: str) -> str:
        """Apply an assortment plan to the live range (cannot be undone)."""
        applied.append(scenario)
        return f"applied {scenario}"

    return apply_plan


def _job_plan(w: World) -> list[Any]:
    """Every scenario in one step (one at a time for a CLI that makes one call at a time),
    then the plan applied."""
    calls = [("scenario", {"message": k}) for k in SCENARIOS]
    head = calls if w.adapter in ("claude", "kimi", "function") else [calls]
    return [*head, ("apply_plan", {"scenario": "cut"})]


async def scenario_job(w: World) -> None:
    """A long scenario comparison as a durable run: one child run per scenario under the
    admission limit, each on the run's stream as it starts and ends (with its seconds), an
    approval asked before the plan is applied (an irreversible write), the plan applied once,
    and — in a conversation — a follow-up's model given the job's answer to explain it."""
    counted: dict[str, int] = {}
    applied: list[str] = []
    helper = _scenarios(w, counted)
    target = None
    if w.adapter not in ("claude", "kimi", "function"):
        from tests.matrix.features import _parallel_target

        target = _parallel_target(w)
    run = {"thread": "job"} if w.mode == "run" else None
    o = await w.go([helper, _apply_plan(applied)], _job_plan(w), target=target, run=run)
    o.succeeded()
    assert counted == dict.fromkeys(SCENARIOS, 1), counted
    kids = [k for k in _kids(w, o.run_id) if k.agent_id == "scenario"]
    assert len(kids) == len(SCENARIOS) and all(k.status is RunStatus.SUCCESS for k in kids), kids
    assert applied == ["cut"], applied
    assert o.pauses, "the irreversible apply_plan was never asked about"
    ended = o.custom("helper_finished")
    assert len(ended) >= len(SCENARIOS), ended
    assert all("seconds" in e for e in ended), ended
    assert o.record.worked_seconds > 0
    if w.mode == "agui":
        names = [e.get("name") for _, e in o.surface if e.get("type") == "CUSTOM"]
        assert names.count("helper_finished") >= len(SCENARIOS), names
    if w.mode == "run" and w.adapter not in CLIS:
        assert w._current is not None
        mark = len(w.models.calls)
        why = await w._current.agent.run("why the cut plan?", user=USER, thread="job")
        # a planned model that does not see the job's tool results plans the job again, and
        # its apply_plan waits for an approval: what matters is what it was given first
        assert why.status in (RunStatus.SUCCESS, RunStatus.PAUSED), (why.status, why.error)
        said = w.models.calls[mark][0] if len(w.models.calls) > mark else ""
        assert "applied cut" in said or "cut: 0.9093" in said, (
            "the follow-up's model had no record of the job"
        )


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


def _crashed(error: BaseException) -> bool:
    if isinstance(error, Crash):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_crashed(e) for e in error.exceptions)
    return False


async def scenario_job_crash(w: World) -> None:
    """The worker running a scenario job dies at its last scenario; another worker takes the
    run over and finishes it: the scenarios that finished are not worked out again."""
    from datetime import UTC, datetime, timedelta

    from trellis.runs import Job

    counted: dict[str, int] = {}
    crash: list[BaseException] = [Crash()]
    plan = [("scenario", {"message": k}) for k in SCENARIOS]  # one at a time: the last one dies
    first = await w.agent([_scenarios(w, counted, crash=crash)], plan, process=0)
    handle = await first.start("compare the scenarios", user=USER)
    worker = first.harness.worker([first])
    claimed = await w.store.claim(worker.worker_id, [first.id])
    assert claimed is not None
    try:
        await first.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=w.store)
        )
    except BaseException as died:  # a framework's task group may wrap it
        if not _crashed(died):
            raise
    else:
        raise AssertionError("the worker did not die")
    held, _ = w.store._leases[handle.id]
    w.store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    second = await w.agent([_scenarios(w, counted, process=1)], plan, process=1, name=first.id)
    assert await second.harness.worker([second]).run_once()
    record = await w.store.get(handle.id)
    assert record is not None and record.status is RunStatus.SUCCESS, (
        record and record.status,
        record and record.error,
    )
    assert counted == dict.fromkeys(SCENARIOS, 1), (
        f"scenarios worked out again after the crash: {counted}"
    )


async def scenario_job_cancel(w: World) -> None:
    """Cancelling a scenario job while its scenarios work cancels every one of them."""
    counted: dict[str, int] = {}
    started = asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        """Work out one scenario (slowly)."""
        started.set()
        await asyncio.sleep(30)
        counted[input] = 1
        return input

    h = w.harness(0)
    helper = h.agents.get("slowscenario") or arm(h.wrap(slow, id="slowscenario"), [])
    calls = [("slowscenario", {"message": k}) for k in SCENARIOS]
    target = None
    plan: list[Any] = calls
    if w.adapter not in ("claude", "kimi", "function"):
        from tests.matrix.features import _parallel_target

        target, plan = _parallel_target(w), [calls]

    async def cancelling(run_id: str) -> None:
        await asyncio.wait_for(started.wait(), STARTED_SECONDS)
        await asyncio.sleep(0.05)
        await w.cancel(run_id, "the user changed their mind")

    o = await w.go([helper], plan, target=target, during=cancelling)
    assert o.status is RunStatus.CANCELLED, (o.status, o.record.error)
    kids = _kids(w, o.run_id)
    assert kids and all(k.status is RunStatus.CANCELLED for k in kids), [k.status for k in kids]
    assert counted == {}, counted


async def spending_recorded(w: World) -> None:
    """What a finished run spent — its tokens, model calls and cost — is on its end (the run's
    ``RUN_FINISHED`` event or its record), not only on each call's span."""
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "tides"})])).succeeded()
    finished = [e for e in o.events if e.type is RunEventType.RUN_FINISHED][-1]
    held = {**(finished.data or {}), **o.record.metadata}
    assert "tokens" in held or "model_calls" in held, (
        f"no spending on the run's end: {sorted(held)}"
    )


async def file_link(w: World) -> None:
    """A run's file, for its user only: the deployment hands out a link per user (the file
    store this probe waits on)."""
    files = getattr(w.harness(0), "files", None)
    assert files is not None, "the deployment has no file store: no run can hand its user a file"


SPENT_GAP: Final = Gap(
    "ADVspent",
    "a run's tokens, model calls and cost are kept in its journal while it works and dropped at its "
    "end: neither its RUN_FINISHED event nor its record says what it spent",
)
FILES_GAP: Final = Gap("ADVfiles", "no file store on main: a run cannot hand its user a file (#91)")


# --------------------------------------------------------------------------- the rows
ADVERSARIAL: Final = [
    Feature(
        "X01",
        "two people on one thread id: each run reads only its person's conversation",
        "ADV (J1)",
        "automatic: the conversation key is the tenant's, the user's and the thread's",
        two_people_one_thread,
        adapters={"function": NO_MODEL},
        modes=_only("run", reason=DIRECT),
        way2=NA("a block keys nothing by thread"),
    ),
    Feature(
        "X02",
        "a message to a busy conversation: queued, interrupting, steering",
        "ADV (A6)",
        "run(mode='queue' | 'interrupt' | 'steer')",
        busy_conversation,
        modes=_only("run", reason=DIRECT),
        way2=NA("the busy-conversation modes are the Harness's"),
    ),
    Feature(
        "X03",
        "a burst of twenty runs at once: each ends well with its own answer",
        "ADV (resource safety)",
        "automatic",
        burst,
        adapters={"function": NO_MODEL},
        modes=_only("run", reason=DIRECT),
        way2=NA("by design: your code runs the block as often as it likes"),
    ),
    Feature(
        "X04",
        "a slow run store: every call 30 ms late, the run still ends well",
        "ADV (resource safety)",
        "automatic",
        slow_store,
        modes=_only("run", "worker", reason="a store's latency is the same on every surface"),
        way2=NA("a block takes no run store"),
    ),
    Feature(
        "X05",
        "malformed tool calls (an unknown tool, a wrong type): answered, and the run goes on",
        "ADV (I)",
        "automatic: an invalid call is not run and the model reads why",
        malformed_calls,
        adapters={"function": NO_MODEL},
        modes=_only(
            "run", "stream", "worker", reason="a call's checks are the same on every surface"
        ),
        way2=NA("your framework checks its own calls"),
    ),
    Feature(
        "X06",
        "a model repeating one call: stopped well before, and the run still answers",
        "ADV (I, E4)",
        "automatic: the stall guard and the step limit",
        looping,
        adapters={"function": NO_MODEL},
        modes=_only("run", "stream", "worker", reason="the stop is the same on every surface"),
        way2=NA("your framework runs its own loop"),
    ),
    Feature(
        "X07",
        "fan-out: twenty-four helpers in one turn, every one answered, at most the admission limit at once",
        "ADV (E2)",
        "automatic: every child run goes through the admission limit",
        fan_out,
        adapters={"function": NA("a function target runs its own calls (asyncio.gather)")},
        modes=_only("run", "worker", reason="the children run the same way on every surface"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X08",
        "nested helpers: three runs deep, each the child of the one above",
        "ADV (F30)",
        "automatic",
        nested,
        modes=_only("run", "worker", reason="the children run the same way on every surface"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X09",
        "a helper that fails and one past its time limit: errors the parent reads, the parent answers",
        "ADV (F30)",
        "automatic",
        failing_helpers,
        adapters={"function": NO_MODEL},
        modes=_only("run", "worker", reason="the children run the same way on every surface"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X10",
        "cancelling a run cancels its helper and its helper's helper",
        "ADV (F05, F30)",
        "agent.cancel / handle.cancel / A2A tasks/cancel",
        cancel_children,
        modes=_only("run", "worker", "agui", "a2a", reason="a cancel's surfaces are F05's"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X11",
        "an approval asked inside a helper pauses the parent; approved, the call runs once",
        "ADV (F04, F30)",
        "automatic: the catalog's tiers, in the child as in the parent",
        approval_in_child,
        modes=_only("run", "worker", "elsewhere", reason="the pause's surfaces are F36's"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X12",
        "a message steered into a run while its helper works: both end well",
        "ADV (A6, F30)",
        "run(mode='steer')",
        steer_while_helping,
        modes=_only("run", reason=DIRECT),
        way2=NA("the busy-conversation modes are the Harness's"),
    ),
    Feature(
        "X13",
        "a file a run makes for its user: a download link only that user can open",
        "ADV (files)",
        "a file store (TRELLIS_FILES_URL)",
        file_link,
        modes=_only("run", reason=DIRECT),
        adapters=dict.fromkeys(ADAPTERS, FILES_GAP),
        way2=NA("no file store yet"),
    ),
    Feature(
        "X14",
        "a scenario job: a durable run, a child per scenario, progress on the stream, an approval before applying, explained after",
        "ADV (scenario jobs)",
        "start() / serve_chat / a worker; children through the admission limit",
        scenario_job,
        modes=_only(
            "run",
            "worker",
            "agui",
            "elsewhere",
            reason="a job's surfaces: run, a durable worker, AG-UI, another process",
        ),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X15",
        "a scenario job whose worker dies mid-way: another worker finishes it, no scenario worked out twice",
        "ADV (scenario jobs, F23)",
        "automatic: the journal and the lease",
        scenario_job_crash,
        modes=_only("worker", reason="a worker's crash is the worker mode's"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X16",
        "a scenario job cancelled while its scenarios work: every scenario cancelled",
        "ADV (scenario jobs, F05)",
        "agent.cancel / handle.cancel / A2A tasks/cancel",
        scenario_job_cancel,
        modes=_only("run", "worker", "agui", reason="a cancel's surfaces are F05's"),
        way2=NA("a block starts no child runs"),
    ),
    Feature(
        "X17",
        "what a finished run spent (tokens, model calls, cost) is on its end, not only on spans",
        "ADV (scenario jobs, cost)",
        "automatic",
        spending_recorded,
        adapters={"function": NO_MODEL} | {a: SPENT_GAP for a in ADAPTERS if a != "function"},
        modes=_only("run", reason="the run's end is the same on every surface"),
        way2=NA("a block keeps no run record"),
    ),
]
