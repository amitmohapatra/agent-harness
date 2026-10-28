"""``delegate``: a ``ReAct``'s helpers (``trellis.harness.delegate``). Offered to a wide task on
a large model; each job a child run that only reads, at once under the adaptive limit
(``admission.py``), on the run's task tree, merged for the agent; a failed, timed-out, refused
or skipped job is reported so; cancelling the run cancels them; the run's budget is theirs;
helpers do not delegate again; with agent-runs the jobs are queued for any worker."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from typing import Any

import pytest

from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.models import Script, ScriptedChat
from trellis import Budget, Harness, ReAct, Runtime, Settings, current, tool
from trellis.contracts import RunEvent, RunEventType, RunRecord, RunStatus
from trellis.harness import admission, delegate
from trellis.harness.admission import Admission
from trellis.harness.agent import Agent
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.runs import LocalRuns
from trellis.harness.runtime import DELEGATED
from trellis.harness.tools.bridge import ONLY_READS

WIDE = "Compare the revenue of Apple, Microsoft, Google and Amazon."
ITEMS = ["Apple", "Microsoft", "Google", "Amazon"]
#: A window with room (a large model's).
ROOM = 128_000

done: list[str] = []
running = {"now": 0, "most": 0}
gates: dict[str, asyncio.Event] = {}


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    done.clear()
    running.update(now=0, most=0)
    gates.clear()
    # the process's admission without the machine's own pressure: what the test sets
    monkeypatch.setattr(admission, "ADMISSION", Admission(machine=None))
    yield


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up.

    Args:
        topic: What to look up.
    """
    running["now"] += 1
    running["most"] = max(running["most"], running["now"])
    try:
        held = gates.get(topic) or gates.get("*")
        if held is not None:
            await held.wait()
        await asyncio.sleep(0.02)
        if topic == "boom":
            raise RuntimeError("the source is down")
        done.append(f"lookup {topic}")
        return f"facts about {topic}"
    finally:
        running["now"] -= 1


@tool(side_effects="write")
def file(note: str) -> str:
    """File a note.

    Args:
        note: The note.
    """
    done.append(f"file {note}")
    return f"filed {note}"


def _first_user(body: dict[str, Any]) -> str:
    return next((str(m["content"]) for m in body["messages"] if m["role"] == "user"), "")


class Routed(Script):
    """The parent's turns in order; a helper's by its part (``Your part:``): ``helper(part,
    calls made)`` — by default one ``lookup`` of its part, then what it found."""

    def __init__(self, turns: list[Any], helper: Any = None) -> None:
        super().__init__(turns)
        self.helper = helper or _looks_up
        self.helpers: list[dict[str, Any]] = []
        #: the parent's requests, in order
        self.parents: list[dict[str, Any]] = []

    def next(self, body: dict[str, Any]) -> Any:
        asked = _first_user(body)
        found = re.search(r"Your part:\n\n(.*?)\n\n", asked, re.S)
        if found is None:
            self.parents.append(body)
            return self.turns.pop(0)
        self.helpers.append(body)
        made = sum(m["role"] == "assistant" for m in body["messages"])
        return self.helper(found.group(1), made)


def _looks_up(part: str, made: int) -> Any:
    return ("lookup", {"topic": part}) if made == 0 else f"found {part}"


def jobs(*parts: str, role: str = "an analyst") -> dict[str, Any]:
    return {"jobs": [{"role": role, "description": p} for p in parts]}


def lead(h: Harness, script: Routed, *, tools: list[Any] | None = None, **kwargs: Any) -> Agent:
    model = ScriptedChat(script=script, model=kwargs.pop("model", "scripted"))
    react = ReAct("You answer business questions.", model, context_window=ROOM, **kwargs)
    return arm(h.wrap(react, id="lead"), tools if tools is not None else [lookup, file])


def merged(script: Routed, n: int = 1) -> dict[str, Any]:
    """What the parent's model read from its ``delegate`` call (its ``n``-th request)."""
    [told] = [m for m in script.parents[n]["messages"] if m["role"] == "tool"][-1:]
    return json.loads(told["content"])


def offered(script: Routed, n: int = 0) -> set[str]:
    return {t["function"]["name"] for t in script.parents[n].get("tools", [])}


async def children(h: Harness, run_id: str) -> list[RunRecord]:
    found = [s async for s in h.runs.iterate(parent_run_id=run_id)]
    return [r for s in found if (r := await h.runs.get(s.run_id)) is not None]


async def until(holds: Any, seconds: float = 10.0) -> None:
    async with asyncio.timeout(seconds):
        while not holds():  # noqa: ASYNC110 - the helpers' tools say so with a counter
            await asyncio.sleep(0.01)


async def test_four_parallel_jobs_run_at_once_and_merge(harness: Harness) -> None:
    gates["*"] = asyncio.Event()
    script = Routed([("delegate", jobs(*ITEMS)), "Apple leads."])
    agent = lead(harness, script)
    task = asyncio.create_task(agent.run(WIDE, user="ada", thread="t1"))
    await until(lambda: running["now"] == 4)  # all four helpers in their lookup at once
    gates["*"].set()
    result = await task
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == "Apple leads."
    assert "delegate" in offered(script)
    told = merged(script)
    assert told["answered"] == 4 and told["of"] == 4
    assert [j["answer"] for j in told["jobs"]] == [f"found {p}" for p in ITEMS]
    assert {j["helper"] for j in told["jobs"]} == {"an analyst"}
    assert sorted(done) == sorted(f"lookup {p}" for p in ITEMS)
    kids = await children(harness, result.run_id)
    assert len(kids) == 4
    for kid in kids:
        assert kid.status is RunStatus.SUCCESS
        assert kid.metadata[DELEGATED] == 1
        assert kid.thread_id == kid.run_id  # its own conversation
        assert kid.agent_id == "lead"
    # a helper is told its part and its role, is offered only what reads, and no delegate
    first = script.helpers[0]
    assert "You are an analyst" in _first_user(first)
    names = {t["function"]["name"] for t in first.get("tools", [])}
    assert "lookup" in names and "file" not in names and "delegate" not in names


async def test_the_limit_holds_however_many_jobs(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "ADMISSION", Admission(start=2, ceiling=2, machine=None))
    parts = [f"item {n}" for n in range(8)]
    script = Routed([("delegate", jobs(*parts)), "done"])
    result = await lead(harness, script).run("For each of these: " + ", ".join(parts), user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    assert merged(script)["answered"] == 8
    assert running["most"] == 2


async def test_the_task_tree_is_on_the_runs_stream(harness: Harness) -> None:
    script = Routed([("delegate", jobs("Apple", "Microsoft")), "done"])
    events: list[RunEvent] = [e async for e in lead(harness, script).stream(WIDE, user="ada")]
    custom = [e.data for e in events if e.type is RunEventType.CUSTOM]
    fanned = [d for d in custom if d["name"] == "delegate"]
    assert fanned == [{"name": "delegate", "jobs": 2, "runs": 2, "limit": 4}]
    phases = [d["name"] for d in custom if d["name"].startswith("helper_")]
    assert phases.count("helper_queued") == 2
    assert phases.count("helper_started") == 2
    finished = [d for d in custom if d["name"] == "helper_finished"]
    assert [d["status"] for d in finished] == ["success", "success"]
    assert all(d["tokens"] > 0 and d["seconds"] >= 0 for d in finished)
    assert {d["helper"] for d in finished} == {"an analyst"}
    assert sorted(d["job"] for d in finished) == [1, 2]


async def test_one_helper_that_fails_is_reported_and_the_others_merge(harness: Harness) -> None:
    script = Routed([("delegate", jobs("Apple", "boom", "Google", "Amazon")), "partly"])
    result = await lead(harness, script).run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)
    statuses = [j["status"] for j in told["jobs"]]
    assert statuses == ["success", "success", "success", "success"]
    # the helper read the lookup's error and said so: a tool error is the helper's to handle
    assert told["jobs"][1]["answer"] == "found boom"


async def test_a_helper_run_that_errors_or_times_out_is_reported(harness: Harness) -> None:
    async def broken(input: str, agent: Runtime) -> str:
        await agent.tools.call("lookup", topic="x")
        raise RuntimeError("the helper broke")

    async def slow(input: str, agent: Runtime) -> str:
        await agent.tools.call("lookup", topic="y")
        await asyncio.sleep(5)
        return "too late"

    async def fine(input: str, agent: Runtime) -> str:
        return str(await agent.tools.call("lookup", topic=input))

    kids = [
        arm(harness.wrap(broken, id="broken"), [lookup]),
        arm(harness.wrap(slow, id="slow", run_timeout=0.2), [lookup]),
        arm(harness.wrap(fine, id="fine"), [lookup]),
    ]
    given = {
        "jobs": [
            {"subagent_type": "broken", "description": "a"},
            {"subagent_type": "slow", "description": "b"},
            {"subagent_type": "fine", "description": "c"},
        ]
    }
    script = Routed([("delegate", given), "two failed"])
    agent = lead(harness, script, tools=[lookup, *kids], subagents=kids)
    result = await agent.run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)
    assert [(j["helper"], j["status"]) for j in told["jobs"]] == [
        ("broken", "error"),
        ("slow", "timeout"),
        ("fine", "success"),
    ]
    assert "the helper broke" in told["jobs"][0]["error"]
    assert told["jobs"][2]["answer"] == "facts about c"
    assert told["answered"] == 1


async def test_writers_are_refused_never_run_beside_each_other(harness: Harness) -> None:
    async def writes(input: str, agent: Runtime) -> str:
        return str(await agent.tools.call("file", note=input))

    writer = arm(harness.wrap(writes, id="writer"), [file])

    def tries_to_write(part: str, made: int) -> Any:
        return ("file", {"note": part}) if made == 0 else f"could not file {part}"

    given = {
        "jobs": [
            {"subagent_type": "writer", "description": "file it"},
            {"role": "a clerk", "description": "note A"},
        ]
    }
    script = Routed([("delegate", given), "nothing filed"], tries_to_write)
    agent = lead(harness, script, tools=[lookup, file, writer], subagents=[writer])
    result = await agent.run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)
    assert told["jobs"][0]["status"] == "refused"
    assert "a helper only reads" in told["jobs"][0]["error"]
    assert told["jobs"][1] == {
        "job": 2,
        "helper": "a clerk",
        "status": "success",
        "answer": "could not file note A",
    }
    assert done == []  # nothing was filed: the helper was never given the tool
    [helper] = script.helpers[:1]
    assert "file" not in {t["function"]["name"] for t in helper.get("tools", [])}


@tool(side_effects="read")
def peek(key: str) -> str:
    """Peek at a key (it reads, it says; the catalog says it writes).

    Args:
        key: The key.
    """
    done.append(f"peek {key}")
    return key


async def test_a_call_the_catalog_says_writes_is_refused_in_a_helper(harness: Harness) -> None:
    catalog = FakeCatalog({"peek": Rule(risk="write")})
    with_governance(harness, Governance(catalog))

    def peeks(part: str, made: int) -> Any:
        return ("peek", {"key": part}) if made == 0 else f"no peek at {part}"

    script = Routed([("delegate", jobs("k1")), ("peek", {"key": "parent"}), "done"], peeks)
    result = await lead(harness, script, tools=[peek]).run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    told = [m["content"] for m in script.helpers[-1]["messages"] if m["role"] == "tool"]
    assert told == [ONLY_READS.format(tool="peek")]
    assert done == ["peek parent"]  # the parent's own call runs (announced): not the helper's


async def test_a_helper_does_not_delegate_again(harness: Harness) -> None:
    def nests(part: str, made: int) -> Any:
        return ("delegate", jobs("deeper", "deeper still")) if made == 0 else "did it alone"

    script = Routed([("delegate", jobs("Apple")), "done"], nests)
    result = await lead(harness, script).run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    assert merged(script)["jobs"][0]["answer"] == "did it alone"
    told = [m["content"] for m in script.helpers[-1]["messages"] if m["role"] == "tool"]
    assert "a helper does not delegate" in told[0]
    assert len(await children(harness, result.run_id)) == 1  # no grandchild


async def test_the_budget_is_shared_and_stops_the_swarm(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "ADMISSION", Admission(start=1, ceiling=1, machine=None))
    script = Routed([("delegate", jobs(*ITEMS)), "what I have"])
    agent = lead(harness, script)
    # the delegate call, then one lookup per helper: the budget is spent after two helpers
    result = await agent.run(WIDE, user="ada", budget=Budget(max_tool_calls=3))
    assert result.status is RunStatus.PARTIAL
    told = merged(script)
    # the second helper spent the last call: it answered with what it had (partial)
    assert [j["status"] for j in told["jobs"]] == ["success", "partial", "skipped", "skipped"]
    assert "budget is spent" in told["jobs"][2]["error"]
    assert done == ["lookup Apple", "lookup Microsoft"]


async def test_cancelling_the_run_cancels_its_helpers(harness: Harness) -> None:
    gates["*"] = asyncio.Event()  # never set: the helpers wait in their lookups
    script = Routed([("delegate", jobs(*ITEMS)), "never"])
    agent = lead(harness, script)
    task = asyncio.create_task(agent.run(WIDE, user="ada"))
    await until(lambda: running["now"] == 4)
    [run_id] = [r for r, rt in agent.running.items() if not rt.delegated]
    await agent.cancel(run_id, reason="enough")
    result = await task
    assert result.status is RunStatus.CANCELLED
    kids = await children(harness, run_id)
    assert len(kids) == 4
    assert {k.status for k in kids} == {RunStatus.CANCELLED}
    assert running["now"] == 0 and done == []


async def test_identical_jobs_run_once(harness: Harness) -> None:
    script = Routed([("delegate", jobs("Apple", "Apple", "Google")), "done"])
    result = await lead(harness, script).run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)
    assert [j["answer"] for j in told["jobs"]] == ["found Apple", "found Apple", "found Google"]
    assert sorted(done) == ["lookup Apple", "lookup Google"]
    assert len(await children(harness, result.run_id)) == 2


async def test_jobs_that_cannot_run_say_why(harness: Harness) -> None:
    given = {
        "jobs": [
            {"description": "  "},
            {"subagent_type": "nobody", "description": "x"},
            {"subagent_type": "nobody", "role": "r", "description": "x"},
            "not an object",
            {"description": "fine"},
        ]
    }
    script = Routed([("delegate", given), "done"])
    result = await lead(harness, script).run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)
    errors = [j.get("error") for j in told["jobs"]]
    assert errors[0] == "the job has no description"
    assert errors[1] == "no sub-agent nobody (the sub-agents: none)"
    assert errors[2] == "give subagent_type or role, not both"
    assert errors[3].startswith("a job is an object")
    assert told["jobs"][4]["status"] == "success" and told["jobs"][4]["helper"] == "a helper"


@pytest.mark.parametrize(
    ("args", "said"),
    [
        ({"jobs": []}, "give at least one job"),
        ({"jobs": [{"description": f"p{n}"} for n in range(delegate.MAX_JOBS + 1)]}, "at most"),
    ],
)
async def test_a_call_with_no_job_or_too_many_is_an_error(
    harness: Harness, args: dict[str, Any], said: str
) -> None:
    script = Routed([("delegate", args), "done"])
    result = await lead(harness, script).run(WIDE, user="ada")
    assert result.status is RunStatus.SUCCESS
    [told] = [m for m in script.parents[1]["messages"] if m["role"] == "tool"]
    assert said in told["content"]


@pytest.mark.parametrize(
    ("request_text", "model", "window"),
    [
        ("What is Apple's revenue?", "scripted", ROOM),  # narrow
        ("Look up Apple, then Microsoft, then Google, then Amazon.", "scripted", ROOM),
        ("Update the records of Apple, Microsoft, Google and Amazon.", "scripted", ROOM),
        (WIDE, "granite-3.3-2b-instruct", ROOM),  # a small model
        (WIDE, "qwen2.5-7b", ROOM),
        (WIDE, "scripted", 8_000),  # a small window
    ],
)
async def test_delegate_is_not_offered_to_a_narrow_or_sequential_task_or_a_small_model(
    harness: Harness, request_text: str, model: str, window: int
) -> None:
    script = Routed(["answered"])
    chat = ScriptedChat(script=script, model=model)
    react = ReAct("You answer business questions.", chat, context_window=window)
    agent = arm(harness.wrap(react, id="lead"), [lookup])
    result = await agent.run(request_text, user="ada")
    assert result.status is RunStatus.SUCCESS
    assert "delegate" not in offered(script)


async def test_a_wide_list_is_offered_delegate(harness: Harness) -> None:
    script = Routed(["answered"])
    agent = lead(harness, script, model="gpt-4o")
    asked = "Revenue of:\n- Apple\n- Microsoft\n- Google\n- Amazon"
    assert (await agent.run(asked, user="ada")).status is RunStatus.SUCCESS
    assert "delegate" in offered(script)
    [spec] = [t for t in script.parents[0]["tools"] if t["function"]["name"] == "delegate"]
    job = spec["function"]["parameters"]["properties"]["jobs"]["items"]
    assert set(job["properties"]) == {"description", "role"}  # no sub-agents: no subagent_type


async def test_a_job_that_asks_a_person_pauses_the_run(harness: Harness) -> None:
    async def asks(input: str, agent: Runtime) -> str:
        said = await agent.ask("Which year?")
        return f"{input} in {said}"

    asker = arm(harness.wrap(asks, id="asker"), [lookup])
    given = {
        "jobs": [
            {"subagent_type": "asker", "description": "Apple"},
            {"role": "an analyst", "description": "Google"},
        ]
    }
    script = Routed([("delegate", given), "merged"])
    agent = lead(harness, script, tools=[lookup, asker], subagents=[asker])
    paused = await agent.run(WIDE, user="ada")
    assert paused.status is RunStatus.PAUSED, paused.error
    assert paused.interrupt is not None
    result = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="2024", reviewer="ada"
    )
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)
    assert [j["answer"] for j in told["jobs"]] == ["Apple in 2024", "found Google"]
    assert done.count("lookup Google") == 1  # the finished helper was not run again


class Remote:
    """A run store that is not this process's own (agent-runs, as a test sees it): the
    in-process one, under another type."""

    def __init__(self) -> None:
        self._local = LocalRuns()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._local, name)


async def test_with_agent_runs_the_jobs_queue_for_a_worker() -> None:
    store = Remote()
    async with Harness(config=Settings(), runs=store) as h:  # type: ignore[arg-type]
        script = Routed([("delegate", jobs("Apple", "Google")), "merged"])
        agent = lead(h, script)
        worker = asyncio.create_task(h.worker([agent], concurrency=4).run())
        try:
            result = await agent.run(WIDE, user="ada", budget=Budget(max_tokens=1000))
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        assert result.status is RunStatus.SUCCESS, result.error
        told = merged(script)
        assert [j["answer"] for j in told["jobs"]] == ["found Apple", "found Google"]
        kids = await children(h, result.run_id)
        assert len(kids) == 2
        for kid in kids:
            assert kid.metadata[DELEGATED] == 1
            assert kid.concurrency_key is not None
            assert kid.concurrency_key.startswith(f"delegate:{result.run_id}:")
            assert kid.metadata["budget"]["max_tokens"] <= 500


async def test_a_queued_job_out_of_time_is_cancelled_and_the_run_answers() -> None:
    store = Remote()
    async with Harness(config=Settings(), runs=store) as h:  # type: ignore[arg-type]
        script = Routed([("delegate", jobs("Apple")), "nothing came back"])
        agent = lead(h, script)
        # no worker: the job waits in the queue for half the run's time, then the run merges
        result = await agent.run(WIDE, user="ada", run_timeout=1.0)
        assert result.status is RunStatus.SUCCESS, result.error
        assert result.answer == "nothing came back"
        assert merged(script)["jobs"][0]["status"] == "timeout"
        [kid] = await children(h, result.run_id)
        assert kid.status is RunStatus.CANCELLED


async def test_a_queued_job_that_asks_a_person_is_reported() -> None:
    store = Remote()
    async with Harness(config=Settings(), runs=store) as h:  # type: ignore[arg-type]

        async def asks(input: str, agent: Runtime) -> str:
            return str(await agent.ask("Which year?"))

        asker = arm(h.wrap(asks, id="asker"), [lookup])
        given = {"jobs": [{"subagent_type": "asker", "description": "Apple"}]}
        script = Routed([("delegate", given), "merged"])
        agent = lead(h, script, tools=[lookup, asker], subagents=[asker])
        worker = asyncio.create_task(h.worker([agent, asker]).run())
        try:
            result = await agent.run(WIDE, user="ada")
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        assert result.status is RunStatus.SUCCESS, result.error
        [job] = merged(script)["jobs"]
        assert job["status"] == "paused" and "asked a person" in job["error"]


async def test_a_job_out_of_time_is_reported_and_the_run_answers(harness: Harness) -> None:
    gates["*"] = asyncio.Event()  # never set: the helper's lookup waits
    script = Routed([("delegate", jobs("Apple")), "out of time"])
    result = await lead(harness, script).run(WIDE, user="ada", run_timeout=1.0)
    assert result.status is RunStatus.SUCCESS, result.error
    assert merged(script)["jobs"][0]["status"] == "timeout"


async def test_a_job_that_raises_is_reported(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the store is down")

    monkeypatch.setattr(delegate, "child_run", broken)
    script = Routed([("delegate", jobs("Apple")), "failed"])
    assert (await lead(harness, script).run(WIDE, user="ada")).status is RunStatus.SUCCESS
    [job] = merged(script)["jobs"]
    assert job == {
        "job": 1,
        "helper": "an analyst",
        "status": "error",
        "error": "RuntimeError: the store is down",
    }


async def test_outside_a_run_delegate_is_an_error() -> None:
    from langchain_core.tools import ToolException

    with pytest.raises(ToolException, match="inside a harness run"):
        await delegate._delegated([{"description": "x"}])


async def test_a_job_with_no_slot_before_the_time_is_up_is_reported(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "ADMISSION", Admission(start=1, ceiling=1, machine=None))
    gates["Apple"] = asyncio.Event()  # never set: the first job holds the only slot
    script = Routed([("delegate", jobs("Apple", "Google")), "out of time"])
    result = await lead(harness, script).run(WIDE, user="ada", run_timeout=1.0)
    assert result.status is RunStatus.SUCCESS, result.error
    told = merged(script)["jobs"]
    assert [j["status"] for j in told] == ["timeout", "timeout"]
    assert told[1]["error"] == "the run's time ran out before it ran"
    assert done == []


async def test_cancelling_the_run_cancels_its_queued_jobs() -> None:
    store = Remote()
    async with Harness(config=Settings(), runs=store) as h:  # type: ignore[arg-type]
        script = Routed([("delegate", jobs("Apple", "Google")), "never"])
        agent = lead(h, script)
        task = asyncio.create_task(agent.run(WIDE, user="ada"))  # no worker: they wait
        found: list[RunRecord] = []

        async def queued() -> bool:
            runs = [r for r, rt in agent.running.items() if not rt.delegated]
            if runs:
                found[:] = await children(h, runs[0])
            return len(found) == 2

        async with asyncio.timeout(10):
            while not await queued():  # noqa: ASYNC110 - the run store says so
                await asyncio.sleep(0.01)
        [run_id] = [r for r, rt in agent.running.items() if not rt.delegated]
        await agent.cancel(run_id)
        assert (await task).status is RunStatus.CANCELLED
        assert {k.status for k in await children(h, run_id)} == {RunStatus.CANCELLED}


@tool(side_effects="read")
async def confirm_year(company: str) -> str:
    """Ask a person which fiscal year to report for a company.

    Args:
        company: The company.
    """
    runtime = current()
    assert runtime is not None
    year = await runtime.ask(f"Which year for {company}?")
    done.append(f"year {company} {year}")
    return f"{company}: {year}"


async def test_a_helper_paused_mid_job_resumes_without_running_its_calls_again(
    harness: Harness,
) -> None:
    """The step-tagged journal: a helper (a ReAct graph of its own) that looked up, then asked
    a person, continues from its checkpoint when the run resumes — its lookup is not made
    again, nor is the other helper, which had finished."""

    def asks(part: str, made: int) -> Any:
        if made == 0:
            return ("lookup", {"topic": part})
        if made == 1 and part == "Apple":
            return ("confirm_year", {"company": part})
        return f"found {part}"

    script = Routed([("delegate", jobs("Apple", "Google")), "merged"], asks)
    agent = lead(harness, script, tools=[lookup, confirm_year])
    paused = await agent.run(WIDE, user="ada")
    assert paused.status is RunStatus.PAUSED, paused.error
    assert paused.interrupt is not None
    assert sorted(done) == ["lookup Apple", "lookup Google"]
    result = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="2024", reviewer="ada"
    )
    assert result.status is RunStatus.SUCCESS, result.error
    assert sorted(done) == ["lookup Apple", "lookup Google", "year Apple 2024"]
    assert [j["status"] for j in merged(script)["jobs"]] == ["success", "success"]
