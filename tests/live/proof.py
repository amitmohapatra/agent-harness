"""What a model-driven live test proves: what the harness did, not what the model said.

A small local model is slow and unreliable at choosing tools, so a live run is checked for
the harness's part of it — the context it pushed, the tools it offered, the calls it ran
(each ended on the stream, journaled, decided by governance), and that the run ended in a
final status with its events, within its time limit — never for the answer being right.
Where the model made none of the calls a test needs, the same target is run again with a
scripted model (its *twin*, ``tests.support.adapters``) against the same live services, so
the harness's tool path is still proven end to end.

The time limits come from the environment: ``TRELLIS_LIVE_TIMEOUT``, the most a live run may
take (default :data:`DEFAULT_RUN_SECONDS`); a test may take two runs and the services' settling
on top (:data:`TEST_SECONDS`). ``TRELLIS_LIVE_MAX_TOKENS`` caps each model reply (default
:data:`DEFAULT_MAX_TOKENS`: a tool call or a short answer)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar

import httpx

from trellis import Harness, Hooks, Result, Runtime
from trellis.contracts import RunEvent, RunEventType, RunOutcome, ToolCall, ToolOutcome
from trellis.harness.governance import Decision
from trellis.harness.journal import content_key

_Client = TypeVar("_Client")

DEFAULT_RUN_SECONDS: Final = 600.0
DEFAULT_MAX_TOKENS: Final = 200
#: how long the services may take to settle after a run (writes, the catalog's statistics)
SETTLE_SECONDS: Final = 120.0


def _number(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


#: The most one live run may take (its ``timeout``): past it the run ends ``TIMEOUT``.
RUN_SECONDS: Final = _number("TRELLIS_LIVE_TIMEOUT", DEFAULT_RUN_SECONDS)
#: The most a test may take: a live run, its twin, and the services settling.
TEST_SECONDS: Final = int(2 * RUN_SECONDS + SETTLE_SECONDS)
#: The longest model reply.
MAX_TOKENS: Final = int(_number("TRELLIS_LIVE_MAX_TOKENS", DEFAULT_MAX_TOKENS))
#: The errors a run ends with on its model's account, not the harness's: its framework
#: stopped a model that kept calling tools (turns, steps), or the model's server failed on
#: what the model wrote — a local llama.cpp server answers ``500`` to a conversation holding
#: a tool call whose arguments are not JSON, the model's own from the turn before (the
#: harness told the model so, and went on) — or did not answer in time.
MODEL_FAILURES: Final = frozenset(
    {
        "MaxTurnsExceeded",
        "GraphRecursionError",
        "InternalServerError",  # the openai client's 500
        "ServerError",  # the gateway client's (ReAct with a model name)
        "APITimeoutError",
    }
)


class Wire:
    """What the model was sent: each chat request's body, as the gateway got it (an ``httpx``
    request hook on the model's own client)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, request: Any) -> None:
        if request.url.path.endswith("/chat/completions"):
            self.requests.append(json.loads(request.content))

    def client(self, kind: Callable[..., _Client] = httpx.AsyncClient) -> _Client:
        """An HTTP client for the model (``kind``: the one its SDK takes) that records here,
        waiting as long as a run may."""
        return kind(timeout=RUN_SECONDS, event_hooks={"request": [self]})

    def offered(self) -> set[str]:
        """Every tool name any request offered the model."""
        return {
            (t.get("function") or t).get("name")
            for request in self.requests
            for t in request.get("tools") or []
        }

    def unparsed(self) -> int:
        """How many of the model's tool calls, as the last request carried them back, have
        arguments that are not JSON."""
        count = 0
        for message in (self.requests[-1].get("messages") or []) if self.requests else []:
            for call in message.get("tool_calls") or []:
                try:
                    json.loads((call.get("function") or {}).get("arguments") or "{}")
                except ValueError:
                    count += 1
        return count

    def said(self) -> str:
        """Everything the model was sent, as one text."""
        return json.dumps([r.get("messages") for r in self.requests])


class Proof(Hooks):
    """The run as the harness ran it, read from its own hooks: the context it pushed, the
    calls it ran and their outcomes, and its journal as the run ended."""

    def __init__(self) -> None:
        self.context: str | None = None
        self.calls: list[tuple[ToolCall, ToolOutcome]] = []
        self.trajectory: list[tuple[ToolCall, ToolOutcome]] = []
        self.journaled: dict[str, list[Any]] = {}
        self.result: Result | None = None

    async def on_run_start(self, run: Runtime) -> None:
        self.context = run.context

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        self.calls.append((call, outcome))
        return outcome

    async def on_run_end(self, run: Runtime, result: Result) -> None:
        journal = run.replay.journal
        self.trajectory = list(journal.trajectory)
        self.journaled = {k: list(v) for k, v in journal.calls.items()}
        self.result = result


def governed(h: Harness, tenant: str) -> list[tuple[str, Decision]]:
    """Every decision governance makes in ``tenant`` from now on — the harness's own (the
    memory service's catalog), only watched."""
    governance = h.governance(tenant)
    seen: list[tuple[str, Decision]] = []
    check = governance.check

    async def watched(tool: str, args: Mapping[str, Any], **kwargs: Any) -> Decision:
        decision = await check(tool, args, **kwargs)
        seen.append((tool, decision))
        return decision

    governance.check = watched  # type: ignore[method-assign]
    return seen


@dataclass
class Run:
    """One run's events, as streamed, and how long it took."""

    name: str
    events: list[RunEvent] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def finished(self) -> RunEvent:
        return self.events[-1]

    def started(self) -> list[str]:
        """The tools of the calls the run started, in order."""
        return [e.data["tool"] for e in self.events if e.type is RunEventType.TOOL_CALL_START]

    def results(self) -> list[dict[str, Any]]:
        return [e.data for e in self.events if e.type is RunEventType.TOOL_CALL_RESULT]

    def summary(self, wire: Wire | None = None) -> str:
        finished = self.finished
        outcome = finished.outcome.value if finished.outcome else None
        sent = (
            f", {len(wire.requests)} model calls ({wire.unparsed()} with arguments not JSON)"
            if wire is not None
            else ""
        )
        error = (
            f" ({finished.error.code}: {finished.error.message[:120]})" if finished.error else ""
        )
        return (
            f"{self.name}: {outcome}{error} in {self.seconds:.1f}s (limit {RUN_SECONDS:g}s)"
            f"{sent}, calls {self.started()}"
        )


async def streamed(name: str, events: AsyncIterator[RunEvent]) -> Run:
    """The run of ``events`` (``agent.stream(...)``), timed."""
    run, started = Run(name), time.monotonic()
    run.events = [e async for e in events]
    run.seconds = time.monotonic() - started
    print(f"\n[live] {run.summary()}")
    return run


def ended(run: Run) -> None:
    """The run ended in a final status, with its events in order: ``RUN_STARTED`` first,
    ``RUN_FINISHED`` last (an error with its ``RUN_ERROR``), every call started ended once
    with a result, and within its time limit (a little over: the run's end is written)."""
    events = run.events
    assert events and events[0].type is RunEventType.RUN_STARTED, events[:1]
    finished = run.finished
    assert finished.type is RunEventType.RUN_FINISHED, finished
    assert finished.outcome in (RunOutcome.SUCCESS, RunOutcome.ERROR, RunOutcome.TIMEOUT)
    assert [e.sequence for e in events] == sorted(e.sequence for e in events)
    if finished.outcome is not RunOutcome.SUCCESS:
        assert any(e.type is RunEventType.RUN_ERROR for e in events), run.summary()
    ids = {
        kind: [e.tool_call_id for e in events if e.type is kind]
        for kind in (
            RunEventType.TOOL_CALL_START,
            RunEventType.TOOL_CALL_END,
            RunEventType.TOOL_CALL_RESULT,
        )
    }
    starts = ids[RunEventType.TOOL_CALL_START]
    for kind in (RunEventType.TOOL_CALL_END, RunEventType.TOOL_CALL_RESULT):
        assert sorted(map(str, starts)) == sorted(map(str, ids[kind])), run.summary()
    assert len(set(starts)) == len(starts)
    assert run.seconds < RUN_SECONDS + 60, run.summary()


def harness_ok(run: Run) -> None:
    """The run did not fail on the harness's account: it succeeded, ran out of its time (a
    slow model: the harness ended it on its limit), or failed on its model's
    (:data:`MODEL_FAILURES`). Anything else — a tool call that broke the run, a request the
    gateway refused — is the harness's."""
    finished = run.finished
    if finished.outcome is RunOutcome.SUCCESS or finished.outcome is RunOutcome.TIMEOUT:
        return
    assert finished.error is not None and finished.error.code in MODEL_FAILURES, run.summary()


def recorded(run: Run, proof: Proof, decisions: list[tuple[str, Decision]]) -> None:
    """Every call the run started ran through the harness: its result on the stream is the
    outcome its hooks saw, it is on the run's journal (its trajectory, and its output kept
    for a re-run when it succeeded), and governance decided it first."""
    ran = [(call.tool, outcome.status.value) for call, outcome in proof.calls]
    on_stream = [(r["tool"], r["status"]) for r in run.results()]
    assert sorted(on_stream) == sorted(ran), (on_stream, ran)
    assert [c.tool for c, _ in proof.trajectory] == [c.tool for c, _ in proof.calls]
    for call, outcome in proof.calls:
        if outcome.ok:  # a re-run of the run gets this output, and runs nothing
            assert content_key("call", call.tool, call.args) in proof.journaled, call
    decided = [tool for tool, _ in decisions]
    for call, _ in proof.calls:
        assert call.tool in decided, (call.tool, decided)
