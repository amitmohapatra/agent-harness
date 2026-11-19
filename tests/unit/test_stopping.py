"""``stopping``: the one owner of stopping an attempt, and of reading that the run store ended
a run while this process worked on it."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from trellis import Harness
from trellis.contracts import RunRecord, RunStatus
from trellis.harness import stopping
from trellis.harness.asking import RunCancelled
from trellis.harness.runtime import Runtime
from trellis.runs import ConflictError, LeaseLostError, RunsError

ENDED = (RunStatus.SUCCESS, RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.PARTIAL)


def record(status: RunStatus) -> RunRecord:
    """A record with ``status`` and nothing else (unchecked: a paused one names no interrupt)."""
    return RunRecord.model_construct(
        run_id="run_1", tenant_id="default", agent_id="chat", status=status
    )


@pytest.mark.parametrize("status", [RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED])
def test_a_run_the_store_has_not_ended_goes_on(status: RunStatus) -> None:
    assert stopping.ended_elsewhere(record(status)) is None


def test_a_run_the_store_does_not_have_stops_nothing() -> None:
    assert stopping.ended_elsewhere(None) is None


def test_a_run_cancelled_in_the_store_stops_as_cancelled_elsewhere() -> None:
    assert stopping.ended_elsewhere(record(RunStatus.CANCELLED)) == stopping.CANCELLED_ELSEWHERE


@pytest.mark.parametrize("status", ENDED)
def test_a_run_the_store_ended_otherwise_stops_as_ended_elsewhere(status: RunStatus) -> None:
    assert stopping.ended_elsewhere(record(status)) == stopping.ENDED_ELSEWHERE


async def test_an_ending_write_the_store_took_was_made() -> None:
    made: list[str] = []

    async def write() -> None:
        made.append("written")

    assert await stopping.unless_ended(write()) is True and made == ["written"]


@pytest.mark.parametrize("refused", [ConflictError, LeaseLostError])
async def test_an_ending_write_the_store_refused_is_nobodys_to_make(
    refused: type[RunsError],
) -> None:
    async def write() -> None:
        raise refused("run_1 already ended", code="CONFLICT", status=409)

    assert await stopping.unless_ended(write()) is False


async def test_an_ending_write_that_fails_otherwise_fails_its_caller() -> None:
    async def write() -> None:
        raise RuntimeError("the run store is away")

    with pytest.raises(RuntimeError, match="is away"):
        await stopping.unless_ended(write())


async def test_an_attempt_stopped_twice_keeps_the_first_reason_and_is_cancelled_once() -> None:
    cancels: list[str] = []

    class Working:
        def cancel(self) -> None:
            cancels.append("cancelled")

    working: Any = Working()
    runtime = cast(Runtime, SimpleNamespace(cancelled=None, running_in=working))
    assert stopping.stop_attempt(runtime, "the first") is working
    assert stopping.stop_attempt(runtime, "the second") is working
    assert runtime.cancelled == "the first" and cancels == ["cancelled"]
    await asyncio.sleep(0)


async def test_a_cancel_raised_inside_a_stopped_attempt_keeps_the_stops_reason(
    harness: Harness,
) -> None:
    """A run stopped for a reason whose framework then says its own generic cancel (a live
    session's bridge: ``RunCancelled``) ends with the reason it was stopped for."""

    async def stopped_then_cancelled(input: str, agent: Runtime) -> str:
        agent.cancelled = "cancelled by Ada"
        raise RunCancelled("the session stopped")

    async def cancelled(input: str, agent: Runtime) -> str:
        raise RunCancelled("the session stopped")

    async def reason(target: Any, name: str) -> Any:
        events = [e async for e in harness.wrap(target, id=name).stream("x", user="ada")]
        assert events[-1].outcome == "cancelled"
        return events[-1].data["reason"]

    assert await reason(stopped_then_cancelled, "stopped") == "cancelled by Ada"
    assert await reason(cancelled, "cancelled") == "the session stopped"
