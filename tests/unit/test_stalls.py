"""The stall watchdog (``Runtime.model_started``): a model call that gets no answer within its
bound ends the attempt with a ``ModelTimeout``; one that answers, fails or is watched anew
does not; the watchdogs end with the attempt; and the LangChain callbacks that watch every chat
model call of a run (``accounting``) start and end it."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from trellis.contracts import AgentExecutionContext, ErrorCategory, RunEvent
from trellis.harness import accounting, pipeline
from trellis.harness.events import RunEvents
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal, Replay
from trellis.harness.runtime import STALLED, ModelTimeout, Runtime, _current


def _runtime() -> tuple[Runtime, list[RunEvent]]:
    seen: list[RunEvent] = []
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="a"))
    events.listen(seen.append)
    agent: Any = SimpleNamespace(adapter=SimpleNamespace(name="langgraph"), without=frozenset())
    identity = Identity(tenant="t", agent_id="a", run_id="run_1", user="u")
    runtime = Runtime(identity=identity, agent=agent, events=events, replay=Replay(Journal()))
    return runtime, seen


async def test_a_call_with_no_answer_within_its_bound_cancels_the_attempt_as_stalled(
    spans: Any,
) -> None:
    runtime, seen = _runtime()

    async def attempt() -> None:
        runtime.running_in = asyncio.current_task()
        runtime.model_started("step-1", 0.01)
        await asyncio.sleep(5)

    task = asyncio.create_task(attempt())
    with pytest.raises(asyncio.CancelledError) as cut:
        await task
    assert STALLED in cut.value.args
    stalled = runtime.stalled
    assert isinstance(stalled, ModelTimeout) and stalled.category is ErrorCategory.TIMEOUT
    assert stalled.details == {"step": "step-1", "seconds": 0.01} and stalled.retryable
    assert stalled.source == "langgraph"
    assert [e.data["code"] for e in seen] == ["model_timeout"]
    [traced] = spans.get_finished_spans()
    assert traced.name == "model_timeout" and traced.attributes["trellis.step"] == "step-1"


async def test_an_answered_call_is_no_stall_and_a_key_watched_again_starts_over() -> None:
    runtime, seen = _runtime()
    runtime.running_in = asyncio.current_task()
    runtime.model_started("a", 0.02)
    runtime.model_started("a", 0.2)  # watched anew: the first watchdog is off
    runtime.model_ended("never-started")  # nothing to end
    await asyncio.sleep(0.05)
    runtime.model_ended("a")
    await asyncio.sleep(0.2)
    assert runtime.stalled is None and seen == []


async def test_the_watchdogs_end_with_the_attempt_and_a_late_one_does_nothing() -> None:
    runtime, _ = _runtime()
    runtime.running_in = asyncio.current_task()
    runtime.model_started("a", 0.01)
    runtime.model_started("b", 0.01)
    runtime.unwatched()
    await asyncio.sleep(0.03)
    assert runtime.stalled is None
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    runtime.running_in = done
    runtime._stall("late", 1.0)  # the attempt ended: nothing to stop
    runtime.running_in = None
    runtime._stall("none", 1.0)
    assert runtime.stalled is None


async def test_a_second_stall_is_not_said_twice() -> None:
    runtime, seen = _runtime()
    runtime.running_in = asyncio.current_task()
    runtime.stalled = ModelTimeout("already")
    runtime._stall("again", 1.0)
    assert seen == [] and runtime.stalled.message == "already"


async def test_a_stall_the_framework_swallowed_or_wrapped_is_still_the_attempts_error() -> None:
    runtime, _ = _runtime()
    runtime.stalled = ModelTimeout("stalled")
    task = asyncio.current_task()
    assert task is not None
    task.cancel()  # the watchdog's cancellation, which the framework did not let through
    assert pipeline._stalled(runtime) is runtime.stalled
    assert task.cancelling() == 0  # taken back
    assert pipeline._stalled(runtime) is runtime.stalled  # nothing left to take back
    runtime.cancelled = "by a person"
    assert pipeline._stalled(runtime) is None  # a cancelled run's ending is its cancellation
    runtime.cancelled, runtime.stalled = None, None
    assert pipeline._stalled(runtime) is None


async def test_every_langchain_model_call_of_a_run_is_watched() -> None:
    runtime, _ = _runtime()
    calls: list[tuple[str, str]] = []
    runtime.model_started = lambda key, seconds=None: calls.append(("started", key))  # type: ignore[method-assign]
    runtime.model_ended = lambda key: calls.append(("ended", key))  # type: ignore[method-assign]
    callbacks = accounting.UsageAccounting()
    chat, text = uuid.uuid4(), uuid.uuid4()
    token = _current.set(runtime)
    try:
        callbacks.on_chat_model_start({}, [[]], run_id=chat)
        callbacks.on_llm_start({}, ["prompt"], run_id=text)
        callbacks.on_llm_error(RuntimeError("down"), run_id=text)
    finally:
        _current.reset(token)
    assert calls == [("started", str(chat)), ("started", str(text)), ("ended", str(text))]
    calls.clear()  # outside a run: nothing is watched
    callbacks.on_chat_model_start({}, [[]], run_id=chat)
    callbacks.on_llm_start({}, ["prompt"], run_id=text)
    callbacks.on_llm_error(RuntimeError("down"), run_id=text)
    assert calls == []
