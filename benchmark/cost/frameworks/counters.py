"""What the ReAct items are measured by beyond the cost benchmark's row, counted in the arm's
own process per run:

* ``ReAct``'s loop events (``trellis.react.loop``: malformed calls, repairs, strict schemas
  turned on) — each call of the harness's counter counted here too, while
  :func:`loop_counts` is open (:class:`LoopCounts`);
* the answer tool's calls (:class:`AnswerTurns`): the model calls that answered ``output=``
  through LangChain's answer tool (``ToolStrategy``), one turn more than a native answer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from trellis.harness import telemetry
from trellis.harness.telemetry import LoopEvent


class LoopCounts:
    """The harness's loop events while it is open (:func:`loop_counts`), by event: every
    ``telemetry.metrics.loop`` call counted here too (the counter itself still counts)."""

    def __init__(self) -> None:
        self.seen: Counter[str] = Counter()

    def now(self) -> dict[str, int]:
        """Every event's count so far."""
        return {event.value: self.seen[event.value] for event in LoopEvent}

    @staticmethod
    def since(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        """The events between two readings."""
        return {k: after[k] - before.get(k, 0) for k in after}


@contextmanager
def loop_counts() -> Iterator[LoopCounts]:
    """A :class:`LoopCounts` of the loop events this process counts until the block ends."""
    counts = LoopCounts()
    metrics = telemetry.metrics
    shadowed = vars(metrics).get("loop")
    counting = shadowed or type(metrics).loop

    def loop(event: LoopEvent) -> None:
        counts.seen[event.value] += 1
        counting(event)

    metrics.loop = loop  # type: ignore[method-assign]
    try:
        yield counts
    finally:
        if shadowed is None:
            del metrics.loop
        else:
            metrics.loop = shadowed  # type: ignore[method-assign]


class AnswerTurns(BaseCallbackHandler):
    """The model replies that call one of ``names`` (the answer tool of ``output=``)."""

    run_inline = True

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names
        self.calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                called = getattr(message, "tool_calls", None) or []
                self.calls += sum(1 for c in called if c.get("name") in self.names)


_TURNS: ContextVar[AnswerTurns | None] = ContextVar("fwbench_answer_turns", default=None)
register_configure_hook(_TURNS, inheritable=True)


@contextmanager
def answer_turns(output: Any) -> Iterator[AnswerTurns]:
    """An :class:`AnswerTurns` of every chat model call in this context, for the answer tool
    of ``output`` (a pydantic model: the tool takes its class's name; none without one)."""
    name = getattr(output, "__name__", None)
    turns = AnswerTurns(frozenset({name} if name else ()))
    token = _TURNS.set(turns)
    try:
        yield turns
    finally:
        _TURNS.reset(token)
