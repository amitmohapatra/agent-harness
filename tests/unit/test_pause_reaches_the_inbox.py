"""A paused run has to tell a person what it is asking.

The run store's PAUSED rows are the human inbox: `GET /v1/runs?status=PAUSED` is what a UI
renders, and agent-runs delivers the same `awaiting` payload by webhook. Only the exception's
class name used to get that far, so the inbox showed "GraphInterrupt" and nothing about the
question — the one thing it exists to show.
"""

from __future__ import annotations

from universal_agent_contracts import AgentPaused
from universal_agent_harness.runs.client import _asked


def test_our_own_pause_carries_its_question() -> None:
    asked = _asked(
        AgentPaused(
            "Approve a EUR 240 refund for order 91?",
            expects={"type": "boolean"},
            payload={"order_id": 91},
        )
    )
    assert asked == {
        "question": "Approve a EUR 240 refund for order 91?",
        "expects": {"type": "boolean"},
        "payload": {"order_id": 91},
    }


def test_a_langgraph_interrupt_is_read_structurally() -> None:
    """LangGraph puts interrupt(value) in the exception args as objects with .value.
    Read by shape, not by importing langgraph — which this package must not depend on."""

    class Interrupt:
        def __init__(self, value: object) -> None:
            self.value = value

    class GraphInterrupt(Exception):
        pass

    signal = GraphInterrupt(Interrupt({"question": "Ship it?"}))
    assert _asked(signal) == {"question": {"question": "Ship it?"}}


def test_several_interrupts_are_all_carried() -> None:
    class Interrupt:
        def __init__(self, value: object) -> None:
            self.value = value

    class GraphInterrupt(Exception):
        pass

    signal = GraphInterrupt(Interrupt("first?"), Interrupt("second?"))
    assert _asked(signal) == {"question": ["first?", "second?"]}


def test_a_signal_that_says_nothing_records_nothing() -> None:
    """The reason alone is still recorded; this must not invent a question."""

    class NodeInterrupt(Exception):
        pass

    assert _asked(NodeInterrupt()) is None
    assert _asked(NodeInterrupt("suspended")) is None, "a bare string is not a question"
    assert _asked(None) is None


def test_a_broken_signal_does_not_turn_a_pause_into_a_crash() -> None:
    """A pause is already the delicate path — the run is suspended and waiting on a person.
    An exception raised while reading the question must not take that down."""

    class Hostile(Exception):
        def awaiting(self):
            raise RuntimeError("no")

    assert _asked(Hostile()) is None


def test_a_non_dict_awaiting_is_ignored() -> None:
    class Odd(Exception):
        def awaiting(self):
            return "not a dict"

    assert _asked(Odd()) is None
