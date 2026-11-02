"""``transcript.folded``: a streamed run's messages, by id, and its answer said once."""

from __future__ import annotations

from typing import Any

from trellis.contracts import RunEvent, RunEventType, RunOutcome
from trellis.harness.transcript import folded


def _event(n: int, kind: RunEventType, **fields: Any) -> RunEvent:
    return RunEvent(tenant_id="t", run_id="r", sequence=n, type=kind, **fields)


def test_a_streamed_message_is_one_item_and_the_answer_is_not_repeated() -> None:
    events = [
        _event(1, RunEventType.TEXT_MESSAGE_START, message_id="m", data={"role": "assistant"}),
        _event(2, RunEventType.TEXT_MESSAGE_CONTENT, message_id="m", data={"delta": "hel"}),
        _event(3, RunEventType.TEXT_MESSAGE_CONTENT, message_id="m", data={"delta": "lo"}),
        _event(4, RunEventType.TEXT_MESSAGE_END, message_id="m", data={}),
        _event(5, RunEventType.RUN_FINISHED, outcome=RunOutcome.SUCCESS, data={"result": "hello"}),
    ]
    assert folded(events) == [
        {"type": "message", "id": "m", "role": "assistant", "text": "hello"},
        {"type": "end", "outcome": "success"},
    ]
