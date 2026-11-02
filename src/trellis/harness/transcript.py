"""A run's transcript — any run's, a sub-agent's included — read from its run log (agent-runs'
event log, which every attempt writes: :mod:`trellis.harness.runlog`), a page at a time.

The log already says everything the transcript shows, so nothing new is recorded: the
assistant's messages (``TEXT_MESSAGE_*``, by message id), each tool call with its arguments
and its result (``TOOL_CALL_START``/``ARGS``/``RESULT``, by call id; the result as the stream
previews it, ``events.PREVIEW_CHARS``), and how the run ended (``RUN_FINISHED``, whose
``result`` is the answer of a run that streamed no text). A page is the
log's events past a position (``after``), at most ``limit``; an item a page cuts in two comes
back in both pages under one ``id``, the later part completing the earlier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from trellis.contracts import RunEvent, RunEventType, RunRecord

#: The most events a transcript page reads (agent-runs' page limit).
PAGE: Final = 500


def folded(events: Sequence[RunEvent]) -> list[dict[str, Any]]:
    """``events`` (in the log's order) as the transcript's items: messages, tool calls, the
    end."""
    items: list[dict[str, Any]] = []
    open_: dict[str, dict[str, Any]] = {}

    def item(key: str, **fresh: Any) -> dict[str, Any]:
        found = open_.get(key)
        if found is None:
            found = open_[key] = fresh
            items.append(found)
        return found

    for event in events:
        kind = event.type
        if kind in (RunEventType.TEXT_MESSAGE_START, RunEventType.TEXT_MESSAGE_CONTENT):
            said = item(
                f"m:{event.message_id}",
                type="message",
                id=event.message_id,
                role=event.data.get("role", "assistant"),
                text="",
            )
            said["text"] += event.data.get("delta") or ""
        elif kind in (
            RunEventType.TOOL_CALL_START,
            RunEventType.TOOL_CALL_ARGS,
            RunEventType.TOOL_CALL_RESULT,
        ):
            call = item(f"c:{event.tool_call_id}", type="tool_call", id=event.tool_call_id)
            if "tool" in event.data:
                call["tool"] = event.data["tool"]
            if kind is RunEventType.TOOL_CALL_ARGS:
                call["args"] = event.data.get("args")
            elif kind is RunEventType.TOOL_CALL_RESULT:
                call["result"] = event.data.get("output")
                call["status"] = event.data.get("status")
        elif kind is RunEventType.RUN_FINISHED and event.outcome is not None:
            answer = event.data.get("result")
            said_last = next((i["text"] for i in reversed(items) if i["type"] == "message"), None)
            if isinstance(answer, str) and answer and answer != said_last:
                # a run not streamed logs no text: its answer is on its end
                items.append({"type": "message", "role": "assistant", "text": answer})
            items.append({"type": "end", "outcome": event.outcome.value})
    return items


async def transcript(
    store: Any, record: RunRecord, *, after: int = 0, limit: int = PAGE
) -> dict[str, Any]:
    """A page of the transcript of ``record``'s run: the run, its question and status, its
    items past position ``after`` of its log, the position to read on from (``after``) and
    whether there is more (``more``: a full page, or a run still going)."""
    entries = await store.events(record.run_id, after=after, limit=limit, tenant=record.tenant_id)
    last = entries[-1].position if entries else after
    return {
        "run_id": record.run_id,
        "agent_id": record.agent_id,
        "parent_run_id": record.parent_run_id,
        "status": record.status.value,
        "input": record.input,
        "items": folded([entry.event for entry in entries]),
        "after": last,
        "more": len(entries) == limit or not record.status.final,
    }


__all__ = ["PAGE", "folded", "transcript"]
