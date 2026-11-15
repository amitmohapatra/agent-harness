"""Turns of a wrapped agent, every pause answered the way the case says, records collected."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from common import harness, records, uid

#: how a case answers a pause: (interrupt) -> (decision, answer) or None to stop
Decide = Callable[[Any], tuple[str, Any] | None]


def approve(_: Any) -> tuple[str, Any]:
    return ("approve", None)


async def wrapped(
    target: Any,
    arm: str,
    agent_id: str,
    questions: Sequence[str],
    *,
    decide: Decide = approve,
    thread: str | None = "same",
    wrap: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    pauses: int = 4,
    state_of: Any = None,
) -> dict[str, Any]:
    """Each question a run (one conversation unless ``thread=None``), its pauses answered by
    ``decide``; the answers, statuses, the questions asked and (``stack``) the records."""
    user = f"{agent_id}-{uid()}"
    conv = uid() if thread == "same" else thread
    answers: list[Any] = []
    statuses: list[str] = []
    asked: list[dict[str, Any]] = []
    ids: list[str] = []
    errors: list[str | None] = []
    async with harness(arm) as h:
        agent = h.wrap(target, id=agent_id, **(wrap or {}))
        for q in questions:
            r = await agent.run(q, user=user, thread=conv, **(run or {}))
            for _ in range(pauses):
                if r.interrupt is None:
                    break
                i = r.interrupt
                asked.append(
                    {
                        "reason": str(i.reason),
                        "question": i.question,
                        "tool_call": i.tool_call.model_dump(mode="json") if i.tool_call else None,
                        "options": [str(o) for o in i.options][:6],
                    }
                )
                choice = decide(i)
                if choice is None:
                    break
                r = await agent.resume(i.interrupt_id, choice[0], answer=choice[1], reviewer="lee")
            answers.append(r.answer)
            statuses.append(str(r.status))
            errors.append(str(r.error) if r.error else None)
            ids.append(r.run_id)
        recs = [records(i) for i in ids] if arm == "stack" else None
        state: dict[str, Any] = {}
        if state_of is not None and conv is not None:
            key = await h.conversation(user=user, thread=conv)
            snap = await state_of.aget_state({"configurable": {"thread_id": key}})
            state = {
                "todos": snap.values.get("todos"),
                "files": sorted((snap.values.get("files") or {}).keys()),
            }
    return {
        "answers": answers,
        "statuses": statuses,
        "asked": asked,
        "run_ids": ids,
        "errors": errors,
        "user": user,
        "records": recs,
        **state,
    }
