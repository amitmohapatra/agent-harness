"""Scenario 8 — schedules made in chat (SCH): the user asks for a report later; the schedule fires
while the user is chatting, a worker runs it, and its reply lands in the right thread — heard at
once on the open chat's thread stream (AG-UI ``/threads/{thread}/events``), and kept in the
conversation for a chat that is closed. One case waits for agent-runs' ticker itself (a real
minute or two: ``in_minutes=1`` plus the schedule's stable 0-59 s spread); the others fire the
schedule at once (``POST /v1/schedules/{id}/fire``), the same path from the queued run on.
A schedule in a zone with daylight saving keeps its wall-clock hour across the change."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from tests.scenarios.brains import command, parse, token, wrapped
from tests.scenarios.kit import TOOLS, Served, eventually, free_port, history
from trellis import Harness
from trellis.contracts import RunStatus
from trellis.harness.agui.sse import decode

pytestmark = [pytest.mark.timeout(900), pytest.mark.plan("SCH")]


async def _schedules(h: Harness, user: str, agent_id: str) -> list:
    return [
        s for s in (await h.runs.schedules.list(on_behalf_of=user)).items if s.agent_id == agent_id
    ]


@pytest.mark.parametrize(
    ("adapter", "how"),
    [
        ("react", "ticker"),
        ("react", "fire"),
        ("langgraph", "fire"),
        ("openai_agents", "fire"),
        ("function", "fire"),
        ("kimi", "fire"),
    ],
)
async def test_a_schedule_made_in_chat_fires_into_its_thread(
    h: Harness, adapter: str, how: str, uid: str, tmp_path: Path
) -> None:
    agent_id = f"s8-{adapter}-{uid}"
    port = free_port()
    spec = {
        "mode": "both",
        "port": port,
        "agents": [{"adapter": adapter, "id": agent_id}],
        "lease": 10,
        "hold": 10,
    }
    served = await Served(spec, tmp_path, "srv").started()
    user, open_thread, closed_thread = f"ada-{uid}", f"open-{uid}", f"closed-{uid}"
    report = token(f"r{uid[:5]}", 9)
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", headers={"x-user": user}, timeout=300
        ) as c:

            async def say(thread: str, text: str) -> dict:
                sent = await c.post(
                    f"/agui/{agent_id}/run",
                    json={
                        "threadId": thread,
                        "messages": [{"id": text[:12], "role": "user", "content": text}],
                    },
                )
                assert sent.status_code == 200, sent.text
                return decode(sent.text)[-1][1]

            when = {"in_minutes": 1} if how == "ticker" else {"when": "0 9 * * 1"}
            for thread, tok in ((open_thread, report), (closed_thread, token(f"q{uid[:5]}", 9))):
                made = await say(
                    thread,
                    f"{token(uid, 1)} {command('schedule_task', task=f'{tok} send my report', **when)}",
                )
                assert made.get("outcome") == {"type": "success"}, made
            schedules = await _schedules(h, user, agent_id)
            assert len(schedules) == 2, schedules
            # the open chat listens on its thread stream while the user keeps chatting
            heard: list[str] = []

            async def listen() -> None:
                async with c.stream(
                    "GET", f"/agui/{agent_id}/threads/{open_thread}/events"
                ) as stream:
                    async for line in stream.aiter_lines():
                        heard.append(line)

            listening = asyncio.create_task(listen())
            await asyncio.sleep(1)
            if how == "fire":
                for s in schedules:
                    await h.runs.schedules.fire(s.schedule_id)
            chat = await say(open_thread, f"{token(uid, 2)} meanwhile, hello")
            assert chat.get("outcome") == {"type": "success"} and parse(
                chat["result"]
            ).mine == token(uid, 2)

            async def delivered() -> bool:
                return report.upper() in "\n".join(heard)

            assert await eventually(delivered, within=200 if how == "ticker" else 90, every=1), (
                heard[-5:]
            )
            listening.cancel()

        # the closed chat: its reply is in the conversation when the user opens it
        async def kept() -> bool:
            said = await history(h, user=user, agent_id=agent_id, thread=closed_thread)
            return any(
                token(f"q{uid[:5]}", 9).upper() in content
                for role, content in said
                if role.upper() == "ASSISTANT"
            )

        assert await eventually(kept, within=120, every=2)
        fired = [r async for r in h.runs.iterate(agent_id=agent_id)]
        scheduled = [
            r
            for r in fired
            if r.thread_id in (open_thread, closed_thread) and r.status is RunStatus.SUCCESS
        ]
        assert len(scheduled) >= 4  # two chat turns making them, the chat turn, the fired runs
    finally:
        served.stop()
        for s in await _schedules(h, user, agent_id):
            await h.runs.schedules.delete(s.schedule_id)
    assert served.violations() == []


@pytest.mark.plan("SCH")
async def test_a_schedule_keeps_its_wall_clock_hour_across_daylight_saving(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    """``0 9 * * *`` in New York fires at 09:00 there on both sides of a change: 13:00 UTC in
    summer, 14:00 UTC in winter — what agent-runs says the next fire is, and what the chat tool
    says to the user."""
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s8d-{uid}")
    made = await agent.schedule(
        "0 9 * * *",
        user=f"ada-{uid}",
        input=f"{token(uid, 1)} morning report",
        tz="America/New_York",
    )
    try:
        assert made.next_fire_at is not None
        local = made.next_fire_at.astimezone(ZoneInfo("America/New_York"))
        assert (local.hour, local.minute) == (9, 0), made.next_fire_at
        offset = made.next_fire_at.astimezone(UTC).hour - local.hour
        assert offset in (4, 5), offset  # EDT or EST, whichever the next morning is in
        assert datetime.now(UTC) < made.next_fire_at
    finally:
        await h.runs.schedules.delete(made.schedule_id)
