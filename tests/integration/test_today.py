"""What day it is: memories and earlier conversations carry dates, and a model told no "today"
has nothing to measure them against. The harness says the day in the model's instructions — to
the day, in the user's zone when their app gives it, else in UTC said as not theirs — the same
line for every model call of a run (the run's start), after the instructions that never change.
Where the framework says the date itself (Claude Code, Kimi Code), the harness adds none."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agents import Agent
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from tests.integration.test_agui_frontend import PATH, served
from tests.integration.test_claude import options, started_with
from tests.integration.test_fit_window import request
from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.kimi import PlannedEndpoint, kimi_target, needs_kimi, text_of
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, current, middleware, tool
from trellis.contracts import RunStatus
from trellis.harness import middleware as ours
from trellis.harness import snapshots
from trellis.harness.automatic import JournalBackend
from trellis.harness.sandbox import DELIVERS_HERE

#: late on the 11th in UTC, early on the 12th in Kolkata
NOW = datetime(2026, 11, 11, 23, 30, tzinfo=UTC)
IN_UTC = "It is now Wed 2026-11-11 (UTC; the user's zone is not known)."
IN_KOLKATA = "It is now Thu 2026-11-12 (Asia/Kolkata, the user's zone)."


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    now = [NOW]
    monkeypatch.setattr(snapshots, "now", lambda: now[0])
    return now


def waiting(clock: list[datetime]) -> Any:
    @tool(side_effects="read")
    def wait_a_day() -> str:
        """Waits until the same time tomorrow."""
        clock[0] += timedelta(days=1)
        return "a day later"

    return wait_a_day


def instructions(body: dict[str, Any]) -> str:
    """A chat completions request's system message."""
    [system] = [m for m in body["messages"] if m["role"] == "system"]
    return text_of(system["content"])


@pytest.mark.parametrize("deep", [False, True], ids=["create_agent", "deep-agents"])
async def test_react_is_told_the_day_after_its_instructions_the_same_all_run(
    harness: Harness, clock: list[datetime], deep: bool
) -> None:
    model = ScriptedChat([("wait_a_day", {}), "done"])
    target = ReAct(
        system_prompt="You answer.", model=model, backend=JournalBackend() if deep else None
    )
    agent = arm(harness.wrap(target, id=f"today-{deep}"), [waiting(clock)])
    result = await agent.run("what day is it?", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    first, second = (instructions(body) for body in model.requests)
    # the instructions, which never change, then the day: one line, to the day
    assert first.startswith("You answer.") and first.endswith(f"\n\n{IN_UTC}")
    assert first.count("It is now") == 1
    # the run's start: the same text for every call of the run, though a day went by
    assert second == first
    # in the instructions, not in the user's turn beside the memory context
    said = [text_of(m["content"]) for m in model.requests[-1]["messages"] if m["role"] == "user"]
    assert not any("It is now" in text for text in said)
    # ``schedule_task`` keeps its hour, as it is when the tools are listed
    described = [
        t["function"]["description"]
        for body in model.requests
        for t in body["tools"]
        if t["function"]["name"] == "schedule_task"
    ]
    assert (
        "It is now Wed 2026-11-11, 23:00-23:59 (UTC; the user's zone is not known)."
        in (described[0])
    )


async def test_the_day_is_the_users_when_their_app_says_their_zone(
    harness: Harness, clock: list[datetime]
) -> None:
    """The app says the person's zone (AG-UI's ``forwardedProps.timezone``): the day is theirs,
    which late in the evening in UTC is already tomorrow in Kolkata."""
    model = ScriptedChat(["Thursday"])
    agent = harness.wrap(ReAct(system_prompt="You answer.", model=model), id="today-zone")
    body = {
        "threadId": "t1",
        "messages": [{"id": "m", "role": "user", "content": "what day is it?"}],
        "forwardedProps": {"timezone": "Asia/Kolkata"},
    }
    async with served(agent) as http:
        assert (await http.post(f"{PATH}/run", json=body)).status_code == 200
    assert instructions(model.requests[0]) == f"You answer.\n\n{IN_KOLKATA}"


@pytest.mark.parametrize("given", ["middleware", "hooks", "none"])
async def test_a_create_agent_graph_is_told_the_day_once(
    harness: Harness, clock: list[datetime], given: str
) -> None:
    """A ``create_agent`` graph reads the day after its system prompt, once: built with the
    harness's ``middleware()``, with ``ModelHooks()`` alone, or with neither (the harness runs
    its model calls through a handler of its own, ``middleware.narrowing``, which a graph given
    ``ModelHooks()`` alone has too)."""
    model = ScriptedChatModel(turns=["Wednesday"])
    stack = {"middleware": middleware(), "hooks": [ours.ModelHooks()], "none": []}[given]
    graph = create_agent(model, system_prompt="You help.", middleware=stack)
    await harness.wrap(graph, id=f"today-graph-{given}").run("what day is it?", user="u1")
    [system] = [m for m in model.seen[0] if isinstance(m, SystemMessage)]
    assert system.text == f"You help.\n\n{IN_UTC}"


async def test_instructions_given_as_blocks_keep_their_blocks(clock: list[datetime]) -> None:
    """Instructions marked for a provider's prompt cache (content blocks) stay as they are: the
    day is one more block after them, so what is cached is what never changes. Outside a run
    nothing is added."""
    blocks = [{"type": "text", "text": "You help.", "cache_control": {"type": "ephemeral"}}]
    given = request([], [])
    given = given.override(system_message=SystemMessage(content=blocks))  # type: ignore[arg-type]
    assert current() is None and ours._dated(given, None) is given
    run: Any = type("Run", (), {"started_at": NOW, "frontend": None})()
    system = ours._dated(given, run).system_message
    assert system is not None
    assert system.content == [*blocks, {"type": "text", "text": IN_UTC}]
    assert ours._dated(ours._dated(given, run), run).system_message == system  # said once
    bare = ours._dated(given.override(system_message=None), run).system_message
    assert bare is not None and bare.content == IN_UTC


async def test_an_openai_agent_is_told_the_day_after_its_instructions(
    harness: Harness, clock: list[datetime]
) -> None:
    model = ScriptedModel([("wait_a_day", {}), "done"])
    target = Agent(name="worker", instructions="You answer.", model=model)
    agent = arm(harness.wrap(target, id="today-openai"), [waiting(clock)])
    result = await agent.run("what day is it?", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert model.system == [f"You answer.\n\n{IN_UTC}"] * 2
    clock[0] = NOW
    bare = ScriptedModel(["done"])
    await harness.wrap(Agent(name="bare", model=bare), id="today-bare").run("hi", user="u1")
    assert bare.system == [IN_UTC]  # no instructions of its own: the day alone


async def test_claude_code_says_the_date_itself_so_the_harness_adds_none(
    harness: Harness, clock: list[datetime], tmp_path: Path
) -> None:
    """Claude Code tells its model the date in its own context ("Today's date is ..." with the
    first prompt, whatever the system prompt): a second one from the harness would be a
    duplicate, in another zone when the two differ."""
    target = options(tmp_path, [{"text": "Wednesday"}], system_prompt="You help.")
    agent = harness.wrap(target, id="today-claude", without={"schedules"})
    await agent.run("what day is it?", user="u1")
    started = started_with(tmp_path)
    assert started["system_prompt"] == f"You help.\n\n{DELIVERS_HERE}"
    assert started["prompt"] == "what day is it?" and not started["context"]


@needs_kimi
async def test_kimi_code_says_the_date_itself_so_the_harness_adds_none(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi Code discloses the date to its model itself (a reminder at the start of the
    conversation, as its own prompt says): the harness adds no line of its own. Pinned on Kimi
    itself, so a Kimi that stops saying the date fails here."""
    async with PlannedEndpoint().serving() as endpoint:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="today-kimi")
        result = await agent.run("what day is it?", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    said = endpoint.said()
    assert "It is now" not in said
    assert "The current date is disclosed through reminders" in said
    assert f"{datetime.now().astimezone():%Y-%m-%d}" in said  # Kimi's own, the host's day
