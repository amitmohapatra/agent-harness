"""A live Claude session's rules where no process is needed to show them
(``bridges.claude_live``): what a request gets when no turn can take it, and what a turn is
charged of the process's running totals."""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace
from typing import Any

from claude_agent_sdk import HookMatcher, PermissionResultDeny
from claude_agent_sdk._internal.message_parser import parse_message

from trellis.harness.bounded import LRU
from trellis.harness.bridges.claude_live import (
    LEFT,
    UNATTENDED,
    LiveSession,
    LiveSessions,
    Turn,
)

REFUSED = {"content": [{"type": "text", "text": UNATTENDED}], "is_error": True}


def refusal(answer: Any) -> tuple[str, str]:
    said = answer["hookSpecificOutput"]
    return said["permissionDecision"], said["permissionDecisionReason"]


async def test_with_no_turn_to_take_it_a_tool_call_is_refused_and_counted() -> None:
    """Between two turns nothing can check a call: whichever way the CLI asks — its
    ``PreToolUse`` hook, its permission check, a harness tool's call — the answer is no."""

    async def never(given: Any, tool_use_id: str | None, context: Any) -> Any:
        raise AssertionError("a hook of no turn's")

    session = LiveSession("conversation", (), LiveSessions())
    hooks = session._hooks(
        {"PreToolUse": [HookMatcher(hooks=[never])], "Stop": [HookMatcher(hooks=[never])]}
    )
    asked = await hooks["PreToolUse"][0].hooks[0]({"tool_name": "Bash"}, "toolu_1", None)
    assert refusal(asked) == ("deny", UNATTENDED)
    assert await hooks["Stop"][0].hooks[0]({}, None, None) == {}  # not a tool call: it passes
    denied = await session._can_use_tool("Bash", {"command": "rm -rf build"}, None)
    assert isinstance(denied, PermissionResultDeny) and denied.message == UNATTENDED
    assert await session._call("refund", {"order": "o1"}) == REFUSED
    assert session.refused == {"Bash": 2, "refund": 1}


async def test_the_main_agents_call_in_a_turn_that_was_left_ends_that_turn() -> None:
    """A left turn's result is still to come: the main agent's next call ends the turn — the
    hook's ``continue: false``, the permission check's ``interrupt``, which leave background
    work as it is. A subagent's call (the CLI names it) is refused and no more."""

    async def never(given: Any, tool_use_id: str | None, context: Any) -> Any:
        raise AssertionError("a hook of no turn's")

    session = LiveSession("conversation", (), LiveSessions())
    session.owed = 1
    relay = session._hooks({"PreToolUse": [HookMatcher(hooks=[never])]})["PreToolUse"][0].hooks[0]
    ended = await relay({"tool_name": "Bash"}, "toolu_1", None)
    assert refusal(ended) == ("deny", LEFT)
    assert ended["continue_"] is False and ended["stopReason"] == LEFT
    theirs = await relay({"tool_name": "Bash", "agent_id": "a1"}, "toolu_2", None)
    assert refusal(theirs) == ("deny", UNATTENDED) and "continue_" not in theirs
    denied = await session._can_use_tool("Bash", {}, SimpleNamespace(agent_id=None))
    assert (denied.message, denied.interrupt) == (LEFT, True)
    denied = await session._can_use_tool("Bash", {}, SimpleNamespace(agent_id="a1"))
    assert (denied.message, denied.interrupt) == (UNATTENDED, False)


async def test_a_request_under_way_when_its_turn_is_left_is_refused() -> None:
    """A turn that is left (cancelled, paused) takes its run with it: what the CLI had asked
    it and was still waiting for is cancelled, and answered as with no turn."""
    never = asyncio.Event()

    async def hook(given: Any, tool_use_id: str | None, context: Any) -> Any:
        await never.wait()

    async def can_use_tool(name: str, args: dict[str, Any], context: Any) -> Any:
        await never.wait()

    options = SimpleNamespace(
        hooks={"PreToolUse": [HookMatcher(hooks=[hook])]},
        can_use_tool=can_use_tool,
        session_store=None,
    )
    runtime: Any = SimpleNamespace(agent=SimpleNamespace(sessions=LRU()))
    session = LiveSession("conversation", (), LiveSessions())
    turn = Turn(runtime, options, None, contextvars.copy_context())
    session._attach(turn)
    relay = session._hooks(options.hooks)["PreToolUse"][0].hooks[0]
    asked = asyncio.gather(
        relay({"tool_name": "Bash"}, "toolu_1", None),
        session._can_use_tool("Bash", {"command": "make"}, None),
    )
    await asyncio.sleep(0.01)
    assert len(turn.calls) == 2
    session._detach(turn)
    hooked, denied = await asked
    assert refusal(hooked) == ("deny", UNATTENDED)
    assert isinstance(denied, PermissionResultDeny) and denied.message == UNATTENDED
    await session.close()
    await session.close()  # closed once: a second time is no change
    assert session.closed and session._timer is None


def result(cost: float | None, **usage: Any) -> Any:
    """A result as the bundled CLI sends one (its usage holds more than token counts)."""
    return parse_message(
        {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 1,
            "session_id": "s1",
            "result": "ok",
            "total_cost_usd": cost,
            "usage": {**usage, "server_tool_use": {"web_search_requests": 0}},
        }
    )


def test_a_turn_is_charged_what_the_processs_running_cost_grew_by() -> None:
    """As the bundled CLI reports: a result's cost is its process's running total, its usage
    the turn's own."""
    session = LiveSession("conversation", (), LiveSessions())
    first = session.charged(result(0.0012, input_tokens=200, output_tokens=40))
    assert (first.total_cost_usd, first.usage["input_tokens"]) == (0.0012, 200)
    second = session.charged(result(0.0016, input_tokens=100, output_tokens=20))
    assert round(second.total_cost_usd, 6) == 0.0004
    assert (second.usage["input_tokens"], second.usage["output_tokens"]) == (100, 20)
    # a total that did not grow is no running total: taken as the turn's own
    third = session.charged(result(0.001, input_tokens=40, output_tokens=5))
    assert third.total_cost_usd == 0.001
    assert session.charged(result(None, input_tokens=50, output_tokens=6)).total_cost_usd is None
