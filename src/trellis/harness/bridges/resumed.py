"""What a session resumed after a pause is told, whichever agent keeps the session (Claude
Code, an ACP agent, the OpenAI Agents SDK): the run goes on, and the call it paused on is
made again — or is not, when a person denied it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from trellis.contracts import ToolCall

#: What a resumed session is told: the run goes on, and the call it paused on is made again —
#: or the call its process stopped in, which the session may hold with the harness's
#: ``tools.convert.mcp.STOPPED`` as its result (the worker died while it ran, and the session
#: was resumed on the same machine): a model told only "no result yet" read that error as the
#: call's result and never made it.
RESUMED: Final = (
    "This task was paused (a person was asked, or the process stopped) and goes on now. If "
    "your last tool call has no result yet, or its result says the run ended before it "
    "finished, call that tool again with the same arguments: it runs once. Then carry on "
    "with the task."
)
#: What a session resumed after a person answered about one tool call — its approval, or a
#: question it asked while it ran — is told: that call, by the name Claude knows it by. The
#: session holds the call's result as "waiting for a person's approval", which :data:`RESUMED`'s
#: "no result yet" does not describe: a model told only that took the waiting call for a
#: failure and tried other tools instead (or asked the person again itself).
RESUMED_CALL: Final = (
    "This task was paused at your call to {tool}: it was waiting for a person's approval, and "
    "the person has answered. Call {tool} again now with the same arguments: it runs once and "
    "returns its result. Then carry on with the task."
)
#: ... and after a person answered a question that call asked while it ran
#: (``Pending.inside``): named with its arguments, which a session whose transcript ends
#: before the call does not hold (told to repeat a call it had no record of, Claude made none).
#: Whatever arguments it is made again with, it runs as it was (``Replay.made``).
RESUMED_ASKED: Final = (
    "This task was paused at your call to {tool} with the arguments {args}: it was waiting for "
    "a person's answer, and the person has answered. Call {tool} again now with those "
    "arguments: it runs once and returns its result. Then carry on with the task."
)
#: ... and after a person denied that call: it does not run, so it returns no result to wait
#: for, and the session is told so instead of being told to expect one.
RESUMED_DENIED: Final = (
    "This task was paused at your call to {tool}: it was waiting for a person's approval, and "
    "the person denied it. That call will not run; do not make it again. Carry on with the "
    "task without it."
)


def resumed_call(call: ToolCall, name: str, *, denied: bool, asked: bool) -> str:
    """What a session resumed after a person answered about ``call`` is told (``name``: the
    tool as its agent calls it): the call was denied, or asked them a question, or was
    approved."""
    told = RESUMED_DENIED if denied else RESUMED_ASKED if asked else RESUMED_CALL
    return told.format(tool=name, args=json.dumps(call.args, default=str))
