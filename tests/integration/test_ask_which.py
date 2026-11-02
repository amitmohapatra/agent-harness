"""``ReAct``'s clarify policy (``trellis.harness.clarify``): when several records match what the
question names and nothing says which is meant, the model asks which one, listing them, rather
than answering for the closest (the break-and-fix gate's "black slim jeans": 14 matches, one
picked). Said once, after the team's own instructions, in every model call of ``ReAct``; never
in ``middleware()`` for a team's own graph."""

from __future__ import annotations

from typing import Any

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, middleware, stored
from trellis.contracts import RunStatus
from trellis.harness import clarify


def system_of(request: dict[str, Any]) -> str:
    [system] = [m for m in request["messages"] if m["role"] == "system"]
    return str(system["content"])


async def test_every_model_call_of_react_says_to_ask_which_one(harness: Harness) -> None:
    model = ScriptedChat(["Which one: the Navy or the Black?"])
    result = await harness.wrap(ReAct("You answer stock questions.", model), id="which").run(
        "north stock of the slim jeans?", user="u"
    )
    assert result.status is RunStatus.SUCCESS, result.error
    said = system_of(model.requests[0])
    assert said == f"You answer stock questions.\n\n{clarify.ASK_WHICH}"


async def test_a_react_with_no_instructions_of_its_own_says_it_alone(harness: Harness) -> None:
    model = ScriptedChat(["ok"])
    await harness.wrap(ReAct("", model), id="bare").run("hi", user="u")
    assert system_of(model.requests[0]) == clarify.ASK_WHICH


def test_the_policy_is_react_s_own() -> None:
    assert all(type(m).__name__ != "AskWhich" for m in middleware())
    assert clarify.AskWhich().name == "AskWhich"
    assert stored  # a stored prompt is rendered ahead of it by ModelHooks (its own tests)
