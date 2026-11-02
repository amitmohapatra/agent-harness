"""``ReAct``'s clarify policy: when several records match what a question names and nothing
says which is meant, the model asks which one — listing them — instead of answering for the
closest match.

A small model given a search with fourteen matches answered for the one whose name was closest
and listed a few others (the break-and-fix gate's "black slim jeans", ReAct and OpenAI Agents
alike), though the team's instructions said to ask when a question is ambiguous: the rule said
*that* it should ask, not *when* a question is. :data:`ASK_WHICH` says when, once, after the
team's own instructions (a fixed text: a provider's prompt cache keeps it). It is not keyed on
how a question is worded: what the tools found decides.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Final

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

ASK_WHICH: Final = (
    "When what the question names matches several records (a search returns more than one, "
    "or more than it shows) and nothing in the question or the conversation says which is "
    "meant, do not answer for the closest match: ask which one, listing the matches."
)


class AskWhich(AgentMiddleware):
    """:data:`ASK_WHICH` after the instructions of every model call (``ReAct``'s own: not in
    ``middleware()``)."""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        own = request.system_message.text if request.system_message is not None else ""
        said = f"{own}\n\n{ASK_WHICH}" if own else ASK_WHICH
        return await handler(request.override(system_message=SystemMessage(content=said)))
