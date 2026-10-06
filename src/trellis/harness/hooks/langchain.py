"""The hooks around a LangChain agent's model calls (``create_agent``, Deep Agents): LangChain's
own middleware, ``wrap_model_call``.

A graph's model calls are its own once it is compiled, so the middleware goes in when it is
built — as its tools do (``h.tools``):

    graph = create_agent(model, tools=await h.tools(...), middleware=[ModelHooks()])
    graph = create_deep_agent(model=model, tools=..., middleware=[ModelHooks()])

In a wrapped run it runs the run's hooks; code that runs the graph itself passes its own
(``ModelHooks(Audit())``). A call a ``before_model`` hook returns is the call made (its
messages, its system message); a failed call is ``on_error("model", ...)``.

It also offers the model only the tools of the parts the run uses: a graph's harness tools are
bound when it is built (``h.tools``), and a part turned off afterwards — ``h.wrap(without=)``,
a run's ``without=`` — would still be offered (a call of it is refused: it is off in the run).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from trellis.harness.hooks import Hooks, ModelCall, running
from trellis.harness.runtime import current
from trellis.harness.tools.convert.langchain import FEATURE


class ModelHooks(AgentMiddleware):
    """``before_model``, ``after_model`` and ``on_error`` of the run's hooks (a wrapped
    run's), then of ``hooks``, around every model call of the agent."""

    def __init__(self, *hooks: Hooks) -> None:
        super().__init__()
        self.given = list(hooks)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        request = _offered(request)
        hooks = running(*self.given)
        if not hooks:
            return await handler(request)
        name = getattr(request.model, "model_name", None)
        asked = ModelCall(
            "langgraph",
            list(request.messages),
            model=name if isinstance(name, str) else None,
            system=request.system_message,
        )
        call = await hooks.model(asked)
        if call is not asked:
            request = request.override(messages=call.messages, system_message=call.system)
        try:
            reply = await handler(request)
        except Exception as exc:
            await hooks.failed("model", exc)
            raise
        await hooks.answered(call, reply)
        return reply


def _offered(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the harness tools of the parts the run is without left out."""
    runtime = current()
    if runtime is None:
        return request
    kept = [
        t
        for t in request.tools
        if (feature := (getattr(t, "metadata", None) or {}).get(FEATURE)) is None
        or runtime.uses(feature)
    ]
    return request if len(kept) == len(request.tools) else request.override(tools=kept)
