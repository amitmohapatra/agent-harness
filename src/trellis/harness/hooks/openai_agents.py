"""The hooks around an OpenAI Agents SDK run's model calls: the SDK's own ``RunHooks``.

A wrapped agent's runs get it from the harness (``Runner.run(..., hooks=ModelHooks())``); code
that runs the SDK itself passes its hooks: ``Runner.run(agent, input,
hooks=ModelHooks(Audit()))``. The SDK reports its model calls (``on_llm_start``,
``on_llm_end``) and takes nothing back: a call a ``before_model`` hook returns is not sent
instead, and a failed call is not reported (``on_error`` sees the run's failure).
"""

from __future__ import annotations

from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse

from trellis.harness.hooks import Hooks, ModelCall, running


class ModelHooks(RunHooks[Any]):
    """``before_model`` and ``after_model`` of the run's hooks (a wrapped run's), then of
    ``hooks``, around every model call of the SDK's run."""

    def __init__(self, *hooks: Hooks) -> None:
        self.given = list(hooks)
        #: the call each agent of the run is making, by its name
        self._calls: dict[str, ModelCall] = {}

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        model = agent.model if isinstance(agent.model, str) else None
        call = ModelCall("openai_agents", list(input_items), model=model, system=system_prompt)
        self._calls[agent.name] = call
        await running(*self.given).model(call)

    async def on_llm_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], response: ModelResponse
    ) -> None:
        await running(*self.given).answered(self._calls.pop(agent.name), response)
