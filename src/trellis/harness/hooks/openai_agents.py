"""The hooks around an OpenAI Agents SDK run's model calls: the SDK's own ``RunHooks``.

A wrapped agent's runs get it from the harness (``Runner.run(..., hooks=ModelHooks())``); code
that runs the SDK itself passes its hooks: ``Runner.run(agent, input,
hooks=ModelHooks(Audit()))``. The SDK reports its model calls (``on_llm_start``,
``on_llm_end``) and takes nothing back: a call a ``before_model`` hook returns is not sent
instead, and a failed call is not reported (``on_error`` sees the run's failure). In a run,
every reply's usage counts toward the run's budget (``budget.py``), and an agent handed off
to is a native helper of the run (``bridges.helpers``: its run record, its turns).
"""

from __future__ import annotations

from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse

from trellis.harness import budget
from trellis.harness.bridges.helpers import Helpers, Kind
from trellis.harness.hooks import Hooks, ModelCall, running
from trellis.harness.runtime import current


class ModelHooks(RunHooks[Any]):
    """``before_model`` and ``after_model`` of the run's hooks (a wrapped run's), then of
    ``hooks``, around every model call of the SDK's run; ``chained`` — the team's own
    ``RunHooks`` — is told every lifecycle event after them, as the SDK would tell it. With
    ``helpers`` (a wrapped run's), each agent handed off to — other than ``starting``, the
    agent the run began with — is a helper of the run until it hands off or answers."""

    def __init__(
        self,
        *hooks: Hooks,
        chained: RunHooks[Any] | None = None,
        helpers: Helpers | None = None,
        starting: str | None = None,
    ) -> None:
        self.given = list(hooks)
        self.chained = chained if chained is not None else RunHooks()
        self.helpers = helpers
        self.starting = starting
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
        await self.chained.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], response: ModelResponse
    ) -> None:
        call = self._calls.pop(agent.name)
        runtime = current()
        if runtime is not None:  # what the run spent (``budget.py``)
            usage = response.usage
            details = getattr(usage, "input_tokens_details", None)
            budget.account(
                runtime,
                model=call.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached=int(getattr(details, "cached_tokens", 0) or 0),
            )
        await running(*self.given).answered(call, response)
        await self.chained.on_llm_end(context, agent, response)

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        await self.chained.on_agent_start(context, agent)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        if self.helpers is not None:
            await self.helpers.end(agent.name, output=output if isinstance(output, str) else None)
        await self.chained.on_agent_end(context, agent, output)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        if self.helpers is not None:
            await self.helpers.end(from_agent.name)  # it handed its work on
            if to_agent.name != self.starting:
                task = self.helpers.runtime.task
                await self.helpers.start(
                    to_agent.name, kind=Kind.HANDOFF, name=to_agent.name, task=task
                )
        await self.chained.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        await self.chained.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        await self.chained.on_tool_end(context, agent, tool, result)
