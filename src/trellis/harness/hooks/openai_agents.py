"""The hooks around an OpenAI Agents SDK run's model calls: the SDK's own ``RunHooks``.

A wrapped agent's runs get it from the harness (``Runner.run(..., hooks=ModelHooks())``); code
that runs the SDK itself passes its hooks: ``Runner.run(agent, input,
hooks=ModelHooks(Audit()))``. The SDK reports its model calls (``on_llm_start``,
``on_llm_end``) and takes nothing back: a call a ``before_model`` hook returns is not sent
instead, and a failed call is not reported (``on_error`` sees the run's failure). In a run,
every reply's usage counts toward the run's budget (``budget.py``), every call is watched for a
stall from its start to its reply (``Runtime.model_started``: the provider's hosted tools run
inside it), and the hosted tool calls a reply carries — file search, web search, code
interpreter, image generation, a hosted MCP server's tools — are recorded as the run's tool
calls, with the step's duration (``tools.bridge.observed``).
"""

from __future__ import annotations

import time
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse

from trellis.harness import budget
from trellis.harness.hooks import Hooks, ModelCall, running
from trellis.harness.runtime import current


class ModelHooks(RunHooks[Any]):
    """``before_model`` and ``after_model`` of the run's hooks (a wrapped run's), then of
    ``hooks``, around every model call of the SDK's run."""

    def __init__(self, *hooks: Hooks) -> None:
        self.given = list(hooks)
        #: the call each agent of the run is making, by its name, and when it began
        self._calls: dict[str, tuple[ModelCall, float]] = {}

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        model = agent.model if isinstance(agent.model, str) else None
        call = ModelCall("openai_agents", list(input_items), model=model, system=system_prompt)
        self._calls[agent.name] = (call, time.time())
        runtime = current()
        if runtime is not None:
            runtime.model_started(_step(agent))
        await running(*self.given).model(call)

    async def on_llm_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], response: ModelResponse
    ) -> None:
        call, started = self._calls.pop(agent.name)
        runtime = current()
        if runtime is not None:  # what the run spent (``budget.py``)
            runtime.model_ended(_step(agent))
            from trellis.harness.adapters.openai_agents import hosted_calls  # noqa: PLC0415

            await hosted_calls(runtime, response.output, started=started)
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


def _step(agent: Agent[Any]) -> str:
    """The key a model call of ``agent`` is watched under (one at a time per agent)."""
    return f"openai_agents:{agent.name}"
