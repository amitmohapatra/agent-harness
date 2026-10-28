"""What the harness watches of an OpenAI Agents SDK run's model calls, through the SDK's own
``RunHooks``: a wrapped agent's runs get it from the harness (``Runner.run(...,
hooks=ModelHooks())``). It runs no hooks of yours: a model call is the SDK's own, observed with your
agent's own ``AgentHooks`` (``Agent(hooks=...)``: ``on_llm_start``, ``on_llm_end``). In a run, every
reply's usage counts toward the run's budget (``budget.py``), every call is watched for a stall from
its start to its reply (``Runtime.model_started``: the provider's hosted tools run inside it), and
the hosted tool calls a reply carries — file search, web search, code interpreter, image generation,
a hosted MCP server's tools — are recorded as the run's tool calls, with the step's duration
(``tools.bridge.observed``), and its function calls repeating an earlier one of it (the same tool,
the same arguments) are kept for the harness's tools to answer without running them
(``Runtime.repeats``: ``tools.convert.openai_agents``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse
from openai.types.responses import ResponseFunctionToolCall

from trellis.harness import budget
from trellis.harness.runtime import current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import arguments_of


class ModelHooks(RunHooks[Any]):
    """Around every model call of a harness run of the SDK: the stall watch, the hosted tool
    calls recorded, the usage counted toward the run's budget."""

    def __init__(self) -> None:
        #: the model each agent of the run is calling, by its name, and when the call began
        self._calls: dict[str, tuple[str | None, float]] = {}

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        model = agent.model if isinstance(agent.model, str) else None
        self._calls[agent.name] = (model, time.time())
        runtime = current()
        if runtime is not None:
            runtime.model_started(_step(agent))

    async def on_llm_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], response: ModelResponse
    ) -> None:
        model, started = self._calls.pop(agent.name)
        runtime = current()
        if runtime is not None:  # what the run spent (``budget.py``)
            runtime.model_ended(_step(agent))
            runtime.repeats = _repeats(response.output)
            from trellis.harness.adapters.openai_agents import hosted_calls  # noqa: PLC0415

            await hosted_calls(runtime, response.output, started=started)
            usage = response.usage
            details = getattr(usage, "input_tokens_details", None)
            budget.account(
                runtime,
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached=int(getattr(details, "cached_tokens", 0) or 0),
            )


def _step(agent: Agent[Any]) -> str:
    """The key a model call of ``agent`` is watched under (one at a time per agent)."""
    return f"openai_agents:{agent.name}"


def _repeats(items: Sequence[Any]) -> dict[tuple[str, str, str], str]:
    """A reply's function calls repeating an earlier one of it (``tools.bridge.firsts``), by
    what the SDK hands each call's tool: its id, its tool, its arguments as the model wrote
    them (``Runtime.repeats``)."""
    calls = [i for i in items if isinstance(i, ResponseFunctionToolCall)]
    firsts = bridge.firsts((c.call_id, c.name, _parsed(c.arguments)) for c in calls)
    return {
        (c.call_id, c.name, c.arguments): firsts[c.call_id] for c in calls if c.call_id in firsts
    }


def _parsed(text: str) -> Any:
    """A call's arguments as an object (the same in any key order), or as written when they are
    not one."""
    args = arguments_of(text)
    return text if isinstance(args, str) else args
