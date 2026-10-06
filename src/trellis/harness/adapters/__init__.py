"""Framework detection: the target's type picks its adapter.

Checked by the class's module before anything is imported, so wrapping a plain function
never imports a framework, and wrapping a LangGraph graph never imports the OpenAI SDK.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from trellis.contracts import ConfigurationError
from trellis.harness.adapters.base import Adapter, ToolFormat
from trellis.harness.adapters.claude import ClaudeAdapter
from trellis.harness.adapters.function import FunctionAdapter
from trellis.harness.adapters.langgraph import LangGraphAdapter
from trellis.harness.adapters.openai_agents import OpenAIAgentsAdapter
from trellis.harness.tools.base import Tool


def detect(target: Any) -> Adapter:
    module = type(target).__module__
    if module.startswith("langgraph."):
        from langgraph.pregel import Pregel

        if isinstance(target, Pregel):
            return LangGraphAdapter(target)
    if module.startswith("agents."):
        from agents import Agent

        if isinstance(target, Agent):
            return OpenAIAgentsAdapter()
    if module.startswith("claude_agent_sdk."):
        from claude_agent_sdk import ClaudeAgentOptions

        if isinstance(target, ClaudeAgentOptions):
            return ClaudeAdapter()
    if inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
        getattr(target, "__call__", None)  # noqa: B004 - an object with an async __call__
    ):
        return FunctionAdapter()
    raise ConfigurationError(
        f"cannot wrap {type(target).__name__}: pass a compiled LangGraph graph (Deep Agents "
        "included, ReAct(...) among them), an OpenAI Agents Agent, ClaudeAgentOptions, or an async "
        "function (input, agent) -> answer"
    )


def convert(tool_format: ToolFormat, tools: Sequence[Tool]) -> Any:
    """``tools`` in a framework's own tool type; ``None`` when there are none."""
    if not tools or tool_format == "none":
        return None
    if tool_format == "langchain":
        from trellis.harness.tools.convert import langchain as module
    elif tool_format == "openai_agents":
        from trellis.harness.tools.convert import openai_agents as module  # type: ignore[no-redef]
    else:
        from trellis.harness.tools.convert import claude as module  # type: ignore[no-redef]
    return module.convert(tools)
