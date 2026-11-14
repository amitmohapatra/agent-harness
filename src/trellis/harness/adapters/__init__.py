"""Framework detection: the target's type picks its adapter.

Checked by the class's module before anything is imported, so wrapping a plain function
never imports a framework, and wrapping a LangGraph graph never imports the OpenAI SDK. An
OpenAI Agents ``Agent`` is told by its type wherever its class is defined (a subclass in your
own module): the SDK is imported already when one exists.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence
from typing import Any

from trellis.contracts import ConfigurationError
from trellis.harness.adapters.acp import ACPAdapter, ACPAgent
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
    sdk = sys.modules.get("agents")  # imported wherever an Agent, or a subclass of it, exists
    agent = getattr(sdk, "Agent", None)
    if isinstance(agent, type) and isinstance(target, agent):
        return OpenAIAgentsAdapter(target)
    if module.startswith(("agents.realtime.", "agents.voice.")):
        raise ConfigurationError(
            f"cannot wrap {type(target).__name__}: the OpenAI Agents SDK runs it over a live "
            "audio or realtime session of its own (RealtimeRunner, VoicePipeline), not "
            "Runner.run, so the harness cannot run it. Keep running it yourself and plug the "
            "blocks in (docs/blocks/openai-agents.md), or wrap the Agent its workflow runs"
        )
    if module.startswith("claude_agent_sdk."):
        from claude_agent_sdk import ClaudeAgentOptions

        if isinstance(target, ClaudeAgentOptions):
            return ClaudeAdapter()
    if isinstance(target, ACPAgent):
        return ACPAdapter()
    if inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
        getattr(target, "__call__", None)  # noqa: B004 - an object with an async __call__
    ):
        return FunctionAdapter()
    raise ConfigurationError(
        f"cannot wrap {type(target).__name__}: pass a compiled LangGraph graph (Deep Agents "
        "included, ReAct(...) among them), an OpenAI Agents Agent, ClaudeAgentOptions, an ACP "
        "agent (kimi()), or an async function (input, agent) -> answer"
    )


def convert(tool_format: ToolFormat, tools: Sequence[Tool]) -> Any:
    """``tools`` in a framework's own tool type; ``None`` when there are none."""
    if not tools or tool_format == "none":
        return None
    if tool_format == "langchain":
        from trellis.harness.tools.convert import langchain as module
    elif tool_format == "openai_agents":
        from trellis.harness.tools.convert import openai_agents as module  # type: ignore[no-redef]
    elif tool_format == "mcp":
        from trellis.harness.tools.convert import mcp as module  # type: ignore[no-redef]
    else:
        from trellis.harness.tools.convert import claude as module  # type: ignore[no-redef]
    return module.convert(tools)
