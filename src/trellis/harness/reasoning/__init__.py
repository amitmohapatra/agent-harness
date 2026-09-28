"""Reasoning loops an agent can use instead of writing its own.

These are plain async functions over :class:`AgentRuntime`, not a framework. That is the
point: the harness already gives ``runtime.model`` and ``runtime.tools`` tracing, metering,
policy and deadlines, so a loop written against them is instrumented wherever it runs —
inside a LangGraph node, a Celery task or a bare ``asyncio.run``.
"""

from trellis.harness.reasoning.assembler import ContextAssembler
from trellis.harness.reasoning.react import ReActStep, ReActTrace, react

__all__ = ["ContextAssembler", "ReActStep", "ReActTrace", "react"]
