"""The frameworks the harness wraps, and what it knows of each outside its adapter — kept here,
and only here.

Each framework is one :class:`Registration`: its adapter's name, the tool type
``h.tools(framework=...)`` builds for it (and the other names that call takes for it), the
distributions it runs on (``compat.py`` checks their versions) and whether a target of it has
tools of its own that the harness does not run (a sub-agent's risk, ``subagents.py``). A new
framework is one more entry in :data:`REGISTERED`, beside its adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from trellis.harness.adapters.base import ToolFormat


def _none(target: Any) -> bool:
    """A target with no tools of its own (a function: its tools are the harness's)."""
    return False


def _all(target: Any) -> bool:
    """A target whose framework always has tools of its own (Claude's built-in tools)."""
    return True


def _agent_tools(target: Any) -> bool:
    """An OpenAI Agents agent's own tools or handoffs."""
    return bool(target.tools or target.handoffs)


def _graph_tools(target: Any) -> bool:
    """A graph's tools not built with ``h.tools``."""
    from trellis.harness.adapters.langgraph import bound_tools  # noqa: PLC0415 - lazy
    from trellis.harness.harness import TOOLBOX  # noqa: PLC0415 - harness imports this

    return any(TOOLBOX not in (getattr(t, "metadata", None) or {}) for t in bound_tools(target))


@dataclass(frozen=True, slots=True)
class Registration:
    """What the harness knows of one framework outside its adapter."""

    #: the adapter's ``name`` (the source of a run's failure)
    name: str
    #: the tool type ``h.tools(framework=name)`` builds (``None``: ``h.tools`` builds none)
    tool_format: ToolFormat | None
    #: the distributions it runs on, whose versions ``compat.check`` checks
    distributions: tuple[str, ...] = ()
    #: the other names ``h.tools(framework=)`` takes for it
    also: tuple[str, ...] = ()
    #: whether a target has tools of its own, which the harness does not run
    own_tools: Callable[[Any], bool] = _none


REGISTERED: Final[tuple[Registration, ...]] = (
    Registration(
        "langgraph",
        "langchain",
        ("langgraph", "langchain-core", "langchain", "deepagents", "langchain-openai"),
        # Deep Agents builds a LangGraph graph: LangChain tools too
        also=("deepagents",),
        own_tools=_graph_tools,
    ),
    Registration("openai_agents", "openai_agents", ("openai-agents",), own_tools=_agent_tools),
    Registration("claude_agent_sdk", "claude", ("claude-agent-sdk",), own_tools=_all),
    Registration("function", None),
)
#: Each registration by its adapter's name.
BY_NAME: Final[Mapping[str, Registration]] = {r.name: r for r in REGISTERED}


def tool_formats() -> dict[str, ToolFormat]:
    """The tool type ``h.tools`` builds, by each name it takes for a framework."""
    return {
        name: r.tool_format
        for r in REGISTERED
        if r.tool_format is not None
        for name in (r.name, *r.also)
    }


__all__ = ["BY_NAME", "REGISTERED", "Registration", "tool_formats"]
