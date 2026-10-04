"""A tool as the harness holds it: what it is (a contracts ``ToolSpec``), how to run it, and
when a person must approve a call.

Every source — a local function, an MCP tool the Bifrost virtual key allows, an A2A agent, an
OpenAPI operation, the memory service's agent tools — resolves to :class:`Tool`\\ s. The native
converters (``tools.convert``) wrap a ``Tool`` in the framework's own tool type, and every
call goes through the bridge (policy, approval, journal, recording) before ``run``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from trellis.contracts import ToolSpec

SideEffects = Literal["read", "write", "irreversible"]

#: What an unknown tool is assumed to do: something, but nothing a person must approve.
DEFAULT_SIDE_EFFECTS: SideEffects = "write"

Runner = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    run: Runner = field(repr=False)
    #: Bifrost Code Mode meta-tool: its nested calls are recorded from the gateway's log.
    code_mode: bool = False
    #: the catalog's ``approve_when`` expression: a call asks exactly when it holds
    approve_when: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def side_effects(self) -> str:
        return self.spec.side_effects


class Source(Protocol):
    """Something ``tools=[...]`` accepts. Resolved again each time the toolbox lists its
    definitions (``toolbox.TOOLS_TTL_SECONDS``)."""

    async def resolve(self) -> list[Tool]: ...
