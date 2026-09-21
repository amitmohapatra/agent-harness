"""Policy providers (§48). The core ships a permissive default and an allow/deny list;
an OPA or policy-service provider implements the same protocol later."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

from universal_agent_contracts.context import AgentExecutionContext
from universal_agent_contracts.messages import AgentRequest
from universal_agent_contracts.model import ModelRequest
from universal_agent_contracts.tool import ToolCall


class NoOpPolicyProvider:
    """Allows everything. The default: the harness does not invent an authorization model."""

    name = "noop"

    async def authorize_execution(self, request: AgentRequest) -> bool:
        return True

    async def authorize_tool(self, context: AgentExecutionContext, call: ToolCall) -> bool:
        return True

    async def authorize_model(self, context: AgentExecutionContext, request: ModelRequest) -> bool:
        return True


class AllowListPolicyProvider:
    """A simple, auditable default for deployments that want one: explicit allow lists.

    ``None`` means "no restriction on this dimension". A denial returns the reason, which
    the harness surfaces in the ``POLICY`` error rather than a bare boolean.
    """

    name = "allowlist"

    def __init__(
        self,
        *,
        agents: Iterable[str] | None = None,
        tools: Iterable[str] | None = None,
        models: Iterable[str] | None = None,
        tenants: Iterable[str] | None = None,
    ) -> None:
        self.agents = frozenset(agents) if agents is not None else None
        self.tools = frozenset(tools) if tools is not None else None
        self.models = frozenset(models) if models is not None else None
        self.tenants = frozenset(tenants) if tenants is not None else None

    async def authorize_execution(self, request: AgentRequest) -> bool | str:
        ctx = request.context
        if self.tenants is not None and ctx.tenant_id not in self.tenants:
            return f"tenant {ctx.tenant_id!r} is not allowed to run agents"
        if self.agents is not None and ctx.agent_id not in self.agents:
            return f"agent {ctx.agent_id!r} is not in the allow list"
        return True

    async def authorize_tool(self, context: AgentExecutionContext, call: ToolCall) -> bool | str:
        if self.tools is not None and call.tool not in self.tools:
            return f"tool {call.tool!r} is not in the allow list"
        return True

    async def authorize_model(
        self, context: AgentExecutionContext, request: ModelRequest
    ) -> bool | str:
        if self.models is not None and (request.model or "") not in self.models:
            return f"model {request.model!r} is not in the allow list"
        return True


class CallablePolicyProvider:
    """Adapts plain callables, for applications that already have an authorization function."""

    name = "callable"

    def __init__(
        self,
        execution: Any = None,
        tool: Any = None,
        model: Any = None,
    ) -> None:
        self._execution, self._tool, self._model = execution, tool, model

    async def authorize_execution(self, request: AgentRequest) -> bool | str:
        return await _call(self._execution, request)

    async def authorize_tool(self, context: AgentExecutionContext, call: ToolCall) -> bool | str:
        return await _call(self._tool, context, call)

    async def authorize_model(
        self, context: AgentExecutionContext, request: ModelRequest
    ) -> bool | str:
        return await _call(self._model, context, request)


async def _call(fn: Any, *args: Any) -> bool | str:
    if fn is None:
        return True
    result = fn(*args)
    return await result if inspect.isawaitable(result) else result
