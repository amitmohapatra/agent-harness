"""The OpenAI adapter's subclasses of the SDK's own classes: an agent's MCP server governed
(:class:`GovernedMCPServer`) and a model name on Bifrost that learns its route
(:class:`Learning`). They import the SDK where the module is loaded, so
``adapters.openai_agents`` — which a harness with no OpenAI agent imports too — loads this
module only when it first needs one (``_governed_server``, ``_learning_class``)."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import openai
from agents.mcp import MCPServer
from agents.models.interface import Model
from mcp.types import CallToolResult, TextContent

from trellis.contracts import ToolError, ToolOutcome, ToolSpec, is_pause_signal
from trellis.harness.adapters.openai_agents import (
    NOT_SERVED,
    WORDED,
    GovernedServer,
    _refused_on_chat,
    _worded,
)
from trellis.harness.runtime import current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool, data_of
from trellis.harness.tools.mcp import side_effects_of

if TYPE_CHECKING:
    from trellis.harness.gateways import Bifrost


class GovernedMCPServer(MCPServer, GovernedServer):  # type: ignore[misc]
    """An OpenAI agent's MCP server whose calls go through the bridge (``GovernedServer``):
    everything else is the inner server's own."""

    def __init__(self, inner: Any) -> None:  # the SDK's own state is the inner server's
        self._inner = inner
        self._annotations: dict[str, Any] = {}

    def _get_failure_error_function(self, agent_failure_error_function: Any) -> Any:
        """The server's own way of telling the model a call failed (else its agent's),
        except while the run is pausing: a pause raises out of the SDK's run rather than
        becoming an error text."""
        theirs = self._inner._get_failure_error_function(agent_failure_error_function)
        if theirs is None:
            return None

        def told(context: Any, error: Exception) -> Any:
            runtime = current()
            if runtime is not None and runtime.pending is not None:
                raise error
            text = theirs(context, error)  # the server's own way, which still runs
            worded = _worded(error)
            return text if worded is None else worded

        return told

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def name(self) -> str:
        return str(self._inner.name)

    async def connect(self) -> None:
        await self._inner.connect()

    async def cleanup(self) -> None:
        await self._inner.cleanup()

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> Any:
        tools = await self._inner.list_tools(run_context, agent)
        self._annotations.update((t.name, t.annotations) for t in tools)
        return tools

    async def list_prompts(self) -> Any:
        return await self._inner.list_prompts()

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return await self._inner.get_prompt(name, arguments)

    # what the SDK's base class answers itself (no tools cached, no resources): the inner
    # server's own
    @property
    def cached_tools(self) -> Any:
        return self._inner.cached_tools

    async def list_resources(self, cursor: str | None = None) -> Any:
        return await self._inner.list_resources(cursor)

    async def list_resource_templates(self, cursor: str | None = None) -> Any:
        return await self._inner.list_resource_templates(cursor)

    async def read_resource(self, uri: str) -> Any:
        return await self._inner.read_resource(uri)

    def _spec(self, tool_name: str) -> ToolSpec:
        """The server's tool as it listed it: its annotations for its risk and what it says."""
        annotations = self._annotations.get(tool_name)
        return ToolSpec(
            name=tool_name,
            source="mcp",
            server=self.name,
            side_effects=side_effects_of(annotations),
            **data_of(annotations, strict=False),
        )

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None, meta: Any = None
    ) -> Any:
        inner = self._inner
        if current() is None:
            return await inner.call_tool(tool_name, arguments, meta)
        raised: list[Exception] = []

        async def run(args: dict[str, Any]) -> Any:
            try:
                result = await inner.call_tool(tool_name, args, meta)
            except Exception as error:
                raised.append(error)
                raise
            if result.is_error:
                said = "".join(getattr(c, "text", "") for c in result.content)
                raise ToolError(said or f"{tool_name} failed", source="mcp")
            return result.model_dump(mode="json", by_alias=True, exclude_none=True)

        # the server's own tries, when it makes them (``max_retry_attempts``), are its only
        retried = bool(getattr(inner, "max_retry_attempts", 0))
        tool = Tool(self._spec(tool_name), run, retried=retried)
        return _mcp_result(await bridge.call(tool, dict(arguments or {})), raised)


def _mcp_result(outcome: ToolOutcome, raised: list[Exception]) -> CallToolResult:
    """What the SDK is handed of a governed MCP call: the server's result; a failure of the
    call itself raised again — the SDK's to tell, or to raise, in the words of the run's
    ``after_tool`` hooks when they worded it (``_worded``); else what the bridge says, as the
    tool's result."""
    if outcome.ok and isinstance(outcome.output, dict):
        return CallToolResult.model_validate(outcome.output)
    text = outcome.output if isinstance(outcome.output, str) else json.dumps(outcome.output)
    hooked = bridge.hooked(outcome)
    if raised and not is_pause_signal(raised[-1]) and not (hooked and outcome.ok):
        if hooked:
            setattr(raised[-1], WORDED, text)
        raise raised[-1]
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=not outcome.ok)


class Learning(Model):  # type: ignore[misc]
    """A model name on Bifrost, asked on the Responses API until the gateway answers that it
    does not serve the route for it (``NOT_SERVED``, on the model's first call there: nothing
    was asked, no tool has run for it) — then, and from then on for this gateway and model
    (``Bifrost.served``), on chat completions, as before the Responses route. What a chat
    completions request cannot carry is refused first, each feature named
    (``_refused_on_chat``). One fallback per call at most; a model that has answered on
    Responses never falls back (its refusal is the run's error)."""

    def __init__(self, gateway: Bifrost, name: str, responses: Any, chat: Any) -> None:
        self.gateway, self.name, self.responses, self.chat = gateway, name, responses, chat

    def _falls_back(self, error: Exception, args: Any, kwargs: Any) -> bool:
        unserved = isinstance(error, openai.APIStatusError) and error.status_code in NOT_SERVED
        if not unserved or not self.gateway.falls_back(self.name):
            return False
        asked = inspect.signature(Model.get_response).bind(self, *args, **kwargs)
        _refused_on_chat(self.name, asked.arguments)
        return True

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        if not self.gateway.keeps_responses(self.name):
            return await self.chat.get_response(*args, **kwargs)
        try:
            response = await self.responses.get_response(*args, **kwargs)
        except Exception as error:
            if not self._falls_back(error, args, kwargs):
                raise
        else:
            self.gateway.served(self.name, responses=True)
            return response
        response = await self.chat.get_response(*args, **kwargs)
        self.gateway.served(self.name, responses=False)  # it answered there: kept
        return response

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if self.gateway.keeps_responses(self.name):
            try:
                async for event in self.responses.stream_response(*args, **kwargs):
                    self.gateway.served(self.name, responses=True)
                    yield event
                return
            except Exception as error:  # refused before its first event: nothing was said
                if not self._falls_back(error, args, kwargs):
                    raise
        async for event in self.chat.stream_response(*args, **kwargs):
            self.gateway.served(self.name, responses=False)
            yield event
