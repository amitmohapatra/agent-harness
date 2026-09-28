"""``BifrostModel``: the OpenAI Agents SDK's ``Model`` interface over the harness's gateway.

Design §8 said "model provider pointed at Bifrost". Done literally — an ``AsyncOpenAI``
client with ``base_url`` set — that constructs a provider SDK client inside trellis code and
routes every model call around ``runtime.model``, so the span, the token and cost metrics,
the policy check and the execution's deadline stop applying. The gateway would still be the
only thing holding a key, but nothing else the harness promises would survive.

So the adapter implements the SDK's own seam instead. ``BifrostModel`` converts the SDK's
input items into a :class:`ModelRequest`, hands it to the harness model port, and converts
the :class:`ModelResponse` back into the SDK's output items and ``Usage``.

The item classes come from ``agents.items``, which is where the SDK keeps its item
vocabulary — so this adapter builds exactly the objects the runner expects without importing
a provider SDK of its own (the SDK's bundled LiteLLM provider reaches for
``openai.types.responses`` directly; that is the import this package does not make).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Final

from agents.items import (
    ModelResponse as SDKModelResponse,
)

# ``agents.items`` is where the SDK keeps its item vocabulary — ``MessageOutputItem.raw_item``
# *is* a ``ResponseOutputMessage`` — but it re-exports these three without declaring them, so a
# type checker calls the import private. The alternative is importing them from
# ``openai.types.responses`` (which is what the SDK's own LiteLLM provider does), and that would
# put a provider SDK into trellis code: the one thing ``tests/unit/test_architecture.py``
# forbids. The objects are the same either way, so the import path stays here and the
# diagnostic is suppressed deliberately rather than the rule being bent.
from agents.items import (
    ResponseFunctionToolCall,  # pyright: ignore[reportPrivateImportUsage]
    ResponseOutputMessage,  # pyright: ignore[reportPrivateImportUsage]
    ResponseOutputText,  # pyright: ignore[reportPrivateImportUsage]
    TResponseInputItem,
    TResponseStreamEvent,
)
from agents.models.interface import Model, ModelProvider, ModelTracing
from agents.usage import Usage
from trellis.contracts.model import ModelRequest, ModelResponse

from trellis.harness.runtime.propagation import require_runtime

__all__ = ["BifrostModel", "BifrostModelProvider", "to_model_request"]

#: Where a run's model calls resolve from when no client was bound at construction.
HINT: Final = "run the agent through harness.openai_agents.wrap(...) / .agent(...)"
#: The id the SDK gives an item a provider did not name.
SYNTHETIC_ID: Final = "trellis-message"
#: ``ModelSettings`` fields the gateway's chat API understands, in its own spelling.
_SETTINGS: Final = {
    "temperature": "temperature",
    "top_p": "top_p",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "max_tokens": "max_tokens",
    "tool_choice": "tool_choice",
    "parallel_tool_calls": "parallel_tool_calls",
    "top_logprobs": "top_logprobs",
}


def to_model_request(
    system_instructions: str | None,
    input: str | list[TResponseInputItem],
    *,
    model: str | None = None,
    tools: Sequence[Any] = (),
    settings: Any = None,
) -> ModelRequest:
    """The SDK's call, as the harness's model contract sees it."""
    messages: list[dict[str, Any]] = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    if isinstance(input, str):
        messages.append({"role": "user", "content": input})
    else:
        messages.extend(_wire(item) for item in input)
    return ModelRequest(
        model=model,
        messages=[m for m in messages if m],
        tools=_tool_schemas(tools),
        params=_params(settings),
    )


def _wire(item: Any) -> dict[str, Any]:
    """One SDK input item in the gateway's chat shape.

    The SDK's items are Responses-API shaped: a message carries a list of content parts, a
    tool call and its output are separate items. The gateway speaks chat completions, where
    a tool call rides on the assistant turn and its result is a ``tool`` turn.
    """
    raw = item if isinstance(item, dict) else item.model_dump()
    kind = raw.get("type", "message")
    if kind == "function_call":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": raw.get("call_id") or raw.get("id"),
                    "type": "function",
                    "function": {
                        "name": raw.get("name"),
                        "arguments": raw.get("arguments") or "{}",
                    },
                }
            ],
        }
    if kind == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": raw.get("call_id"),
            "content": _as_text(raw.get("output")),
        }
    return {"role": raw.get("role", "user"), "content": _as_text(raw.get("content"))}


def _as_text(content: Any) -> str:
    """Responses content (a string, or a list of parts) as the chat API's plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            block = part if isinstance(part, dict) else getattr(part, "model_dump", dict)()
            parts.append(str(block.get("text") or block.get("output") or ""))
        return "".join(parts)
    return str(content)


def _tool_schemas(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """The agent's function tools in the schema the gateway expects.

    Only function tools are projected: a hosted tool (web search, the code interpreter) is
    executed by a provider the gateway does not proxy, so offering it would promise the model
    something nothing can run.
    """
    schemas = []
    for tool in tools:
        schema = getattr(tool, "params_json_schema", None)
        if schema is None:
            continue
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": schema,
                },
            }
        )
    return schemas


def _params(settings: Any) -> dict[str, Any]:
    if settings is None:
        return {}
    params: dict[str, Any] = {}
    for field, wire_name in _SETTINGS.items():
        value = getattr(settings, field, None)
        if value is not None:
            params[wire_name] = value
    return params


def _output_items(response: ModelResponse) -> list[Any]:
    """The harness's model response as the SDK's output items."""
    items: list[Any] = []
    if response.text:
        items.append(
            ResponseOutputMessage(
                id=SYNTHETIC_ID,
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        text=response.text, type="output_text", annotations=[], logprobs=[]
                    )
                ],
            )
        )
    for index, call in enumerate(response.tool_calls):
        function = call.get("function") or {}
        call_id = call.get("id") or f"call_{index}"
        arguments = function.get("arguments")
        items.append(
            ResponseFunctionToolCall(
                id=call_id,
                call_id=call_id,
                type="function_call",
                name=function.get("name") or "",
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
            )
        )
    return items


def _usage(response: ModelResponse) -> Usage:
    usage = response.usage
    if usage is None:
        return Usage(requests=1)
    return Usage(
        requests=1,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


class BifrostModel(Model):
    """The SDK's ``Model``, answered by the harness's instrumented model client.

        agent = Agent(name="triage", model=harness.openai_agents.model())

    With no ``client`` the running execution's ``runtime.model`` answers, so one ``Agent``
    built at import time serves every run and every call is still traced, metered,
    policy-checked and deadline-bounded.
    """

    def __init__(self, *, client: Any = None, model: str | None = None) -> None:
        self._client = client
        self.model_name = model

    def _resolve(self) -> Any:
        return self._client if self._client is not None else require_runtime(hint=HINT).model

    async def get_response(  # noqa: PLR0917 - the SDK's Model interface is positional
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
        **_: Any,
    ) -> SDKModelResponse:
        request = to_model_request(
            system_instructions,
            input,
            model=self.model_name,
            tools=tools,
            settings=model_settings,
        )
        client = self._resolve()
        if output_schema is not None and not output_schema.is_plain_text():
            response = await client.structured(request, output_schema.json_schema())
            response = _as_text_response(response)
        else:
            response = await client.invoke(request)
        return SDKModelResponse(
            output=_output_items(response),
            usage=_usage(response),
            response_id=previous_response_id,
        )

    async def stream_response(  # noqa: PLR0917 - the SDK's Model interface is positional
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Not implemented, and it says so rather than pretending.

        The SDK's streaming contract is a sequence of Responses-API *server* events
        (``response.created``, ``response.output_text.delta``, … ) which a chat-completions
        gateway does not produce; synthesising them convincingly is a translation layer of
        its own, and one that silently emitted a wrong event order would break the runner in
        ways a caller could not diagnose. ``Runner.run`` (non-streaming) is fully supported,
        and the harness's own ``RunEvent`` stream is what a UI should watch — that is what
        the AG-UI surface consumes. See this adapter's README.
        """
        raise NotImplementedError(
            "trellis-harness-openai-agents does not implement SDK streaming: use Runner.run "
            "and watch the harness RunEvent stream (see the adapter README)"
        )
        yield  # pragma: no cover - makes this an async generator, as the interface requires


def _as_text_response(response: ModelResponse) -> ModelResponse:
    """A structured answer, rendered as the text the SDK's output schema will validate."""
    if response.text:
        return response
    return response.model_copy(update={"text": json.dumps(response.data)})


class BifrostModelProvider(ModelProvider):
    """Resolves every model name to the gateway, so ``Agent(model="...")`` keeps working."""

    def __init__(self, *, client: Any = None, default_model: str | None = None) -> None:
        self._client = client
        self.default_model = default_model

    def get_model(self, model_name: str | None) -> Model:
        return BifrostModel(client=self._client, model=model_name or self.default_model)
