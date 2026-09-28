"""``BifrostChatModel``: a LangChain chat model whose every call leaves through the harness.

Design §8 proposed reaching Bifrost with ``init_chat_model("openai:...", base_url=...)``.
That works, and it imports ``langchain_openai`` and through it ``openai`` — a provider SDK
in trellis code, which is the one thing the platform does not allow (and which
``tests/unit/test_architecture.py`` now fails on). It would also route the call around
``runtime.model``, so the span, the token and cost metrics, the model policy check and the
execution's remaining deadline would all stop applying to a Deep Agents model call.

So the adapter implements LangChain's own seam instead. ``BifrostChatModel`` is a
``BaseChatModel`` that converts LangChain messages into a :class:`ModelRequest`, hands it to
whatever model client it was given — ``runtime.model`` (instrumented, over Bifrost) in a
harnessed run — and converts the :class:`ModelResponse` back into an ``AIMessage``. The
conversion is the whole cost of the decision; what it buys is one credential envelope, one
spend record and one instrumented call path for every framework.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import tool_call as make_tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from trellis.contracts.model import ModelRequest, ModelResponse

from trellis.harness.execution.sync import run_sync
from trellis.harness_deepagents.binding import active_runtime

__all__ = ["BifrostChatModel", "to_model_request"]

#: LangChain message types mapped onto the roles the gateway's chat API knows.
_ROLES: dict[str, str] = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
}


def to_model_request(
    messages: list[BaseMessage],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> ModelRequest:
    """LangChain messages as the harness's model contract sees them."""
    return ModelRequest(
        model=model,
        messages=[_wire(message) for message in messages],
        tools=list(tools or []),
        params=dict(params or {}),
    )


def _wire(message: BaseMessage) -> dict[str, Any]:
    """One LangChain message in the gateway's wire shape.

    ``content`` may be a list of blocks on a multimodal message; the gateway's chat API
    accepts that list unchanged, so it is passed through rather than flattened into text
    that would silently drop an image.
    """
    wire: dict[str, Any] = {
        "role": _ROLES.get(message.type, message.type),
        "content": message.content,
    }
    call_id = getattr(message, "tool_call_id", None)
    if call_id:
        wire["tool_call_id"] = call_id
    calls = getattr(message, "tool_calls", None)
    if calls:
        wire["tool_calls"] = [
            {
                "id": call.get("id"),
                "type": "function",
                "function": {
                    "name": call.get("name"),
                    "arguments": json.dumps(call.get("args") or {}),
                },
            }
            for call in calls
        ]
        # an assistant turn that only calls tools has no text, and some providers refuse
        # a null content alongside tool_calls
        wire["content"] = message.content or ""
    return wire


def _from_response(response: ModelResponse) -> AIMessage:
    """The harness's model response as the ``AIMessage`` LangChain expects."""
    calls = []
    invalid: list[dict[str, Any]] = []
    for raw in response.tool_calls:
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except (TypeError, ValueError):
            # a model that emits malformed JSON is a normal occurrence, and LangChain has a
            # place for it: the agent can see the broken call and retry rather than crash
            invalid.append(
                {
                    "name": function.get("name"),
                    "args": arguments,
                    "id": raw.get("id"),
                    "error": "the model's tool arguments are not valid JSON",
                    "type": "invalid_tool_call",
                }
            )
            continue
        calls.append(make_tool_call(name=function.get("name") or "", args=args, id=raw.get("id")))
    usage = response.usage
    return AIMessage(
        content=response.text or "",
        tool_calls=calls,
        invalid_tool_calls=invalid,
        response_metadata={
            "model_name": response.model,
            "provider": response.provider,
            "finish_reason": response.finish_reason,
        },
        usage_metadata=(
            {
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "total_tokens": usage.total_tokens or 0,
            }
            if usage is not None
            else None
        ),
    )


class BifrostChatModel(BaseChatModel):
    """A ``BaseChatModel`` over a harness model client. The only model Deep Agents sees.

        model = BifrostChatModel(client=runtime.model, model_name="gemini-3.8-flash")

    ``client`` is anything implementing the harness's model port (``invoke``): the
    instrumented ``runtime.model`` in a harnessed run, a :class:`BifrostModelClient`
    outside one, or a scripted double in a test. Left unset it resolves the running
    execution's ``runtime.model`` per call, so one compiled graph serves every run.
    """

    client: Any = None
    model_name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "trellis-bifrost"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """What LangChain caches and traces on. No credential is ever part of it: the model
        client holds the virtual key and this object only knows a model name."""
        return {"model_name": self.model_name, "params": self.params}

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Declare the tools for the next call, in the schema the gateway expects.

        LangChain's own ``convert_to_openai_tool`` does the projection, so a ``@tool``
        function, a Pydantic model and a raw dict all arrive in one shape.
        """
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        params = dict(self.params)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        bound = self.__class__(
            client=self.client,
            model_name=self.model_name,
            params={**params, **kwargs},
            bound_tools=schemas,
        )
        return bound

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self.client if self.client is not None else active_runtime().model
        response = await client.invoke(self._request(messages, stop, kwargs))
        return ChatResult(generations=[ChatGeneration(message=_from_response(response))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """The synchronous path, on the harness's shared bridge loop.

        Present because ``BaseChatModel`` requires it and a sync graph is legal. Prefer the
        async path: a synchronous call from inside a running loop blocks that loop's thread
        until the gateway answers.
        """
        return run_sync(self._agenerate(messages, stop, None, **kwargs))

    def _request(
        self, messages: list[BaseMessage], stop: list[str] | None, kwargs: dict[str, Any]
    ) -> ModelRequest:
        params = {**self.params, **kwargs}
        if stop:
            params["stop"] = stop
        return to_model_request(
            messages, model=self.model_name, tools=self.bound_tools, params=params
        )
