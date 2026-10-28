"""The techniques a ``ReAct`` arm is measured with — each a library's piece or a model server's
native switch, given to ``ReAct(middleware=[...])`` by a variant (``variants.TECHNIQUES``); none
is the harness's until the benchmark keeps it. Nothing here is logic of its own: a
:class:`Settings` only passes settings the model's client or server already takes.

* ``think`` — the chat template's own thinking switch: llama.cpp's ``chat_template_kwargs``
  (``{"thinking": true}``) in the request body, read by Granite 3.3's template on a call
  offered no tools (its ``elif thinking``); llama.cpp's Granite format splits the thoughts
  from the answer. Run the reasoning class with ``--sizes 0`` (no tools) for it to apply;
* ``slot-reuse`` — llama.cpp's ``id_slot`` and ``cache_prompt``: each run's calls on one slot,
  its prompt kept there between calls (``TRELLIS_BENCH_SLOTS``: the server's ``-np``);
* ``temp0``, ``serial-tools`` — the OpenAI API's ``temperature`` 0 and
  ``parallel_tool_calls`` off on every call offered tools;
* ``anthropic-cache`` — ``langchain_anthropic``'s ``AnthropicPromptCachingMiddleware`` (an
  Anthropic model only);
* ``openai-cache-key`` — the OpenAI API's ``prompt_cache_key``, one per agent (an OpenAI model).

Worked examples (``examples``) come from the memory service's own retrieval: the reasoning
class's dev items added as documents (:func:`seed_examples`), the similar ones pushed into a
run's context as any document is.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from benchmark.cost import reasoning
from trellis.harness.runtime import current

Handler = Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]

#: The request body's switch of a chat template's thinking (llama.cpp, ``--jinja``).
THINKING: Final = {"chat_template_kwargs": {"thinking": True}}
#: The slots a llama.cpp server serves at once (its ``-np``), when the environment says.
SLOTS_ENV: Final = "TRELLIS_BENCH_SLOTS"


def merged(settings: Mapping[str, Any], more: Mapping[str, Any]) -> dict[str, Any]:
    """``settings`` with ``more``, a request body's fields (``extra_body``) added to its own."""
    found = {**settings, **more}
    if "extra_body" in settings and "extra_body" in more:
        found["extra_body"] = {**settings["extra_body"], **more["extra_body"]}
    return found


class Settings(AgentMiddleware):
    """Model settings the client or server takes (``settings``), passed on every call — or only
    on those offered tools (``with_tools``)."""

    def __init__(self, label: str, *, with_tools: bool = False, **settings: Any) -> None:
        super().__init__()
        self.label = label
        self.with_tools = with_tools
        self.settings = settings

    @property
    def name(self) -> str:
        return f"Settings[{self.label}]"

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Handler
    ) -> ModelResponse[Any]:
        if self.with_tools and not request.tools:
            return await handler(request)
        return await handler(
            request.override(model_settings=merged(request.model_settings, self.settings))
        )


class SlotReuse(AgentMiddleware):
    """Each run's model calls on one llama.cpp slot (``id_slot``: its run's, of ``slots``), the
    prompt kept there between calls (``cache_prompt``)."""

    def __init__(self, slots: int | None = None) -> None:
        super().__init__()
        self.slots = slots or int(os.environ.get(SLOTS_ENV, "2"))

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Handler
    ) -> ModelResponse[Any]:
        runtime = current()
        if runtime is None:
            return await handler(request)
        slot = int(hashlib.sha256(runtime.run_id.encode()).hexdigest(), 16) % self.slots
        body = {"extra_body": {"id_slot": slot, "cache_prompt": True}}
        return await handler(request.override(model_settings=merged(request.model_settings, body)))


def _anthropic_cache() -> AgentMiddleware:
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

    return AnthropicPromptCachingMiddleware()


#: The techniques by the name their variant has, each a function making its middleware.
TECHNIQUES: Final[dict[str, Callable[[], AgentMiddleware]]] = {
    "think": lambda: Settings("think", extra_body=THINKING),
    "slot-reuse": SlotReuse,
    "temp0": lambda: Settings("temp0", with_tools=True, temperature=0),
    "serial-tools": lambda: Settings("serial-tools", with_tools=True, parallel_tool_calls=False),
    "anthropic-cache": _anthropic_cache,
    "openai-cache-key": lambda: Settings("openai-cache-key", prompt_cache_key="trellis-react"),
}


# --------------------------------------------------------------------------- examples
#: The user a worked-examples arm runs as: the examples are its documents, no other arm's.
EXAMPLES_USER: Final = "trellis-bench-examples"


def example_of(item: dict[str, Any]) -> str:
    """A dev item as the solved example memory keeps."""
    return f"Solved example.\nQuestion: {item['input']}\nAnswer: {item['metadata']['answer']}\n"


async def seed_examples(h: Any, user: str) -> None:
    """The reasoning class's dev items added as solved examples for ``user`` (never a test
    item): the memory service's own retrieval brings the similar ones into a run's context."""
    if h.memory is None:
        raise RuntimeError("the worked-examples arm needs memory: --memory (MEMORY_URL)")
    for item in reasoning.items():
        if item["metadata"]["split"] != "dev":
            continue
        name = re.sub(r"\W+", "-", item["metadata"]["task"]).strip("-")
        await h.add_document(
            (f"{name}.txt", example_of(item).encode(), "text/plain"),
            user=user,
            title=f"Solved example: {item['metadata']['task']}",
        )
