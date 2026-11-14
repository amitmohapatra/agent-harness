"""A provider's own tools in ``ReAct``: web search, web fetch, code execution — the tools a
provider runs itself, inside the model's step, given as LangChain gives them, a ``dict`` the
chat model binds as it is (Anthropic's ``{"type": "web_search_20250305", "name":
"web_search"}``, OpenAI's ``{"type": "web_search"}``):

    ReAct("You answer with sources.", "anthropic/claude-sonnet-4-5",
          tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}])

``create_agent`` binds them to the model and runs none of them (its tool node has none): the
provider does, and its reply says what each call asked and found (LangChain's standard
``server_tool_call`` and ``server_tool_result`` content blocks). The harness adds what it adds
to every tool:

* **the catalog's risk** — each is a tool of the run's set, published to the catalog; before
  each model call governance decides it as a call with no arguments (the catalog's risk, else
  its declared one: :data:`DECLARED`), under the provider's own flow, as an OpenAI agent's hosted
  tools are (a ``write`` announced, ``Decision.unasked``). No approval can stand before a call
  the provider makes inside its step, so a tool whose decision asks a person is not offered to
  that model call (said once on the run): the catalog offers it by declaring a risk that
  needs none;
* **records** — each call the reply reports is the run's tool call (``tools.bridge.observed``):
  on its stream, traced, counted, recorded in the memory service, with what it was asked and
  what it found.

A ``dict`` that is a tool a client runs (a function's schema, Anthropic's ``bash``,
``text_editor``, ``computer``, ``memory``, OpenAI's ``computer_use_preview``, ``shell``) has no
executor in the graph: refused, saying what to give instead."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from trellis.contracts import ConfigurationError, ToolOutcome, ToolSpec, ToolStatus
from trellis.harness.adapters.base import builtin
from trellis.harness.bounded import SAID_ONCE, LRUSet
from trellis.harness.runtime import current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import UNDECLARED

#: What each provider tool does, by the name it is called by (a type's name before its
#: version date: ``web_search_20250305`` is ``web_search``), where nobody declared it in the
#: catalog: a search, a fetch or a file search reads; code runs in the provider's own sandbox
#: (a write the provider keeps to itself: announced, as ``sandbox_exec`` is). A tool of none of
#: these names says nothing: it asks, so it is offered once the catalog declares it.
DECLARED: Final[Mapping[str, str]] = {
    "web_search": "read",
    "web_search_preview": "read",
    "web_fetch": "read",
    "file_search": "read",
    "tool_search_tool_regex": "read",
    "tool_search_tool_bm25": "read",
    "image_generation": "read",
    "code_execution": "write",
    "code_interpreter": "write",
}

#: The provider tools a client runs, by their name (as :data:`DECLARED`): the graph has no
#: executor for them.
CLIENT_RUN: Final = frozenset(
    {
        "function",
        "custom",
        "bash",
        "text_editor",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "computer",
        "computer_use_preview",
        "memory",
        "local_shell",
        "shell",
        "apply_patch",
    }
)

#: LangChain's standard name of a server call the provider names otherwise (its
#: ``server_tool_call`` block's ``name``): Anthropic's ``code_execution`` is LangChain's
#: ``code_interpreter``.
STANDARD: Final[Mapping[str, str]] = {"code_execution": "code_interpreter"}

#: A server result's keys that are the provider's opaque state, not what was found: left out of
#: the record (Anthropic's encrypted page content, a search result's encrypted index).
OPAQUE: Final = frozenset({"encrypted_content", "encrypted_index"})

#: The runs told a provider tool was held back, by run and tool (said once each).
_HELD_SAID: LRUSet[tuple[str, str]] = LRUSet(maxsize=SAID_ONCE)


def name_of(tool: Mapping[str, Any]) -> str:
    """A provider tool's name: its ``name``, else its type (OpenAI's ``{"type":
    "web_search"}``)."""
    return str(tool.get("name") or tool.get("type") or "")


def kind_of(tool: Mapping[str, Any]) -> str:
    """What a provider tool is: its type without the version date (``web_search_20250305`` is
    ``web_search``), else its name."""
    kind = str(tool.get("type") or tool.get("name") or "")
    stem, _, date = kind.rpartition("_")
    return stem if stem and date.isdigit() and len(date) == 8 else kind


def checked(tools: Sequence[Mapping[str, Any]]) -> list[ToolSpec]:
    """The provider tools given to ``ReAct`` as specs of the run's tool set (the framework's:
    ``BUILTIN``), each with its declared risk (:data:`DECLARED`); a tool a client runs is
    refused (:data:`CLIENT_RUN`)."""
    specs: list[ToolSpec] = []
    for tool in tools:
        kind, name = kind_of(tool), name_of(tool)
        if kind in CLIENT_RUN or name in CLIENT_RUN or "function" in tool:
            raise ConfigurationError(
                f"ReAct(tools=) got {name or kind!r} as a dict, a tool a client runs, and the "
                "graph has no code for it: give a function (or tool(...)) that does it; a dict "
                "is a tool the provider runs itself (web search, web fetch, code execution)"
            )
        if not name:
            raise ConfigurationError(f"a provider tool has a type or a name: {dict(tool)!r}")
        spec = ToolSpec(
            name=name,
            description=f"the provider's {kind}, run inside the model's step",
            side_effects=DECLARED.get(kind, UNDECLARED),
        )
        specs.append(builtin(spec))
    return specs


class ProviderTools(AgentMiddleware):
    """Around each model call of a ``ReAct`` given provider tools (``specs``, by name): each
    offered as governance decides it now, and each server call the reply reports recorded as
    the run's tool call (the module's doc). Outside a harness run it steps aside."""

    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        super().__init__()
        self.specs = {s.name: s for s in specs}
        #: a server call's name (LangChain's standard one too) to the tool it is a call of
        self.called = {STANDARD.get(n, n): s for n, s in self.specs.items()} | self.specs

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        if runtime is None:
            return await handler(request)
        held = await self._held(runtime)
        if held:
            request = request.override(
                tools=[t for t in request.tools if not (isinstance(t, dict) and name_of(t) in held)]
            )
        started = time.time()
        reply = await handler(request)
        await self._recorded(runtime, reply, started)
        return reply

    async def _held(self, runtime: Any) -> set[str]:
        """The provider tools whose decision asks a person: not offered to this call, said
        once a run each."""
        governance = runtime.agent.harness._governance(runtime.tenant)
        held: set[str] = set()
        for name, spec in self.specs.items():
            decision = await governance.check(
                name, {}, side_effects=spec.side_effects, agent_id=runtime.agent_id
            )
            if not decision.unasked().asks:
                continue
            held.add(name)
            if (runtime.run_id, name) not in _HELD_SAID:
                _HELD_SAID.add((runtime.run_id, name))
                runtime.events.warning(
                    "PROVIDER_TOOL_HELD",
                    f"{name} runs at the provider inside the model's step, where no approval "
                    f"can stand, and governance asks a person about it ({decision.reason}): "
                    "it is not offered; a risk in the tool catalog that needs no approval "
                    "offers it",
                )
        return held

    async def _recorded(self, runtime: Any, reply: ModelResponse[Any], started: float) -> None:
        """Each server call ``reply`` reports, with its result, as the run's tool call."""
        for message in (m for m in reply.result if isinstance(m, AIMessage)):
            blocks = message.content_blocks
            found = {
                str(b.get("tool_call_id")): b for b in blocks if b["type"] == "server_tool_result"
            }
            for block in blocks:
                if block["type"] != "server_tool_call":
                    continue
                name = str(block.get("name") or "")
                spec = self.called.get(name) or ToolSpec(name=name, source="local")
                ref = str(block.get("id") or f"{runtime.run_id}:{name}")
                result = found.get(ref)
                failed = result is not None and result.get("status") == "error"
                output = _plain(result.get("output")) if result is not None else None
                outcome = ToolOutcome(
                    tool=spec.name,
                    status=ToolStatus.ERROR if failed else ToolStatus.OK,
                    output=output,
                    error_class="ProviderToolError" if failed else None,
                )
                args = block.get("args")
                await bridge.observed(
                    runtime,
                    spec,
                    dict(args) if isinstance(args, Mapping) else {},
                    outcome,
                    ref=ref,
                    started=started,
                )


def _plain(output: Any) -> Any:
    """A server result's output without the provider's opaque state (:data:`OPAQUE`)."""
    if isinstance(output, Mapping):
        return {k: _plain(v) for k, v in output.items() if k not in OPAQUE}
    if isinstance(output, list):
        return [_plain(v) for v in output]
    return output


__all__ = ["DECLARED", "ProviderTools", "checked"]
