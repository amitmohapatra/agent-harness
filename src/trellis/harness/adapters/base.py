"""The adapter contract: four functions per framework, and one check.

* ``prepare_input(target, input, context, history)`` — the framework's input, with the pushed
  memory context as a system message (``thread_store(target, options)``: the store the
  framework keeps the thread's messages in; when ``history.owner`` says it owns them, the
  context is asked for without the recent conversation) and, where memory owns the
  conversation and the framework keeps no store, its recent messages (``history``) as the
  framework's own messages, as its checkpoint would replay them (``replays_history``; a
  framework that takes no history of messages gets them in the context's text instead);
* ``invoke(target, native_input, run)`` / ``stream(...)`` — run it (the stream yields text
  deltas, then an :class:`Output` with what ``invoke`` would have returned);
* ``extract(target, output)`` — the answer, the transcript, and the framework's own pause;
* ``resume_input(target, native_input, pending, resolution)`` — what continues a paused run;
* ``check_options(options)`` — refuses framework options (``framework_options=``) its run call
  cannot take: they reach the framework's own run call unchanged (``runtime.framework_options``);
* ``native(target)`` — the tools the framework gives the model itself, beside the harness's
  (an OpenAI Agents agent's own, hosted, MCP servers' and handoff tools — its handoff agents'
  and agent tools' agents' too —, Claude's built-ins and its MCP servers' tools, the tools a
  compiled graph's tool nodes run, a Deep Agent's sub-agents'), as specs, sorted by name, each
  once (:func:`unique`): with the harness's they are the run's tool set
  (``Runtime.tool_set``), which the tool hints, ``tool_search``, the catalog and the
  narrowing read.

``run`` carries the per-run tools (already converted by ``tools.convert`` for the format the
adapter names) and the runtime. Adapters never wrap models, never re-implement a loop, and
use only their framework's public API.

``narrows`` says how far the tool schemas sent to the model follow the tool hints
(``Runtime.offers``): ``"turn"`` — every model call sees the tools offered at that moment (a
``create_agent`` graph, Deep Agents' and ``ReAct`` among them: ``middleware.HarnessTools``, or
the handler ``middleware.narrowing`` adds to the copy the harness runs; OpenAI Agents through
``FunctionTool.is_enabled``); ``"run"`` — the tools offered when the run starts (Claude: the
CLI lists an MCP server's tools once per query); ``"none"`` — the target binds its model's
tools in its own code (a hand-written LangGraph graph) or has no model (a function).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, Protocol

from trellis.contracts import ConfigurationError, InterruptResolution, ToolSpec
from trellis.harness.history import Said
from trellis.harness.journal import Pending
from trellis.harness.runtime import HELPER as HELPER_TAG
from trellis.harness.runtime import Runtime
from trellis.harness.tools.base import Tool

#: The ``RunStart.metadata`` key a run's own ``framework_options=`` is kept under (its JSON
#: values: what a later attempt — a resume, a worker's — runs with).
FRAMEWORK_OPTIONS: Final = "framework_options"


def merged(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Framework options ``over`` (a run's) on ``base`` (its agent's): a key both give as a
    mapping (a LangGraph config's ``configurable``, ``metadata``) is merged the same way, key by
    key; any other value of ``over``'s replaces ``base``'s."""
    found = dict(base)
    for key, value in over.items():
        given = found.get(key)
        if isinstance(value, Mapping) and isinstance(given, Mapping):
            found[key] = merged(given, value)
        else:
            found[key] = value
    return found


ToolFormat = Literal["langchain", "openai_agents", "claude", "none"]
Narrowing = Literal["turn", "run", "none"]


@dataclass(frozen=True, slots=True)
class Output:
    """The last item of a stream: what ``invoke`` returns."""

    value: Any


@dataclass(frozen=True, slots=True)
class NativePause:
    """A pause the framework itself reported: LangGraph's ``interrupt`` (its value is our
    interrupt when ``ask`` raised it), or an OpenAI Agents ``needs_approval`` tool."""

    value: Any
    native_id: str | None = None
    #: the framework's serialised run, when it resumes from one
    state: dict[str, Any] | None = None
    #: the tool call waiting for approval, when that is what the framework paused on
    tool: str | None = None
    args: dict[str, Any] | None = None
    #: the framework's handle of each harness pause raised at once (calls made together may
    #: each ask), by the harness's interrupt id
    ids: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Extracted:
    answer: Any
    #: what the agent said in this attempt - its tool calls and their results too, where the
    #: framework reports them - in order
    transcript: list[Said] = field(default_factory=list)
    pause: NativePause | None = None


@dataclass(slots=True)
class Invocation:
    """One attempt, as an adapter sees it."""

    runtime: Runtime
    tools: Sequence[Tool]
    #: ``tools`` in the adapter's own format (``None`` when there are none)
    native_tools: Any = None


class Adapter(Protocol):
    name: ClassVar[str]
    tool_format: ToolFormat
    #: the target's tools are fixed when it is built (a compiled graph): ``tools=`` is refused
    #: at wrap time; the memory tools come from ``h.memory_tools()`` instead
    fixed_tools: bool
    narrows: Narrowing
    #: the framework takes a conversation's history as its own messages (``prepare_input``'s
    #: ``history``); else memory's window comes in the context's text
    replays_history: bool

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None: ...

    def prepare_input(
        self,
        target: Any,
        input: Any,
        context: str | None,
        history: Sequence[Said] = (),
    ) -> Any: ...

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any: ...

    def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]: ...

    def extract(self, target: Any, output: Any) -> Extracted: ...

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any: ...

    def check_options(self, options: Mapping[str, Any]) -> None: ...

    async def native(self, target: Any) -> list[ToolSpec]: ...


#: The tag of a framework's own tool that hands the work to another agent — Deep Agents'
#: ``task``, an OpenAI agent's handoff or agent tool, Claude's ``Agent`` —: it joins the run's
#: tool set (hints, the catalog, records) but the narrowing never takes it away.
HELPER: Final = HELPER_TAG


def helper(spec: ToolSpec) -> ToolSpec:
    """``spec`` tagged :data:`HELPER`, its risk ``read``: the call only hands the work on — each
    of the other agent's own calls is governed on its own."""
    return spec.model_copy(update={"tags": [*spec.tags, HELPER], "side_effects": "read"})


def unique(specs: Iterable[ToolSpec]) -> list[ToolSpec]:
    """Tool specs sorted by name, each name once (the first found wins): a stable tool set,
    whatever order the framework lists them in (the memory context's cache key holds)."""
    found: dict[str, ToolSpec] = {}
    for spec in specs:
        found.setdefault(spec.name, spec)
    return [found[name] for name in sorted(found)]


def no_options(options: Mapping[str, Any], target: str, why: str) -> None:
    """``check_options`` of a target with no framework run call to hand options to."""
    if options:
        raise ConfigurationError(
            f"framework_options= goes to a framework's own run call, which {target} does not "
            f"have ({why}): {', '.join(map(repr, options))} cannot be passed"
        )


def query_of(input: Any) -> str:
    """The text a memory lookup uses: the input itself, the last user message, or the
    first text field of a structured input."""
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        messages = input.get("messages")
        if isinstance(messages, list):
            return query_of(messages)
        for key in ("query", "question", "input", "prompt", "text", "message"):
            value = input.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(input, list):
        for message in reversed(input):
            role = (
                message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
            )
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if role in ("user", "human") and isinstance(content, str):
                return content
    return ""


#: Where a target (or its ``model``) says how many tokens its model reads: an attribute of
#: either name, or a LangChain chat model's ``profile["max_input_tokens"]``.
WINDOW_ATTRIBUTES: Final = ("context_window", "max_input_tokens")
#: What a graph ``ReAct(...)`` built says of itself in its config's metadata: its gateway
#: model's name (when it was given one) and its context window.
MODEL_METADATA: Final = "trellis_model"
WINDOW_METADATA: Final = "trellis_context_window"


#: The attribute a graph ``ReAct(...)`` built carries its declared tools on (``tools=``, its
#: ``subagents=`` among them): what ``h.wrap`` gives its runs.
TOOLS_ATTRIBUTE: Final = "trellis_tools"


def declared_tools(target: Any) -> list[Any]:
    """The tools ``target``'s builder declared for the harness to run (``ReAct(tools=,
    subagents=)``); none for any other target: its tools are the framework's own."""
    return list(getattr(target, TOOLS_ATTRIBUTE, None) or ())


def metadata_of(target: Any) -> dict[str, Any]:
    """The metadata a compiled graph's config carries (``{}`` for any other target)."""
    config = getattr(target, "config", None)
    found = config.get("metadata") if isinstance(config, dict) else None
    return found if isinstance(found, dict) else {}


def context_window(target: Any) -> int | None:
    """The context window of the target's model, in tokens, when the target or its model
    says, or the graph's metadata (``ReAct``) does (``None`` otherwise: a compiled graph of
    your own, a model named by a string)."""
    for holder in (target, getattr(target, "model", None)):
        profile = getattr(holder, "profile", None)
        found = [getattr(holder, name, None) for name in WINDOW_ATTRIBUTES]
        found.append(profile.get("max_input_tokens") if isinstance(profile, dict) else None)
        found.append(metadata_of(holder).get(WINDOW_METADATA))
        for value in found:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None
