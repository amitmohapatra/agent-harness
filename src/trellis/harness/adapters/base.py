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
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol

from trellis.contracts import ConfigurationError, InterruptResolution, ToolSpec
from trellis.harness.history import Said
from trellis.harness.journal import Pending
from trellis.harness.runtime import BUILTIN as BUILTIN_TAG
from trellis.harness.runtime import HELPER as HELPER_TAG
from trellis.harness.runtime import Runtime
from trellis.harness.tools.base import Tool

if TYPE_CHECKING:
    from trellis.harness.bridges.async_tasks import AsyncTasks

#: The ``RunStart.metadata`` key a run's own ``framework_options=`` is kept under (its JSON
#: values: what a later attempt — a resume, a worker's — runs with).
FRAMEWORK_OPTIONS: Final = "framework_options"


#: The framework options that are the deployment's, by adapter: given on ``h.wrap`` (the
#: deployment's code), never changed by a run's own ``framework_options`` (:func:`deployed`) —
#: which MCP servers the agent reaches, what it runs without asking.
DEPLOYMENT_ONLY: Final[Mapping[str, Mapping[str, str]]] = {
    "claude_agent_sdk": {
        "mcp_servers": "which MCP servers it reaches",
        "allowed_tools": "the tools it runs without asking",
        "permission_mode": "a permission mode that lets calls through without asking",
        "extra_args": "the CLI's own flags: its permission mode, allowed tools, MCP servers",
        "settings": "Claude Code's settings: its permission rules",
    },
    "acp": {"mcp_servers": "which MCP servers it reaches"},
}
#: The permission modes a run may still choose for itself: they let nothing through unasked.
ASKING_MODES: Final = frozenset({"default", "plan"})


def deployed(adapter: str, wrapped: Mapping[str, Any], own: Mapping[str, Any]) -> None:
    """Refuse a run's own framework options (``own``) that change what the deployment gave on
    ``h.wrap`` (``wrapped``) of :data:`DEPLOYMENT_ONLY`: a run's caller acts in a user's name,
    and those are the administrator's. A permission mode that asks (:data:`ASKING_MODES`) is
    the run's to choose."""
    kept = DEPLOYMENT_ONLY.get(adapter, {})
    changed = [
        key
        for key, value in own.items()
        if key in kept
        and value != wrapped.get(key)
        and not (key == "permission_mode" and value in ASKING_MODES)
    ]
    if changed:
        said = "; ".join(f"{key!r} ({kept[key]})" for key in changed)
        raise ConfigurationError(
            f"a run's framework_options cannot set {said}: that is the deployment's, given on "
            "h.wrap(framework_options=...) or the agent's own options"
        )


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


ToolFormat = Literal["langchain", "openai_agents", "claude", "mcp", "none"]
Narrowing = Literal["turn", "run", "none"]
HistoryIn = Literal["messages", "context"]
#: What runs code in an agent (``Adapter.code``): a shell or sandbox of its own (``own``),
#: Deep Agents' ``execute`` on the sandbox the harness gives it (``default``: ``ReAct``'s
#: ``sandbox.backend.CodeBackend``), or nothing (``none``: the harness gives it ``sandbox_exec``).
Code = Literal["own", "default", "none"]
#: The attribute marking a backend whose ``execute`` is the harness's default code
#: (``sandbox.backend.CodeBackend``), read with no import of it.
DEFAULT_CODE: Final = "trellis_default_code"


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
    #: the JSON Schema of the answer the framework takes (LangGraph's
    #: ``interrupt(response_schema=...)``), which it checks the answer against
    schema: dict[str, Any] | None = None
    #: the agent's plan when it paused, as its graph's state keeps it (``todos``: LangChain's
    #: ``TodoListMiddleware``, Deep Agents' ``write_todos``)
    plan: list[Any] | None = None


@dataclass(frozen=True, slots=True)
class Extracted:
    answer: Any
    #: what the agent said in this attempt - its tool calls and their results too, where the
    #: framework reports them - in order
    transcript: list[Said] = field(default_factory=list)
    pause: NativePause | None = None
    #: the run's question as the framework's own conversation keeps it, when it holds one: what
    #: the run records as asked (a graph's middleware may have changed it: redacted)
    asked: str | None = None


@dataclass(slots=True)
class Invocation:
    """One attempt, as an adapter sees it."""

    runtime: Runtime
    tools: Sequence[Tool]
    #: ``tools`` in the adapter's own format (``None`` when there are none)
    native_tools: Any = None


class Keeps:
    """What an adapter whose framework keeps nothing of the kind answers with: no background
    tasks of its own, nothing checked where it is wrapped, nothing running between runs,
    nothing kept of a session, nothing going for a cancelled child. An adapter that keeps
    one of these says so by defining it. And what one that does not say otherwise is taken
    to have: tools of its framework's own (:meth:`own_tools`)."""

    @property
    def tasks(self) -> AsyncTasks | None:
        return None

    def wrapped(self, target: Any, options: Mapping[str, Any], gateway: Any) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def session_gone(self, session: str) -> None:
        return None

    async def child_cancelled(self, run_id: str, reason: str | None) -> None:
        return None

    def kept_children(self, run_id: str) -> list[str]:
        return []

    def own_tools(self, target: Any, sources: Sequence[Any]) -> bool:
        return True  # Claude's built-in tools, an ACP agent's own: the safe answer

    async def held_runs(self, target: Any, runtime: Any) -> frozenset[str]:
        return frozenset()

    def code(self, target: Any, options: Mapping[str, Any]) -> Code:
        return "own"  # Claude Code's shell, an ACP agent's (Kimi Code's): theirs, as they are

    def give_shell(self, target: Any) -> bool | None:
        return None  # no shell of the framework's own for the harness to turn on


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
    #: where a conversation's earlier turns from the run store go, with memory off
    #: (``Agent.recalled``): its own messages (``prepare_input``'s ``history``; a function's
    #: ``agent.history``), or the context's text
    history_in: HistoryIn
    #: the framework keeps a session of its own (the journal's ``session``), which the
    #: conversation's next turn continues (``Agent.sessions``, ADR 0005)
    sessions: bool

    @property
    def tasks(self) -> AsyncTasks | None:
        """The framework's own background tasks this process holds the runs of (Deep
        Agents' async tasks: ``bridges.async_tasks``); ``None`` for a framework with none."""
        ...

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None: ...

    def wrapped(self, target: Any, options: Mapping[str, Any], gateway: Any) -> None:
        """Where the agent is wrapped (``h.wrap``), with the options every one of its runs
        is given and the harness's model route: what the framework checks of its own."""
        ...

    async def aclose(self) -> None:
        """The agent is closing: what the framework keeps running between runs in this
        process ends."""
        ...

    def session_gone(self, session: str) -> None:
        """``session`` is kept for no conversation's next turn any more (``Agent.sessions``):
        the framework lets go of what it keeps of it."""
        ...

    async def child_cancelled(self, run_id: str, reason: str | None) -> None:
        """The run ``run_id`` ended with the run that started it: what the framework keeps
        going for it outside any attempt stops (``children.stop_kept``)."""
        ...

    def kept_children(self, run_id: str) -> list[str]:
        """The children of run ``run_id`` the framework keeps at work in this process (a live
        session's background subagents, Deep Agents' async tasks): what stops with the run
        even when the run store cannot list its children (``children.end_children``)."""
        ...

    def own_tools(self, target: Any, sources: Sequence[Any]) -> bool:
        """Whether ``target`` has tools of its framework's own, whose effects the harness does
        not know (``sources``: the harness tools it declares): wrapped as a tool, the agent
        then writes whatever its declared tools say (``subagents``)."""
        ...

    def steers(self, target: Any) -> bool:
        """Whether ``target`` takes a message between two of its steps (``mode="steer"``,
        ``Runtime.steered``): its framework has a hook there the harness uses."""
        ...

    def asks(self, target: Any) -> bool:
        """Whether ``target``'s framework gives its model a question tool of its own, which
        the harness asks as the run's question: such a run is offered no ``ask_user``
        (``Agent.tools_for``)."""
        ...

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

    async def held_runs(self, target: Any, runtime: Any) -> frozenset[str]:
        """The runs whose turns the framework's thread store, kept in this process's RAM, holds
        of ``runtime``'s thread (none: it keeps none, or does not say whose they are): memory's
        rendered window is left out where they are every turn the run store lists
        (``Agent.replays_itself``)."""
        ...

    async def native(self, target: Any) -> list[ToolSpec]: ...

    def workdir(self, target: Any, options: Mapping[str, Any]) -> Path | None:
        """The folder on this machine the framework works and writes in (``options``: the run's
        framework options), whose ``outputs/`` is the user's (``sandbox.delivered``); ``None``
        for a framework with no such folder."""
        ...

    def code(self, target: Any, options: Mapping[str, Any]) -> Code:
        """What runs code in ``target`` run with ``options`` (:data:`Code`): the harness gives
        an agent with none its own code tool (``sandbox.code_for``)."""
        ...

    def give_shell(self, target: Any) -> bool | None:
        """Turn on, for ``target``, the framework's own shell that it has and that runs
        nothing as it was built — on the working folder the harness gives the run
        (``sandbox.code_for``, which then adds no tool of its own): ``True`` when it is on;
        ``None`` when the framework has none to turn on; ``False`` when ``target`` must have
        no shell at all (the framework's own rule: a Deep Agent with permissions over its
        files)."""
        ...


#: The tag of a framework's own tool that hands the work to another agent — Deep Agents'
#: ``task``, an OpenAI agent's handoff or agent tool, Claude's ``Agent`` —: it joins the run's
#: tool set (hints, the catalog, records) but the narrowing never takes it away.
HELPER: Final = HELPER_TAG


#: The tag of a tool the framework gives the model of itself (Claude Code's and Kimi's built-ins,
#: Deep Agents' own): of the run's tool set, but the narrowing's threshold counts only the run's
#: own tools (``Runtime.own_tools``), so the framework's never narrow a run's few away.
BUILTIN: Final = BUILTIN_TAG


def builtin(spec: ToolSpec) -> ToolSpec:
    """``spec`` tagged :data:`BUILTIN`."""
    return spec.model_copy(update={"tags": [*spec.tags, BUILTIN]})


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
#: model's name (when it was given one), its context window and its instructions' text (what
#: the replayed conversation's budget leaves room for).
MODEL_METADATA: Final = "trellis_model"
WINDOW_METADATA: Final = "trellis_context_window"
PROMPT_METADATA: Final = "trellis_system_prompt"
#: What one model call's request may hold, as a share of the window: what Deep Agents'
#: summarization budgets (its ``_input_budget``: 95%, less the reply's tokens), so a request
#: ``FitWindow`` fitted is one the summarization after it never refuses.
FIT_SHARE: Final = 0.95


#: The attribute a graph ``ReAct(...)`` built carries its declared tools on (``tools=``, its
#: ``subagents=`` among them): what ``h.wrap`` gives its runs.
TOOLS_ATTRIBUTE: Final = "trellis_tools"
#: Where a ``ReAct`` graph keeps the specs of the provider's own tools it binds to its model
#: (``provider_tools``: run by the provider, so in no tool node), of the run's tool set.
PROVIDER_TOOLS: Final = "trellis_provider_tools"


def declared_tools(target: Any) -> list[Any]:
    """The tools ``target``'s builder declared for the harness to run (``ReAct(tools=,
    subagents=)``); none for any other target: its tools are the framework's own."""
    return list(getattr(target, TOOLS_ATTRIBUTE, None) or ())


def metadata_of(target: Any) -> dict[str, Any]:
    """The metadata a compiled graph's config carries (``{}`` for any other target)."""
    config = getattr(target, "config", None)
    found = config.get("metadata") if isinstance(config, dict) else None
    return found if isinstance(found, dict) else {}


def model_name(target: Any) -> str | None:
    """The name of the target's model when the graph's metadata says it (``ReAct`` given a
    model name); ``None`` otherwise."""
    found = metadata_of(target).get(MODEL_METADATA)
    return found if isinstance(found, str) else None


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
