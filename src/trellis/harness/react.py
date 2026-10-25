"""``ReAct(...)``: a tool-calling agent for teams with no framework — Deep Agents'
``create_deep_agent``, every feature of it native, with the harness's middleware on top:

    agent = h.wrap(
        ReAct("You answer stock questions.", "gpt-4o", tools=[quote], subagents=[research]),
        id="stock",
    )

It returns the compiled graph (a LangGraph target: ``LangGraphAdapter`` runs it); nothing of the
loop is the harness's. What it is built with:

* **Deep Agents, native** — its file tools on a backend (``backend=``: the graph's state by
  default — a :class:`~.automatic.JournalBackend`, kept by the run's journal — a disk, a store,
  a composite of them, or the run's sandbox: ``docker_sandbox()``), a large tool result saved as
  a file with a preview in its place, summarization near the window's end, dangling tool calls
  answered, prompt caching, its sub-agents (``subagents=``: the ``task`` tool), ``skills=``,
  ``memory_files=`` (its ``AGENTS.md``-style memory), ``permissions=``, ``response_format``,
  parallel tool calls;
* **the harness's** — :func:`middleware`, what Deep Agents has no piece for: the run's tools per
  model call (narrowed per turn: the tool hints, the memory and gateway MCP tools), governed
  with full parity; the model hooks, ``call_timeout`` and model routing; ``max_steps`` then a
  final answer; ``max_repeats``; older results cleared past half the window and read back; the
  run's budget; planning and Deep Agents' scratch tools offered on a long task
  (``trellis.harness.automatic``); optional tool guidance and an answer check; and the graph's
  checkpoint in the run (:class:`~.middleware.RunCheckpointer`: a resume continues in place);
* **yours** — ``middleware=[...]``: any LangChain or Deep Agents middleware, added before
  ``ModelHooks``; one named as the harness's is refused (naming the parameter that sets it),
  one named as Deep Agents' replaces it, as Deep Agents does.

``middleware(...)`` is the harness's part on its own: ``create_agent(...,
middleware=middleware(...))`` or ``create_deep_agent(..., middleware=middleware(...))``.

A model name is routed per call as the harness running it says (``trellis.harness.gateways``):
through Bifrost (``BIFROST_URL``, the virtual key ``BIFROST_VIRTUAL_KEY``), else to its provider
through the provider's own SDK and key. Any LangChain chat model is used as it is. The context
window is ``context_window``, else the model's LangChain profile; when neither says, the
window-scaled features are off, said once.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Final, cast

from pydantic import BaseModel

from trellis.contracts import ConfigurationError
from trellis.harness.adapters.base import (
    MODEL_METADATA,
    OWN_CHECKPOINTER,
    OWN_STORE,
    TOOLS_ATTRIBUTE,
    WINDOW_METADATA,
    context_window,
)
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.gateways import routed_chat_model
from trellis.harness.prompts import Stored

log = logging.getLogger("trellis.models")

#: The share of the window past which older tool results are cleared from the request, the
#: last :data:`KEEP_RESULTS` kept, and the share a clearing frees at least (the cached prompt
#: breaks only when it pays).
CLEAR_AT: Final = 0.5
KEEP_RESULTS: Final = 3
MIN_FREED: Final = 0.1
#: What a cleared result reads as (the tool message keeps its call's id).
CLEARED: Final = (
    "[result cleared to keep the context small: read_result(id=<this call's id>) reads it again]"
)
#: ``tool_guidance=None``: the guidance goes with a call offering at least this many tools
#: (measured: docs/frameworks/react.md#tool-choice).
GUIDE_FROM: Final = 20
#: Deep Agents' file tools that read, and those that change files: on a backend outside the
#: graph's state they are governed as any tool, with these risks.
FILE_READS: Final = frozenset({"ls", "read_file", "glob", "grep"})
FILE_WRITES: Final = frozenset({"write_file", "edit_file", "delete"})


def _checked(
    *,
    max_steps: int,
    max_repeats: int,
    call_timeout: float | None,
    context_window: int | None,
    system_prompt: Any,
) -> None:
    """The loop settings ``ReAct`` and :func:`middleware` share, checked once, the same way."""
    if call_timeout is not None and call_timeout <= 0:
        raise ConfigurationError("call_timeout is a number of seconds over 0")
    for name, value, least in (("max_steps", max_steps, 1), ("max_repeats", max_repeats, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ConfigurationError(f"{name} is a number of steps, {least} or more: not {value!r}")
    if context_window is not None and context_window <= 0:
        raise ConfigurationError("context_window is a number of tokens over 0")
    if system_prompt is not None and not isinstance(system_prompt, Stored):
        raise ConfigurationError(
            "middleware(system_prompt=) is a stored prompt (stored(...)): give a text to the "
            "graph's own system_prompt"
        )


def middleware(
    *,
    max_steps: int = 12,
    max_repeats: int = 3,
    call_timeout: float | None = None,
    context_window: int | None = None,
    tool_guidance: bool | None = None,
    answer_check: bool | None = None,
    capabilities: Mapping[str, bool] | None = None,
    system_prompt: Stored | None = None,
) -> list[Any]:
    """The harness's middleware for a ``create_agent`` or ``create_deep_agent`` graph — what
    ``ReAct`` adds to Deep Agents, the same settings meaning the same:

        graph = create_deep_agent(model, tools=[quote], middleware=middleware(max_steps=20),
                                  checkpointer=RunCheckpointer())
        agent = h.wrap(graph, id="stock")

    In order: planning on a long task (:class:`~.automatic.Planning`); the answer check
    (``answer_check``: ``None`` checks an invocation whose context
    holds a rule, ``True`` every one, ``False`` none — :class:`~.middleware.AnswerCheck`); the
    run's tools per model call, narrowed per turn, the memory and gateway MCP tools among them
    (:class:`~.middleware.HarnessTools`); the graph's own tools governed with full parity
    (:class:`~.middleware.GovernTools`); ``max_repeats`` (:class:`~.middleware.StallGuard`);
    ``max_steps`` then one answer without tools (:class:`~.middleware.StepLimit`); the run's
    budget (:class:`~.automatic.Budgeted`); on a known ``context_window``, older tool results
    cleared past half of it and ``read_result`` to read them back; the tool guidance
    (``tool_guidance``: ``None`` with :data:`GUIDE_FROM` tools or more on a known window with
    room for it, ``True`` always, ``False`` never — and ``capabilities``, what the agent can and
    cannot do, on every call);
    and :class:`~.middleware.ModelHooks` (the run's hooks, the span, ``call_timeout`` and the
    run's time left, a ``stored(...)`` ``system_prompt`` pinned for the run, the gateway's
    headers on a routed model). Outside a harness run each steps aside."""
    return _middleware(
        max_steps=max_steps,
        max_repeats=max_repeats,
        call_timeout=call_timeout,
        context_window=context_window,
        tool_guidance=tool_guidance,
        answer_check=answer_check,
        capabilities=capabilities,
        system_prompt=system_prompt,
    )


def _middleware(
    *,
    max_steps: int,
    max_repeats: int,
    call_timeout: float | None,
    context_window: int | None,
    tool_guidance: bool | None,
    answer_check: bool | None,
    capabilities: Mapping[str, bool] | None,
    system_prompt: Stored | None,
    held: frozenset[str] = frozenset(),
    files: bool = True,
    summarization: Any = None,
) -> list[Any]:
    """:func:`middleware`, and what only ``ReAct`` knows: the Deep Agents tools a long task is
    offered (``held``: on its own backend, the scratch file tools, and its general-purpose
    ``task`` helper when it has no sub-agents of its own), whether ``read_file`` is offered
    from the start (``files``: its files are not in the graph's state), and Deep Agents'
    ``summarization``, placed after what narrows the tools so it counts what the model is
    sent."""
    _checked(
        max_steps=max_steps,
        max_repeats=max_repeats,
        call_timeout=call_timeout,
        context_window=context_window,
        system_prompt=system_prompt,
    )
    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

    from trellis.harness import automatic
    from trellis.harness import middleware as ours

    window = context_window
    clearing = (
        [
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=int(CLEAR_AT * window),
                        clear_at_least=int(MIN_FREED * window),
                        keep=KEEP_RESULTS,
                        exclude_tools=(ours.READ_RESULT, ours.READ_FILE),
                        placeholder=CLEARED,
                    )
                ]
            )
        ]
        if window is not None
        else []
    )
    reads = (
        [ours.ReadTools(CLEARED if window is not None else None, files=files)]
        if window is not None or not files
        else []
    )
    # automatic, the guidance needs room: about 450 tokens a call, which a window under
    # automatic.MIN_WINDOW (or one not known) cannot spare beside many tools (measured: at 50
    # tools on an 8,192-token window it pushes the request past the window)
    guided = tool_guidance or (
        tool_guidance is None and window is not None and window >= automatic.MIN_WINDOW
    )
    guidance = (
        [
            ours.ToolGuidance(
                ours.TOOL_GUIDANCE if guided else None,
                capabilities=capabilities,
                from_tools=GUIDE_FROM if tool_guidance is None else 1,
            )
        ]
        if guided or capabilities
        else []
    )
    check = (
        [ours.AnswerCheck(with_rules_only=answer_check is None)]
        if answer_check is not False
        else []
    )
    return [
        # what a long task is offered, decided before anything reads the request's tools
        automatic.Planning(window=window),
        *([automatic.LongTaskTools(window=window, tools=held)] if held else []),
        *check,
        ours.HarnessTools(),
        ours.GovernTools(),
        ours.StallGuard(max_repeats),
        ours.StepLimit(max_steps),
        automatic.Budgeted(),
        *clearing,
        *([summarization] if summarization is not None else []),
        *reads,
        *guidance,
        ours.ModelHooks(timeout=call_timeout, prompt=system_prompt),
    ]


def ReAct(
    system_prompt: str | Stored,
    model: str | Any,
    *,
    tools: Sequence[Any] = (),
    subagents: Sequence[Any] = (),
    skills: list[str] | None = None,
    memory_files: list[str] | None = None,
    permissions: list[Any] | None = None,
    backend: Any = None,
    store: Any = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    state_schema: type[Any] | None = None,
    context_schema: type[Any] | None = None,
    cache: Any = None,
    max_steps: int = 12,
    max_repeats: int = 3,
    call_timeout: float | None = None,
    context_window: int | None = None,
    effort: str | None = None,
    tool_guidance: bool | None = None,
    answer_check: bool | None = None,
    capabilities: Mapping[str, bool] | None = None,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
) -> Any:
    """A Deep Agents graph (``create_deep_agent``) answering with ``system_prompt`` (its
    instructions: a string, or ``stored("name", **vars)``, a stored prompt of the gateway pinned
    for each run) and ``model`` (a model name, routed per call — Bifrost, else the provider's own
    key — or a LangChain chat model), its answer a ``response_format`` (a pydantic model or a
    JSON schema) when given.

    ``tools`` are what it may call: functions, ``tool(...)``, ``mcp(...)``, ``openapi(...)`` —
    run by the harness (governed, journaled, recorded) — and LangChain tools, which the graph
    runs itself (governed as the harness's). ``subagents`` are what it hands tasks to: Deep
    Agents' own (a ``SubAgent`` dict — ``name``, ``description``, ``system_prompt``, ``mode``,
    ``tools``, ``model``, ``middleware``, ``skills``, ``permissions``, ``response_format``; its
    ``interrupt_on`` is the catalog's: its tools named there ask a person — a
    ``CompiledSubAgent`` ``{"name", "description", "runnable"}``, a ``ReAct`` graph among them),
    each called through Deep Agents' ``task`` tool, and wrapped agents (``h.wrap(...)``) and
    ``a2a(url)``, each a tool of its own; every one a child run. Code runs in
    ``backend=docker_sandbox()``. Every run also has the MCP tools the virtual key allows and the
    memory tools.

    Deep Agents' own, passed as they are: ``skills``, ``memory_files`` (its ``memory``:
    ``AGENTS.md``-style files loaded into the instructions), ``permissions``, ``backend`` (by
    default the graph's state, kept in the run's journal), ``state_schema``, ``context_schema``,
    ``cache``; ``store`` and ``checkpointer`` are for the graph run on its own (``h.wrap``
    refuses them: a wrapped agent's runs keep their checkpoint in the run, and its store is the
    memory service).

    The loop's settings are :func:`middleware`'s: ``max_steps``, ``max_repeats``,
    ``call_timeout``, ``context_window`` (else the model's profile), ``tool_guidance``,
    ``answer_check``, ``capabilities``; ``effort`` is the reasoning effort sent with every call
    of a model name. ``middleware`` adds middleware."""
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import StateBackend
        from deepagents.middleware.summarization import create_summarization_middleware

        from trellis.harness import automatic
        from trellis.harness import middleware as ours
    except ImportError as exc:  # pragma: no cover - the react extra installs them
        raise ConfigurationError(
            "ReAct needs the react extra: pip install 'trellis-harness[react]'"
        ) from exc
    _checked(
        max_steps=max_steps,
        max_repeats=max_repeats,
        call_timeout=call_timeout,
        context_window=context_window,
        system_prompt=None,
    )
    if effort is not None and not isinstance(model, str):
        raise ConfigurationError("effort is for a model name: set it on your own chat model")
    named = model if isinstance(model, str) else None
    # the model call's bound is ``ModelHooks``' (with the run's time left): applied once, not
    # again by the model's own client
    chat = routed_chat_model(model, effort=effort) if isinstance(model, str) else model
    window, source = window_of(chat, context_window)
    if window is None:
        _unknown(named or type(chat).__name__)
    else:
        log.info("the context window of %s: %d tokens (%s)", named or "the model", window, source)
    own, run = _split(tools)
    native, helpers = _subagents(
        subagents, call_timeout=call_timeout, max_steps=max_steps, max_repeats=max_repeats
    )
    profiled = _profiled(chat, window)
    files_backend = backend if backend is not None else automatic.JournalBackend()
    # files outside the graph's state (a disk, a store, a sandbox) are the agent's from the
    # start, and governed; in its state, a long task is offered the scratch tools — and the
    # general-purpose helper, when the agent has no sub-agent of its own
    outside = not isinstance(files_backend, StateBackend)
    held = (frozenset() if outside else automatic.SCRATCH) | (
        frozenset() if native else frozenset({automatic.TASK})
    )
    defaults = _middleware(
        max_steps=max_steps,
        max_repeats=max_repeats,
        call_timeout=call_timeout,
        context_window=window,
        tool_guidance=tool_guidance,
        answer_check=answer_check,
        capabilities=capabilities,
        system_prompt=system_prompt if isinstance(system_prompt, Stored) else None,
        held=held,
        files=outside or bool(skills) or bool(memory_files),
        summarization=_placed(create_summarization_middleware(profiled, files_backend), PLACED),
    )
    graph = create_deep_agent(
        profiled,
        tools=own,
        system_prompt=None if isinstance(system_prompt, Stored) else system_prompt,
        # Deep Agents' summarization stands after the harness's narrowing, which it counts
        middleware=[_nothing(SUMMARIZATION), *stacked(defaults, middleware)],
        subagents=cast(Any, native),  # Deep Agents' own dicts, as given
        skills=skills,
        memory=memory_files,
        permissions=permissions,
        backend=files_backend,
        response_format=response_format,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer or ours.RunCheckpointer(),
        store=store,
        cache=cache,
        name="react",
    )
    _declared(graph, backend=files_backend, outside=outside, helper_only=not native)
    # what the harness reads of the graph: the model's name, the window when known (the
    # pushed memory context's budget is a share of it), and the tools it runs for the agent
    graph = graph.with_config(metadata={MODEL_METADATA: named, WINDOW_METADATA: window})
    setattr(graph, TOOLS_ATTRIBUTE, [*run, *helpers])
    setattr(graph, OWN_CHECKPOINTER, checkpointer is not None)
    setattr(graph, OWN_STORE, store is not None)
    return graph


def _split(tools: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """``tools`` as the graph's own (LangChain tools: its tool node runs them, governed) and
    the harness's (functions, ``tool()``, ``mcp()``, ``openapi()``: the run's toolbox). A
    sandbox is a backend, and an agent a sub-agent: either given here is refused, saying so."""
    from langchain_core.tools import BaseTool

    from trellis.harness.agent import Agent
    from trellis.harness.sandbox import SandboxSource
    from trellis.harness.tools.sources import A2ASource

    for given in tools:
        if isinstance(given, SandboxSource):
            raise ConfigurationError(
                "ReAct runs code in its backend: ReAct(backend=docker_sandbox()), not "
                "tools=[sandbox()]"
            )
        if isinstance(given, Agent | A2ASource):
            raise ConfigurationError(
                f"an agent is a sub-agent: ReAct(subagents=[...]), not tools= ({given!r})"
            )
    own = [t for t in tools if isinstance(t, BaseTool)]
    return own, [t for t in tools if not isinstance(t, BaseTool)]


def _subagents(
    subagents: Sequence[Any], *, call_timeout: float | None, max_steps: int, max_repeats: int
) -> tuple[list[dict[str, Any]], list[Any]]:
    """``subagents=`` as Deep Agents' (its ``SubAgent``, ``CompiledSubAgent`` and
    ``AsyncSubAgent`` dicts, :func:`_native`) and the harness's (wrapped agents and
    ``a2a(url)``, each a tool whose call is a child run)."""
    from langgraph.pregel import Pregel

    from trellis.harness.agent import Agent
    from trellis.harness.tools.sources import A2ASource

    native: list[dict[str, Any]] = []
    helpers: list[Any] = []
    for given in subagents:
        if isinstance(given, Agent | A2ASource):
            helpers.append(given)
        elif isinstance(given, Mapping) and {"name", "description"} <= set(given):
            native.append(
                _native(
                    dict(given),
                    call_timeout=call_timeout,
                    max_steps=max_steps,
                    max_repeats=max_repeats,
                )
            )
        elif isinstance(given, Pregel):
            raise ConfigurationError(
                "a graph is a sub-agent with its name and what it is for: "
                '{"name": ..., "description": ..., "runnable": graph} (Deep Agents\' '
                "CompiledSubAgent)"
            )
        else:
            raise ConfigurationError(
                "subagents= takes Deep Agents' sub-agents (a dict with name and description), "
                f"wrapped agents (h.wrap(...)) and a2a(url), not a {type(given).__name__}"
            )
    return native, helpers


def _native(
    spec: dict[str, Any], *, call_timeout: float | None, max_steps: int, max_repeats: int
) -> dict[str, Any]:
    """A Deep Agents sub-agent as ``ReAct`` gives it: a compiled one (``runnable``) or a remote
    one (``graph_id``) as it is; a declarative one with its model name routed as ``ReAct``'s,
    the loop's guards (``StallGuard``, ``StepLimit``, ``ModelHooks``: its own middleware of
    those names kept) and its ``interrupt_on`` turned into its tools' declared risk — a tool it
    names asks a person (``irreversible``), as the catalog says; a name that is none of its
    tools (a Deep Agents tool of its own) stays Deep Agents' approval, which the harness asks
    once."""
    if "runnable" in spec or "graph_id" in spec:
        return spec
    from trellis.harness import middleware as ours

    if isinstance(spec.get("model"), str):
        spec["model"] = routed_chat_model(spec["model"])
    given = list(spec.get("middleware") or [])
    names = {m.name for m in given}
    guards = [
        ours.StallGuard(max_repeats),
        ours.StepLimit(max_steps),
        ours.ModelHooks(timeout=call_timeout),
    ]
    spec["middleware"] = [*given, *(g for g in guards if g.name not in names)]
    gates = spec.pop("interrupt_on", None) or {}
    asked = {name for name, config in gates.items() if config}
    if "tools" in spec:
        tools = [_asking(t) if _name_of(t) in asked else t for t in spec["tools"]]
        asked -= {name for t in tools if (name := _name_of(t)) is not None}
        spec["tools"] = tools
    if asked:
        spec["interrupt_on"] = {name: gates[name] for name in sorted(asked)}
    return spec


def _name_of(tool: Any) -> str | None:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    return name if isinstance(name, str) else None


def _asking(tool: Any) -> Any:
    """``tool`` as a LangChain tool declared ``irreversible`` (governance asks a person)."""
    from langchain_core.tools import BaseTool
    from langchain_core.tools import tool as made

    found = tool if isinstance(tool, BaseTool) else made(tool)
    return found.model_copy(
        update={"metadata": {**(found.metadata or {}), "side_effects": "irreversible"}}
    )


def _declared(graph: Any, *, backend: Any, outside: bool, helper_only: bool) -> None:
    """What the harness reads of Deep Agents' tools — the agent's and its sub-agents' — set on
    them: on a ``backend`` outside the graph's state (``outside``) the file tools are governed
    as any tool, reading or writing (``middleware.OUTSIDE_STATE``); ``execute`` on a backend
    that runs nothing, and the ``task`` tool when its only helper is the general-purpose one
    (``helper_only``: offered on a long task), are the graph's machinery, not tools of the
    agent's set (``middleware.MACHINERY``)."""
    from deepagents.middleware.filesystem import supports_execution

    from trellis.harness.adapters.langgraph import TASK_TOOL, graph_tools
    from trellis.harness.middleware import MACHINERY, OUTSIDE_STATE

    runs = supports_execution(backend)
    for tool in graph_tools(graph):
        marks: dict[str, Any] = {}
        if outside and tool.name in FILE_READS | FILE_WRITES:
            risk = "read" if tool.name in FILE_READS else "write"
            marks = {OUTSIDE_STATE: True, "side_effects": risk}
        elif (tool.name == "execute" and not runs) or (tool.name == TASK_TOOL and helper_only):
            marks = {MACHINERY: True}
        if marks:
            tool.metadata = {**(tool.metadata or {}), **marks}


#: What a default middleware of ``ReAct`` is set by, where a parameter sets it: a given one of
#: the same name is refused, naming it (it would have replaced the default silently).
SET_BY: Final = {
    "StepLimit": "max_steps=",
    "StallGuard": "max_repeats=",
    "ModelHooks": "call_timeout= and system_prompt=stored(...) (and h.wrap(hooks=) for the hooks)",
    "ToolGuidance": "tool_guidance= and capabilities=",
    "AnswerCheck": "answer_check=",
    "ContextEditingMiddleware": "context_window=",
}


#: The default middleware a measurement turns off (:func:`defaults_off`), by name.
_OFF: ContextVar[frozenset[str]] = ContextVar("trellis_react_off", default=frozenset())


@contextlib.contextmanager
def defaults_off(*names: str) -> Iterator[None]:
    """``ReAct`` graphs built inside leave out the default middleware ``names`` — the
    harness's, and Deep Agents' (replaced by one that does nothing: Deep Agents replaces a
    middleware of the same name) — a measurement of what each is worth (``benchmark.cost``'s
    ``main-<piece>`` variants). Not an option of ``ReAct``: a graph built for an agent has every
    default."""
    token = _OFF.set(frozenset(names))
    try:
        yield
    finally:
        _OFF.reset(token)


def stacked(defaults: Sequence[Any], given: Sequence[Any]) -> list[Any]:
    """The middleware the graph is built with: the defaults, then the given ones, added before
    the last default (``ModelHooks``, so its span shows what the model is sent). A given one
    named as a default is refused, naming what sets that default: it never replaces it."""
    names = {m.name for m in defaults}
    taken = [m.name for m in given if m.name in names]
    if taken:
        said = "; ".join(
            f"{name} is ReAct's own ({SET_BY.get(name, 'always on')})" for name in taken
        )
        raise ConfigurationError(f"ReAct(middleware=) adds middleware, it replaces none: {said}")
    off = _OFF.get()
    kept = [m for m in defaults[:-1] if m.name not in off]
    stand_ins = [_nothing(name) for name in sorted(off - names)]
    return [*kept, *stand_ins, *given, defaults[-1]]


#: Deep Agents' summarization, by its name, and by the name ``ReAct`` places it under: after
#: the harness's middleware that narrow the tools, so its budget counts what the model is sent.
SUMMARIZATION: Final = "SummarizationMiddleware"
PLACED: Final = "ReActSummarization"


def _placed(found: Any, name: str) -> Any:
    """``found`` (a middleware) named ``name``: Deep Agents keeps one middleware per name."""
    found.__class__ = _named_class(type(found), name)
    return found


def _nothing(name: str) -> Any:
    """A middleware that does nothing, named ``name`` (the Deep Agents one it stands in for)."""
    from langchain.agents.middleware import AgentMiddleware

    return _named_class(AgentMiddleware, name)()


@functools.lru_cache(maxsize=SAID_ONCE)  # one class per (base, name), not one per build
def _named_class(base: type, name: str) -> type:
    """A subclass of ``base`` whose middleware ``name`` is ``name``."""
    return type(name, (base,), {"__module__": __name__, "name": property(lambda self: name)})


def window_of(model: Any, given: int | None = None) -> tuple[int | None, str]:
    """The context window ``ReAct`` builds with, and what said so: ``given``, else the chat
    model's LangChain profile (``profile``); ``(None, "unknown")`` when neither says."""
    if given is not None:
        return given, "given"
    found = context_window(model)
    return (found, "profile") if found is not None else (None, "unknown")


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per name, of the most recent
def _unknown(name: str) -> None:
    log.warning(
        "the context window of %s is not known: clearing old tool results and the long-task "
        "tools are off (ReAct(context_window=) turns them on)",
        name,
    )


def _profiled(model: Any, window: int | None) -> Any:
    """The model, its profile naming the window ReAct uses (what Deep Agents' summarization's
    thresholds are shares of): a copy, when it is given and the model says otherwise; the
    model as it is when the window is not known (Deep Agents' own thresholds)."""
    profile = getattr(model, "profile", None)
    if window is None or (isinstance(profile, dict) and profile.get("max_input_tokens") == window):
        return model
    return model.model_copy(update={"profile": {**(profile or {}), "max_input_tokens": window}})
