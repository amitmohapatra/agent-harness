"""``ReAct(...)``: a tool-calling agent for teams with no framework — LangChain's
``create_agent`` loop, with the native middleware a long run needs and the harness's on top.

    agent = h.wrap(ReAct(system="You answer stock questions.", model="provider/model"),
                   id="stock", tools=[quote])

It returns the compiled graph (a LangGraph target: ``LangGraphAdapter`` runs it); nothing of
the loop is the harness's. What it is built with:

* **native, on by default** — LangChain's ``ContextEditingMiddleware`` (older tool results
  cleared from the request past half the context window, the last :data:`KEEP_RESULTS` kept;
  only when the window is known), Deep Agents' summarization (the older turns summarized near
  the window's end — Deep Agents' own thresholds when it is not known — the history they held
  saved where ``read_file`` reads it, oversized arguments truncated first, a context overflow
  summarized and retried), Deep Agents' ``FilesystemMiddleware`` on the run's journal (a large
  tool result saved as a file, a head-and-tail preview in its place, ``read_file`` reading it),
  Deep Agents' ``PatchToolCallsMiddleware`` (a call left unanswered gets an answer), parallel
  tool calls and ``response_format`` for ``output=``;
* **native, on when a task needs them** (``trellis.harness.automatic``) — planning and scratch
  files on a long task, on a known window with room for them;
* **the harness's** (``trellis.harness.middleware``) — :class:`~.HarnessTools` (the run's
  tools per model call, sorted; writes in the model's order), :class:`~.ModelHooks` (the
  hooks, the ``chat`` span, ``model_timeout`` and the run's time left, the ``prompt`` pinned
  for the run, the gateway's attribution and trace headers), :class:`~.StepLimit`
  (``max_steps``, then one answer without tools), :class:`~.StallGuard` (``max_repeats``),
  :func:`~.read_result` (a cleared result read back, :class:`~.ReadTools` offering the read
  tools once they have something to read), the run's budget when it has one and
  :class:`~.RunCheckpointer` (the graph's checkpoint in the run: a resume continues in place);
* **yours** — ``middleware=[...]``: any LangChain or Deep Agents middleware (planning, the full
  filesystem, sub-agents, approvals, call limits, PII, tool selection...), added before
  ``ModelHooks``; one named as a default one is refused, naming the parameter that sets it.

A model name is routed per call as the harness running it says (``trellis.harness.gateways``:
``Harness(config=)``'s settings over the environment's): through Bifrost (``BIFROST_URL``, the
virtual key ``BIFROST_VIRTUAL_KEY``; the deny-all MCP scope; the gateway owns retries,
fallbacks and prompt caching), else to its provider through the provider's own SDK and key
(``gpt-4o``, ``openai:gpt-4o``, ``anthropic:claude-...``; the client's own retries). Any
LangChain chat model is used as it is. The context window is
``context_window``, else the model's LangChain profile (``model.profile``); when neither says,
the window-scaled features are off, said once.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Final

from pydantic import BaseModel

from trellis.contracts import ConfigurationError
from trellis.harness.adapters.base import MODEL_METADATA, WINDOW_METADATA, context_window
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.gateways import routed_chat_model
from trellis.harness.prompts import Prompt

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


def ReAct(
    system: str,
    model: str | Any,
    *,
    output: type[BaseModel] | dict[str, Any] | None = None,
    max_steps: int = 12,
    max_repeats: int = 3,
    model_timeout: float | None = None,
    context_window: int | None = None,
    prompt: str | Prompt | None = None,
    prompt_vars: dict[str, Any] | None = None,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
    tool_guidance: bool = False,
    answer_check: bool = False,
    capabilities: Mapping[str, bool] | None = None,
    effort: str | None = None,
) -> Any:
    """A ``create_agent`` graph answering with ``system`` and ``model`` (a gateway model name, or a
    LangChain chat model), its answer an ``output`` (a pydantic model or a JSON schema) when given.
    ``max_steps`` model calls with tools, then one without (:class:`~.StepLimit`); ``max_repeats``
    steps repeating one call stop it (:class:`~.StallGuard`); a model call takes at most
    ``model_timeout`` seconds; ``prompt`` is a prompt of the harness's prompt sources (``"name"``,
    ``"name@version"``, a ``Prompt``) pinned for the run — a stored prompt of the gateway selected
    by every call, any other rendered into the instructions (``prompt_vars`` fill its
    ``{{variables}}``); ``middleware`` adds middleware (one named as a default is refused);
    ``checkpointer`` replaces the run's (:class:`~.RunCheckpointer`). ``tool_guidance`` adds how to
    choose among tools to every call's instructions (:class:`~.ToolGuidance`); ``answer_check`` has
    the model check its answer against the request once before it is final, calling the tools still
    needed, and the first change against the rules before it is made (:class:`~.AnswerCheck`: one
    more model call each); ``capabilities`` tells the model what it can and cannot do beyond its
    tools (``{"accept an image upload": False}``), so it answers such a question plainly. ``effort``
    is the reasoning effort sent with every call of a model name (``reasoning_effort``; the gateway
    passes it on)."""
    try:
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
        from deepagents.middleware.summarization import create_summarization_middleware
        from langchain.agents import create_agent
        from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

        from trellis.harness import automatic
        from trellis.harness import middleware as ours
    except ImportError as exc:  # pragma: no cover - the react extra installs them
        raise ConfigurationError(
            "ReAct needs the react extra: pip install 'trellis-harness[react]'"
        ) from exc
    if model_timeout is not None and model_timeout <= 0:
        raise ConfigurationError("model_timeout is a number of seconds over 0")
    for name, value in (("max_steps", max_steps), ("max_repeats", max_repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{name} is a number of steps, 1 or more: not {value!r}")
    if context_window is not None and context_window <= 0:
        raise ConfigurationError("context_window is a number of tokens over 0")
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
    backend = automatic.JournalBackend()
    defaults = [
        automatic.JournalFiles(backend, tools=["read_file", *sorted(automatic.SCRATCH)]),
        # what a long task is offered, decided before anything reads the request's tools
        automatic.Planning(window=window),
        automatic.ScratchFiles(window=window),
        *([ours.AnswerCheck()] if answer_check else []),
        ours.HarnessTools(),
        ours.GovernTools(),
        ours.StallGuard(max_repeats),
        ours.StepLimit(max_steps),
        automatic.Budgeted(),
        PatchToolCallsMiddleware(),
        *(
            [
                ContextEditingMiddleware(
                    edits=[
                        ClearToolUsesEdit(
                            trigger=int(CLEAR_AT * window),
                            clear_at_least=int(MIN_FREED * window),
                            keep=KEEP_RESULTS,
                            exclude_tools=(ours.READ_RESULT, "read_file"),
                            placeholder=CLEARED,
                        )
                    ]
                )
            ]
            if window is not None
            else []
        ),
        create_summarization_middleware(_profiled(chat, window), backend),
        ours.ReadTools(CLEARED),
        *(
            [
                ours.ToolGuidance(
                    None if not tool_guidance else ours.TOOL_GUIDANCE, capabilities=capabilities
                )
            ]
            if tool_guidance or capabilities
            else []
        ),
        ours.ModelHooks(
            timeout=model_timeout,
            prompt=prompt,
            prompt_vars=prompt_vars,
            routed=isinstance(model, str),
        ),
    ]
    graph = create_agent(
        chat,
        tools=[ours.read_result()],
        system_prompt=system,
        middleware=stacked(defaults, middleware),
        response_format=output,
        checkpointer=checkpointer or ours.RunCheckpointer(),
        name="react",
    )
    # what the harness reads of the graph: the model's name, the window when known (the
    # pushed memory context's budget is a share of it)
    return graph.with_config(metadata={MODEL_METADATA: named, WINDOW_METADATA: window})


#: What a default middleware of ``ReAct`` is set by, where a parameter sets it: a given one of
#: the same name is refused, naming it (it would have replaced the default silently).
SET_BY: Final = {
    "StepLimit": "max_steps=",
    "StallGuard": "max_repeats=",
    "ModelHooks": "model_timeout= and prompt= (and h.wrap(hooks=) for the hooks)",
    "RunCheckpointer": "checkpointer=",
    "ToolGuidance": "tool_guidance= and capabilities=",
    "AnswerCheck": "answer_check=",
}


#: The default middleware a measurement turns off (:func:`defaults_off`), by name.
_OFF: ContextVar[frozenset[str]] = ContextVar("trellis_react_off", default=frozenset())


@contextlib.contextmanager
def defaults_off(*names: str) -> Iterator[None]:
    """``ReAct`` graphs built inside leave out the default middleware ``names`` (a measurement
    of what each is worth: ``benchmark.cost``'s ``main-<piece>`` variants). Not an option of
    ``ReAct``: a graph built for an agent has every default."""
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
    return [*kept, *given, defaults[-1]]


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
    """The model, its profile naming the window ReAct uses (what the summarization's
    thresholds are shares of): a copy, when it is given and the model says otherwise; the model
    as it is when the window is not known (Deep Agents' own thresholds)."""
    profile = getattr(model, "profile", None)
    if window is None or (isinstance(profile, dict) and profile.get("max_input_tokens") == window):
        return model
    return model.model_copy(update={"profile": {**(profile or {}), "max_input_tokens": window}})
