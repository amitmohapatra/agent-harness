"""``ReAct(...)``: a tool-calling agent for teams with no framework — LangChain's
``create_agent`` loop, with the native middleware a long run needs and the harness's on top.

    agent = h.wrap(ReAct(system="You answer stock questions.", model="provider/model"),
                   id="stock", tools=[quote])

It returns the compiled graph (a LangGraph target: ``LangGraphAdapter`` runs it); nothing of
the loop is the harness's. What it is built with:

* **native, on by default** — LangChain's ``ContextEditingMiddleware`` (older tool results
  cleared from the request past half the context window, the last :data:`KEEP_RESULTS` kept),
  Deep Agents' summarization (the older turns summarized near the window's end, the history
  they held saved where ``read_file`` reads it, oversized arguments truncated first, a context
  overflow summarized and retried), Deep Agents' ``FilesystemMiddleware`` with ``read_file``
  only (a large tool result saved as a file, a head-and-tail preview in its place), Deep
  Agents' ``PatchToolCallsMiddleware`` (a call left unanswered gets an answer), parallel tool
  calls and ``response_format`` for ``output=``;
* **the harness's** (``trellis.harness.middleware``) — :class:`~.HarnessTools` (the run's
  tools per model call, sorted; writes in the model's order), :class:`~.ModelHooks` (the
  hooks, the ``chat`` span, ``model_timeout`` and the run's time left, the stored ``prompt``
  pinned for the run), :class:`~.StepLimit` (``max_steps``, then one answer without tools),
  :class:`~.StallGuard` (``max_repeats``), :func:`~.read_result` (a cleared result read back)
  and :class:`~.RunCheckpointer` (the graph's checkpoint in the run: a resume continues in
  place);
* **yours** — ``middleware=[...]``: any LangChain or Deep Agents middleware (planning, the full
  filesystem, sub-agents, approvals, call limits, PII...), placed before ``ModelHooks``; one
  named as a default one (``SummarizationMiddleware``, ``FilesystemMiddleware``...) replaces it.

A model name is a model of the gateway (``BIFROST_URL``, the virtual key
``BIFROST_VIRTUAL_KEY``), asked through ``ChatOpenAI`` with the gateway's deny-all MCP scope;
any LangChain chat model is used as it is. The context window is ``context_window``, else the
model's profile, else :data:`CONTEXT_WINDOW`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from bifrost_sdk import NO_GATEWAY_TOOLS
from pydantic import BaseModel

from trellis.contracts import ConfigurationError
from trellis.harness.adapters.base import MODEL_METADATA, WINDOW_METADATA, context_window
from trellis.harness.clients.bifrost import prompt_ref
from trellis.harness.settings import Settings

#: The window assumed when neither ``context_window`` nor the model says, in tokens.
CONTEXT_WINDOW: Final = 128_000
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
#: Retries of a model call the model client makes itself (after a timeout, a 408, 409, 429
#: or 5xx, honouring ``Retry-After``), within ``model_timeout`` and the run's time.
MODEL_RETRIES: Final = 2
#: The ``Authorization`` a gateway without a virtual key is sent (the client needs one).
NO_KEY: Final = "no-virtual-key"


def ReAct(
    system: str,
    model: str | Any,
    *,
    output: type[BaseModel] | None = None,
    max_steps: int | None = None,
    max_repeats: int | None = None,
    model_timeout: float | None = None,
    context_window: int | None = None,
    prompt: str | None = None,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
) -> Any:
    """A ``create_agent`` graph answering with ``system`` and ``model`` (a gateway model name,
    or a LangChain chat model), its answer an ``output`` model when given. ``max_steps`` model
    calls with tools, then one without (:class:`~.StepLimit`); ``max_repeats`` steps repeating
    one call stop it (:class:`~.StallGuard`); a model call takes at most ``model_timeout``
    seconds; ``prompt`` is a stored prompt of the gateway (``"name"``, ``"name@version"``)
    every call selects; ``middleware`` adds or replaces middleware by name; ``checkpointer``
    replaces the run's (:class:`~.RunCheckpointer`)."""
    try:
        from deepagents.backends import StateBackend
        from deepagents.middleware import FilesystemMiddleware
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
        from deepagents.middleware.summarization import create_summarization_middleware
        from langchain.agents import create_agent
        from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

        from trellis.harness import middleware as ours
    except ImportError as exc:  # pragma: no cover - the react extra installs them
        raise ConfigurationError(
            "ReAct needs the react extra: pip install 'trellis-harness[react]'"
        ) from exc
    if model_timeout is not None and model_timeout <= 0:
        raise ConfigurationError("model_timeout is a number of seconds over 0")
    if context_window is not None and context_window <= 0:
        raise ConfigurationError("context_window is a number of tokens over 0")
    if prompt is not None:
        prompt_ref(prompt)
        if not isinstance(model, str):
            raise ConfigurationError(
                "prompt= needs a gateway model name: the gateway prepends the stored prompt"
            )
    window = context_window or _window(model) or CONTEXT_WINDOW
    chat = _gateway_model(model, timeout=model_timeout, window=window)
    backend = StateBackend()
    defaults = [
        FilesystemMiddleware(backend=backend, tools=["read_file"]),
        ours.HarnessTools(),
        ours.StallGuard(max_repeats or ours.MAX_REPEATS),
        ours.StepLimit(max_steps or ours.MAX_STEPS),
        PatchToolCallsMiddleware(),
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
        ),
        create_summarization_middleware(_profiled(chat, window), backend),
        ours.ModelHooks(timeout=model_timeout, prompt=prompt),
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
    named = model if isinstance(model, str) else None
    return graph.with_config(metadata={MODEL_METADATA: named, WINDOW_METADATA: window})


def stacked(defaults: Sequence[Any], given: Sequence[Any]) -> list[Any]:
    """The middleware the graph is built with: the defaults, each replaced by a given one of
    the same name, then the other given ones before the last default (``ModelHooks``, so its
    span shows what the model is sent)."""
    stack = list(defaults)
    names = [m.name for m in stack]
    added = []
    for m in given:
        if m.name in names:
            stack[names.index(m.name)] = m
        else:
            added.append(m)
    return [*stack[:-1], *added, stack[-1]]


def _window(model: Any) -> int | None:
    return None if isinstance(model, str) else context_window(model)


def _gateway_model(model: Any, *, timeout: float | None, window: int) -> Any:
    """A chat model of the gateway for a model name (``ChatOpenAI`` on ``BIFROST_URL``, with the
    deny-all MCP scope and the window as its profile); any other model as it is."""
    if not isinstance(model, str):
        return model
    from langchain_openai import ChatOpenAI

    settings = Settings.from_env()
    if settings.bifrost_url is None:
        raise ConfigurationError("ReAct with a model name needs BIFROST_URL")
    return ChatOpenAI(
        model=model,
        base_url=settings.bifrost_url,
        api_key=settings.bifrost_virtual_key or NO_KEY,  # type: ignore[arg-type]
        default_headers=dict(NO_GATEWAY_TOOLS),
        timeout=timeout,
        max_retries=MODEL_RETRIES,
        stream_usage=True,
        profile={"max_input_tokens": window},
    )


def _profiled(model: Any, window: int) -> Any:
    """The model, its profile naming the window (what the summarization's thresholds are
    shares of): a copy, when the model does not say."""
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict) and isinstance(profile.get("max_input_tokens"), int):
        return model
    return model.model_copy(update={"profile": {**(profile or {}), "max_input_tokens": window}})
