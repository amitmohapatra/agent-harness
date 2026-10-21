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
  overflow summarized and retried), Deep Agents' ``FilesystemMiddleware`` on the run's journal
  (a large tool result saved as a file, a head-and-tail preview in its place, ``read_file``
  reading it), Deep
  Agents' ``PatchToolCallsMiddleware`` (a call left unanswered gets an answer), parallel tool
  calls and ``response_format`` for ``output=``;
* **native, on when a task needs them** (``trellis.harness.automatic``) — planning, scratch
  files and the ``task`` helper on a long task;
* **the harness's** (``trellis.harness.middleware``) — :class:`~.HarnessTools` (the run's
  tools per model call, sorted; writes in the model's order), :class:`~.ModelHooks` (the
  hooks, the ``chat`` span, ``model_timeout`` and the run's time left, the ``prompt`` pinned
  for the run), :class:`~.StepLimit` (``max_steps``, then one answer without tools),
  :class:`~.StallGuard` (``max_repeats``), :func:`~.read_result` (a cleared result read back,
  :class:`~.ReadTools` offering the read tools once they have something to read) and
  :class:`~.RunCheckpointer` (the graph's checkpoint in the run: a resume continues in place);
* **yours** — ``middleware=[...]``: any LangChain or Deep Agents middleware (planning, the full
  filesystem, sub-agents, approvals, call limits, PII...), placed before ``ModelHooks``; one
  named as a default one (``SummarizationMiddleware``, ``FilesystemMiddleware``...) replaces it.

A model name is a model of the gateway (``BIFROST_URL``, the virtual key
``BIFROST_VIRTUAL_KEY``), asked through ``ChatOpenAI`` with the gateway's deny-all MCP scope —
or, for a vendor's model whose own API the gateway serves, through that API (``ChatAnthropic``,
OpenAI's Responses API; ``native_client``); any LangChain chat model is used as it is. The
context window is ``context_window``, else the model's profile, else the gateway's model
listing, else the table of known models, else :data:`CONTEXT_WINDOW` (``models.window_of``).
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, Final

from bifrost_sdk import NO_GATEWAY_TOOLS
from pydantic import BaseModel

from trellis.contracts import ConfigurationError
from trellis.harness import models
from trellis.harness.adapters.base import MODEL_METADATA, WINDOW_METADATA, context_window
from trellis.harness.prompts import Prompt
from trellis.harness.settings import Settings

log = logging.getLogger("trellis.models")

#: The window assumed when nothing says, in tokens (``models.window_of``).
CONTEXT_WINDOW: Final = models.DEFAULT_WINDOW
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
#: or 5xx, honouring ``Retry-After``), within ``model_timeout`` and the run's time. Bifrost
#: retries no model call of its own unless a provider's ``network_config.max_retries`` says
#: (its default is 0), so these are the only retries by default; an error the gateway gives
#: after it tried the request's fallbacks (or its own) is not retried here: every model was
#: already tried (:func:`_gateway_answered`).
MODEL_RETRIES: Final = 2
#: The ``Authorization`` a gateway without a virtual key is sent (the client needs one).
NO_KEY: Final = "no-virtual-key"


def ReAct(
    system: str,
    model: str | Any,
    *,
    output: type[BaseModel] | dict[str, Any] | None = None,
    max_steps: int | None = None,
    max_repeats: int | None = None,
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
    fallback_models: Sequence[str | Any] = (),
) -> Any:
    """A ``create_agent`` graph answering with ``system`` and ``model`` (a gateway model name,
    or a LangChain chat model), its answer an ``output`` model when given. ``max_steps`` model
    calls with tools, then one without (:class:`~.StepLimit`); ``max_repeats`` steps repeating
    one call stop it (:class:`~.StallGuard`); a model call takes at most ``model_timeout``
    seconds; ``prompt`` is a prompt of the harness's prompt sources (``"name"``,
    ``"name@version"``, a ``Prompt``) pinned for the run — a stored prompt of the gateway
    selected by every call, any other rendered into the instructions (``prompt_vars`` fill its
    ``{{variables}}``); ``middleware`` adds or replaces middleware by name; ``checkpointer``
    replaces the run's (:class:`~.RunCheckpointer`). ``tool_guidance`` adds how to choose among
    tools to every call's instructions (:class:`~.ToolGuidance`); ``answer_check`` has the
    model check its answer against the request once before it is final, calling the tools
    still needed, and the first change against the rules before it is made
    (:class:`~.AnswerCheck`: one more model call each); ``capabilities`` tells the model what
    it can and cannot do beyond its tools (``{"accept an image upload": False}``), so it
    answers such a question plainly. ``effort`` fixes the reasoning effort of every step of a
    model that takes one (else each step's own, :class:`~.automatic.Effort`).
    ``fallback_models`` are tried in order when a model call fails (names, as ``model``, or
    chat models; :class:`~.automatic.Fallback`). ``output`` may be a pydantic model or a JSON
    schema."""
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
    if context_window is not None and context_window <= 0:
        raise ConfigurationError("context_window is a number of tokens over 0")
    named = model if isinstance(model, str) else None
    detected = _detected(named, given=context_window, profile=_window(model))
    window = detected.tokens
    known = None if detected.source == "default" else window
    chat = _gateway_model(model, timeout=model_timeout, window=window)
    native = _native_model(named, timeout=model_timeout, window=window) if named else None
    # a model the gateway serves falls back there (Bifrost's own fallbacks, never two
    # layers); the harness falls back only to a model of the user's own, or for one
    forwarded = [m for m in fallback_models if named and isinstance(m, str)]
    fallbacks = [
        _built(m, timeout=model_timeout, window=window)
        for m in fallback_models
        if not (named and isinstance(m, str))
    ]
    gateway = [c for c in (chat, native) if named and c is not None]
    backend = automatic.JournalBackend()
    defaults = [
        # the vendor's own API, when the gateway serves it, for every call (outermost)
        *([automatic.NativeClient(native)] if native is not None else []),
        automatic.JournalFiles(backend, tools=["read_file", *sorted(automatic.SCRATCH)]),
        # what the model is offered, decided before anything reads the request's tools: the
        # tools the task is about, and what a long task needs
        automatic.ToolSearch(window=window),
        automatic.Planning(window=window),
        automatic.Helpers(backend, window=window),
        automatic.ScratchFiles(window=window),
        *([ours.AnswerCheck()] if answer_check else []),
        ours.HarnessTools(),
        ours.StallGuard(max_repeats or ours.MAX_REPEATS),
        ours.StepLimit(max_steps or ours.MAX_STEPS),
        automatic.Budgeted(),
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
        automatic.StructuredRepair(),
        *([automatic.Fallback(*fallbacks)] if fallbacks else []),
        *([automatic.GatewayFallbacks(forwarded, gateway)] if forwarded else []),
        automatic.PromptCache(),
        automatic.Effort(effort),
        ours.ModelHooks(timeout=model_timeout, prompt=prompt, prompt_vars=prompt_vars),
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
    # what the harness reads of the graph: the judge's fallback model, the window when known
    # (the pushed memory context's budget is a share of it)
    return graph.with_config(metadata={MODEL_METADATA: named, WINDOW_METADATA: known})


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


def _detected(name: str | None, *, given: int | None, profile: int | None) -> models.Window:
    """The model's window (``models.window_of``): the gateway asked for a model name."""
    settings = Settings.from_env() if name is not None else None
    return models.window_of(
        name,
        given=given,
        profile=profile,
        base_url=settings.bifrost_url if settings is not None else None,
        key=settings.bifrost_virtual_key if settings is not None else None,
    )


def detected_window(model: Any, context_window: int | None = None) -> models.Window:
    """The context window ``ReAct(model, context_window=)`` builds with, and what said so
    (``given``, ``profile``, ``gateway``, ``server``, ``table``, ``default``)."""
    named = model if isinstance(model, str) else None
    return _detected(named, given=context_window, profile=_window(model))


def _built(model: str | Any, *, timeout: float | None, window: int) -> Any:
    """A model as ``ReAct`` asks it: a name its vendor's own client when the gateway serves
    it, else the gateway's generic one; a chat model as it is."""
    if not isinstance(model, str):
        return model
    native = _native_model(model, timeout=timeout, window=window)
    return native if native is not None else _gateway_model(model, timeout=timeout, window=window)


def _native_model(name: str, *, timeout: float | None, window: int) -> Any | None:
    """The vendor's own client for a vendor's model name, on the gateway's endpoint for that
    vendor's API — ``ChatAnthropic`` on its Anthropic endpoint, ``ChatOpenAI`` on its Responses
    API — when the gateway serves it (``models.serves``); ``None`` otherwise (the generic
    client is used, as for any other name; said once in the log)."""
    vendor = models.vendor_of(name)
    settings = Settings.from_env()
    if vendor is None or settings.bifrost_url is None:
        return None
    key = settings.bifrost_virtual_key or NO_KEY
    if not models.serves(settings.bifrost_url, key, vendor):
        _generic(name, vendor)
        return None
    common: dict[str, Any] = {
        "default_headers": dict(NO_GATEWAY_TOOLS),
        "timeout": timeout,
        "max_retries": MODEL_RETRIES,
        "profile": {"max_input_tokens": window},
    }
    if vendor == "anthropic":
        root = settings.bifrost_url.rstrip("/").removesuffix("/v1")
        return _anthropic()(
            model_name=models.bare(name),
            base_url=f"{root}/anthropic",
            api_key=key,  # type: ignore[arg-type]
            stream_usage=True,
            **common,
        )
    import openai
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=name,
        base_url=settings.bifrost_url,
        api_key=key,  # type: ignore[arg-type]
        use_responses_api=True,
        stream_usage=True,
        http_async_client=openai.DefaultAsyncHttpxClient(
            event_hooks={"response": [_gateway_answered]}
        ),
        **common,
    )


@functools.cache
def _generic(name: str, vendor: str) -> None:
    log.info(
        "%s: the gateway does not serve %s's own API; the generic client is used", name, vendor
    )


def _gateway_model(model: Any, *, timeout: float | None, window: int) -> Any:
    """A chat model of the gateway for a model name (``ChatOpenAI`` on ``BIFROST_URL``, with the
    deny-all MCP scope and the window as its profile); any other model as it is."""
    if not isinstance(model, str):
        return model
    import openai
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
        # a client of its own: langchain-openai shares one per gateway URL across the process,
        # whose pooled connections break the models of a later event loop
        http_async_client=openai.DefaultAsyncHttpxClient(
            event_hooks={"response": [_gateway_answered]}
        ),
    )


async def _gateway_answered(response: Any) -> None:
    """A gateway's error after it tried every model — the request's ``fallbacks``, or its own
    (the error's ``extra_fields.routing_info.is_fallback``) — is not retried by the client
    (``x-should-retry: false``, which the openai and anthropic SDKs honour): retrying would
    try them all again. Any other error is retried as the client does."""
    if response.status_code < 400:
        return
    try:
        sent = json.loads(response.request.content or b"{}")
        walked = isinstance(sent, dict) and bool(sent.get("fallbacks"))
        if not walked:
            said = json.loads(await response.aread() or b"{}")
            info = (said.get("extra_fields") or {}).get("routing_info") or {}
            walked = isinstance(info, dict) and info.get("is_fallback") is True
    except Exception:
        return
    if walked:
        response.headers["x-should-retry"] = "false"


def _anthropic_http(**params: Any) -> Any:
    """The HTTP client of a gateway ``ChatAnthropic``: its own (as the generic client's), the
    gateway's errors read by :func:`_gateway_answered`."""
    from langchain_anthropic._client_utils import _AsyncHttpxClientWrapper

    return _AsyncHttpxClientWrapper(**params, event_hooks={"response": [_gateway_answered]})


@functools.cache
def _anthropic() -> type[Any]:
    """``ChatAnthropic`` on the gateway's Anthropic endpoint, with an HTTP client of its own
    (:func:`_anthropic_http`; LangChain shares one per URL across the process)."""
    import anthropic
    from langchain_anthropic import ChatAnthropic

    class GatewayAnthropic(ChatAnthropic):
        @cached_property
        def _async_client(self) -> anthropic.AsyncClient:
            params = self._client_params
            http = {k: params[k] for k in ("base_url", "timeout") if k in params}
            return anthropic.AsyncClient(**params, http_client=_anthropic_http(**http))

    return GatewayAnthropic


def _profiled(model: Any, window: int) -> Any:
    """The model, its profile naming the window ReAct detected (what the summarization's
    thresholds are shares of): a copy, when the model says otherwise or nothing."""
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict) and profile.get("max_input_tokens") == window:
        return model
    return model.model_copy(update={"profile": {**(profile or {}), "max_input_tokens": window}})
