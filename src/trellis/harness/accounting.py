"""Every LangChain chat model call made inside a harness run, accounted to the run's budget
(``trellis.harness.budget``), watched for a stall (``Runtime.model_started``) and for a reply
cut at the model's output limit with no answer (``Runtime.cut_off``, :func:`cut_off`): a LangChain
callback, attached to every LangChain run of the process (``register_configure_hook``), that
watches each call from its start to its reply or its error and accounts each reply's usage —
whatever graph made the call (``ReAct``'s loop, its summarization, a grader, a ``create_agent``
or LangGraph graph of the team's own, with or without ``ModelHooks``, a function's own model).
Each call's pace (seconds per output token) and a ``429`` from the provider or its gateway are
what the child runs' adaptive limit reads of the provider (``admission.py``). Outside a harness
run it does nothing."""

from __future__ import annotations

import functools
import logging
import time
from contextvars import ContextVar
from typing import Any, Final
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tracers.context import register_configure_hook

from trellis.harness import admission, budget
from trellis.harness.bounded import LRU, SAID_ONCE
from trellis.harness.gateways import Capability
from trellis.harness.runtime import current

log = logging.getLogger("trellis.harness")


#: Where LangChain keeps why a reply ended (``response_metadata``), by provider: an
#: OpenAI-compatible endpoint's ``finish_reason``, Anthropic's ``stop_reason``, Bedrock
#: Converse's ``stopReason``, Ollama's ``done_reason``.
STOP_KEYS: Final = ("finish_reason", "stop_reason", "stopReason", "done_reason")
#: What they say when the reply reached the model's output limit (its ``max_tokens``):
#: ``length`` (OpenAI-compatible, Ollama), ``max_tokens`` (Anthropic, Bedrock), ``MAX_TOKENS``
#: (Gemini).
AT_OUTPUT_LIMIT: Final = frozenset({"length", "max_tokens", "MAX_TOKENS"})


def cut_off(message: AIMessage) -> bool:
    """Whether ``message`` is a reply cut at the model's output limit before it gave an answer:
    no text, no tool call, and the provider says it reached the limit."""
    if message.tool_calls or message.text.strip():
        return False
    said = message.response_metadata or {}
    return any(said.get(key) in AT_OUTPUT_LIMIT for key in STOP_KEYS)


class UsageAccounting(BaseCallbackHandler):
    """Each chat model reply's usage (and the gateway's cost), accounted to the current run;
    each call watched for a stall from its start to its end."""

    run_inline = True  # in the run's own task: ``current()`` is the run

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[Any], *, run_id: UUID, **kwargs: Any
    ) -> None:
        runtime = current()
        if runtime is not None:
            runtime.model_started(str(run_id))
            _STARTED[str(run_id)] = time.monotonic()
            _routed_or_said(runtime, serialized)

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        runtime = current()
        if runtime is not None:
            runtime.model_started(str(run_id))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        runtime = current()
        if runtime is not None:
            runtime.model_ended(str(run_id))
            _STARTED.pop(str(run_id))
            if _status(error) == RATE_LIMITED:
                admission.ADMISSION.throttled()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        runtime = current()
        if runtime is None:
            return
        runtime.model_ended(str(run_id))
        began = _STARTED.pop(str(run_id))
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if isinstance(message, AIMessage):
                    budget.account_reply(runtime, message)
                    runtime.cut_off = cut_off(message)
                    out = int((message.usage_metadata or {}).get("output_tokens") or 0)
                    if began is not None:
                        admission.ADMISSION.paced(time.monotonic() - began, out)


#: The status a rate-limited model call fails with.
RATE_LIMITED: Final = 429
#: When each chat model call under way started, by its LangChain run id (bounded: a call that
#: never ends is forgotten).
_STARTED: Final[LRU[str, float]] = LRU(1024)


def _status(error: BaseException) -> int | None:
    """The HTTP status a model client's error carries (its own, or its response's)."""
    found = getattr(error, "status_code", None)
    if found is None:
        found = getattr(getattr(error, "response", None), "status_code", None)
    return found if isinstance(found, int) else None


#: Where a LangChain chat model says it sends its calls (its serialized constructor
#: arguments), by the arguments' names.
BASE_URL_ARGUMENTS: Final = ("openai_api_base", "base_url", "anthropic_api_url", "api_base")


def _routed_or_said(runtime: Any, serialized: dict[str, Any] | None) -> None:
    """With Bifrost serving the harness (a route with ``Capability.ATTRIBUTION``), a chat model
    call that does not go through it — a model object built with a client of its own, in a
    graph of the team's — is said once per agent and model (it is not routed: the harness
    cannot change a model object's client)."""
    gateway = runtime.agent.harness.model_gateway
    if not gateway.has(Capability.ATTRIBUTION) or not serialized:
        return
    kind = (serialized.get("id") or ["?"])[-1]
    if kind == "RoutedChatModel":
        return
    given = serialized.get("kwargs") or {}
    urls = [given.get(name) for name in BASE_URL_ARGUMENTS]
    if any(isinstance(url, str) and url.startswith(gateway.origin) for url in urls):
        return
    name = given.get("model_name") or given.get("model") or kind
    _unrouted(runtime.agent_id, str(kind), str(name))


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and model, of the most recent
def _unrouted(agent_id: str, kind: str, model: str) -> None:
    log.warning(
        "%s: its chat model %s (%s) calls its provider directly, not Bifrost: build it with a "
        "model name (ReAct(model='...')) or point its client at LLM_BASE_URL",
        agent_id,
        model,
        kind,
        extra={"agent_id": agent_id, "model": model},
    )


#: The callback, on every LangChain run of the process (inherited by the runs it starts).
ACCOUNTING: ContextVar[UsageAccounting | None] = ContextVar(
    "trellis_usage_accounting",
    default=UsageAccounting(),  # noqa: B039 - it holds no state: one for every context
)
register_configure_hook(ACCOUNTING, inheritable=True)
