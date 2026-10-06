"""A tool as the harness holds it: what it is (a contracts ``ToolSpec``) and how to run it.

Every source — a local function, an MCP tool the Bifrost virtual key allows, an A2A agent, an
OpenAPI operation, the memory service's agent tools — resolves to :class:`Tool`\\ s. The native
converters (``tools.convert``) wrap a ``Tool`` in the framework's own tool type, and every
call goes through the bridge (governance, approval, journal, recording) before ``run``.

How a call is run is the same for every source, wrapped (the bridge) or not (``governed``):
one executor, :func:`execute` — at most its ``timeout``, a call that only reads (or is
idempotent) tried again after an error that may pass (:func:`retried`), a sync function in a
worker thread (:func:`invoked`), and a call that timed out told to the model as
:func:`timed_out` says (for one that does more than read: its effect is unknown).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from trellis.contracts import (
    AgentError,
    ConfigurationError,
    ErrorCategory,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness.features import Feature

SideEffects = Literal["read", "write", "irreversible"]

#: What an unknown tool is assumed to do: something, but nothing a person must approve.
DEFAULT_SIDE_EFFECTS: SideEffects = "write"

Runner = Callable[[dict[str, Any]], Awaitable[Any]]

#: A JSON schema ``type`` and the Python values that are one (a bool is not a number).
JSON_TYPES: Final[dict[str, tuple[type, ...]]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


#: Retries of a call that only reads (or is idempotent) after an error that may pass (the
#: error's own ``retryable``, or its category's): within the call's timeout, after a random
#: wait under :data:`RETRY_BACKOFF_SECONDS`, doubled for each retry. A call that does more
#: than read runs once.
READ_RETRIES: Final = 2
RETRY_BACKOFF_SECONDS: Final = 0.5
#: How long one call of a remote tool may take unless its ``timeout=`` says: an OpenAPI
#: operation (and the fetch of its document) and an A2A exchange (``remote()`` too) alike.
REMOTE_TIMEOUT_SECONDS: Final = 120.0
#: The ``ToolOutcome.metadata`` flag of a call whose effect is not known: it does more than
#: read, and timed out or was running when its worker died. Its ``error_class`` says so too
#: (:data:`OUTCOME_UNKNOWN`), which the memory service's tool records keep.
UNKNOWN: Final = "unknown"
OUTCOME_UNKNOWN: Final = "OutcomeUnknown"


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    run: Runner = field(repr=False)
    #: the part of what the harness does that this tool is (``without=`` turns it off): an
    #: MCP tool, a Bifrost Code Mode meta-tool (its nested calls are recorded from the
    #: gateway's log), a skill's or the memory service's tool; ``None``: the agent's own
    feature: Feature | None = None
    #: The most one call may take, in seconds, retries included (``None``: no limit of its
    #: own; the run's time still bounds it).
    timeout: float | None = None
    #: A call cut by a crash continues where it was when it runs again (a sub-agent: its run
    #: keeps a journal of its own), so it is run again rather than reported of unknown effect.
    resumable: bool = False

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout <= 0:
            raise ConfigurationError(f"{self.spec.name}: a timeout is a number of seconds over 0")

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def side_effects(self) -> str:
        return self.spec.side_effects


class Source(Protocol):
    """Something ``tools=[...]`` accepts. Resolved again each time the toolbox lists its
    definitions (``toolbox.TOOLS_TTL_SECONDS``)."""

    async def resolve(self) -> list[Tool]: ...


class ToolTimeout(ToolError):
    """A call that did not finish in its time (``governed``). ``unknown``: it does more than
    read, so it may or may not have taken effect."""

    code = "TOOL_TIMEOUT"
    category = ErrorCategory.TIMEOUT

    def __init__(self, message: str, *, unknown: bool) -> None:
        super().__init__(message, source="tools")
        self.unknown = unknown


def retries_of(spec: ToolSpec, *, reads: bool) -> int:
    """How often a call is tried again: :data:`READ_RETRIES` for one that only ``reads`` (its
    risk as governance sees it) or is idempotent, none for anything else."""
    return READ_RETRIES if reads or spec.idempotent else 0


async def execute(
    tool: Tool,
    args: dict[str, Any],
    *,
    reads: bool,
    within: AbstractAsyncContextManager[Any] | None = None,
) -> tuple[ToolOutcome, Exception | None]:
    """One call of ``tool`` within its time — ``within`` (a run's call: ``Runtime.limited``,
    the tool's ``timeout`` and the run's), else the tool's ``timeout`` — tried again after an
    error that may pass when it ``reads`` (or is idempotent). The outcome, and what the call
    raised: an error (``ERROR``), or out of time a :class:`ToolTimeout` (``TIMEOUT``: for a
    call that does more than read, its effect is unknown — ``metadata["unknown"]``)."""
    attempts = 0

    async def once() -> Any:
        nonlocal attempts
        attempts += 1
        return await tool.run(args)

    began = time.monotonic()
    try:
        async with within or asyncio.timeout(tool.timeout):
            output = await retried(once, retries=retries_of(tool.spec, reads=reads))
    except Exception as exc:
        if AgentError.of(exc).category is not ErrorCategory.TIMEOUT:
            failed = ToolOutcome(
                tool=tool.name,
                status=ToolStatus.ERROR,
                output=f"{tool.name} failed: {exc}",
                error_class=type(exc).__name__,
                attempts=attempts,
            )
            return failed, exc
        took = time.monotonic() - began
        text = timed_out(tool.name, took=took, limit=tool.timeout, unknown=not reads)
        late = ToolOutcome(
            tool=tool.name,
            status=ToolStatus.TIMEOUT,
            output=text,
            error_class=type(exc).__name__ if reads else OUTCOME_UNKNOWN,
            attempts=attempts,
            metadata={} if reads else {UNKNOWN: True},
        )
        timeout = ToolTimeout(text, unknown=not reads)
        timeout.__cause__ = exc
        return late, timeout
    return ToolOutcome(tool=tool.name, output=output, attempts=attempts), None


async def retried(call: Callable[[], Awaitable[Any]], *, retries: int) -> Any:
    """``call()``, tried again up to ``retries`` times after an error that may pass, with a
    jittered backoff. The caller's timeout bounds the whole of it."""
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:
            if attempt == retries or not AgentError.of(exc).retryable:
                raise
        await asyncio.sleep(random.uniform(0, RETRY_BACKOFF_SECONDS * 2**attempt))
        attempt += 1


async def invoked(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    """What ``fn(**kwargs)`` returns: an async function awaited, a sync one run in a worker
    thread (so it blocks neither the run's other work nor its timeout — though a thread,
    once started, cannot be stopped: it runs on after a timeout, its result unused)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    result = await asyncio.to_thread(fn, **kwargs)
    return await result if inspect.isawaitable(result) else result


def timed_out(tool: str, *, took: float, limit: float | None, unknown: bool) -> str:
    """What the model reads about a call that ran out of time — its ``limit``, once it ``took``
    that long, or a timeout of its own (after what it took) —: for one that does more than
    read (``unknown``), that it may have taken effect."""
    seconds = limit if limit is not None and took >= limit else round(took, 1)
    text = f"{tool} timed out after {seconds:g}s"
    if unknown:
        text += "; it may or may not have taken effect: check before calling it again"
    return text


def interrupted(tool: str) -> str:
    """What the model reads about a call that does more than read and was running when its
    worker died: whether it took effect is not known."""
    return (
        f"{tool} was interrupted by a crash; it may or may not have taken effect: check "
        "before calling it again"
    )


def arguments_problem(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Why ``args`` do not fit a tool's input ``schema``, or ``None``: a light check — required
    fields, the basic types of the declared ones, unknown fields where none are allowed; the
    tool itself validates the rest (a local tool through pydantic). What a model's call is
    checked with (``ReAct``), and a reviewer's edited arguments (``Agent.resume``)."""
    missing = [name for name in schema.get("required") or [] if name not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    properties: dict[str, Any] = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = [name for name in args if name not in properties]
        if unknown:
            return f"unknown argument(s): {', '.join(unknown)}"
    for name, value in args.items():
        expected = (properties.get(name) or {}).get("type")
        allowed = JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if allowed is None:
            continue
        wrong_bool = isinstance(value, bool) and expected != "boolean"
        if wrong_bool or not isinstance(value, allowed):
            return f"{name} must be of type {expected}"
    return None
