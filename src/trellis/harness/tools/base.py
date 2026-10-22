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
import functools
import inspect
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Literal, Protocol

from jsonschema import SchemaError
from jsonschema.exceptions import relevance
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

from trellis.contracts import (
    AgentError,
    ConfigurationError,
    ErrorCategory,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.contracts.errors import is_pause_signal
from trellis.harness.bounded import MAXSIZE
from trellis.harness.features import Feature

log = logging.getLogger("trellis.harness")

SideEffects = Literal["read", "write", "irreversible"]


class SideEffect(StrEnum):
    """What a tool does (:data:`SideEffects`), for comparing a value read as text."""

    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


class ToolOrigin(StrEnum):
    """Where a tool comes from (the contracts' ``ToolSpec.source``)."""

    LOCAL = "local"
    MCP = "mcp"
    MEMORY = "memory"
    OPENAPI = "openapi"
    A2A = "a2a"


#: What an unknown tool is assumed to do: something, but nothing a person must approve.
DEFAULT_SIDE_EFFECTS: SideEffects = "write"

Runner = Callable[[dict[str, Any]], Awaitable[Any]]

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
            retries = retries_of(tool.spec, reads=reads)
            output = await retried(once, retries=retries)
    except Exception as exc:  # noqa: BLE001 - the outcome the model reads
        if is_pause_signal(exc):  # it asked a person: the caller lets it through
            return ToolOutcome(tool=tool.name, status=ToolStatus.CANCELLED), exc
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


async def retried(
    call: Callable[[], Awaitable[Any]],
    *,
    retries: int,
    left: Callable[[], float | None] | None = None,
) -> Any:
    """``call()``, tried again up to ``retries`` times after an error that may pass, with a
    jittered backoff — or, when the error says when to come back (``Retry-After``), then: a
    wait longer than the time ``left`` (the run's, the call's) is not waited, the error is
    raised. The caller's timeout bounds the whole of it."""

    def passes(exc: BaseException) -> bool:
        if is_pause_signal(exc) or not isinstance(exc, Exception):
            return False
        if not AgentError.of(exc).retryable:
            return False
        after, remaining = retry_after(exc), (left or _left)()
        return after is None or remaining is None or after <= remaining

    def wait(state: RetryCallState) -> float:
        backoff = random.uniform(0, RETRY_BACKOFF_SECONDS * 2 ** (state.attempt_number - 1))  # noqa: S311 - jitter, not a secret
        error = state.outcome.exception() if state.outcome is not None else None
        return max(backoff, (retry_after(error) if error is not None else None) or 0.0)

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(retries + 1),
        retry=retry_if_exception(passes),
        wait=wait,
        sleep=asyncio.sleep,
        reraise=True,
    ):
        with attempt:
            return await call()
    raise AssertionError("unreachable: tenacity returns or raises")  # pragma: no cover


def retry_after(exc: BaseException) -> float | None:
    """When an error says to come back, in seconds: its own ``retry_after``, or the
    ``Retry-After`` header of the HTTP response it carries (seconds; a date is not read)."""
    for held in (exc, exc.__cause__):
        if held is None:
            continue
        said = getattr(held, "retry_after", None)
        response = getattr(held, "response", None)
        headers = getattr(response, "headers", None)
        if said is None and headers is not None:
            said = headers.get("retry-after")
        if said is None:
            continue
        try:
            return max(0.0, float(said))
        except (TypeError, ValueError):
            return None
    return None


def _left() -> float | None:
    """The time the code running now may still take (``Runtime.remaining``), in a run."""
    from trellis.harness.runtime import current  # noqa: PLC0415 - runtime imports tools

    runtime = current()
    return runtime.remaining() if runtime is not None else None


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


#: What the model is told after the arguments of its call could not be used.
FIX_ARGUMENTS: Final = "Call it again with arguments that fit its schema."


def not_run(tool: str, problem: str) -> str:
    """What the model reads about a call that was not run for its arguments' ``problem``."""
    return f"{tool} was not run: {problem}. {FIX_ARGUMENTS}"


def arguments_of(raw: str | None) -> dict[str, Any] | str:
    """The arguments of a call, from the text the model wrote, as an object — or what is wrong
    with them."""
    try:
        args = json.loads(raw) if raw else {}
    except ValueError as exc:
        return f"its arguments are not valid JSON ({exc})"
    return args if isinstance(args, dict) else "its arguments must be a JSON object"


def arguments_problem(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Why ``args`` do not fit a tool's input ``schema`` (JSON Schema, validated by
    ``jsonschema``), or ``None``: the missing required fields, the unknown ones where none are
    allowed, else the first other mismatch (a type, an enum, a bound...). A schema that is not
    valid JSON Schema checks nothing (said once): the tool itself validates its arguments.
    What every call is checked with before it runs (``tools.bridge``), and a reviewer's edited
    arguments (``Agent.resume``)."""
    validator = _validator(json.dumps(schema, sort_keys=True, default=str))
    if validator is None:
        return None
    errors = list(validator.iter_errors(args))
    if not errors:
        return None
    # the arguments as a whole (the schema's own ``required`` and ``properties``) first
    top = {e.validator for e in errors if not e.path}
    missing = [n for n in schema.get("required") or [] if n not in args]
    if "required" in top and missing:
        return f"missing required argument(s): {', '.join(missing)}"
    closed = schema.get("additionalProperties") is False  # declared here, not by a $ref
    unknown = [n for n in args if closed and n not in (schema.get("properties") or {})]
    if "additionalProperties" in top and unknown:
        return f"unknown argument(s): {', '.join(unknown)}"
    error = min(errors, key=relevance)
    where = ".".join(str(p) for p in error.absolute_path)
    if error.validator == "type":
        expected = error.validator_value
        kinds = " or ".join(expected) if isinstance(expected, list) else str(expected)
        return f"{where or 'the arguments'} must be of type {kinds}"
    return f"{where}: {error.message}" if where else error.message


@functools.lru_cache(maxsize=MAXSIZE)
def _validator(schema: str) -> Validator | None:
    """The validator of a tool's input schema (as sorted JSON: one per schema, kept), or
    ``None`` for one that is not valid JSON Schema."""
    loaded = json.loads(schema)
    cls = validator_for(loaded)
    try:
        cls.check_schema(loaded)
    except SchemaError as exc:
        log.warning(
            "a tool's input schema is not valid JSON Schema (%s): its arguments are not "
            "checked before it runs",
            exc.message,
        )
        return None
    return cls(loaded)
