"""``HarnessTracer``: the one place that decides what a span says.

Every span the harness emits goes through here, so three policies are applied exactly once
and cannot be forgotten at a call site:

* **capture** — whether a payload may be attached at all (§26), per category;
* **redaction** — what a permitted payload looks like once redacted (§27);
* **sampling** — whether this execution is traced at all (§28), decided once per run.

The tracer is cheap when disabled: no provider calls, no dict building, a shared no-op span.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

from universal_agent_harness.config.settings import CaptureConfig
from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import MetricsRecorder
from universal_agent_harness.telemetry.redaction import DefaultRedactor
from universal_agent_harness.telemetry.sampling import SamplingDecision
from universal_agent_harness.telemetry.span import NOOP_SPAN

#: capture category -> (input flag, output flag). Memory content has its own flag because
#: it is the most sensitive payload the harness ever sees.
_CAPTURE_FLAGS = {
    "agent": ("inputs", "outputs"),
    "model": ("inputs", "outputs"),
    "prompt": ("inputs", "outputs"),
    "tool": ("inputs", "outputs"),
    "memory": ("memory_content", "memory_content"),
}


class TracedSpan:
    """A span plus the capture/redaction policy that applies to it."""

    __slots__ = ("_capture", "_category", "_redactor", "span")

    def __init__(self, span: Any, *, category: str, capture: CaptureConfig, redactor: Any) -> None:
        self.span = span
        self._category = category
        self._capture = capture
        self._redactor = redactor

    # -- attributes ---------------------------------------------------------------
    def set(self, **attributes: Any) -> None:
        clean = self._redactor.redact_attributes(attributes)
        if clean:
            self.span.set_attributes(clean)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self.set(**dict(attributes))

    def event(self, name: str, **attributes: Any) -> None:
        self.span.add_event(name, self._redactor.redact_attributes(attributes))

    # -- payloads (capture-gated) --------------------------------------------------
    def set_input(self, value: Any, *, category: str | None = None) -> None:
        if self._allowed(category or self._category, index=0):
            self.span.set_attribute(N.INPUT, _as_attribute(self._redactor.redact_input(value)))

    def set_output(self, value: Any, *, category: str | None = None) -> None:
        if self._allowed(category or self._category, index=1):
            self.span.set_attribute(N.OUTPUT, _as_attribute(self._redactor.redact_output(value)))

    def _allowed(self, category: str, *, index: int) -> bool:
        flags = _CAPTURE_FLAGS.get(category)
        return bool(flags) and bool(getattr(self._capture, flags[index], False))

    # -- status --------------------------------------------------------------------
    def error(self, error: BaseException | str, **attributes: Any) -> None:
        self.span.record_error(error, **self._redactor.redact_attributes(attributes))

    def ok(self) -> None:
        self.span.set_status_ok()


_NOOP_TRACED = TracedSpan(
    NOOP_SPAN, category="internal", capture=CaptureConfig(), redactor=DefaultRedactor()
)


class HarnessTracer:
    """Creates the harness's spans. One per harness instance; safe to share across tasks."""

    def __init__(
        self,
        provider: Any,
        *,
        capture: CaptureConfig | None = None,
        redactor: Any | None = None,
        metrics: MetricsRecorder | None = None,
        enabled: bool = True,
        decision: SamplingDecision | None = None,
    ) -> None:
        self.provider = provider
        self.capture = capture or CaptureConfig()
        self.redactor = redactor or DefaultRedactor()
        self.metrics = metrics or MetricsRecorder(provider, enabled=enabled)
        self.enabled = enabled
        self.decision = decision

    def for_decision(self, decision: SamplingDecision) -> HarnessTracer:
        """A tracer bound to one execution's sampling decision (spans off when not sampled)."""
        if decision.sampled and self.enabled:
            clone = HarnessTracer.__new__(HarnessTracer)
            clone.__dict__.update(self.__dict__)
            clone.decision = decision
            return clone
        clone = HarnessTracer.__new__(HarnessTracer)
        clone.__dict__.update(self.__dict__)
        clone.decision = decision
        clone.enabled = False
        # metrics stay on: a sampled-out run still counts (§28/§35).
        return clone

    # -- generic span ---------------------------------------------------------------
    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = N.KIND_INTERNAL,
        category: str = "agent",
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[TracedSpan]:
        if not self._span_enabled(category):
            yield _NOOP_TRACED
            return
        clean = self.redactor.redact_attributes(attributes or {})
        with self.provider.start_span(name, kind=kind, attributes=clean) as raw:
            yield TracedSpan(raw, category=category, capture=self.capture, redactor=self.redactor)

    def _span_enabled(self, category: str) -> bool:
        return self.enabled

    # -- named spans ----------------------------------------------------------------
    def agent_span(
        self, context: AgentExecutionContext, *, skills: list[str] | None = None, **extra: Any
    ) -> Any:
        return self.span(
            N.AGENT_RUN,
            kind=N.KIND_AGENT,
            category="agent",
            attributes={
                **context_attributes(context, self.capture),
                N.AGENT_SKILL: skills,
                **extra,
            },
        )

    def model_span(self, request: Any, **extra: Any) -> Any:
        return self.span(
            N.MODEL_INVOKE,
            kind=N.KIND_MODEL,
            category="model",
            attributes={
                N.MODEL_NAME: getattr(request, "model", None),
                N.MODEL_PROVIDER: getattr(request, "provider", None),
                N.MODEL_PROFILE: getattr(request, "profile", None),
                N.PROMPT_ID: getattr(request, "prompt_id", None),
                N.PROMPT_VERSION: getattr(request, "prompt_version", None),
                **extra,
            },
        )

    def tool_span(self, tool: str, **extra: Any) -> Any:
        return self.span(
            N.TOOL_CALL,
            kind=N.KIND_TOOL,
            category="tool",
            attributes={N.TOOL_NAME: tool, **extra},
        )

    #: Memory operation -> (span name, whether it is a retrieval).
    MEMORY_OPERATIONS: ClassVar[dict[str, tuple[str, bool]]] = {
        "retrieve": (N.MEMORY_RETRIEVE, True),
        "recall": (N.MEMORY_RECALL, True),
        "observe": (N.MEMORY_OBSERVE, False),
        "remember": (N.MEMORY_REMEMBER, False),
        "forget": (N.MEMORY_FORGET, False),
        "list": (N.MEMORY_LIST, True),
        "history": (N.MEMORY_HISTORY, True),
        "graph": (N.MEMORY_GRAPH, True),
        "ingest": (N.MEMORY_INGEST, False),
        "verify": (N.MEMORY_VERIFY, True),
    }

    def memory_span(self, operation: str, **extra: Any) -> Any:
        name, is_read = self.MEMORY_OPERATIONS.get(operation, (N.MEMORY_OBSERVE, False))
        return self.span(
            name,
            kind=N.KIND_RETRIEVAL if is_read else N.KIND_INTERNAL,
            category="memory",
            attributes={"memory.operation": operation, **extra},
        )

    def artifact_span(self, artifact_type: str, **extra: Any) -> Any:
        return self.span(
            N.ARTIFACT_CREATE,
            kind=N.KIND_INTERNAL,
            category="agent",
            attributes={N.ARTIFACT_TYPE: artifact_type, **extra},
        )

    def flush(self, timeout_seconds: float = 5.0) -> None:
        self.provider.flush(timeout_seconds)


def context_attributes(context: AgentExecutionContext, capture: CaptureConfig) -> dict[str, Any]:
    """Identity attributes for a span. ``user_id``/``thread_id`` are capture-gated (§26)."""
    attributes: dict[str, Any] = {
        N.AGENT_ID: context.agent_id,
        N.AGENT_RUN_ID: context.agent_run_id,
        N.AGENT_PARENT_RUN_ID: context.parent_agent_run_id,
        N.AGENT_GROUP_ID: context.agent_group_id,
        N.TENANT_ID: context.tenant_id,
        N.WORKSPACE_ID: context.workspace_id,
        N.TASK_ID: context.task_id,
        N.WORK_ID: context.work_id,
        N.REQUEST_ID: context.request_id,
        N.CORRELATION_ID: context.correlation_id,
        N.TURN_ID: context.turn_id,
    }
    if capture.thread_id:
        attributes[N.THREAD_ID] = context.thread_id
        attributes[N.SESSION_ID] = context.session_id
    if capture.user_id:
        attributes[N.USER_ID] = context.user_id
    return {k: v for k, v in attributes.items() if v is not None}


def _as_attribute(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)[:8000]
    except (TypeError, ValueError):
        return str(value)[:8000]


class Stopwatch:
    """Monotonic elapsed-milliseconds helper used by every instrumented call site."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def reset(self) -> None:
        self._start = time.perf_counter()
