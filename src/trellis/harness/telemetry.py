"""OpenTelemetry with the GenAI semantic conventions, and scores on the run's trace.

Spans (every attribute passes the redactor first):

* ``invoke_agent <agent>`` — one per attempt of a run (``gen_ai.operation.name=invoke_agent``,
  ``gen_ai.agent.id``/``.name``, ``gen_ai.conversation.id`` = the thread), carrying the trace
  attributes Langfuse maps: ``langfuse.trace.name`` (the agent id), ``user.id``,
  ``session.id`` (the thread), ``langfuse.trace.tags``, ``langfuse.trace.metadata.*`` (run id,
  tenant, framework, attempt), and the run's input and output;
* ``execute_tool <tool>`` — one per tool call (``gen_ai.tool.name``, ``gen_ai.tool.call.id``,
  ``gen_ai.tool.call.arguments``/``.result``);
* ``chat <model>`` — one per model call the harness makes itself (the ``ReAct`` target:
  ``gen_ai.request.model``, ``gen_ai.usage.input_tokens``/``output_tokens``...); a framework's
  own model calls are its instrumentation's;
* ``retrieve memory`` — the pushed context;
* ``score <name>`` — a grounding score or a person's feedback, in the run's trace.

Every attempt of a run is in one trace whose id is derived from the run id
(:func:`trace_id_of`), so a resume in another process, a score computed later and
``h.feedback`` all land on the trace the run started.

Export: ``OTEL_EXPORTER_OTLP_ENDPOINT`` (+ ``OTEL_EXPORTER_OTLP_HEADERS``) installs an OTLP/HTTP
exporter — Langfuse's endpoint, or a collector (``deploy/otel-collector.yaml``) that sends
every span to Datadog and the GenAI spans to Langfuse. An application that configured OTel
itself keeps its provider.

Scores: Langfuse takes scores through its public API, not OTLP. When the OTLP headers carry
Langfuse's ``Authorization: Basic`` credentials, and the endpoint is Langfuse's
(``…/api/public/otel``) or the headers name its host (``x-langfuse-host``, which a collector
ignores), a score is also posted to ``/api/public/scores`` on the run's trace. Otherwise the
score is only the ``score`` span, which every backend receives.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Literal

import httpx
from opentelemetry import context as otel_context
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace

from trellis.harness.redaction import DEFAULT as REDACTOR
from trellis.harness.redaction import redact_attributes
from trellis.harness.settings import Settings

log = logging.getLogger("trellis.telemetry")

TRACER_NAME: Final = "trellis"
#: The OTLP/HTTP traces path, appended to ``OTEL_EXPORTER_OTLP_ENDPOINT`` (the OTel spec).
TRACES_PATH: Final = "/v1/traces"
#: Langfuse's OTLP endpoint path, and so how its host is recognised in an endpoint.
LANGFUSE_OTLP_PATH: Final = "/api/public/otel"
#: The OTLP header that names Langfuse's host when traces go through a collector.
LANGFUSE_HOST_HEADER: Final = "x-langfuse-host"
LANGFUSE_SCORES_PATH: Final = "/api/public/scores"
SCORES_TIMEOUT_SECONDS: Final = 10.0

_tracer = trace.get_tracer(TRACER_NAME)
_meter = otel_metrics.get_meter(TRACER_NAME)
_runs = _meter.create_counter("trellis.runs", description="runs finished, by outcome")
_tools = _meter.create_counter("trellis.tool_calls", description="tool calls, by status")
_writes = _meter.create_counter(
    "trellis.writes.failed", description="background writes that failed"
)


class _Metrics:
    """The counters, behind names that say what happened."""

    @staticmethod
    def run_finished(agent_id: str, outcome: str) -> None:
        _runs.add(1, {"agent": agent_id, "outcome": outcome})

    @staticmethod
    def tool_called(tool: str, status: str) -> None:
        _tools.add(1, {"tool": tool, "status": status})

    @staticmethod
    def write_failed(label: str) -> None:
        _writes.add(1, {"write": label})


metrics = _Metrics()


# --------------------------------------------------------------------------- trace ids


def trace_id_of(run_id: str) -> int:
    """The trace every attempt of ``run_id`` belongs to (128 bits of its SHA-256)."""
    return int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:16], "big") or 1


def trace_hex(run_id: str) -> str:
    """The trace id as Langfuse and every OTLP backend show it."""
    return format(trace_id_of(run_id), "032x")


def _run_context(run_id: str) -> otel_context.Context:
    """A remote parent in the run's trace, so the attempt's span joins it."""
    span_id = int.from_bytes(hashlib.sha256(f"{run_id}:root".encode()).digest()[:8], "big") or 1
    parent = trace.SpanContext(
        trace_id=trace_id_of(run_id),
        span_id=span_id,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(trace.NonRecordingSpan(parent))


# --------------------------------------------------------------------------- spans


@dataclass(frozen=True, slots=True)
class RunTrace:
    """Who a run is for, as trace attributes."""

    run_id: str
    agent_id: str
    tenant: str
    user: str
    thread: str | None
    framework: str
    attempt: int = 1

    def attributes(self) -> dict[str, Any]:
        session = self.thread or self.run_id
        return {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.id": self.agent_id,
            "gen_ai.agent.name": self.agent_id,
            "gen_ai.conversation.id": session,
            "langfuse.observation.type": "agent",
            "langfuse.trace.name": self.agent_id,
            "langfuse.trace.tags": [self.agent_id, self.framework],
            "langfuse.trace.metadata.run_id": self.run_id,
            "langfuse.trace.metadata.tenant": self.tenant,
            "langfuse.trace.metadata.framework": self.framework,
            "user.id": self.user,
            "session.id": session,
            "trellis.run_id": self.run_id,
            "trellis.tenant": self.tenant,
            "trellis.attempt": self.attempt,
        }


@contextmanager
def agent_span(run: RunTrace, task: str) -> Iterator[trace.Span]:
    """The attempt's span, in the run's trace; ``output(span, answer)`` records the answer."""
    with _tracer.start_as_current_span(
        f"invoke_agent {run.agent_id}", context=_run_context(run.run_id)
    ) as current:
        if current.is_recording():
            attributes = {**run.attributes(), "langfuse.observation.input": _text(task)}
            current.set_attributes(redact_attributes(attributes))
        yield current


@contextmanager
def tool_span(
    name: str, call_id: str, args: Any, *, source: str, tier: str
) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span(f"execute_tool {name}") as current:
        if current.is_recording():
            attributes = {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": call_id,
                "gen_ai.tool.type": "extension" if source == "mcp" else "function",
                "gen_ai.tool.call.arguments": _text(args),
                "langfuse.observation.type": "tool",
                "trellis.tool.source": source,
                "trellis.tool.tier": tier,
            }
            current.set_attributes(redact_attributes(attributes))
        yield current


@contextmanager
def model_span(model: str, messages: Any) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span(f"chat {model}", kind=trace.SpanKind.CLIENT) as current:
        if current.is_recording():
            attributes = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": model.split("/", 1)[0] if "/" in model else "bifrost",
                "gen_ai.request.model": model,
                "langfuse.observation.type": "generation",
                "langfuse.observation.input": _text(messages),
            }
            current.set_attributes(redact_attributes(attributes))
        yield current


@contextmanager
def retrieval_span(query: str) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span("retrieve memory") as current:
        if current.is_recording():
            attributes = {
                "gen_ai.operation.name": "retrieve",
                "langfuse.observation.type": "retriever",
                "langfuse.observation.input": _text(query),
            }
            current.set_attributes(redact_attributes(attributes))
        yield current


def output(span: trace.Span, value: Any, *, key: str = "langfuse.observation.output") -> None:
    """Record what a span produced (redacted, bounded)."""
    if value is not None and span.is_recording():
        span.set_attributes(redact_attributes({key: _text(value)}))


def usage(span: trace.Span, response: Mapping[str, Any]) -> None:
    """A chat-completions response's model, usage and finish reasons, as GenAI attributes."""
    if not span.is_recording():
        return
    found: dict[str, Any] = {"gen_ai.response.model": response.get("model")}
    counts = response.get("usage") or {}
    found["gen_ai.usage.input_tokens"] = counts.get("prompt_tokens")
    found["gen_ai.usage.output_tokens"] = counts.get("completion_tokens")
    reasons = [c.get("finish_reason") for c in response.get("choices") or [] if c]
    found["gen_ai.response.finish_reasons"] = [r for r in reasons if r]
    span.set_attributes(redact_attributes({k: v for k, v in found.items() if v not in (None, [])}))


def score_span(run_id: str, name: str, value: float | str, comment: str | None) -> None:
    """A score as a span in the run's trace (what every OTLP backend receives)."""
    attributes = {
        "langfuse.observation.type": "evaluator",
        "trellis.run_id": run_id,
        "trellis.score.name": name,
        "trellis.score.value": value,
        "trellis.score.comment": comment,
    }
    with _tracer.start_as_current_span(
        f"score {name}", context=_run_context(run_id), attributes=redact_attributes(attributes)
    ) as current:
        current.add_event("score", redact_attributes({"name": name, "value": value}))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(REDACTOR.redact_input(value))


# --------------------------------------------------------------------------- scores

ScoreType = Literal["NUMERIC", "CATEGORICAL"]


class Scores:
    """Langfuse's public scores API, reached with the OTLP exporter's own credentials."""

    def __init__(self, host: str, authorization: str, *, client: httpx.AsyncClient | None = None):
        self.host = host.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self.host,
            headers={"Authorization": authorization},
            timeout=SCORES_TIMEOUT_SECONDS,
        )

    @classmethod
    def of(cls, settings: Settings) -> Scores | None:
        """The scores API the OTLP settings reach, or ``None`` (scores stay spans)."""
        authorization = settings.otlp_headers.get("authorization", "")
        if not authorization.lower().startswith("basic "):
            return None
        host = settings.otlp_headers.get(LANGFUSE_HOST_HEADER)
        endpoint = settings.otlp_endpoint or ""
        if host is None and LANGFUSE_OTLP_PATH in endpoint:
            host = endpoint.split(LANGFUSE_OTLP_PATH, 1)[0]
        return cls(host, authorization) if host else None

    async def post(
        self,
        run_id: str,
        name: str,
        value: float | str,
        *,
        data_type: ScoreType,
        comment: str | None = None,
        key: str,
    ) -> None:
        """One score on the run's trace; ``key`` makes a retry update rather than add."""
        body: dict[str, Any] = {
            "id": key,
            "traceId": trace_hex(run_id),
            "name": name,
            "value": value,
            "dataType": data_type,
        }
        if comment:
            body["comment"] = comment
        response = await self._client.post(LANGFUSE_SCORES_PATH, json=body)
        response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- export


def configure(settings: Settings) -> bool:
    """Install an OTLP exporter when the deployment asked for one. Returns whether it did."""
    if not settings.otlp_endpoint:
        return False
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        log.info("an OpenTelemetry tracer provider is already installed; keeping it")
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError:
        log.warning(
            "OTLP export is configured but not installed: pip install 'trellis-harness[otel]'"
        )
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": "trellis-harness"}))
    exporter = OTLPSpanExporter(
        endpoint=traces_url(settings.otlp_endpoint), headers=settings.otlp_headers
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return True


def traces_url(endpoint: str) -> str:
    """The traces URL an OTLP/HTTP base endpoint means (a full ``…/v1/traces`` is kept)."""
    base = endpoint.rstrip("/")
    return base if base.endswith(TRACES_PATH) else base + TRACES_PATH
