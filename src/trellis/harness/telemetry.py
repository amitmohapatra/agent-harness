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
* ``score <name>`` — a grounding score, an evaluator's or a person's feedback, in the run's trace
  (or the trace a score names: a trace the team's own tracing made).

A run an evaluation runs (``evals.evaluate``) is an item of a Langfuse experiment: inside
:func:`experiment`, every span above carries Langfuse's experiment attributes
(``langfuse.experiment.*``), as its own SDK's experiment runner sets them — what Langfuse v4
builds experiments from (v3 links dataset runs through ``POST /api/public/dataset-run-items``,
which the harness also sends). A callable evaluated with no harness gets a root span of its own
(:func:`item_span`) in the trace of the run id the evaluation made for it.

Every attempt of a run is in one trace whose id is derived from the run id
(:func:`trace_id_of`), so a resume in another process, a score computed later and
``h.feedback`` all land on the trace the run started. A sub-agent's run (``Agent.as_tool``)
works inside its parent's tool call: its spans are in the parent's trace, under that call.

Export: ``OTEL_EXPORTER_OTLP_ENDPOINT`` (+ ``OTEL_EXPORTER_OTLP_HEADERS``) installs an OTLP/HTTP
exporter — Langfuse's endpoint, or a collector (``deploy/otel-collector.yaml``) that sends
every span to Datadog and the GenAI spans to Langfuse. An application that configured OTel
itself keeps its provider.

Scores and datasets: Langfuse takes scores through its public API, not OTLP. When the OTLP
headers carry Langfuse's ``Authorization: Basic`` credentials, and the endpoint is Langfuse's
(``…/api/public/otel``) or the headers name its host (``x-langfuse-host``, which a collector
ignores), a score is also posted to ``/api/public/scores`` on the run's trace, and
:class:`Langfuse` reads datasets and links runs to a dataset run (``evaluate``). Otherwise the
score is only the ``score`` span, which every backend receives.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Literal
from urllib.parse import quote

import httpx
from opentelemetry import context as otel_context
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace

from trellis.contracts import ConfigurationError
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
#: A dataset by name, its items page by page, and a run's trace linked to a dataset run
#: (Langfuse's public API: ``fern/apis/server/definition`` in its repository).
LANGFUSE_DATASET_PATH: Final = "/api/public/v2/datasets/{name}"
LANGFUSE_ITEMS_PATH: Final = "/api/public/dataset-items"
LANGFUSE_RUN_ITEMS_PATH: Final = "/api/public/dataset-run-items"
#: Dataset items one page asks for (Langfuse's default page size).
LANGFUSE_PAGE: Final = 50
SCORES_TIMEOUT_SECONDS: Final = 10.0

#: Langfuse's experiment attributes, as its Python SDK names them (``_client/attributes.py``).
EXPERIMENT_ID: Final = "langfuse.experiment.id"
EXPERIMENT_NAME: Final = "langfuse.experiment.name"
EXPERIMENT_DESCRIPTION: Final = "langfuse.experiment.description"
EXPERIMENT_METADATA: Final = "langfuse.experiment.metadata"
EXPERIMENT_DATASET_ID: Final = "langfuse.experiment.dataset.id"
EXPERIMENT_ITEM_ID: Final = "langfuse.experiment.item.id"
EXPERIMENT_ITEM_EXPECTED_OUTPUT: Final = "langfuse.experiment.item.expected_output"
EXPERIMENT_ITEM_METADATA: Final = "langfuse.experiment.item.metadata"
EXPERIMENT_ITEM_ROOT_OBSERVATION_ID: Final = "langfuse.experiment.item.root_observation_id"
LANGFUSE_ENVIRONMENT: Final = "langfuse.environment"
#: The environment the SDK puts an experiment item's spans in.
EXPERIMENT_ENVIRONMENT: Final = "sdk-experiment"
#: The longest propagated value the SDK keeps (a longer one is dropped, not cut).
PROPAGATED_MAX_CHARS: Final = 200

_tracer = trace.get_tracer(TRACER_NAME)
_meter = otel_metrics.get_meter(TRACER_NAME)
_runs = _meter.create_counter("trellis.runs", description="runs finished, by outcome")
_tools = _meter.create_counter("trellis.tool_calls", description="tool calls, by status")
_writes = _meter.create_counter(
    "trellis.writes.failed", description="background writes that failed"
)
_undelivered = _meter.create_counter(
    "trellis.writes.undelivered",
    description="background writes given up by this process, by outcome (spooled or lost)",
)
_queue_wait = _meter.create_histogram(
    "trellis.runs.queue_wait",
    unit="s",
    description="how long a queued run waited for a worker to claim it",
)
_rate_limited = _meter.create_counter(
    "trellis.runs.rate_limited",
    description="calls agent-runs refused with 429 (its rate limit) after the SDK's retries",
)
_events_lost = _meter.create_counter(
    "trellis.run_events.undelivered",
    description="run events agent-runs' event log did not take",
)
_notified = _meter.create_counter(
    "trellis.notifications", description="notifications of a pause, by provider and outcome"
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

    @staticmethod
    def write_undelivered(label: str, outcome: str) -> None:
        _undelivered.add(1, {"write": label, "outcome": outcome})

    @staticmethod
    def queue_waited(agent_id: str, seconds: float) -> None:
        _queue_wait.record(max(0.0, seconds), {"agent": agent_id})

    @staticmethod
    def rate_limited(operation: str) -> None:
        _rate_limited.add(1, {"operation": operation})

    @staticmethod
    def events_undelivered(count: int) -> None:
        _events_lost.add(count)

    @staticmethod
    def notified(provider: str, outcome: str) -> None:
        _notified.add(1, {"provider": provider, "outcome": outcome})


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
    return _parent(trace_id_of(run_id), f"{run_id}:root")


def _parent(trace_id: int, seed: str) -> otel_context.Context:
    """A remote parent in the trace ``trace_id``, its span id derived from ``seed``."""
    span_id = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big") or 1
    parent = trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(trace.NonRecordingSpan(parent))


# --------------------------------------------------------------------------- experiments


def serialized(value: Any) -> str | None:
    """A value as Langfuse's SDK writes it into an attribute: text as it is, anything else as
    JSON (``None`` stays ``None``)."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def flattened(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Metadata as the SDK propagates it: nested keys joined with dots, values serialized, none
    left out."""
    found: dict[str, str] = {}

    def walk(path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                walk(f"{path}.{key}", nested)
            return
        text = serialized(value)
        if text is not None:
            found[path] = text

    for key, value in (metadata or {}).items():
        walk(str(key), value)
    return found


@dataclass(slots=True)
class Experiment:
    """An evaluated run's place in a Langfuse experiment: the dataset run (``id``, ``name``), the
    dataset and item, and the run's root span once it started (``root_observation_id``)."""

    id: str
    name: str
    item_id: str
    dataset_id: str | None = None
    description: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    expected_output: str | None = None
    item_metadata: dict[str, str] = field(default_factory=dict)
    root_observation_id: str | None = None

    def propagated(self) -> dict[str, Any]:
        """What every span of the run carries (the SDK's propagated experiment attributes; a
        value over :data:`PROPAGATED_MAX_CHARS` is left out, as the SDK leaves it out)."""
        values: dict[str, str | None] = {
            EXPERIMENT_ID: self.id,
            EXPERIMENT_NAME: self.name,
            EXPERIMENT_DATASET_ID: self.dataset_id,
            EXPERIMENT_ITEM_ID: self.item_id,
            EXPERIMENT_ITEM_ROOT_OBSERVATION_ID: self.root_observation_id,
        }
        values.update({f"{EXPERIMENT_METADATA}.{k}": v for k, v in self.metadata.items()})
        values.update({f"{EXPERIMENT_ITEM_METADATA}.{k}": v for k, v in self.item_metadata.items()})
        if self.root_observation_id is not None:
            values[LANGFUSE_ENVIRONMENT] = EXPERIMENT_ENVIRONMENT
        return {k: v for k, v in values.items() if v is not None and len(v) <= PROPAGATED_MAX_CHARS}

    def root(self) -> dict[str, Any]:
        """What the run's root span carries: the propagated attributes, the description and the
        expected output."""
        values = self.propagated()
        if self.description is not None:
            values[EXPERIMENT_DESCRIPTION] = self.description
        if self.expected_output is not None:
            values[EXPERIMENT_ITEM_EXPECTED_OUTPUT] = self.expected_output
        return values


_experiment: ContextVar[Experiment | None] = ContextVar("trellis_experiment", default=None)


@contextmanager
def experiment(item: Experiment) -> Iterator[Experiment]:
    """Run the code inside as ``item`` of a Langfuse experiment: the spans it starts carry the
    experiment's attributes (the first ``invoke_agent`` span is the item's root observation)."""
    token = _experiment.set(item)
    try:
        yield item
    finally:
        _experiment.reset(token)


def _experimented(span: trace.Span, *, root: bool = False) -> None:
    """Put the current experiment's attributes on ``span`` (a recording one)."""
    current = _experiment.get()
    if current is None:
        return
    if root and current.root_observation_id is None:
        current.root_observation_id = format(span.get_span_context().span_id, "016x")
    span.set_attributes(redact_attributes(current.root() if root else current.propagated()))


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
    #: the agent's version (``h.wrap(..., version=)``), when it has one
    version: str | None = None
    #: the run in whose tool call this one works (a sub-agent's run): its spans are in that
    #: run's trace, under the call, and the trace's own attributes stay the parent's
    parent: str | None = None

    def attributes(self) -> dict[str, Any]:
        session = self.thread or self.run_id
        versioned = (
            {}
            if self.version is None
            else {
                "gen_ai.agent.version": self.version,
                "langfuse.version": self.version,
            }
        )
        traced = (
            {"trellis.parent_run_id": self.parent}
            if self.parent is not None
            else {
                "langfuse.trace.name": self.agent_id,
                "langfuse.trace.tags": [self.agent_id, self.framework],
                "langfuse.trace.metadata.run_id": self.run_id,
                "langfuse.trace.metadata.tenant": self.tenant,
                "langfuse.trace.metadata.framework": self.framework,
            }
        )
        return {
            **versioned,
            **traced,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.id": self.agent_id,
            "gen_ai.agent.name": self.agent_id,
            "gen_ai.conversation.id": session,
            "langfuse.observation.type": "agent",
            "user.id": self.user,
            "session.id": session,
            "trellis.run_id": self.run_id,
            "trellis.tenant": self.tenant,
            "trellis.attempt": self.attempt,
        }


@contextmanager
def agent_span(run: RunTrace, task: str) -> Iterator[trace.Span]:
    """The attempt's span, in the run's trace — a sub-agent's run's under its parent's tool
    call; ``output(span, answer)`` records the answer."""
    attributes = {**run.attributes(), "langfuse.observation.input": _text(task)}
    if run.parent is None:
        with _root_span(run.run_id, run.agent_id, attributes) as current:
            yield current
        return
    with _tracer.start_as_current_span(f"invoke_agent {run.agent_id}") as current:
        if current.is_recording():
            current.set_attributes(redact_attributes(attributes))
            _experimented(current)
        yield current


@contextmanager
def item_span(run_id: str, name: str, input: Any, *, user: str) -> Iterator[trace.Span]:
    """The span of a callable an evaluation runs on one item (``evaluate(callable, ...)``), in
    the trace of the ``run_id`` the evaluation gave it: what :func:`agent_span` is to a wrapped
    agent's run. The callable's own spans, if it makes any, are its children."""
    attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": name,
        "langfuse.observation.type": "agent",
        "langfuse.trace.name": name,
        "langfuse.observation.input": _text(input),
        "user.id": user,
        "trellis.run_id": run_id,
    }
    with _root_span(run_id, name, attributes) as current:
        yield current


@contextmanager
def _root_span(run_id: str, name: str, attributes: dict[str, Any]) -> Iterator[trace.Span]:
    with _tracer.start_as_current_span(
        f"invoke_agent {name}", context=_run_context(run_id)
    ) as current:
        if current.is_recording():
            current.set_attributes(redact_attributes(attributes))
            _experimented(current, root=True)
        yield current


@contextmanager
def tool_span(
    name: str, call_id: str, args: Any, *, source: str, action: str
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
                "trellis.governance.action": action,
            }
            current.set_attributes(redact_attributes(attributes))
            _experimented(current)
        yield current


@contextmanager
def model_span(
    model: str, messages: Any, *, extra: Mapping[str, Any] | None = None
) -> Iterator[trace.Span]:
    """A model call's ``chat`` span; ``extra`` attributes say more about the request (the
    stored prompt it selects)."""
    with _tracer.start_as_current_span(f"chat {model}", kind=trace.SpanKind.CLIENT) as current:
        if current.is_recording():
            attributes = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": model.split("/", 1)[0] if "/" in model else "bifrost",
                "gen_ai.request.model": model,
                "langfuse.observation.type": "generation",
                "langfuse.observation.input": _text(messages),
                **(extra or {}),
            }
            current.set_attributes(redact_attributes(attributes))
            _experimented(current)
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
            _experimented(current)
        yield current


def decided(span: trace.Span, decision: Mapping[str, Any]) -> None:
    """A person's decision the attempt goes on with (its kind, reviewer, comment and reach),
    as the attempt span's ``decision`` event and ``trellis.decision.*`` attributes."""
    if span.is_recording():
        found = {f"trellis.decision.{k}": v for k, v in decision.items() if v is not None}
        attributes = redact_attributes(found)
        span.set_attributes(attributes)
        span.add_event("decision", attributes)


def attribute(name: str, value: str) -> None:
    """An attribute of the span current now (the run's span, in its pipeline)."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(name, value)


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


def score_span(
    trace_id: str, name: str, value: float | str, comment: str | None, *, run_id: str | None = None
) -> None:
    """A score as a span in the trace ``trace_id`` (32 hex characters: a run's is
    :func:`trace_hex`) — what every OTLP backend receives."""
    attributes = {
        "langfuse.observation.type": "evaluator",
        "trellis.run_id": run_id,
        "trellis.score.name": name,
        "trellis.score.value": value,
        "trellis.score.comment": comment,
    }
    parent = _parent(int(trace_id, 16), f"{run_id or trace_id}:root")
    with _tracer.start_as_current_span(
        f"score {name}", context=parent, attributes=redact_attributes(attributes)
    ) as current:
        current.add_event("score", redact_attributes({"name": name, "value": value}))
        if current.is_recording():
            _experimented(current)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(REDACTOR.redact_input(value))


# --------------------------------------------------------------------------- scores

ScoreType = Literal["NUMERIC", "BOOLEAN", "CATEGORICAL"]


class Langfuse:
    """Langfuse's public API as far as the harness uses it — scores on a run's trace, datasets
    and dataset runs — reached with the OTLP exporter's own credentials."""

    def __init__(self, host: str, authorization: str, *, client: httpx.AsyncClient | None = None):
        self.host = host.rstrip("/")
        self._client = client or httpx.AsyncClient(
            base_url=self.host,
            headers={"Authorization": authorization},
            timeout=SCORES_TIMEOUT_SECONDS,
        )

    @classmethod
    def of(cls, settings: Settings) -> Langfuse | None:
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
        trace_id: str,
        name: str,
        value: float | str,
        *,
        data_type: ScoreType,
        comment: str | None = None,
        key: str,
    ) -> None:
        """One score on the trace ``trace_id`` (32 hex characters: a run's is
        :func:`trace_hex`); ``key`` makes a retry update rather than add."""
        body: dict[str, Any] = {
            "id": key,
            "traceId": trace_id,
            "name": name,
            "value": value,
            "dataType": data_type,
        }
        if comment:
            body["comment"] = comment
        response = await self._client.post(LANGFUSE_SCORES_PATH, json=body)
        response.raise_for_status()

    async def dataset(self, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The dataset ``name`` (its ``id``, ``projectId``...) and its active items, in
        Langfuse's order, every page of them. A dataset Langfuse does not have is a
        ``ConfigurationError``."""
        found = await self._client.get(LANGFUSE_DATASET_PATH.format(name=quote(name, safe="")))
        if found.status_code == httpx.codes.NOT_FOUND:
            raise ConfigurationError(f"Langfuse has no dataset {name!r}")
        found.raise_for_status()
        items: list[dict[str, Any]] = []
        page, pages = 1, 1
        while page <= pages:
            response = await self._client.get(
                LANGFUSE_ITEMS_PATH,
                params={"datasetName": name, "page": page, "limit": LANGFUSE_PAGE},
            )
            response.raise_for_status()
            body = response.json()
            items.extend(i for i in body["data"] if i.get("status", "ACTIVE") == "ACTIVE")
            pages = int((body.get("meta") or {}).get("totalPages") or 0)
            page += 1
        return found.json(), items

    async def link(
        self,
        run_id: str,
        *,
        run_name: str,
        item_id: str,
        metadata: dict[str, Any],
        description: str | None = None,
    ) -> str | None:
        """Link the run's trace to the dataset run ``run_name`` as the result of the dataset
        item ``item_id`` (the dataset run is created by its first item); the dataset run's id.
        Langfuse v3 (and a self-hosted v3) takes it; v4 builds the run from the spans'
        experiment attributes instead (:class:`Experiment`)."""
        body: dict[str, Any] = {
            "runName": run_name,
            "datasetItemId": item_id,
            "traceId": trace_hex(run_id),
            "metadata": metadata,
        }
        if description is not None:
            body["runDescription"] = description
        response = await self._client.post(LANGFUSE_RUN_ITEMS_PATH, json=body)
        response.raise_for_status()
        run = response.json().get("datasetRunId")
        return run if isinstance(run, str) and run else None

    def dataset_run_url(self, project_id: str, dataset_id: str, run_id: str) -> str:
        """Where Langfuse shows a dataset run (an experiment)."""
        return f"{self.host}/project/{project_id}/datasets/{dataset_id}/runs/{run_id}"

    def trace_url(self, run_id: str) -> str:
        """Where Langfuse shows the run's trace (its ``/trace/{id}`` page finds the project)."""
        return f"{self.host}/trace/{trace_hex(run_id)}"

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- export


#: The OTLP/HTTP metrics path, appended to ``OTEL_EXPORTER_OTLP_ENDPOINT``.
METRICS_PATH: Final = "/v1/metrics"
#: How often the counters and histograms are exported.
METRICS_EXPORT_MILLIS: Final = 60_000


def configure(settings: Settings) -> bool:
    """Install an OTLP exporter when the deployment asked for one (and one for the metrics,
    unless the endpoint is Langfuse's, which takes traces only). Returns whether the traces'
    exporter was installed."""
    if not settings.otlp_endpoint:
        return False
    _measured(settings.otlp_endpoint, settings.otlp_headers)
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


def _measured(endpoint: str, headers: Mapping[str, str]) -> bool:
    """Export the counters and histograms over OTLP/HTTP to ``<endpoint>/v1/metrics``, every
    :data:`METRICS_EXPORT_MILLIS` — unless the endpoint is Langfuse's (no metrics there), the
    application installed a meter provider itself (kept), or the extra is not installed."""
    if LANGFUSE_OTLP_PATH in endpoint:
        return False
    # the API's default stands in until a provider is set (its class is private to it)
    if type(otel_metrics.get_meter_provider()).__name__ != "_ProxyMeterProvider":
        log.info("an OpenTelemetry meter provider is already installed; keeping it")
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: PLC0415
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
        from opentelemetry.sdk.metrics.export import (  # noqa: PLC0415
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    except ImportError:
        return False  # configure() says how to install the extra
    exporter = OTLPMetricExporter(endpoint=metrics_url(endpoint), headers=dict(headers))
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=METRICS_EXPORT_MILLIS)
    resource = Resource.create({"service.name": "trellis-harness"})
    otel_metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
    return True


#: How long closing the harness waits for the spans still queued to be exported.
FLUSH_TIMEOUT_MS: Final = 5_000


async def flush() -> None:
    """Export the spans still queued (the batch processor sends every few seconds): closing
    the harness - a worker's shutdown, a script's end - must not leave the last runs' traces
    behind. The export blocks, so it runs off the event loop."""
    for provider in (trace.get_tracer_provider(), otel_metrics.get_meter_provider()):
        force_flush = getattr(provider, "force_flush", None)
        if force_flush is not None:
            await asyncio.to_thread(force_flush, FLUSH_TIMEOUT_MS)


def traces_url(endpoint: str) -> str:
    """The traces URL an OTLP/HTTP base endpoint means (a full ``…/v1/traces`` is kept)."""
    base = endpoint.rstrip("/")
    return base if base.endswith(TRACES_PATH) else base + TRACES_PATH


def metrics_url(endpoint: str) -> str:
    """The metrics URL an OTLP/HTTP base endpoint means (a traces URL's base, too)."""
    base = endpoint.rstrip("/").removesuffix(TRACES_PATH)
    return base if base.endswith(METRICS_PATH) else base + METRICS_PATH
