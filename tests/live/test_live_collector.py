"""``deploy/otel-collector.yaml`` as shipped, in the real OpenTelemetry Collector (contrib): the
harness exports to it, and only the GenAI / agent spans reach the Langfuse exporter — with the
caller's own ``Authorization`` — while every span goes to the Datadog pipeline. Langfuse is
played by a local OTLP receiver; the Datadog exporter gets a dummy key (its sends fail, which
the collector logs; what it would send is what the ``traces/all`` pipeline receives).

Needs docker (``otel/opentelemetry-collector-contrib``); skipped without it."""

from __future__ import annotations

import gzip
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from tests.live.support import eventually
from trellis.harness import telemetry
from trellis.harness.telemetry import RunTrace

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "otel/opentelemetry-collector-contrib:latest"
AUTH = "Basic cGstbGY6c2stbGY="


def _docker() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


needs_docker = pytest.mark.skipif(not _docker(), reason="needs docker")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Langfuse(BaseHTTPRequestHandler):
    """An OTLP/HTTP traces receiver standing in for Langfuse's endpoint."""

    received: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["content-length"]))
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    attributes = {a.key: a.value for a in span.attributes}
                    _Langfuse.received.append(
                        (span.name, {"auth": self.headers["authorization"], **attributes})
                    )
        self.send_response(200)
        self.send_header("content-type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def collector() -> Iterator[str]:
    receiver = ThreadingHTTPServer(("0.0.0.0", _free_port()), _Langfuse)
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    otlp, health = _free_port(), _free_port()
    name = f"trellis-collector-{uuid.uuid4().hex[:6]}"
    subprocess.run(
        [
            "docker", "run", "-d", "--rm", "--name", name,
            "-p", f"{otlp}:4318", "-p", f"{health}:13133",
            "--add-host", "host.docker.internal:host-gateway",
            "-e", f"LANGFUSE_OTLP_ENDPOINT=http://host.docker.internal:{receiver.server_port}",
            "-e", "DD_API_KEY=dummy", "-e", "DD_SITE=datadoghq.com",
            "-v", f"{ROOT / 'deploy' / 'otel-collector.yaml'}:/etc/otelcol-contrib/config.yaml",
            IMAGE,
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{health}/").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        yield f"http://127.0.0.1:{otlp}"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        receiver.shutdown()


@needs_docker
async def test_the_collector_sends_only_genai_spans_to_langfuse_with_the_callers_auth(
    collector: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": "trellis-live"}))
    exporter = OTLPSpanExporter(
        endpoint=telemetry.traces_url(collector), headers={"authorization": AUTH}
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("live")
    monkeypatch.setattr(telemetry, "_tracer", tracer)
    run_id = f"run_{uuid.uuid4().hex}"
    run = RunTrace(
        run_id=run_id, agent_id="live-agent", tenant="acme", user="u", thread="th",
        framework="react",
    )  # fmt: skip
    with telemetry.agent_span(run, "what is in stock?"):
        with telemetry.tool_span("stock", "c1", {"sku": "a"}, source="local", tier="auto"):
            pass
        with tracer.start_as_current_span("GET /inventory"):  # an HTTP span: Datadog's only
            pass
    telemetry.score_span(run_id, "grounding", 0.9, None)
    provider.force_flush()

    async def arrived() -> bool:
        return {"invoke_agent live-agent", "execute_tool stock", "score grounding"} <= {
            name for name, _ in _Langfuse.received
        }

    assert await eventually(arrived, within=30, every=0.5)
    names = {name for name, _ in _Langfuse.received}
    assert "GET /inventory" not in names  # filtered out of the Langfuse pipeline
    agent = next(a for n, a in _Langfuse.received if n == "invoke_agent live-agent")
    assert agent["auth"] == AUTH  # the caller's Langfuse credentials, forwarded
    assert agent["langfuse.trace.name"].string_value == "live-agent"
    assert agent["session.id"].string_value == "th"
    provider.shutdown()
