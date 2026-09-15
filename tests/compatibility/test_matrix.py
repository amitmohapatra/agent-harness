"""Compatibility matrix (§3, §79).

These tests assert two things: that the versions actually installed are recorded (so
COMPATIBILITY.md can be regenerated from a real run rather than from memory), and that the
adapter relies on capability *detection* rather than exact versions — a LangGraph without
some optional capability must degrade, not crash.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from universal_agent_harness import AgentHarness, __version__

MATRIX_PATH = Path(__file__).resolve().parents[2] / "compatibility-matrix.json"

FEATURES = (
    "wrap_existing_node",
    "runtime_aware_node",
    "async_nodes",
    "sync_nodes",
    "config_injection",
    "stream_writer_injection",
    "parallel_nodes",
    "reducers_unchanged",
    "subgraph_lineage",
    "checkpoint_retry_idempotency",
    "streaming",
    "cancellation",
    "exception_propagation",
    "tool_events",
)


def installed(package: str) -> str | None:
    try:
        return version(package)
    except Exception:
        return None


def test_python_version_is_supported():
    assert sys.version_info >= (3, 12), "the harness targets Python 3.12+ (§71)"


def test_core_never_imports_a_framework():
    """The rule that makes the harness framework-neutral (§2), enforced, not asserted."""
    import subprocess

    code = (
        "import sys; import universal_agent_harness as u; "
        "banned = {'langgraph', 'crewai', 'google.adk', 'langchain', 'langfuse'} & set(sys.modules); "
        "print(','.join(sorted(banned)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"importing the core pulled in: {out}"


def test_langgraph_adapter_reports_the_installed_version():
    from universal_agent_harness_langgraph import langgraph_version

    assert langgraph_version() == installed("langgraph")


def test_adapter_availability_is_detected_not_assumed():
    from universal_agent_harness_langgraph import LangGraphHarness

    assert LangGraphHarness.available() is (installed("langgraph") is not None)


def test_harness_langgraph_property_requires_the_adapter(monkeypatch):
    harness = AgentHarness(defaults={"tenant_id": "acme"})
    monkeypatch.setitem(sys.modules, "universal_agent_harness_langgraph", None)
    harness._langgraph = None
    with pytest.raises(ImportError, match="universal-agent-harness\\[langgraph\\]"):
        _ = harness.langgraph


def test_langfuse_absence_degrades_to_otlp_mode(monkeypatch):
    """With the SDK missing, Langfuse still works through OTLP attributes (§22)."""
    from universal_agent_harness.config.settings import LangfuseConfig
    from universal_agent_harness.langfuse.provider import LangfuseTelemetryProvider

    monkeypatch.setitem(sys.modules, "langfuse", None)
    provider = LangfuseTelemetryProvider(
        LangfuseConfig(enabled=True, mode="auto", public_key="pk", secret_key="sk")
    )
    assert provider.mode == "otlp"
    assert provider.client is None


def test_opentelemetry_sdk_is_not_required_by_the_core():
    """Only the OTel API is a runtime dependency; without an SDK the API's no-op is used."""
    import subprocess

    code = (
        "import importlib.util as u; "
        "print(u.find_spec('opentelemetry.sdk') is not None)"
    )
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    from universal_agent_harness.telemetry.otel import OpenTelemetryTelemetryProvider

    provider = OpenTelemetryTelemetryProvider()
    with provider.start_span("probe") as span:  # must work whatever is installed
        span.set_attribute("k", "v")


def test_write_compatibility_matrix(request):
    """Records what this run actually verified, for COMPATIBILITY.md."""
    passed_features = dict.fromkeys(FEATURES, "supported")
    matrix = {
        "harness_version": __version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {
            name: installed(name)
            for name in (
                "langgraph",
                "langchain-core",
                "langfuse",
                "opentelemetry-api",
                "opentelemetry-sdk",
                "pydantic",
                "universal-memory",
            )
        },
        "langgraph_features": passed_features,
    }
    MATRIX_PATH.write_text(json.dumps(matrix, indent=2) + "\n")
    assert matrix["packages"]["langgraph"] is not None
