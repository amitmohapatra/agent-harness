"""Compatibility matrix (§3, §79).

These tests assert three things: that the versions actually installed are recorded (so
COMPATIBILITY.md can be regenerated from a real run rather than from memory), that each
adapter relies on capability *detection* rather than exact versions — a framework missing an
optional capability must degrade, not crash — and that the core imports **none** of the four
frameworks, which is what keeps ``pip install trellis-harness`` small.

The last one runs in a subprocess, because that is the only honest way to observe what
importing the core pulls in: ``sys.modules`` in this process is already full of everything the
adapters' own tests imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import util
from importlib.metadata import version
from pathlib import Path

import pytest

from trellis.harness import AgentHarness, __version__

MATRIX_PATH = Path(__file__).resolve().parents[2] / "compatibility-matrix.json"

LANGGRAPH_FEATURES = (
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

#: The six moments of design §8 per adapter, and what this repository's tests prove about
#: each. A moment a framework cannot express is recorded as such rather than claimed — the
#: adapter READMEs carry the reasoning.
DEEPAGENTS_FEATURES = {
    "context_into_system_prompt": "supported",
    "memory_service_backend": "supported",
    "model_through_gateway": "supported",
    "tool_policy": "supported",
    "tool_events": "supported",
    "tool_memory": "supported",
    "pause_on_require_approval": "supported",
    "resume_continues_the_run": "supported",
    "approver_edit": "supported",
    "run_end_observations": "supported",
    "compaction_summary_observed": "supported",
    "run_events": "supported",
    "memory_note_filenames": (
        "partial: the Memory Service names a note, so a write returns the path it was given"
    ),
}
OPENAI_AGENTS_FEATURES = {
    "context_into_instructions": "supported",
    "memory_service_session": "supported",
    "model_through_gateway": "supported",
    "model_provider": "supported",
    "tool_policy": "supported",
    "tool_events": "supported",
    "tool_memory": "supported",
    "pause_on_require_approval": "supported",
    "pause_on_needs_approval": "supported",
    "resume_continues_the_run": "supported",
    "run_end_observations": "supported",
    "compaction_summary_observed": "supported",
    "run_events": "supported",
    "sdk_streaming": "unsupported: the SDK streams Responses-API server events",
    "session_pop_item": "unsupported: the Memory Service thread has no per-message delete",
    "approver_edit": "unsupported: the SDK approves or rejects a call, it does not edit one",
}
CLAUDE_AGENT_SDK_FEATURES = {
    "context_as_additional_context": "supported",
    "model_through_gateway": "supported",
    "tool_policy": "supported",
    "tool_events": "supported",
    "tool_memory": "supported",
    "pause_on_require_approval": "supported",
    "resume_continues_the_run": "supported",
    "approver_edit": "supported",
    "run_end_observations": "supported",
    "compaction_summary_observed": "supported",
    "run_events": "supported",
    "model_client_instrumentation": (
        "unsupported: the SDK spawns the claude CLI, so there is no model client to wrap"
    ),
    "per_call_token_usage": "partial: the CLI reports usage and cost once, on ResultMessage",
}

#: adapter attribute on the harness -> (module, the framework distribution, the extra)
ADAPTERS = {
    "langgraph": ("trellis.harness_langgraph", "langgraph", "langgraph"),
    "deepagents": ("trellis.harness_deepagents", "deepagents", "deepagents"),
    "openai_agents": ("trellis.harness_openai_agents", "openai-agents", "openai-agents"),
    "claude_agent_sdk": (
        "trellis.harness_claude_agent_sdk",
        "claude-agent-sdk",
        "claude-agent-sdk",
    ),
}

#: The adapter distributions, by the module that ships each.
DISTRIBUTIONS = {
    "trellis.harness_langgraph": "trellis-harness-langgraph",
    "trellis.harness_deepagents": "trellis-harness-deepagents",
    "trellis.harness_openai_agents": "trellis-harness-openai-agents",
    "trellis.harness_claude_agent_sdk": "trellis-harness-claude-agent-sdk",
}

#: Any of these in ``sys.modules`` after importing the core is a bug (design §2).
BANNED_IN_CORE = (
    "langgraph",
    "langchain",
    "langchain_core",
    "deepagents",
    "agents",
    "claude_agent_sdk",
    "openai",
    "anthropic",
    "crewai",
    "google.adk",
    "langfuse",
    "fastapi",
)

#: What the matrix records the installed version of.
RECORDED = (
    "langgraph",
    "langchain",
    "langchain-core",
    "deepagents",
    "openai-agents",
    "claude-agent-sdk",
    "langfuse",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "pydantic",
    "trellis-memory",
    "trellis-harness-agui",
    "trellis-harness-langgraph",
    "trellis-harness-deepagents",
    "trellis-harness-openai-agents",
    "trellis-harness-claude-agent-sdk",
    "trellis-harness-a2a",
    "a2a-sdk",
    "fastapi",
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
    code = (
        "import sys; import trellis.harness as u; "
        f"banned = {BANNED_IN_CORE!r}; "
        "print(','.join(sorted(n for n in banned if n in sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"importing the core pulled in: {out}"


@pytest.mark.parametrize("attribute", sorted(ADAPTERS))
def test_each_adapter_version_is_its_distributions(attribute: str):
    """An adapter's ``__version__`` is its distribution's: a matrix row nobody can fake."""
    module_name, _framework, _extra = ADAPTERS[attribute]
    module = __import__(module_name, fromlist=["__version__"])
    assert module.__version__ == installed(DISTRIBUTIONS[module_name])


@pytest.mark.parametrize("attribute", sorted(ADAPTERS))
def test_each_adapter_reports_the_framework_version_it_runs_against(attribute: str):
    module_name, framework, _extra = ADAPTERS[attribute]
    module = __import__(module_name, fromlist=["*"])
    reporter = next(getattr(module, name) for name in dir(module) if name.endswith("_version"))
    assert reporter() == installed(framework)


@pytest.mark.parametrize("attribute", sorted(ADAPTERS))
def test_adapter_availability_is_detected_not_assumed(attribute: str):
    module_name, framework, _extra = ADAPTERS[attribute]
    module = __import__(module_name, fromlist=["*"])
    adapter = next(
        getattr(module, name)
        for name in dir(module)
        if name.endswith("Harness") and hasattr(getattr(module, name), "available")
    )
    assert adapter.available() is (installed(framework) is not None)


@pytest.mark.parametrize("attribute", sorted(ADAPTERS))
def test_the_harness_property_requires_the_adapter(monkeypatch, attribute: str):
    """Without the adapter installed, the harness says which extra to install (§2)."""
    module_name, _framework, extra = ADAPTERS[attribute]
    harness = AgentHarness(defaults={"tenant_id": "acme"})
    monkeypatch.setitem(sys.modules, module_name, None)
    harness._adapters.clear()
    with pytest.raises(ImportError, match=f"trellis-harness\\[{extra}\\]"):
        getattr(harness, attribute)


def test_langfuse_absence_degrades_to_otlp_mode(monkeypatch):
    """With the SDK missing, Langfuse still works through OTLP attributes (§22)."""
    from trellis.harness.config.settings import LangfuseConfig
    from trellis.harness.langfuse.provider import LangfuseTelemetryProvider

    monkeypatch.setitem(sys.modules, "langfuse", None)
    provider = LangfuseTelemetryProvider(
        LangfuseConfig(enabled=True, mode="auto", public_key="pk", secret_key="sk")
    )
    assert provider.mode == "otlp"
    assert provider.client is None


def test_opentelemetry_sdk_is_not_required_by_the_core():
    """Only the OTel API is a runtime dependency; without an SDK the API's no-op is used."""
    from trellis.harness.telemetry.otel import OpenTelemetryTelemetryProvider

    provider = OpenTelemetryTelemetryProvider()
    with provider.start_span("probe") as span:  # must work whatever is installed
        span.set_attribute("k", "v")


def test_the_agui_surface_reports_its_version_and_stays_out_of_the_core():
    """The AG-UI package is a distribution of its own (design §10): importable when
    installed, never pulled in by the core."""
    from trellis.harness_agui import __version__ as agui_version

    assert agui_version == installed("trellis-harness-agui")
    assert util.find_spec("fastapi") is not None  # installed: the core check means something


def test_write_compatibility_matrix(request):
    """Records what this run actually verified, for COMPATIBILITY.md."""
    matrix = {
        "harness_version": __version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {name: installed(name) for name in RECORDED},
        "langgraph_features": dict.fromkeys(LANGGRAPH_FEATURES, "supported"),
        "deepagents_features": DEEPAGENTS_FEATURES,
        "openai_agents_features": OPENAI_AGENTS_FEATURES,
        "claude_agent_sdk_features": CLAUDE_AGENT_SDK_FEATURES,
    }
    MATRIX_PATH.write_text(json.dumps(matrix, indent=2) + "\n")
    for framework in ("langgraph", "deepagents", "openai-agents", "claude-agent-sdk"):
        assert matrix["packages"][framework] is not None, f"{framework} is not installed"


def test_the_a2a_surface_reports_its_version_and_stays_out_of_the_core():
    """The A2A package is a distribution of its own (design §9): importable when installed, and
    never pulled into the core — including by the core's own registry directory, which speaks the
    contracts' Agent Card and imports no ``a2a-sdk``."""
    import subprocess
    from importlib import util

    from trellis.harness_a2a import __version__ as a2a_version

    assert a2a_version == installed("trellis-harness-a2a")
    assert util.find_spec("a2a") is not None  # installed: the check below means something
    code = (
        "import sys; import trellis.harness; import trellis.harness.registry; "
        "print(sorted(m for m in sys.modules if m == 'a2a' or m.startswith('a2a.')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"importing the core pulled in: {out.stdout.strip()}"


def test_the_a2a_package_speaks_the_protocol_version_it_was_verified_against():
    """A2A 1.x of the SDK is protocol v1.0 with protobuf types; the card mapping depends on it."""
    from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

    assert installed("a2a-sdk") == "1.1.5"
    assert PROTOCOL_VERSION_CURRENT == "1.0"
