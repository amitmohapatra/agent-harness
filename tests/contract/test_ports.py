"""Every shipped implementation actually satisfies the port it claims (§27).

These are the tests that keep "core depends only on protocols" true: if an implementation
drifts from its protocol, this fails before an adapter does at runtime.
"""

from __future__ import annotations

import pytest

from universal_agent_harness.artifacts.stores import (
    FileArtifactStore,
    InMemoryArtifactStore,
    NoArtifactStore,
)
from universal_agent_harness.config.settings import LangfuseConfig
from universal_agent_harness.contracts.ports import (
    AgentPolicyProvider,
    AgentRegistryClient,
    ArtifactClient,
    EvaluationProvider,
    EvaluationSink,
    MemoryPort,
    ModelClient,
    TelemetryProvider,
    TelemetryRedactor,
    ToolClient,
)
from universal_agent_harness.evaluation.events import (
    CollectingEvaluationSink,
    CompositeEvaluationSink,
    LoggingEvaluationSink,
    NoOpEvaluationProvider,
)
from universal_agent_harness.langfuse.provider import LangfuseTelemetryProvider
from universal_agent_harness.memory.runtime import NoOpMemoryRuntime
from universal_agent_harness.models.providers import DirectModelClient, UnconfiguredModelClient
from universal_agent_harness.policy.providers import (
    AllowListPolicyProvider,
    CallablePolicyProvider,
    NoOpPolicyProvider,
)
from universal_agent_harness.registry.client import InMemoryAgentRegistry, NoOpAgentRegistry
from universal_agent_harness.telemetry.composite import CompositeTelemetryProvider
from universal_agent_harness.telemetry.noop import NoOpTelemetryProvider
from universal_agent_harness.telemetry.otel import OpenTelemetryTelemetryProvider
from universal_agent_harness.telemetry.redaction import DefaultRedactor, NoOpRedactor
from universal_agent_harness.tools.local import CallableToolClient, LocalToolClient, NoToolsClient


@pytest.mark.parametrize(
    ("protocol", "instance"),
    [
        (ModelClient, UnconfiguredModelClient()),
        (ModelClient, DirectModelClient(lambda prompt: "ok")),
        (ToolClient, LocalToolClient()),
        (ToolClient, NoToolsClient()),
        (ToolClient, CallableToolClient(lambda t, a: None)),
        (ArtifactClient, InMemoryArtifactStore()),
        (ArtifactClient, NoArtifactStore()),
        (MemoryPort, NoOpMemoryRuntime()),
        (TelemetryProvider, NoOpTelemetryProvider()),
        (TelemetryProvider, OpenTelemetryTelemetryProvider()),
        (TelemetryProvider, CompositeTelemetryProvider([])),
        (TelemetryProvider, LangfuseTelemetryProvider(LangfuseConfig())),
        (TelemetryRedactor, DefaultRedactor()),
        (TelemetryRedactor, NoOpRedactor()),
        (AgentPolicyProvider, NoOpPolicyProvider()),
        (AgentPolicyProvider, AllowListPolicyProvider()),
        (AgentPolicyProvider, CallablePolicyProvider()),
        (AgentRegistryClient, NoOpAgentRegistry()),
        (AgentRegistryClient, InMemoryAgentRegistry()),
        (EvaluationProvider, NoOpEvaluationProvider()),
        (EvaluationSink, LoggingEvaluationSink()),
        (EvaluationSink, CollectingEvaluationSink()),
        (EvaluationSink, CompositeEvaluationSink([])),
    ],
)
def test_implementation_satisfies_protocol(protocol, instance):
    assert isinstance(instance, protocol), f"{type(instance).__name__} != {protocol.__name__}"


def test_file_artifact_store_satisfies_the_port(tmp_path):
    assert isinstance(FileArtifactStore(tmp_path), ArtifactClient)


def test_memory_runtime_satisfies_the_port(harness, context):
    runtime = harness.memory_factory.create(context, tracer=harness.tracer)
    assert isinstance(runtime, MemoryPort)


def test_langfuse_evaluation_provider_satisfies_the_port():
    from universal_agent_harness.langfuse.evaluation import LangfuseEvaluationProvider

    assert isinstance(LangfuseEvaluationProvider(client=object()), EvaluationProvider)
