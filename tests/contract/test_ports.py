"""Every shipped implementation actually satisfies the port it claims (§27).

These are the tests that keep "core depends only on protocols" true: if an implementation
drifts from its protocol, this fails before an adapter does at runtime.
"""

from __future__ import annotations

import pytest
from trellis.contracts.ports import (
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

from trellis.harness.artifacts.stores import (
    FileArtifactStore,
    InMemoryArtifactStore,
    NoArtifactStore,
)
from trellis.harness.config.settings import LangfuseConfig
from trellis.harness.evaluation.events import (
    CollectingEvaluationSink,
    CompositeEvaluationSink,
    LoggingEvaluationSink,
    NoOpEvaluationProvider,
)
from trellis.harness.langfuse.provider import LangfuseTelemetryProvider
from trellis.harness.memory.runtime import NoOpMemoryRuntime
from trellis.harness.models.providers import DirectModelClient, UnconfiguredModelClient
from trellis.harness.policy.providers import (
    AllowListPolicyProvider,
    CallablePolicyProvider,
    NoOpPolicyProvider,
)
from trellis.harness.registry.client import InMemoryAgentRegistry, NoOpAgentRegistry
from trellis.harness.telemetry.composite import CompositeTelemetryProvider
from trellis.harness.telemetry.noop import NoOpTelemetryProvider
from trellis.harness.telemetry.otel import OpenTelemetryTelemetryProvider
from trellis.harness.telemetry.redaction import DefaultRedactor, NoOpRedactor
from trellis.harness.tools.local import CallableToolClient, LocalToolClient, NoToolsClient


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
    from trellis.harness.langfuse.evaluation import LangfuseEvaluationProvider

    assert isinstance(LangfuseEvaluationProvider(client=object()), EvaluationProvider)


def test_the_online_judge_satisfies_the_judge_port():
    """Nothing else in the harness implements ``Judge``; if this drifts, the interceptor finds
    out at the end of a real turn."""
    from trellis.contracts.ports import Judge

    from trellis.harness.evaluation.judge import GroundedJudge

    assert isinstance(GroundedJudge(), Judge)


def test_the_temporal_adapters_satisfy_the_run_and_schedule_ports():
    """A deployment chooses its durability engine by configuration, which is only true while
    both adapters answer the same two protocols the agent-runs client does."""
    from trellis.contracts.ports import RunStore, Scheduler

    temporal = pytest.importorskip(
        "trellis.harness_temporal", reason="needs trellis-harness[temporal]"
    )
    assert isinstance(temporal.TemporalRunStore("localhost:7233"), RunStore)
    assert isinstance(temporal.TemporalScheduler("localhost:7233"), Scheduler)


def test_the_phase_3_adapters_satisfy_their_ports():
    """The run store client is the contracts ``RunStore``; every sink is an ``EventSink``."""
    from trellis.contracts.ports import EventSink, RunStore

    from trellis.harness.events import (
        CollectingEventSink,
        CompositeEventSink,
        FilteringEventSink,
        WebhookEventSink,
    )
    from trellis.harness.runs import NoRunStore, RunStoreClient

    assert isinstance(NoRunStore(), RunStore)
    assert isinstance(RunStoreClient("http://runs.test", api_key="k"), RunStore)
    for sink in (
        CollectingEventSink(),
        CompositeEventSink([]),
        FilteringEventSink(CollectingEventSink(), ["RUN_FINISHED"]),
        WebhookEventSink("https://hooks.example/run", secret="s" * 32, verify_targets=False),
    ):
        assert isinstance(sink, EventSink), type(sink).__name__
