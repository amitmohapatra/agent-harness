"""Universal Agent Harness: a runtime layer around agents you already have.

    from universal_agent_harness import AgentHarness

    harness = AgentHarness(memory=memory_client)
    wrapped = harness.wrap(existing_agent, agent_id="inventory-agent")
    result = await wrapped(payload, context=context)

The core is framework-neutral: it imports no agent framework, and framework support lives
in adapters (``universal-agent-harness-langgraph``). See ARCHITECTURE.md for the layering
and COMPATIBILITY.md for what is tested against which versions.
"""

from universal_agent_harness.artifacts import (
    ArtifactRuntime,
    FileArtifactStore,
    InMemoryArtifactStore,
)
from universal_agent_harness.config import HarnessConfig
from universal_agent_harness.contracts import (
    OBSERVATION_KINDS,
    AgentCancelledError,
    AgentDescriptor,
    AgentError,
    AgentEvalEvent,
    AgentExecutionContext,
    AgentRequest,
    AgentResult,
    AgentStatus,
    AgentTimeoutError,
    AgentWarning,
    ArtifactRef,
    Claim,
    ConfigurationError,
    ErrorCategory,
    EvidenceRef,
    HarnessError,
    LifecycleEvent,
    MemoryObservation,
    ModelError,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PolicyDeniedError,
    RecommendedAction,
    SkillDescriptor,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolSpec,
)
from universal_agent_harness.evaluation import CollectingEvaluationSink
from universal_agent_harness.execution import RetryPolicy, run_sync
from universal_agent_harness.harness import AgentHarness, __version__
from universal_agent_harness.interceptors import BaseInterceptor, Order
from universal_agent_harness.memory import MemoryPolicy
from universal_agent_harness.models import DirectModelClient
from universal_agent_harness.policy import AllowListPolicyProvider, CallablePolicyProvider
from universal_agent_harness.runtime import (
    AgentRuntime,
    CancellationToken,
    current_context,
    current_runtime,
    trace_headers,
)
from universal_agent_harness.telemetry import DefaultRedactor, HarnessTracer
from universal_agent_harness.tools import LocalToolClient, wrap_tool

__all__ = [
    "OBSERVATION_KINDS",
    "AgentCancelledError",
    "AgentDescriptor",
    "AgentError",
    "AgentEvalEvent",
    "AgentExecutionContext",
    "AgentHarness",
    "AgentRequest",
    "AgentResult",
    "AgentRuntime",
    "AgentStatus",
    "AgentTimeoutError",
    "AgentWarning",
    "AllowListPolicyProvider",
    "ArtifactRef",
    "ArtifactRuntime",
    "BaseInterceptor",
    "CallablePolicyProvider",
    "CancellationToken",
    "Claim",
    "CollectingEvaluationSink",
    "ConfigurationError",
    "DefaultRedactor",
    "DirectModelClient",
    "ErrorCategory",
    "EvidenceRef",
    "FileArtifactStore",
    "HarnessConfig",
    "HarnessError",
    "HarnessTracer",
    "InMemoryArtifactStore",
    "LifecycleEvent",
    "LocalToolClient",
    "MemoryObservation",
    "MemoryPolicy",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "Order",
    "PolicyDeniedError",
    "RecommendedAction",
    "RetryPolicy",
    "SkillDescriptor",
    "ToolCall",
    "ToolError",
    "ToolOutcome",
    "ToolSpec",
    "__version__",
    "current_context",
    "current_runtime",
    "run_sync",
    "trace_headers",
    "wrap_tool",
]
