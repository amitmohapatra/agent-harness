"""trellis-harness: a runtime layer around agents you already have.

    from trellis.harness import AgentHarness

    harness = AgentHarness(memory=memory_client)
    wrapped = harness.wrap(existing_agent, agent_id="inventory-agent")
    result = await wrapped(payload, context=context)

The core is framework-neutral: it imports no agent framework, and framework support lives
in adapters (``trellis-harness-langgraph``). See ARCHITECTURE.md for the layering
and COMPATIBILITY.md for what is tested against which versions.
"""

from typing import Any

from trellis.contracts import (
    OBSERVATION_KINDS,
    AgentCancelledError,
    AgentDescriptor,
    AgentError,
    AgentEvalEvent,
    AgentExecutionContext,
    AgentRequest,
    AgentResponse,
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

from trellis.harness.artifacts import (
    ArtifactRuntime,
    FileArtifactStore,
    InMemoryArtifactStore,
)
from trellis.harness.config import HarnessConfig
from trellis.harness.evaluation import CollectingEvaluationSink
from trellis.harness.execution import RetryPolicy, run_sync
from trellis.harness.harness import AgentHarness, __version__
from trellis.harness.interceptors import BaseInterceptor, Order
from trellis.harness.memory import MemoryPolicy
from trellis.harness.models import BifrostModelClient, DirectModelClient, tool_schemas
from trellis.harness.policy import AllowListPolicyProvider, CallablePolicyProvider
from trellis.harness.reasoning import ReActStep, ReActTrace, react
from trellis.harness.runtime import (
    AgentRuntime,
    CancellationToken,
    current_context,
    current_runtime,
    trace_headers,
)
from trellis.harness.telemetry import DefaultRedactor, HarnessTracer
from trellis.harness.tools import LocalToolClient, wrap_tool

__all__ = [
    "OBSERVATION_KINDS",
    "AgentCancelledError",
    "AgentDescriptor",
    "AgentError",
    "AgentEvalEvent",
    "AgentExecutionContext",
    "AgentHarness",
    "AgentRequest",
    "AgentResponse",
    "AgentRuntime",
    "AgentStatus",
    "AgentTimeoutError",
    "AgentWarning",
    "AllowListPolicyProvider",
    "ArtifactRef",
    "ArtifactRuntime",
    "BaseInterceptor",
    "BifrostModelClient",
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
    "ReActStep",
    "ReActTrace",
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
    "react",
    "run_sync",
    "tool_schemas",
    "trace_headers",
    "wrap_tool",
]


#: Framework adapters are separate distributions, re-exported here when installed.
#:
#: The split is a dependency decision, not an organisational one: ``langgraph`` is 38
#: packages and ~29 MB, and this package installs into somebody else's application. A
#: plain-Python user must be able to ``pip install trellis-harness`` without a
#: graph framework arriving behind it.
#:
#: What that should *not* cost is a second import line. PEP 562 lets the name live here and
#: the dependency stay optional: ``from trellis.harness import LangGraphHarness``
#: works when the extra is installed, and says how to install it when it is not.
_ADAPTERS = {"LangGraphHarness": "trellis.harness_langgraph"}


def __getattr__(name: str) -> Any:
    module = _ADAPTERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import importlib  # noqa: PLC0415 - only on the adapter path

        return getattr(importlib.import_module(module), name)
    except ImportError as exc:  # pragma: no cover - documented degradation
        raise ImportError(
            f"{name} needs the adapter: pip install 'trellis-harness[langgraph]'"
        ) from exc


def __dir__() -> list[str]:
    """Adapters are discoverable in a REPL even before one is imported."""
    return sorted([*__all__, *_ADAPTERS])
