"""Deep Agents adapter for the trellis-harness.

    from trellis.harness import AgentHarness

    harness = AgentHarness(memory=memory_client, model=bifrost, defaults={"tenant_id": "acme"})

    run = harness.deepagents.agent(agent_id="researcher", tools=[search])
    answer = await run({"messages": [{"role": "user", "content": "how much stock?"}]},
                       context=context)

Installing this package is what makes ``harness.deepagents`` available; the harness core
never imports Deep Agents. See README.md for the six bindings and for what Deep Agents
cannot express.
"""

from trellis.harness_deepagents.adapter import DeepAgentsHarness, deepagents_version
from trellis.harness_deepagents.backend import MEMORIES_ROOT, MemoryServiceBackend
from trellis.harness_deepagents.binding import active_runtime
from trellis.harness_deepagents.middleware import FRAMEWORK, TrellisMiddleware
from trellis.harness_deepagents.models import BifrostChatModel, to_model_request

__version__ = "0.1.0"

__all__ = [
    "FRAMEWORK",
    "MEMORIES_ROOT",
    "BifrostChatModel",
    "DeepAgentsHarness",
    "MemoryServiceBackend",
    "TrellisMiddleware",
    "__version__",
    "active_runtime",
    "deepagents_version",
    "to_model_request",
]
