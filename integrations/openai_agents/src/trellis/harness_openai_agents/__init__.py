"""OpenAI Agents SDK adapter for the trellis-harness.

    from trellis.harness import AgentHarness

    harness = AgentHarness(memory=memory_client, model=bifrost, defaults={"tenant_id": "acme"})

    run = harness.openai_agents.agent(agent_id="inventory", tools=[lookup])
    result = await run("how much stock of SKU-1?", context=context)

Installing this package is what makes ``harness.openai_agents`` available; the harness core
never imports the SDK. See README.md for the six bindings and for what the SDK cannot
express.
"""

from trellis.harness_openai_agents.adapter import (
    PENDING_RESULT_KEY,
    OpenAIAgentsHarness,
    openai_agents_version,
)
from trellis.harness_openai_agents.hooks import (
    FRAMEWORK,
    TrellisRunHooks,
    tool_input_guardrail,
    tool_output_guardrail,
)
from trellis.harness_openai_agents.models import (
    BifrostModel,
    BifrostModelProvider,
    to_model_request,
)
from trellis.harness_openai_agents.session import MemoryServiceSession

__version__ = "0.1.0"

__all__ = [
    "FRAMEWORK",
    "PENDING_RESULT_KEY",
    "BifrostModel",
    "BifrostModelProvider",
    "MemoryServiceSession",
    "OpenAIAgentsHarness",
    "TrellisRunHooks",
    "__version__",
    "openai_agents_version",
    "to_model_request",
    "tool_input_guardrail",
    "tool_output_guardrail",
]
