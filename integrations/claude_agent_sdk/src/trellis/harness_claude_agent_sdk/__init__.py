"""Claude Agent SDK adapter for the trellis-harness.

    from trellis.harness import AgentHarness

    harness = AgentHarness(memory=memory_client, model=bifrost, defaults={"tenant_id": "acme"})

    run = harness.claude_agent_sdk.agent(
        agent_id="reviewer", system_prompt="You review pull requests.",
        allowed_tools=["Read", "Grep"], gateway_url="http://localhost:8091",
    )
    answer = await run("what changed in src/?", context=context)

Installing this package is what makes ``harness.claude_agent_sdk`` available; the harness
core never imports the SDK. See README.md for the six bindings and for what the SDK cannot
express — this is the adapter with the most of those, because the SDK drives a CLI rather
than calling a model.
"""

from trellis.harness_claude_agent_sdk.adapter import (
    GATEWAY_URL_VAR,
    ClaudeAgentSDKHarness,
    claude_agent_sdk_version,
)
from trellis.harness_claude_agent_sdk.gateway import (
    ANTHROPIC_PREFIX,
    BASE_URL_VAR,
    TOKEN_VAR,
    anthropic_base_url,
    gateway_env,
)
from trellis.harness_claude_agent_sdk.hooks import (
    FRAMEWORK,
    STEP,
    TrellisHooks,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "ANTHROPIC_PREFIX",
    "BASE_URL_VAR",
    "FRAMEWORK",
    "GATEWAY_URL_VAR",
    "STEP",
    "TOKEN_VAR",
    "ClaudeAgentSDKHarness",
    "TrellisHooks",
    "Verdict",
    "__version__",
    "anthropic_base_url",
    "claude_agent_sdk_version",
    "gateway_env",
]
