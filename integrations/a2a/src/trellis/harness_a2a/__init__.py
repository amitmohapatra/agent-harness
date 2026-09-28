"""The A2A surface and client of trellis-harness (design §9).

Two halves, one protocol:

* **Serving** — :class:`~trellis.harness_a2a.server.A2AServer` publishes a harness agent as an A2A
  agent: a card generated from the Registry entity and the ``AgentDescriptor``, streaming from the
  run's ``RunEvent``s, ``input-required`` from a pause and a resume from the next message on the
  same task, push notifications on the harness's signed webhook rules, and a task store backed by
  the run store.
* **Calling** — :class:`~trellis.harness_a2a.client.A2AAgentClient` implements the harness
  ``ToolClient`` port, so the agents a Registry lists are tools the planner can choose, next to
  local functions and the gateway's MCP tools.

The harness core imports nothing of this package, and this package is the only place ``a2a-sdk``
is imported.
"""

from trellis.harness_a2a.card import (
    TEAM_KEY_HEADER,
    TEAM_KEY_SCHEME,
    agent_card,
    card_location,
    from_sdk_card,
    team_key_scheme,
    to_sdk_card,
)
from trellis.harness_a2a.client import A2AAgentClient, MappingCredentials, tool_name
from trellis.harness_a2a.executor import HarnessAgentExecutor
from trellis.harness_a2a.identity import (
    EXTENSION_URI,
    IDENTITY_HEADER,
    FixedIdentity,
    IdentityRefused,
    IdentityResolver,
    TrustedHeaderIdentity,
    identity_headers,
)
from trellis.harness_a2a.push import HarnessPushNotifier
from trellis.harness_a2a.server import A2AServer
from trellis.harness_a2a.tasks import HarnessTaskStore, task_from_run
from trellis.harness_a2a.translate import Update, update_for

__version__ = "0.1.0"

__all__ = [
    "EXTENSION_URI",
    "IDENTITY_HEADER",
    "TEAM_KEY_HEADER",
    "TEAM_KEY_SCHEME",
    "A2AAgentClient",
    "A2AServer",
    "FixedIdentity",
    "HarnessAgentExecutor",
    "HarnessPushNotifier",
    "HarnessTaskStore",
    "IdentityRefused",
    "IdentityResolver",
    "MappingCredentials",
    "TrustedHeaderIdentity",
    "Update",
    "__version__",
    "agent_card",
    "card_location",
    "from_sdk_card",
    "identity_headers",
    "task_from_run",
    "team_key_scheme",
    "to_sdk_card",
    "tool_name",
    "update_for",
]
