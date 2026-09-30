"""The A2A surface: ``agent.serve_a2a(app, url)`` publishes an agent to other agents.

One argument decides where everything is: JSON-RPC is served at ``url``'s path and the card
at ``{url}/.well-known/agent-card.json`` (the well-known path at the origin when ``url`` has no
path), so the card, the routes and the address a caller was given cannot disagree. The task id
is the run id; a pause is ``input-required``; push notifications are signed (``push``).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    PROTOCOL_VERSION_CURRENT,
    TransportProtocol,
)

from trellis.harness.surfaces.a2a.executor import RunExecutor
from trellis.harness.surfaces.a2a.identity import (
    EXTENSION_DESCRIPTION,
    EXTENSION_URI,
    HeaderIdentity,
    UserResolver,
)
from trellis.harness.surfaces.a2a.push import PushNotifier
from trellis.harness.surfaces.a2a.tasks import RunTaskStore, owner

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

MODES: Final = ["text/plain", "application/json"]
VERSION: Final = "1"


def mount(app: Any, agent: Agent, *, url: str, identity: UserResolver | None = None) -> None:
    """Add the card and JSON-RPC routes for ``agent`` to a FastAPI/Starlette ``app``."""
    user_of = identity or HeaderIdentity(agent.harness)
    configs = InMemoryPushNotificationConfigStore(owner_resolver=owner(user_of))
    notifier = PushNotifier(configs)
    card = agent_card(agent, url)
    tasks = RunTaskStore(agent.harness.runs, agent_id=agent.id, user_of=user_of)
    handler = DefaultRequestHandler(
        agent_executor=RunExecutor(agent, user_of, tasks),
        task_store=tasks,
        agent_card=card,
        push_config_store=configs,
        push_sender=notifier,
        push_url_validator=notifier.validate_url,
    )
    path = urlsplit(url).path.rstrip("/")
    app.router.routes.extend(
        [
            *create_agent_card_routes(card, card_url=path + AGENT_CARD_WELL_KNOWN_PATH),
            *create_jsonrpc_routes(handler, path or "/"),
        ]
    )


def agent_card(agent: Agent, url: str) -> AgentCard:
    """The card: the agent's id and description, one JSON-RPC interface, streaming, push, and
    the trusted-identity extension. No credential is ever part of it."""
    description = _description(agent)
    return AgentCard(
        name=agent.id,
        description=description,
        version=VERSION,
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=True,
            extensions=[AgentExtension(uri=EXTENSION_URI, description=EXTENSION_DESCRIPTION)],
        ),
        default_input_modes=MODES,
        default_output_modes=MODES,
        skills=[AgentSkill(id=agent.id, name=agent.id, description=description, tags=[agent.id])],
    )


def _description(agent: Agent) -> str:
    """What the target says about itself, where it says anything."""
    target = agent.target
    for attribute in ("description", "handoff_description"):
        value = getattr(target, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if inspect.isfunction(target) and target.__doc__:
        return inspect.cleandoc(target.__doc__).split("\n\n")[0]
    return f"the {agent.id} agent"
