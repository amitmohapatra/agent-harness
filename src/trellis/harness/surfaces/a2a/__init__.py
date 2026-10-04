"""The A2A surface: ``agent.serve_a2a(app, url)`` publishes an agent to other agents.

One argument decides where everything is: JSON-RPC is served at ``url``'s path and the card
at ``{url}/.well-known/agent-card.json`` (the well-known path at the origin when ``url`` has no
path), so the card, the routes and the address a caller was given cannot disagree. The task id
is the run id; a pause is ``input-required``; push notifications are signed (``push``).

The A2A SDK serves both routes as plain Starlette routes, which FastAPI leaves out of the app's
OpenAPI document; on a FastAPI app the surface adds a description of each (tag ``a2a``) — routes
that document, placed after the SDK's, so they never answer a request themselves.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
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
from fastapi import FastAPI, HTTPException

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
#: What the surface tells an app's OpenAPI document about itself.
TAG: Final = {
    "name": "a2a",
    "description": "A2A surface (serve_a2a): the agent card and A2A JSON-RPC (task id = run "
    "id, context id = thread; streaming answers are server-sent events).",
    "externalDocs": {"description": "A2A protocol", "url": "https://a2a-protocol.org"},
}
_JSONRPC_REQUEST: Final = {
    "type": "object",
    "required": ["jsonrpc", "method"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "integer", "null"]},
        "method": {"enum": sorted(JsonRpcDispatcher.METHOD_TO_MODEL)},
        "params": {"type": "object", "description": "the method's request, as A2A defines it"},
    },
}


def mount(app: Any, agent: Agent, *, url: str, identity: UserResolver | None = None) -> None:
    """Add the card and JSON-RPC routes for ``agent`` to a FastAPI/Starlette ``app``."""
    user_of = identity or HeaderIdentity(agent.harness)
    configs = InMemoryPushNotificationConfigStore(owner_resolver=owner(user_of))
    notifier = PushNotifier(configs)
    card = agent_card(agent, url)
    tasks = RunTaskStore(
        agent.harness.runs, agent_id=agent.id, user_of=user_of, tenant=agent.harness.tenant
    )
    handler = DefaultRequestHandler(
        agent_executor=RunExecutor(agent, user_of, tasks),
        task_store=tasks,
        agent_card=card,
        push_config_store=configs,
        push_sender=notifier,
        push_url_validator=notifier.validate_url,
    )
    path = urlsplit(url).path.rstrip("/")
    card_path = path + AGENT_CARD_WELL_KNOWN_PATH
    app.router.routes.extend(
        [
            *create_agent_card_routes(card, card_url=card_path),
            *create_jsonrpc_routes(handler, path or "/"),
        ]
    )
    if isinstance(app, FastAPI):
        _describe(app, agent, rpc_path=path or "/", card_path=card_path)


def _describe(app: FastAPI, agent: Agent, *, rpc_path: str, card_path: str) -> None:
    """The SDK's routes in the app's OpenAPI document: documenting routes after them."""
    tags = app.openapi_tags or []
    if not any(t.get("name") == TAG["name"] for t in tags):
        app.openapi_tags = [*tags, TAG]
    app.add_api_route(
        card_path,
        served_by_the_sdk,
        methods=["GET"],
        tags=["a2a"],
        summary=f"The {agent.id} agent's A2A card",
        description="Its id, description, the JSON-RPC interface at its URL, streaming, push "
        "notifications and the trusted-identity extension. No credential is ever part of it.",
        responses={
            200: {
                "description": "the AgentCard",
                "content": {"application/json": {"schema": {"type": "object"}}},
            }
        },
    )
    app.add_api_route(
        rpc_path,
        served_by_the_sdk,
        methods=["POST"],
        tags=["a2a"],
        summary=f"A2A JSON-RPC for the {agent.id} agent",
        description="One JSON-RPC 2.0 request per call. SendStreamingMessage and "
        "SubscribeToTask answer server-sent events (task status updates, the `result` "
        "artifact); the others answer one JSON-RPC response. The caller's identity is the "
        "trusted `x-trellis-identity` header (or the deployment's `identity`).",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": _JSONRPC_REQUEST}},
            }
        },
        responses={
            200: {
                "description": "a JSON-RPC response, or (streaming methods) its events",
                "content": {
                    "application/json": {"schema": {"type": "object"}},
                    "text/event-stream": {"schema": {"type": "string"}},
                },
            }
        },
    )
    app.openapi_schema = None


async def served_by_the_sdk() -> None:
    """A documenting route: the SDK's route before it answers every request it describes."""
    raise HTTPException(500, "the A2A route is served by the SDK's route ahead of this one")


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
