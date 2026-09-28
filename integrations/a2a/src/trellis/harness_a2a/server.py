"""The A2A server: one harness agent, its card, and the routes that serve both (design §9).

```mermaid
flowchart LR
  subgraph Served["A2AServer"]
    CARD["GET /.well-known/agent-card.json<br/>card from Registry entity + AgentDescriptor"]
    RPC["POST /<br/>SendMessage · SendStreamingMessage · GetTask · CancelTask · push configs"]
  end
  RPC --> EX["HarnessAgentExecutor"] --> H["AgentHarness"]
  RPC --> TS["HarnessTaskStore → RunStore"]
  RPC --> PN["HarnessPushNotifier (signed, target-checked)"]
  CARD --> REG["AI Registry: card URL written back"]
```

What this class is for is agreement. The published card says an agent answers at ``url``; the
routes have to be *at* that URL, the card document has to be where the card says it is, and the
Registry has to be told the same string. One constructor argument (``url``) decides all three, so
they cannot drift.

Mount the app (or its routes) at the host root of that URL:

```python
server = await A2AServer.from_registry(
    harness, agent=refund_agent, url="https://agents.example.com/a2a",
    registry=harness.registry, identity=TrustedHeaderIdentity(allowed_tenants={"acme"}),
)
app.mount("/", server.app())          # or: server.add_to_fastapi(app)
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from a2a.server.agent_execution import SimpleRequestContextBuilder
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    PushNotificationConfigStore,
    TaskStore,
)
from a2a.types import AgentCard as SDKAgentCard
from trellis.contracts.a2a import AgentCard, AgentProvider
from trellis.contracts.descriptors import AgentDescriptor

from trellis.harness.registry.directory import (
    AGENT_CARD_PATH,
    CARD_URL_METADATA,
    RegistryAgentDirectory,
)
from trellis.harness.runtime.logging import get_logger
from trellis.harness_a2a.card import agent_card, card_location, to_sdk_card
from trellis.harness_a2a.executor import HarnessAgentExecutor
from trellis.harness_a2a.identity import IdentityResolver
from trellis.harness_a2a.push import HarnessPushNotifier
from trellis.harness_a2a.tasks import HarnessTaskStore

if TYPE_CHECKING:  # pragma: no cover - typing only, so starlette stays a runtime concern
    from starlette.applications import Starlette
    from starlette.routing import BaseRoute

log = get_logger("trellis.harness_a2a.server")

#: Where JSON-RPC is served when the published URL has no path of its own.
DEFAULT_RPC_PATH: Final = "/"


class A2AServer:
    """A harness agent published as an A2A agent."""

    def __init__(
        self,
        harness: Any,
        *,
        agent: Any,
        url: str,
        agent_id: str | None = None,
        identity: IdentityResolver | None = None,
        entity: Mapping[str, Any] | None = None,
        directory: RegistryAgentDirectory | None = None,
        task_store: TaskStore | None = None,
        push_notifier: HarnessPushNotifier | None = None,
        push_config_store: PushNotificationConfigStore | None = None,
        webhook_secret: str | None = None,
        provider: AgentProvider | None = None,
        security_schemes: Mapping[str, Mapping[str, Any]] | None = None,
        security: Sequence[Mapping[str, Sequence[str]]] | None = None,
        extensions: Sequence[Mapping[str, Any]] = (),
        documentation_url: str | None = None,
        rest: bool = False,
        enable_v0_3_compat: bool = False,
        card_path: str = AGENT_CARD_PATH,
    ) -> None:
        """``url`` is the public A2A endpoint; every path below is derived from it.

        ``entity`` is the Registry entity for this agent (see :meth:`from_registry`, which reads
        it). Push notifications are on when a notifier is passed or ``webhook_secret`` is given —
        the capability is advertised on the card only then, because advertising a capability the
        deployment has not configured is how a caller ends up waiting for a notification that
        cannot arrive.
        """
        self.harness = harness
        self.url = url
        self.directory = directory
        self.executor = HarnessAgentExecutor(
            harness, agent=agent, agent_id=agent_id, identity=identity
        )
        self.descriptor: AgentDescriptor = self.executor.agent.descriptor
        self.task_store: TaskStore = task_store or HarnessTaskStore(
            getattr(harness, "runs", None), identity=identity
        )
        self.push_configs = push_config_store or InMemoryPushNotificationConfigStore()
        self.push_notifier = push_notifier or (
            HarnessPushNotifier(self.push_configs, secret=webhook_secret)
            if webhook_secret
            else None
        )
        self._owns_notifier = push_notifier is None and self.push_notifier is not None
        self.card: AgentCard = agent_card(
            self.descriptor,
            url=url,
            entity=entity,
            streaming=True,
            push_notifications=self.push_notifier is not None,
            provider=provider,
            security_schemes=security_schemes,
            security=security,
            extensions=extensions,
            documentation_url=documentation_url,
            card_path=card_path,
        )
        self.sdk_card: SDKAgentCard = to_sdk_card(self.card)
        self.handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=self.task_store,
            agent_card=self.sdk_card,
            push_config_store=self.push_configs if self.push_notifier else None,
            push_sender=self.push_notifier,
            push_url_validator=self.push_notifier.validate_url if self.push_notifier else None,
            request_context_builder=SimpleRequestContextBuilder(task_store=self.task_store),
        )
        self.rpc_path = urlsplit(url).path or DEFAULT_RPC_PATH
        self.card_path = urlsplit(self.card_url).path or card_path
        self.rest = rest
        self.enable_v0_3_compat = enable_v0_3_compat

    @classmethod
    async def from_registry(
        cls,
        harness: Any,
        *,
        agent: Any,
        url: str,
        registry: Any = None,
        audience: str | None = None,
        publish_card: bool = True,
        allow_local_card_url: bool = False,
        **options: Any,
    ) -> A2AServer:
        """Build a server whose card comes from the Registry entity, and tell the Registry where
        that card is.

        A registry that is unreachable, or an entity that does not exist, degrades to a card built
        from the descriptor alone — loudly. The alternative (refusing to start) would make a
        catalogue outage an agent outage, which is the property the Registry's own data-plane
        design exists to avoid.
        """
        client = registry if registry is not None else getattr(harness, "registry", None)
        entity: Mapping[str, Any] | None = None
        directory: RegistryAgentDirectory | None = None
        agent_id = _declared_agent_id(harness, agent, options.get("agent_id"))
        if client is not None and callable(getattr(client, "entity", None)):
            directory = RegistryAgentDirectory(
                client, audience=audience, allow_local=allow_local_card_url
            )
            try:
                entity = await client.entity(agent_id)
            except Exception as exc:
                log.warning("a2a.registry_unreachable", error=str(exc), serving="descriptor only")
            if entity is None:
                log.warning(
                    "a2a.agent_not_in_registry",
                    agent_id=agent_id,
                    detail="serving a card built from the descriptor; nobody approved this agent",
                )
        server = cls(harness, agent=agent, url=url, entity=entity, directory=directory, **options)
        if publish_card:
            await server.publish_card()
        return server

    # ------------------------------------------------------------------ the card
    @property
    def card_url(self) -> str:
        """Where this server's card document is published."""
        return str(self.card.metadata.get(CARD_URL_METADATA) or card_location(self.url))

    async def publish_card(self) -> bool:
        """Write the card's location onto the Registry entity. Never fatal (design §9)."""
        if self.directory is None:
            return False
        await self.directory.publish(self.card)
        return True

    # ------------------------------------------------------------------ the routes
    def routes(self) -> list[BaseRoute]:
        """The card route and the transport routes, at the paths the card publishes."""
        built: list[BaseRoute] = [
            *create_agent_card_routes(self.sdk_card, card_url=self.card_path),
            *create_jsonrpc_routes(
                self.handler, self.rpc_path, enable_v0_3_compat=self.enable_v0_3_compat
            ),
        ]
        if self.rest:
            built.extend(
                create_rest_routes(
                    self.handler,
                    enable_v0_3_compat=self.enable_v0_3_compat,
                    path_prefix=self.rpc_path.rstrip("/"),
                )
            )
        return built

    def app(self) -> Starlette:
        """A Starlette app serving exactly this agent, closing the handler on shutdown.

        The close matters: the SDK's request handler owns per-task producer and consumer tasks, and
        a process that exits without ``aclose()`` leaves them behind.
        """
        from starlette.applications import Starlette  # noqa: PLC0415 - optional at import time

        return Starlette(routes=self.routes(), lifespan=self._lifespan)

    @asynccontextmanager
    async def _lifespan(self, _app: Any) -> AsyncIterator[None]:
        """Starlette 1.x has no ``on_shutdown``: the close hangs off the lifespan instead."""
        try:
            yield
        finally:
            await self.aclose()

    def add_to_fastapi(self, app: Any) -> None:
        """Add the same routes to an existing FastAPI application.

        The application's own lifespan is its to own, so it has to call :meth:`aclose` on shutdown
        (``app.router.lifespan_context``, or a ``@asynccontextmanager`` lifespan of its own).
        """
        from a2a.server.routes import add_a2a_routes_to_fastapi  # noqa: PLC0415 - needs fastapi

        add_a2a_routes_to_fastapi(
            app,
            agent_card_routes=create_agent_card_routes(self.sdk_card, card_url=self.card_path),
            jsonrpc_routes=create_jsonrpc_routes(
                self.handler, self.rpc_path, enable_v0_3_compat=self.enable_v0_3_compat
            ),
        )

    async def aclose(self) -> None:
        """Release the handler's background tasks and the notifier's client."""
        await self.handler.aclose()
        if self._owns_notifier and self.push_notifier is not None:
            await self.push_notifier.aclose()


def _declared_agent_id(harness: Any, agent: Any, agent_id: str | None) -> str:
    """The id this agent will run under, before it is wrapped.

    Needed because the Registry entity has to be read *before* the card is built, and the harness
    derives the same id when it wraps: an explicit id, else an already-wrapped agent's descriptor,
    else the callable's name — qualified with the deployment's product key either way.
    """
    descriptor = getattr(agent, "descriptor", None)
    if descriptor is not None and getattr(descriptor, "agent_id", None):
        return str(descriptor.agent_id)  # already wrapped: already qualified
    return str(harness.qualify(agent_id or getattr(agent, "__name__", "agent")))


__all__ = ["DEFAULT_RPC_PATH", "A2AServer"]
