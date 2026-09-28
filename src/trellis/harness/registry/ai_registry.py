"""The AI Registry as the harness's agent registry (§47).

The registry is a *control plane*: it owns what tools and agents exist, who may see them,
and what version is current. Its own architecture note is the rule this client obeys —
servers serve from an in-memory manifest and contact the registry only at bootstrap and for
deltas, so that the registry being down degrades freshness rather than availability.

So discovery reads the **manifest** (data plane, API-key auth, ETag-cached) rather than the
entity CRUD API (control plane, user JWT). Three consequences worth stating:

* A 304 costs nothing, which is what makes polling a reasonable fallback when no Redis
  channel is configured.
* ``views`` are *pre-resolved per audience by the registry*. This client picks a view; it
  must never merge overlays itself, or two surfaces will disagree about what a tool is.
* A cached manifest survives a registry outage. An agent that cannot reach the registry
  keeps running on last-known-good instead of failing to start.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final

from trellis.contracts.descriptors import AgentDescriptor

from trellis.harness.events.targets import TargetRefused, validate_url
from trellis.harness.runtime.logging import get_logger

log = get_logger("trellis.harness.registry.ai")

_NOT_MODIFIED = 304
_ERROR = 400

#: Where an entity is written. Configuration, not a guess: the entity API belongs to the
#: registry product. ``{entity_id}`` is the manifest's own id for the entity.
DEFAULT_ENTITY_PATH: Final = "/v1/entities/{entity_id}"
#: The entity spec field holding an agent's A2A Agent Card location. The harness both writes
#: and reads it, so the name is the platform's; the two aliases are for an entity a person
#: filled in by hand before this existed.
CARD_URL_FIELD: Final = "a2a_card_url"
CARD_URL_ALIASES: Final = (CARD_URL_FIELD, "card_url", "agent_card_url")


@dataclass(slots=True)
class Reconciliation:
    """How the registry's agents and this process's agents line up.

    The registry decides *what exists and who may see it*; code decides *what it does*. They
    are bound by **name** — the harness's ``agent_id`` must equal the registry entity's
    ``name``, exactly as the MCP SDK binds a handler with ``@server.tool("get_invoice")``.
    """

    #: Listed by the registry and implemented here. These are the agents that actually run.
    bound: list[str] = field(default_factory=list)
    #: Listed by the registry with no local handler. Fail-safe, never fail-crash: they are
    #: not served, and the warning says so — a deploy that dropped a handler should be loud
    #: but must not take the rest of the process down with it.
    unbound: list[str] = field(default_factory=list)
    #: Implemented here but not listed. Not exposed: the registry is the source of truth for
    #: exposure, which is what stops a deployment shipping an agent nobody approved.
    unregistered: list[str] = field(default_factory=list)

    @property
    def in_sync(self) -> bool:
        return not self.unbound and not self.unregistered


class AIRegistryClient:
    """Reads agents from an AI Registry product manifest.

    Implements the harness's registry port (``register``/``heartbeat``) and adds the
    discovery a planner needs. Registration is deliberately a no-op unless a control-plane
    token is supplied: the registry treats "what exists" as reviewed configuration, not
    something a process announces about itself at startup.
    """

    name = "ai-registry"

    def __init__(
        self,
        base_url: str,
        *,
        product_key: str,
        api_key: str,
        timeout: float = 10.0,
        client: Any = None,
        control_plane_token: str | None = None,
        entity_path: str = DEFAULT_ENTITY_PATH,
    ) -> None:
        """``control_plane_token`` and ``entity_path`` enable the one control-plane write the
        harness performs: publishing an agent's A2A card location (:meth:`publish_card_url`).
        The path is configuration because the entity API belongs to the registry product, not
        to this client — without a token the write is skipped loudly rather than guessed."""
        import httpx  # noqa: PLC0415 - optional at import time, required to construct

        if not base_url or not product_key or not api_key:
            raise ValueError("AIRegistryClient needs base_url, product_key and api_key")
        if "{entity_id}" not in entity_path:
            raise ValueError("entity_path must contain {entity_id}")
        self.product_key = product_key
        self._httpx = httpx
        #: Sent per request, not baked into the client. A caller supplying its own
        #: ``httpx.AsyncClient`` (for pooling, for a proxy, for tests) would otherwise get a
        #: silently unauthenticated client and a 401 that looks like a bad key.
        self._auth = {"X-API-Key": api_key}
        self._control_plane_token = control_plane_token
        self.entity_path = entity_path
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
        )
        self._owns_client = client is None
        self._etag: str | None = None
        self._manifest: dict[str, Any] | None = None

    # ------------------------------------------------------------------ identity
    def qualified(self, name: str) -> str:
        """The globally unique id for an agent the registry knows as ``name``.

        The registry enforces ``UNIQUE(product_id, type, name)`` — so "refund-agent" is
        unique *within a product*, and two teams may each own one. Nothing downstream knows
        about products: the Memory Service keys a private memory as
        ``principal:{tenant}/agent:{agent_id}``, scoped by tenant alone. Hand it a bare name
        and two products inside one tenant share a memory scope, which is a cross-team data
        leak that looks exactly like a working system.

        So the identity that leaves this process is ``{product_key}:{name}``.

        ``:`` rather than ``/`` on purpose: ``safe_id`` keeps ``:`` and rewrites ``/`` to a
        hyphen, which would turn ``billing/refund-agent`` into ``billing-refund-agent`` —
        indistinguishable from product ``billing-refund``'s agent ``agent``. A separator that
        survives sanitisation is the only one that can carry a guarantee.

        A name that already carries a product prefix is returned unchanged, so a process
        serving several products can declare fully-qualified ids itself.
        """
        return name if ":" in name else f"{self.product_key}:{name}"

    # ------------------------------------------------------------------ data plane
    async def manifest(self, *, refresh: bool = True) -> dict[str, Any]:
        """The product manifest, re-fetched only when the registry says it changed.

        Returns the cached copy when the registry answers 304 **or** cannot be reached at
        all — the second case is the one that matters: a registry outage must not take the
        agents down with it.
        """
        if not refresh and self._manifest is not None:
            return self._manifest
        headers = dict(self._auth)
        if self._etag:
            headers["If-None-Match"] = self._etag
        try:
            response = await self._client.get(
                f"/v1/products/{self.product_key}/manifest", headers=headers
            )
        except self._httpx.HTTPError as exc:
            if self._manifest is None:
                raise
            log.warning("registry.unreachable", error=str(exc), serving="last-known-good")
            return self._manifest
        if response.status_code == _NOT_MODIFIED and self._manifest is not None:
            return self._manifest
        if response.status_code >= _ERROR:
            if self._manifest is not None:
                log.warning("registry.error", status=response.status_code)
                return self._manifest
            raise RuntimeError(f"registry returned {response.status_code}: {response.text[:200]}")
        self._manifest = dict(response.json())
        self._etag = response.headers.get("ETag") or self._etag
        return self._manifest

    async def agents(
        self, *, audience: str | None = None, refresh: bool = True
    ) -> list[dict[str, Any]]:
        """Enabled agents, as the given audience sees them.

        Falls back to the manifest's ``default_audience`` — the registry's own answer for a
        caller with no entitlement — rather than to "show everything", which would leak an
        internal agent to an external surface the first time an audience was misspelled.
        """
        manifest = await self.manifest(refresh=refresh)
        wanted = audience or manifest.get("default_audience") or "external"
        found = []
        for entity in manifest.get("entities", []):
            if entity.get("type") != "agent":
                continue
            view = (entity.get("views") or {}).get(wanted)
            if not view or not view.get("enabled"):
                continue
            found.append(
                {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    #: What the harness must use as its agent_id — see qualified().
                    "agent_id": self.qualified(str(entity.get("name"))),
                    "version": entity.get("version"),
                    **(view.get("spec") or {}),
                }
            )
        return found

    async def all_agents(self, *, refresh: bool = True) -> list[dict[str, Any]]:
        """Every agent in the product, whatever any audience can see.

        The set a deployment is accountable for. :meth:`agents` answers the narrower,
        per-request question of what a given caller is entitled to discover.
        """
        manifest = await self.manifest(refresh=refresh)
        return [
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "agent_id": self.qualified(str(e.get("name"))),
                "version": e.get("version"),
            }
            for e in manifest.get("entities", [])
            if e.get("type") == "agent"
        ]

    async def entity(
        self, agent_id: str, *, type: str = "agent", refresh: bool = False
    ) -> dict[str, Any] | None:
        """The manifest entity behind ``agent_id`` (qualified or bare), or ``None``.

        Served from the cached manifest by default: an Agent Card is built per served agent and
        a card write happens once per deployment, so neither needs a network hop to learn
        something the registry pushes on change.
        """
        wanted = self.qualified(agent_id)
        for entity in (await self.manifest(refresh=refresh)).get("entities", []):
            if entity.get("type") != type:
                continue
            if self.qualified(str(entity.get("name"))) == wanted:
                return dict(entity)
        return None

    async def card_url(self, agent_id: str, *, refresh: bool = False) -> str | None:
        """Where this agent's A2A Agent Card is published, as the registry holds it.

        Read from any audience's view: the card's *location* is one fact about the agent, not a
        per-audience overlay, and an entity that only enables an internal view still has one.
        """
        entity = await self.entity(agent_id, refresh=refresh)
        if entity is None:
            return None
        for view in (entity.get("views") or {}).values():
            spec = (view or {}).get("spec") or {}
            for field_name in CARD_URL_ALIASES:
                value = spec.get(field_name)
                if value:
                    return str(value)
        return None

    async def output_schema(self, agent_id: str, *, refresh: bool = False) -> dict[str, Any] | None:
        """What this agent has declared it returns, or ``None`` if it declared nothing.

        Describes ``AgentResponse.data`` only. The envelope around it — status, claims,
        evidence, error — is fixed by the contract and identical for every agent, so an
        entity that restated it would be the contract copied once per agent, drifting from
        the day it next changed.

        Served from the cached manifest by default: this is called on every completed run,
        and a network hop per turn to re-read something the registry pushes on change would
        be paying for freshness that is already free.
        """
        for entity in (await self.manifest(refresh=refresh)).get("entities", []):
            if entity.get("type") != "agent" or self.qualified(str(entity.get("name"))) != agent_id:
                continue
            for view in (entity.get("views") or {}).values():
                schema = (view.get("spec") or {}).get("output_schema")
                if schema:
                    # Identical across audiences: an override may narrow what a caller
                    # *sends*, never what the agent returns.
                    return dict(schema)
        return None

    async def tools(
        self, *, audience: str | None = None, refresh: bool = True
    ) -> list[dict[str, Any]]:
        """Enabled tools, as the given audience sees them."""
        manifest = await self.manifest(refresh=refresh)
        wanted = audience or manifest.get("default_audience") or "external"
        return [
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "version": e.get("version"),
                **((e.get("views") or {}).get(wanted, {}).get("spec") or {}),
            }
            for e in manifest.get("entities", [])
            if e.get("type") == "tool" and ((e.get("views") or {}).get(wanted) or {}).get("enabled")
        ]

    async def reconcile(self, declared: Iterable[str], *, refresh: bool = True) -> Reconciliation:
        """Compare what the registry lists against what this process implements.

        Call it at startup. It answers the only question that matters when metadata lives in
        one place and behaviour in another: *is what we are about to serve the thing that was
        approved?* Drift in either direction is a real deployment error, and the direction
        tells you which team to talk to.

        Deliberately **not** audience-filtered. An audience answers "who may call this?",
        which is a property of a request, not of a deployment: an agent visible only to
        ``internal`` is still an approved agent, and checking it against the manifest's
        default audience would report it as unregistered on every start. Whether a caller
        may reach it is decided per request, by :meth:`agents`.
        """
        listed = {a["agent_id"] for a in await self.all_agents(refresh=refresh)}
        # Declared names may be bare (this product) or already qualified (a process serving
        # several products). Comparing raw names would report every agent of every other
        # product as unregistered.
        local = {self.qualified(name) for name in declared}
        result = Reconciliation(
            bound=sorted(listed & local),
            unbound=sorted(listed - local),
            unregistered=sorted(local - listed),
        )
        for name in result.unbound:
            log.warning(
                "registry.agent_unbound",
                agent_id=name,
                detail="listed in the registry but no local handler; it will not be served",
            )
        for name in result.unregistered:
            log.warning(
                "registry.agent_unregistered",
                agent_id=name,
                detail="implemented here but not listed in the registry; it is not exposed",
            )
        return result

    # ------------------------------------------------------------------ control plane
    async def publish_card_url(
        self, agent_id: str, card_url: str, *, allow_local: bool = False
    ) -> bool:
        """Write this agent's A2A card location onto its registry entity (design §9).

        The one control-plane write the harness performs, and it publishes a *location*, not an
        agent: the entity still has to exist, so a deployment cannot expose something nobody
        approved. ``True`` when the registry accepted it.

        Loud and never fatal in every other case — no control-plane token, no such entity, a
        rejected request: discovery keeps working off whatever the entity already holds, and the
        warning says which team to talk to. The URL passes the same checks as a webhook target
        first, because a published card URL is a fetch target for every other agent.
        """
        try:
            target = validate_url(card_url, allow_local=allow_local)
        except TargetRefused as exc:
            log.warning("registry.card_url_refused", agent_id=agent_id, error=str(exc))
            return False
        if not self._control_plane_token:
            log.warning(
                "registry.card_url_not_published",
                agent_id=agent_id,
                card_url=target,
                detail="no control-plane token configured; publish the card URL out of band",
            )
            return False
        entity = await self.entity(agent_id)
        if entity is None or not entity.get("id"):
            log.warning(
                "registry.card_url_not_published",
                agent_id=agent_id,
                detail="no registry entity of type agent with this name",
            )
            return False
        path = self.entity_path.format(entity_id=entity["id"])
        try:
            response = await self._client.patch(
                path,
                json={"spec": {CARD_URL_FIELD: target}},
                headers={
                    **self._auth,
                    "Authorization": f"Bearer {self._control_plane_token}",
                },
            )
        except self._httpx.HTTPError as exc:
            log.warning("registry.card_url_write_failed", agent_id=agent_id, error=str(exc))
            return False
        if response.status_code >= _ERROR:
            log.warning(
                "registry.card_url_write_failed",
                agent_id=agent_id,
                status=response.status_code,
                path=path,
            )
            return False
        # the cached manifest now disagrees with the registry about this one field
        self._etag = None
        log.info("registry.card_url_published", agent_id=agent_id, card_url=target)
        return True

    # ------------------------------------------------------------------ the harness port
    async def register(self, descriptor: AgentDescriptor) -> None:
        """Announce this agent. Deliberately a no-op against the data plane.

        Creating an entity is a control-plane write behind a user session and a review
        surface. A process that could register itself would let a deployment add an agent
        nobody approved, which is the property the registry exists to prevent.
        """
        log.debug("registry.register.skipped", agent_id=descriptor.agent_id, reason="control-plane")

    async def heartbeat(self, descriptor: AgentDescriptor, *, status: str = "healthy") -> None:
        """Liveness is reported by re-reading the manifest, not by announcing anything.

        The registry's data plane has no endpoint a process may write to, and inventing one would
        fail in production for every deployment. What a heartbeat *is* here: a check, against the
        manifest this process already holds, that the entity behind this agent still exists and is
        still bound to a local handler. :class:`RegistrySync` polls the manifest itself each cycle,
        so the check is free — and a heartbeat never fetches anything, because a liveness ping that
        makes a network call is a second thing that can fail.
        """
        if self._manifest is None:
            log.debug("registry.heartbeat.no_manifest", agent_id=descriptor.agent_id, status=status)
            return
        entity = await self.entity(descriptor.agent_id, refresh=False)
        if entity is None:
            log.warning(
                "registry.heartbeat.unregistered",
                agent_id=descriptor.agent_id,
                status=status,
                detail="implemented here but not listed in the registry; it is not exposed",
            )
            return
        log.debug("registry.heartbeat", agent_id=descriptor.agent_id, status=status)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["CARD_URL_FIELD", "DEFAULT_ENTITY_PATH", "AIRegistryClient", "Reconciliation"]
