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
from typing import Any

from universal_agent_contracts.descriptors import AgentDescriptor

from universal_agent_harness.runtime.logging import get_logger

log = get_logger("universal_agent_harness.registry.ai")

_NOT_MODIFIED = 304
_ERROR = 400


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
    ) -> None:
        import httpx  # noqa: PLC0415 - optional at import time, required to construct

        if not base_url or not product_key or not api_key:
            raise ValueError("AIRegistryClient needs base_url, product_key and api_key")
        self.product_key = product_key
        self._httpx = httpx
        #: Sent per request, not baked into the client. A caller supplying its own
        #: ``httpx.AsyncClient`` (for pooling, for a proxy, for tests) would otherwise get a
        #: silently unauthenticated client and a 401 that looks like a bad key.
        self._auth = {"X-API-Key": api_key}
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

    # ------------------------------------------------------------------ the harness port
    async def register(self, descriptor: AgentDescriptor) -> None:
        """Announce this agent. Deliberately a no-op against the data plane.

        Creating an entity is a control-plane write behind a user session and a review
        surface. A process that could register itself would let a deployment add an agent
        nobody approved, which is the property the registry exists to prevent.
        """
        log.debug("registry.register.skipped", agent_id=descriptor.agent_id, reason="control-plane")

    async def heartbeat(self, descriptor: AgentDescriptor, *, status: str = "healthy") -> None:
        log.debug("registry.heartbeat.skipped", agent_id=descriptor.agent_id, status=status)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["AIRegistryClient", "Reconciliation"]
