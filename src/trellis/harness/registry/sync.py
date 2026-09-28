"""Keeping the deployment, the Registry and the gateway in agreement (design §9, §14).

``reconcile()`` answers the startup question — *is what we are about to serve the thing that was
approved?* This module keeps answering it while the process runs, and extends it to the second
half of the catalogue: the Registry owns which tools exist, so Bifrost's MCP clients are
configured *from* it rather than by hand, which is what stops a tool nobody approved appearing
in a model's tool list.

```mermaid
sequenceDiagram
  participant S as RegistrySync
  participant R as AI Registry
  participant B as Bifrost
  S->>R: reconcile (declared agents vs manifest)
  loop every cycle
    S->>R: GET manifest (If-None-Match)
    R-->>S: 304, or a new manifest
    S->>S: ManifestDelta(added / removed / changed) -> log + metric
    S->>R: heartbeat (the entity is still there, still bound)
    S->>B: mcp.add / mcp.update / mcp.remove from tool entities
  end
```

Three properties, in order of importance:

1. **Never fatal.** Every cycle is wrapped: a registry outage, a gateway 500 or a malformed
   entity is logged, counted and slept off. The agents keep serving on last-known-good.
2. **Loud.** Drift in either direction is a warning with the names in it, plus a metric, because
   the direction tells you which team to talk to.
3. **It only removes what it added.** A gateway MCP client this job never configured is left
   alone, however tempting the symmetry: deleting a hand-registered server because a catalogue
   does not mention it is the kind of "reconciliation" that takes production down.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from trellis.contracts.descriptors import AgentDescriptor

from trellis.harness.events.targets import TargetRefused, validate_url
from trellis.harness.registry.ai_registry import Reconciliation
from trellis.harness.runtime.logging import get_logger
from trellis.harness.telemetry.metrics import REGISTRY_DRIFT, REGISTRY_SYNC, MetricsRecorder

log = get_logger("trellis.harness.registry.sync")

#: The gateway forbids hyphens in an MCP client's name and routes tool calls by that name.
NAME_SEPARATOR: Final = "_"
#: How long a failed cycle waits before the next one, and the ceiling on that wait.
BACKOFF_SECONDS: Final = 5.0
MAX_BACKOFF_SECONDS: Final = 300.0
#: The floor between two cycles, whatever woke them. A channel that signals constantly (or one
#: that ends as soon as it is read) would otherwise turn the sync job into a busy loop against
#: the registry, which is the opposite of what a change channel is for.
MIN_CYCLE_SECONDS: Final = 1.0
#: Connection types the gateway accepts (its own enum).
CONNECTION_TYPES: Final = frozenset({"http", "sse", "stdio"})
#: What this job will configure from catalogue data without being asked twice. ``stdio`` makes the
#: gateway run a local process, so a catalogue entry must not be able to introduce one by itself:
#: the gateway refuses stdio to unauthenticated callers for the same reason.
DEFAULT_CONNECTION_TYPES: Final = frozenset({"http", "sse"})


@runtime_checkable
class DeltaChannel(Protocol):
    """A registry that offers a change channel (the manifest's own ``channel`` block).

    Deliberately an async iterator and nothing more: a deployment that already runs Redis wires
    its own subscriber in, and the harness imports no bus client. Every yield means "re-read the
    manifest"; the payload is logged and otherwise ignored, since the manifest is the truth.
    """

    def watch(self) -> AsyncIterator[Any]: ...


@dataclass(slots=True, frozen=True)
class ManifestDelta:
    """What moved between two manifests, by entity name (``type:name``)."""

    seq: Any = None
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


@dataclass(slots=True)
class MCPReconciliation:
    """What the sync job did to the gateway's MCP clients, and what it refused to do."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unbound: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    @property
    def in_sync(self) -> bool:
        return not (self.added or self.updated or self.removed)


@dataclass(slots=True, frozen=True)
class ToolBinding:
    """A Registry tool entity's MCP binding, as the gateway needs it.

    Deliberately just the two fields the catalogue owns. Forwarding the rest of an entity's ``mcp``
    block as gateway options would let whoever writes a catalogue entry configure a client every
    team's key can reach; gateway options belong in the gateway's own configuration.
    """

    name: str
    connection_type: str
    connection_string: str | None = None

    def changes_from(self, existing: Mapping[str, Any]) -> dict[str, Any]:
        """What to send to ``mcp.update`` so the gateway matches this binding; empty when it
        already does."""
        changes: dict[str, Any] = {}
        if str(existing.get("connection_type") or "") != self.connection_type:
            changes["connection_type"] = self.connection_type
        current = existing.get("connection_string") or existing.get("connection_url")
        if self.connection_string is not None and str(current or "") != self.connection_string:
            changes["connection_string"] = self.connection_string
        return changes


def mcp_client_name(entity_name: str) -> str:
    """The gateway's name for a Registry tool: hyphens become underscores.

    The gateway refuses hyphens and prefixes every tool it lists with the client's name, so this
    mapping decides what the model sees. It is deliberately not clever: two entities that would
    collide here are refused by the caller rather than silently merged into one server.
    """
    name = entity_name.strip().replace("-", NAME_SEPARATOR).replace(".", NAME_SEPARATOR)
    if not name or not name.replace(NAME_SEPARATOR, "").isalnum():
        raise ValueError(f"{entity_name!r} cannot be an MCP client name")
    return name


def tool_binding(entity: Mapping[str, Any], *, allow_local: bool = False) -> ToolBinding | None:
    """The MCP binding a tool entity declares, or ``None`` when it declares none.

    Accepts the nested spelling (``mcp: {connection_type, connection_string}``) and the flat one
    (``connection_type``/``connection_string``, with ``url`` as an alias), because the Registry's
    spec blob is another product's shape and a deployment may have filled either in. A tool with
    no binding is not configurable here: the caller logs it as unbound.

    An ``http``/``sse`` binding is a URL the *gateway* will fetch on behalf of every key allowed to
    use that client, so it passes the same target checks a webhook does: a catalogue entry pointing
    at ``169.254.169.254`` or an internal service is refused rather than configured.
    """
    name = str(entity.get("name") or "")
    if not name:
        return None
    nested = entity.get("mcp") if isinstance(entity.get("mcp"), Mapping) else None
    block: Mapping[str, Any] = nested if nested is not None else entity
    connection_type = str(block.get("connection_type") or block.get("type") or "").lower()
    if connection_type not in CONNECTION_TYPES:
        return None
    connection = block.get("connection_string") or block.get("url") or block.get("connection_url")
    target = str(connection) if connection else None
    if target is not None and connection_type in ("http", "sse"):
        try:
            target = validate_url(target, allow_local=allow_local)
        except TargetRefused as exc:
            log.warning("registry.mcp_target_refused", tool=name, error=str(exc))
            return None
    return ToolBinding(name=name, connection_type=connection_type, connection_string=target)


class RegistrySync:
    """The startup reconciliation, on a loop, plus the gateway's MCP clients."""

    def __init__(
        self,
        registry: Any,
        *,
        descriptors: Iterable[AgentDescriptor] = (),
        gateway: Any = None,
        telemetry: Any = None,
        audience: str | None = None,
        heartbeat_seconds: float = 60.0,
        poll_seconds: float = 30.0,
        channel: DeltaChannel | None = None,
        min_cycle_seconds: float = MIN_CYCLE_SECONDS,
        remove_unlisted_mcp_clients: bool = True,
        connection_types: Iterable[str] = DEFAULT_CONNECTION_TYPES,
        allow_local_targets: bool = False,
    ) -> None:
        """``gateway`` is a ``bifrost_sdk.Bifrost`` (or the harness's ``BifrostModelClient``,
        whose gateway is reused) — without one no MCP client is configured. ``channel`` is the
        registry's change channel when the deployment has one; without it the ETag poll is the
        subscription. ``remove_unlisted_mcp_clients`` only ever removes clients this job
        configured itself."""
        if not callable(getattr(registry, "manifest", None)):
            raise TypeError("RegistrySync needs an AIRegistryClient")
        self._registry = registry
        self._gateway = getattr(gateway, "gateway", None) or gateway
        if self._gateway is not None and not hasattr(self._gateway, "mcp"):
            raise TypeError("RegistrySync needs a Bifrost client (or a BifrostModelClient)")
        self.descriptors = list(descriptors)
        self.audience = audience
        self.heartbeat_seconds = max(1.0, heartbeat_seconds)
        self.poll_seconds = max(1.0, poll_seconds)
        self.channel = channel
        self.min_cycle_seconds = max(0.0, min_cycle_seconds)
        self.connection_types = frozenset(connection_types) & CONNECTION_TYPES
        self.allow_local_targets = allow_local_targets
        self.remove_unlisted_mcp_clients = remove_unlisted_mcp_clients
        self.metrics = MetricsRecorder(telemetry, enabled=telemetry is not None)
        #: The names this job configured in the gateway: the only ones it may remove.
        self._owned: set[str] = set()
        self._snapshot: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._last_cycle = 0.0
        self.cycles = 0
        self.failures = 0

    # ------------------------------------------------------------------ one-shot operations
    async def reconcile(self) -> Reconciliation:
        """The registry's agents against this process's, by name (see ``AIRegistryClient``)."""
        result = await self._registry.reconcile([d.agent_id for d in self.descriptors])
        self.metrics.count(
            REGISTRY_SYNC, operation="reconcile", status="in_sync" if result.in_sync else "drift"
        )
        for kind, names in (("unbound", result.unbound), ("unregistered", result.unregistered)):
            if names:
                self.metrics.count(REGISTRY_DRIFT, float(len(names)), operation=kind)
        return result

    async def heartbeat(self, *, status: str = "healthy") -> None:
        """Say this process is still serving each declared agent (and check it is still listed)."""
        for descriptor in self.descriptors:
            await self._registry.heartbeat(descriptor, status=status)
        self.metrics.count(REGISTRY_SYNC, operation="heartbeat", status=status)

    async def poll(self) -> ManifestDelta:
        """Re-read the manifest and report what moved. A 304 costs nothing and yields no delta."""
        manifest = await self._registry.manifest(refresh=True)
        current = _snapshot(manifest)
        previous, self._snapshot = self._snapshot, current
        if previous is None:
            return ManifestDelta(seq=manifest.get("seq"))
        delta = ManifestDelta(
            seq=manifest.get("seq"),
            added=tuple(sorted(set(current) - set(previous))),
            removed=tuple(sorted(set(previous) - set(current))),
            changed=tuple(
                sorted(k for k in set(current) & set(previous) if current[k] != previous[k])
            ),
        )
        if not delta.empty:
            log.info(
                "registry.manifest_delta",
                seq=delta.seq,
                added=list(delta.added),
                removed=list(delta.removed),
                changed=list(delta.changed),
            )
            self.metrics.count(
                REGISTRY_SYNC,
                float(len(delta.added) + len(delta.removed) + len(delta.changed)),
                operation="delta",
            )
        return delta

    async def configure_mcp_clients(self) -> MCPReconciliation:
        """Make the gateway's MCP clients match the Registry's tool entities (design §3).

        The catalogue decides what exists; the gateway decides what a key may reach. Nothing
        here grants access — it registers servers the Registry already lists — and a name that
        cannot be mapped, or that two entities would share, is refused rather than guessed at.
        """
        result = MCPReconciliation()
        if self._gateway is None:
            return result
        desired = _desired_clients(
            await self._registry.tools(audience=self.audience),
            result,
            self.connection_types,
            allow_local=self.allow_local_targets,
        )
        existing = {_client_name(c): c for c in await self._gateway.mcp.clients()}
        existing.pop("", None)
        for name, binding in desired.items():
            found = existing.get(name)
            try:
                if found is None:
                    await self._gateway.mcp.add(
                        name,
                        connection_type=binding.connection_type,
                        connection_string=binding.connection_string,
                    )
                    result.added.append(name)
                    # only what this job actually wrote is its to remove later
                    self._owned.add(name)
                else:
                    changes = binding.changes_from(_client_config(found))
                    if changes:
                        await self._gateway.mcp.update(_client_id(found, name), **changes)
                        result.updated.append(name)
                        self._owned.add(name)
            except Exception as exc:
                # one gateway refusal is one tool nobody can call, not a reason to stop
                # configuring the rest of the catalogue
                result.refused.append(binding.name)
                log.warning("registry.mcp_write_failed", tool=binding.name, error=str(exc))
        if self.remove_unlisted_mcp_clients:
            for name in sorted((self._owned & set(existing)) - set(desired)):
                try:
                    await self._gateway.mcp.remove(_client_id(existing[name], name))
                except Exception as exc:
                    log.warning("registry.mcp_remove_failed", gateway_name=name, error=str(exc))
                    continue
                result.removed.append(name)
                self._owned.discard(name)
        if result.unbound:
            log.warning(
                "registry.tools_without_mcp_binding",
                tools=result.unbound,
                detail="listed as tools with no MCP connection; the gateway cannot serve them",
            )
        if not result.in_sync:
            log.info(
                "registry.mcp_configured",
                added=result.added,
                updated=result.updated,
                removed=result.removed,
            )
            self.metrics.count(
                REGISTRY_DRIFT,
                float(len(result.added) + len(result.updated) + len(result.removed)),
                operation="mcp",
            )
        return result

    async def cycle(self) -> ManifestDelta:
        """One pass: poll, heartbeat, configure the gateway. Raises nothing the loop cannot sleep
        off — the caller may await it directly in a test, and does see the exception then."""
        delta = await self.poll()
        await self.heartbeat()
        if self._gateway is not None:
            await self.configure_mcp_clients()
        self.cycles += 1
        return delta

    # ------------------------------------------------------------------ the loop
    async def start(self) -> None:
        """Reconcile once, configure the gateway once, then run the loop in the background."""
        with contextlib.suppress(Exception):  # a registry outage must not stop a deployment
            await self.reconcile()
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task = asyncio.create_task(self._run(), name="registry-sync")

    async def aclose(self) -> None:
        self._stopping.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        backoff = 0.0
        while not self._stopping.is_set():
            await self._rate_limit()
            if self._stopping.is_set():
                break
            try:
                self._last_cycle = asyncio.get_running_loop().time()
                await self.cycle()
                backoff = 0.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failures += 1
                backoff = min(MAX_BACKOFF_SECONDS, max(BACKOFF_SECONDS, backoff * 2))
                log.warning("registry.sync_cycle_failed", error=str(exc), retry_in_seconds=backoff)
                self.metrics.count(REGISTRY_SYNC, operation="cycle", status="error")
            await self._wait(backoff or self._interval())

    async def _rate_limit(self) -> None:
        """Keep two cycles at least ``min_cycle_seconds`` apart, however they were triggered."""
        remaining = self.min_cycle_seconds - (asyncio.get_running_loop().time() - self._last_cycle)
        if remaining > 0 and self._last_cycle:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=remaining)

    def _interval(self) -> float:
        """The poll interval, or the longer safety net when a channel is doing the waking."""
        base = min(self.poll_seconds, self.heartbeat_seconds)
        return base if self.channel is None else max(base, self.heartbeat_seconds)

    async def _wait(self, seconds: float) -> None:
        """Sleep, unless the registry's channel says the manifest moved first."""
        if self.channel is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
            return
        try:
            await asyncio.wait_for(self._next_signal(), timeout=seconds)
        except TimeoutError:
            return
        except Exception as exc:  # a dropped channel degrades to polling, loudly
            log.warning("registry.channel_failed", error=str(exc), serving="poll")
            self.channel = None

    async def _next_signal(self) -> None:
        """Wait for one signal from the channel. A channel that ends falls back to polling.

        The stream is closed either way: a deployment's subscriber is usually a real subscription,
        and leaving one unfinalised per cycle leaks it.
        """
        channel = self.channel
        if channel is None:  # pragma: no cover - only reachable if the channel was cleared
            return
        stream = channel.watch()
        try:
            async for signal in stream:
                log.debug("registry.channel_signal", signal=str(signal)[:200])
                return
        finally:
            # an async generator finalises here; an iterator without aclose has nothing to release
            if isinstance(stream, AsyncGenerator):
                with contextlib.suppress(Exception):
                    await stream.aclose()


def _desired_clients(
    entities: Iterable[Mapping[str, Any]],
    result: MCPReconciliation,
    connection_types: frozenset[str],
    *,
    allow_local: bool = False,
) -> dict[str, ToolBinding]:
    """The gateway clients the Registry's tool entities ask for, by gateway name.

    Everything it will not configure is recorded on ``result`` rather than raised: a catalogue
    row nobody can act on is a deployment gap to report, not a reason to stop configuring the
    rest of the catalogue.
    """
    desired: dict[str, ToolBinding] = {}
    #: A name two entities wanted is refused for every later entity too, not handed to the third.
    collided: set[str] = set()
    for entity in entities:
        binding = tool_binding(entity, allow_local=allow_local)
        if binding is None:
            result.unbound.append(str(entity.get("name") or "?"))
            continue
        if binding.connection_type not in connection_types:
            result.refused.append(binding.name)
            log.warning(
                "registry.mcp_connection_type_refused",
                tool=binding.name,
                connection_type=binding.connection_type,
                detail="not in this deployment's allowed MCP connection types",
            )
            continue
        try:
            name = mcp_client_name(binding.name)
        except ValueError as exc:
            result.refused.append(binding.name)
            log.warning("registry.mcp_name_refused", tool=binding.name, error=str(exc))
            continue
        if name in collided or name in desired:
            # every entity that wants this name is refused: the one already accepted is no safer
            # than the new one, and a third must not inherit the slot the first two lost
            result.refused.extend(
                [b.name for b in (desired.pop(name, None),) if b is not None] + [binding.name]
            )
            collided.add(name)
            log.warning(
                "registry.mcp_name_collision",
                tool=binding.name,
                gateway_name=name,
                detail="several tool entities map to one gateway client name; none is used",
            )
            continue
        desired[name] = binding
    return desired


def _snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """``{type:name: version}`` for every entity — enough to name what moved."""
    snapshot: dict[str, Any] = {}
    for entity in manifest.get("entities", []) or ():
        if not isinstance(entity, Mapping):
            continue
        name = str(entity.get("name") or "")
        if name:
            snapshot[f"{entity.get('type') or 'unknown'}:{name}"] = entity.get("version")
    return snapshot


def _client_name(client: Mapping[str, Any]) -> str:
    config = client.get("config")
    if isinstance(config, Mapping) and config.get("name"):
        return str(config["name"])
    return str(client.get("name") or "")


def _client_config(client: Mapping[str, Any]) -> Mapping[str, Any]:
    config = client.get("config")
    return config if isinstance(config, Mapping) else client


def _client_id(client: Mapping[str, Any], name: str) -> str:
    """What the gateway's client endpoints address: its id when it has one, else its name."""
    for key in ("id", "client_id"):
        value = client.get(key)
        if value:
            return str(value)
    return name


__all__: Sequence[str] = [
    "DeltaChannel",
    "MCPReconciliation",
    "ManifestDelta",
    "RegistrySync",
    "ToolBinding",
    "mcp_client_name",
    "tool_binding",
]
