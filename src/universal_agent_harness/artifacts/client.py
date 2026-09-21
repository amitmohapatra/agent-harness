"""``runtime.artifacts``: store a payload, get a reference back, keep results small."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.artifacts import ArtifactRef

from universal_agent_harness.telemetry import names as N

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


class ArtifactRuntime:
    """Per-execution artifact client. Registers every artifact it creates on the result."""

    def __init__(
        self, store: Any, *, runtime_ref: Any = None, inline_max_bytes: int = 64_000
    ) -> None:
        self._store = store
        self._runtime: AgentRuntime | None = runtime_ref
        self.inline_max_bytes = inline_max_bytes
        self.created: list[ArtifactRef] = []

    def attach(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime

    async def put(
        self,
        content: bytes | str,
        *,
        type: str = "blob",
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactRef:
        """Store content. The key defaults to one that is stable across retries (§42)."""
        runtime = self._runtime
        key = idempotency_key
        if key is None and runtime is not None:
            key = runtime.idempotency_key("artifact", type, len(content))
        if runtime is None:
            ref = await self._store.put(
                content, type=type, mime_type=mime_type, metadata=metadata, idempotency_key=key
            )
            self.created.append(ref)
            return ref
        with runtime.tracer.artifact_span(type) as span:
            ref = await self._store.put(
                content, type=type, mime_type=mime_type, metadata=metadata, idempotency_key=key
            )
            span.set(**{N.ARTIFACT_ID: ref.artifact_id, N.ARTIFACT_SIZE: ref.size_bytes})
            span.ok()
        self.created.append(ref)
        return ref

    async def get(self, artifact_id: str) -> bytes | None:
        return await self._store.get(artifact_id)

    def should_offload(self, value: Any) -> bool:
        """Whether a result payload is large enough to belong in a store instead (§43)."""
        if isinstance(value, bytes | str):
            return len(value) > self.inline_max_bytes
        return False
