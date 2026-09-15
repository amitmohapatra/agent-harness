"""Artifact stores (§43). Small, boring, and replaceable through the ``ArtifactClient`` port.

Two implementations ship: an in-process store (tests, single-process apps) and a filesystem
store. Anything durable — object storage, or the Memory Service's own file API — implements
the same port; the harness only ever passes references around.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from universal_agent_harness.contracts.artifacts import ArtifactRef
from universal_agent_harness.contracts.errors import ConfigurationError
from universal_agent_harness.contracts.ids import stable_id


class InMemoryArtifactStore:
    """Keeps artifacts in a dict. Bounded by ``max_items`` so a long-lived process cannot
    grow without limit (§65); the oldest entry is evicted first."""

    name = "memory"

    def __init__(self, max_items: int = 1000) -> None:
        self._items: dict[str, bytes] = {}
        self.max_items = max_items

    async def put(
        self,
        content: bytes | str,
        *,
        type: str = "blob",
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactRef:
        raw = content.encode("utf-8") if isinstance(content, str) else content
        ref = _ref(raw, type=type, mime_type=mime_type, metadata=metadata, key=idempotency_key)
        if ref.artifact_id not in self._items and len(self._items) >= self.max_items:
            self._items.pop(next(iter(self._items)))
        self._items[ref.artifact_id] = raw
        return ref

    async def get(self, artifact_id: str) -> bytes | None:
        return self._items.get(artifact_id)


class FileArtifactStore:
    """Writes artifacts under a directory, named by artifact id."""

    name = "file"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def put(
        self,
        content: bytes | str,
        *,
        type: str = "blob",
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactRef:
        raw = content.encode("utf-8") if isinstance(content, str) else content
        ref = _ref(raw, type=type, mime_type=mime_type, metadata=metadata, key=idempotency_key)
        path = self.root / ref.artifact_id
        if not path.exists():  # content-addressed: identical content is written once
            path.write_bytes(raw)
        return ref.model_copy(update={"uri": path.as_uri()})

    async def get(self, artifact_id: str) -> bytes | None:
        path = self.root / artifact_id
        return path.read_bytes() if path.exists() else None


class NoArtifactStore:
    """Artifacts disabled: putting one raises rather than silently losing data."""

    name = "none"

    async def put(self, content: bytes | str, **kwargs: Any) -> ArtifactRef:
        raise ConfigurationError("artifacts are disabled (harness.artifacts.enabled=false)")

    async def get(self, artifact_id: str) -> bytes | None:
        return None


def _ref(
    raw: bytes,
    *,
    type: str,
    mime_type: str | None,
    metadata: Mapping[str, Any] | None,
    key: str | None,
) -> ArtifactRef:
    checksum = hashlib.sha256(raw).hexdigest()
    return ArtifactRef(
        artifact_id=key or stable_id(checksum, type, prefix="art_"),
        type=type,
        mime_type=mime_type,
        checksum=f"sha256:{checksum}",
        size_bytes=len(raw),
        created_at=datetime.now(UTC),
        metadata=dict(metadata or {}),
    )
