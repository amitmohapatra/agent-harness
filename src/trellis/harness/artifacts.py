"""Payloads too large to travel inside a question (``ask(table=...)``), kept by reference.

The contracts ``ArtifactClient`` port, in this process: an interrupt carries an
``ArtifactRef`` and the chat surface serves the content at ``{path}/artifacts/{id}``. Bounded:
the oldest artifacts are dropped past :data:`MAX_ARTIFACTS`.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Final

from trellis.contracts import ArtifactRef, new_id

MAX_ARTIFACTS: Final = 1024


class Artifacts:
    def __init__(self) -> None:
        self._items: OrderedDict[str, tuple[bytes, str]] = OrderedDict()

    async def put(
        self,
        content: bytes | str,
        *,
        type: str = "blob",
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactRef:
        data = content.encode() if isinstance(content, str) else content
        artifact_id = idempotency_key or new_id("art_")
        mime = mime_type or "application/octet-stream"
        self._items[artifact_id] = (data, mime)
        while len(self._items) > MAX_ARTIFACTS:
            self._items.popitem(last=False)
        return ArtifactRef(
            artifact_id=artifact_id,
            type=type,
            mime_type=mime,
            size_bytes=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
            metadata=dict(metadata or {}),
        )

    async def get(self, artifact_id: str) -> bytes | None:
        found = self._items.get(artifact_id)
        return found[0] if found else None

    def mime_type(self, artifact_id: str) -> str | None:
        found = self._items.get(artifact_id)
        return found[1] if found else None

    async def put_json(self, value: Any) -> ArtifactRef:
        return await self.put(
            json.dumps(value, default=str), type="table", mime_type="application/json"
        )
