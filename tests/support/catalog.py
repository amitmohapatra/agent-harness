"""A tool catalog in process, as governance reads it: conditional on an ETag (the rules'
version), or down; what was published to it and the decisions it was told."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trellis.contracts import Feedback
from trellis.harness.governance.catalog import Rule
from trellis.memory.errors import ValidationError


class FakeCatalog:
    def __init__(self, rules: dict[str, Rule] | None = None) -> None:
        self.rules = rules or {}
        self.asked: list[list[str]] = []
        self.etags_sent: list[str | None] = []
        self.published: list[list[dict[str, Any]]] = []
        self.feedback_sent: list[Feedback] = []
        self.version = 1
        self.down = False
        #: how many publishes are refused (not retried by the background writes)
        self.refuse_publish = 0

    async def read(
        self, names: Sequence[str], *, etag: str | None = None
    ) -> tuple[dict[str, Rule] | None, str | None]:
        self.asked.append(list(names))
        self.etags_sent.append(etag)
        if self.down:
            raise ConnectionError("memory is down")
        current = f'"v{self.version}"'
        if etag == current:
            return None, current
        return {n: r for n, r in self.rules.items() if n in names}, current

    async def publish(self, entries: Sequence[Mapping[str, object]]) -> None:
        if self.refuse_publish:
            self.refuse_publish -= 1
            raise ValidationError("not stored", retryable=False)
        self.published.append([dict(e) for e in entries])

    async def feedback(self, record: Feedback) -> None:
        self.feedback_sent.append(record)
