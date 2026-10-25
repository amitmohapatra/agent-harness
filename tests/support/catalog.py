"""A tool catalog in process, as governance reads it: conditional on an ETag (the rules'
version), or down; what was published to it and the decisions it was told."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trellis.contracts import Feedback
from trellis.harness.governance.catalog import Rule
from trellis.memory.errors import ValidationError
from trellis.memory.models import ApprovalSuggestion


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
        #: the approval suggestions learned, and the rule each writes when accepted
        self.suggested: list[ApprovalSuggestion] = []
        self.learned: dict[str, str] = {}
        self.accepted: list[str] = []
        #: whether reading (``"read"``) or accepting (``"accept"``) suggestions fails
        self.suggestions_fail: set[str] = set()
        #: the agents whose suggestions were read, in order
        self.suggestions_asked: list[str] = []

    def suggest(self, tool: str, rule: str, *, agent: str = "buyer") -> None:
        """A suggestion learned for ``tool`` from ``agent``'s decisions: accepted, ``rule`` is
        its ``approve_when``."""
        found = f"sug_{len(self.suggested) + 1}"
        self.suggested.append(
            ApprovalSuggestion(
                id=found,
                agent_id=agent,
                tool=tool,
                arg_shape="{}",
                suggestion="auto_approve",
                approvals=5,
                rejections=0,
                edits=0,
                support=5,
                approve_rate=1.0,
            )
        )
        self.learned[found] = rule

    async def suggestions(self, agent_id: str) -> Sequence[ApprovalSuggestion]:
        """The agent's own, as the service lists them (in its scope)."""
        self.suggestions_asked.append(agent_id)
        if "read" in self.suggestions_fail:
            raise ConnectionError("suggestions are down")
        return [
            s.model_copy(update={"accepted": s.id in self.accepted})
            for s in self.suggested
            if s.agent_id == agent_id
        ]

    async def accept(self, suggestion_id: str, *, agent_id: str) -> Rule:
        if "accept" in self.suggestions_fail:
            raise ConnectionError("accepting is down")
        found = next(s for s in self.suggested if s.id == suggestion_id)
        if found.agent_id != agent_id:
            raise LookupError("no such approval suggestion")  # the service's 404
        tool = found.tool
        self.accepted.append(suggestion_id)
        self.rules[tool] = Rule(self.rules[tool].risk, approve_when=self.learned[suggestion_id])
        self.version += 1
        return self.rules[tool]

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
