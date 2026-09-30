"""The memory service, as the harness uses it: every call the harness makes goes through here.

``Memory`` is the process's client; ``Memory.bind(identity)`` is a :class:`RunMemory`, the
calls one run makes in its own scope. The SDK's ``MemoryContext`` it wraps (``.ctx``) is also
what tools and nodes get as ``trellis.current().memory``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from trellis.contracts import ToolCall, ToolOutcome, ToolSpec
from trellis.harness.identity import Identity
from trellis.memory import MemoryClient, MemoryContext
from trellis.memory.models import AgentTool, KeyInfo, PromptContext, SideEffects

#: Prompt budget for the pushed context, in tokens.
CONTEXT_TOKEN_BUDGET: Final = 2000
#: Pull tools a read-only key gets: the ones that change nothing.
READ_ONLY_TOOLS: Final = frozenset({"memory_search", "tool_search"})
#: The pull tool that chooses among the run's own tools (which the harness passes).
TOOL_SEARCH: Final = "tool_search"
#: Key roles that may read memory but not change it (``GET /v1/keys/self``).
READ_ONLY_ROLES: Final = frozenset({"reader"})
#: What the harness calls the transcript it writes, so a re-recorded message is stored once.
SOURCE_SYSTEM: Final = "trellis-harness"


def read_only(key: KeyInfo) -> bool:
    return key.role in READ_ONLY_ROLES


@dataclass(frozen=True, slots=True)
class Governance:
    """The catalog's word on one tool: the tier it decided, and the approval rule an
    administrator (or an accepted suggestion) set."""

    risk: SideEffects
    approve_when: str | None = None


class Memory:
    """One memory service for the process."""

    def __init__(
        self, url: str, api_key: str | None, *, client: MemoryClient | None = None
    ) -> None:
        self.client = client or MemoryClient(url, api_key=api_key)
        #: the agent-tool listing: a fixed set, listed once per process
        self.listed: list[ToolSpec] | None = None

    async def key(self) -> KeyInfo:
        """Who ``TRELLIS_API_KEY`` is: its tenant, principal and role."""
        return await self.client.tenant.keys.whoami()

    def bind(self, identity: Identity) -> RunMemory:
        return RunMemory(self, self.client.bind(**identity.scope()))

    def scoped(self, tenant: str, agent_id: str | None = None) -> RunMemory:
        """Calls made outside a run: tenant-wide (the tool catalog) or for one agent (its
        model key)."""
        scope = {"tenant_id": tenant} | ({"agent_id": agent_id} if agent_id else {})
        return RunMemory(self, self.client.bind(**scope))

    async def aclose(self) -> None:
        await self.client.aclose()


class RunMemory:
    """The calls one run makes, in its scope."""

    __slots__ = ("ctx", "memory")

    def __init__(self, memory: Memory, ctx: MemoryContext) -> None:
        self.memory = memory
        self.ctx = ctx

    # ------------------------------------------------------------------ push
    async def context(
        self, query: str, *, tools: Sequence[str] | None, window: bool
    ) -> PromptContext:
        """What the prompt gets, and the tools that fit the task (``tool_candidates``, when
        ``tools`` are given). ``window=False`` when the framework keeps the thread's messages
        itself: the service then leaves the recent conversation out."""
        return await self.ctx.context(
            query, token_budget=CONTEXT_TOKEN_BUDGET, tools=tools, window=window
        )

    # ------------------------------------------------------------------ pull
    async def agent_tools(self, *, read_only: bool) -> list[ToolSpec]:
        if self.memory.listed is None:
            self.memory.listed = [_agent_tool(t) for t in await self.ctx.agent_tools()]
        return [t for t in self.memory.listed if not read_only or t.name in READ_ONLY_TOOLS]

    async def call_agent_tool(
        self, name: str, args: dict[str, object], *, toolbox: Sequence[str] | None = None
    ) -> object:
        """One memory tool, in this run's scope; ``toolbox`` is what ``tool_search`` chooses
        among (the run's own tools)."""
        return await self.ctx.call_agent_tool(name, args, toolbox=toolbox)

    async def tool_hints(self, task: str, available: Sequence[str]) -> Any:
        return await self.ctx.tool_hints(task, available=list(available))

    # ------------------------------------------------------------------ records
    async def record_messages(
        self, messages: Sequence[tuple[str, str]], run_id: str, attempt: int
    ) -> None:
        """One attempt's transcript, in one request. Each message names itself (the run for
        the question, which every attempt asks; the attempt and position for what the agent
        said), and the service stores a message it has seen before once."""
        batch = [
            {
                "role": role.upper(),
                "content": content,
                "source_system": SOURCE_SYSTEM,
                "source_message_id": f"{run_id}:user:{index}"
                if role == "user"
                else f"{run_id}:{attempt}:msg:{index}",
            }
            for index, (role, content) in enumerate(messages)
        ]
        await self.ctx.history.add(batch, idempotency_key=f"{run_id}:{attempt}:transcript")

    async def record_tool(self, call: ToolCall, outcome: ToolOutcome) -> None:
        await self.ctx.record_tool(
            call.tool,
            call.args,
            output=outcome.output,
            status=outcome.status.value,
            error_class=outcome.error_class,
            latency_ms=outcome.latency_ms,
            task=call.task,
            step=call.step,
        )

    async def run_feedback(
        self,
        verdict: Any,
        *,
        source: Any,
        key: str,
        correction: Any = None,
        comment: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        """A verdict on this scope's run (``source`` ``system`` for how it ended, ``human``
        for a person's); ``key`` makes a retry store it once."""
        run_id = self.ctx.scope.agent_run_id
        assert run_id is not None
        await self.ctx.feedback(
            "run",
            run_id,
            verdict,
            correction=correction,
            comment=comment,
            reviewer=reviewer,
            source=source,
            idempotency_key=key,
        )

    async def feedback(self, record: Any) -> None:
        """A contracts ``Feedback`` record as it is (an interrupt's decision)."""
        await self.ctx.feedback(record)

    async def verify(self, answer: str, bundle_id: str) -> float | None:
        """The grounding score of ``answer`` against the context the run was given (the share
        of its claims the evidence supports), or ``None`` for an answer with no checkable
        claim. The service records the verdict as the run's ``judge`` feedback itself; this is
        the same number, for the run's trace."""
        report = await self.ctx.verify(answer, bundle_id=bundle_id)
        if not report.claims:
            return None
        return round(1.0 - report.per_claim_hallucination_rate, 4)

    # ------------------------------------------------------------------ catalog
    async def catalog(self, names: Sequence[str]) -> dict[str, Governance]:
        """What the catalog says about each named tool: the tier it decided, and the rule an
        administrator (or an accepted suggestion) set. Tools it does not know are absent."""
        return {
            entry.name: Governance(risk=entry.risk, approve_when=entry.approve_when or None)
            for entry in await self.ctx.advanced.tools.catalog(names=list(names))
        }

    async def publish_catalog(self, entries: Sequence[dict[str, object]]) -> None:
        await self.ctx.advanced.tools.put_catalog(list(entries))

    async def register_model_key(self, key: str) -> None:
        """The agent-level LLM key the service uses for this agent's memory, in an agent
        scope (the same request from every process and run, so the idempotent PUT repeats
        rather than conflicts; a rotated key is a new request)."""
        agent = self.ctx.scope.agent_id
        digest = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
        await self.ctx.advanced.model_keys.set(key, idempotency_key=f"model-key:{agent}:{digest}")


def catalog_entry(spec: ToolSpec, annotations: dict[str, bool] | None) -> dict[str, object]:
    """A tool as the catalog stores it. ``side_effects`` only where the harness knows them (a
    local tool declares them, an OpenAPI method implies them); an MCP tool sends its server's
    annotations instead and the service derives the tier, so an administrator's stays."""
    entry: dict[str, object] = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema or {"type": "object"},
        "source": spec.source,
        "server": spec.server,
    }
    if spec.source != "mcp" and spec.side_effects in ("read", "write", "irreversible"):
        entry["side_effects"] = spec.side_effects
    if annotations:
        entry["annotations"] = annotations
    return entry


def _agent_tool(tool: AgentTool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema or {"type": "object"},
        source="memory",
        side_effects="read" if tool.name in READ_ONLY_TOOLS else "write",
    )
