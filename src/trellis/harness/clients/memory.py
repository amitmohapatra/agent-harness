"""The memory service, as the harness uses it: every call the harness makes goes through here.

``Memory`` is the process's client; ``Memory.bind(identity)`` is a :class:`RunMemory`, the
calls one run makes in its own scope. The SDK's ``MemoryContext`` it wraps (``.ctx``) is also
what tools and nodes get as ``trellis.current().memory``: the harness adds the pull tools as
tool specs and the records the pipeline queues.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from trellis.contracts import Feedback, ToolCall, ToolOutcome, ToolSpec
from trellis.harness.identity import Identity
from trellis.memory import MemoryClient, MemoryContext
from trellis.memory.models import AgentTool, ContextBundle, GroundingReport, ToolHints

#: Prompt budget for the pushed context, in tokens.
CONTEXT_TOKEN_BUDGET: Final = 2000
#: Tool candidates asked for when ``tool_hints`` is on.
TOOL_HINTS_K: Final = 8
#: Pull tools a ``memory="read"`` agent gets: the ones that change nothing.
READ_ONLY_TOOLS: Final = frozenset(
    {"memory_search", "history_search", "procedures_search", "tool_search"}
)
#: The pull tool that says whether the run achieved its task (the harness then records none).
RECORD_OUTCOME: Final = "record_outcome"
#: The pull tool for tool hints: answered among the run's own tools.
TOOL_SEARCH: Final = "tool_search"
#: What the catalog may call a tool's side effects (anything else is left for it to learn).
SIDE_EFFECTS: Final = frozenset({"read", "write", "irreversible"})


class Memory:
    """One memory service for the process."""

    def __init__(
        self, url: str, api_key: str | None, *, client: MemoryClient | None = None
    ) -> None:
        self.client = client or MemoryClient(url, api_key=api_key)
        #: the agent-tool listing: a fixed set, listed once per process
        self.listed: list[ToolSpec] | None = None

    def bind(self, identity: Identity) -> RunMemory:
        return RunMemory(self, self.client.bind(**identity.scope()))

    def tenant(self, tenant: str) -> RunMemory:
        """Tenant-wide calls made outside a run (the tool catalog, tools built before a run)."""
        return RunMemory(self, self.client.bind(tenant_id=tenant))

    async def aclose(self) -> None:
        await self.client.aclose()


class RunMemory:
    """The calls one run makes, in its scope."""

    __slots__ = ("ctx", "memory")

    def __init__(self, memory: Memory, ctx: MemoryContext) -> None:
        self.memory = memory
        self.ctx = ctx

    # ------------------------------------------------------------------ push
    async def context(self, query: str, *, tools: Sequence[str] | None) -> ContextBundle:
        """What the prompt gets (``.rendered``), and what a judge verifies the answer against."""
        return await self.ctx.context(
            query,
            token_budget=CONTEXT_TOKEN_BUDGET,
            tools=None if tools is None else {"available": list(tools), "k": TOOL_HINTS_K},
        )

    # ------------------------------------------------------------------ pull
    async def agent_tools(self, *, read_only: bool) -> list[ToolSpec]:
        if self.memory.listed is None:
            self.memory.listed = [_agent_tool(t) for t in await self.ctx.agent_tools()]
        return [t for t in self.memory.listed if not read_only or t.name in READ_ONLY_TOOLS]

    async def call_agent_tool(self, name: str, args: dict[str, object]) -> object:
        return await self.ctx.call_agent_tool(name, args)

    async def tool_hints(self, task: str, available: Sequence[str]) -> ToolHints:
        return await self.ctx.tool_hints(task, available=list(available), k=TOOL_HINTS_K)

    # ------------------------------------------------------------------ records
    async def record_messages(
        self, messages: Sequence[tuple[str, str]], run_id: str, attempt: int
    ) -> None:
        """One attempt's transcript. The question is keyed by the run (every attempt asks
        it, and it is stored once); what the agent said, by attempt and position — so a
        retried write stores nothing twice."""
        for index, (role, content) in enumerate(messages):
            if role == "user":
                await self.ctx.chat.user(content, idempotency_key=f"{run_id}:user:{index}")
            else:
                key = f"{run_id}:{attempt}:msg:{index}"
                await self.ctx.chat.assistant(content, idempotency_key=key)

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

    async def outcome(self, *, success: bool, note: str | None) -> None:
        await self.ctx.outcome(success=success, note=note)

    async def feedback(self, feedback: Feedback) -> None:
        await self.ctx.feedback(feedback)

    async def verify(self, answer: str, bundle: ContextBundle) -> GroundingReport:
        return await self.ctx.verify(answer, bundle=bundle)

    # ------------------------------------------------------------------ catalog
    async def side_effects(self, names: Sequence[str]) -> dict[str, str]:
        """What the catalog says each named tool does; tools it does not know are absent."""
        return {
            entry.name: entry.side_effects
            for entry in await self.ctx.advanced.tools.catalog(names=list(names))
            if entry.side_effects is not None
        }

    async def publish_catalog(self, specs: Sequence[ToolSpec]) -> None:
        """Tools whose side effects the harness knows (local, OpenAPI, A2A). A tool whose
        effects are unknown goes in without them: the catalog learns them elsewhere."""
        await self.ctx.advanced.tools.put_catalog([_catalog_entry(s) for s in specs])

    async def register_model_key(self, key: str, agent_id: str) -> None:
        """The agent-level LLM key the service uses for this agent's memory (idempotent)."""
        await self.ctx.advanced.model_keys.set(key, idempotency_key=f"model-key:{agent_id}")


def _catalog_entry(spec: ToolSpec) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema or {"type": "object"},
        "source": spec.source,
        "server": spec.server,
    }
    if spec.side_effects in SIDE_EFFECTS:
        entry["side_effects"] = spec.side_effects
    return entry


def _agent_tool(tool: AgentTool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema or {"type": "object"},
        source="memory",
        side_effects="read" if tool.name in READ_ONLY_TOOLS else "write",
    )
