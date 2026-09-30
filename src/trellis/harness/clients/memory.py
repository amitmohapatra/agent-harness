"""The memory service, as the harness uses it: every call the harness makes goes through here.

``Memory`` is the process's client; ``Memory.bind(identity)`` is a :class:`RunMemory`, the
calls one run makes in its own scope. The SDK's ``MemoryContext`` it wraps (``.ctx``) is also
what tools and nodes get as ``trellis.current().memory``: the harness adds the push context
rendered for a prompt, the pull tools as tool specs, and the records the pipeline queues.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from trellis.contracts import Feedback, ToolCall, ToolOutcome, ToolSpec
from trellis.memory import MemoryClient

from trellis.harness.identity import Identity

#: Prompt budget for the pushed context, in tokens.
CONTEXT_TOKEN_BUDGET: Final = 2000
#: Tool candidates asked for when ``tool_hints`` is on.
TOOL_HINTS_K: Final = 8
#: Pull tools a ``memory="read"`` agent gets: the ones that change nothing.
READ_ONLY_TOOLS: Final = frozenset(
    {"memory_search", "history_search", "procedures_search", "tool_search"}
)
#: What the catalog may call a tool's side effects.
SIDE_EFFECTS: Final = frozenset({"read", "write", "irreversible"})


@dataclass(frozen=True, slots=True)
class Pushed:
    """What ``/v1/context`` returned: the text a prompt gets, and the bundle a judge verifies
    the answer against."""

    text: str
    bundle: Any


class Memory:
    """One memory service for the process."""

    def __init__(self, url: str, api_key: str | None, *, client: Any = None) -> None:
        self.client = client or MemoryClient(url, api_key=api_key)
        #: the agent-tool listing: a fixed set, listed once per process
        self.listed: list[ToolSpec] | None = None

    def bind(self, identity: Identity) -> RunMemory:
        return RunMemory(self, self.client.bind(**identity.scope()))

    async def aclose(self) -> None:
        await self.client.aclose()


class RunMemory:
    """The calls one run makes, in its scope."""

    __slots__ = ("ctx", "memory")

    def __init__(self, memory: Memory, ctx: Any) -> None:
        self.memory = memory
        self.ctx = ctx

    # ------------------------------------------------------------------ push
    async def context(self, query: str, *, tools: Sequence[str] | None) -> Pushed:
        options: dict[str, Any] = {"token_budget": CONTEXT_TOKEN_BUDGET}
        if tools is not None:
            options["tools"] = {"available": list(tools), "k": TOOL_HINTS_K}
        bundle = await self.ctx.context(query, **options)
        return Pushed(text=str(getattr(bundle, "rendered", "") or ""), bundle=bundle)

    # ------------------------------------------------------------------ pull
    async def agent_tools(self, *, read_only: bool) -> list[ToolSpec]:
        if self.memory.listed is None:
            self.memory.listed = [_agent_tool(t) for t in await self.ctx.agent_tools()]
        return [t for t in self.memory.listed if not read_only or t.name in READ_ONLY_TOOLS]

    async def call_agent_tool(self, name: str, args: dict[str, Any]) -> Any:
        return await self.ctx.call_agent_tool(name, args)

    async def tool_hints(self, task: str, available: Sequence[str]) -> Any:
        return await self.ctx.tool_hints(task, available=list(available), k=TOOL_HINTS_K)

    # ------------------------------------------------------------------ records
    async def record_messages(self, messages: Sequence[tuple[str, str]], run_id: str) -> None:
        """The run's transcript, once: each message keyed by run and position, so a retried
        write stores nothing twice."""
        for index, (role, content) in enumerate(messages):
            key = f"{run_id}:msg:{index}"
            if role == "user":
                await self.ctx.chat.user(content, idempotency_key=key)
            else:
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

    async def verify(self, answer: str, bundle: Any) -> Any:
        return await self.ctx.verify(answer, bundle=bundle)

    # ------------------------------------------------------------------ catalog
    async def side_effects(self, names: Sequence[str]) -> dict[str, str]:
        """What the catalog says each named tool does; tools it does not know are absent."""
        found: dict[str, str] = {}
        for entry in await self.ctx.advanced.tools.catalog(names=list(names)):
            effect = _field(entry, "side_effects")
            if effect in SIDE_EFFECTS:
                found[str(_field(entry, "name"))] = str(effect)
        return found

    async def publish_catalog(self, specs: Sequence[ToolSpec]) -> None:
        """Tools whose side effects the harness knows (local, OpenAPI, A2A)."""
        await self.ctx.advanced.tools.put_catalog(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema or {"type": "object"},
                    "side_effects": s.side_effects,
                    "source": s.source,
                    "server": s.server,
                }
                for s in specs
            ]
        )

    async def register_model_key(self, key: str, agent_id: str) -> None:
        """The agent-level LLM key the service uses for this agent's memory (idempotent)."""
        await self.ctx.advanced.model_keys.set(key, idempotency_key=f"model-key:{agent_id}")


def _agent_tool(entry: Any) -> ToolSpec:
    name = str(_field(entry, "name"))
    return ToolSpec(
        name=name,
        description=str(_field(entry, "description") or ""),
        input_schema=_field(entry, "input_schema") or {"type": "object"},
        source="memory",
        side_effects="read" if name in READ_ONLY_TOOLS else "write",
    )


def _field(entry: Any, name: str) -> Any:
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)
