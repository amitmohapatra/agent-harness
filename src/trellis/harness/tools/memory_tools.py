"""Memory as tools (design §5): ``memory.recall`` and ``memory.remember`` offered to the
model, so an agent can ask for more mid-loop instead of the harness guessing up front.

The interceptors still load context before the run and write observations after it; these
tools are the agentic addition, and they go through the same memory runtime, so the same
scope, policy and failure mode apply.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from trellis.contracts import ToolStatus
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.tool import ToolCall, ToolOutcome, ToolSpec

RECALL: Final = "memory.recall"
REMEMBER: Final = "memory.remember"
SOURCE: Final = "memory"
#: What the model calls a memory, mapped onto the service's memory types (a hint).
KINDS: Final[dict[str, str]] = {
    "fact": "SEMANTIC",
    "preference": "PREFERENCE",
    "decision": "DECISION",
    "task": "TASK",
}
#: One sentence, the tool says; a model that pastes a document in is cut here.
REMEMBER_MAX_CHARS: Final = 1000

SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name=RECALL,
        description="Search what is remembered about the user, the thread and the team; "
        "returns the most relevant memories with their ids and content.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look for"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
        },
        source=SOURCE,
    ),
    ToolSpec(
        name=REMEMBER,
        description="Remember something durable about the user or the task, in one sentence.",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "the fact, in one sentence"},
                "kind": {
                    "type": "string",
                    "enum": sorted(KINDS),
                    "description": "what it is",
                    "default": "fact",
                },
            },
            "required": ["content"],
        },
        source=SOURCE,
    ),
)


class MemoryToolClient:
    """Bound to one execution's memory runtime by ``attach``; empty until then."""

    name = "memory"

    def __init__(self, runtime: Any = None) -> None:
        self._runtime = runtime

    def attach(self, runtime: Any) -> None:
        self._runtime = runtime

    async def list_tools(self) -> Sequence[ToolSpec]:
        memory = getattr(self._runtime, "memory", None)
        return list(SPECS) if memory is not None and memory.enabled else []

    def spec(self, tool: str) -> ToolSpec | None:
        return next((s for s in SPECS if s.name == tool), None)

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        memory = self._runtime.memory if self._runtime is not None else None
        if memory is None or not memory.enabled:
            return ToolOutcome(
                tool=call.tool, status=ToolStatus.ERROR, error_class="MemoryDisabled"
            )
        if call.tool == RECALL:
            hits = await memory.recall(
                str(call.args.get("query", "")), limit=int(call.args.get("limit", 5))
            )
            return ToolOutcome(tool=call.tool, status=ToolStatus.OK, output=[_hit(h) for h in hits])
        if call.tool == REMEMBER:
            kind = str(call.args.get("kind", "fact")).lower()
            content = str(call.args.get("content", ""))[:REMEMBER_MAX_CHARS]
            observation = MemoryObservation(
                content=content,
                kind="AGENT_RESULT",
                hints={"memory_type": KINDS.get(kind, KINDS["fact"])},
                # provenance: the model asked for this, through a tool, from this run
                metadata={"source": REMEMBER},
            )
            ack = await memory.observe(observation)
            return ToolOutcome(
                tool=call.tool,
                status=ToolStatus.OK if ack is not None else ToolStatus.ERROR,
                output={"remembered": ack is not None},
            )
        return ToolOutcome(tool=call.tool, status=ToolStatus.ERROR, error_class="UnknownTool")


def _hit(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        return {k: hit.get(k) for k in ("memory_id", "content", "memory_type", "score") if k in hit}
    return {
        "memory_id": getattr(hit, "memory_id", None),
        "content": getattr(hit, "content", None),
        "memory_type": getattr(hit, "memory_type", None),
        "score": getattr(hit, "score", None),
    }
