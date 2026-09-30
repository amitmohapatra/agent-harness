"""An in-process memory service with the SDK's shape: every call the harness makes lands in
``calls``, and the answers are whatever a test sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trellis.memory import MemoryClient  # noqa: F401 - the shape imitated


@dataclass
class Bundle:
    rendered: str


@dataclass
class Report:
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    borderline: int = 0
    judge_consulted: int = 0


AGENT_TOOLS = [
    {
        "name": "memory_search",
        "description": "Search memories.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "memory_remember",
        "description": "Remember a fact.",
        "input_schema": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    },
]


@dataclass
class FakeMemoryService:
    context_text: str = "The user prefers email."
    catalog: dict[str, str] = field(default_factory=dict)
    report: Report | None = None
    fail: set[str] = field(default_factory=set)
    calls: list[tuple[str, dict[str, Any], Any]] = field(default_factory=list)

    def bind(self, **scope: Any) -> FakeContext:
        return FakeContext(self, scope)

    async def aclose(self) -> None:
        return None

    def named(self, name: str) -> list[tuple[dict[str, Any], Any]]:
        return [(scope, payload) for call, scope, payload in self.calls if call == name]


class FakeContext:
    def __init__(self, service: FakeMemoryService, scope: dict[str, Any]) -> None:
        self.service = service
        self.scope = type("Scope", (), scope)()
        self._scope = scope
        self.chat = _Chat(self)
        self.advanced = _Advanced(self)

    def log(self, name: str, payload: Any) -> None:
        if name in self.service.fail:
            raise ConnectionError(f"{name} is down")
        self.service.calls.append((name, self._scope, payload))

    async def context(self, query: str, **options: Any) -> Bundle:
        self.log("context", {"query": query, **options})
        return Bundle(self.service.context_text)

    async def agent_tools(self) -> list[dict[str, Any]]:
        self.log("agent_tools", None)
        return AGENT_TOOLS

    async def call_agent_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.log("call_agent_tool", {"name": name, "args": args})
        return {"result": [f"{name} ok"]}

    async def tool_hints(self, task: str, **options: Any) -> Any:
        self.log("tool_hints", {"task": task, **options})
        return {"candidates": options.get("available", [])[:1]}

    async def record_tool(self, tool: str, args: dict[str, Any], **fields: Any) -> None:
        self.log("record_tool", {"tool": tool, "args": args, **fields})

    async def outcome(self, **fields: Any) -> None:
        self.log("outcome", fields)

    async def feedback(self, feedback: Any) -> None:
        self.log("feedback", feedback)

    async def verify(self, answer: str, **options: Any) -> Any:
        self.log("verify", {"answer": answer, **options})
        return self.service.report


class _Chat:
    def __init__(self, ctx: FakeContext) -> None:
        self.ctx = ctx

    async def user(self, content: str, **fields: Any) -> None:
        self.ctx.log("message", {"role": "user", "content": content, **fields})

    async def assistant(self, content: str, **fields: Any) -> None:
        self.ctx.log("message", {"role": "assistant", "content": content, **fields})


class _Advanced:
    def __init__(self, ctx: FakeContext) -> None:
        self.tools = _Catalog(ctx)
        self.model_keys = _Keys(ctx)


class _Catalog:
    def __init__(self, ctx: FakeContext) -> None:
        self.ctx = ctx

    async def catalog(self, names: list[str]) -> list[dict[str, Any]]:
        self.ctx.log("catalog", names)
        known = self.ctx.service.catalog
        return [{"name": n, "side_effects": known[n]} for n in names if n in known]

    async def put_catalog(self, entries: list[dict[str, Any]]) -> None:
        self.ctx.log("put_catalog", entries)


class _Keys:
    def __init__(self, ctx: FakeContext) -> None:
        self.ctx = ctx

    async def set(self, key: str, **fields: Any) -> None:
        self.ctx.log("model_key", {"key": key, **fields})
