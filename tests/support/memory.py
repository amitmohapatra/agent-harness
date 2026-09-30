"""The memory service's HTTP API, as far as the harness uses it, in process.

Tests talk to it through the real SDK (``FakeMemoryService.client()`` is a ``MemoryClient``
on an ``httpx.MockTransport``), so what the harness sends and reads is exactly what the SDK
sends and reads. Every request lands in ``calls``; the answers are whatever a test sets.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

import httpx

from trellis.memory import MemoryClient
from trellis.memory.models import GroundingReport

URL: Final = "http://memory.test"

AGENT_TOOLS: Final = [
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

#: (method, path pattern) -> the name a call is filed under
ROUTES: Final = [
    ("POST", r"/v1/context", "context"),
    ("GET", r"/v1/agent-tools", "agent_tools"),
    ("POST", r"/v1/agent-tools/(?P<name>[^/]+)", "call_agent_tool"),
    ("POST", r"/v1/tools/hints", "tool_hints"),
    ("POST", r"/v1/tools/invocations", "record_tool"),
    ("POST", r"/v1/runs/(?P<run>[^/]+)/outcome", "outcome"),
    ("POST", r"/v1/feedback", "feedback"),
    ("POST", r"/v1/verify", "verify"),
    ("POST", r"/v1/messages", "message"),
    ("GET", r"/v1/tools", "catalog"),
    ("PUT", r"/v1/tools/catalog", "put_catalog"),
    ("PUT", r"/v1/agents/model-key", "model_key"),
]


@dataclass(frozen=True)
class Call:
    name: str
    scope: dict[str, Any]
    body: Any
    idempotency_key: str | None
    #: the path's parameters (a tool name, a run id)
    path: dict[str, str]
    query: httpx.QueryParams


@dataclass
class FakeMemoryService:
    context_text: str = "The user prefers email."
    #: the catalog's side effects by tool name
    catalog: dict[str, str] = field(default_factory=dict)
    report: GroundingReport = field(default_factory=GroundingReport)
    #: names of the calls that answer 503
    fail: set[str] = field(default_factory=set)
    calls: list[Call] = field(default_factory=list)
    _ids: itertools.count[int] = field(default_factory=itertools.count)

    def client(self) -> MemoryClient:
        transport = httpx.MockTransport(self._handle)
        return MemoryClient(
            URL,
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(base_url=URL, transport=transport),
        )

    def named(self, name: str) -> list[Call]:
        return [call for call in self.calls if call.name == name]

    # ------------------------------------------------------------------ the API
    def _handle(self, request: httpx.Request) -> httpx.Response:
        for method, pattern, name in ROUTES:
            matched = re.fullmatch(pattern, request.url.path)
            if request.method == method and matched:
                body = json.loads(request.content) if request.content else None
                call = Call(
                    name=name,
                    scope=_scope(request, body),
                    body=body,
                    idempotency_key=request.headers.get("idempotency-key"),
                    path=matched.groupdict(),
                    query=request.url.params,
                )
                if name in self.fail:
                    return httpx.Response(503, json={"title": f"{name} is down", "status": 503})
                self.calls.append(call)
                return httpx.Response(200, json=getattr(self, f"_{name}")(call))
        return httpx.Response(404, json={"title": "no such route", "status": 404})

    def _context(self, call: Call) -> dict[str, Any]:
        return {
            "query": call.body["query"],
            "query_type": "GENERAL_SEMANTIC",
            "conversation": {},
            "evidence": {"status": "COMPLETE"},
            "token_budget": call.body.get("token_budget", 0),
            "token_estimate": len(self.context_text) // 4,
            "rendered": self.context_text,
        }

    def _agent_tools(self, call: Call) -> dict[str, Any]:
        return {"tools": AGENT_TOOLS}

    def _call_agent_tool(self, call: Call) -> dict[str, Any]:
        return {"result": [f"{call.path['name']} ok"]}

    def _tool_hints(self, call: Call) -> dict[str, Any]:
        available = call.body.get("available") or []
        return {"candidates": [{"name": n, "score": 1.0} for n in available[:1]]}

    def _record_tool(self, call: Call) -> dict[str, Any]:
        return {"invocation_id": self._id("inv"), "step": call.body.get("step") or 0}

    def _outcome(self, call: Call) -> dict[str, Any]:
        return {"run_id": call.path["run"], "success": call.body["success"], "source": "outcome"}

    def _feedback(self, call: Call) -> dict[str, Any]:
        return {
            "feedback_id": call.body.get("feedback_id") or self._id("fb"),
            "created_at": datetime.now(UTC).isoformat(),
            **call.body,
        }

    def _verify(self, call: Call) -> dict[str, Any]:
        return self.report.model_dump(mode="json")

    def _message(self, call: Call) -> dict[str, Any]:
        n = next(self._ids)
        return {
            "message_id": f"msg_{n}",
            "thread_id": call.scope.get("thread_id") or "thr_1",
            "session_id": "ses_1",
            "turn_id": f"turn_{n}",
            "sequence": n,
        }

    def _catalog(self, call: Call) -> dict[str, Any]:
        names = call.query.get_list("names")
        return {
            "tools": [
                {"tool_id": f"tool_{n}", "name": n, "side_effects": self.catalog[n]}
                for n in names
                if n in self.catalog
            ]
        }

    def _put_catalog(self, call: Call) -> dict[str, Any]:
        return {"tools": [{"tool_id": f"tool_{t['name']}", **t} for t in call.body["tools"]]}

    def _model_key(self, call: Call) -> dict[str, Any]:
        return {"registered": True, "revoked": False, "revision": 1}

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids)}"


def _scope(request: httpx.Request, body: Any) -> dict[str, Any]:
    """The scope a request carries: in its body, else in its headers and query."""
    if isinstance(body, dict) and isinstance(body.get("scope"), dict):
        return body["scope"]
    scope: dict[str, Any] = {k: v for k, v in request.url.params.items() if k != "names"}
    for header, name in (("x-trellis-tenant", "tenant_id"), ("x-trellis-user", "user_id")):
        if header in request.headers:
            scope[name] = request.headers[header]
    return scope
