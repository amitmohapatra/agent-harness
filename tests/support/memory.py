"""The memory service's HTTP API, as far as the harness uses it, in process.

Tests talk to it through the real SDK (``FakeMemoryService.client()`` is a ``MemoryClient`` on
an ``httpx.MockTransport``), so what the harness sends and reads is exactly what the SDK sends
and reads. Request and response shapes are the service's (its ``docs/api``); every request
lands in ``calls``; the answers are whatever a test sets.
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
    {
        "name": "tool_search",
        "description": "Which tool to use next for a task.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
]

#: (method, path pattern) -> the name a call is filed under
ROUTES: Final = [
    ("GET", r"/v1/keys/self", "key"),
    ("POST", r"/v1/context", "context"),
    ("GET", r"/v1/agent-tools", "agent_tools"),
    ("POST", r"/v1/agent-tools/(?P<name>[^/]+)", "call_agent_tool"),
    ("POST", r"/v1/tools/hints", "tool_hints"),
    ("POST", r"/v1/tools/invocations", "record_tool"),
    ("POST", r"/v1/feedback", "feedback"),
    ("POST", r"/v1/verify", "verify"),
    ("POST", r"/v1/messages", "messages"),
    ("GET", r"/v1/tools", "catalog"),
    ("PUT", r"/v1/tools/catalog", "put_catalog"),
    ("PUT", r"/v1/agents/model-key", "model_key"),
    ("POST", r"/v1/documents", "add_document"),
    ("GET", r"/v1/documents/(?P<document_id>[^/]+)", "document"),
]


@dataclass(frozen=True)
class Call:
    name: str
    scope: dict[str, Any]
    body: Any
    idempotency_key: str | None
    #: the path's parameters (a tool name)
    path: dict[str, str]
    query: httpx.QueryParams


@dataclass
class FakeMemoryService:
    context_text: str = "The user prefers email."
    #: what the context says about its evidence; None leaves the field out (an older server)
    evidence_status: str | None = "COMPLETE"
    #: what ``GET /v1/keys/self`` says about the key
    tenant: str | None = "acme"
    role: str = "service"
    #: the catalog by tool name: risk / approve_when / annotations / side_effects
    catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: tool-hint candidates (names), in rank order; ``None``: the first available one
    candidates: list[str] | None = None
    #: the context answers without a ``tool_candidates`` field at all
    omit_candidates: bool = False
    #: candidates for a particular task, over ``candidates``
    candidates_for: dict[str, list[str]] = field(default_factory=dict)
    #: the claims ``/v1/verify`` finds, and how many of them the evidence does not support
    claims: int = 5
    unsupported: int = 1
    #: names of the calls that answer 503
    fail: set[str] = field(default_factory=set)
    #: names of the calls that answer 503 this many times, then succeed
    fail_times: dict[str, int] = field(default_factory=dict)
    #: feedback by id, each stored once however often it is sent
    stored_feedback: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: uploaded documents by id, with the scope they were uploaded in
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    agent_tools: list[dict[str, Any]] = field(default_factory=lambda: list(AGENT_TOOLS))
    _ids: itertools.count[int] = field(default_factory=itertools.count)
    #: the ``source_message_id``s already stored (the service's own message identity)
    _stored: set[str] = field(default_factory=set)

    def client(self, *, max_retries: int = 0) -> MemoryClient:
        transport = httpx.MockTransport(self._handle)
        return MemoryClient(
            URL,
            api_key="test",
            max_retries=max_retries,
            http_client=httpx.AsyncClient(base_url=URL, transport=transport),
        )

    def named(self, name: str) -> list[Call]:
        return [call for call in self.calls if call.name == name]

    # ------------------------------------------------------------------ the API
    def _handle(self, request: httpx.Request) -> httpx.Response:
        for method, pattern, name in ROUTES:
            matched = re.fullmatch(pattern, request.url.path)
            if request.method == method and matched:
                json_body = request.headers.get("content-type", "").startswith("application/json")
                body = json.loads(request.content) if request.content and json_body else None
                if not json_body and request.content:
                    # a multipart upload carries its scope as a JSON form field
                    found = re.search(rb'name="scope"\r\n\r\n(\{.*?\})\r\n', request.content)
                    body = {"scope": json.loads(found.group(1))} if found else None
                call = Call(
                    name=name,
                    scope=_scope(request, body),
                    body=body,
                    idempotency_key=request.headers.get("idempotency-key"),
                    path=matched.groupdict(),
                    query=request.url.params,
                )
                if name in self.fail or self.fail_times.get(name, 0) > 0:
                    if name in self.fail_times:
                        self.fail_times[name] -= 1
                    return httpx.Response(503, json={"title": f"{name} is down", "status": 503})
                self.calls.append(call)
                return httpx.Response(200, json=getattr(self, f"_{name}")(call))
        return httpx.Response(404, json={"title": "no such route", "status": 404})

    def _key(self, call: Call) -> dict[str, Any]:
        return {
            "key_id": "key_1",
            "tenant_id": self.tenant,
            "principal": "svc:harness",
            "role": self.role,
            "may_act_as": ["*"],
        }

    def _context(self, call: Call) -> dict[str, Any]:
        rendered = self.context_text
        available = (call.body.get("tools") or {}).get("available")
        answer: dict[str, Any] = {
            "rendered": rendered,
            "bundle_id": f"bnd_{next(self._ids)}",
            "token_estimate": 0,
        }
        if self.evidence_status is not None:
            answer["evidence_status"] = self.evidence_status
        if available is not None:
            chosen = self._candidates("", available)
            rendered += "\n\n## Tools\nnext: " + (chosen[0] if chosen else "-")
            if not self.omit_candidates:
                answer["tool_candidates"] = chosen
            answer["rendered"] = rendered
        answer["token_estimate"] = len(rendered) // 4
        return answer

    def _agent_tools(self, call: Call) -> dict[str, Any]:
        return {"tools": self.agent_tools}

    def _call_agent_tool(self, call: Call) -> dict[str, Any]:
        return {"result": [f"{call.path['name']} ok"]}

    def _candidates(self, task: str, available: list[str]) -> list[str]:
        names = self.candidates_for.get(
            task, self.candidates if self.candidates is not None else available[:1]
        )
        return [n for n in names if n in available][:8]

    def _tool_hints(self, call: Call) -> dict[str, Any]:
        available = call.body.get("available") or []
        chosen = self._candidates(call.body["task"], available)
        return {
            "candidates": [{"name": n, "score": 1.0 - i / 10} for i, n in enumerate(chosen)],
            "next": chosen[0] if chosen else None,
            "plan": None,
            "prefill": {},
            "missing": [],
        }

    def _record_tool(self, call: Call) -> dict[str, Any]:
        return {"invocation_id": self._id("inv"), "step": call.body.get("step") or 0}

    def _feedback(self, call: Call) -> dict[str, Any]:
        # as the service does: a feedback id already stored answers with the stored record
        feedback_id = call.body.get("feedback_id") or self._id("fb")
        if feedback_id not in self.stored_feedback:
            self.stored_feedback[feedback_id] = {
                "feedback_id": feedback_id,
                "created_at": datetime.now(UTC).isoformat(),
                **call.body,
                **({"review": {"state": "pending"}} if _waits_for_review(call) else {}),
            }
        return self.stored_feedback[feedback_id]

    def _add_document(self, call: Call) -> dict[str, Any]:
        document_id = self._id("doc")
        self.documents[document_id] = call.scope
        return {
            "document_id": document_id,
            "filename": "upload",
            "checksum": "0" * 64,
            "size_bytes": 1,
            "job_ids": [self._id("job")],
        }

    def _document(self, call: Call) -> dict[str, Any]:
        return {
            "document_id": call.path["document_id"],
            "title": "upload",
            "filename": "upload",
            "media_type": "text/plain",
            "size_bytes": 1,
            "checksum": "0" * 64,
            "status": "READY",
            "archive_status": "ARCHIVED",
        }

    def _verify(self, call: Call) -> dict[str, Any]:
        claims = [
            {
                "claim": f"claim {i}",
                "verdict": "supported" if i >= self.unsupported else "unsupported",
            }
            for i in range(self.claims)
        ]
        return {
            "claims": claims,
            "supported": self.claims - self.unsupported,
            "unsupported": self.unsupported,
            "per_claim_hallucination_rate": (
                self.unsupported / self.claims if self.claims else 0.0
            ),
            "feedback_id": self._id("fb"),
        }

    def _messages(self, call: Call) -> dict[str, Any]:
        """One durable append. A message the service has already stored under its
        ``source_message_id`` is acknowledged as ``deduplicated``."""
        thread = call.scope.get("thread_id") or call.scope.get("agent_run_id") or "thr_1"
        acks = []
        for message in call.body["messages"]:
            n = next(self._ids)
            source_id = message.get("source_message_id") or f"anon_{n}"
            acks.append(
                {
                    "message_id": f"msg_{n}",
                    "thread_id": thread,
                    "session_id": "ses_1",
                    "turn_id": f"trn_{n}",
                    "sequence": n,
                    "deduplicated": source_id in self._stored,
                }
            )
            self._stored.add(source_id)
        return {"messages": acks}

    def _catalog(self, call: Call) -> dict[str, Any]:
        names = call.query.get_list("names")
        return {
            "tools": [
                {"tool_id": f"tool_{n}", "name": n, "risk": "write", **self.catalog[n]}
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


def _waits_for_review(call: Call) -> bool:
    """The service's rule (its ADR 0028): a vote waits for the tenant administrator; a run's
    own status citing nothing, a decision on a tool call and an owner's edit of a memory
    do not."""
    body = call.body
    if body.get("target_kind") == "tool_call":
        return False
    if body.get("target_kind") == "memory" and body.get("verdict") not in ("confirm", "approve"):
        return False
    own_run = body.get("target_id") == (body.get("agent_run_id") or call.scope.get("agent_run_id"))
    return not (body.get("source") == "system" and own_run and not body.get("evidence_refs"))


def _scope(request: httpx.Request, body: Any) -> dict[str, Any]:
    """The scope a request carries: in its body, else in its headers and query."""
    if isinstance(body, dict) and isinstance(body.get("scope"), dict):
        return body["scope"]
    scope: dict[str, Any] = {k: v for k, v in request.url.params.items() if k != "names"}
    for header, name in (("x-trellis-tenant", "tenant_id"), ("x-trellis-user", "user_id")):
        if header in request.headers:
            scope[name] = request.headers[header]
    return scope
