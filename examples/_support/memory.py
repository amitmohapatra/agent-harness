"""A scripted memory service, in process: what the examples remember with when ``MEMORY_URL`` is
unset.

It answers the routes the harness and the memory SDK call, in the service's shapes, from a few
fixed facts: the context it pushes is :attr:`ScriptedMemory.context`, a search finds
:attr:`ScriptedMemory.facts`, the tool catalog says what :attr:`ScriptedMemory.catalog` says,
and every write is kept in :attr:`ScriptedMemory.written` so an example can show what was
recorded. It is a stand-in for running the service (``docker compose up`` in
agent-memory-service), never a model of how it ranks or learns.

    memory = ScriptedMemory(context="Ada prefers email.")
    async with Harness(memory=memory.client()) as h: ...
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from trellis.memory import MemoryClient

URL: Final = "http://memory.offline"
#: The memory tools as the service lists them (names, descriptions, input schemas).
AGENT_TOOLS: Final[list[dict[str, Any]]] = json.loads(
    (Path(__file__).with_name("memory_tools.json")).read_text()
)
ROUTES: Final = [
    ("GET", r"/v1/keys/self", "key"),
    ("POST", r"/v1/context", "context_"),
    ("POST", r"/v1/recall", "recall"),
    ("POST", r"/v1/memories", "remember"),
    ("GET", r"/v1/agent-tools", "agent_tools"),
    ("POST", r"/v1/agent-tools/(?P<name>[^/]+)", "call_tool"),
    ("POST", r"/v1/tools/hints", "hints"),
    ("POST", r"/v1/tools/invocations", "record_tool"),
    ("POST", r"/v1/feedback", "feedback"),
    ("POST", r"/v1/verify", "verify"),
    ("POST", r"/v1/messages", "messages"),
    ("GET", r"/v1/tools", "read_catalog"),
    ("PUT", r"/v1/tools/catalog", "publish"),
    ("PUT", r"/v1/agents/model-key", "model_key"),
    ("POST", r"/v1/documents", "add_document"),
    ("GET", r"/v1/documents/(?P<document_id>[^/]+)", "document"),
]
#: the uploaded file's bytes in a multipart body
FILE_PART: Final = rb'filename="[^"]*"\r\n(?:[^\r]*\r\n)*\r\n(.*?)\r\n--'
CREATED: Final = {
    "messages": 202,
    "record_tool": 202,
    "add_document": 202,
    "feedback": 201,
    "remember": 201,
}


@dataclass
class ScriptedMemory:
    #: what the pushed context says about the user and the task
    context: str = "The user prefers email and short answers."
    #: what ``memory_search`` (the model's tool, or ``agent.memory.search``) finds
    facts: list[str] = field(default_factory=lambda: ["The user prefers email."])
    #: the tool catalog by tool name: ``risk`` (read, write, irreversible), ``approve_when``
    catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    tenant: str = "default"
    #: every write, by route: ``messages``, ``record_tool``, ``feedback``, ``add_document``...
    written: Counter[str] = field(default_factory=Counter)
    #: every call, by route: ``context_`` (a pushed context), ``recall``, ``read_catalog``...
    asked: Counter[str] = field(default_factory=Counter)
    documents: dict[str, str] = field(default_factory=dict)
    _ids: itertools.count[int] = field(default_factory=itertools.count)

    def client(self) -> MemoryClient:
        """The memory SDK's client, talking to this service."""
        transport = httpx.MockTransport(self.handle)
        return MemoryClient(
            URL,
            api_key="offline",
            max_retries=0,
            http_client=httpx.AsyncClient(base_url=URL, transport=transport),
        )

    # ------------------------------------------------------------------ the routes
    def handle(self, request: httpx.Request) -> httpx.Response:
        for method, pattern, name in ROUTES:
            matched = re.fullmatch(pattern, request.url.path)
            if request.method == method and matched:
                json_body = request.headers.get("content-type", "").startswith("application/json")
                body = json.loads(request.content) if request.content and json_body else {}
                self.asked[name] += 1
                answer = getattr(self, f"_{name}")(body, request, **matched.groupdict())
                if name in CREATED:
                    self.written[name] += 1
                return httpx.Response(CREATED.get(name, 200), json=answer)
        return httpx.Response(404, json={"code": "NOT_FOUND", "status": 404, "title": "Not found"})

    def _key(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {
            "key_id": "key_offline",
            "tenant_id": self.tenant,
            "principal": "svc:examples",
            "role": "service",
            "may_act_as": ["*"],
        }

    def _context_(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        rendered = self.context
        answer: dict[str, Any] = {"bundle_id": self._id("bnd"), "evidence_status": "COMPLETE"}
        available = (body.get("tools") or {}).get("available")
        if available:  # the tool hints: here, every tool fits
            answer["tools"] = [{"name": n, "confidence": 0.9} for n in available[:8]]
        answer |= {"rendered": rendered, "token_estimate": len(rendered) // 4}
        return answer

    def _recall(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {"items": self._found()}

    def _remember(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        self.facts.append(str(body.get("content", "")))
        return {"memory_id": self._id("mem"), "deduplicated": False, "job_ids": []}

    def _found(self) -> list[dict[str, Any]]:
        return [
            {"id": f"mem_{n}", "kind": "memory", "text": fact, "observed_on": "2026-10-01"}
            for n, fact in enumerate(self.facts)
        ]

    def _agent_tools(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {"tools": AGENT_TOOLS}

    def _call_tool(self, body: Any, request: httpx.Request, name: str) -> dict[str, Any]:
        args = body.get("args") or {}
        if name == "memory_search":
            return {"result": self._found()}
        if name == "tool_search":
            return {"result": {"tools": []}}
        if name == "memory_remember":
            self.facts.append(str(args.get("content", "")))
        self.written[name] += 1
        return {"result": {"id": self._id("mem"), "stored": True}}

    def _hints(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {"tools": []}

    def _record_tool(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {"invocation_id": self._id("inv"), "step": 0, "args_hash": "0", "recorded": True}

    def _feedback(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        scope = body.get("scope") or {}
        record: dict[str, Any] = {
            "feedback_id": body.get("feedback_id") or self._id("fb"),
            "tenant_id": self.tenant,
            "user_id": scope.get("user_id"),
            "agent_id": scope.get("agent_id"),
            "agent_run_id": body.get("agent_run_id") or scope.get("agent_run_id"),
            "target_kind": body["target_kind"],
            "target_id": body["target_id"],
            "verdict": body["verdict"],
            "correction": body.get("correction"),
            "comment": body.get("comment"),
            "reviewer": body.get("reviewer"),
            "source": body.get("source") or "human",
            "evidence_refs": [],
            "metadata": {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        if record["source"] == "human" and record["target_kind"] == "run":
            record["review"] = {"state": "pending"}  # a person's verdict waits for the admin
        return record

    def _verify(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        claims = [{"claim": "the answer", "verdict": "supported"}]
        return {
            "claims": claims,
            "supported": 1,
            "unsupported": 0,
            "per_claim_hallucination_rate": 0.0,
            "feedback_id": self._id("fb"),
        }

    def _messages(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        acks = []
        for _ in body.get("messages") or []:
            n = next(self._ids)
            acks.append(
                {
                    "message_id": f"msg_{n}",
                    "thread_id": "thr_offline",
                    "session_id": "ses_offline",
                    "turn_id": f"trn_{n}",
                    "sequence": n,
                    "deduplicated": False,
                }
            )
        return {"messages": acks}

    def _read_catalog(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        names = request.url.params.get_list("names")
        return {"tools": [_entry(n, self.catalog[n]) for n in names if n in self.catalog]}

    def _publish(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {"tools": [_entry(t["name"], t) for t in body.get("tools") or []]}

    def _model_key(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        return {"registered": True, "revoked": False, "revision": 1}

    def _add_document(self, body: Any, request: httpx.Request) -> dict[str, Any]:
        document_id = self._id("doc")
        found = re.search(FILE_PART, request.content, re.S)
        self.documents[document_id] = found.group(1).decode(errors="replace") if found else ""
        self.context += f"\n\n## Documents\n{self.documents[document_id]}"
        return {
            "document_id": document_id,
            "filename": "upload",
            "checksum": "0" * 64,
            "size_bytes": len(self.documents[document_id]),
            "job_ids": [self._id("job")],
        }

    def _document(self, body: Any, request: httpx.Request, document_id: str) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "tenant_id": self.tenant,
            "title": "upload",
            "filename": "upload",
            "media_type": "text/plain",
            "size_bytes": 1,
            "checksum": "0" * 64,
            "status": "READY",
            "archive_status": "ARCHIVED",
            "created_at": "2026-10-01T00:00:00Z",
            "last_error": None,
        }

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids)}"


def _entry(name: str, fields: dict[str, Any]) -> dict[str, Any]:
    """A catalog entry as the service answers it."""
    entry: dict[str, Any] = {
        "tool_id": f"tool_{name}",
        "name": name,
        "version": 1,
        "description": "",
        "input_schema": {"type": "object"},
        "required": [],
        "argument_entity_types": {},
        "side_effects": None,
        "source": "manual",
        "server": None,
        "examples": [],
        "annotations": {},
        "approve_when": None,
        "schema_hash": "0" * 16,
        "workspace_id": None,
    }
    entry.update({k: v for k, v in fields.items() if k in entry or k == "risk"})
    entry.setdefault("risk", entry["side_effects"] or "write")
    return entry
