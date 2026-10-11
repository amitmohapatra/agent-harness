"""The memory service's HTTP API, as far as the harness uses it, in process.

Tests talk to it through the real SDK (``FakeMemoryService.client()`` is a ``MemoryClient`` on
an ``httpx.MockTransport``), so what the harness sends and reads is exactly what the SDK sends
and reads. Request and response shapes, statuses and error documents are the service's — and
checked: every request and every answer is validated against the service's committed
``docs/openapi.json`` (``violations``; ``tests/conftest.py`` fails a test that has any). Every
request lands in ``calls``; the answers are whatever a test sets, and a test can make any call
fail the ways the service fails (a status with its problem document, a timeout, garbage).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from tests.support.openapi import MEMORY_OPENAPI, OpenAPI
from trellis.memory import MemoryClient

URL: Final = "http://memory.test"
#: The toolbox the service hints tools for (the memory service's ``TOOL_HINTS_MIN``); the
#: learned skills come back for any toolbox.
HINTS_MIN_TOOLS: Final = 5
#: The memory service's committed OpenAPI document: every request the harness sends and every
#: answer this fake gives is checked against it (``FakeMemoryService.violations``).
CONTRACT: Final = OpenAPI.load(MEMORY_OPENAPI)
#: Every fake made, so a test's fakes can be checked when it ends (``tests/conftest.py``).
MADE: list[FakeMemoryService] = []

#: The service's agent tools exactly as it lists them (its ``AgentTools.specs()``: names,
#: descriptions, lean input schemas), committed beside this module.
AGENT_TOOLS: Final[list[dict[str, Any]]] = json.loads(
    (Path(__file__).with_name("memory_agent_tools.json")).read_text()
)
#: The memory tools' names, in the service's order.
MEMORY_TOOLS: Final = [t["name"] for t in AGENT_TOOLS]
#: The status each route answers a success with, where it is not 200.
CREATED: Final = {"messages": 202, "record_tool": 202, "add_document": 202}

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
    #: what the context says about its evidence (COMPLETE, INCOMPLETE, INSUFFICIENT)
    evidence_status: str = "COMPLETE"
    #: what ``GET /v1/keys/self`` says about the key
    tenant: str | None = "acme"
    role: str = "service"
    #: the catalog by tool name: risk / approve_when / annotations / side_effects
    catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: tool-hint candidates (names), in rank order; ``None``: the first available one
    candidates: list[str] | None = None
    #: the context answers without a ``tools`` field at all
    omit_candidates: bool = False
    #: the "Learned skills for this task" lines the context carries for an agent with tools
    learned_skills: str = ""
    #: candidates for a particular task, over ``candidates``
    candidates_for: dict[str, list[str]] = field(default_factory=dict)
    #: the claims ``/v1/verify`` finds, and how many of them the evidence does not support
    claims: int = 5
    unsupported: int = 1
    #: the items of the context the claims were checked against
    evidence: int = 3
    #: names of the calls that answer 503 (DEPENDENCY_UNAVAILABLE)
    fail: set[str] = field(default_factory=set)
    #: names of the calls that answer 503 this many times, then succeed
    fail_times: dict[str, int] = field(default_factory=dict)
    #: names of the calls that fail another way: an HTTP status (its problem document, with
    #: ``Retry-After`` on a 429/503), ``"timeout"`` (no answer in time) or ``"malformed"`` (a
    #: 200 whose body is not JSON)
    failures: dict[str, int | str] = field(default_factory=dict)
    #: the ``Retry-After`` (seconds) a 429 or 503 carries
    retry_after: str = "0"
    #: whether model keys can be registered (the service's credential encryption is
    #: configured); without, ``PUT /v1/agents/model-key`` answers what the service does
    model_keys: bool = True
    #: feedback by id, each stored once however often it is sent
    stored_feedback: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: uploaded documents by id, with the scope they were uploaded in
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: how an uploaded document ends (READY or FAILED), after how many reads answer STAGED
    document_outcome: str = "READY"
    document_reads_staged: int = 1
    calls: list[Call] = field(default_factory=list)
    #: what the requests and answers got wrong against the service's OpenAPI document
    violations: list[str] = field(default_factory=list)
    agent_tools: list[dict[str, Any]] = field(default_factory=lambda: list(AGENT_TOOLS))
    #: whether the catalog listing answers with an ETag (and honours If-None-Match)
    etags: bool = True
    _ids: itertools.count[int] = field(default_factory=itertools.count)
    #: the ``source_message_id``s already stored (the service's own message identity)
    _stored: set[str] = field(default_factory=set)
    #: reads of each document so far
    _reads: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        MADE.append(self)

    def client(self, *, max_retries: int = 0) -> MemoryClient:
        transport = httpx.MockTransport(self._checked)
        return MemoryClient(
            URL,
            api_key="test",
            max_retries=max_retries,
            http_client=httpx.AsyncClient(base_url=URL, transport=transport),
        )

    def named(self, name: str) -> list[Call]:
        return [call for call in self.calls if call.name == name]

    # ------------------------------------------------------------------ the API
    def _checked(self, request: httpx.Request) -> httpx.Response:
        response = self._handle(request)
        self.violations += CONTRACT.request(request) + CONTRACT.response(request, response)
        return response

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
                refused = self._refusal(name, request)
                if refused is not None:
                    return refused
                self.calls.append(call)
                answer = getattr(self, f"_{name}")(call)
                status, answer = (
                    answer if isinstance(answer, tuple) else (CREATED.get(name, 200), answer)
                )
                if name == "catalog" and self.etags:
                    return _conditional(request, answer)
                return httpx.Response(status, json=answer)
        return problem(404, request.url.path)

    def _refusal(self, name: str, request: httpx.Request) -> httpx.Response | None:
        """How the call fails, when a test said it does (``None``: it is answered)."""
        if name in self.fail or self.fail_times.get(name, 0) > 0:
            if name in self.fail_times:
                self.fail_times[name] -= 1
            return problem(503, request.url.path, retry_after=self.retry_after)
        failure = self.failures.get(name)
        if failure == "timeout":
            raise httpx.ReadTimeout(f"{name} timed out", request=request)
        if failure == "malformed":
            return httpx.Response(200, content=b"<html>proxy error</html>")
        if isinstance(failure, int):
            return problem(failure, request.url.path, retry_after=self.retry_after)
        if name == "model_key" and not self.model_keys:
            return problem(
                503,
                request.url.path,
                detail="Agent credential encryption is not configured",
                retry_after=self.retry_after,
            )
        return None

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
        requested = call.body.get("tools") or {}
        available = requested.get("available")
        answer: dict[str, Any] = {
            "rendered": rendered,
            "bundle_id": f"bnd_{next(self._ids)}",
            "token_estimate": 0,
            "evidence_status": self.evidence_status,
        }
        if available is not None and self.learned_skills:
            # the service offers what the agent learned to any agent with tools
            rendered += "\n\n## Learned skills for this task\n" + self.learned_skills
            answer["rendered"] = rendered
        if (
            available is not None
            and requested.get("hints", True)
            and len(available) >= HINTS_MIN_TOOLS
        ):
            chosen = self._candidates("", available)
            rendered += "\n\n## Tools\n" + "\n".join(
                f"- {n} (confidence {0.9 - i / 10:.2f})" for i, n in enumerate(chosen)
            )
            if not self.omit_candidates:
                answer["tools"] = [
                    {"name": n, "confidence": round(0.9 - i / 10, 2)} for i, n in enumerate(chosen)
                ]
            answer["rendered"] = rendered
        answer["token_estimate"] = len(rendered) // 4
        return answer

    def _agent_tools(self, call: Call) -> dict[str, Any]:
        return {"tools": self.agent_tools}

    def _call_agent_tool(self, call: Call) -> dict[str, Any]:
        """What each tool returns (the service's ``AgentTools`` handlers): search items with
        ids, a write's id, a profile block, ``tool_search``'s hints."""
        name, args = call.path["name"], call.body.get("args") or {}
        results: dict[str, Any] = {
            "memory_search": [
                {
                    "id": "mem_1",
                    "kind": "memory",
                    "text": f"{name} found: the user prefers email",
                    "observed_on": "2026-09-01",
                },
                {
                    "id": "chk_1",
                    "kind": "chunk",
                    "text": "Returns: 30 days.",
                    "document_id": "doc_1",
                },
            ],
            "memory_remember": {"id": self._id("mem"), "deduplicated": False},
            "memory_update": {"id": self._id("mem"), "supersedes": args.get("id")},
            "memory_forget": {"id": args.get("id"), "forgotten": True},
            "profile_edit": {"block": args.get("block"), "text": args.get("new"), "version": 2},
        }
        if name == "tool_search":
            return {
                "result": self._hints(str(args.get("task", "")), call.body.get("toolbox") or [])
            }
        return {"result": results[name]}

    def _candidates(self, task: str, available: list[str]) -> list[str]:
        names = self.candidates_for.get(
            task, self.candidates if self.candidates is not None else available[:1]
        )
        return [n for n in names if n in available][:8]

    def _tool_hints(self, call: Call) -> dict[str, Any]:
        return self._hints(call.body["task"], call.body.get("available") or [])

    def _hints(self, task: str, available: list[str]) -> dict[str, Any]:
        """The service's hints view: each fitting tool with its confidence, success rate,
        argument values found and what is still missing; the learned plan."""
        chosen = self._candidates(task, available)
        tools = [
            {
                "name": n,
                "confidence": round(0.9 - i / 10, 2),
                "success_rate": 0.8,
                "args": {"sku": "SKU-1"},
                "missing": [{"arg": "qty", "question": "How many?"}],
            }
            | ({"next": True} if i == 0 else {})
            for i, n in enumerate(chosen)
        ]
        hints: dict[str, Any] = {"tools": tools}
        if chosen:
            plan = {"id": "proc_1", "name": task or "the task", "steps": chosen}
            hints["plan"] = plan | {"success_rate": 0.75, "runs": 4}
        return hints

    def _record_tool(self, call: Call) -> dict[str, Any]:
        args = json.dumps(call.body.get("args") or {}, sort_keys=True).encode()
        return {
            "invocation_id": self._id("inv"),
            "step": call.body.get("step") or 0,
            "args_hash": hashlib.sha256(args).hexdigest()[:16],
            "recorded": True,
        }

    def _feedback(self, call: Call) -> tuple[int, dict[str, Any]]:
        """As the service does: the record, ``201`` when stored, ``200`` with the stored one
        when its id was stored before; a vote waits for review."""
        body, scope = call.body, call.scope
        feedback_id = body.get("feedback_id") or self._id("fb")
        if feedback_id in self.stored_feedback:
            return 200, self.stored_feedback[feedback_id]
        record: dict[str, Any] = {
            "feedback_id": feedback_id,
            "tenant_id": scope.get("tenant_id") or self.tenant or "default",
            "workspace_id": scope.get("workspace_id"),
            "user_id": scope.get("user_id"),
            "agent_id": scope.get("agent_id"),
            "agent_run_id": body.get("agent_run_id") or scope.get("agent_run_id"),
            "trace_id": None,
            "target_kind": body["target_kind"],
            "target_id": body["target_id"],
            "verdict": body["verdict"],
            "correction": body.get("correction"),
            "score": body.get("score"),
            "comment": body.get("comment"),
            "reviewer": body.get("reviewer"),
            "source": body.get("source") or "human",
            "evidence_refs": body.get("evidence_refs") or [],
            "metadata": body.get("metadata") or {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        if _waits_for_review(call):
            record["review"] = {"state": "pending"}
        self.stored_feedback[feedback_id] = record
        return 201, record

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
        """A document's life: STAGED (its parse job queued or running) for the first reads,
        then READY — or FAILED, with the parse error."""
        document_id = call.path["document_id"]
        reads = self._reads[document_id] = self._reads.get(document_id, 0) + 1
        staged = reads <= self.document_reads_staged
        status = "STAGED" if staged else self.document_outcome
        return {
            "document_id": document_id,
            "tenant_id": self.tenant or "default",
            "title": "upload",
            "filename": "upload",
            "media_type": "text/plain",
            "size_bytes": 1,
            "checksum": "0" * 64,
            "status": status,
            "archive_status": "STAGED" if staged else "ARCHIVED",
            "created_at": "2026-10-01T00:00:00Z",
            "last_error": "the file could not be parsed" if status == "FAILED" else None,
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
            "evidence_count": self.evidence,
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
        return {"tools": [_catalog_tool(n, self.catalog[n]) for n in names if n in self.catalog]}

    def _put_catalog(self, call: Call) -> dict[str, Any]:
        return {"tools": [_catalog_tool(t["name"], t) for t in call.body["tools"]]}

    def _model_key(self, call: Call) -> dict[str, Any]:
        return {"registered": True, "revoked": False, "revision": 1}

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids)}"


def _catalog_tool(name: str, fields: dict[str, Any]) -> dict[str, Any]:
    """A catalog entry as the service answers it: every member, the tier derived as the
    service derives it (``side_effects`` when set, else the annotations)."""
    annotations = fields.get("annotations") or {}
    derived = (
        "read"
        if annotations.get("readOnlyHint")
        else "irreversible"
        if annotations.get("destructiveHint")
        else "write"
    )
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
    entry.setdefault("risk", entry["side_effects"] or derived)
    return entry


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


#: The service's problem codes by status (its ``api/errors.py``), and whether a retry may pass.
PROBLEMS: Final[dict[int, tuple[str, bool]]] = {
    401: ("AUTHENTICATION", False),
    403: ("SCOPE_DENIED", False),
    404: ("NOT_FOUND", False),
    409: ("CONFLICT", False),
    413: ("VALIDATION", False),
    422: ("VALIDATION", False),
    429: ("RATE_LIMIT", True),
    503: ("DEPENDENCY_UNAVAILABLE", True),
    504: ("TIMEOUT", True),
}


def problem(
    status: int, instance: str, *, detail: str | None = None, retry_after: str = "0"
) -> httpx.Response:
    """An RFC 9457 problem document as the memory service writes it."""
    code, retryable = PROBLEMS.get(status, ("INTERNAL", False))
    body: dict[str, Any] = {
        "type": f"urn:trellis:problem:{code.lower().replace('_', '-')}",
        "title": code.replace("_", " ").capitalize(),
        "status": status,
        "detail": detail or f"{code.lower()} (the fake memory service)",
        "instance": instance,
        "code": code,
        "retryable": retryable,
        "request_id": "req_fake",
    }
    if status == 422:
        body["details"] = {"errors": [{"loc": ["body"], "msg": "invalid"}]}
    headers = {"content-type": "application/problem+json"}
    if retryable and status in (429, 503):
        headers["retry-after"] = retry_after
    return httpx.Response(status, content=json.dumps(body).encode(), headers=headers)


def _conditional(request: httpx.Request, answer: Any) -> httpx.Response:
    """A listing with a strong ETag of its content: ``304`` when the client already has it."""
    digest = hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest()[:16]
    etag = f'"{digest}"'
    if request.headers.get("if-none-match") == etag:
        return httpx.Response(304, headers={"etag": etag})
    return httpx.Response(200, json=answer, headers={"etag": etag})


def _scope(request: httpx.Request, body: Any) -> dict[str, Any]:
    """The scope a request carries: in its body, else in its headers and query."""
    if isinstance(body, dict) and isinstance(body.get("scope"), dict):
        return body["scope"]
    scope: dict[str, Any] = {k: v for k, v in request.url.params.items() if k != "names"}
    for header, name in (("x-trellis-tenant", "tenant_id"), ("x-trellis-user", "user_id")):
        if header in request.headers:
            scope[name] = request.headers[header]
    return scope
