"""The memory service, as the harness uses it: every call the harness makes goes through here,
except the blocks usable without ``Harness`` — the tool catalog (``governance/catalog.py``) and
the grounding check (``evals.grounding_score``) — which call the SDK's ``MemoryContext`` itself.

``Memory`` is the process's client; ``Memory.bind(identity)`` is a :class:`RunMemory`, the
calls one run makes in its own scope. The SDK's ``MemoryContext`` it wraps (``.ctx``) is also
what tools and nodes get as ``trellis.current().memory``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Final

from trellis.contracts import (
    ConfigurationError,
    Endpoint,
    FeedbackSource,
    FeedbackTargetKind,
    ToolCall,
    ToolOutcome,
    ToolSpec,
)
from trellis.contracts import Feedback as Decision
from trellis.harness.fresh import Fresh
from trellis.harness.governance.catalog import MemoryCatalog
from trellis.harness.history import Said, Speaker, recorded, said_of
from trellis.harness.identity import Identity, conversation_key
from trellis.memory import MemoryClient, MemoryContext
from trellis.memory.errors import NotFoundError
from trellis.memory.models import (
    AgentTool,
    DocumentInfo,
    EvidenceRef,
    Feedback,
    KeyInfo,
    PromptContext,
)

#: Prompt budget for the pushed context, in tokens, when the model's context window is not
#: known; with it known, :data:`CONTEXT_SHARE` of the window, between that and
#: :data:`CONTEXT_TOKEN_MAX`.
CONTEXT_TOKEN_BUDGET: Final = 2000
CONTEXT_SHARE: Final = 0.05
CONTEXT_TOKEN_MAX: Final = 8000
#: The pull tools that change nothing: listed with side_effects "read", the rest "write".
READ_ONLY_TOOLS: Final = frozenset({"memory_search", "tool_search"})
#: The pull tool that chooses among the run's own tools: the harness answers it with
#: ``Tools.hints``, which passes the run's toolbox.
TOOL_SEARCH: Final = "tool_search"
#: What the harness calls the transcript it writes, so a re-recorded message is stored once.
SOURCE_SYSTEM: Final = "trellis-harness"
#: The ``RunStart.metadata`` key a run's ``document_ids=`` is kept under.
DOCUMENTS: Final = "document_ids"
#: The keys of a run's own outcome record's metadata that keep what its pushed context showed —
#: its bundle and the ids of the memories in it (at most :data:`CITED_MAX`) — so a person's
#: verdict on an unverified run cites the memories the model was shown. Flat: the service keeps
#: a caller's metadata at most two levels deep.
PUSHED_BUNDLE: Final = "pushed_bundle_id"
PUSHED_MEMORIES: Final = "pushed_memory_ids"
#: The most memories a person's verdict on a run cites (the memory service's bound on a
#: verdict's evidence references).
CITED_MAX: Final = 50
#: How long the agent-tool listing is kept before it is listed again; while the service
#: cannot be reached, the last listing is kept and asked for again after the retry interval.
AGENT_TOOLS_TTL_SECONDS: Final = 600.0
AGENT_TOOLS_RETRY_SECONDS: Final = 30.0


class Window(StrEnum):
    """Whether, and how, a run's context carries the thread's recent messages."""

    #: left out: the framework keeps the conversation itself
    NONE = "none"
    #: as a section of the rendered context (a framework that takes no history of messages)
    RENDERED = "rendered"
    #: verbatim, for the framework to replay as its own conversation
    MESSAGES = "messages"


class Memory:
    """One memory service for the process."""

    def __init__(self, client: MemoryClient) -> None:
        self.client = client
        self._lister = self.client.bind()
        self._agent_tools = Fresh(
            self._list_agent_tools,
            what="the memory service's agent tools",
            ttl=AGENT_TOOLS_TTL_SECONDS,
            retry=AGENT_TOOLS_RETRY_SECONDS,
        )

    @property
    def listed(self) -> list[ToolSpec] | None:
        """The agent tools last listed (``None`` before the first listing)."""
        return self._agent_tools.value

    async def agent_tools(self, ctx: MemoryContext) -> list[ToolSpec]:
        """The memory service's agent tools (a fixed set, listed in the asking run's scope):
        listed again every :data:`AGENT_TOOLS_TTL_SECONDS`, the last listing kept while the
        service is down; raises only when they were never listed."""
        self._lister = ctx
        return list(await self._agent_tools.get())

    async def _list_agent_tools(self) -> list[ToolSpec]:
        return [_agent_tool(t) for t in await self._lister.agent_tools()]

    async def key(self) -> KeyInfo:
        """Who ``TRELLIS_API_KEY`` is: its tenant, principal and role."""
        return await self.client.tenant.keys.whoami()

    def bind(self, identity: Identity) -> RunMemory:
        """A run's calls, in its scope. A sub-agent's run is derived from its parent's the
        SDK's way (``MemoryContext.agent``): it names its parent run, so it reads the
        parent's hand-off notes (``RUN`` visibility) and its own go up to the parent, and its
        agent group, which ``memory_remember(scope="group")`` binds to."""
        if identity.parent_run is None:
            return RunMemory(self, self.client.bind(**identity.scope()))
        parent = self.client.bind(**identity.parent_scope())
        child = parent.agent(
            identity.agent_id, agent_run_id=identity.run_id, agent_group_id=identity.group
        )
        return RunMemory(self, child)

    def for_user(self, tenant: str, user: str, thread: str | None = None) -> RunMemory:
        """Calls made for a person outside a run (adding a document they can retrieve) — in
        their conversation ``thread``, the memory service's thread its runs write to
        (``identity.conversation_key``)."""
        scope = {"tenant_id": tenant, "user_id": user} | (
            {"thread_id": conversation_key(tenant, user, thread)} if thread else {}
        )
        return RunMemory(self, self.client.bind(**scope))

    def scoped(self, tenant: str, agent_id: str | None = None) -> RunMemory:
        """Calls made outside a run: tenant-wide (the tool catalog) or for one agent (its
        model key)."""
        scope = {"tenant_id": tenant} | ({"agent_id": agent_id} if agent_id else {})
        return RunMemory(self, self.client.bind(**scope))

    def replay(self, record: Mapping[str, Any]) -> Callable[[], Awaitable[object]] | None:
        """The write a spooled record describes (:meth:`RunMemory.record`), in its scope;
        ``None`` for a record of no write this client replays."""
        run = RunMemory(self, self.client.bind(**record["scope"]))
        args = record["args"]
        match record["op"]:
            case "record_messages":
                messages = [said_of(m) for m in args["messages"]]
                return lambda: run.record_messages(messages, args["run_id"], args["attempt"])
            case "record_tool":
                call = ToolCall.model_validate(args["call"])
                outcome = ToolOutcome.model_validate(args["outcome"])
                return lambda: run.record_tool(call, outcome)
            case "run_feedback":
                return lambda: run.run_feedback(**args)
            case "feedback":
                decision = Decision.model_validate(args["record"])
                return lambda: run.feedback(decision)
            case "publish_catalog":
                return lambda: MemoryCatalog(run.ctx).publish(args["entries"])
        return None


class RunMemory:
    """The calls one run makes, in its scope."""

    __slots__ = ("ctx", "memory")

    def __init__(self, memory: Memory, ctx: MemoryContext) -> None:
        self.memory = memory
        self.ctx = ctx

    def record(self, op: str, **args: Any) -> dict[str, Any]:
        """A write of this scope as data — the method (``op``) and its arguments as JSON — so
        the write spool can keep it and :meth:`Memory.replay` run it after a restart."""
        scope = self.ctx.scope.model_dump(mode="json", exclude_none=True)
        return {"op": op, "scope": scope, "args": args}

    # ------------------------------------------------------------------ push
    async def context(
        self,
        query: str,
        *,
        tools: Sequence[str] | None,
        window: Window,
        hints: bool,
        budget: int = CONTEXT_TOKEN_BUDGET,
        document_ids: Sequence[str] | None = None,
    ) -> PromptContext:
        """What the prompt gets (at most ``budget`` tokens): with the agent's ``tools``, the
        skills it learned for the task and - unless ``hints`` is off - the tools that fit.
        ``window``: the thread's recent messages left out (``Window.NONE``: the framework
        keeps them itself), in the rendered text (``RENDERED``), or verbatim as
        ``conversation.messages`` (``MESSAGES``) for the framework to replay as its own.
        ``document_ids`` narrows the documents its passages come from to those (``None``:
        every one the scope may read)."""
        return await self.ctx.context(
            query,
            token_budget=budget,
            tools=tools,
            hints=hints,
            window=window is not Window.NONE,
            window_format="messages" if window is Window.MESSAGES else "rendered",
            document_ids=document_ids,
        )

    # ------------------------------------------------------------------ pull
    async def agent_tools(self) -> list[ToolSpec]:
        return await self.memory.agent_tools(self.ctx)

    async def call_agent_tool(self, name: str, args: dict[str, object]) -> object:
        """One memory tool, in this run's scope."""
        return await self.ctx.call_agent_tool(name, args)

    async def tool_hints(self, task: str, available: Sequence[str]) -> Any:
        return await self.ctx.tool_hints(task, available=list(available))

    # ------------------------------------------------------------------ records
    async def record_messages(self, messages: Sequence[Said], run_id: str, attempt: int) -> None:
        """One attempt's transcript - its tool calls and their results with it - in one
        request. Each message names itself (the run for the question, which every attempt
        asks; the attempt and position for what the agent said), and the service stores a
        message it has seen before once."""
        batch = [
            recorded(said)
            | {
                "source_system": SOURCE_SYSTEM,
                "source_message_id": f"{run_id}:user:{index}"
                if said.role is Speaker.USER
                else f"{run_id}:{attempt}:msg:{index}",
            }
            for index, said in enumerate(messages)
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
        evidence_refs: Sequence[EvidenceRef] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Feedback:
        """A verdict on this scope's run (``source`` ``system`` for how it ended, ``human``
        for a person's); ``key`` makes a retry store it once; ``evidence_refs`` are the
        memories its answer cited, whose confidence the verdict moves; ``metadata`` is kept
        with it (the run's own: :data:`PUSHED_MEMORIES`, what its context showed). The stored
        record comes back: a person's verdict waits for the tenant administrator (``review.state``
        pending, ADR 0028 of the memory service) where the run's own status is applied at
        once."""
        run_id = self.ctx.scope.agent_run_id
        assert run_id is not None
        return await self.ctx.feedback(
            FeedbackTargetKind.RUN.value,
            run_id,
            verdict,
            correction=correction,
            comment=comment,
            reviewer=reviewer,
            source=source,
            evidence_refs=evidence_refs,
            metadata=metadata,
            idempotency_key=key,
        )

    async def cited(self) -> list[EvidenceRef]:
        """The memories this scope's run's answer rests on, each once, at most
        :data:`CITED_MAX`: what the grounding judge's verdict on it cites (``/v1/verify``
        resolves the answer's claims against the run's context bundle and records the
        supporting memories with its verdict); a run that was not verified (most runs: the
        check takes a sample) cites the memories its pushed context showed, which its own
        outcome record keeps (:data:`PUSHED_MEMORIES`). Empty when the service holds no
        feedback on the run (``NotFoundError``), or neither cites a memory."""
        run_id = self.ctx.scope.agent_run_id
        assert run_id is not None
        try:
            page = await self.ctx.feedback.page_for(FeedbackTargetKind.RUN.value, run_id)
        except NotFoundError:
            return []
        judged = [f for f in page.items if f.source == FeedbackSource.JUDGE.value]
        refs = {ref.source_id: ref for verdict in judged for ref in verdict.evidence_refs}
        if not refs:
            own = [f for f in page.items if f.source == "system"]  # the run's own outcome
            shown = own[0].metadata.get(PUSHED_MEMORIES, ()) if own else ()
            refs = {mid: EvidenceRef(source_type="memory", source_id=mid) for mid in shown}
        return list(refs.values())[:CITED_MAX]

    async def feedback(self, record: Any) -> None:
        """A contracts ``Feedback`` record as it is (an interrupt's decision). Its id is the
        idempotency key: an interrupt's feedback id is fixed by its run and interrupt, so the
        client may retry a failed send and the service stores and counts it once."""
        await self.ctx.feedback(record, idempotency_key=record.feedback_id)

    # ------------------------------------------------------------------ documents, model key
    async def add_document(
        self,
        file: Any,
        *,
        title: str | None = None,
        visibility: str | None = None,
        wait: float | None = 60.0,
    ) -> DocumentInfo:
        """Upload a file into this scope's document memory - bytes, a path, or a (filename,
        bytes, media_type) tuple - and, unless ``wait`` is None, wait until it is indexed.
        Context for this user (and thread) then cites it like any other document."""
        handle = await self.ctx.advanced.documents.add(
            file,
            title=title,
            visibility=visibility,  # type: ignore[arg-type]
        )
        if wait is None:
            return await self.ctx.advanced.documents.document(handle.document_id)
        return await self.ctx.advanced.documents.wait_ready(handle.document_id, max_wait=wait)

    async def document(self, document_id: str) -> DocumentInfo:
        """A document as this scope's user may read it: the memory service refuses one they
        may not (``AuthorizationError``) and one that does not exist (``NotFoundError``)."""
        return await self.ctx.advanced.documents.document(document_id)

    async def register_endpoint(self, endpoint: Endpoint) -> None:
        """The model endpoint the service calls for this agent's memory work, in an agent
        scope: the whole ``Endpoint``, its key in plain (``SecretStr`` masks it; the service
        keeps it encrypted). The same request from every process and run, so the idempotent PUT
        repeats rather than conflicts; a rotated key or a moved endpoint is a new request. An
        answer that holds no endpoint — the replay of an earlier request whose registration the
        service has since dropped (a migration, a cleared store) — is sent again, as a new
        request."""
        agent = self.ctx.scope.agent_id
        wire = endpoint.model_dump(mode="json")
        wire["api_key"] = endpoint.api_key.get_secret_value() if endpoint.api_key else None
        said = json.dumps(wire, sort_keys=True).encode()
        digest = hashlib.blake2b(said, digest_size=8).hexdigest()
        key = f"model-key:{agent}:{digest}"
        status = await self.ctx.advanced.model_keys.set(wire, idempotency_key=key)
        if not status.registered or status.endpoint is None:
            await self.ctx.advanced.model_keys.set(
                wire, idempotency_key=f"{key}:{uuid.uuid4().hex}"
            )


def documents(document_ids: Sequence[str]) -> list[str]:
    """``document_ids=`` checked: a list of document ids (``h.add_document``'s), each once,
    at least one — an empty list would narrow the context to no document at all."""
    if isinstance(document_ids, str):
        raise ConfigurationError("document_ids= is a list of ids: [document_id], not one id")
    ids = list(dict.fromkeys(document_ids))
    if not ids or not all(isinstance(i, str) and i for i in ids):
        raise ConfigurationError(
            "document_ids= names the documents a run's context draws on: one id or more "
            "(h.add_document(...).document_id); leave it out for all of them"
        )
    return ids


def context_budget(window: int | None) -> int:
    """The pushed context's token budget for a model reading ``window`` tokens (``None``: not
    known)."""
    if not window:
        return CONTEXT_TOKEN_BUDGET
    return max(CONTEXT_TOKEN_BUDGET, min(CONTEXT_TOKEN_MAX, int(window * CONTEXT_SHARE)))


def _agent_tool(tool: AgentTool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema or {"type": "object"},
        source="memory",
        side_effects="read" if tool.name in READ_ONLY_TOOLS else "write",
    )
