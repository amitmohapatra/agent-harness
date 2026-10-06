"""The memory service, as the harness uses it: every call the harness makes goes through here,
except the blocks usable without ``Harness`` — the tool catalog (``governance/catalog.py``) and
the grounding check (``evals.grounding_score``) — which call the SDK's ``MemoryContext`` itself.

``Memory`` is the process's client; ``Memory.bind(identity)`` is a :class:`RunMemory`, the
calls one run makes in its own scope. The SDK's ``MemoryContext`` it wraps (``.ctx``) is also
what tools and nodes get as ``trellis.current().memory``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final

from trellis.contracts import Feedback as Decision
from trellis.contracts import ToolCall, ToolOutcome, ToolSpec
from trellis.harness.fresh import Fresh
from trellis.harness.governance.catalog import MemoryCatalog
from trellis.harness.identity import Identity
from trellis.memory import MemoryClient, MemoryContext
from trellis.memory.models import (
    AgentTool,
    DocumentInfo,
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
#: How long the agent-tool listing is kept before it is listed again; while the service
#: cannot be reached, the last listing is kept and asked for again after the retry interval.
AGENT_TOOLS_TTL_SECONDS: Final = 600.0
AGENT_TOOLS_RETRY_SECONDS: Final = 30.0


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
        return RunMemory(self, self.client.bind(**identity.scope()))

    def for_user(self, tenant: str, user: str, thread: str | None = None) -> RunMemory:
        """Calls made for a person outside a run (adding a document they can retrieve)."""
        scope = {"tenant_id": tenant, "user_id": user} | ({"thread_id": thread} if thread else {})
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
                messages = [(role, content) for role, content in args["messages"]]
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
        window: bool,
        budget: int = CONTEXT_TOKEN_BUDGET,
    ) -> PromptContext:
        """What the prompt gets (at most ``budget`` tokens), and the tools that fit the task
        (``tools``, when ``tools`` are given). ``window=False`` when the framework keeps the
        thread's messages itself: the service then leaves the recent conversation out."""
        return await self.ctx.context(query, token_budget=budget, tools=tools, window=window)

    # ------------------------------------------------------------------ pull
    async def agent_tools(self) -> list[ToolSpec]:
        return await self.memory.agent_tools(self.ctx)

    async def call_agent_tool(self, name: str, args: dict[str, object]) -> object:
        """One memory tool, in this run's scope."""
        return await self.ctx.call_agent_tool(name, args)

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
    ) -> Feedback:
        """A verdict on this scope's run (``source`` ``system`` for how it ended, ``human``
        for a person's); ``key`` makes a retry store it once. The stored record comes back:
        a person's verdict waits for the tenant administrator (``review.state`` pending,
        ADR 0028 of the memory service) where the run's own status is applied at once."""
        run_id = self.ctx.scope.agent_run_id
        assert run_id is not None
        return await self.ctx.feedback(
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

    async def register_model_key(self, key: str) -> None:
        """The agent-level LLM key the service uses for this agent's memory, in an agent
        scope (the same request from every process and run, so the idempotent PUT repeats
        rather than conflicts; a rotated key is a new request)."""
        agent = self.ctx.scope.agent_id
        digest = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
        await self.ctx.advanced.model_keys.set(key, idempotency_key=f"model-key:{agent}:{digest}")


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
