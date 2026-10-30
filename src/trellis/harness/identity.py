"""Who a run is for: tenant, user, thread, agent and run — derived once, used everywhere.

What leaves the process for a run says who it is for in one trusted header,
:data:`IDENTITY_HEADER` (:func:`identity_headers`): the calls a run makes to other agents
(A2A), and its MCP tool calls through the gateway, which forwards it to the MCP servers whose
``allowed_extra_headers`` name it. What arrives with it — set (and overwritten) by the
platform's authenticating edge — is read the same way on both surfaces (:func:`trusted_user`:
``serve_a2a``, ``serve_chat``): without it a call is refused (``401``) unless the deployment
names its callers another way (``identity=``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from trellis.contracts import AgentExecutionContext, RunRecord

#: The trusted identity header (lower case: the A2A SDK hands a server lower-cased headers).
IDENTITY_HEADER: Final = "x-trellis-identity"
#: A trusted identity header longer than this is refused rather than parsed.
HEADER_MAX_CHARS: Final = 4096
#: The ``RunStart.metadata`` key a sub-agent's run keeps its agent group under.
AGENT_GROUP: Final = "agent_group"
#: The prefix of a conversation's key (:func:`conversation_key`).
CONVERSATION_PREFIX: Final = "conv_"


def conversation_key(tenant: str, user: str, thread: str) -> str:
    """The key the conversation ``thread`` of ``user`` in ``tenant`` keeps its state under —
    the framework's thread (a LangGraph ``thread_id``), the memory service's thread, the
    turn a busy conversation waits for, the run store's ``concurrency_key``, the trace's
    session: a thread id is the caller's name for a chat, unique only for one user, so two
    users (or tenants) naming theirs alike never share one. A digest of the three, so it is
    the same wherever it is made, of fixed length (the run store and the memory service keep
    ids of up to 200 characters) and safe in a URL path (the memory service's
    ``/v1/threads/{id}``)."""
    named = json.dumps([tenant, user, thread], separators=(",", ":"))
    return CONVERSATION_PREFIX + hashlib.sha256(named.encode()).hexdigest()[:32]


class Identity(BaseModel):
    """The scope of one run. ``user`` is required: memory, approvals and the inbox are all
    somebody's, and a run for nobody would be scoped to everybody."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str
    user: str
    agent_id: str
    run_id: str
    thread: str | None = None
    workspace: str | None = None
    #: the run this one is a child run of (a sub-agent's, a nested run): what it reads of the
    #: parent's hand-off notes, and where its own reports go up to
    parent_run: str | None = None
    #: the agent group a sub-agent's run works in: its delegation tree, named by the run at
    #: its top (``memory_remember(scope="group")`` binds to it)
    group: str | None = None

    @classmethod
    def of(cls, record: RunRecord) -> Identity:
        """A run's, from its record: a scheduled run acts for whom it runs on behalf of, and —
        having no thread — its transcript is its own; a sub-agent's run names its parent and
        its group."""
        group = record.metadata.get(AGENT_GROUP)
        return cls(
            tenant=record.tenant_id,
            user=record.user_id or record.on_behalf_of or "system",
            agent_id=record.agent_id,
            run_id=record.run_id,
            thread=record.thread_id or record.run_id,
            workspace=record.workspace_id,
            parent_run=record.parent_run_id,
            group=group if isinstance(group, str) else None,
        )

    @property
    def conversation(self) -> str:
        """The key the run's conversation keeps its state under (:func:`conversation_key`);
        a run with no conversation of its own is its own (its run id)."""
        if self.thread is None or self.thread == self.run_id:
            return self.run_id
        return conversation_key(self.tenant, self.user, self.thread)

    @property
    def own_group(self) -> str:
        """The agent group the runs this one starts (its sub-agents', its framework's
        helpers') work in: its own, else the one it names by being at the top of the tree."""
        return self.group or self.run_id

    def scope(self) -> dict[str, Any]:
        """The memory service scope of a run with no parent (a sub-agent's run is derived from
        its parent's: :meth:`parent_scope`)."""
        return _present(
            tenant_id=self.tenant,
            user_id=self.user,
            agent_id=self.agent_id,
            agent_run_id=self.run_id,
            thread_id=self.conversation,
            workspace_id=self.workspace,
        )

    def parent_scope(self) -> dict[str, Any]:
        """A sub-agent's run's parent, as the memory service scope the child's is derived from
        (``MemoryContext.agent``): the same person, conversation and workspace, the parent's
        run."""
        return _present(
            tenant_id=self.tenant,
            user_id=self.user,
            agent_run_id=self.parent_run,
            thread_id=self.conversation,
            workspace_id=self.workspace,
        )

    def context(self) -> AgentExecutionContext:
        """The contracts execution context (events, interrupts and feedback are built on it)."""
        return AgentExecutionContext.create(
            tenant_id=self.tenant,
            agent_id=self.agent_id,
            agent_run_id=self.run_id,
            user_id=self.user,
            thread_id=self.thread,
            workspace_id=self.workspace,
        )


def _present(**fields: str | None) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


def identity_headers(tenant: str, user: str) -> dict[str, str]:
    """The outbound trusted identity, for a call this process makes for a run."""
    fields = {"tenant_id": tenant, "user_id": user}
    return {IDENTITY_HEADER: json.dumps(fields, separators=(",", ":"), sort_keys=True)}


class IdentityRefused(PermissionError):
    """The call carries an identity this deployment will not act on."""


def trusted_user(raw: str | None, served: str | None) -> str:
    """The user a trusted identity header names: ``{"tenant_id": ..., "user_id": ...}``, its
    tenant (when it names one) the one ``TRELLIS_API_KEY`` speaks for (``served``). No header,
    or anything else, is :class:`IdentityRefused`: nobody is anonymous."""
    if not raw:
        raise IdentityRefused(
            f"no {IDENTITY_HEADER} header: the platform's authenticating edge sets it, or the "
            "deployment names the caller with identity=..."
        )
    if len(raw) > HEADER_MAX_CHARS:
        raise IdentityRefused("the platform identity header is too long")
    try:
        claimed = json.loads(raw)
    except ValueError as exc:
        raise IdentityRefused("the platform identity header is not JSON") from exc
    if not isinstance(claimed, Mapping):
        raise IdentityRefused("the platform identity header must be a JSON object")
    tenant = claimed.get("tenant_id")
    if tenant and tenant != served:
        raise IdentityRefused(f"tenant {tenant!r} is not served here")
    user = claimed.get("user_id")
    if not user:
        raise IdentityRefused("the platform identity header names no user")
    return str(user)
