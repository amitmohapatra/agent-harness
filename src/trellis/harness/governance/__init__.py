"""Governance: whether a tool call runs, runs and is announced, or waits for a person — and
the tool catalog that decision reads, kept fresh.

The one place the decision is made (:meth:`Governance.check`): a ``read`` tool runs, a
``write`` tool runs and is announced, an ``irreversible`` tool asks; the catalog's ``risk``
overrides the tool's own, and its ``approve_when`` decides when the call asks
(``decision.py``). Without a catalog (no memory service) the tools' own risks decide.

Two ways in. Wrapped (``h.wrap``): every harness tool call goes through the bridge, which
checks it with :meth:`Harness.governance` of the run's tenant, then pauses the run, announces
the call or runs it. Pluggable: :meth:`Governance.from_env` in any code, and
:meth:`Governance.check` or :func:`governed` around the team's own tools; :meth:`publish`
tells the catalog about the tools (an administrator sets ``approve_when`` on them there), and
:meth:`decided` tells the memory service what a person decided, which it learns approval
suggestions from.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, Literal

from trellis.contracts import (
    AgentExecutionContext,
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    stable_id,
)
from trellis.harness.governance.catalog import (
    CATALOG_UNREAD,
    Catalog,
    MemoryCatalog,
    Published,
    Rule,
    Rules,
    digest,
    entry,
)
from trellis.harness.governance.decision import Action, Decision, decide
from trellis.harness.hooks import Chain, Deny, Hooks, denied
from trellis.harness.journal import content_key
from trellis.harness.settings import Settings
from trellis.harness.tools.base import Tool, execute, invoked
from trellis.memory import MemoryClient

#: How a publish is sent: given the entries and the send itself, run it — the harness queues it
#: with its background writes, which keep the entries on disk when it cannot be delivered.
Submit = Callable[[list[dict[str, object]], Callable[[], Awaitable[None]]], Awaitable[None]]
#: What a person may decide about a call that asked.
Verdict = Literal["approve", "reject", "edit"]
VERDICTS: Final = frozenset({"approve", "reject", "edit"})


class Rejected(Exception):
    """The call was not run: the person who was asked rejected it."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(f"{decision.tool} was not run: the approver rejected it")
        self.decision = decision


class Denied(Exception):
    """The call was not run: a ``before_tool`` hook denied it (``outcome``: what the model
    reads)."""

    def __init__(self, outcome: ToolOutcome) -> None:
        super().__init__(str(outcome.output))
        self.outcome = outcome


class Governance:
    """The decision for one tenant's tool calls. ``catalog`` is where the rules are read
    (``None``: the tools' own risks decide); ``submit`` sends a publish (``None``: at once);
    ``tenant`` and ``agent_id`` are whom :meth:`decided` attributes a decision to (the key's
    own tenant is enough for everything else)."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        *,
        submit: Submit | None = None,
        tenant: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.tenant = tenant
        self.agent_id = agent_id
        self._rules = Rules(catalog) if catalog is not None else None
        self._submit = submit
        self._published = Published()
        #: the memory client :meth:`from_env` made, closed by :meth:`aclose`
        self._client: MemoryClient | None = None

    @classmethod
    def from_env(
        cls,
        *,
        agent_id: str | None = None,
        tenant: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Governance:
        """Governance from the deployment's environment: the memory service's catalog with
        ``MEMORY_URL`` and ``TRELLIS_API_KEY`` (in ``tenant``, or the key's own), or — with
        ``MEMORY_URL`` unset — the tools' own risks only."""
        settings = Settings.from_env(environ)
        if settings.memory_url is None:
            return cls(tenant=tenant, agent_id=agent_id)
        if settings.api_key is None:
            raise ConfigurationError(
                "MEMORY_URL needs TRELLIS_API_KEY: the memory service refuses a call without a key"
            )
        client = MemoryClient(settings.memory_url, api_key=settings.api_key)
        scope = {"tenant_id": tenant} if tenant is not None else {}
        governance = cls(MemoryCatalog(client.bind(**scope)), tenant=tenant, agent_id=agent_id)
        governance._client = client
        return governance

    async def check(
        self, tool: str, args: Mapping[str, Any], *, side_effects: str = "write"
    ) -> Decision:
        """The decision for a call of ``tool`` with ``args``. ``side_effects`` is what the tool
        says it does (``read``, ``write``, ``irreversible``); the catalog's word overrides it,
        and while the catalog cannot be read every tool that does more than read asks."""
        if self._rules is not None:
            rules = await self._rules.get([tool])
            if rules is None:
                if side_effects != "read":
                    return decide(tool, side_effects, CATALOG_UNREAD, args)
            elif (rule := rules.get(tool)) is not None:
                return decide(tool, rule.risk, rule.approve_when, args)
        return decide(tool, side_effects, None, args)

    async def rules(self, names: Sequence[str]) -> dict[str, Rule | None]:
        """The catalog's rule for each of ``names``: ``None`` for a tool it says nothing about
        — and for every tool while it cannot be read (:meth:`check` then asks for each one
        that does more than read)."""
        found = await self._rules.get(names) if self._rules is not None else None
        return {name: (found or {}).get(name) for name in names}

    async def publish(
        self,
        specs: Sequence[ToolSpec],
        *,
        annotations: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Tell the catalog about ``specs`` (an MCP tool with its server's ``annotations``, by
        name), once per content: an entry is sent again only when it changed, or when its
        last send failed. Nothing without a catalog."""
        catalog = self.catalog
        if catalog is None:
            return
        hints = annotations or {}
        published = self._published
        fresh = published.fresh([entry(spec, hints.get(spec.name)) for spec in specs])
        if not fresh:
            return
        digests = {digest(e) for e in fresh}
        published.pending |= digests

        async def send() -> None:
            try:
                await catalog.publish(fresh)
                published.done |= digests  # stored: not sent again
            finally:
                published.pending -= digests  # failed: sent again by the next publish

        if self._submit is None:
            await send()
        else:
            await self._submit(fresh, send)

    async def decided(
        self,
        decision: Decision,
        verdict: Verdict,
        *,
        reviewer: str,
        run_id: str,
        user: str,
        edited: Mapping[str, Any] | None = None,
    ) -> None:
        """What ``reviewer`` decided about the call ``decision`` asked for, in the run
        ``run_id`` acting for ``user``: the ``TOOL_CALL`` feedback the memory service learns
        approval suggestions from (``edited``: the arguments of an ``edit``). Sent once however
        often it is retried. Nothing without a catalog."""
        if verdict not in VERDICTS:
            raise ValueError(f"a decision is approve, reject or edit, not {verdict!r}")
        if self.catalog is None:
            return
        if self.tenant is None or self.agent_id is None:
            raise ConfigurationError(
                "a decision is recorded for an agent in a tenant: pass agent_id= and tenant="
            )
        args = dict(decision.args)
        key = content_key("call", decision.tool, args)
        interrupt = Interrupt(
            interrupt_id=stable_id(run_id, key, prefix="int_"),
            tenant_id=self.tenant,
            run_id=run_id,
            reason=InterruptReason.APPROVAL,
            question=decision.question,
            tool_call=ToolCall(tool=decision.tool, args=args, idempotency_key=key),
        )
        resolution = InterruptResolution(
            interrupt_id=interrupt.interrupt_id,
            run_id=run_id,
            decision=InterruptDecision(verdict.upper()),
            reviewer=reviewer,
            payload=dict(edited) if edited is not None else None,
        )
        context = AgentExecutionContext.create(
            tenant_id=self.tenant, agent_id=self.agent_id, agent_run_id=run_id, user_id=user
        )
        record = resolution.to_feedback(interrupt, context)
        assert record is not None  # an approve, reject or edit of a call is always feedback
        await self.catalog.feedback(record)

    async def aclose(self) -> None:
        """Close the memory client :meth:`from_env` made."""
        if self._client is not None:
            await self._client.aclose()


def governed(
    fn: Callable[..., Any],
    governance: Governance,
    *,
    name: str | None = None,
    side_effects: str = "write",
    timeout: float | None = None,
    on_ask: Callable[[Decision], Any],
    on_announce: Callable[[Decision], Any] | None = None,
    hooks: Sequence[Hooks] = (),
) -> Callable[..., Awaitable[Any]]:
    """``fn`` (sync or async) as an async callable whose every call — with keyword arguments,
    the tool's arguments — is checked first. A call that asks runs ``on_ask(decision)`` (sync
    or async): ``True`` runs it, ``False`` raises :class:`Rejected`, a dict runs it with
    those (edited) arguments, and an exception (LangGraph's ``interrupt``) propagates. A call
    that is announced runs ``on_announce(decision)`` first, when given.

    It runs as a harness tool call does: at most ``timeout`` seconds (a sync ``fn`` in a
    worker thread), tried again after an error that may pass when it only reads; out of time
    it raises :class:`~trellis.harness.tools.base.ToolTimeout`, whose message is what the
    model should read (for a call that does more than read: that it may have taken effect).

    ``hooks`` (``trellis.harness.hooks``) run around each call as around a harness tool call:
    ``before_tool`` first — a ``Deny`` raises :class:`Denied`, a ``Rewrite`` changes the
    arguments governance checks and the call gets, an ``Ask`` asks (``on_ask``) whatever
    governance says (the decision carries the ``Ask``'s ``assignee``, ``component`` and
    ``props``) —, ``on_error("tool", ...)`` when it fails, ``after_tool`` on its outcome (the
    call returns that outcome's output)."""
    tool = Tool(
        ToolSpec(name=name or fn.__name__, side_effects=side_effects),
        lambda args: invoked(fn, **args),
        timeout=timeout,
    )

    chain = Chain(hooks)

    @functools.wraps(fn)
    async def call(**args: Any) -> Any:
        hooked, verdict = await chain.tool(ToolCall(tool=tool.name, args=args))
        if isinstance(verdict, Deny):
            raise Denied(denied(hooked, verdict))
        args = hooked.args
        decision = await governance.check(tool.name, args, side_effects=side_effects)
        if verdict is not None:
            decision = decision.asking(
                verdict.question,
                assignee=verdict.assignee,
                component=verdict.component,
                props=verdict.props,
            )
        if decision.asks:
            answer = await _settled(on_ask(decision))
            if answer is False:
                raise Rejected(decision)
            if isinstance(answer, Mapping):
                args = dict(answer)
            elif answer is not True:
                raise TypeError(
                    f"on_ask answered {answer!r}: True (run), False (reject) or a dict of "
                    "edited arguments"
                )
        elif decision.announces and on_announce is not None:
            await _settled(on_announce(decision))
        outcome, error = await execute(tool, args, reads=decision.risk == "read")
        if error is not None:
            await chain.failed("tool", error)
        outcome = await chain.done(hooked.model_copy(update={"args": args}), outcome)
        if error is not None and not outcome.ok:
            raise error
        return outcome.output

    call.__name__ = tool.name
    return call


async def _settled(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


__all__ = ["Action", "Decision", "Denied", "Governance", "Rejected", "governed"]
