"""Hooks: your code around a run and its tool calls — every adapter, wrapped or not.

    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Ask | Rewrite | None:
            if call.tool == "refund" and await frozen(call.args["account"]):
                return Deny("the account is frozen")
            return None

    h = Harness(hooks=[Guard()])                      # every agent of this harness
    agent = h.wrap(target, id="support", hooks=[Audit()])  # this agent, after the harness's

Model calls have no hook here: each framework has its own callbacks for them (LangChain and
Deep Agents middleware — ``ReAct(middleware=[...])`` —, the OpenAI Agents SDK's ``AgentHooks``,
the Claude Agent SDK's hooks), and a hook class that still defines ``before_model`` or
``after_model`` is refused where it is given (:class:`Chain`), never silently skipped.

Hooks are for custom logic a rule cannot express — a refusal, a rewrite, a decision on data
outside the call, who reviews and on which screen. When a call needs a person is a rule, and
rules live in one place: the tool catalog's ``risk`` and ``approve_when``
(``trellis.harness.governance``).

Subclass :class:`Hooks` and override what you need (every method does nothing by default);
hooks run in order — the harness's, then the agent's. Where they fire:

* ``on_run_start(run)`` / ``on_run_end(run, result)`` — each attempt of a run, in the pipeline
  (``run`` is the :class:`~trellis.harness.runtime.Runtime`; a resumed run starts again);
* ``before_tool(call)`` / ``after_tool(call, outcome)`` — every harness tool call, in the bridge
  (every adapter); Way 2: ``governed(..., hooks=)``. ``before_tool`` returns ``None`` (go on),
  :class:`Deny` (not run: the model reads why), :class:`Ask` (a person approves it first, as
  governance's approvals) or :class:`Rewrite` (run with these arguments; the next hook sees
  them). The decision is journaled: a resumed run replays it instead of asking the hooks again.
  ``after_tool`` may return another outcome (what the model reads, journaled and recorded);
* ``on_error(stage, error)`` — a run that failed or ran out of time (``"run"``), a model call
  that failed (``"model"``), a tool call that failed or timed out (``"tool"``);
* ``on_output(item)`` — what leaves for the user, and only that: the answer ``agent.run``,
  ``agent.resume`` and ``RunHandle.result`` return (:attr:`OutputKind.ANSWER`), and every event
  of ``agent.stream``, ``agent.events`` and ``serve_chat``'s AG-UI stream — text deltas, tool
  calls, custom events, the finish with its result (:attr:`OutputKind.EVENT`). What it returns
  is what is sent: mask your users' personal data here. Memory, the journal, A2A, the run's
  log, traces and what the model and the tools see are not touched. It is synchronous (it runs
  on every streamed delta, in order: keep it fast, no I/O), and with no hook overriding it
  nothing is wrapped at all.

A hook that raises fails what it hooks — the call (an error the framework sees), or the run —
except ``on_run_end`` and ``on_error``, which run once the outcome is decided: what they raise
is logged, and ``on_output``, which fails closed: what it raises is logged and the item is sent
as :data:`WITHHELD` instead, never as it was.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal

from trellis.contracts import (
    ConfigurationError,
    RunEvent,
    RunEventType,
    ToolCall,
    ToolOutcome,
    ToolStatus,
)
from trellis.harness.runtime import current

if TYPE_CHECKING:
    from trellis.harness.result import Result
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.hooks")

#: Hook methods ``Hooks`` no longer has, and what to use instead (said when a hook defines one).
REMOVED_HOOKS: Final = dict.fromkeys(
    ("before_model", "after_model"),
    "a model call is each framework's own: ReAct(middleware=[...]) or create_agent's "
    "LangChain middleware, the OpenAI Agents SDK's AgentHooks (Agent(hooks=...)), the Claude "
    "Agent SDK's hooks",
)
#: Where an error was met: the run, a model call, a tool call.
Stage = Literal["run", "model", "tool"]


@dataclass(frozen=True, slots=True)
class Deny:
    """The call is not run; the model reads ``reason``."""

    reason: str


@dataclass(frozen=True, slots=True)
class Ask:
    """The call waits for a person's approval (``assignee``: whose; else anyone's), asked
    ``question`` — as a governance approval: approve, edit, reject or cancel. ``component``
    names your own review screen and ``props`` its data (``Interrupt.component``/``props``,
    passed as they are; a surface without that screen shows the approval)."""

    question: str
    assignee: str | None = None
    component: str | None = None
    props: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Rewrite:
    """The call runs with ``args`` instead (governance decides on them)."""

    args: dict[str, Any]


#: What ``before_tool`` decides.
Verdict = Deny | Ask | Rewrite | None


class OutputKind(StrEnum):
    """What an :class:`Output` holds."""

    #: the run's answer, as ``agent.run``/``agent.resume``/``RunHandle.result`` return it
    ANSWER = "answer"
    #: one event of the run's stream as a user is shown it (``agent.stream``, AG-UI)
    EVENT = "event"


#: What the user is sent in place of an item whose ``on_output`` hook raised.
WITHHELD: Final = "[withheld: an on_output hook failed]"


@dataclass(frozen=True, slots=True)
class Output:
    """One thing leaving for the user: ``value`` is the answer (``ANSWER``) or a
    :class:`~trellis.contracts.RunEvent` (``EVENT``). Return it, or ``replace(value=...)``."""

    kind: OutputKind
    value: Any

    def replace(self, value: Any) -> Output:
        return Output(self.kind, value)

    def withheld(self) -> Output:
        """This item with nothing of what it carried: the answer :data:`WITHHELD`; an event
        of the same type and ids whose data is withheld (a finish keeps its outcome)."""
        if self.kind is OutputKind.ANSWER:
            return self.replace(WITHHELD)
        event: RunEvent = self.value
        data = {"result": WITHHELD} if event.type is RunEventType.RUN_FINISHED else {}
        if "delta" in event.data:
            data["delta"] = WITHHELD
        if "tool" in event.data:
            data["tool"] = event.data["tool"]
        return self.replace(event.model_copy(update={"data": {"withheld": WITHHELD, **data}}))


class Hooks:
    """Subclass and override what you need; each method does nothing by default."""

    async def on_run_start(self, run: Runtime) -> None:
        return None

    async def on_run_end(self, run: Runtime, result: Result) -> None:
        return None

    async def before_tool(self, call: ToolCall) -> Verdict:
        return None

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        return outcome

    async def on_error(self, stage: Stage, error: Exception) -> None:
        return None

    def on_output(self, item: Output) -> Output:
        return item


class Chain:
    """Hooks run in order: what the harness calls them through. Empty, it is false, and the
    harness skips it."""

    def __init__(self, hooks: Sequence[Hooks]) -> None:
        self.hooks = list(hooks)
        for hook in self.hooks:
            for name in REMOVED_HOOKS:
                if hasattr(hook, name):
                    raise ConfigurationError(
                        f"{type(hook).__name__}.{name} was removed: {REMOVED_HOOKS[name]}"
                    )
        #: the hooks that override ``on_output``: none, and nothing sent is wrapped
        self.shows = [h for h in self.hooks if type(h).on_output is not Hooks.on_output]

    def __bool__(self) -> bool:
        return bool(self.hooks)

    async def started(self, run: Runtime) -> None:
        for hook in self.hooks:
            await hook.on_run_start(run)

    async def ended(self, run: Runtime, result: Result) -> None:
        for hook in self.hooks:
            try:
                await hook.on_run_end(run, result)
            except Exception:
                log.exception("on_run_end of %s failed", type(hook).__name__)

    async def tool(self, call: ToolCall) -> tuple[ToolCall, Deny | Ask | None]:
        """The call as the hooks leave it (a :class:`Rewrite` is the next hook's call), and the
        first :class:`Deny` or :class:`Ask`, which the hooks after it are not asked about."""
        for hook in self.hooks:
            verdict = await hook.before_tool(call)
            if isinstance(verdict, Rewrite):
                call = call.model_copy(update={"args": dict(verdict.args)})
            elif verdict is not None:
                return call, verdict
        return call, None

    async def done(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        for hook in self.hooks:
            outcome = await hook.after_tool(call, outcome)
        return outcome

    def shown(self, item: Output) -> Output:
        """``item`` as the ``on_output`` hooks leave it, in order; withheld when one raises."""
        for hook in self.shows:
            try:
                item = hook.on_output(item)
            except Exception:
                log.exception("on_output of %s failed: the item is withheld", type(hook).__name__)
                return item.withheld()
        return item

    def answer(self, result: Result) -> Result:
        """``result`` as its user gets it: the answer through the ``on_output`` hooks."""
        if not self.shows or result.answer is None:
            return result
        shown = self.shown(Output(OutputKind.ANSWER, result.answer))
        return result.model_copy(update={"answer": shown.value})

    def event(self, event: RunEvent) -> RunEvent:
        """``event`` as a user watching the run is shown it (the ``on_output`` hooks)."""
        return self.shown(Output(OutputKind.EVENT, event)).value if self.shows else event

    async def failed(self, stage: Stage, error: Exception) -> None:
        for hook in self.hooks:
            try:
                await hook.on_error(stage, error)
            except Exception:
                log.exception("on_error of %s failed", type(hook).__name__)


def running(*given: Hooks) -> Chain:
    """The hooks of the run this code is in (a wrapped run's: the harness's and its agent's),
    then ``given`` — what a framework's own hook mechanism calls, wrapped or not."""
    runtime = current()
    return Chain([*(runtime.agent.hooks.hooks if runtime is not None else ()), *given])


def denied(call: ToolCall, deny: Deny) -> ToolOutcome:
    """The outcome of a call a hook denied: what the model reads."""
    return ToolOutcome(
        tool=call.tool,
        status=ToolStatus.REJECTED,
        output=f"{call.tool} was not run: {deny.reason}",
        error_class="Denied",
    )


def noted(verdict: Deny | Ask | None) -> dict[str, Any] | None:
    """A decision as the journal keeps it."""
    if verdict is None:
        return None
    return {"kind": type(verdict).__name__, **dataclasses.asdict(verdict)}


def read(note: dict[str, Any] | None) -> Deny | Ask | None:
    """A decision the journal kept (:func:`noted`)."""
    if note is None:
        return None
    fields = {k: v for k, v in note.items() if k != "kind"}
    return Deny(**fields) if note["kind"] == "Deny" else Ask(**fields)


__all__ = [
    "WITHHELD",
    "Ask",
    "Deny",
    "Hooks",
    "Output",
    "OutputKind",
    "Rewrite",
    "Stage",
    "Verdict",
]
