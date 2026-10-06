"""Hooks: your code around a run, its model calls and its tool calls — every adapter, wrapped
or not.

    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Ask | Rewrite | None:
            if call.tool == "refund" and call.args.get("amount", 0) > 1000:
                return Ask("A refund over 1000: approve it?")
            return None

    h = Harness(hooks=[Guard()])                      # every agent of this harness
    agent = h.wrap(target, id="support", hooks=[Audit()])  # this agent, after the harness's

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
* ``before_model(call)`` / ``after_model(call, reply)`` — every model call: ``ReAct``'s own;
  LangChain's and Deep Agents' through their middleware (``middleware.ModelHooks``, given
  to ``create_agent(middleware=[...])``); the OpenAI Agents SDK's through its ``RunHooks``
  (``hooks.openai_agents.ModelHooks``, which the harness passes to ``Runner.run`` itself). A
  ``before_model`` that returns a call rewrites it where the framework lets it (``ReAct``,
  LangChain); the OpenAI Agents SDK reports its calls only. The Claude Agent SDK's model calls
  are the CLI's, and a plain function makes none: no model hooks there;
* ``on_error(stage, error)`` — a run that failed or ran out of time (``"run"``), a model call
  that failed (``"model"``), a tool call that failed or timed out (``"tool"``).

A hook that raises fails what it hooks — the call (an error the framework sees), or the run —
except ``on_run_end`` and ``on_error``, which run once the outcome is decided: what they raise
is logged.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from trellis.contracts import ToolCall, ToolOutcome, ToolStatus
from trellis.harness.runtime import current

if TYPE_CHECKING:
    from trellis.harness.result import Result
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.hooks")

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


@dataclass(frozen=True, slots=True)
class ModelCall:
    """One model call as its framework makes it: ``framework`` (the adapter's name),
    ``messages`` in that framework's own form (chat-completions dicts for ``react``, LangChain
    messages for ``langgraph``, Responses input items for ``openai_agents``), the ``model`` when
    it is named, and the ``system`` prompt where the framework keeps it apart (LangChain's
    ``SystemMessage``, the OpenAI Agents SDK's text)."""

    framework: str
    messages: list[Any]
    model: str | None = None
    system: Any = None


class Hooks:
    """Subclass and override what you need; each method does nothing by default."""

    async def on_run_start(self, run: Runtime) -> None:
        return None

    async def on_run_end(self, run: Runtime, result: Result) -> None:
        return None

    async def before_model(self, call: ModelCall) -> ModelCall | None:
        return None

    async def after_model(self, call: ModelCall, reply: Any) -> None:
        return None

    async def before_tool(self, call: ToolCall) -> Verdict:
        return None

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        return outcome

    async def on_error(self, stage: Stage, error: Exception) -> None:
        return None


class Chain:
    """Hooks run in order: what the harness calls them through. Empty, it is false, and the
    harness skips it."""

    def __init__(self, hooks: Sequence[Hooks]) -> None:
        self.hooks = list(hooks)

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

    async def model(self, call: ModelCall) -> ModelCall:
        """The call as the hooks leave it: each one's rewrite is the next one's call."""
        for hook in self.hooks:
            call = await hook.before_model(call) or call
        return call

    async def answered(self, call: ModelCall, reply: Any) -> None:
        for hook in self.hooks:
            await hook.after_model(call, reply)

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


__all__ = ["Ask", "Deny", "Hooks", "ModelCall", "Rewrite", "Stage", "Verdict"]
