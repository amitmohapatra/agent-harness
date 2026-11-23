"""A run's budget: the most it may spend — tokens, dollars, tool calls — counted as it runs and
kept in its journal, so a resume goes on from what was spent, and a sub-agent's spending counts
toward its parent's too. ``h.wrap(..., budget=Budget(...))`` sets it for every run of the
agent, ``agent.run(..., budget=...)`` for one run (kept with its record, for its later
attempts); a run given none has no cap (it is still counted). Only the run's own cap is the
harness's: a tenant's or a team's budget is the gateway's (Bifrost's virtual-key budgets).

Counted: every model reply the harness sees (``ReAct`` and any graph with ``ModelHooks``, an
OpenAI Agents run's hooks) — its input, output and cached tokens, and the dollars the gateway
says it cost (Bifrost's ``usage.cost``; a reply it put no cost on is counted in tokens, said
once in the log) — and every harness tool call. Enforced: at a limit no new tool call starts
(the calls under way finish), and a ``ReAct`` asks its model once more, with no tools, for the
best answer with what it has; the run ends ``PARTIAL``, its error ``BUDGET_EXCEEDED``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError
from trellis.harness import telemetry
from trellis.harness.bounded import SAID_ONCE

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.run")

#: The ``RunStart.metadata`` key a run's own budget is kept under.
BUDGET: Final = "budget"
#: The error code of a run that ended at its budget.
BUDGET_EXCEEDED: Final = "BUDGET_EXCEEDED"
#: What a run's model is told when its budget is spent.
SPENT: Final = (
    "This run's budget is spent. Do not call tools: answer now with what you have, and say "
    "that the budget was reached and what is left undone."
)
#: What a run's model is told when its step limit is reached (``ReAct``'s ``max_steps``): its
#: one last call, with no tool, answers.
LIMIT: Final = (
    "This run has reached its limit of {limit} steps. Do not call tools: answer now with what "
    "you found, and say what is left undone."
)
#: The same, told in a framework's session that the conversation's next turn continues (ACP):
#: kept there, :data:`LIMIT` read as still in force, and a later turn answered
#: "I've reached the tool-call limit" without calling a tool.
TURN_LIMIT: Final = (
    "This turn has reached its limit of {limit} steps. Do not call tools now: answer with what "
    "you found, and say what is left undone. The limit is for this turn only: the user's next "
    "message starts a new turn, with your tools."
)
#: What a model is told when its reply was cut at its output limit (its ``max_tokens``) before
#: it gave an answer — the reply spent on working, such as rows summed by hand (the
#: break-and-fix gate: two LangGraph turns at 4,096 tokens) —: its one more call, with no tool,
#: answers.
CUT_OFF: Final = (
    "Your last reply reached the output limit before it gave an answer. Do not call tools now "
    "and do not write out data or working: answer in a few sentences with what you found, and "
    "say what is left undone."
)
#: Why a run that would end with no answer ends ``ERROR`` instead (``pipeline``): an empty
#: answer reported as a success hid the failure (the break-and-fix gate's two LangGraph turns).
NO_ANSWER: Final = "NO_ANSWER"


@dataclass(frozen=True)
class Budget:
    """The most one run may spend: ``max_tokens`` (input and output, every model call),
    ``max_cost`` in US dollars as the gateway reports them, and ``max_tool_calls`` harness tool
    calls (each ``None``: no limit). A sub-agent's run counts toward its parent's."""

    max_tokens: int | None = None
    max_cost: float | None = None
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_cost", "max_tool_calls"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ConfigurationError(f"a budget's {name} is a number from 0")

    def kept(self) -> dict[str, Any]:
        """What a run's record keeps of it (its limits that are set)."""
        found = {
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "max_tool_calls": self.max_tool_calls,
        }
        return {name: value for name, value in found.items() if value is not None}

    @classmethod
    def of(cls, kept: Any) -> Budget | None:
        """It again, from a run's record (``None``: the run kept none)."""
        return cls(**kept) if isinstance(kept, dict) else None

    def over(self, spent: dict[str, float]) -> str | None:
        """The limit ``spent`` has reached, said (``None``: none)."""
        for name, used in (
            ("max_tokens", spent.get("tokens", 0)),
            ("max_cost", spent.get("cost", 0.0)),
            ("max_tool_calls", spent.get("tool_calls", 0)),
        ):
            limit = getattr(self, name)
            if limit is not None and used >= limit:
                return f"{name} {limit:g} reached ({used:g} spent)"
        return None


#: What a run's end says it spent (``spent``), each counted from 0.
SPENT_KEYS: Final = (
    "tokens",
    "cached_tokens",
    "cache_written_tokens",
    "cost",
    "model_calls",
    "tool_calls",
)


def spent(runtime: Runtime) -> dict[str, float]:
    """What the run has spent, its sub-agents' runs and its earlier attempts included: its
    tokens (input and output), the input tokens read from and written to the prompt cache,
    dollars (as the gateway said them), model calls
    and harness tool calls — what its ``RUN_FINISHED`` event carries (``spent``)."""
    kept = runtime.replay.journal.spent
    return {name: kept.get(name, 0.0 if name == "cost" else 0) for name in SPENT_KEYS}


def _runs(runtime: Runtime) -> Iterator[Runtime]:
    """The run and the runs it works inside (a sub-agent's parent, theirs)."""
    at: Runtime | None = runtime
    while at is not None:
        yield at
        at = at.parent


def account(
    runtime: Runtime,
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cached: int = 0,
    written: int = 0,
    cost: float | None = None,
    calls: int = 1,
) -> None:
    """One model reply's spending, on the run and every run it works inside: ``input_tokens``
    counts every prompt token, of which ``cached`` were read from the provider's prompt cache
    and ``written`` written to it (priced apart: a cache write costs more than plain input);
    ``calls``: 0 for a cost that came apart from its replies. A reply in a thread (a sync
    graph node's model) is accounted on the run's loop, its next turn: never while the loop
    reads the spending to save it (``Runtime.spent_more``)."""
    spent: dict[str, Any] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached": cached,
        "written": written,
        "cost": cost,
        "calls": calls,
    }
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if runtime.running_in is not None:
            loop = runtime.running_in.get_loop()
            loop.call_soon_threadsafe(functools.partial(_account, runtime, **spent))
            return
    _account(runtime, **spent)


def _account(
    runtime: Runtime,
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cached: int,
    written: int,
    cost: float | None,
    calls: int,
) -> None:
    if model and calls and runtime.called is None:
        runtime.called = model  # the model its framework calls, when no builder said
    capped = any(r.budget is not None and r.budget.max_cost for r in _runs(runtime))
    if cost is None and capped:
        _no_cost(model or "a model")
    for run in _runs(runtime):
        spent = run.replay.journal.spent
        spent["tokens"] = spent.get("tokens", 0) + input_tokens + output_tokens
        spent["cached_tokens"] = spent.get("cached_tokens", 0) + cached
        spent["cache_written_tokens"] = spent.get("cache_written_tokens", 0) + written
        spent["cost"] = spent.get("cost", 0.0) + (cost or 0.0)
        spent["model_calls"] = spent.get("model_calls", 0) + calls
    runtime.spent_more()  # saved at once: a worker that dies next has it said


def reported(runtime: Runtime, **spent: Any) -> None:
    """A model reply its framework made and reported, which no span of the harness's covers
    (Claude Code's, Kimi Code's, an OpenAI agent's): accounted as :func:`account` takes it,
    and a ``chat`` generation with its usage and cost in the run's trace
    (``telemetry.generation``)."""
    account(runtime, **spent)
    telemetry.generation(
        spent["model"],
        input_tokens=spent["input_tokens"],
        output_tokens=spent["output_tokens"],
        cached=spent.get("cached", 0),
        written=spent.get("written", 0),
        cost=spent.get("cost"),
        calls=spent.get("calls", 1),
    )


def account_reply(runtime: Runtime, reply: Any) -> None:
    """A LangChain model reply's usage (its ``AIMessage``'s ``usage_metadata``, and the cost
    the gateway put in its ``token_usage``), accounted."""
    from langchain_core.messages import AIMessage  # noqa: PLC0415 - the react extra

    messages = getattr(reply, "result", None) or [reply]
    message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if message is None:
        return
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    details = usage.get("input_token_details") or {}
    metadata = message.response_metadata or {}
    account(
        runtime,
        model=metadata.get("model_name"),
        cost=gateway_cost(metadata),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached=int(details.get("cache_read") or 0),
        written=int(details.get("cache_creation") or 0),
    )


def gateway_cost(metadata: dict[str, Any]) -> float | None:
    """What the gateway said a reply cost, in US dollars (Bifrost's ``usage.cost.total_cost``,
    which LangChain keeps in the reply's ``token_usage``), if it said."""
    usage = metadata.get("token_usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    total = cost.get("total_cost") if isinstance(cost, dict) else None
    return float(total) if isinstance(total, int | float) and not isinstance(total, bool) else None


def count_tool_call(runtime: Runtime, *, times: int = 1) -> None:
    """One harness tool call, on the run and every run it works inside (``times=-1``: one
    counted that is not over — it asked a person, and is made again when the run resumes)."""
    for run in _runs(runtime):
        spent = run.replay.journal.spent
        spent["tool_calls"] = spent.get("tool_calls", 0) + times


def reached(runtime: Runtime) -> str | None:
    """The limit the run — or a run it works inside — has reached (``None``: none, or no
    budget)."""
    said = (_over(run) for run in _runs(runtime))
    return next((s for s in said if s is not None), None)


def stopped(runtime: Runtime, spent: str) -> None:
    """The run stopped at a budget — its own or that of a run it works inside: it and every
    run up to the one whose limit it was end ``PARTIAL`` (a parent whose budget cut its
    sub-agent's work short has a partial answer too)."""
    runs = list(_runs(runtime))
    last = next((n for n, run in enumerate(runs) if _over(run)), len(runs))
    for run in runs[: last + 1]:
        run.budget_spent = spent


def _over(run: Runtime) -> str | None:
    cap: Budget | None = run.budget
    return cap.over(run.replay.journal.spent) if cap is not None else None


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per name, of the most recent
def _no_cost(model: str) -> None:
    log.warning(
        "the gateway put no cost on %s's replies: their tokens are counted, their dollars are "
        "not (a max_cost budget does not see them)",
        model,
    )


__all__ = [
    "BUDGET",
    "BUDGET_EXCEEDED",
    "LIMIT",
    "SPENT",
    "SPENT_KEYS",
    "TURN_LIMIT",
    "Budget",
    "account",
    "account_reply",
    "count_tool_call",
    "gateway_cost",
    "reached",
    "spent",
    "stopped",
]
