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

import functools
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError
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
    cost: float | None = None,
) -> None:
    """One model reply's spending, on the run and every run it works inside."""
    capped = any(r.budget is not None and r.budget.max_cost for r in _runs(runtime))
    if cost is None and capped:
        _no_cost(model or "a model")
    for run in _runs(runtime):
        spent = run.replay.journal.spent
        spent["tokens"] = spent.get("tokens", 0) + input_tokens + output_tokens
        spent["cached_tokens"] = spent.get("cached_tokens", 0) + cached
        spent["cost"] = spent.get("cost", 0.0) + (cost or 0.0)
        spent["model_calls"] = spent.get("model_calls", 0) + 1


def account_reply(runtime: Runtime, reply: Any) -> None:
    """A LangChain model reply's usage (its ``AIMessage``'s ``usage_metadata``, and the cost
    the gateway put in its ``token_usage``), accounted."""
    from langchain_core.messages import AIMessage  # noqa: PLC0415 - the react extra

    messages = getattr(reply, "result", None) or [reply]
    message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if message is None:
        return
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    metadata = message.response_metadata or {}
    account(
        runtime,
        model=metadata.get("model_name"),
        cost=gateway_cost(metadata),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached=int((usage.get("input_token_details") or {}).get("cache_read") or 0),
    )


def gateway_cost(metadata: dict[str, Any]) -> float | None:
    """What the gateway said a reply cost, in US dollars (Bifrost's ``usage.cost.total_cost``,
    which LangChain keeps in the reply's ``token_usage``), if it said."""
    usage = metadata.get("token_usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    total = cost.get("total_cost") if isinstance(cost, dict) else None
    return float(total) if isinstance(total, int | float) and not isinstance(total, bool) else None


def count_tool_call(runtime: Runtime) -> None:
    """One harness tool call, on the run and every run it works inside."""
    for run in _runs(runtime):
        spent = run.replay.journal.spent
        spent["tool_calls"] = spent.get("tool_calls", 0) + 1


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
    "SPENT",
    "Budget",
    "account",
    "account_reply",
    "count_tool_call",
    "gateway_cost",
    "reached",
    "stopped",
]
