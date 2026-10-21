"""A run's budget: the most it may spend — tokens, dollars, tool calls — counted as it runs and
kept in its journal, so a resume goes on from what was spent, and a sub-agent's spending counts
toward its parent's too. ``h.wrap(..., budget=Budget(...))`` sets it; every run has the default
:class:`Budget` otherwise (``without={"budget"}``: no limit).

Counted: every model reply the harness sees (``ReAct`` and any graph with ``ModelHooks``, an
OpenAI Agents run's hooks) — its input, output and cached tokens, and its dollars: what the
gateway said it cost (Bifrost's ``usage.cost``), else priced from the table
(``trellis.harness.prices``, dated and sourced; a model it has not is counted in tokens and said
once in the log) — and every harness tool call. Only the run's own cap is the harness's: a
tenant's or a team's budget is the gateway's (Bifrost's virtual-key budgets). Enforced: at a
limit no new tool call starts (the calls under way finish), and a ``ReAct`` asks its model once
more, with no tools, for the best answer with what it has; the run ends ``PARTIAL``, its
error ``BUDGET_EXCEEDED``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError
from trellis.harness import prices

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.run")

#: A run's tokens (input and output, every model call) unless its budget says.
DEFAULT_MAX_TOKENS: Final = 2_000_000
#: A run's harness tool calls unless its budget says.
DEFAULT_MAX_TOOL_CALLS: Final = 200
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
    ``max_cost`` in US dollars (``None``: no limit; priced from the price table), and
    ``max_tool_calls`` harness tool calls. A sub-agent's run counts toward its parent's."""

    max_tokens: int | None = DEFAULT_MAX_TOKENS
    max_cost: float | None = None
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_cost", "max_tool_calls"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ConfigurationError(f"a budget's {name} is a number from 0")


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
) -> None:
    """One model reply's spending, on the run and every run it works inside, and the model that
    answered it (as the reply names it), on the run."""
    if cost is None:  # the gateway did not say: the price table's
        price = prices.price_of(model) if model else None
        if price is None and model:
            _unpriced(model)
        cost = (
            prices.dollars(
                price,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=cached,
                written=written,
            )
            if price is not None
            else 0.0
        )
    if model:
        answered = runtime.replay.journal.answered_by
        answered[model] = answered.get(model, 0) + 1
    for run in _runs(runtime):
        spent = run.replay.journal.spent
        spent["tokens"] = spent.get("tokens", 0) + input_tokens + output_tokens
        spent["cached_tokens"] = spent.get("cached_tokens", 0) + cached
        spent["cost"] = spent.get("cost", 0.0) + cost
        spent["model_calls"] = spent.get("model_calls", 0) + 1


def account_reply(runtime: Runtime, reply: Any) -> None:
    """A LangChain model reply's usage (its ``AIMessage``'s ``usage_metadata``), accounted."""
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


def count_tool_call(runtime: Runtime) -> None:
    """One harness tool call, on the run and every run it works inside."""
    for run in _runs(runtime):
        spent = run.replay.journal.spent
        spent["tool_calls"] = spent.get("tool_calls", 0) + 1


def reached(runtime: Runtime) -> str | None:
    """The limit the run — or a run it works inside — has reached (``None``: none), unless it
    is without its budget."""
    if not runtime.uses("budget"):
        return None
    for run in _runs(runtime):
        budget: Budget = run.agent.budget
        spent = run.replay.journal.spent
        for name, used in (
            ("max_tokens", spent.get("tokens", 0)),
            ("max_cost", spent.get("cost", 0.0)),
            ("max_tool_calls", spent.get("tool_calls", 0)),
        ):
            limit = getattr(budget, name)
            if limit is not None and used >= limit:
                return f"{name} {limit:g} reached ({used:g} spent)"
    return None


def stopped(runtime: Runtime, spent: str) -> None:
    """The run stopped at a budget — its own or that of a run it works inside: it and every
    run up to the one whose limit it was end ``PARTIAL`` (a parent whose budget cut its
    sub-agent's work short has a partial answer too)."""
    runs = list(_runs(runtime))
    last = next((n for n, run in enumerate(runs) if _over(run, run.agent.budget)), len(runs))
    for run in runs[: last + 1]:
        run.budget_spent = spent


def _over(run: Runtime, budget: Budget) -> bool:
    spent = run.replay.journal.spent
    return any(
        limit is not None and used >= limit
        for limit, used in (
            (budget.max_tokens, spent.get("tokens", 0)),
            (budget.max_cost, spent.get("cost", 0.0)),
            (budget.max_tool_calls, spent.get("tool_calls", 0)),
        )
    )


@functools.cache
def _unpriced(model: str) -> None:
    log.warning("%s is not in the price table: its tokens are counted, its dollars are not", model)


__all__ = [
    "BUDGET_EXCEEDED",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TOOL_CALLS",
    "SPENT",
    "Budget",
    "account",
    "account_reply",
    "count_tool_call",
    "gateway_cost",
    "reached",
    "stopped",
]
