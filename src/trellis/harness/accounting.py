"""Every LangChain chat model call made inside a harness run, accounted to the run's budget
(``trellis.harness.budget``): a LangChain callback, attached to every LangChain run of the
process (``register_configure_hook``), that accounts each reply's usage — whatever graph made
the call (``ReAct``'s loop, its summarization, a grader, a ``create_agent`` or LangGraph graph
of the team's own, with or without ``ModelHooks``). Outside a harness run it does nothing."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tracers.context import register_configure_hook

from trellis.harness import budget
from trellis.harness.runtime import current


class UsageAccounting(BaseCallbackHandler):
    """Each chat model reply's usage (and the gateway's cost), accounted to the current run."""

    run_inline = True  # in the run's own task: ``current()`` is the run

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        runtime = current()
        if runtime is None:
            return
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if isinstance(message, AIMessage):
                    budget.account_reply(runtime, message)


#: The callback, on every LangChain run of the process (inherited by the runs it starts).
ACCOUNTING: ContextVar[UsageAccounting | None] = ContextVar(
    "trellis_usage_accounting",
    default=UsageAccounting(),  # noqa: B039 - it holds no state: one for every context
)
register_configure_hook(ACCOUNTING, inheritable=True)
