"""The run's LangChain callbacks: the hook every model call and tool call of a graph's run
reaches — its subgraphs, its ``Send`` branches and a Deep Agent's ``task`` sub-agents included,
since LangGraph hands the run's callbacks to everything it runs inside it.

* every model reply's usage (and the gateway's cost) counts toward the run's budget
  (``budget.py``) — the ``ModelHooks`` middleware then counts nothing again;
* a Deep Agent's ``task`` call is a native helper of the run (``bridges.helpers``): its run
  record (lineage) and its turns.
"""

from __future__ import annotations

from typing import Any, Final
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from trellis.contracts.errors import is_pause_signal
from trellis.harness import budget
from trellis.harness.bridges.helpers import Helpers, Kind, text_of
from trellis.harness.runtime import Runtime

#: Deep Agents' sub-agent tool, and the arguments it takes (``SubAgentMiddleware``).
TASK: Final = "task"
TASK_ARGUMENTS: Final = frozenset({"description", "subagent_type"})


class GraphCallbacks(AsyncCallbackHandler):
    """The callbacks of one attempt of a graph's run (``runtime``)."""

    #: a helper's record is written before its sub-agent runs, and a failure to write it is
    #: the run's (LangChain would only log it)
    run_inline = True
    raise_error = True

    def __init__(self, runtime: Runtime, helpers: Helpers) -> None:
        self.runtime = runtime
        self.helpers = helpers

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            message = getattr(generations[0], "message", None) if generations else None
            if message is not None:
                budget.account_reply(self.runtime, message)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if (serialized or {}).get("name") == TASK and set(inputs or {}) >= TASK_ARGUMENTS:
            assert inputs is not None
            await self.helpers.start(
                str(run_id),
                kind=Kind.TASK,
                name=str(inputs["subagent_type"]),
                task=str(inputs["description"]),
            )

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        await self.helpers.end(str(run_id), output=text_of(output))

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if not is_pause_signal(error):  # a pause: the next attempt ends it
            await self.helpers.end(str(run_id), error=f"{type(error).__name__}: {error}")
