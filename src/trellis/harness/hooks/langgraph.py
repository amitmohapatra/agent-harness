"""The run's LangChain callbacks for the helpers a graph spawns itself: LangGraph hands the
run's callbacks to everything it runs inside it — its subgraphs, its ``Send`` branches, a Deep
Agent's ``task`` sub-agents — so a ``task`` call is seen starting and ending here, and is a
native helper of the run (``bridges.helpers``: its run record, its turns). What every model
call spends is counted by ``accounting``, the one owner of that."""

from __future__ import annotations

from typing import Any, Final
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from trellis.contracts.errors import is_pause_signal
from trellis.harness.bridges.helpers import Helpers, Kind, text_of

#: Deep Agents' sub-agent tool, and the arguments it takes (``SubAgentMiddleware``).
TASK: Final = "task"
TASK_ARGUMENTS: Final = frozenset({"description", "subagent_type"})


class HelperCallbacks(AsyncCallbackHandler):
    """The helper callbacks of one attempt of a graph's run."""

    #: a helper's record is written before its sub-agent runs, and a failure to write it is
    #: the run's (LangChain would only log it)
    run_inline = True
    raise_error = True

    def __init__(self, helpers: Helpers) -> None:
        self.helpers = helpers

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
