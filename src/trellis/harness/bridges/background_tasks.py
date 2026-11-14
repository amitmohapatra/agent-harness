"""``BackgroundTasks``: the harness's background tasks (``trellis.harness.background``) as a
LangChain middleware with Deep Agents' own async sub-agent tools — ``start_async_task``,
``check_async_task``, ``update_async_task``, ``cancel_async_task`` and ``list_async_tasks``, their
arguments Deep Agents' (``StartAsyncTaskSchema``...), its task list in the graph's state
(``AsyncSubAgentState``: ``async_tasks``, its ``AsyncTask`` records) — for sub-agents that run in
the harness: a ``ReAct``'s wrapped agents and Deep Agents' own sub-agents (``SubAgent``,
``CompiledSubAgent``). The model sees the interface Deep Agents gives a sub-agent on a server;
the task runs here, with no server to host.

As Deep Agents' ``AsyncSubAgentMiddleware`` does, it reads its tasks' outcomes itself: when a
turn starts (``abefore_agent``), before each model call (``abefore_model``) and after a final
answer (``aafter_model``, giving the model another turn when one came), each new outcome a
notification after the conversation's messages (``background.notification``). A turn starts
by asking the questions its tasks wait on, in a run a person reads (``Runtime.attended``),
and handing the answers on (``background.ask``, ``background.answered``). The tools are offered
where ``ask_user`` is — a chat a person reads, where a turn of its own can be delivered — and a
run elsewhere hands its sub-agents tasks in the foreground (``task``, a wrapped agent's tool).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from deepagents.middleware.async_subagents import (
    AsyncSubAgentState,
    CancelAsyncTaskSchema,
    CheckAsyncTaskSchema,
    ListAsyncTasksSchema,
    StartAsyncTaskSchema,
    UpdateAsyncTaskSchema,
)
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import hook_config
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from trellis.harness import background
from trellis.harness.adapters.langgraph import notified
from trellis.harness.agent import Agent
from trellis.harness.automatic import without_tools
from trellis.harness.background import TOOLS
from trellis.harness.middleware import ASYNC_TASK_DONE
from trellis.harness.runtime import current

#: The state key of a ``ReAct`` graph's model calls so far in the run (LangChain's
#: ``ModelCallLimitMiddleware``, counted per thread: a ``ReAct``'s thread is its run).
STEPS: Final = "thread_model_call_count"
#: Deep Agents' argument schema of each tool.
SCHEMAS: Final[Mapping[str, type]] = {
    "start_async_task": StartAsyncTaskSchema,
    "check_async_task": CheckAsyncTaskSchema,
    "update_async_task": UpdateAsyncTaskSchema,
    "cancel_async_task": CancelAsyncTaskSchema,
    "list_async_tasks": ListAsyncTasksSchema,
}


class BackgroundTasks(AgentMiddleware):
    """Deep Agents' five async sub-agent tools for ``subagents`` (``background.Offered``: a
    wrapped agent, or a Deep Agents sub-agent spec whose compiled graph the ``task`` tool runs,
    given by :meth:`compiled` once the graph is built), run in the harness, the task list in the
    graph's state. ``described`` and ``strategies`` as ``Offered`` takes them; ``max_steps``,
    the graph's step limit (``ReAct``'s, counted per run): past it no model turn is left after
    a final answer, and an outcome is told after the turn (``background.left``) rather than
    read into a turn the model never gets."""

    state_schema = AsyncSubAgentState

    def __init__(
        self,
        subagents: Mapping[str, Agent | None],
        *,
        described: Mapping[str, str],
        strategies: Mapping[str, str] | None = None,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.offered = background.Offered(subagents, described=described, strategies=strategies)
        self.tools = [self._tool(name) for name in TOOLS]

    def compiled(self, graphs: Mapping[str, Any]) -> None:
        """The compiled graphs of the Deep Agents sub-agents among the sub-agents, by name
        (``adapters.langgraph.named_subagent_graphs``)."""
        self.offered.found(graphs)

    def _tool(self, name: str) -> Any:
        """One of the five tools as a LangChain tool: Deep Agents' name and arguments, its work
        the harness's (``Offered.call``), saying what it does (``background.SIDE_EFFECTS``; the
        result of a start, an update and a cancel a record of what was done:
        ``middleware.ASYNC_TASK_DONE``)."""
        offered = self.offered

        async def run(runtime: ToolRuntime, **args: Any) -> Any:
            tasks = runtime.state.get("async_tasks") or {}
            said, changed = await offered.call(name, current(), tasks, args)
            if not changed:
                return said
            message = ToolMessage(said, tool_call_id=runtime.tool_call_id)
            return Command(update={"messages": [message], "async_tasks": changed})

        done = ASYNC_TASK_DONE if name in background.DONE else {}
        return StructuredTool.from_function(
            name=name,
            coroutine=run,
            description=offered.descriptions[name],
            infer_schema=False,
            args_schema=SCHEMAS[name],
            metadata={**done, "side_effects": background.SIDE_EFFECTS[name]},
        )

    # ------------------------------------------------------------------ offered where they work
    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        if runtime is not None and runtime.attended:
            return await handler(request)
        return await handler(without_tools(request, TOOLS))

    # ------------------------------------------------------------------ the pull
    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """A turn starts: the outcome a turn of the chat's own was started for recorded (it
        is told once), the questions the tasks wait on asked (in a run a person reads), then
        the new outcomes told."""
        run = current()
        if run is None:
            return None
        messages = state.get("messages") or []
        noted = (notified(messages[-1]) if messages else None) or {}
        event = noted.get("data") if noted.get("source") == background.SOURCE else None
        recorded = background.recorded(event, state.get("async_tasks") or {})
        tasks = {**(state.get("async_tasks") or {}), **recorded}
        if run.attended:
            await self.offered.asked(run, tasks)
        update = await _pulled(run, tasks)
        update["async_tasks"] = {**recorded, **update.get("async_tasks", {})}
        return update if update["async_tasks"] or update.get("messages") else None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        run = current()
        if run is None:
            return None
        return await _pulled(run, state.get("async_tasks") or {}) or None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """After a final answer, the tasks read once more: a new outcome gives the model another
        turn (each needs an outcome of its own; it counts as any model call, against
        ``max_steps`` and the run's budget)."""
        run = current()
        messages = state.get("messages") or []
        if run is None or not messages or getattr(messages[-1], "tool_calls", None):
            return None
        if self.max_steps is not None and state.get(STEPS, 0) > self.max_steps:
            return None  # no model turn is left: the outcome is told after the turn instead
        update = await _pulled(run, state.get("async_tasks") or {})
        return {**update, "jump_to": "model"} if update.get("messages") else update or None


async def _pulled(run: Any, tasks: Mapping[str, Any]) -> dict[str, Any]:
    """``background.pulled`` as a graph state update: the records, and a notification message
    per new outcome (``background.notification``)."""
    updated, told = await background.pulled(run, tasks)
    found: dict[str, Any] = {"async_tasks": updated} if updated else {}
    if told:
        found["messages"] = [background.notification(update) for update in told]
    return found
