"""LangGraph compiled graphs — and Deep Agents, whose ``create_deep_agent`` returns one.

* input: ``{"messages": [...]}`` with the memory context as a leading system message (a
  string input becomes one user message; any other state dict passes through untouched).
  The context message has a fixed id, so a checkpointed thread holds at most one — each turn
  replaces it in place (``add_messages``) — and it leaves out the recent conversation, which
  the checkpointer already holds;
* run: ``ainvoke``/``astream`` with ``version="v2"``, on the LangGraph thread named by the
  run's thread (or the run id);
* pause: with a checkpointer, ``trellis.current().ask`` *is* LangGraph's ``interrupt``, and a
  resume is ``Command(resume=...)``; without one the run is re-executed from its input and the
  journal answers the questions already asked. A resume where the checkpointer no longer holds
  the pause (an ``InMemorySaver`` in another process: a worker, a replica) answers a harness
  pause from the journal the same way (:func:`holds`); a graph's own pause cannot be answered
  there, and the run fails saying so. LangChain's ``HumanInTheLoopMiddleware`` (Deep
  Agents' ``interrupt_on``) pauses with its own request: it is an approval of the calls it
  holds, and the harness's decision is turned into the ``{"decisions": [...]}`` it resumes
  with (:func:`hitl_response`);
* tools: fixed when the graph is compiled, so harness tools come from ``h.tools(...)`` at
  build time — unless the graph takes them per model call (``middleware.HarnessTools``, as
  ``ReAct(...)`` does): then ``h.wrap(tools=)`` gives them, every model call is offered the
  tools the run offers at that moment (``narrows="turn"``), and the memory tools come with them.
  A ``create_agent`` graph (Deep Agents' too) built with ``h.tools`` narrows per model call
  as well: the harness runs a copy of it whose model calls go through one more
  ``awrap_model_call`` handler (``middleware.narrowing``), so the tool hints and a part turned
  off reach the model; a hand-written graph binds its model's tools in its own code, which
  the harness cannot reach (``narrows="none"``: every tool it bound is offered);
* a graph that checkpoints into its run (``middleware.RunCheckpointer``, as ``ReAct(...)``
  does) runs on the run's own thread (``thread_id`` is the run id): its checkpoint is in the
  run's journal, so a resume — in any process, after a pause or a crash — continues where it
  stopped, and it holds no conversation of its own (the memory service gives the recent turns);
* store: a graph compiled without a store of its own is run with the memory service as its
  long-term store (``bridges.store.MemoryStore``: ``runtime.store``, ``get_store()``, Deep
  Agents' ``StoreBackend``) while the run has memory and its memory tools; a graph's own store
  is kept (said once);
* framework options: the ``RunnableConfig`` keys the run is given (``recursion_limit``,
  ``configurable``, ``tags``...) go into the config of ``ainvoke``/``astream``; the harness's
  ``configurable.thread_id`` wins over one given. The graph's own run arguments
  (:data:`RUN_ARGUMENTS`: ``context`` for ``create_agent(context_schema=)``, ``durability``) go
  to the call beside it;
* streaming: the text of each AI message chunk (a string or content blocks), and what a node or
  tool writes to the graph's stream (``get_stream_writer()``) as a ``graph_write`` event.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar, Final

from trellis.contracts import ConfigurationError, InterruptDecision, InterruptResolution
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    NativePause,
    Output,
    ToolFormat,
)
from trellis.harness.asking import answer_of
from trellis.harness.bridges import said
from trellis.harness.bridges.helpers import Helpers
from trellis.harness.bridges.history import Thread, owns, thread_for
from trellis.harness.journal import Journal, Pending
from trellis.harness.runtime import DEFERRED, MARKER, Runtime, reason_of

#: The journal key a graph's own ``interrupt(...)`` (not ``ask``) is filed under.
FOREIGN: Final = "langgraph"
#: The journal key of a ``HumanInTheLoopMiddleware`` pause (Deep Agents' ``interrupt_on``).
HITL: Final = "langchain_hitl"
#: The state key a graph declares when it takes the run's tools per model call
#: (``middleware.HarnessTools``; kept here so detecting it imports no LangChain).
HARNESS_TOOLS: Final = "trellis_harness_tools"
#: The id of the memory context message: one per thread, replaced every turn.
CONTEXT_MESSAGE_ID: Final = "trellis-memory-context"
#: The keyword arguments of ``ainvoke``/``astream`` a run may give beside its config: the graph's
#: runtime context (``create_agent(context_schema=)``, a node's ``runtime.context``) and how it
#: saves its checkpoints.
RUN_ARGUMENTS: Final = ("context", "durability")
#: The ``CUSTOM`` event name of what a node or tool wrote to the graph's stream
#: (``get_stream_writer()``).
GRAPH_WRITE: Final = "graph_write"
#: The qualified name of a ``create_agent`` graph's async model node (Deep Agents' too).
MODEL_NODE: Final = "create_agent.<locals>.amodel_node"


class LangGraphAdapter:
    name: ClassVar[str] = "langgraph"

    def __init__(self, target: Any = None) -> None:
        #: the graph takes the run's tools per model call (``middleware.HarnessTools``): it
        #: converts them itself, so nothing is converted for it here
        per_call = HARNESS_TOOLS in (getattr(target, "channels", None) or {})
        self.tool_format: ToolFormat = "none" if per_call else "langchain"
        self.fixed_tools = not per_call
        #: the graph run in its place: a ``create_agent`` graph whose model calls narrow its
        #: bound tools to those the run offers (``None``: ``target`` itself is run)
        self.per_call = per_call
        self.narrowed = None if per_call else _narrowed(target)
        per_turn = per_call or self.narrowed is not None
        self.narrows: Narrowing = "turn" if per_turn else "none"

    def runs(self, target: Any, *, agent_id: str, memory: bool) -> Any:
        """The graph the harness runs: ``target``, or — where the history policy puts the
        framework on the run's own thread (``bridges.history``: ``Thread.RUN``) — a copy of it on
        the harness's ``RunCheckpointer``, its checkpoint in the run's journal."""
        if run_scoped(target):
            return target
        own = target.checkpointer if _checkpointed(target) else None
        if thread_for(own, bridged=False, agent_id=agent_id, memory=memory) is not Thread.RUN:
            return target
        from trellis.harness.middleware import RunCheckpointer

        graph = target.copy({"checkpointer": RunCheckpointer()})
        self.narrowed = None if self.per_call else _narrowed(graph)
        return graph

    def keeps_conversation(self, target: Any, runtime: Runtime) -> bool:
        """Whether the graph's own checkpointer owns the conversation's history
        (``bridges.history``: a durable one does; the run's own, or one kept in RAM, does not)."""
        if run_scoped(target):
            return False
        own = target.checkpointer if _checkpointed(target) else None
        memory = runtime.run_memory is not None
        return owns(thread_for(own, bridged=False, agent_id=runtime.agent_id, memory=memory))

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any:
        from langchain_core.messages import SystemMessage

        system = [SystemMessage(content=context, id=CONTEXT_MESSAGE_ID)] if context else []
        if isinstance(input, str):
            return {"messages": [*system, {"role": "user", "content": input}]}
        if isinstance(input, list):
            return {"messages": [*system, *input]}
        if isinstance(input, dict) and isinstance(input.get("messages"), list):
            return {**input, "messages": [*system, *input["messages"]]}
        return input

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        helpers = Helpers(run.runtime, self.name)
        error: BaseException | None = None
        graph, config = self._graph(target, run), self._config(target, run, helpers)
        try:
            return await graph.ainvoke(
                await _continued(graph, target, native_input, run, config),
                config,
                version="v2",
                **_run_arguments(run),
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error)

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from langchain_core.messages import AIMessage
        from langgraph.types import GraphOutput

        values: Any = None
        interrupts: tuple[Any, ...] = ()
        helpers = Helpers(run.runtime, self.name)
        error: BaseException | None = None
        graph, config = self._graph(target, run), self._config(target, run, helpers)
        try:
            async for part in graph.astream(
                await _continued(graph, target, native_input, run, config),
                config,
                stream_mode=["messages", "values", "custom"],
                version="v2",
                **_run_arguments(run),
            ):
                if part["type"] == "messages":
                    chunk = part["data"][0]
                    # the text of a string or of content blocks (an Anthropic model's, reasoning's)
                    text = chunk.text if isinstance(chunk, AIMessage) else ""
                    if text:
                        yield text
                elif part["type"] == "custom":
                    run.runtime.events.custom(GRAPH_WRITE, value=part["data"])
                else:  # "values"
                    values = part["data"]
                    interrupts = part["interrupts"] or interrupts
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error)
        yield Output(GraphOutput(value=values, interrupts=interrupts))

    def extract(self, target: Any, output: Any) -> Extracted:
        state = output.value
        answer: Any = None
        if isinstance(state, dict):
            answer = state.get("structured_response")
            if answer is None:
                answer = _last_ai_text(state.get("messages") or [])
        transcript = [("assistant", answer)] if isinstance(answer, str) and answer else []
        pause = None
        # a call waiting for its turn behind a pause asks nothing (``DEFERRED``)
        raised = [
            i
            for i in output.interrupts
            if not (isinstance(i.value, dict) and i.value.get(DEFERRED))
        ]
        if raised:
            ours = [i for i in raised if isinstance(i.value, dict) and i.value.get(MARKER)]
            if not ours and not _checkpointed(target):
                raise ConfigurationError(
                    "the graph called interrupt() without a checkpointer, so it cannot be "
                    "resumed: compile it with one, or ask through trellis.current().ask"
                )
            first = (ours or raised)[0]
            # calls made at once may each have asked: the pause names each one's interrupt
            ids = {i.value.get("interrupt_id"): i.id for i in ours}
            pause = NativePause(value=first.value, native_id=first.id, ids=ids)
        return Extracted(answer=answer, transcript=transcript, pause=pause)  # type: ignore[arg-type]

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        if pending.native_id is None or not _checkpointed(target):
            return native_input
        from langgraph.types import Command

        if pending.key == HITL:
            value: Any = hitl_response(pending.interrupt.payload or {}, resolution)
        elif pending.key == FOREIGN:
            value = answer_of(resolution)
        else:
            value = resolution.model_dump(mode="json")
        return Command(resume={pending.native_id: value})

    def check_options(self, options: Mapping[str, Any]) -> None:
        from langchain_core.runnables import RunnableConfig

        known = [*RunnableConfig.__annotations__, *RUN_ARGUMENTS]
        unknown = [key for key in options if key not in known]
        if unknown:
            raise ConfigurationError(
                f"framework_options for a graph are RunnableConfig keys and the run's own "
                f"arguments: {', '.join(map(repr, unknown))} is none of {', '.join(known)}"
            )
        if not isinstance(options.get("configurable", {}), Mapping):
            raise ConfigurationError("framework_options' configurable is a dict")

    def _graph(self, target: Any, run: Invocation) -> Any:
        """The graph run for ``target``: its narrowing copy, when it has one — given the
        run's store (the memory service: ``bridges.store``) when it has none of its own."""
        graph = target if self.narrowed is None else self.narrowed
        store = run_store(target, run.runtime)
        return graph if store is None else graph.copy({"store": store})

    @staticmethod
    def _config(target: Any, run: Invocation, helpers: Helpers) -> dict[str, Any]:
        """The run's config: its framework options, the harness's thread id over theirs, and
        the run's callbacks (``hooks.langgraph.GraphCallbacks``: its spending, its helpers)
        beside theirs."""
        from trellis.harness.hooks.langgraph import GraphCallbacks

        runtime = run.runtime
        if _checkpointed(target):
            from langgraph.types import interrupt

            runtime.suspend = interrupt
        options = {k: v for k, v in runtime.framework_options.items() if k not in RUN_ARGUMENTS}
        thread = thread_of(target, runtime)
        runtime.replies_counted = True  # by the callbacks: the middleware counts none again
        callbacks = _with_handler(options.get("callbacks"), GraphCallbacks(runtime, helpers))
        return {
            **options,
            "callbacks": callbacks,
            "configurable": {**options.get("configurable", {}), "thread_id": thread},
        }


def _with_handler(given: Any, handler: Any) -> Any:
    """The run's callbacks: ``handler`` beside the ones the run was given (a list of handlers,
    or a callback manager — copied, the given one is not changed)."""
    if given is None:
        return [handler]
    if isinstance(given, list):
        return [*given, handler]
    manager = given.copy()
    manager.add_handler(handler, inherit=True)
    return manager


def _run_arguments(run: Invocation) -> dict[str, Any]:
    """The graph's own run arguments among the run's framework options (:data:`RUN_ARGUMENTS`)."""
    options = run.runtime.framework_options
    return {key: options[key] for key in RUN_ARGUMENTS if key in options}


def run_store(target: Any, runtime: Any) -> Any | None:
    """The store a run of ``target`` is given (``bridges.store.MemoryStore``): with memory on
    for the run and its memory tools not turned off (``without=``), the memory service is the
    graph's long-term store — a store it was compiled with is not used beside it (said once);
    ``None`` otherwise (the graph's own store, or none, as it was built)."""
    if runtime.run_memory is None or not runtime.uses("memory_pull"):
        return None
    if getattr(target, "store", None) is not None:
        said(
            f"store:{runtime.agent_id}",
            "%s: the graph's own store is not used while memory is on: the memory service is "
            "its long-term store (without={'memory'} runs it on its own store)",
            runtime.agent_id,
            extra={"agent_id": runtime.agent_id},
        )
    import asyncio

    from trellis.harness.bridges.store import MemoryStore

    return MemoryStore(runtime.run_memory.ctx, asyncio.get_running_loop())


def _narrowed(target: Any) -> Any | None:
    """``target`` narrowing its model's tools per call (``middleware.narrowing``), when it is
    a ``create_agent`` graph (Deep Agents' too); else ``None``."""
    model = (getattr(target, "nodes", None) or {}).get("model")
    node: Any = getattr(getattr(model, "bound", None), "afunc", None)
    if getattr(node, "__qualname__", None) != MODEL_NODE:
        return None
    from trellis.harness.middleware import narrowing

    return narrowing(target, node)


def thread_of(target: Any, runtime: Any) -> str:
    """The LangGraph thread a run of ``target`` runs on: the run's own for a graph that
    checkpoints into its run, else the run's conversation (or the run)."""
    return runtime.run_id if run_scoped(target) else runtime.thread or runtime.run_id


async def holds(target: Any, thread: str, native_id: str, journal: Journal) -> bool:
    """Whether the graph's checkpointer still holds the pause ``native_id`` on ``thread`` —
    what a resume in place needs. An ``InMemorySaver`` holds it in the process that paused,
    and nowhere else; a ``RunCheckpointer`` holds it in the run's journal, everywhere. (A
    pause with a native id is a checkpointed graph's.)"""
    if run_scoped(target):
        return bool(journal.graph)
    state = await target.aget_state({"configurable": {"thread_id": thread}})
    return any(paused.id == native_id for paused in state.interrupts)


def run_scoped(target: Any) -> bool:
    """Whether the graph checkpoints into its run (``middleware.RunCheckpointer``)."""
    if not _checkpointed(target):
        return False
    from trellis.harness.middleware import RunCheckpointer

    return isinstance(target.checkpointer, RunCheckpointer)


async def _continued(
    graph: Any, target: Any, native_input: Any, run: Invocation, config: dict[str, Any]
) -> Any:
    """What the graph is run with. Run again without a decision to resume with (after a crash,
    or an error that may pass), a graph continues the turn it was cut in from its checkpoint
    (``None``) rather than starting a new turn: one whose checkpoint the run's journal holds,
    or one on a checkpointer of its own whose thread has steps still to run."""
    from langgraph.types import Command

    if isinstance(native_input, Command):
        return native_input
    if run_scoped(target):
        return None if run.runtime.replay.journal.graph else native_input
    if _checkpointed(target) and run.runtime.attempt > 1:
        state = await graph.aget_state(config)
        if state.next:  # the turn a crash cut, its next steps checkpointed
            return None
    return native_input


def is_hitl(value: Any) -> bool:
    """Whether an interrupt's value is a ``HumanInTheLoopMiddleware`` request (its
    ``action_requests`` and their ``review_configs``)."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("action_requests"), list)
        and bool(value["action_requests"])
        and isinstance(value.get("review_configs"), list)
    )


def hitl_response(request: dict[str, Any], resolution: InterruptResolution) -> dict[str, Any]:
    """The ``{"decisions": [...]}`` a ``HumanInTheLoopMiddleware`` pause resumes with — one
    decision per call it holds, in order — for a harness decision:

    * ``approve`` approves every call;
    * ``edit`` runs the first call (the one the interrupt shows) with the edited arguments and
      approves the others;
    * ``reject`` rejects every call, with the reviewer's reason (``answer=``) as the message the
      model reads;
    * ``answer`` responds for every call with the answer (the tool does not run).

    An answer (or edit) that already is ``{"decisions": [...]}`` is sent as it is. A decision a
    call's ``allowed_decisions`` does not include raises ``ConfigurationError``."""
    raw = resolution.payload if resolution.decision is InterruptDecision.EDIT else resolution.answer
    if isinstance(raw, dict) and isinstance(raw.get("decisions"), list):
        return raw
    actions: list[dict[str, Any]] = request["action_requests"]
    configs: list[dict[str, Any]] = request["review_configs"]
    decisions = [_decision(resolution, action, first=n == 0) for n, action in enumerate(actions)]
    for action, config, decision in zip(actions, configs, decisions, strict=False):
        allowed = config.get("allowed_decisions") or []
        if decision["type"] not in allowed:
            raise ConfigurationError(
                f"{action.get('name')} does not allow the decision {decision['type']!r} "
                f"(allowed: {', '.join(allowed)})"
            )
    return {"decisions": decisions}


def _decision(resolution: InterruptResolution, action: dict[str, Any], *, first: bool) -> Any:
    decision = resolution.decision
    if decision is InterruptDecision.REJECT:
        reason = reason_of(resolution)
        return {"type": "reject", "message": reason} if reason else {"type": "reject"}
    if decision is InterruptDecision.EDIT and first:
        return {
            "type": "edit",
            "edited_action": {"name": action["name"], "args": resolution.payload},
        }
    if decision is InterruptDecision.ANSWER:
        answer = resolution.answer
        message = answer if isinstance(answer, str) else json.dumps(answer, default=str)
        return {"type": "respond", "message": message}
    return {"type": "approve"}


def bound_tools(target: Any) -> list[Any]:
    """The tools a compiled graph's tool nodes run (Deep Agents included)."""
    found: list[Any] = []
    for node in getattr(target, "nodes", {}).values():
        found.extend(getattr(getattr(node, "bound", None), "tools_by_name", {}).values())
    return found


def _checkpointed(target: Any) -> bool:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    return isinstance(getattr(target, "checkpointer", None), BaseCheckpointSaver)


def _last_ai_text(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = (
                message.text if isinstance(getattr(message, "text", None), str) else message.content
            )
            if isinstance(text, str) and text:
                return text
    return None
