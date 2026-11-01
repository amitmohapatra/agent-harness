"""LangGraph compiled graphs — and Deep Agents, whose ``create_deep_agent`` returns one.

* input: ``{"messages": [...]}`` with the memory context as a leading system message (a
  string input becomes one user message; any other state dict passes through untouched).
  The context message has a fixed id, so a checkpointed thread holds at most one — each turn
  replaces it in place (``add_messages``) — and it leaves out the recent conversation, which
  the checkpointer already holds. Where memory keeps the conversation instead, its recent
  messages follow the context as LangChain's own — ``HumanMessage``, ``AIMessage`` with its
  ``tool_calls``, ``ToolMessage`` — the messages a checkpointer would replay. A
  ``create_agent`` graph (``ReAct``'s, Deep Agents') sends its model the context as the first
  part of the turn's question instead, after the history a prompt cache keeps
  (``middleware.with_question``);
* transcript: the messages the run added after the question — tool calls and their results
  with the answer — so the conversation memory keeps replays as the framework's would;
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
* tools: fixed when the graph is compiled — unless the graph takes them per model call
  (``middleware.HarnessTools``, as ``ReAct(...)`` and ``middleware()`` do): then the run's
  tools (``ReAct(tools=)``, the gateway's MCP tools, the memory tools) are offered on every
  model call as the run offers them at that moment (``narrows="turn"``). A hand-built graph
  takes the memory tools from ``h.memory_tools()`` at build time. A ``create_agent`` graph
  (Deep Agents' too) without ``HarnessTools`` narrows per model call as well: the harness
  runs a copy of it whose model calls go through one more ``awrap_model_call`` handler
  (``middleware.narrowing``), so the tool hints and a part turned off reach the model; a
  hand-written graph binds its model's tools in its own code, which
  the harness cannot reach (``narrows="none"``: every tool it bound is offered);
* a graph that checkpoints into its run (``middleware.RunCheckpointer``, as ``ReAct(...)``
  does) runs on the run's own thread (``thread_id`` is the run id): its checkpoint is in the
  run's journal, so a resume — in any process, after a pause or a crash — continues where it
  stopped, and it holds no conversation of its own (the memory service gives the recent turns).
  A graph built with ``middleware()`` and compiled without a checkpointer is run as a copy
  that has one (:func:`checkpointing`); one compiled with a checkpointer of its own keeps it,
  its checkpoints marked with the run (:data:`RUN_METADATA`), and the attempt after a crash
  continues from the run's latest one, as a resume does, rather than starting the turn again;
* steps: a checkpointed graph's harness calls are journaled with the step they are made in
  (the checkpoint the step started from: ``Replay.where``), and an attempt that continues from
  a checkpoint passes over the records of the steps before it (``Replay.resumed``): a later
  step's identical call runs, and is asked about, again;
* store: a graph compiled without a store of its own is run, while memory is on, with the
  memory service as its long-term store (``bridges.store.MemoryStore``: ``runtime.store``,
  ``get_store()``, Deep Agents' ``StoreBackend``); a graph's own store is kept (said once);
* helpers: a Deep Agent's ``task`` sub-agent is a full child run of the run
  (``bridges.helpers``, from the governed ``task`` call: ``middleware.governed_call``): given
  the memory context for its task, its tool calls governed as the run's and recorded under its
  run, its outcome recorded; its model calls count toward the run's budget (``accounting``),
  as do those of the graph's subgraphs and ``Send`` branches, which run inside the run;
* framework options: the ``RunnableConfig`` keys the run is given (``recursion_limit``,
  ``configurable``, ``tags``...) go into the config of ``ainvoke``/``astream``; a
  ``configurable.thread_id`` among them is refused (the thread is ``run(thread=)``, one owner).
  The graph's own run arguments
  (:data:`RUN_ARGUMENTS`: ``context`` for a ``context_schema``, ``durability``) go to the call
  beside it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, ClassVar, Final

from trellis.contracts import ConfigurationError, InterruptDecision, InterruptResolution, ToolSpec
from trellis.harness.adapters.base import (
    Extracted,
    HistoryIn,
    Invocation,
    Narrowing,
    NativePause,
    Output,
    ToolFormat,
    builtin,
    helper,
    unique,
)
from trellis.harness.asking import answer_of
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.bridges.helpers import Helpers
from trellis.harness.history import Called, Said, Speaker
from trellis.harness.journal import Journal, Pending
from trellis.harness.runtime import DEFERRED, MARKER, Runtime, reason_of

log = logging.getLogger("trellis.harness")

#: The journal key a graph's own ``interrupt(...)`` (not ``ask``) is filed under.
FOREIGN: Final = "langgraph"
#: The journal key of a ``HumanInTheLoopMiddleware`` pause (Deep Agents' ``interrupt_on``).
HITL: Final = "langchain_hitl"
#: The state key a graph declares when it takes the run's tools per model call
#: (``middleware.HarnessTools``; kept here so detecting it imports no LangChain).
HARNESS_TOOLS: Final = "trellis_harness_tools"
#: The id of the memory context message: one per thread, replaced every turn.
CONTEXT_MESSAGE_ID: Final = "trellis-memory-context"
#: The qualified name of a ``create_agent`` graph's async model node (Deep Agents' too).
MODEL_NODE: Final = "create_agent.<locals>.amodel_node"
#: The graph's own run arguments a run's framework options may carry, beside its config.
RUN_ARGUMENTS: Final = ("context", "durability")
#: Deep Agents' tool that runs a sub-agent.
TASK_TOOL: Final = "task"
#: The checkpoint metadata key a graph with a checkpointer of its own has its run's checkpoints
#: marked with (the run's id, from the run's config ``metadata``).
RUN_METADATA: Final = "trellis_run_id"
#: The ``configurable`` key under which LangGraph hands a task, by namespace, the checkpoint
#: its step started from (``CONFIG_KEY_CHECKPOINT_MAP``; the root graph's under ``""``).
CHECKPOINT_MAP: Final = "checkpoint_map"


class LangGraphAdapter:
    name: ClassVar[str] = "langgraph"

    replays_history = True
    history_in: HistoryIn = "messages"
    sessions = False

    def __init__(self, target: Any = None) -> None:
        # every model call of the graph accounted to the run's budget, ModelHooks or not
        from trellis.harness import accounting  # noqa: F401

        # and every tool node's calls governed, the harness's middleware or not
        from trellis.harness.middleware import declare_files, govern_tool_nodes

        govern_tool_nodes()
        # a Deep Agent's file tools on a disk, a store or a sandbox change real files: governed
        declare_files(target)

        #: the graph takes the run's tools per model call (``middleware.HarnessTools``): it
        #: converts them itself, so nothing is converted for it here
        per_call = HARNESS_TOOLS in (getattr(target, "channels", None) or {})
        self.tool_format: ToolFormat = "none" if per_call else "langchain"
        self.fixed_tools = not per_call
        #: the graph run in its place: a ``create_agent`` graph whose model calls narrow its
        #: bound tools to those the run offers (``None``: ``target`` itself is run)
        self.narrowed = None if per_call else _narrowed(target)
        if self.narrowed is not None:
            _without_middleware(str(getattr(target, "name", None) or "LangGraph"))
        per_turn = per_call or self.narrowed is not None
        self.narrows: Narrowing = "turn" if per_turn else "none"
        #: the graph's own tools (:meth:`native`), once read
        self._native: list[ToolSpec] | None = None

    def steers(self, target: Any) -> bool:
        """A graph with the harness's middleware (``ReAct``, ``middleware()``): its
        ``HarnessTools`` takes a steered message before each model call
        (``AgentMiddleware.before_model``); any other graph has no hook the harness owns."""
        return HARNESS_TOOLS in (getattr(target, "channels", None) or {})

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        """The graph's checkpointer, unless it checkpoints into its run."""
        if not _checkpointed(target) or run_scoped(target):
            return None
        return target.checkpointer

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        from langchain_core.messages import SystemMessage

        system = [SystemMessage(content=context, id=CONTEXT_MESSAGE_ID)] if context else []
        system += [langchain_message(said) for said in history]
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
        try:
            given = await _continued(target, native_input, run)
            return await self._graph(target, run.runtime).ainvoke(
                given,
                self._config(target, run),
                version="v2",
                **_run_arguments(run.runtime),
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
        try:
            given = await _continued(target, native_input, run)
            async for part in self._graph(target, run.runtime).astream(
                given,
                self._config(target, run),
                stream_mode=["messages", "values"],
                version="v2",
                **_run_arguments(run.runtime),
            ):
                if part["type"] == "messages":
                    chunk = part["data"][0]
                    if (
                        isinstance(chunk, AIMessage)
                        and isinstance(chunk.content, str)
                        and chunk.content
                    ):
                        yield chunk.content
                elif part["type"] == "values":
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
        messages: list[Any] = []
        if isinstance(state, dict):
            messages = state.get("messages") or []
            answer = state.get("structured_response")
            if answer is None:
                answer = _last_ai_text(messages)
        transcript = added(messages) or (
            [Said(Speaker.ASSISTANT, answer)] if isinstance(answer, str) and answer else []
        )
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
                f"framework_options for a graph are RunnableConfig keys and its run arguments: "
                f"{', '.join(map(repr, unknown))} is none of {', '.join(known)}"
            )
        configurable = options.get("configurable", {})
        if not isinstance(configurable, Mapping):
            raise ConfigurationError("framework_options' configurable is a dict")
        if "thread_id" in configurable:
            raise ConfigurationError(
                "framework_options' configurable.thread_id: the harness runs the graph on the "
                "run's thread (or the run's own, for a graph that checkpoints into its run): "
                "pass agent.run(thread=...) instead"
            )

    async def native(self, target: Any) -> list[ToolSpec]:
        """What the graph's tool nodes run (:func:`graph_tools`: a Deep Agent's ``task``
        sub-agents' tools too), as specs — the harness's own (converted: the run's toolbox
        has them) are not among them, and a tool a node binds to its model in its own code
        cannot be read; the graph's state tools (its plan, its virtual files) are not tools of
        the run's set, unless their backend is outside the state (``OUTSIDE_STATE``), nor is
        the graph's own machinery (``MACHINERY``: ``ReAct``'s idle ``execute``, its
        general-purpose helper). Deep Agents' own (``task``, its file tools outside the
        state) are tagged ``BUILTIN``: the framework's, not counted for the narrowing. A graph
        that takes the run's tools per model call (``ReAct``, ``middleware()``) has its own
        LangChain tools and sub-agents here, beside the run's toolbox. Read once per graph."""
        if self._native is None:
            from trellis.harness.middleware import MACHINERY, OUTSIDE_STATE, STATE_TOOLS, spec_of
            from trellis.harness.tools.convert.langchain import made

            found = (
                _own_spec(spec_of(t), outside=bool((t.metadata or {}).get(OUTSIDE_STATE)))
                for t in graph_tools(target)
                if (t.name not in STATE_TOOLS or (t.metadata or {}).get(OUTSIDE_STATE))
                and not (t.metadata or {}).get(MACHINERY)
                and not made(t)
            )
            self._native = unique(found)
        return self._native

    def _graph(self, target: Any, runtime: Runtime) -> Any:
        """The graph run for ``target``: its narrowing copy, when it has one — given the
        memory service as its store (:func:`memory_store`) when it has none of its own."""
        graph = target if self.narrowed is None else self.narrowed
        store = memory_store(target, runtime)
        return graph if store is None else graph.copy({"store": store})

    @staticmethod
    def _config(target: Any, run: Invocation) -> dict[str, Any]:
        """The run's config: its framework options (less its run arguments) and the harness's
        thread id (theirs names none: ``check_options``)."""
        runtime = run.runtime
        if _checkpointed(target):
            from langgraph.types import interrupt

            runtime.suspend = interrupt
        options = {k: v for k, v in runtime.framework_options.items() if k not in RUN_ARGUMENTS}
        thread = thread_of(target, runtime)
        config = {
            **options,
            "configurable": {**options.get("configurable", {}), "thread_id": thread},
        }
        if _checkpointed(target) and not run_scoped(target):  # its checkpoints: the run's
            config["metadata"] = {**options.get("metadata", {}), RUN_METADATA: runtime.run_id}
        return config


def _own_spec(spec: ToolSpec, *, outside: bool) -> ToolSpec:
    """A graph's own tool as the run's tool set has it: Deep Agents' ``task`` a helper, and it
    and Deep Agents' file tools outside the state (``outside``) the framework's built-ins."""
    if spec.name == TASK_TOOL:
        return builtin(helper(spec))
    return builtin(spec) if outside else spec


def _run_arguments(runtime: Runtime) -> dict[str, Any]:
    """The graph's own run arguments among the run's framework options (:data:`RUN_ARGUMENTS`)."""
    options = runtime.framework_options
    return {key: options[key] for key in RUN_ARGUMENTS if key in options}


def memory_store(target: Any, runtime: Runtime) -> Any | None:
    """The store a run of ``target`` is given (``bridges.store.MemoryStore``): the memory
    service while memory is on for the run, its memory tools too (``memory_pull``: the store is
    the graph reading memory), and the graph was compiled with no store of its own (one it has
    is kept, said once); else ``None`` — the graph runs as it was built."""
    if runtime.run_memory is None or not runtime.uses("memory_pull"):
        return None
    if getattr(target, "store", None) is not None:
        _own_store(runtime.agent_id, type(target.store).__name__)
        return None
    from trellis.harness.bridges.store import MemoryStore

    return MemoryStore(runtime.run_memory.ctx, asyncio.get_running_loop())


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per graph name, of the most recent
def _without_middleware(name: str) -> None:
    """Said when a ``create_agent`` or Deep Agents graph is wrapped without the harness's
    middleware (``trellis.middleware()``): it still runs governed and narrowed, but without
    what the middleware gives."""
    log.warning(
        "the graph %s is wrapped without the harness's middleware: its tools are governed and "
        "narrowed, but it has no memory or gateway MCP tools, no step limit or stall guard, no "
        "model spans, call_timeout or budget stop, and no checkpoint in the run — build it with "
        "middleware=trellis.middleware() (its checkpoint is then kept in the run)",
        name,
    )


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and store type, of the most recent
def _own_store(agent_id: str, store: str) -> None:
    log.warning(
        "the graph was compiled with a store of its own: it keeps it, and its long-term memory "
        "is not the memory service's (compile it without one to use the memory service)",
        extra={"agent_id": agent_id, "store": store},
    )


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
    checkpoints into its run, else the run's conversation — its tenant's, its user's
    (``Identity.conversation``: two users naming their threads alike never share one), or
    the run's own."""
    return runtime.run_id if run_scoped(target) else runtime.conversation


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


def checkpointing(target: Any) -> Any:
    """``target`` as the harness runs it: a graph built with ``middleware()`` (it takes the
    run's tools per model call) and compiled without a checkpointer, as a copy checkpointing
    into its run (``middleware.RunCheckpointer``: a resume continues where it stopped) — the
    graph itself left as it is; any other as it is (a checkpointer of its own, or
    ``checkpointer=False``, stays the graph's)."""
    if HARNESS_TOOLS not in (getattr(target, "channels", None) or {}):
        return target
    if getattr(target, "checkpointer", False) is not None:
        return target
    from trellis.harness.middleware import RunCheckpointer

    return target.copy({"checkpointer": RunCheckpointer()})


async def _continued(target: Any, native_input: Any, run: Invocation) -> Any:
    """What the graph is run with — and, a checkpointed graph, where the attempt's calls are
    made (``Replay.where``: the step's checkpoint) and the checkpoint it continues from
    (``Replay.resumed``). A graph run again without a decision to resume with (after a crash,
    or an error that may pass) continues from the run's latest checkpoint (``None``) rather
    than starting a new turn: one its journal holds (``RunCheckpointer``), or one its own
    checkpointer holds for this run (:data:`RUN_METADATA`); a graph whose checkpointer holds
    none of the run's starts from its input."""
    from langgraph.types import Command

    if not _checkpointed(target):
        return native_input
    runtime = run.runtime
    replay = runtime.replay
    thread = thread_of(target, runtime)
    replay.where = _stepping(thread)
    resumed = isinstance(native_input, Command)
    if run_scoped(target):
        if not replay.journal.graph:
            return native_input
        replay.resumed = _latest(replay.journal.graph)
        return native_input if resumed else None
    if runtime.attempt == 1:
        return native_input
    state = await target.aget_state({"configurable": {"thread_id": thread}})
    marked: Mapping[str, object] = state.metadata or {}
    if marked.get(RUN_METADATA) != runtime.run_id:
        return native_input
    configurable: Mapping[str, object] = state.config.get("configurable") or {}
    latest = configurable.get("checkpoint_id")
    replay.resumed = latest if isinstance(latest, str) else None
    return native_input if resumed else None


def _latest(graph: Mapping[str, Any]) -> str:
    """The id of the root graph's checkpoint the run's journal holds; one that holds a
    sub-graph's and not the root's is damaged (``CheckpointUnreadable``)."""
    from trellis.harness.middleware import CheckpointUnreadable

    latest = (graph.get("") or {}).get("id")
    if not isinstance(latest, str):
        raise CheckpointUnreadable(
            "the run's journal holds a graph checkpoint without the graph's own: the run stops "
            "rather than start the graph over, which would make its calls again; start a new run"
        )
    return latest


def _stepping(thread: str) -> Callable[[], str | None]:
    """Where a call is made now: the checkpoint the graph's step started from, read from the
    config LangGraph gives the step's task on ``thread`` (:data:`CHECKPOINT_MAP`); ``None``
    outside one of its steps."""
    from langgraph.config import get_config

    def where() -> str | None:
        try:
            configurable = get_config().get("configurable") or {}
        except RuntimeError:  # not inside a step of a graph
            return None
        if configurable.get("thread_id") != thread:
            return None
        step = (configurable.get(CHECKPOINT_MAP) or {}).get("")
        return step if isinstance(step, str) else None

    return where


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


#: How deep :func:`graph_tools` looks for a ``task`` tool's sub-agents: a sub-agent's own
#: sub-agents, and theirs.
SUBAGENT_DEPTH: Final = 3


def graph_tools(target: Any, depth: int = SUBAGENT_DEPTH) -> list[Any]:
    """The tools a compiled graph's tool nodes run, and — a Deep Agent's ``task`` tool — the
    tools of the sub-agent graphs it runs (:func:`subagent_graphs`), down to ``depth``."""
    found = [t for t in bound_tools(target) if isinstance(getattr(t, "name", None), str)]
    if depth <= 0:
        return found
    for tool in list(found):
        if tool.name != TASK_TOOL:
            continue
        for graph in subagent_graphs(tool):
            found.extend(graph_tools(graph, depth - 1))
    return found


def subagent_graphs(task: Any) -> list[Any]:
    """The sub-agent graphs a Deep Agents ``task`` tool runs, as its code holds them (the
    graphs its sub-agents were compiled to, by name); ``[]`` when they cannot be read (a
    release laid out otherwise: its sub-agents' tools are then left out of the run's set)."""
    from langgraph.pregel import Pregel

    seen: set[int] = set()
    graphs: dict[int, Any] = {}
    pending = [getattr(task, "coroutine", None), getattr(task, "func", None)]
    while pending:
        fn = pending.pop()
        if fn is None or id(fn) in seen or not hasattr(fn, "__closure__"):
            continue
        seen.add(id(fn))
        for cell in fn.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            if isinstance(value, dict):
                graphs.update((id(v), v) for v in value.values() if isinstance(v, Pregel))
            elif callable(value) and hasattr(value, "__closure__"):
                pending.append(value)
    return list(graphs.values())


def _checkpointed(target: Any) -> bool:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    return isinstance(getattr(target, "checkpointer", None), BaseCheckpointSaver)


def langchain_message(said: Said) -> Any:
    """``said`` as LangChain's own message: an assistant's with its tool calls, a tool's with
    the call it answers."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if said.role is Speaker.USER:
        return HumanMessage(content=said.content)
    if said.role is Speaker.ASSISTANT:
        calls = [
            {"id": c.id, "name": c.name, "args": dict(c.arguments), "type": "tool_call"}
            for c in said.tool_calls
        ]
        return AIMessage(content=said.content, tool_calls=calls)
    if said.role is Speaker.TOOL:
        return ToolMessage(content=said.content, tool_call_id=said.tool_call_id or "")
    return SystemMessage(content=said.content)


def added(messages: Sequence[Any]) -> list[Said]:
    """What the run said after its question (the last user message not steered into it): each
    answer with the tool calls it made, and their results, and each message steered into it
    (``middleware.STEERED``, with its id); ``[]`` without a question."""
    asked = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if getattr(messages[i], "type", None) == "human" and not _steered(messages[i])
        ),
        None,
    )
    if asked is None:
        return []
    out: list[Said] = []
    for m in messages[asked + 1 :]:
        kind = getattr(m, "type", None)
        if kind == "human":  # steered into the run
            out.append(Said(Speaker.USER, _text(m), id=m.id))
        elif kind == "ai":
            calls = tuple(
                Called(c["id"] or "", c["name"], c.get("args") or {})
                for c in getattr(m, "tool_calls", None) or ()
            )
            if calls or _text(m):
                out.append(Said(Speaker.ASSISTANT, _text(m), tool_calls=calls))
        elif kind == "tool":
            out.append(Said(Speaker.TOOL, _text(m), tool_call_id=m.tool_call_id))
    return out


def _steered(message: Any) -> bool:
    from trellis.harness.middleware import STEERED

    return STEERED in (getattr(message, "response_metadata", None) or {})


def _text(message: Any) -> str:
    """A message's text (its text parts, when its content is a list of them)."""
    text = message.text if isinstance(getattr(message, "text", None), str) else message.content
    return text if isinstance(text, str) else ""


def _last_ai_text(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and (text := _text(message)):
            return text
    return None
