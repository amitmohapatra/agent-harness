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
* run: ``astream`` with ``version="v2"`` (``agent.run``), or ``astream_events`` with
  ``version="v3"`` (``agent.stream``: the graph's stream transformers shape what is streamed,
  and the answer's text is its text content blocks), on the LangGraph thread named by the
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
  stopped, and it holds no conversation of its own (the memory service gives the recent turns)
  but the state its conversation carries from turn to turn (:data:`CARRIED`).
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
  as do those of the graph's subgraphs and ``Send`` branches, which run inside the run. A
  Deep Agents async sub-agent's task (``start_async_task``) is a child run too, outliving the
  turn as the task does: held while it runs on its server, ended when a later check, list or
  cancel says how it ended, and cancelled with the turn that started it
  (``bridges.async_tasks``);
* framework options: the ``RunnableConfig`` keys the run is given (``recursion_limit``,
  ``configurable``, ``tags``...) go into the config of ``ainvoke``/``astream``; a
  ``configurable.thread_id`` among them is refused (the thread is ``run(thread=)``, one owner).
  The graph's own run arguments
  (:data:`RUN_ARGUMENTS`: ``context`` for a ``context_schema``, ``durability``) go to the call
  beside it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import json
import logging
import warnings
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel

from trellis.contracts import ConfigurationError, InterruptDecision, InterruptResolution, ToolSpec
from trellis.harness.adapters.base import (
    DEFAULT_CODE,
    PROVIDER_TOOLS,
    Code,
    Extracted,
    HistoryIn,
    Invocation,
    Keeps,
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
from trellis.harness.bridges.async_tasks import AsyncTasks
from trellis.harness.bridges.helpers import helping
from trellis.harness.events import STREAMED
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
#: The ``response_metadata`` key marking a run's input message as its question although the
#: user did not type it (a notification run's: ``background.notify``): what the run says comes
#: after it (:func:`added`).
ASKED: Final = "trellis_asked"
#: The checkpoint metadata key a graph with a checkpointer of its own has its run's checkpoints
#: marked with (the run's id, from the run's config ``metadata``).
RUN_METADATA: Final = "trellis_run_id"
#: The ``configurable`` key under which LangGraph hands a task, by namespace, the checkpoint
#: its step started from (``CONFIG_KEY_CHECKPOINT_MAP``; the root graph's under ``""``).
CHECKPOINT_MAP: Final = "checkpoint_map"
#: The ``configurable`` key under which LangGraph hands a task its id (``CONFIG_KEY_TASK_ID``):
#: the same for each run of the task its ``RetryPolicy`` makes.
TASK_ID: Final = "__pregel_task_id"
#: The result a new turn gives a call its thread's last message holds with none: the run that
#: made it was stopped (a newer message interrupted it, its time ran out) or left waiting for
#: a person's approval, and a provider refuses a call with no result — the thread could not go
#: on.
STOPPED_CALL: Final = (
    "{tool} has no result: its run stopped, or waited for a person's approval, before the call "
    "came back."
)
#: The node of Deep Agents' ``PatchToolCallsMiddleware``: a graph that answers such calls itself.
PATCHES_CALLS: Final = "PatchToolCallsMiddleware.before_agent"
#: The graph state channels a conversation carries from one turn to the next where the graph
#: has no thread of its own to hold them (:func:`carries`: its thread is its run, or it has
#: none), as a thread of the graph's own would: Deep Agents' list of the background tasks its
#: async sub-agents run (``AsyncSubAgentState``), which its task tools read a task id against;
#: its files kept in the graph's state (``StateBackend``: "persists across turns"), and its
#: todos. Never the conversation itself (``messages``: the memory window's, or the graph's).
#: A turn starts with what the conversation's last turn ended with (``Runtime.carried``; kept
#: in the process and, up to ``pipeline.CARRIED_BYTES``, in the file store: ``pipeline``). So
#: does the agent a langgraph-swarm graph last handed the conversation to (:data:`ACTIVE_AGENT`):
#: its next turn is that agent's.
ACTIVE_AGENT: Final = "active_agent"  # langgraph-swarm's ``SwarmState.active_agent``
CARRIED: Final = ("async_tasks", "files", "todos", ACTIVE_AGENT)


class LangGraphAdapter(Keeps):
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
        #: its model calls are the harness's to change (``HarnessTools``, the narrowing copy):
        #: an earlier turn's result is marked in the request (``middleware._stale_noted``)
        self.marks_earlier = per_turn
        #: the graph's own tools (:meth:`native`), once read
        self._native: list[ToolSpec] | None = None
        #: the async tasks (Deep Agents' ``start_async_task``) its runs started that this
        #: process holds the runs of (``bridges.async_tasks``)
        self._tasks = AsyncTasks()

    @property
    def tasks(self) -> AsyncTasks:
        return self._tasks

    async def child_cancelled(self, run_id: str, reason: str | None) -> None:
        """The run ``run_id`` ended with the run that started it: an async task's that this
        process holds is cancelled on its server (``children.stop_kept``)."""
        await self.tasks.stopped(run_id, reason)

    def kept_children(self, run_id: str) -> list[str]:
        return self.tasks.kept_for(run_id)

    def own_tools(self, target: Any, sources: Sequence[Any]) -> bool:
        """A graph's own tools; a ``ReAct``'s are only the harness's own read tools."""
        from trellis.harness.middleware import STATE_TOOLS

        return any(getattr(t, "name", None) not in STATE_TOOLS for t in bound_tools(target))

    def code(self, target: Any, options: Mapping[str, Any]) -> Code:
        """A graph's: Deep Agents' ``execute`` on the sandbox the harness gives the agent
        (``default``: ``ReAct``'s ``CodeBackend``), or on a backend of its own that runs
        commands, or LangChain's shell (``own``); else ``none``."""
        found = {_code_of(t) for t in graph_tools(target)}
        return "default" if "default" in found else ("own" if "own" in found else "none")

    def steers(self, target: Any) -> bool:
        """A graph with the harness's middleware (``ReAct``, ``middleware()``): its
        ``HarnessTools`` takes a steered message before each model call
        (``AgentMiddleware.before_model``); any other graph has no hook the harness owns."""
        return HARNESS_TOOLS in (getattr(target, "channels", None) or {})

    def asks(self, target: Any) -> bool:
        return False  # LangGraph's ``interrupt`` is a node's code: its model has no question tool

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        """The graph's checkpointer, unless it checkpoints into its run. A durable one is
        said once (:data:`DUPLICATE_STORAGE`)."""
        if not _checkpointed(target) or run_scoped(target):
            return None
        if (name := durable_checkpointer(target)) and name not in _warned:
            _warned.add(name)
            log.warning(DUPLICATE_STORAGE.format(name=name))
        return target.checkpointer

    async def held_runs(self, target: Any, runtime: Any) -> frozenset[str]:
        """The runs that wrote the checkpoints the graph's own checkpointer holds of the run's
        thread (:data:`RUN_METADATA`)."""
        if not _checkpointed(target) or run_scoped(target):
            return frozenset()
        config = {"configurable": {"thread_id": thread_of(target, runtime)}}
        marks = [c.metadata.get(RUN_METADATA) async for c in target.checkpointer.alist(config)]
        return frozenset(m for m in marks if isinstance(m, str))

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        from langchain_core.messages import SystemMessage

        # the context is sent with the user's message: checked as that message is
        context = screened(target, context) if context else context
        system = [SystemMessage(content=context, id=CONTEXT_MESSAGE_ID)] if context else []
        system += [langchain_message(said) for said in history]
        if isinstance(input, str) and not takes_text(target):
            return {"messages": [*system, {"role": "user", "content": input}]}
        if isinstance(input, list):
            return {"messages": [*system, *input]}
        if not isinstance(input, dict):
            return input
        # what the conversation carries is the harness's to give (:func:`_continued`), never
        # an input's: whoever starts a run names no task list for it
        given = {name: value for name, value in input.items() if name not in CARRIED}
        if isinstance(given.get("messages"), list):
            given["messages"] = [*system, *given["messages"]]
        return given

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        from langgraph.types import GraphOutput

        values: Any = None
        interrupts: list[Any] = []
        async with helping(run.runtime, self.name):
            given = await _continued(target, native_input, run)
            graph, config = self._graph(target, run.runtime), self._config(target, run)
            try:
                # ``ainvoke``, the ``custom`` stream mode beside its ``values``: what the
                # graph's nodes write to the stream is the run's (:func:`_written`)
                async for part in graph.astream(
                    given,
                    config,
                    stream_mode=["values", "custom"],
                    version="v2",
                    **_run_arguments(run.runtime, target),
                ):
                    if part["type"] == "custom":
                        _written(run.runtime, part["data"])
                    else:
                        values = part["data"]
                        interrupts.extend(part["interrupts"])
            except Exception as exc:
                values, interrupts[:] = await _asked_again(graph, config, exc)
            _ended_with(target, run.runtime, values)
            return GraphOutput(value=values, interrupts=tuple(interrupts))

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        """The run streamed as LangGraph streams a graph to its consumers. A graph given
        stream transformers of its own (:func:`transforms`: ``create_agent``'s
        ``transformers=``, a middleware's — ``PIIMiddleware`` redacts the text on the wire) is
        streamed through them (:func:`_transformed`), and what reaches the run is what they
        let through; any other by its stream modes (:func:`_streamed`). The answer's text is
        each model reply's text as it arrives — a text content block's too, a model's
        reasoning not (:class:`_Text`); the root graph's ``values`` are its state and its
        pauses, and what its nodes write (``custom``) is the run's."""
        from langgraph.types import GraphOutput

        values: Any = None
        interrupts: tuple[Any, ...] = ()
        text = _Text()
        async with helping(run.runtime, self.name):
            given = await _continued(target, native_input, run)
            graph, config = self._graph(target, run.runtime), self._config(target, run)
            source = _transformed if transforms(graph) else _streamed
            try:
                async for kind, data, held in source(
                    graph, given, config, _run_arguments(run.runtime, target)
                ):
                    if kind == "messages":
                        if said := text.of(*data):
                            yield said
                    elif kind == "custom":
                        _written(run.runtime, data)
                    else:  # "values"
                        values, interrupts = data, tuple(held) or interrupts
            except Exception as exc:
                values, again = await _asked_again(graph, config, exc)
                interrupts = tuple(again)
        _ended_with(target, run.runtime, values)
        yield Output(GraphOutput(value=values, interrupts=interrupts))

    def extract(self, target: Any, output: Any) -> Extracted:
        state = fields_of(output.value)
        answer: Any = output.value
        messages: list[Any] = []
        structured = state.get("structured_response") if isinstance(state, Mapping) else None
        if structured is not None:
            answer = structured
        elif isinstance(state, Mapping) and isinstance(state.get("messages"), list):
            answer = _answer_text(state["messages"])
        if isinstance(state, Mapping) and isinstance(state.get("messages"), list):
            messages = state["messages"]
        transcript = added(messages) or (
            [Said(Speaker.ASSISTANT, answer)] if isinstance(answer, str) and answer else []
        )
        if structured is not None:
            transcript = _structured(transcript, structured)
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
            schema = first.response_schema  # surfaced as its JSON Schema
            pause = NativePause(
                value=first.value,
                native_id=first.id,
                ids=ids,
                schema=schema if isinstance(schema, dict) else None,
                plan=state.get(PLAN) if isinstance(state, Mapping) else None,
            )
        return Extracted(answer, transcript, pause, asked=asked(messages))  # type: ignore[arg-type]

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

    def workdir(self, target: Any, options: Mapping[str, Any]) -> Path | None:
        return None  # no folder of its own: a sandbox's outputs/ are its outputs

    async def native(self, target: Any) -> list[ToolSpec]:
        """What the graph's tool nodes run (:func:`graph_tools`: a Deep Agent's ``task``
        sub-agents' tools too), as specs — the harness's own (converted: the run's toolbox
        has them) are not among them, and a tool a node binds to its model in its own code
        cannot be read; the graph's state tools (its plan, its virtual files) are not tools of
        the run's set, unless their backend is outside the state (``OUTSIDE_STATE``), nor is
        the graph's own machinery (``MACHINERY``: ``ReAct``'s idle ``execute``, its
        general-purpose helper), nor Deep Agents' ``execute`` on a backend that runs no command
        (:func:`idle_execute`). Deep Agents' own (``task``, its file tools outside the
        state) are tagged ``BUILTIN``: the framework's, not counted for the narrowing. A graph
        that takes the run's tools per model call (``ReAct``, ``middleware()``) has its own
        LangChain tools and sub-agents here, beside the run's toolbox. Read once per graph."""
        if self._native is None:
            from trellis.harness.middleware import (
                MACHINERY,
                OUTSIDE_STATE,
                STATE_TOOLS,
                framework_made,
                spec_of,
            )
            from trellis.harness.tools.convert.langchain import made

            found = (
                _own_spec(spec_of(t), outside=bool((t.metadata or {}).get(OUTSIDE_STATE)))
                for t in graph_tools(target)
                if (
                    t.name not in STATE_TOOLS
                    or (t.metadata or {}).get(OUTSIDE_STATE)
                    or not framework_made(t)
                )
                and not (t.metadata or {}).get(MACHINERY)
                and not made(t)
                and not idle_execute(t)
            )
            provided = getattr(target, PROVIDER_TOOLS, None) or []  # ``ReAct``'s provider tools
            self._native = unique([*found, *provided])
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
        runtime.deciding = _task
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


class _Text:
    """The answer's text in a graph's ``messages`` events (``astream_events(version="v3")``),
    after its stream transformers: a model's text content block, delta by delta and then the
    rest of the block its ``content-block-finish`` holds beyond what its deltas gave (a
    transformer may keep the tail of a block back until it finishes: ``PIIMiddleware``'s
    keeps the last characters, a redaction could still reach into them); a message a node
    returns whole (a model that does not stream, a hook's reply), its text. Reasoning,
    tool-call and other blocks are not the answer's text."""

    def __init__(self) -> None:
        #: what each text block (by the model call's run id and the block's index) has given
        self._given: dict[tuple[str, Any], str] = {}

    def of(self, payload: Any, metadata: Mapping[str, Any] | None) -> str:
        from langchain_core.messages import AIMessage

        if isinstance(payload, AIMessage):
            return _text(payload)
        if not isinstance(payload, dict):
            return ""
        key = (str((metadata or {}).get("run_id") or ""), payload.get("index"))
        kind = payload.get("event")
        if kind == "content-block-delta":
            delta = payload.get("delta") or {}
            said = delta.get("text") if delta.get("type") == "text-delta" else None
            if not isinstance(said, str):
                return ""
            self._given[key] = self._given.get(key, "") + said
            return said
        if kind == "content-block-finish":
            given = self._given.pop(key, "")
            block = payload.get("content") or {}
            whole = block.get("text") if block.get("type") == "text" else None
            if isinstance(whole, str) and whole.startswith(given):
                return whole[len(given) :]
        return ""


#: One event of a graph's stream as the run reads it, whichever way the graph is streamed:
#: its kind (``messages``, ``custom``, ``values``), its data, and the pauses a ``values`` holds.
Streamed = tuple[str, Any, Sequence[Any]]


def transforms(graph: Any) -> bool:
    """Whether ``graph`` was given stream transformers of its own — a middleware's, its
    builder's (``transformers=``) — beside the projections ``create_agent`` gives every graph
    (tool calls, sub-agents: they add to the stream and change nothing a model said). Those
    decide what the wire may carry, so the graph is streamed through them."""
    from langchain.agents._subagent_transformer import SubagentTransformer
    from langgraph.prebuilt import ToolCallTransformer

    given = getattr(graph, "stream_transformers", ())
    return any(t is not ToolCallTransformer and t is not SubagentTransformer for t in given)


async def _streamed(
    graph: Any, given: Any, config: Any, arguments: Mapping[str, Any]
) -> AsyncIterator[Streamed]:
    """A graph's stream by its stream modes (``astream``): each model message as it arrives,
    the root graph's state and pauses, what its nodes write."""
    async for part in graph.astream(
        given, config, stream_mode=["messages", "values", "custom"], version="v2", **arguments
    ):
        yield part["type"], part["data"], part.get("interrupts") or ()


async def _transformed(
    graph: Any, given: Any, config: Any, arguments: Mapping[str, Any]
) -> AsyncIterator[Streamed]:
    """A graph's stream through its stream transformers (``astream_events(version="v3")``):
    every event passes them first, and the root graph's are the run's."""
    from langchain_core._api import LangChainBetaWarning

    with warnings.catch_warnings():  # the harness's choice: said to no user of it
        warnings.simplefilter("ignore", LangChainBetaWarning)
        # returns without awaiting anything: no other task runs under this filter
        events = await graph.astream_events(
            given, config, version="v3", transformers=[_custom_mode()], **arguments
        )
    async with events:
        async for event in events:
            method, params = event["method"], event["params"]
            if method == "messages" or not params["namespace"]:
                yield method, params["data"], params.get("interrupts") or ()


@functools.cache
def _custom_mode() -> Any:
    """A stream transformer that asks the graph for its ``custom`` stream mode, and nothing
    else: ``astream_events(version="v3")`` streams only the modes its transformers ask for,
    and LangGraph's own ``CustomTransformer`` would take the ``custom`` projection a graph
    may already register."""
    from langgraph.stream import StreamTransformer

    class CustomMode(StreamTransformer):
        required_stream_modes = ("custom",)

        def init(self) -> dict[str, Any]:
            return {}

        def process(self, event: Any) -> bool:
            return True

    return CustomMode


async def _asked_again(graph: Any, config: Any, error: Exception) -> tuple[Any, list[Any]]:
    """The graph's state and its pauses, still open, after the answer a resume gave a pause
    was not one its ``response_schema`` takes (``is_invalid_resume``: a check the schema's
    JSON form, checked first, could not make — a validator of its own): LangGraph saved
    nothing, so the question is asked again rather than the run failing on it. Any other
    error is raised as it was."""
    from langgraph.errors import is_invalid_resume

    if not is_invalid_resume(error):
        raise error
    log.warning("an answer was not one the graph's interrupt takes, asked again: %s", error)
    state = await graph.aget_state(config)
    return state.values, list(state.interrupts)


def _written(runtime: Runtime, value: Any) -> None:
    """What a node of the graph wrote to LangGraph's ``custom`` stream mode
    (``get_stream_writer()``, a middleware's ``runtime.stream_writer``): a ``CUSTOM`` event of
    the run (:data:`~trellis.harness.events.STREAMED`), its ``value`` the write as JSON."""
    from pydantic_core import to_jsonable_python

    runtime.events.custom(STREAMED, value=to_jsonable_python(value, fallback=str))


def _own_spec(spec: ToolSpec, *, outside: bool) -> ToolSpec:
    """A graph's own tool as the run's tool set has it: Deep Agents' ``task`` a helper, and it
    and Deep Agents' file tools outside the state (``outside``) the framework's built-ins."""
    if spec.name == TASK_TOOL:
        return builtin(helper(spec))
    return builtin(spec) if outside else spec


def _run_arguments(runtime: Runtime, target: Any) -> dict[str, Any]:
    """The graph's own run arguments among the run's framework options (:data:`RUN_ARGUMENTS`).
    A worker's run of a graph that checkpoints into its run saves each step's checkpoint
    before the next step starts (LangGraph's ``durability="sync"``, unless the run asks
    otherwise): LangGraph's default saves it beside the next step, so a write of that step
    could run, and its worker die, before the step that planned it is saved, and the worker
    after it would plan and make the write again."""
    options = runtime.framework_options
    arguments = {key: options[key] for key in RUN_ARGUMENTS if key in options}
    if runtime.worker_id is not None and run_scoped(target):
        arguments.setdefault("durability", "sync")
    return arguments


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
        "narrowed, and its results kept for read_file and query_result as ReAct's are, but it "
        "has no memory or gateway MCP tools, no step limit or stall "
        "guard, no "
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


#: Checkpointers that keep nothing past their process: no second copy of anything.
EPHEMERAL_CHECKPOINTERS: Final = frozenset({"InMemorySaver", "MemorySaver"})
#: What is said, once per process and kind, of a graph that brings a durable checkpointer.
DUPLICATE_STORAGE: Final = (
    "the graph brings its own durable checkpointer ({name}): Trellis memory and runs already "
    "keep the conversation and the run's state, so both are stored twice; compile the graph "
    "without it (or with InMemorySaver) to keep one copy"
)
_warned: set[str] = set()


def durable_checkpointer(target: Any) -> str | None:
    """The kind of the durable checkpointer a graph brings of its own (one that keeps state
    past the process, e.g. a Postgres saver), or ``None``: none, an in-memory one, or the
    run's own (``RunCheckpointer``)."""
    if not _checkpointed(target) or run_scoped(target):
        return None
    name = type(target.checkpointer).__name__
    return None if name in EPHEMERAL_CHECKPOINTERS else name


def run_scoped(target: Any) -> bool:
    """Whether the graph checkpoints into its run (``middleware.RunCheckpointer``)."""
    if not _checkpointed(target):
        return False
    from trellis.harness.middleware import RunCheckpointer

    return isinstance(target.checkpointer, RunCheckpointer)


def carries(target: Any) -> bool:
    """Whether a conversation carries state of ``target`` from turn to turn: a graph with one
    of the channels :data:`CARRIED` names and no thread of its own that holds it — it
    checkpoints into its run, or not at all."""
    channels = getattr(target, "channels", None) or {}
    own = _checkpointed(target) and not run_scoped(target)
    return not own and any(name in channels for name in CARRIED)


def _ended_with(target: Any, runtime: Any, values: Any) -> None:
    """What the run's conversation carries to its next turn (:data:`CARRIED`), read from the
    state the graph ended its turn with (``values``, whole: a checkpoint holds a channel kept
    as deltas — Deep Agents' ``files`` — only now and then), over what the turn began with
    (:func:`carried_of`)."""
    if carries(target) and isinstance(values, Mapping):
        runtime.carried = carried_of(values, getattr(runtime, "carried", None) or {})


def carried_of(values: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    """The channels of a graph's state its conversation carries (:data:`CARRIED`), as
    ``values`` holds them now: those it holds anything in — Deep Agents sets its list of
    tasks, empty, on every turn, and a turn that started none carries nothing — and those the
    turn began with (``before``) and emptied, carried empty. An entry of a mapping the turn
    began with and no longer holds (a file it deleted) is carried as ``None``: what merges
    the conversation's states (``tools.results.merged``) keeps the entries of both, and Deep
    Agents' files take ``None`` for a file that is gone."""
    out: dict[str, Any] = {}
    for name in CARRIED:
        now, was = values.get(name), before.get(name)
        if not now and not was:
            continue
        if isinstance(now, Mapping) and isinstance(was, Mapping):
            gone = {key: None for key, entry in was.items() if entry is not None and key not in now}
            out[name] = {**now, **gone}
        else:
            out[name] = now
    return out


def carried_plain(values: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    """``before`` with the carried channels a checkpoint's ``values`` hold whole — a channel
    kept as deltas is not in every checkpoint, and is read where the turn ends
    (:func:`_ended_with`)."""
    whole = {n: values[n] for n in CARRIED if isinstance(values.get(n), dict | list)}
    return {**before, **carried_of(whole, {n: before[n] for n in whole if n in before})}


def checkpointing(target: Any) -> Any:
    """``target`` as the harness runs it: a graph built with ``middleware()`` (it takes the
    run's tools per model call) and compiled without a checkpointer, as a copy checkpointing
    into its run (``middleware.RunCheckpointer``: a resume continues where it stopped) — the
    graph itself left as it is; any other as it is (a checkpointer of its own, or
    ``checkpointer=False``, stays the graph's). A graph checkpointing into its run — so, or
    built with a ``RunCheckpointer`` — reads its checkpoint back strictly
    (``middleware.STRICT_SERDE``), its own state's types allowed (``state_types``)."""
    from trellis.harness.middleware import RunCheckpointer, state_types

    saver = getattr(target, "checkpointer", False)
    if saver is None and HARNESS_TOOLS in (getattr(target, "channels", None) or {}):
        saver = RunCheckpointer()
    if not isinstance(saver, RunCheckpointer):
        return target
    # its strict serializer is told the graph's own state types, by LangGraph's own means
    typed = saver.with_allowlist(state_types(target))
    return target if typed is target.checkpointer else target.copy({"checkpointer": typed})


async def _continued(target: Any, native_input: Any, run: Invocation) -> Any:
    """What the graph is run with — and, a checkpointed graph, where the attempt's calls are
    made (``Replay.where``: the step's checkpoint) and the checkpoint it continues from
    (``Replay.resumed``). A graph run again without a decision to resume with (after a crash,
    or an error that may pass) continues from the run's latest checkpoint (``None``) rather
    than starting a new turn: one its journal holds (``RunCheckpointer``), or one its own
    checkpointer holds for this run (:data:`RUN_METADATA`); a graph whose checkpointer holds
    none of the run's starts from its input."""
    from langgraph.types import Command

    runtime = run.runtime
    # a new turn's state starts with what its conversation carries (:data:`CARRIED`)
    carried = getattr(runtime, "carried", None)
    started = (
        _carrying(target, native_input, carried)
        if carried and isinstance(native_input, dict)
        else native_input
    )
    if carried and ACTIVE_AGENT in carried:  # whose turn it is, on the run's stream
        runtime.events.custom(ACTIVE_AGENT, agent=carried[ACTIVE_AGENT])
    if not _checkpointed(target):
        return started
    replay = runtime.replay
    thread = thread_of(target, runtime)
    replay.where = _stepping(thread)
    resumed = isinstance(native_input, Command)
    if run_scoped(target):
        if not replay.journal.graph:
            return started
        replay.resumed = _latest(replay.journal.graph)
        return native_input if resumed else None
    state = await target.aget_state({"configurable": {"thread_id": thread}})
    marked: Mapping[str, object] = state.metadata or {}
    if runtime.attempt == 1 or marked.get(RUN_METADATA) != runtime.run_id:
        return native_input if resumed else _answered(target, native_input, state.values)
    configurable: Mapping[str, object] = state.config.get("configurable") or {}
    latest = configurable.get("checkpoint_id")
    replay.resumed = latest if isinstance(latest, str) else None
    return native_input if resumed else None


def _carrying(target: Any, native_input: dict[str, Any], carried: Mapping[str, Any]) -> Any:
    """A new turn's input with what its conversation carries: merged into it — or, when the
    graph's input leaves a carried channel out (LangChain's ``OmitFromInput``:
    ``TodoListMiddleware``'s ``todos``, which an input would drop), written to the graph's
    state as the turn starts (``Command(update=)``), what is not a channel of the graph left
    out as an input leaves it."""
    from langgraph.types import Command

    started = {**native_input, **carried}
    taken = target.get_input_jsonschema().get("properties") or {}
    if all(name in taken for name in carried):
        return started
    return Command(update={k: v for k, v in started.items() if k in target.channels})


def _answered(target: Any, native_input: Any, values: Any) -> Any:
    """A new turn's input, the calls of its thread's last message that have no result answered
    first (:data:`STOPPED_CALL`) — unless the graph answers them itself (Deep Agents'
    ``PatchToolCallsMiddleware``) or its state holds no messages."""
    from langchain_core.messages import AIMessage, ToolMessage

    held = values.get("messages") if isinstance(values, Mapping) else None
    given = native_input.get("messages") if isinstance(native_input, Mapping) else None
    if not held or not isinstance(given, list) or PATCHES_CALLS in target.nodes:
        return native_input
    last = held[-1]
    calls = last.tool_calls if isinstance(last, AIMessage) else []
    if not calls:
        return native_input
    stopped = [
        ToolMessage(
            STOPPED_CALL.format(tool=c["name"]),
            name=c["name"],
            tool_call_id=c["id"],
            status="error",
        )
        for c in calls
    ]
    return {**native_input, "messages": [*stopped, *given]}


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


def _task() -> str | None:
    """The LangGraph task a call is made in (its id, the same for each run of the task its
    ``RetryPolicy`` makes; a tool node's call is one of its task's): the decision a harness
    call comes from (``Runtime.deciding``); ``None`` outside a task of a graph."""
    from langgraph.config import get_config

    try:
        task = (get_config().get("configurable") or {}).get(TASK_ID)
    except RuntimeError:  # not inside a step of a graph
        return None
    return task if isinstance(task, str) else None


#: The ``type`` of a ``HumanInTheLoopMiddleware(interrupt_mode="per_call")`` request: one
#: call's approval (``ToolApprovalRequest``), answered with one decision.
TOOL_APPROVAL: Final = "tool_approval"
#: The key of an approval's payload holding the JSON Schema of the decision the request takes
#: (LangGraph's ``Interrupt.response_schema``: a per-call request's allowed decisions, an
#: edit's arguments as its tool takes them).
DECISION_SCHEMA: Final = "response_schema"


#: The state key the agent's plan is kept under (LangChain's ``TodoListMiddleware``, Deep Agents'
#: ``write_todos``), and the key of an approval's payload that holds it.
PLAN: Final = "todos"
#: The decisions a ``HumanInTheLoopMiddleware`` request may allow, in the order they are said.
DECISIONS: Final = ("approve", "edit", "reject", "respond")
#: What a person reads of a ``HumanInTheLoopMiddleware`` request (:func:`hitl_said`): its plan,
#: each step with its status, and the decisions it takes.
PLAN_SAID: Final = "Plan:\n{steps}"
STEP_SAID: Final = "- {content} ({status})"
DECISIONS_SAID: Final = "Decisions: {decisions}."


def is_hitl(value: Any) -> bool:
    """Whether an interrupt's value is a ``HumanInTheLoopMiddleware`` request: a batch's (its
    ``action_requests`` and their ``review_configs``) or one call's (``interrupt_mode=
    "per_call"``: a :data:`TOOL_APPROVAL`)."""
    if not isinstance(value, dict):
        return False
    if value.get("type") == TOOL_APPROVAL:
        return isinstance(value.get("name"), str) and isinstance(value.get("args"), dict)
    return (
        isinstance(value.get("action_requests"), list)
        and bool(value["action_requests"])
        and isinstance(value.get("review_configs"), list)
    )


def hitl_calls(request: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The calls a ``HumanInTheLoopMiddleware`` request holds, each its tool and arguments,
    in order: a batch's, or the one call of a per-call request."""
    actions = [request] if request.get("type") == TOOL_APPROVAL else request["action_requests"]
    return [(str(a.get("name")), dict(a.get("args") or {})) for a in actions]


def hitl_allowed(request: Mapping[str, Any]) -> list[str]:
    """The decisions a ``HumanInTheLoopMiddleware`` request takes, as :data:`DECISIONS` orders
    them: those every call of a batch allows (its ``allowed_decisions``), or those a per-call
    request's schema of them names (:data:`DECISION_SCHEMA`; none known without it)."""
    if request.get("type") == TOOL_APPROVAL:
        schema = request.get(DECISION_SCHEMA, {})
        branches = [schema, *schema.get("$defs", {}).values()]
        named = {b.get("properties", {}).get("type", {}).get("const") for b in branches}
    else:
        named = set.intersection(*(set(c["allowed_decisions"]) for c in request["review_configs"]))
    return [d for d in DECISIONS if d in named]


def hitl_said(request: Mapping[str, Any]) -> str:
    """What a person is asked of a ``HumanInTheLoopMiddleware`` request, in the library's own
    words: an approval of its first call (and how many more it holds), each call's
    ``description``, the agent's plan when it keeps one (:data:`PLAN`), and the decisions the
    request takes (:func:`hitl_allowed`)."""
    calls = hitl_calls(request)
    more = len(calls) - 1
    said = f"Approve {calls[0][0]}?"
    if more:
        said += f" (and {more} more call{'s' if more > 1 else ''})"
    actions = [request] if request.get("type") == TOOL_APPROVAL else request["action_requests"]
    described = "\n".join(str(a["description"]) for a in actions if a.get("description"))
    plan = request.get(PLAN, ())
    steps = [STEP_SAID.format(content=t.get("content"), status=t.get("status")) for t in plan]
    allowed = ", ".join(hitl_allowed(request))
    parts = (
        f"{said} {described}" if described else said,
        PLAN_SAID.format(steps="\n".join(steps)) if steps else "",
        DECISIONS_SAID.format(decisions=allowed) if allowed else "",
    )
    return "\n".join(p for p in parts if p)


def hitl_response(request: dict[str, Any], resolution: InterruptResolution) -> dict[str, Any]:
    """What a ``HumanInTheLoopMiddleware`` pause resumes with, for a harness decision: a
    batch's ``{"decisions": [...]}``, one decision per call it holds, in order — or a per-call
    request's one decision:

    * ``approve`` approves every call;
    * ``edit`` runs the call with the edited arguments — one call: a batch of several takes
      one decision for each, given as ``{"decisions": [...]}``;
    * ``reject`` rejects every call, with the reviewer's reason (``answer=``) as the message the
      model reads;
    * ``answer`` responds for every call with the answer (the tool does not run).

    An answer (or edit) that already is what the request takes (``{"decisions": [...]}``, a
    per-call decision) is sent as it is. A decision a call does not allow (its
    ``allowed_decisions``, a per-call request's schema of them: :data:`DECISION_SCHEMA`), or an
    edit of a batch in one set of arguments, raises ``ConfigurationError``: the run stays
    paused, for an answer it takes."""
    raw = resolution.payload if resolution.decision is InterruptDecision.EDIT else resolution.answer
    if request.get("type") == TOOL_APPROVAL:
        given = raw if isinstance(raw, dict) and "type" in raw else None
        decision = given or _decision(resolution, request)
        _allowed(request, decision)
        return decision
    if isinstance(raw, dict) and isinstance(raw.get("decisions"), list):
        return raw
    actions: list[dict[str, Any]] = request["action_requests"]
    configs: list[dict[str, Any]] = request["review_configs"]
    if resolution.decision is InterruptDecision.EDIT and len(actions) > 1:
        raise ConfigurationError(
            f"the approval holds {len(actions)} calls, and an edit's arguments are one call's: "
            'give one decision for each call, in order, as {"decisions": [...]}'
        )
    decisions = [_decision(resolution, action) for action in actions]
    for action, config, decision in zip(actions, configs, decisions, strict=False):
        allowed = config.get("allowed_decisions") or []
        if decision["type"] not in allowed:
            raise ConfigurationError(
                f"{action.get('name')} does not allow the decision {decision['type']!r} "
                f"(allowed: {', '.join(allowed)})"
            )
    return {"decisions": decisions}


def hitl_approved(
    request: Mapping[str, Any], resolution: InterruptResolution
) -> list[tuple[str, Any]]:
    """The calls a ``HumanInTheLoopMiddleware`` request's resume runs as a person decided —
    each call its decision approves, as the model made it, or edits, as edited
    (:func:`hitl_response`); a rejected or answered call runs nothing."""
    approved: list[tuple[str, Any]] = []
    for (name, args), decision in _hitl_decided(request, resolution):
        if decision.get("type") == "approve":
            approved.append((name, args))
        elif decision.get("type") == "edit":
            edited = decision.get("edited_action") or {}
            approved.append((str(edited.get("name", name)), edited.get("args") or {}))
    return approved


def hitl_rejected(
    request: Mapping[str, Any], resolution: InterruptResolution
) -> list[tuple[str, dict[str, Any]]]:
    """The calls a ``HumanInTheLoopMiddleware`` request's resume rejects as a person decided,
    each as the model made it: a decision on each, not a failed call of the run."""
    return [
        call
        for call, decision in _hitl_decided(request, resolution)
        if decision.get("type") == "reject"
    ]


def _hitl_decided(
    request: Mapping[str, Any], resolution: InterruptResolution
) -> list[tuple[tuple[str, dict[str, Any]], dict[str, Any]]]:
    """Each call a ``HumanInTheLoopMiddleware`` request holds, with the decision its resume
    sends for it (:func:`hitl_response`). Never a cancel: a cancelled run takes no attempt
    (``Agent._continue``), so no resume is sent."""
    sent = hitl_response(dict(request), resolution)
    decisions = [sent] if request.get("type") == TOOL_APPROVAL else sent["decisions"]
    return list(zip(hitl_calls(request), decisions, strict=False))


def _allowed(request: Mapping[str, Any], decision: Any) -> None:
    """A per-call request's decision, checked against the schema of those it takes
    (:data:`DECISION_SCHEMA`), as LangGraph checks it on the resume: refused here, the run
    stays paused rather than failing on it."""
    from jsonschema.exceptions import best_match
    from jsonschema.validators import validator_for

    schema = request.get(DECISION_SCHEMA)
    if not isinstance(schema, dict):
        return
    error = best_match(validator_for(schema)(schema).iter_errors(decision))
    if error is not None:
        raise ConfigurationError(
            f"{request.get('name')} does not take the decision {decision!r}: {error.message}"
        )


def _decision(resolution: InterruptResolution, action: Mapping[str, Any]) -> Any:
    decision = resolution.decision
    if decision is InterruptDecision.REJECT:
        reason = reason_of(resolution)
        return {"type": "reject", "message": reason} if reason else {"type": "reject"}
    if decision is InterruptDecision.EDIT:
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


def idle_execute(tool: Any) -> bool:
    """Whether ``tool`` is Deep Agents' ``execute`` on a backend that runs no command (its
    default ``StateBackend``): ``create_deep_agent`` lists it whatever the backend, and its
    ``FilesystemMiddleware`` takes it off each model call there — no tool of the run's that
    runs code (``tools.export.runs_code``)."""
    from deepagents.middleware.filesystem import supports_execution

    backend = _execute_backend(tool)
    return backend is not None and not supports_execution(backend)


def _execute_backend(tool: Any) -> Any | None:
    """The backend of Deep Agents' ``execute`` ``tool``, read from the middleware its function
    closes over (``self``); ``None`` for any other tool."""
    if getattr(tool, "name", None) != "execute":
        return None
    try:
        owner = inspect.getclosurevars(tool.coroutine or tool.func).nonlocals.get("self")
    except (AttributeError, TypeError, ValueError):  # not Deep Agents' own
        return None
    return getattr(owner, "backend", None)


def _code_of(tool: Any) -> Code:
    """What ``tool`` runs code with (``LangGraphAdapter.code``): Deep Agents' ``execute`` on the
    harness's default code (``default``) or on a backend that runs commands (``own``),
    LangChain's ``shell`` (``own``); else ``none``."""
    backend = _execute_backend(tool)
    if getattr(getattr(backend, "default", backend), DEFAULT_CODE, False):
        return "default"  # a ``CodeBackend``, or a composite on one (the skills' route)
    shell = getattr(tool, "name", None) == "shell" or not (backend is None or idle_execute(tool))
    return "own" if shell else "none"


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
    """The sub-agent graphs a Deep Agents ``task`` tool runs (:func:`named_subagent_graphs`),
    each once."""
    return list({id(g): g for g in named_subagent_graphs(task).values()}.values())


def named_subagent_graphs(task: Any) -> dict[str, Any]:
    """The sub-agent graphs a Deep Agents ``task`` tool runs, by name, as its code holds them
    (the graphs its sub-agents were compiled to); ``{}`` when they cannot be read (a release
    laid out otherwise: its sub-agents' tools are then left out of the run's set, and it has no
    background tasks)."""
    from langgraph.pregel import Pregel

    seen: set[int] = set()
    graphs: dict[str, Any] = {}
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
                graphs.update((k, v) for k, v in value.items() if isinstance(v, Pregel))
            elif callable(value) and hasattr(value, "__closure__"):
                pending.append(value)
    return graphs


def _checkpointed(target: Any) -> bool:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    return isinstance(getattr(target, "checkpointer", None), BaseCheckpointSaver)


def langchain_message(said: Said) -> Any:
    """``said`` as LangChain's own message: an assistant's with its tool calls, a tool's with
    the call it answers, a notification marked as one (``add_notification_metadata``)."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        add_notification_metadata,
    )

    if said.role is Speaker.USER and said.notified:
        data = said.notified.get("data")
        return add_notification_metadata(
            HumanMessage(content=said.content),
            source=str(said.notified.get("source")),
            data=data if isinstance(data, dict) else None,
        )
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


def input_screens(target: Any) -> list[Any]:
    """The ``PIIMiddleware`` instances of a ``create_agent`` graph that check the user's
    message before each model call (``apply_to_input``), in the graph's order: read from the
    ``before_model`` nodes ``create_agent`` made of them. ``[]`` for any other graph."""
    from langchain.agents.middleware import PIIMiddleware

    found: list[Any] = []
    for node in (getattr(target, "nodes", None) or {}).values():
        bound = getattr(node, "bound", None)
        for hook in (getattr(bound, "func", None), getattr(bound, "afunc", None)):
            owner = getattr(hook, "__self__", None)
            screens = isinstance(owner, PIIMiddleware) and owner.apply_to_input
            if screens and not any(owner is m for m in found):
                found.append(owner)
    return found


def screened(target: Any, text: str) -> str:
    """``text`` the harness sends with the user's message (the memory context), checked as the
    graph checks that message: the rule of each of its ``PIIMiddleware`` instances applied
    (:func:`input_screens`: its detector and strategy — redacted, masked or hashed; a
    ``block`` raises the library's ``PIIDetectionError``, which ends the run as a blocked
    question does). As it is for a graph that checks no input."""
    from langchain.agents.middleware._redaction import apply_strategy

    for screen in input_screens(target):
        if matches := screen.detector(text):
            text = apply_strategy(text, matches, screen.strategy)
    return text


def _asked_at(messages: Sequence[Any]) -> int | None:
    """Where the run's question is in ``messages``: the message marked :data:`ASKED`, else the
    last user message neither steered into the run nor added by the harness (``notified``)."""
    humans = [i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"]
    marked = [i for i in humans if ASKED in (getattr(messages[i], "response_metadata", None) or {})]
    typed = [i for i in humans if not _steered(messages[i]) and notified(messages[i]) is None]
    found = marked or typed
    return found[-1] if found else None


def asked(messages: Sequence[Any]) -> str | None:
    """The run's question as the graph's conversation keeps it (:func:`_asked_at`): its text
    after the graph's own middleware (``PIIMiddleware`` redacts it there); ``None`` without
    one, or when it holds no text."""
    at = _asked_at(messages)
    return (_text(messages[at]) or None) if at is not None else None


def added(messages: Sequence[Any]) -> list[Said]:
    """What the run said after its question (:func:`_asked_at`): each answer
    with the tool calls it made, and their results, each message steered into it
    (``middleware.STEERED``, with its id) and each notification it was given (``Said.notified``:
    a background task's update); ``[]`` without a question."""
    at = _asked_at(messages)
    if at is None:
        return []
    out: list[Said] = []
    for m in messages[at + 1 :]:
        kind = getattr(m, "type", None)
        if kind == "human" and (noted := notified(m)) is not None:
            out.append(Said(Speaker.USER, _text(m), id=m.id, notified=noted))
        elif kind == "human":  # steered into the run
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


def _structured(transcript: list[Said], answer: Any) -> list[Said]:
    """The turn as said, ending with its structured ``answer`` as the assistant's message (its
    JSON), as LangChain's provider strategy ends it (the model's reply is that JSON). Its tool
    strategy ends with the model's call of the schema's tool and LangChain's message answering
    it (``Returning structured response: ...``) — the only way ``create_agent`` ends on a
    tool's message — which are no steps of the turn (the run makes no such call): they give
    way to the answer."""
    from pydantic_core import to_jsonable_python

    if not transcript or transcript[-1].role is not Speaker.TOOL:
        return transcript
    said = json.dumps(to_jsonable_python(answer, fallback=str))
    return [*transcript[:-2], Said(Speaker.ASSISTANT, said)]


def notified(message: Any) -> dict[str, Any] | None:
    """The notification metadata of a message the harness or a framework added, which the user
    did not type — a background task's update, a summary of earlier turns (langchain-core's
    ``get_notification_metadata``); ``None`` for any other message."""
    from langchain_core.messages import BaseMessage, get_notification_metadata

    if not isinstance(message, BaseMessage):
        return None
    found = get_notification_metadata(message)
    return None if found is None else dict(found)


def _steered(message: Any) -> bool:
    from trellis.harness.middleware import STEERED

    return STEERED in (getattr(message, "response_metadata", None) or {})


def _text(message: Any) -> str:
    """A message's text (its text parts, when its content is a list of them)."""
    text = message.text if isinstance(getattr(message, "text", None), str) else message.content
    return text if isinstance(text, str) else ""


def _answer_text(messages: list[Any]) -> str | None:
    """The answer of a graph whose state is its conversation: the text of the tool result it
    ended with (a ``return_direct`` tool's: ``create_agent`` ends its loop on it), else of its
    model's last reply that has any."""
    if messages and getattr(messages[-1], "type", None) == "tool":
        return _text(messages[-1])
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and (text := _text(message)):
            return text
    return None


def fields_of(value: Any) -> Any:
    """A graph's output as the mapping of its state's fields — a state LangGraph gives as its
    Pydantic model or dataclass (``version="v2"`` coerces the output to the state's schema), as
    one; any other output (a ``@entrypoint``'s return value, a state dict) as it is."""
    if isinstance(value, BaseModel):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    return value


def takes_text(target: Any) -> bool:
    """Whether the graph's input schema takes a string as it is (a ``@entrypoint`` typed to
    take one, or not typed at all): a run's string input is then given to it unchanged, where
    a graph whose input is its state gets it as the conversation's user message."""
    schema: dict[str, Any] = target.get_input_jsonschema()
    said = [schema, *schema.get("anyOf", ())]
    constrained = {"type", "anyOf", "oneOf", "allOf", "properties", "$ref"} & schema.keys()
    return not constrained or any(k.get("type") == "string" for k in said)
