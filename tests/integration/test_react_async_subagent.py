"""An investigation, kept as tests: "start a slow task in the background, keep chatting in the
same chat, get the result later" through ``ReAct`` with Deep Agents' own async sub-agents (an
``AsyncSubAgent`` dict, ``{"name", "description", "graph_id", "url"}``, in ``subagents=``), one
user, one thread.

The remote sub-agent is a stand-in Agent Protocol server (FastAPI on a local port, below): the
five routes the ``langgraph_sdk`` client calls from ``deepagents.middleware.async_subagents``
— a thread made, a run started on it, the run read, the thread read, the run cancelled — whose
run takes :data:`TASK_SECONDS` and then answers :data:`RESULT`. No model is called: the chat
model is scripted, and reads a task id as a model would, from what it is shown.

Each test asserts what a user needs; a test whose docstring starts "Today:" pins an observation
that is not what a user needs, as the evidence of a finding. What they show:

* T1 and T2: the task starts, the turn ends at once with the task id, the next message is
  answered while the task runs. The five tools say what they do by default (:data:`RISKS`):
  a status check or the list never asks a person, a start, an update or a cancel is announced.
* T3: Deep Agents keeps its task list in the graph's state (``async_tasks``) and a ``ReAct``
  graph's state is its run's, so the harness carries that list from a conversation's turn to
  its next one (``adapters.langgraph.CARRIED``): ``check``, ``list``, ``update`` and
  ``cancel`` find the task in a later turn, memory on or off, and another user's thread of
  the same name does not. It is kept in the process that ran the turn and in the file store
  (``harness.files``): a turn another server runs, or one after a restart, starts with it;
  two turns that overlap (one waiting for an approval, a worker's still running) each keep
  their tasks.
* Nothing tells the chat when the task ends; cancelling the turn leaves the task running (the
  next turn's list has it); with no ``url`` it cannot start; memory on, the start's result in
  a later turn's history is left as it was said, and a status check's is marked out of date.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Final

import httpx
import pytest
import uvicorn
from deepagents import create_deep_agent
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk._async.runs import RunsClient

from tests.contract.test_runs_wire import RunsService
from tests.support.arm import arm, declaring
from tests.support.chat_model import ScriptedChatModel
from tests.support.journals import journal_of
from tests.support.openapi import runs_contract
from trellis import Harness, ReAct, Settings, middleware, tool
from trellis.contracts import ConfigurationError, RunEventType, RunOutcome, RunStatus
from trellis.harness import pipeline
from trellis.harness.adapters.langgraph import bound_tools, notified
from trellis.harness.agent import Agent
from trellis.harness.agui import threads as threads_module
from trellis.harness.agui.sse import decode
from trellis.harness.agui.threads import Threads
from trellis.harness.harness import LOCAL_TENANT
from trellis.harness.identity import conversation_key
from trellis.harness.runs import LocalRuns
from trellis.harness.tools.base import max_age
from trellis.runs import Job

#: How long the background task takes on the server, and what it answers.
TASK_SECONDS: Final = 1.5
RESULT: Final = "OPTIMUM: x=42, cost=7"

USER: Final = "ada"
THREAD: Final = "chat-1"
TOOLS: Final = (
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
)
#: What the five tools do, as the harness declares Deep Agents' own (``middleware.ASYNC_TASKS``).
RISKS: Final = {
    "start_async_task": "write",
    "update_async_task": "write",
    "cancel_async_task": "write",
    "check_async_task": "read",
    "list_async_tasks": "read",
}
START: Final = (
    "start_async_task",
    {"description": "Find the optimum.", "subagent_type": "optimizer"},
)
#: Everything a start sends the server (``client.runs.create``): the graph, the task's text, the
#: SDK's stream defaults and (Deep Agents 0.7.30 on) the config naming the run that started it
#: (``deepagents_parent``) — no idempotency key, no metadata.
SENT: Final = {
    "assistant_id",
    "input",
    "stream_mode",
    "stream_resumable",
    "stream_subgraphs",
    "config",
}
#: What Deep Agents' ``list_async_tasks`` says of a thread with no task.
NO_TASKS: Final = "No async subagent tasks tracked."


def asked_to_do(server: StandIn) -> list[tuple[str, str]]:
    """What the server was asked to do — a thread made, a run started or cancelled (its POSTs)
    — leaving out the reads of a task's run and thread with which Deep Agents (0.7.27 on)
    pulls a task's status itself, before each model call and after a final answer."""
    return [call for call in server.calls if call[0] == "POST"]


def only_read(server: StandIn, since: int = 0) -> bool:
    """Whether every call the server got from ``since`` on reads a task (a status pull)."""
    return all(method == "GET" for method, _ in server.calls[since:])


# --------------------------------------------------------------------------- the stand-in server
class StandIn:
    """An Agent Protocol server standing for a remote sub-agent's: exactly the routes the
    ``langgraph_sdk`` client calls from Deep Agents' async sub-agent tools. A run is ``running``
    for ``seconds`` and then ``success``, its thread's last message :data:`RESULT`."""

    def __init__(self, seconds: float = TASK_SECONDS) -> None:
        self.seconds = seconds
        #: every task runs until :meth:`release`, whatever the clock says (:meth:`hold`)
        self.held = False
        self.runs: dict[str, dict[str, Any]] = {}  # by thread id (the task id): its latest run
        self.calls: list[tuple[str, str]] = []  # (method, path), in order
        self.headers: list[dict[str, str]] = []
        self.bodies: list[Any] = []
        self.app = FastAPI()
        self.url = ""
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        app = self.app

        @app.post("/threads")  # client.threads.create()
        async def create_thread(request: Request) -> dict[str, Any]:
            await self._saw(request)
            return {"thread_id": str(uuid.uuid4()), "status": "idle", "values": None}

        @app.post("/threads/{thread_id}/runs")  # client.runs.create(thread_id=, ...)
        async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
            body = await self._saw(request)
            self.runs[thread_id] = {
                "run_id": str(uuid.uuid4()),
                "thread_id": thread_id,
                "assistant_id": body["assistant_id"],
                "status": "running",
                "input": body["input"],
                "started": time.monotonic(),
            }
            return self._run(thread_id)

        @app.get("/threads/{thread_id}/runs/{run_id}")  # client.runs.get(thread_id=, run_id=)
        async def get_run(thread_id: str, run_id: str, request: Request) -> dict[str, Any]:
            await self._saw(request)
            return self._run(thread_id)

        @app.get("/threads/{thread_id}")  # client.threads.get(thread_id=)
        async def get_thread(thread_id: str, request: Request) -> dict[str, Any]:
            await self._saw(request)
            run = self.runs[thread_id]
            said = list(run["input"]["messages"])
            if self.status(thread_id) == "success":
                said.append({"role": "assistant", "content": RESULT})
            return {"thread_id": thread_id, "status": "idle", "values": {"messages": said}}

        @app.post("/threads/{thread_id}/runs/{run_id}/cancel")  # client.runs.cancel(...)
        async def cancel_run(thread_id: str, run_id: str, request: Request) -> None:
            await self._saw(request)
            self.runs[thread_id]["status"] = "cancelled"

    async def _saw(self, request: Request) -> Any:
        self.calls.append((request.method, request.url.path))
        self.headers.append(dict(request.headers))
        body = await request.json() if await request.body() else None
        self.bodies.append(body)
        return body

    def _run(self, thread_id: str) -> dict[str, Any]:
        run = self.runs[thread_id]
        public = {k: run[k] for k in ("run_id", "thread_id", "assistant_id")}
        return {**public, "status": self.status(thread_id)}

    def status(self, task_id: str) -> str:
        run = self.runs[task_id]
        over = not self.held and time.monotonic() - run["started"] >= self.seconds
        return "success" if run["status"] == "running" and over else run["status"]

    def hold(self) -> None:
        """Every task runs until :meth:`release`, whatever the clock says: a turn that ends
        meanwhile did not wait for its task, which could not end before it — proved by the
        order things happen in, not by how long they took."""
        self.held = True

    def release(self) -> None:
        """Every task held ends now (``success``)."""
        self.held, self.seconds = False, 0.0

    @property
    def task_id(self) -> str:
        """The one task started here (a task's id is its thread's)."""
        [found] = self.runs
        return found

    async def finished(self) -> None:
        """Waits until every task started here has ended."""
        while any(self.status(t) == "running" for t in self.runs):  # noqa: ASYNC110 - by the clock
            await asyncio.sleep(0.05)

    def start(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self._server = uvicorn.Server(uvicorn.Config(self.app, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, args=([sock],), daemon=True)
        self._thread.start()
        while not self._server.started:
            time.sleep(0.01)

    def stop(self) -> None:
        assert self._server is not None and self._thread is not None
        self._server.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Iterator[StandIn]:
    found = StandIn()
    found.start()
    yield found
    found.stop()


# --------------------------------------------------------------------------- the scripted model
class Model(ScriptedChatModel):
    """A scripted model whose turn may be a function of the messages it is shown — it reads a
    task id where a model would, in a tool's result or in the conversation — and that stops
    answering from its ``hold_from``-th call (a turn still working)."""

    hold_from: int | None = None

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        if self.hold_from is not None and len(self.seen) >= self.hold_from:
            await asyncio.sleep(3600)
        return await super()._agenerate(messages, *args, **kwargs)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        if self.turns and callable(self.turns[0]):
            self.turns[0] = self.turns[0](messages)
        return super()._generate(messages, *args, **kwargs)


_ID: Final = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def id_in(messages: list[BaseMessage]) -> str | None:
    """The last task id in what the model is shown."""
    found = [i for m in messages for i in _ID.findall(str(m.content))]
    return found[-1] if found else None


def reports_the_id(messages: list[BaseMessage]) -> str:
    """The answer of a model that starts a task as the tool says to: it reports the task id."""
    return f"The optimization is running in the background. Task id: {id_in(messages[-1:])}"


def calls(name: str, **args: Any) -> Callable[[list[BaseMessage]], Any]:
    """A call of ``name`` with the task id the model can read in what it is shown."""

    def turn(messages: list[BaseMessage]) -> Any:
        found = id_in(messages)
        if found is None:
            return "I see no task id in this conversation."
        return (name, {"task_id": found, **args})

    return turn


def says_what_the_tool_said(messages: list[BaseMessage]) -> str:
    return str(messages[-1].content)


def tool_results(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- the chat
def subagent(server: StandIn, **more: Any) -> Any:
    spec = {"name": "optimizer", "description": "Runs the slow optimization, in the background."}
    return {**spec, "graph_id": "optimizer", "url": server.url, **more}


def chat(harness: Harness, server: StandIn, model: Model, **more: Any) -> Agent:
    """A wrapped ``ReAct`` with the async sub-agent — and no word in the tool catalog on Deep
    Agents' five tools."""
    return harness.wrap(ReAct("You help.", model, subagents=[subagent(server)], **more), id="bg")


async def say(agent: Agent, text: str, **more: Any) -> Any:
    return await agent.run(text, user=USER, thread=THREAD, **more)


# --------------------------------------------------------------------------- T1, T2, T3
async def test_t1_the_task_starts_and_the_turn_ends_at_once_with_its_id(
    harness: Harness, server: StandIn
) -> None:
    model = Model(turns=[START, reports_the_id])
    agent = chat(harness, server, model)
    server.hold()
    first = await say(agent, "run the optimization")
    assert first.status is RunStatus.SUCCESS and first.interrupt is None, first.error
    assert set(TOOLS) <= set(model.requests[0]["tools"])  # the five tools, offered to the model
    started = [("POST", "/threads"), ("POST", f"/threads/{server.task_id}/runs")]
    assert asked_to_do(server) == started and only_read(server, since=2)
    assert server.bodies[1]["input"]["messages"][0]["content"] == "Find the optimum."
    assert tool_results(model.seen[1]) == [f"Launched async subagent. task_id: {server.task_id}"]
    assert server.task_id in first.answer  # the reply names the task
    # the turn ended while its task could not (held): it did not wait for the task, which
    # is still running
    assert server.status(server.task_id) == "running"


async def test_t2_the_next_message_is_answered_while_the_task_runs(
    harness: Harness, server: StandIn
) -> None:
    model = Model(turns=[START, reports_the_id, "4"])
    agent = chat(harness, server, model)
    server.hold()
    await say(agent, "run the optimization")
    second = await say(agent, "what is 2+2?")
    assert second.status is RunStatus.SUCCESS and second.answer == "4", second.error
    # answered while the task could not end (held): not queued behind it
    assert server.status(server.task_id) == "running"
    # and the second turn only read how the task goes (it runs: nothing to tell the model)
    assert len(asked_to_do(server)) == 2 and only_read(server, since=2)
    assert not [m for m in model.seen[2] if notified(m) is not None]


async def later_turn(harness: Harness, server: StandIn, *turns: Any) -> tuple[Model, Any]:
    """T1, T2, the task ended on the server, then a third turn scripted with ``turns``."""
    model = Model(turns=[START, reports_the_id, "4"])
    agent = chat(harness, server, model)
    await say(agent, "run the optimization")
    await say(agent, "what is 2+2?")
    await server.finished()
    model.turns += turns
    return model, await say(agent, "is the optimization done?")


async def test_t3_a_later_turn_gets_the_result(harness: Harness, server: StandIn) -> None:
    model, third = await later_turn(
        harness, server, calls("check_async_task"), says_what_the_tool_said
    )
    assert third.status is RunStatus.SUCCESS, third.error
    assert RESULT in tool_results(model.seen[-1])[-1]
    assert RESULT in third.answer


async def test_t3_a_later_turn_lists_the_task(harness: Harness, server: StandIn) -> None:
    _, third = await later_turn(harness, server, ("list_async_tasks", {}), says_what_the_tool_said)
    assert third.status is RunStatus.SUCCESS, third.error
    assert f"task_id: {server.task_id}  agent: optimizer  status: success" in third.answer


async def test_a_later_turn_cancels_the_task(harness: Harness, server: StandIn) -> None:
    server.seconds = 60
    model = Model(turns=[START, reports_the_id, calls("cancel_async_task"), "cancelled"])
    agent = chat(harness, server, model)
    await say(agent, "run the optimization")
    await say(agent, "stop the optimization")
    assert server.status(server.task_id) == "cancelled"


async def test_a_later_turn_sends_the_task_new_instructions(
    harness: Harness, server: StandIn
) -> None:
    turns = [START, reports_the_id, calls("update_async_task", message="Minimise cost."), "sent"]
    agent = chat(harness, server, Model(turns=turns))
    await say(agent, "run the optimization")
    await say(agent, "minimise cost instead")
    [*_, sent] = [b for b in server.bodies if b is not None]
    assert sent["input"]["messages"][0]["content"] == "Minimise cost."


# --------------------------------------------------------------------------- a. state across turns
async def test_a_the_task_list_is_carried_whatever_history_the_next_turn_sees(
    harness: Harness, server: StandIn
) -> None:
    """Memory off, the next turn's history is the earlier questions and answers (the run
    store's): the task id is in it only because the model's answer said it, T1's tool result
    is not — and the id the model reads there is found in the task list the conversation
    carries, so the server is asked for its result."""
    model, third = await later_turn(
        harness, server, calls("check_async_task"), says_what_the_tool_said
    )
    shown = model.seen[3]  # the third turn's first model call
    assert not tool_results(shown)
    said = [str(m.content) for m in shown if isinstance(m, AIMessage)]
    assert said == [
        f"The optimization is running in the background. Task id: {server.task_id}",
        "4",
    ]
    assert RESULT in third.answer
    assert len(asked_to_do(server)) == 2  # the rest read the task


async def test_a_with_memory_the_start_in_the_history_is_not_marked_out_of_date(
    memory_harness: Harness, server: StandIn
) -> None:
    """Memory on, the next turn's history holds T1's call and its result (the id in it) as it
    was said: a start's result records what was done, it is no value to read again, and
    nothing advises calling ``start_async_task`` again — which would start a second task. A
    status check's earlier result is a value that changes, and is rightly marked."""
    model = Model(turns=[START, reports_the_id, calls("check_async_task"), "It runs.", "Yes."])
    agent = chat(memory_harness, server, model)
    server.hold()  # it runs throughout, however long the turns take
    await say(agent, "run the optimization")
    second = await say(agent, "is the optimization done?")
    assert second.status is RunStatus.SUCCESS, second.error
    started = f"Launched async subagent. task_id: {server.task_id}"
    assert tool_results(model.seen[2]) == [started]  # the second turn's history
    assert '"status": "running"' in tool_results(model.seen[3])[-1]  # the id, found
    await say(agent, "are you sure?")
    kept = tool_results(model.seen[4])
    assert kept[0] == started and len(server.runs) == 1
    assert kept[1].startswith("[out of date: call check_async_task again for the current value]")
    # and the context's note on out-of-date values names the check alone
    assert "The values check_async_task returned in earlier turns" in str(model.seen[4])


async def test_a_another_users_thread_of_the_same_name_has_no_task(
    harness: Harness, server: StandIn
) -> None:
    """The list is the conversation's — its user's thread: another user naming a thread
    alike is shown none, and asked for the first user's task id is refused it."""
    model = Model(turns=[START, reports_the_id])
    agent = chat(harness, server, model)
    await say(agent, "run the optimization")
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    asked = len(server.calls)
    other = await agent.run("which tasks are there?", user="bob", thread=THREAD)
    assert other.answer == NO_TASKS
    model.turns += [("check_async_task", {"task_id": server.task_id}), says_what_the_tool_said]
    other = await agent.run("how is that task?", user="bob", thread=THREAD)
    assert other.answer == f"No tracked task found for task_id: {server.task_id!r}"
    assert len(server.calls) == asked  # the server was asked nothing for bob
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    mine = await say(agent, "which tasks are there?")
    assert f"task_id: {server.task_id}  agent: optimizer  status: running" in mine.answer


async def test_a_within_one_turn_the_task_is_started_and_checked(
    harness: Harness, server: StandIn
) -> None:
    """The tools and the stand-in do work together while the graph's state is there: one turn
    starts the task and, the task over, reads its result."""
    server.seconds = 0.2
    turns = [START, calls("check_async_task"), says_what_the_tool_said]
    model = Model(turns=turns, stall=0.3)
    done = await say(chat(harness, server, model), "run the optimization and wait for it")
    assert done.status is RunStatus.SUCCESS, done.error
    assert RESULT in done.answer and '"status": "success"' in done.answer
    assert len(asked_to_do(server)) == 2 and only_read(server, since=2)


async def test_a_a_graph_with_its_own_checkpointer_keeps_the_task_list_across_turns(
    harness: Harness, server: StandIn
) -> None:
    """Not ``ReAct``: ``create_deep_agent`` with ``middleware()`` and a checkpointer of its own
    runs on the conversation's LangGraph thread, so its state — the task list — is the next
    turn's: T1, T2 and T3 all work, and another user's same-named thread has no task."""
    model = Model(turns=[START, reports_the_id, "4"])
    graph = create_deep_agent(
        model, subagents=[subagent(server)], middleware=middleware(), checkpointer=InMemorySaver()
    )
    agent = harness.wrap(graph, id="own-thread")
    server.hold()
    first = await say(agent, "run the optimization")
    assert server.task_id in first.answer
    assert (await say(agent, "what is 2+2?")).answer == "4"
    assert server.status(server.task_id) == "running"  # both turns ended while it could not
    server.release()
    await server.finished()
    model.turns += [calls("check_async_task"), says_what_the_tool_said]
    third = await say(agent, "is the optimization done?")
    assert RESULT in third.answer, third.answer
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    fourth = await say(agent, "which tasks are there?")
    assert f"task_id: {server.task_id}  agent: optimizer  status: success" in fourth.answer
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    other = await agent.run("which tasks are there?", user="bob", thread=THREAD)
    assert other.answer == NO_TASKS


# --------------------------------------------------------------------------- b. governance
async def test_b_the_five_tools_say_what_they_do(harness: Harness, server: StandIn) -> None:
    """The five tools are Deep Agents', and say nothing of what they do themselves: the
    harness says it for them, as it does of Deep Agents' ``execute`` — the run's tool set has
    each with its risk, with no word in the tool catalog."""
    agent = chat(harness, server, Model(turns=[]))
    own = await agent.adapter.native(agent.target)
    assert {spec.name: spec.side_effects for spec in own if spec.name in TOOLS} == RISKS


async def test_b_a_writes_result_is_no_value_to_read_again_and_a_reads_is(
    harness: Harness, server: StandIn
) -> None:
    """Beside what each does, the harness says how long its result stays current: a start, an
    update and a cancel return the record of what was done (never out of date: no advice to
    call it again), a status and the list a value that changes (live, as any tool's)."""
    agent = chat(harness, server, Model(turns=[]))
    own = {spec.name: spec for spec in await agent.adapter.native(agent.target)}
    assert {name: max_age(own[name]) for name in TOOLS} == {
        "start_async_task": None,
        "update_async_task": None,
        "cancel_async_task": None,
        "check_async_task": 0.0,
        "list_async_tasks": 0.0,
    }


async def test_b_a_task_tool_that_says_how_long_its_result_stays_current_keeps_its_word(
    harness: Harness, server: StandIn
) -> None:
    graph = create_deep_agent(
        Model(turns=[]), subagents=[subagent(server)], middleware=middleware()
    )
    [start] = [t for t in bound_tools(graph) if t.name == "start_async_task"]
    start.metadata = {"max_age": 60}
    agent = harness.wrap(graph, id="bg")
    own = {spec.name: spec for spec in await agent.adapter.native(agent.target)}
    assert max_age(own["start_async_task"]) == 60
    assert own["start_async_task"].side_effects == "write"  # and what it does is still said


@pytest.mark.parametrize("name", TOOLS)
async def test_b_a_read_runs_and_a_write_is_announced_and_none_asks_a_person(
    harness: Harness, server: StandIn, name: str
) -> None:
    """A status check and the list are reads: they run. A start, an update and a cancel are
    writes of Deep Agents' own tools, announced on the run's stream and not asked about."""
    args = {"description": "d", "subagent_type": "optimizer", "task_id": "t", "message": "m"}
    model = Model(turns=[(name, args), "done"])
    agent = chat(harness, server, model)
    events = [e async for e in agent.stream("go", user=USER, thread=THREAD)]
    assert events[-1].type is RunEventType.RUN_FINISHED
    assert events[-1].outcome is RunOutcome.SUCCESS  # it ran: no pause for an approval
    assert [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_START] == [name]
    announced = [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"]
    assert announced == ([name] if RISKS[name] == "write" else [])


async def test_b_the_catalog_still_decides_a_start_declared_irreversible_asks(
    harness: Harness, server: StandIn
) -> None:
    """What the harness says of the five tools is a default: the tool catalog's word comes
    first, so an app that wants a person to approve every start says so there."""
    declaring(harness, start_async_task="irreversible")
    agent = chat(harness, server, Model(turns=[START, reports_the_id]))
    paused = await say(agent, "run the optimization")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "start_async_task" and server.calls == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer=USER)
    assert done.status is RunStatus.SUCCESS and len(server.runs) == 1, done.error


# --------------------------------------------------------------------------- c. journal, retries
@tool(side_effects="irreversible")
def publish(text: str) -> str:
    """Publish a text to the customers' page, once a person approved it.

    Args:
        text: The text to publish.
    """
    return f"published {text}"


async def test_c_a_resume_after_the_start_starts_no_second_task(
    harness: Harness, server: StandIn
) -> None:
    """The call is recorded on the run (its trajectory) and its result kept in the graph's
    checkpoint in the run's journal: a turn paused after the start and resumed continues from
    there."""
    model = Model(turns=[START, ("publish", {"text": "started"}), "done"])
    agent = arm(chat(harness, server, model), [publish])
    paused = await say(agent, "run the optimization and publish that it started")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    journal = await journal_of(harness.runs, await harness.runs.get(paused.run_id))
    [(made, outcome)] = journal.trajectory
    assert made.tool == "start_async_task" and server.task_id in str(outcome.output)
    assert journal.graph  # the graph's checkpoint: the tool's message and async_tasks
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer=USER)
    assert done.status is RunStatus.SUCCESS and done.answer == "done", done.error
    assert len(server.runs) == 1 and len(model.seen) == 3  # one task; the model not asked again


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


@pytest.fixture
async def workers() -> AsyncIterator[tuple[Harness, Harness]]:
    """Two worker processes: two harnesses sharing nothing but the run store."""
    store = LocalRuns()
    pair = (Harness(config=Settings(), runs=store), Harness(config=Settings(), runs=store))
    yield pair
    for h in pair:
        await h.aclose()


async def crash_then_finish(
    workers: tuple[Harness, Harness], server: StandIn, model: Model, tools: list[Any]
) -> Any:
    """The turn queued, run by the first worker, which dies mid-run; the second finishes it."""
    first, second = workers
    dying, taking_over = (chat(h, server, model, tools=tools) for h in (first, second))
    store = first.runs
    assert isinstance(store, LocalRuns)
    handle = await dying.start("run the optimization", user=USER, thread=THREAD)
    worker = first.worker([dying])
    claimed = await store.claim(worker.worker_id, [dying.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await dying.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await second.worker([taking_over]).run_once()
    result = await handle.result(wait=5)
    assert (await handle.status()).attempt == 2
    return result


async def test_c_a_crash_after_the_start_starts_no_second_task(
    workers: tuple[Harness, Harness], server: StandIn
) -> None:
    crashes: list[BaseException] = [Crash()]

    @lc_tool
    def boom() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    boom.metadata = {"side_effects": "read"}
    model = Model(turns=[START, ("boom", {}), reports_the_id])
    result = await crash_then_finish(workers, server, model, [boom])
    assert result.status is RunStatus.SUCCESS, result.error
    assert len(server.runs) == 1  # the retried turn did not start the task again
    assert server.task_id in tool_results(model.seen[-1])[0]  # its result, from the checkpoint


async def test_c_today_a_crash_inside_the_start_leaves_a_task_nobody_holds_the_id_of(
    workers: tuple[Harness, Harness], server: StandIn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today: the worker dies after the server took the run and before the call's result was
    kept. The retried turn does not start it blind — the model is told the effect is unknown
    — but the request carries no idempotency key and the task's id was never kept, so the
    task runs on with nothing able to check, cancel or find it."""
    server.seconds = math.inf  # it runs on as long as nothing stops it: never by the clock
    crashes: list[BaseException] = [Crash()]
    create = RunsClient.create

    async def dies_once(self: RunsClient, *args: Any, **kwargs: Any) -> Any:
        run = await create(self, *args, **kwargs)
        if crashes:
            raise crashes.pop()
        return run

    monkeypatch.setattr(RunsClient, "create", dies_once)
    model = Model(turns=[START, says_what_the_tool_said])
    result = await crash_then_finish(workers, server, model, [])
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == (
        "start_async_task was interrupted by a crash; it may or may not have taken effect: "
        "check before calling it again"
    )
    assert len(server.runs) == 1 and server.status(server.task_id) == "running"
    assert set(server.bodies[1]) == SENT  # no idempotency key: nothing to find it again by
    assert all(id_in(shown) is None for shown in model.seen)  # its id reached no one


# --------------------------------------------------------------------------- d. events
async def test_d_the_stream_shows_the_call_and_nothing_tells_an_idle_chat_the_task_ended(
    harness: Harness, server: StandIn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn's stream shows the start as any tool call (announced: a write), its result
    naming the task. Deep Agents reads the task's status while the turn runs (0.7.27 on) and
    wakes an idle parent only on an Agent Server of its own, which a ``url`` sub-agent's parent
    is not: once the turn ended nothing asks the server, no run starts in the thread when the
    task ends, and the thread's stream stays silent."""
    monkeypatch.setattr(threads_module, "POLL_SECONDS", 0.01)
    model = Model(turns=[START, reports_the_id])
    agent = chat(harness, server, model)
    server.hold()  # it runs until the test ends it, below
    events = [e async for e in agent.stream("run the optimization", user=USER, thread=THREAD)]
    assert [e.type.value for e in events] == [
        "RUN_STARTED",
        "CUSTOM",  # tool_notice: the write, announced
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    notice = events[1].data
    assert (notice["name"], notice["tool"]) == ("tool_notice", "start_async_task")
    [told] = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert told["output"] == f"Launched async subagent. task_id: {server.task_id}"

    chunks: list[str] = []

    async def follow() -> None:
        async for chunk in Threads(agent).stream(LOCAL_TENANT, USER, THREAD, None):
            chunks.append(chunk)

    asked = len(server.calls)
    following = asyncio.create_task(follow())
    try:
        await asyncio.sleep(0.3)  # the thread's stream, open: T1's run replayed
        replayed = len(chunks)
        assert replayed and server.status(server.task_id) == "running"
        server.release()
        await server.finished()
        await asyncio.sleep(0.5)  # the task ended while no turn runs
        assert len(chunks) == replayed  # nothing reached the chat
    finally:
        following.cancel()
        await asyncio.gather(following, return_exceptions=True)
    assert len(asked_to_do(server)) == 2 and only_read(server, since=2)  # while the turn ran
    assert len(server.calls) == asked  # once it ended, nobody asked the server how it went
    runs = [r async for r in harness.runs.iterate(agent_id="bg", thread_id=THREAD)]
    assert len(runs) == 1  # and no run was started in the thread to say so


# --------------------------------------------------------------------------- a worker's turn
async def test_c_a_worker_of_the_process_runs_the_next_turn_with_the_task_list(
    workers: tuple[Harness, Harness], server: StandIn
) -> None:
    """The next message queued, and run by a worker of the process that ran the turn before:
    the list that turn ended with is there."""
    server.seconds = 0.2
    first, _ = workers
    model = Model(turns=[START, reports_the_id, calls("check_async_task"), says_what_the_tool_said])
    agent = chat(first, server, model)
    await say(agent, "run the optimization")
    await server.finished()
    handle = await agent.start("is the optimization done?", user=USER, thread=THREAD)
    assert await first.worker([agent]).run_once()
    done = await handle.result(wait=5)
    assert done.status is RunStatus.SUCCESS and RESULT in done.answer, done.error


async def test_c_a_worker_of_another_process_runs_the_next_turn_with_the_task_list(
    workers: tuple[Harness, Harness], server: StandIn
) -> None:
    """The list a turn ended with is kept in the file store too (``harness.files``: the folder
    or bucket the servers share). A worker of another process that runs the conversation's
    next turn — nothing of the first turn in its memory — starts with it: the id its history
    has (the first turn's answer) is found, and the server is asked for the result."""
    server.seconds = 0.2
    _, second = workers
    model = Model(turns=[START, reports_the_id, calls("check_async_task"), says_what_the_tool_said])
    here, there = (chat(h, server, model) for h in workers)
    await say(here, "run the optimization")
    await server.finished()
    handle = await here.start("is the optimization done?", user=USER, thread=THREAD)
    assert await second.worker([there]).run_once()
    done = await handle.result(wait=5)
    assert done.status is RunStatus.SUCCESS and RESULT in done.answer, done.error
    assert len(asked_to_do(server)) == 2 and only_read(server, since=2)


# --------------------------------------------------------------------------- servers, restarts
def kept_files(harness: Harness) -> list[str]:
    """Every file in the harness's file store, by its path below the store's root."""
    store = harness.files
    found = store.fs.find(store.root) if store.fs.exists(store.root) else []
    return sorted(str(path)[len(store.root) + 1 :] for path in found)


async def test_c_after_a_restart_the_next_turn_has_the_task_list(
    workers: tuple[Harness, Harness], server: StandIn
) -> None:
    """The server that ran the first turn is gone, its agent and what it kept in memory with
    it; the one started in its place — the same run store, the same file store — runs the
    conversation's next turns: the task is checked and listed."""
    server.seconds = 0.2
    first, second = workers
    await say(chat(first, server, Model(turns=[START, reports_the_id])), "run the optimization")
    await first.aclose()
    await server.finished()
    listing = [("list_async_tasks", {}), says_what_the_tool_said]
    model = Model(turns=[calls("check_async_task"), says_what_the_tool_said, *listing])
    restarted = chat(second, server, model)
    third = await say(restarted, "is the optimization done?")
    assert third.status is RunStatus.SUCCESS and RESULT in third.answer, third.answer
    listed = await say(restarted, "which tasks are there?")
    assert f"task_id: {server.task_id}  agent: optimizer  status: success" in listed.answer


async def test_c_two_servers_taking_turns_each_find_the_tasks_the_other_started(
    workers: tuple[Harness, Harness], server: StandIn
) -> None:
    """A conversation's turns run where they land. The first server starts a task, the second
    another; back on the first — whose memory holds the list its own turn ended with, without
    the second's task — the list has both, as it has on the second."""
    server.seconds = 60
    listing = [("list_async_tasks", {}), says_what_the_tool_said]
    model = Model(turns=[START, reports_the_id, START, reports_the_id, *listing, *listing])
    here, there = (chat(h, server, model) for h in workers)
    await say(here, "run the optimization")
    await say(there, "run another one")
    assert len(server.runs) == 2
    for agent in (here, there):
        listed = await say(agent, "which tasks are there?")
        for task in server.runs:
            assert f"task_id: {task}  agent: optimizer  status: running" in listed.answer


async def test_c_another_users_thread_and_another_agent_on_another_server_have_no_task(
    workers: tuple[Harness, Harness], server: StandIn
) -> None:
    """What the file store keeps is the conversation's and the agent's, as what the process
    keeps is: on another server, another user naming a thread alike is shown no task and
    refused the first user's task id, and so is another agent in the first user's thread."""
    _, second = workers
    model = Model(turns=[START, reports_the_id])
    here, there = (chat(h, server, model) for h in workers)
    await say(here, "run the optimization")
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    asked = len(server.calls)
    other = await there.run("which tasks are there?", user="bob", thread=THREAD)
    assert other.answer == NO_TASKS
    refused = f"No tracked task found for task_id: {server.task_id!r}"
    model.turns += [("check_async_task", {"task_id": server.task_id}), says_what_the_tool_said]
    other = await there.run("how is that task?", user="bob", thread=THREAD)
    assert other.answer == refused
    another = second.wrap(ReAct("You help.", model, subagents=[subagent(server)]), id="other")
    model.turns += [("check_async_task", {"task_id": server.task_id}), says_what_the_tool_said]
    assert (await say(another, "how is that task?")).answer == refused
    assert len(server.calls) == asked  # the server was asked nothing for either


async def test_c_the_kept_list_is_none_of_the_users_files(
    harness: Harness, server: StandIn
) -> None:
    """Where it is kept: one small file per turn that kept it, in a folder per conversation and
    agent, under its tenant's and its user's names, in a folder of the store that is no tenant's (none starts with a dot) —
    none of the files kept for the user, whose listings show nothing. It holds Deep Agents'
    record of each task (ids, the sub-agent's name, a status, three times) and nothing of
    the sub-agent's server: no address, no header."""
    spec = subagent(server, headers={"authorization": "Bearer of-the-app"})
    model = Model(turns=[START, reports_the_id])
    agent = harness.wrap(ReAct("You help.", model, subagents=[spec]), id="bg")
    await say(agent, "run the optimization")
    folder = f".carried/{LOCAL_TENANT}/{USER}/{conversation_key(LOCAL_TENANT, USER, THREAD)}/bg"
    [path] = kept_files(harness)
    where, _, name = path.rpartition("/")
    assert where == folder and name.endswith(pipeline.CARRIED_SUFFIX)
    written = await harness.files.read(folder, name)
    assert written is not None and len(written) < 1_000
    kept = json.loads(written)
    assert list(kept) == ["async_tasks"] and list(kept["async_tasks"]) == [server.task_id]
    assert set(kept["async_tasks"][server.task_id]) == {
        "task_id",
        "agent_name",
        "thread_id",
        "run_id",
        "status",
        "created_at",
        "last_checked_at",
        "last_updated_at",
    }
    assert server.url not in written.decode() and "Bearer" not in written.decode()
    mine = f"{LOCAL_TENANT}/{USER}"
    assert await harness.files.files(mine) == [] and await harness.files.folders(mine) == []


async def test_c_a_turn_that_started_no_task_keeps_nothing_in_the_file_store(
    harness: Harness, server: StandIn
) -> None:
    agent = chat(harness, server, Model(turns=["4"]))
    assert (await say(agent, "what is 2+2?")).answer == "4"
    assert kept_files(harness) == []


@pytest.mark.parametrize("failing", ["files", "read", "put"])
async def test_c_a_file_store_out_of_reach_leaves_the_list_in_the_process(
    harness: Harness,
    server: StandIn,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing: str,
) -> None:
    """The file store cannot be read, or written (a bucket's credentials, a full disk): it is
    said in the log and the turn goes on. The list is kept in the process, as it was before
    the store kept it too: the next turn there finds the task."""

    async def refused(*args: Any, **kwargs: Any) -> Any:
        raise OSError("the bucket is out of reach")

    monkeypatch.setattr(harness.files, failing, refused)
    server.seconds = 0.2
    model = Model(turns=[START, reports_the_id, calls("check_async_task"), says_what_the_tool_said])
    agent = chat(harness, server, model)
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        first = await say(agent, "run the optimization")
        await server.finished()
        done = await say(agent, "is the optimization done?")
    assert first.status is RunStatus.SUCCESS and RESULT in done.answer, done.answer
    assert "the bucket is out of reach" in caplog.text
    assert bool(kept_files(harness)) is (failing != "put")  # written, where it can be


async def test_c_a_list_past_the_bound_is_kept_in_the_process_only(
    harness: Harness,
    server: StandIn,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """What the file store keeps for a conversation is bounded (``pipeline.CARRIED_BYTES``): a
    list past it is not written — said in the log — and stays the process's."""
    monkeypatch.setattr(pipeline, "CARRIED_BYTES", 64)
    server.seconds = 0.2
    model = Model(turns=[START, reports_the_id, calls("check_async_task"), says_what_the_tool_said])
    agent = chat(harness, server, model)
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        await say(agent, "run the optimization")
    assert kept_files(harness) == [] and "over the 64 bytes" in caplog.text
    await server.finished()
    assert RESULT in (await say(agent, "is the optimization done?")).answer


@pytest.mark.parametrize("written", [b'{"async_tasks": {"cut off', b'["not", "a", "state"]'])
async def test_c_a_kept_list_that_cannot_be_read_back_is_left_out_and_said(
    workers: tuple[Harness, Harness],
    server: StandIn,
    caplog: pytest.LogCaptureFixture,
    written: bytes,
) -> None:
    """What the store holds for the conversation is not what a turn kept there (a write cut
    off, another program's file): another server's turn starts without a list, said in the
    log — the id is refused, as it was before the store kept the list — and the turn that
    next ends with tasks writes the file anew."""
    server.seconds = 60
    first, second = workers
    turns = [START, reports_the_id, calls("check_async_task"), says_what_the_tool_said]
    model = Model(turns=[*turns, START, reports_the_id])
    here, there = (chat(h, server, model) for h in workers)
    await say(here, "run the optimization")
    [path] = kept_files(first)
    folder, _, name = path.rpartition("/")
    await first.files.put(folder, name, written, source="test")
    started = server.task_id
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        done = await say(there, "is the optimization done?")
    assert done.answer == f"No tracked task found for task_id: {started!r}"
    assert "what the conversation carries" in caplog.text
    await say(there, "run another one")
    [kept] = kept_files(second)  # the turn's own file; the one that held no state removed
    assert not kept.endswith(name)
    again = await second.files.read(folder, kept.rpartition("/")[2])
    assert again is not None and len(json.loads(again)["async_tasks"]) == 1


async def test_c_with_agent_runs_as_the_run_store_another_server_has_the_task_list(
    server: StandIn,
) -> None:
    """The same with agent-runs as the run store: two servers, each with the client of the
    one service (a stand-in that checks every exchange against agent-runs' OpenAPI
    document), sharing the file store. The second server's turn has the list, every exchange
    is the document's, and nothing of the list is in a run's record."""
    service = RunsService(runs_contract())
    server.seconds = 0.2
    model = Model(turns=[START, reports_the_id, calls("check_async_task"), says_what_the_tool_said])
    async with (
        Harness(config=Settings(), runs=service.client()) as first,
        Harness(config=Settings(), runs=service.client()) as second,
    ):
        await say(chat(first, server, model), "run the optimization")
        await server.finished()
        done = await say(chat(second, server, model), "is the optimization done?")
    assert done.status is RunStatus.SUCCESS and RESULT in done.answer, done.answer
    assert service.violations == [], "\n".join(service.violations)
    records = [r.model_dump_json() for r in service.store._runs.values()]
    assert len(records) == 2 and not [r for r in records if "async_tasks" in r]


# --------------------------------------------------------------------------- turns that overlap
async def test_g_a_turn_approved_after_another_turn_ended_loses_neither_turns_task(
    harness: Harness, server: StandIn
) -> None:
    """A turn starts a task and waits for a person's approval; the conversation's next turn
    starts another task and ends; the first, approved, ends after it. What each ended with is
    merged task by task, not put in the other's place: the list has both."""
    server.seconds = 60
    publishing = [START, ("publish", {"text": "started"})]
    listing = [("list_async_tasks", {}), says_what_the_tool_said]
    model = Model(turns=[*publishing, START, reports_the_id, "done", *listing])
    agent = arm(chat(harness, server, model), [publish])
    paused = await say(agent, "run the optimization and publish that it started")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    second = await say(agent, "run another one")
    assert second.status is RunStatus.SUCCESS and len(server.runs) == 2, second.error
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer=USER)
    assert done.status is RunStatus.SUCCESS and done.answer == "done", done.error
    listed = await say(agent, "which tasks are there?")
    for task in server.runs:
        assert f"task_id: {task}  agent: optimizer  status: running" in listed.answer


async def test_g_a_turn_waiting_for_an_approval_has_its_task_in_the_next_turns_list(
    harness: Harness, server: StandIn
) -> None:
    """A task started before a turn waits for a person's approval runs on the server whatever
    the person decides: the list a paused turn holds is kept as an ended turn's is, so the
    conversation's next turn — while the first still waits — lists the task."""
    server.seconds = 60
    listing = [("list_async_tasks", {}), says_what_the_tool_said]
    model = Model(turns=[START, ("publish", {"text": "started"}), *listing])
    agent = arm(chat(harness, server, model), [publish])
    paused = await say(agent, "run the optimization and publish that it started")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    listed = await say(agent, "which tasks are there?")
    assert f"task_id: {server.task_id}  agent: optimizer  status: running" in listed.answer


async def test_g_a_turn_cancelled_at_its_approval_leaves_its_task_for_the_next_turn_to_find(
    harness: Harness, server: StandIn
) -> None:
    """The person cancels the turn at its approval: the turn ends there, no attempt runs, and
    the task it had started is still running on the server. The next turn lists it and
    cancels it by the id read there."""
    server.seconds = 60
    stopping = [("list_async_tasks", {}), calls("cancel_async_task"), "It is stopped."]
    model = Model(turns=[START, ("publish", {"text": "started"}), *stopping])
    agent = arm(chat(harness, server, model), [publish])
    paused = await say(agent, "run the optimization and publish that it started")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    ended = await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer=USER)
    assert ended.status is RunStatus.CANCELLED and server.status(server.task_id) == "running"
    await say(agent, "is the optimization running? stop it")
    assert (
        f"task_id: {server.task_id}  agent: optimizer  status: running"
        in tool_results(model.seen[-2])[-1]
    )
    assert server.status(server.task_id) == "cancelled"


async def test_g_a_workers_turn_still_running_while_another_turn_ends_loses_neither_turns_task(
    harness: Harness, server: StandIn
) -> None:
    """A worker's turn is not held behind the conversation's turn in the process: it starts a
    task and is still working (a slow tool) while a message sent directly starts another and
    ends; the worker's turn ends after it. The list has both tasks."""
    server.seconds = 60
    entered, released = asyncio.Event(), asyncio.Event()

    @lc_tool
    async def wait() -> str:
        """Waits until it is let go."""
        entered.set()
        await released.wait()
        return "waited"

    wait.metadata = {"side_effects": "read"}
    slow = [START, ("wait", {}), START, reports_the_id, "started"]
    model = Model(turns=[*slow, ("list_async_tasks", {}), says_what_the_tool_said])
    agent = chat(harness, server, model, tools=[wait])
    handle = await agent.start("run the optimization, then wait", user=USER, thread=THREAD)
    working = asyncio.create_task(harness.worker([agent]).run_once())
    await entered.wait()
    second = await say(agent, "run another one")
    assert second.status is RunStatus.SUCCESS and len(server.runs) == 2, second.error
    released.set()
    assert await working
    assert (await handle.result(wait=5)).answer == "started"
    listed = await say(agent, "which tasks are there?")
    for task in server.runs:
        assert f"task_id: {task}  agent: optimizer  status: running" in listed.answer


def ending(run_id: str, task: str) -> Any:
    """A turn of one conversation ending with the task it started in its list (as much of a
    run as keeping it reads)."""
    return SimpleNamespace(
        tenant="t",
        user="u",
        conversation="c1",
        run_id=run_id,
        carried={"async_tasks": {task: {"task_id": task, "status": "running"}}},
    )


async def test_g_two_turns_ending_together_both_keep_their_task_in_the_file_store(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two turns of a conversation end together, each keeping the list it ended with in the
    file store (read what is kept, merge, write; the store slow to answer a read): they keep
    it one after the other, so the second merges over the first's and neither task is lost."""
    agent = harness.wrap(create_deep_agent(model=Model(turns=["hi"]), tools=[]), id="desk")
    read = harness.files.read

    async def slowly(folder: str, name: str) -> bytes | None:
        found = await read(folder, name)  # what is kept now, answered a while later
        await asyncio.sleep(0.05)
        return found

    monkeypatch.setattr(harness.files, "read", slowly)
    await asyncio.gather(
        pipeline._keep_carried(agent, ending("run_x", "task_x")),
        pipeline._keep_carried(agent, ending("run_y", "task_y")),
    )
    kept = await pipeline._carried(agent, ending("run_z", "none"))
    assert set(kept["async_tasks"]) == {"task_x", "task_y"}


@pytest.mark.parametrize("writes_first", ["here", "there"])
async def test_g_two_servers_in_the_worst_order_both_keep_their_task(
    workers: tuple[Harness, Harness], monkeypatch: pytest.MonkeyPatch, writes_first: str
) -> None:
    """Two servers end turns of one conversation together, in the worst order: each reads
    what is kept before either writes, then one writes and removes what it read, then the
    other. The file store has no conditional write, so neither writes over a file of the
    other's: each keeps its own, and the next turn, on either server, finds both tasks. What a
    turn read is removed once its file holds it: one file each is left, then one in all once
    a later turn merged them."""
    here, there = (
        h.wrap(create_deep_agent(model=Model(turns=["hi"]), tools=[]), id="desk") for h in workers
    )
    # a list the conversation had before, kept as an earlier turn left it
    await pipeline._keep_carried(here, ending("run_0", "task_0"))
    order = ["here", "there"] if writes_first == "here" else ["there", "here"]
    both_read, turn = asyncio.Event(), {name: asyncio.Event() for name in order}
    waiting: list[str] = []
    for name, h in zip(("here", "there"), workers, strict=True):
        files = h.files

        async def put(*args: Any, _put: Any = files.put, _name: str = name, **kw: Any) -> Any:
            waiting.append(_name)  # it has read what is kept
            if len(waiting) == 2:
                both_read.set()
            await both_read.wait()
            if _name != order[0]:
                await turn[order[0]].wait()  # the other has written and removed what it read
            return await _put(*args, **kw)

        monkeypatch.setattr(files, "put", put)

    async def ending_turn(name: str, agent: Any, task: str) -> None:
        await pipeline._keep_carried(agent, ending(f"run_{task}", task))
        turn[name].set()

    await asyncio.gather(ending_turn("here", here, "x"), ending_turn("there", there, "y"))
    for agent in (here, there):
        kept = await pipeline._carried(agent, ending("run_z", "none"))
        assert set(kept["async_tasks"]) == {"task_0", "x", "y"}
    assert len(kept_files(workers[0])) == 2  # each turn's own; the earlier one removed
    await pipeline._keep_carried(there, ending("run_w", "w"))
    [last] = kept_files(workers[1])
    assert "run_w" in last
    kept = await pipeline._carried(here, ending("run_z", "none"))
    assert set(kept["async_tasks"]) == {"task_0", "x", "y", "w"}


async def test_g_a_file_another_turn_removed_while_it_was_read_is_passed_over(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A file listed but gone by the time it is read was merged into another turn's file:
    passed over, said nowhere. One that cannot be removed is said in the log and left for the
    next turn to merge and remove."""
    agent = harness.wrap(create_deep_agent(model=Model(turns=["hi"]), tools=[]), id="desk")
    await pipeline._keep_carried(agent, ending("run_x", "task_x"))
    files = harness.files
    read = files.read

    async def gone(folder: str, name: str) -> bytes | None:
        await files.remove(folder, name)  # another turn merged it and removed it meanwhile
        return await read(folder, name)

    monkeypatch.setattr(files, "read", gone)
    assert await pipeline._carried(agent, ending("run_z", "none")) == {}
    monkeypatch.setattr(files, "read", read)
    await pipeline._keep_carried(agent, ending("run_x", "task_x"))

    async def stuck(folder: str, name: str) -> None:
        raise PermissionError("read-only bucket")

    monkeypatch.setattr(files, "remove", stuck)
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        await pipeline._keep_carried(agent, ending("run_y", "task_y"))
    assert "was not removed: PermissionError: read-only bucket" in caplog.text
    assert len(kept_files(harness)) == 2
    kept = await pipeline._carried(agent, ending("run_z", "none"))
    assert set(kept["async_tasks"]) == {"task_x", "task_y"}


# --------------------------------------------------------------------------- e. cancel
async def test_e_today_cancelling_the_turn_leaves_the_task_running_for_the_next_turn_to_find(
    harness: Harness, server: StandIn
) -> None:
    """Today: the turn cancelled after the start (its model still working), the task is not
    cancelled on the server. The next turn's history is the cancelled turn's question only —
    no id — but the list the cancelled turn ended with is carried: the task is listed, and
    cancelled by the id read there."""
    server.seconds = 60
    model = Model(turns=[START, "never said"], hold_from=1)
    agent = chat(harness, server, model)
    turn = asyncio.create_task(say(agent, "run the optimization"))
    while not server.runs or len(model.seen) < 1:  # noqa: ASYNC110 - the server's own thread
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # the start returned; the model's next call hangs
    [run_id] = agent.running
    await agent.cancel(run_id, reason="the user stopped the turn")
    cancelled = await turn
    assert cancelled.status is RunStatus.CANCELLED and cancelled.answer is None
    assert server.status(server.task_id) == "running"
    started = [("POST", "/threads"), ("POST", f"/threads/{server.task_id}/runs")]
    assert asked_to_do(server) == started  # its status read before the model call, no more
    model.hold_from = None
    model.turns = [("list_async_tasks", {}), calls("cancel_async_task"), "It is stopped."]
    await say(agent, "is the optimization running? stop it")
    assert id_in(model.seen[1]) is None  # the history names no task
    assert (
        f"task_id: {server.task_id}  agent: optimizer  status: running"
        in tool_results(model.seen[2])[-1]
    )
    assert server.status(server.task_id) == "cancelled"


# --------------------------------------------------------------------------- a busy conversation
#: How ``list_async_tasks`` names the task while it runs.
LISTED: Final = "task_id: {}  agent: optimizer  status: running"


@pytest.mark.parametrize("mode", ["queue", "interrupt", "steer"])
async def test_a_message_to_a_busy_conversation_that_has_a_background_task(
    harness: Harness, server: StandIn, mode: str
) -> None:
    """What a message does when its conversation's turn is under way (``mode``, as
    ``docs/runs.md`` says it), in a conversation that carries a task list: it waits its turn
    (``queue``), takes the turn from the working run (``interrupt``) or is given to it before
    its next model call (``steer``) — and in each case the list is the one the first turn
    left: the task is found, none is started again, and it goes on running on its server."""
    server.seconds = 60
    started, release = asyncio.Event(), asyncio.Event()

    @tool(side_effects="read")
    async def hold() -> str:
        """Wait for the record to come in."""
        started.set()
        await release.wait()
        return "held"

    model = Model(turns=[START, reports_the_id])
    target = ReAct("You help.", model, tools=[hold], subagents=[subagent(server)])
    agent = harness.wrap(target, id="bg")
    await say(agent, "run the optimization")
    listing = [("list_async_tasks", {}), says_what_the_tool_said]
    listed = LISTED.format(server.task_id)
    # the working turn calls ``hold``; what follows is whichever turn's model call comes next
    model.turns += [("hold", {}), *([] if mode == "interrupt" else ["looked it up"]), *listing]
    first = asyncio.create_task(say(agent, "Look it up."))
    await asyncio.wait_for(started.wait(), 10)
    second = asyncio.create_task(say(agent, "Which tasks are there?", mode=mode))
    await asyncio.sleep(0.3)  # it arrives while the first works
    if mode == "interrupt":
        a = await asyncio.wait_for(first, 10)
        assert a.status is RunStatus.CANCELLED  # the newer message took over
        release.set()
        b = await asyncio.wait_for(second, 10)
        # the turn that took over starts with the list the conversation carries
        assert b.status is RunStatus.SUCCESS and listed in b.answer, b.error
    else:
        assert not second.done()  # it waits, or the first run took it
        release.set()
        a, b = await asyncio.wait_for(asyncio.gather(first, second), 10)
        assert a.status is RunStatus.SUCCESS and a.answer == "looked it up", a.error
        assert b.status is RunStatus.SUCCESS and b.run_id != a.run_id, b.error
        if mode == "queue":
            assert listed in b.answer  # its own turn, after the first, with the list
        else:
            assert b.answer == a.answer  # the working run answered both
            said = [str(m.content) for m in model.seen[-1]]
            assert "Which tasks are there?" in said  # given to it before that model call
            third = await say(agent, "And now?")  # the list outlived the steered turn
            assert third.status is RunStatus.SUCCESS and listed in third.answer, third.error
    assert not model.turns  # every scripted turn was asked for: no call went astray
    assert len(server.runs) == 1 and server.status(server.task_id) == "running"


# --------------------------------------------------------------------------- f. configuration
async def test_f_what_the_server_is_sent(harness: Harness, server: StandIn) -> None:
    """The spec's ``url`` and ``headers`` are all an app configures: the SDK's default
    ``x-auth-scheme: langsmith`` is sent unless ``headers`` says otherwise, the spec's own
    headers ride along, and the request names the graph and the task's text. Of the harness's
    run, Deep Agents (0.7.30 on) sends only the graph's LangGraph thread — a ``ReAct``'s is its
    run — as the task thread's ``parent_thread_id`` and the run's ``deepagents_parent``: the
    run's id, nothing of its user, tenant or chat thread."""
    spec = subagent(server, headers={"authorization": "Bearer of-the-app"})
    model = Model(turns=[START, reports_the_id])
    agent = harness.wrap(ReAct("You help.", model, subagents=[spec]), id="bg")
    done = await say(agent, "run the optimization")
    assert done.status is RunStatus.SUCCESS, done.error
    for sent in server.headers:
        assert sent["x-auth-scheme"] == "langsmith"
        assert sent["authorization"] == "Bearer of-the-app"
        assert not [name for name in sent if "trellis" in name]
    [run_id] = [r.run_id async for r in harness.runs.iterate(agent_id="bg")]
    assert server.bodies[0] == {
        "metadata": {"subagent_name": "optimizer", "parent_thread_id": run_id}
    }
    started = server.bodies[1]
    assert set(started) == SENT and started["assistant_id"] == "optimizer"
    assert started["config"] == {"configurable": {"deepagents_parent": {"thread_id": run_id}}}
    assert started["input"] == {"messages": [{"role": "user", "content": "Find the optimum."}]}
    sent = json.dumps(server.bodies)  # an id's hex may spell a user's name: whole values
    assert f'"{USER}"' not in sent and f'"{THREAD}"' not in sent


async def test_f_today_without_a_url_the_task_cannot_start(harness: Harness) -> None:
    """Today: with no ``url`` the SDK looks for a LangGraph server in this process
    (``langgraph_api.server``), which a harness process is not: the start fails as a tool
    result the model reads, and the turn goes on."""
    spec = {"name": "optimizer", "description": "Runs the optimization.", "graph_id": "optimizer"}
    model = Model(turns=[START, says_what_the_tool_said])
    agent = harness.wrap(ReAct("You help.", model, subagents=[spec]), id="bg")
    done = await say(agent, "run the optimization")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer.startswith("Failed to launch async subagent 'optimizer':"), done.answer


# --------------------------------------------------------------------------- the AG-UI routes
async def test_through_the_ag_ui_routes_t1_t2_and_t3_work(server: StandIn) -> None:
    """The same chat over ``serve_chat``: ``POST /agui/run`` per message."""
    harness = Harness(config=Settings())
    model = Model(turns=[START, reports_the_id, "4"])
    agent = chat(harness, server, model)
    app = FastAPI()
    agent.serve_chat(app, identity=lambda request: request.headers.get("x-user", ""))
    transport = httpx.ASGITransport(app=app)

    async def post(http: httpx.AsyncClient, text: str) -> list[dict[str, Any]]:
        messages = [{"id": str(uuid.uuid4()), "role": "user", "content": text}]
        response = await http.post("/agui/run", json={"threadId": THREAD, "messages": messages})
        assert response.status_code == 200, response.text
        return [event for _, event in decode(response.text)]

    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://agui", headers={"x-user": USER}
        ) as http:
            server.hold()
            first = await post(http, "run the optimization")
            assert first[-1]["outcome"] == {"type": "success"}, first[-1]
            assert server.task_id in first[-1]["result"]
            [told] = [e for e in first if e["type"] == "TOOL_CALL_RESULT"]
            assert server.task_id in told["content"]
            second = await post(http, "what is 2+2?")
            assert second[-1]["result"] == "4"
            # both turns ended while the task could not (held): neither waited for it
            assert server.status(server.task_id) == "running"
            server.release()
            await server.finished()
            model.turns += [calls("check_async_task"), says_what_the_tool_said]
            third = await post(http, "is the optimization done?")
            assert RESULT in third[-1]["result"]
    finally:
        await harness.aclose()


#: The AG-UI event types of a turn that calls no tool, as ``main`` streamed them before the
#: task list was carried (9e60203, the same ``POST /agui/run`` on the same scripted chat).
PLAIN_TURN: Final = [
    "RUN_STARTED",
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END",
    "RUN_FINISHED",
]


async def test_through_the_ag_ui_routes_a_turn_with_no_task_streams_what_it_did_before(
    server: StandIn,
) -> None:
    """Carrying the task list adds nothing to a turn's stream: a turn that calls no tool
    streams the same event types in a thread with no task, and in the thread that carries one
    (T2, after T1 started it) — the types ``main`` streamed (:data:`PLAIN_TURN`)."""
    harness = Harness(config=Settings())
    model = Model(turns=["4", START, reports_the_id, "4"])
    agent = chat(harness, server, model)
    app = FastAPI()
    agent.serve_chat(app, identity=lambda request: request.headers.get("x-user", ""))
    transport = httpx.ASGITransport(app=app)

    async def types(http: httpx.AsyncClient, text: str, thread: str) -> list[str]:
        messages = [{"id": str(uuid.uuid4()), "role": "user", "content": text}]
        response = await http.post("/agui/run", json={"threadId": thread, "messages": messages})
        assert response.status_code == 200, response.text
        return [event["type"] for _, event in decode(response.text)]

    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://agui", headers={"x-user": USER}
        ) as http:
            assert await types(http, "what is 2+2?", "no-task") == PLAIN_TURN
            server.hold()  # it runs throughout, however long the turns take
            started = await types(http, "run the optimization", THREAD)
            assert "TOOL_CALL_RESULT" in started and server.status(server.task_id) == "running"
            assert await types(http, "what is 2+2?", THREAD) == PLAIN_TURN  # T2, list carried
    finally:
        await harness.aclose()


# --------------------------------------------------------------------------- g. Deep Agents' pull
def says_the_update(messages: list[BaseMessage]) -> str:
    """The answer of a model told how a task went: what the update says."""
    [update] = [m for m in messages if notified(m) is not None]
    return f"Done: {RESULT}" if RESULT in str(update.content) else "Not yet."


async def test_g_the_next_turn_is_told_how_the_task_went_before_its_model_call(
    memory_harness: Harness, memory_service: Any, server: StandIn
) -> None:
    """Deep Agents (0.7.27 on) reads its tasks' statuses itself when a turn starts: the task
    over, the next turn's model is given its result — a user-role message the user did not type,
    marked with langchain-core's notification metadata (source ``async_subagent``) — after the
    user's question, which stays the turn's. The harness records it as a notification, and the
    task's record the read updated is carried: the turn after is not told again."""
    server.seconds = 0.2
    model = Model(turns=[START, reports_the_id, says_the_update, "Nothing new."])
    agent = chat(memory_harness, server, model)
    await say(agent, "run the optimization")
    await server.finished()
    second = await say(agent, "anything new?")
    assert second.answer == f"Done: {RESULT}", second.error
    shown = model.seen[2]
    found = notified(shown[-1])
    assert found is not None and found["source"] == "async_subagent"
    assert found["data"]["status"] == "success" and found["data"]["task_id"] == server.task_id
    assert shown[-2].text.endswith("anything new?")  # the user's question, then the update
    kept = [m for msgs in memory_service.threads.values() for m in msgs]
    noted = [m for m in kept if "notification" in (m["custom_metadata"].get("run_record") or {})]
    assert [m["role"] for m in noted] == ["USER"]
    asked = len(server.calls)
    third = await say(agent, "and now?")
    assert third.answer == "Nothing new."
    # the earlier update is in its history, as a notification; nothing new was read or told
    assert len([m for m in model.seen[3] if notified(m) is not None]) == 1
    assert len(server.calls) == asked  # the task is over: its status is not read again


async def test_g_a_task_that_ends_while_the_turn_works_gives_the_model_another_turn(
    harness: Harness, server: StandIn
) -> None:
    """After the turn's final answer Deep Agents reads its tasks once more: one that ended
    meanwhile gives the model another turn, which tells its result."""
    server.seconds = 0.2
    model = Model(turns=[START, "Started.", says_the_update], stall=0.3)
    done = await say(chat(harness, server, model), "run the optimization")
    assert done.status is RunStatus.SUCCESS and done.answer == f"Done: {RESULT}", done.error
    assert len(model.seen) == 3


async def test_g_the_extra_turn_counts_against_max_steps(harness: Harness, server: StandIn) -> None:
    """That extra turn is a model call as any: past ``max_steps`` it is not made, and the run
    ends with the answer it had — not LangChain's "Model call limits exceeded"."""
    server.seconds = 0.2
    model = Model(turns=[START, "Started.", says_the_update], stall=0.3)
    done = await say(chat(harness, server, model, max_steps=1), "run the optimization")
    assert done.status is RunStatus.SUCCESS and done.answer == "Started.", done.error
    assert len(model.seen) == 2


# --------------------------------------------------------------------------- h. what a spec carries
def test_h_a_webhook_is_checked_as_every_url_the_harness_hands_on(server: StandIn) -> None:
    """``webhook`` is a URL the sub-agent's server is told to call with each run's outcome:
    checked as the harness checks every URL it hands on (``outbound.validate_url``: https, a
    public host), at build time."""
    ReAct(
        "You help.",
        Model(turns=[]),
        subagents=[subagent(server, webhook="https://93.184.216.34/h")],
    )
    for refused in ("http://hooks.example/h", "https://localhost/h", "https://10.0.0.1/h"):
        with pytest.raises(ConfigurationError, match="optimizer: its webhook"):
            ReAct("You help.", Model(turns=[]), subagents=[subagent(server, webhook=refused)])


@pytest.mark.filterwarnings("ignore:The feature `forked subagents` is in beta")
async def test_h_a_start_that_sends_the_conversation_asks_a_person(server: StandIn) -> None:
    """With ``mode="fork"`` the server is sent the conversation at launch, which cannot be taken
    back: ``start_async_task`` is irreversible — a person approves it, unless the catalog says
    otherwise. The others say what they say for any async sub-agent."""
    graph = ReAct("You help.", Model(turns=[]), subagents=[subagent(server, mode="fork")])
    harness = Harness(config=Settings())
    try:
        agent = harness.wrap(graph, id="bg")
        own = {s.name: s.side_effects for s in await agent.adapter.native(agent.target)}
        assert {name: own[name] for name in TOOLS} == {**RISKS, "start_async_task": "irreversible"}
    finally:
        await harness.aclose()
