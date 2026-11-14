"""An investigation, kept as tests: which adapter can "start a slow task in the background, keep
chatting in the same chat, get the result in a later turn" with what its framework itself
provides — the question ``test_react_async_subagent`` answers for ``ReAct``, asked of every
other target the harness wraps. One user (:data:`USER`), one thread (:data:`THREAD`), three
turns: T1 starts the task and ends at once, T2 is answered while it runs, T3 gets its result.
No model is called (each is scripted) and no provider's server: the remote sub-agent is the
stand-in Agent Protocol server of ``test_react_async_subagent``, the shell a local process.

Each test asserts what a user needs; one named ``today_`` pins an observation that is not what
a user needs, as the evidence of a finding. What they show:

* **A LangGraph graph** — Deep Agents' async sub-agents (``AsyncSubAgentMiddleware``: five
  tools, the task list in the graph's state). With the harness's ``middleware()`` and no
  checkpointer of its own (``create_deep_agent``, or a plain ``create_agent`` given Deep Agents'
  middleware) the graph checkpoints into its run and the conversation carries the list
  (``adapters.langgraph.CARRIED``): all three turns work. With a checkpointer of its own the
  list is in its thread: all three work, on another server too when the checkpointer is kept
  outside the process. With neither (no ``middleware()``, no checkpointer) the conversation
  carries the list too, read from the state each turn ends with: all three work — but a turn
  cancelled after its start has no end to read it from ("today"). The five tools say what
  they do in every shape.
* **An OpenAI Agents SDK agent** — the SDK has no background run, no background sub-agent and
  no ``background`` model setting; what it has is its sandbox's shell (``SandboxAgent`` with
  ``Shell``): ``exec_command`` returns after ``yield_time_ms`` with the id of a process still
  running, and ``write_stdin`` reads it later. The process lives in the sandbox session: with a
  session the app keeps (``SandboxRunConfig(session=)``) all three turns work; with a session
  per run (``SandboxRunConfig(client=)``) the process ends with T1 ("today"). The session is
  the app's, one for every user, and the SDK's shell tools are not governed ("today").
* **A function** — its code queues a child run (``agent.start`` inside the run) and answers
  with the run's id; a worker runs it; a later turn reads the run by the id its history holds.
* **An ACP agent (Kimi Code)** — each turn runs in an agent process of its own, ended with the
  turn, its scheduling turned off; the next turn resumes the session in a new process. Whatever
  the agent left running in its process is gone ("today"). No ``kimi`` is installed here: what
  Kimi's own background tasks do across that is not tested.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, Final

import pytest
from agents import Agent as OpenAIAgent
from agents import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient
from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver, PersistentDict

from tests.integration.test_acp import fake, recorded
from tests.integration.test_claude_background_session import (
    BACKGROUND,
    alive,
    live_of,
    open_sessions,
    served,
)
from tests.integration.test_claude_background_session import options as claude_options
from tests.integration.test_react_async_subagent import (
    NO_TASKS,
    RESULT,
    RISKS,
    START,
    THREAD,
    TOOLS,
    USER,
    Model,
    StandIn,
    calls,
    chat,
    reports_the_id,
    say,
    says_what_the_tool_said,
    subagent,
)
from tests.support.arm import arm
from tests.support.openai_model import ScriptedModel
from trellis import Harness, Runtime, Settings, current, middleware, tool
from trellis.contracts import RunEventType, RunOutcome, RunStatus
from trellis.harness.adapters.acp import KIMI_ENV
from trellis.harness.adapters.langgraph import run_scoped
from trellis.harness.agent import Agent
from trellis.harness.harness import LOCAL_TENANT
from trellis.harness.identity import conversation_key


@pytest.fixture
def server() -> Iterator[StandIn]:
    found = StandIn()
    found.start()
    yield found
    found.stop()


# --------------------------------------------------------------------------- A. LangGraph
Build = Callable[[Model, StandIn], Any]


def deep_agent(model: Model, server: StandIn, **more: Any) -> Any:
    """A1: Deep Agents as it is built, with the harness's middleware."""
    return create_deep_agent(model, subagents=[subagent(server)], middleware=middleware(), **more)


def plain_agent(model: Model, server: StandIn, *, ours: bool = True, **more: Any) -> Any:
    """A2: LangChain's ``create_agent`` given Deep Agents' middleware for async sub-agents
    (``ours``: after the harness's)."""
    theirs = AsyncSubAgentMiddleware(async_subagents=[subagent(server)])
    own = [*middleware(), theirs] if ours else [theirs]
    return create_agent(model, tools=[], middleware=own, **more)


#: The shapes whose task list lasts from turn to turn, and whether the conversation carries it
#: (the graph checkpoints into its run) or the graph's own thread holds it.
LASTING: Final[dict[str, tuple[Build, bool]]] = {
    "a1 deep agent, middleware()": (deep_agent, True),
    "a2 create_agent, middleware() and Deep Agents' middleware": (plain_agent, True),
    "a2 create_agent, Deep Agents' middleware alone, its own checkpointer": (
        lambda m, s: plain_agent(m, s, ours=False, checkpointer=InMemorySaver()),
        False,
    ),
}
#: The shapes with no thread of their own and no checkpoint in the run: the conversation
#: carries the list from the state each turn ends with.
BARE: Final[dict[str, Build]] = {
    "deep agent as Deep Agents builds it": lambda m, s: create_deep_agent(
        m, subagents=[subagent(s)]
    ),
    "create_agent, Deep Agents' middleware alone": lambda m, s: plain_agent(m, s, ours=False),
}


async def risks(agent: Agent) -> dict[str, Any]:
    """What the run's tool set says the five tools do."""
    own = await agent.adapter.native(agent.target)
    return {spec.name: spec.side_effects for spec in own if spec.name in TOOLS}


async def t1_and_t2(agent: Agent, server: StandIn) -> None:
    """T1: the task starts and the turn ends with its id. T2: the next message is answered
    while the task runs. The task cannot end until both did (``StandIn.hold``): neither
    waited for it. Then it ends."""
    server.hold()
    first = await say(agent, "run the optimization")
    assert first.status is RunStatus.SUCCESS, first.error
    assert server.task_id in first.answer
    second = await say(agent, "what is 2+2?")
    assert second.answer == "4", second.error
    assert server.status(server.task_id) == "running" and len(server.calls) == 2
    server.release()


async def t3_list_and_another_user(agent: Agent, server: StandIn, model: Model) -> None:
    """T3: the task over, a later turn gets its result and lists it; another user's thread of
    the same name is shown no task and refused the id."""
    model.turns += [calls("check_async_task"), says_what_the_tool_said]
    third = await say(agent, "is the optimization done?")
    assert third.status is RunStatus.SUCCESS and RESULT in third.answer, third.answer
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    listed = await say(agent, "which tasks are there?")
    assert f"task_id: {server.task_id}  agent: optimizer  status: success" in listed.answer
    asked = len(server.calls)
    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    other = await agent.run("which tasks are there?", user="bob", thread=THREAD)
    assert other.answer == NO_TASKS
    model.turns += [("check_async_task", {"task_id": server.task_id}), says_what_the_tool_said]
    other = await agent.run("how is that task?", user="bob", thread=THREAD)
    assert other.answer == f"No tracked task found for task_id: {server.task_id!r}"
    assert len(server.calls) == asked  # the server was asked nothing for bob


@pytest.mark.parametrize(("build", "carried"), LASTING.values(), ids=LASTING)
async def test_a_langgraph_graph_starts_a_task_chats_on_and_gets_its_result(
    harness: Harness, server: StandIn, build: Build, carried: bool
) -> None:
    model = Model(turns=[START, reports_the_id, "4"])
    agent = harness.wrap(build(model, server), id="bg")
    assert run_scoped(agent.target) is carried  # where the list is kept from turn to turn
    assert await risks(agent) == RISKS
    await t1_and_t2(agent, server)
    await server.finished()
    await t3_list_and_another_user(agent, server, model)


@pytest.mark.parametrize("build", BARE.values(), ids=BARE)
async def test_a_a_graph_with_no_middleware_and_no_checkpointer_keeps_the_task_list(
    harness: Harness, server: StandIn, build: Build
) -> None:
    """A graph wrapped without the harness's ``middleware()`` and compiled without a
    checkpointer has no thread and no checkpoint in its run (said when it is wrapped): its
    state ends with each turn. The conversation carries the task list all the same, read from
    the state each turn ends with: all three turns work, another user's thread has no task,
    and the five tools say what they do."""
    model = Model(turns=[START, reports_the_id, "4"])
    agent = harness.wrap(build(model, server), id="bg")
    assert getattr(agent.target, "checkpointer", None) is None
    assert await risks(agent) == RISKS
    await t1_and_t2(agent, server)
    await server.finished()
    await t3_list_and_another_user(agent, server, model)


@pytest.mark.parametrize("build", BARE.values(), ids=BARE)
async def test_a_such_a_graphs_streamed_turn_keeps_the_task_list_too(
    harness: Harness, server: StandIn, build: Build
) -> None:
    """The same graph, its turn streamed (``agent.stream``: the chat's route) on one server
    and the next turn run on another: the list is read from the state the stream ended with,
    and found there."""
    other = Harness(config=Settings())
    try:
        model = Model(turns=[START, reports_the_id, ("list_async_tasks", {}), "listed"])
        here, there = (h.wrap(build(model, server), id="bg") for h in (harness, other))
        events = [e async for e in here.stream("run the optimization", user=USER, thread=THREAD)]
        assert events[-1].outcome is RunOutcome.SUCCESS
        assert (await say(there, "which tasks are there?")).answer == "listed"
        listed = str(model.seen[-1][-1].content)
        assert f"task_id: {server.task_id}  agent: optimizer  status: running" in listed
    finally:
        await other.aclose()


async def test_a_today_such_a_graphs_cancelled_turn_leaves_its_task_out_of_the_list(
    harness: Harness, server: StandIn
) -> None:
    """Today: a graph with no checkpointer has its state read when its turn ends, and a turn
    cancelled after the start has no end to read it from: the next turn's list does not have
    the task (a graph with ``middleware()`` checkpoints each step into its run, and keeps it:
    ``test_react_async_subagent``). The task itself is cancelled with its turn."""
    server.seconds = 60
    model = Model(turns=[START, "never said"], hold_from=1)
    agent = harness.wrap(create_deep_agent(model, subagents=[subagent(server)]), id="bg")
    turn = asyncio.create_task(say(agent, "run the optimization"))
    while not server.runs or len(model.seen) < 1:  # noqa: ASYNC110 - the server's own thread
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # the start returned; the model's next call hangs
    [run_id] = agent.running
    await agent.cancel(run_id, reason="the user stopped the turn")
    assert (await turn).status is RunStatus.CANCELLED
    assert server.status(server.task_id) == "cancelled"
    model.hold_from = None
    model.turns = [("list_async_tasks", {}), says_what_the_tool_said]
    assert (await say(agent, "which tasks are there?")).answer == NO_TASKS


def kept_in(folder: Path) -> InMemorySaver:
    """A checkpointer that outlives its process, of what is installed here (no
    ``langgraph-checkpoint-sqlite`` or ``-postgres`` is): LangGraph's ``InMemorySaver`` on its
    own ``PersistentDict``\\ s — read from ``folder``'s files when made, written to them when
    closed."""
    names = iter(("storage", "writes", "blobs"))

    def kept(*default: Any) -> PersistentDict:
        found = PersistentDict(*default, filename=str(folder / next(names)))
        if Path(found.filename).exists():
            found.load()
        return found

    return InMemorySaver(factory=kept)  # type: ignore[arg-type]  # a factory of them, by file


def written_in(folder: Path) -> set[str]:
    return {kept.name for kept in folder.iterdir()}


async def test_a3_another_server_sharing_the_graphs_checkpointer_gets_the_result(
    server: StandIn, tmp_path: Path
) -> None:
    """A3: the graph's own checkpointer kept outside the process (:func:`kept_in`). One server
    — a harness, a graph and a checkpointer of its own — runs T1 and T2 and shuts down;
    another, sharing nothing with it but the checkpointer's files, runs T3: the thread it
    reads holds the conversation and the task list."""
    first, second = Harness(config=Settings()), Harness(config=Settings())
    try:
        model = Model(turns=[START, reports_the_id, "4"])
        with kept_in(tmp_path) as saver:
            here = first.wrap(deep_agent(model, server, checkpointer=saver), id="bg")
            await t1_and_t2(here, server)
        await first.aclose()
        assert written_in(tmp_path) == {"storage", "writes", "blobs"}
        await server.finished()
        model = Model(turns=[])  # the id is read in the thread the checkpointer holds
        with kept_in(tmp_path) as saver:
            there = second.wrap(deep_agent(model, server, checkpointer=saver), id="bg")
            assert not [r async for r in second.runs.iterate(agent_id="bg")]  # no run of it
            await t3_list_and_another_user(there, server, model)
    finally:
        await first.aclose()
        await second.aclose()


# --------------------------------------------------------------------------- B. OpenAI Agents
#: The slow command, and what it prints when it ends.


def slow(gate: Path) -> str:
    """The command T1 starts: it waits until the test opens ``gate`` (a file it makes), then
    prints the result and says it is done (``<gate>.done``) — so a turn that returns while the
    gate is shut cannot have waited for it, whatever the time each took."""
    return f"until [ -e {gate} ]; do sleep 0.05; done; echo '{RESULT}'; touch {gate}.done"


def is_open(gate: Path, *, done: bool = False) -> bool:
    """Whether the test opened ``gate`` (``done``: and the command is done)."""
    return (gate.with_name(f"{gate.name}.done") if done else gate).exists()


def open_gate(gate: Path) -> None:
    gate.touch()


async def opened(gate: Path) -> None:
    """The gate opened, and the command done with what it prints."""
    open_gate(gate)
    async with asyncio.timeout(30):
        while not is_open(gate, done=True):  # noqa: ASYNC110 - another process's file
            await asyncio.sleep(0.02)


#: How the SDK's ``exec_command`` names a process still running when it returns.
_PROCESS: Final = re.compile(r"session (?:ID )?(\d+)")


class Shelled(ScriptedModel):
    """A scripted model whose turn may be a function of the input it is sent: it reads a
    process id where a model would, in a tool's result or in the conversation."""

    def _next(self, input: Any, system: str | None) -> list[Any]:
        if self.turns and callable(self.turns[0]):
            self.turns[0] = self.turns[0](input)
        return super()._next(input, system)


def process_in(input: Any) -> int:
    """The last process id in what the model is shown."""
    return int(_PROCESS.findall(str(input))[-1])


def reports_the_process(input: Any) -> str:
    return f"The optimization is running in the background, in session {process_in(input)}"


def polls(input: Any) -> Any:
    """``write_stdin`` with nothing to write: the SDK's way to read a running process."""
    return ("write_stdin", {"session_id": process_in(input), "chars": "", "yield_time_ms": 500})


def says_what_the_shell_said(input: Any) -> str:
    return str(input[-1]["output"])


def shell_chat(harness: Harness, model: Shelled, box: SandboxRunConfig) -> Agent:
    target = SandboxAgent(name="box", instructions="Work.", model=model, capabilities=[Shell()])
    return harness.wrap(target, id="bg", framework_options={"run_config": RunConfig(sandbox=box)})


def shell_turns(gate: Path) -> list[Any]:
    """T1: the command started (:func:`slow`), to return after a tenth of a second whatever
    it does."""
    command = {"cmd": slow(gate), "yield_time_ms": 100}
    return [("exec_command", command), reports_the_process, "4"]


@pytest.fixture
async def session() -> AsyncIterator[Any]:
    """A sandbox session the app keeps: made before the chat, closed after it."""
    client = UnixLocalSandboxClient()
    found = await client.create()
    await found.start()
    yield found
    await found.shutdown()
    await client.delete(found)


async def shell_t1_and_t2(agent: Agent, model: Shelled, gate: Path) -> int:
    """T1 and T2 with the SDK's shell, both ended while the command's gate is still shut (so
    neither waited for it: it cannot end before the gate opens); the id of the process T1
    left running."""
    first = await say(agent, "run the optimization")
    assert first.status is RunStatus.SUCCESS, first.error
    started = str(model.inputs[1][-1]["output"])  # what exec_command told the model
    assert "Process running with session ID" in started and RESULT not in started
    second = await say(agent, "what is 2+2?")
    assert second.answer == "4", second.error
    assert not is_open(gate)  # both turns ended with the command still waiting
    return process_in(first.answer)


async def test_b_an_openai_sandbox_agents_shell_runs_a_command_across_the_turns_of_a_chat(
    harness: Harness, session: Any, tmp_path: Path
) -> None:
    """The SDK's own provision, through the harness: with a sandbox session the app keeps, the
    process ``exec_command`` left running in T1 is still the session's in T3, where
    ``write_stdin`` reads what it printed and how it ended — the id read from the
    conversation's history (the first turn's answer; memory off, no tool result is in it)."""
    gate = tmp_path / "gate"
    model = Shelled(shell_turns(gate))
    agent = shell_chat(harness, model, SandboxRunConfig(session=session))
    await shell_t1_and_t2(agent, model, gate)
    await opened(gate)
    model.turns += [polls, says_what_the_shell_said]
    third = await say(agent, "is the optimization done?")
    assert third.status is RunStatus.SUCCESS, third.error
    assert RESULT in third.answer and "Process exited with code 0" in third.answer
    assert {"exec_command", "write_stdin"} <= set(model.tools[0])  # the SDK's two tools


async def test_b_today_with_a_sandbox_session_per_run_the_command_ends_with_its_turn(
    harness: Harness, tmp_path: Path
) -> None:
    """Today: given a sandbox client and no session (as the docs' example wraps a
    ``SandboxAgent``), the SDK makes a session for each ``Runner.run`` and ends it, its
    processes too, when the run ends: T1 and T2 work, and T3 is told there is no such
    process."""
    gate = tmp_path / "gate"
    model = Shelled(shell_turns(gate))
    agent = shell_chat(harness, model, SandboxRunConfig(client=UnixLocalSandboxClient()))
    process = await shell_t1_and_t2(agent, model, gate)
    open_gate(gate)  # nothing is left to open it for: the process ended with its session
    model.turns += [polls, says_what_the_shell_said]
    third = await say(agent, "is the optimization done?")
    assert f"write_stdin failed: PTY session not found: {process}" in third.answer
    assert RESULT not in third.answer


async def test_b_today_the_kept_session_is_every_users_and_its_shell_is_not_governed(
    harness: Harness, session: Any, tmp_path: Path
) -> None:
    """Today: the session is the app's — a run option of the agent, one for all its runs — so
    another user's turn that names the process reads it. And the SDK's shell tools are made
    inside ``Runner.run`` (a ``SandboxAgent``'s capabilities): they are not of the run's tool
    set, their calls pass no governance (nothing says what ``exec_command`` does, and nobody is
    asked) and are not on the run's stream."""
    gate = tmp_path / "gate"
    model = Shelled(shell_turns(gate)[:2])
    agent = shell_chat(harness, model, SandboxRunConfig(session=session))
    assert await agent.adapter.native(agent.target) == []
    events = [e async for e in agent.stream("run the optimization", user=USER, thread=THREAD)]
    assert events[-1].type is RunEventType.RUN_FINISHED
    assert events[-1].outcome is RunOutcome.SUCCESS  # it ran: nobody was asked
    assert not [e for e in events if e.type is RunEventType.TOOL_CALL_START]
    process = process_in(model.inputs[1])
    await opened(gate)
    model.turns += [
        ("write_stdin", {"session_id": process, "chars": "", "yield_time_ms": 500}),
        says_what_the_shell_said,
    ]
    other = await agent.run("how is that process?", user="bob", thread=THREAD)
    assert RESULT in other.answer


# --------------------------------------------------------------------------- D. a function
_RUN: Final = re.compile(r"run_\w+")


async def test_d_a_function_queues_a_child_run_and_a_later_turn_reads_its_result(
    harness: Harness,
) -> None:
    """A function has no framework: its background task is a run of another agent it queues
    (``agent.start`` inside the run: a child of the turn, in its thread, worked by a worker in
    its own time). T1 answers with the run's id at once; T2 is answered while the worker runs
    it; T3 reads the run from the run store by the id its history holds (memory off: the
    earlier turns' questions and answers) — and another user's thread has no such history.
    The run works until the test lets it end (``finish``): T1 and T2 end while it cannot."""
    started, finish = asyncio.Event(), asyncio.Event()

    async def optimize(input: str, agent: Runtime) -> str:
        started.set()
        await finish.wait()
        return RESULT

    optimizer = harness.wrap(optimize, id="optimizer")

    async def chat(input: str, agent: Runtime) -> str:
        if input == "run the optimization":
            queued = await optimizer.start("Find the optimum.", user=agent.user)
            return f"The optimization is running in the background. Run id: {queued.id}"
        if input == "what is 2+2?":
            return "4"
        said = " ".join(str(s.content) for s in agent.history or [])
        ids = _RUN.findall(said)
        if not ids:
            return "I see no run id in this conversation."
        record = await harness.runs.get(ids[-1], tenant=agent.tenant)
        assert record is not None
        return f"{record.status.value}: {record.output}"

    agent = harness.wrap(chat, id="bg")
    first = await say(agent, "run the optimization")  # nothing works the run yet: not waited for
    assert first.status is RunStatus.SUCCESS, first.error
    [run_id] = _RUN.findall(first.answer)
    working = asyncio.create_task(harness.worker([optimizer]).run_once())
    await started.wait()  # the worker has the run, which waits for the test
    second = await say(agent, "what is 2+2?")
    assert second.answer == "4"  # answered while the run could not end
    assert (await say(agent, "is the optimization done?")).answer == "RUNNING: None"
    finish.set()
    assert await working
    third = await say(agent, "is the optimization done?")
    assert third.answer == f"SUCCESS: {RESULT}"
    child = await harness.runs.get(run_id)
    assert child is not None
    assert (child.parent_run_id, child.thread_id) == (first.run_id, THREAD)  # the turn's child
    other = await agent.run("is the optimization done?", user="bob", thread=THREAD)
    assert other.answer == "I see no run id in this conversation."


# --------------------------------------------------------------------------- C. an ACP agent
async def test_c_today_an_acp_agents_process_ends_with_each_turn_and_its_session_goes_on(
    harness: Harness, tmp_path: Path
) -> None:
    """Today: what the harness does around an ACP agent (Kimi Code) from turn to turn, with
    the scripted agent — no ``kimi`` is installed here. Each turn of a chat runs in an agent
    process of its own, started with Kimi's scheduling off and ended with the turn; the next
    turn starts another and resumes the session the last one ended in. So the conversation
    goes on, and nothing the agent left running in its process does."""
    script = [{"text": "started"}, {"stop": "end_turn"}, {"text": "4"}]
    agent = harness.wrap(fake(tmp_path, script), id="bg")
    first = await say(agent, "run the optimization")
    second = await say(agent, "what is 2+2?")
    assert (first.answer, second.answer) == ("started", "4"), second.error
    record = recorded(tmp_path)
    assert len(record["started"]) == 2  # a process per turn
    assert [p["env"]["KIMI_DISABLE_CRON"] for p in record["started"]] == ["1", "1"]
    assert KIMI_ENV == {"KIMI_DISABLE_CRON": "1"}
    new, resumed = record["sessions"]
    assert (new["how"], resumed["how"]) == ("new", "resume") and new["id"] == resumed["id"]
    assert agent.adapter.steers(agent.target) is False  # and no message reaches a turn under way


# --------------------------------------------------------------------------- E. side by side
#: The conversation every agent below is asked in: the same user's thread of the same name.
KEY: Final = conversation_key(LOCAL_TENANT, USER, THREAD)
LISTED: Final = "task_id: {}  agent: optimizer  status: running"


async def test_e_the_task_list_a_deep_agent_carries_is_handed_to_no_other_adapter(
    harness: Harness, server: StandIn, tmp_path: Path
) -> None:
    """The list is kept by the agent whose graph left it (``Agent.results``), under the
    conversation's key — which names no agent. A Claude Agent SDK agent, an OpenAI Agents SDK
    agent and a function on the same harness, asked by the same user in a thread of the same
    name, start their turns with no carried state (``Runtime.carried``), keep none, and
    answer; the Deep Agents graph's own next turn still finds its task."""
    server.seconds = 60
    model = Model(turns=[START, reports_the_id])
    deep = chat(harness, server, model)
    await say(deep, "run the optimization")
    assert list(deep.results.state(KEY)) == ["async_tasks"]  # what its conversation carries
    started: dict[str, list[dict[str, Any]]] = {"claude": [], "openai": [], "function": []}

    def carried_by(name: str) -> Any:
        @tool(side_effects="read")
        def carried() -> str:
            """What this run's conversation carried into it."""
            runtime = current()
            assert runtime is not None
            started[name].append(dict(runtime.carried))
            return "looked"

        return carried

    async def function(input: str, agent: Runtime) -> str:
        started["function"].append(dict(agent.carried))
        return "looked"

    turn = [{"tool": "carried"}, {"text": "{last}"}]
    asked = ScriptedModel([("carried", {}), "looked", ("carried", {}), "looked"])
    others = {
        "claude": arm(
            harness.wrap(claude_options(tmp_path, [turn, turn]), id="claude"),
            [carried_by("claude")],
        ),
        "openai": arm(
            harness.wrap(OpenAIAgent(name="w", instructions="Work.", model=asked), id="openai"),
            [carried_by("openai")],
        ),
        "function": harness.wrap(function, id="function"),
    }
    for name, other in others.items():
        for question in ("is the optimization done?", "and now?"):  # a first turn, a later one
            done = await say(other, question)
            assert done.status is RunStatus.SUCCESS and done.answer == "looked", (name, done.error)
        assert other.results.state(KEY) == {}, name  # and it kept none for its next turn
    assert started == {"claude": [{}, {}], "openai": [{}, {}], "function": [{}, {}]}
    assert len({turn["pid"] for turn in served(tmp_path)}) == 1  # Claude's turns: one session
    assert len(server.calls) == 2  # none of them reached the sub-agent's server

    model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
    mine = await say(deep, "which tasks are there?")
    assert LISTED.format(server.task_id) in mine.answer  # the graph's own list, as it left it


async def test_e_an_async_sub_agent_and_a_live_claude_session_side_by_side_on_one_harness(
    server: StandIn, tmp_path: Path
) -> None:
    """Both of a day's features in one process: a ``ReAct`` agent whose async sub-agent works
    on its server and a Claude Agent SDK agent whose live session runs a background subagent,
    asked in turn by the same user in threads of the same name. Each keeps its own: the graph
    its task list (and no CLI process), Claude's session its background task and its own
    prompts (and no task list). Closing the harness ends Claude's process; the sub-agent's
    task is its server's, and goes on."""
    server.seconds = 60
    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "you asked: {heard}"}]]
    async with Harness(config=Settings()) as harness:
        model = Model(turns=[START, reports_the_id])
        deep = chat(harness, server, model)
        claude = harness.wrap(claude_options(tmp_path, scripts), id="claude")
        first = await say(deep, "run the optimization")
        began = await say(claude, "research the market")
        assert first.status is RunStatus.SUCCESS and server.task_id in first.answer, first.error
        assert began.status is RunStatus.SUCCESS and began.answer == "started it", began.error
        live = live_of(claude, USER, THREAD)
        assert live is not None and live.working == {"t1": "research the market"}
        assert list(deep.results.state(KEY)) == ["async_tasks"] and not claude.results.state(KEY)
        assert getattr(deep.adapter, "live", None) is None  # a graph keeps no CLI process

        model.turns += [("list_async_tasks", {}), says_what_the_tool_said]
        listed = await say(deep, "which tasks are there?")
        again = await say(claude, "and now?")
        assert LISTED.format(server.task_id) in listed.answer
        # Claude's session heard its own two prompts, none of the graph's turns
        assert again.answer == "you asked: research the market | and now?", again.error
        one, two = served(tmp_path)
        assert two["pid"] == one["pid"] and alive(one["pid"]) and open_sessions(claude) == 1
        assert live_of(claude, USER, THREAD) is live and list(live.working) == ["t1"]
        assert not claude.results.state(KEY) and len(server.runs) == 1
    assert open_sessions(claude) == 0 and live.closed
    assert not alive(one["pid"])  # no process is left behind
    assert server.status(server.task_id) == "running"  # the remote task is not the harness's
