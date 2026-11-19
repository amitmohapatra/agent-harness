"""The OpenAI Agents SDK's own features through the harness, as the SDK does them: the real
``Runner`` with a scripted ``Model`` (no model is paid for)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest
from agents import (
    Agent,
    ModelResponse,
    RunConfig,
    Runner,
    SQLiteSession,
    Tool,
    ToolOutputImage,
    ToolsToFinalOutputResult,
    Usage,
    function_tool,
)
from agents.memory.session import SessionABC
from agents.models.multi_provider import MultiProvider
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider
from pydantic import BaseModel

from tests.support.arm import declaring
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Harness
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness import adapters
from trellis.harness.adapters.openai_agents import (
    OpenAIAgentsAdapter,
    ServerThread,
    _Continue,
    governed_capability,
)
from trellis.harness.gateways import Bifrost
from trellis.harness.governance.catalog import Rule
from trellis.harness.tools import bridge

GOLDEN = Path(__file__).parent / "golden"


def said_by_user(items: Any) -> list[Any]:
    return [i.get("content") for i in items if isinstance(i, dict) and i.get("role") == "user"]


# --------------------------------------------------------------------------- sessions
async def test_a_session_given_on_wrap_is_each_conversations_own(
    harness: Harness, tmp_path: Path
) -> None:
    """A session on ``h.wrap`` is every run's: the harness keeps each conversation (tenant,
    user, thread) under its own ``session_id`` in it — the same class, the same database — so
    bob never reads alice's turns, and alice's thread goes on with its own."""
    own = SQLiteSession("shared", tmp_path / "s.db")
    model = ScriptedModel(["noted", "nothing", "42"])
    agent = harness.wrap(Agent(name="a", model=model), id="s", framework_options={"session": own})
    await agent.run("my secret is 42", user="alice", thread="t1")
    await agent.run("what do you know?", user="bob", thread="t1")
    await agent.run("what is my secret?", user="alice", thread="t1")
    assert said_by_user(model.inputs[1]) == ["what do you know?"]
    assert said_by_user(model.inputs[2]) == ["my secret is 42", "what is my secret?"]
    assert own.session_id == "shared" and await own.get_items() == []  # yours, unchanged
    own.close()


class Keyed(SessionABC):
    """A session that makes its storage key from its id when it is made (as a Redis or a Dapr
    session does)."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._key = f"app:{session_id}"

    async def get_items(self, limit: int | None = None) -> list[Any]:
        return []

    async def add_items(self, items: list[Any]) -> None:
        return None

    async def pop_item(self) -> Any:
        return None

    async def clear_session(self) -> None:
        return None


class Hosted(Keyed):
    """A session whose id is not a plain attribute (a server's conversation)."""

    def __init__(self) -> None:
        self._key = "x"

    @property  # type: ignore[override]
    def session_id(self) -> str:
        return "conv_1"


def test_a_session_that_cannot_be_each_conversations_is_refused_where_it_is_wrapped(
    harness: Harness,
) -> None:
    with pytest.raises(ConfigurationError, match="its _key was made from its session_id"):
        harness.wrap(Agent(name="a"), id="k", framework_options={"session": Keyed("x")})
    with pytest.raises(ConfigurationError, match="not a plain attribute"):
        harness.wrap(Agent(name="a"), id="h", framework_options={"session": Hosted()})
    # given per run, a session is the caller's own for that conversation: kept as it is
    harness.wrap(Agent(name="a"), id="ok")


def test_a_session_wrapping_another_is_made_each_conversations_with_it() -> None:
    from trellis.harness.adapters.openai_agents import per_conversation

    class Wrapping(Keyed):
        def __init__(self, inner: Any) -> None:
            self.session_id = "outer"
            self.underlying = inner

    inner = SQLiteSession("inner")
    scoped = per_conversation(Wrapping(inner), "conv_a")
    assert scoped.session_id == "outer:conv_a"
    assert scoped.underlying.session_id == "inner:conv_a" and inner.session_id == "inner"


# --------------------------------------------------------------------------- run_config (P1)
@pytest.mark.host_execution
async def test_a_run_config_given_as_a_dict_is_the_sdks(harness: Harness) -> None:
    """``run_config`` as a dict, as ``Runner.run`` takes it — and as a queued run keeps it: made
    the SDK's ``RunConfig``; its own ``tool_not_found_behavior`` is kept (a made-up tool ends
    the run, as the SDK does), and a field the SDK does not have is the SDK's error."""
    model = ScriptedModel(["hi", ("nope", {})])
    options = {"run_config": {"workflow_name": "w", "tool_not_found_behavior": "raise_error"}}
    agent = harness.wrap(Agent(name="d", model=model), id="d", framework_options=options)
    assert (await agent.run("q", user="u")).answer == "hi"
    made_up = await agent.run("q", user="u")
    assert made_up.status is RunStatus.ERROR and made_up.error is not None
    assert "Tool nope not found" in made_up.error.message
    bad = await agent.run("q", user="u", framework_options={"run_config": {"no_such": 1}})
    assert bad.error is not None and "Unknown run_config settings: no_such" in bad.error.message


# --------------------------------------------------------------------------- tool outputs
class Ship(BaseModel):
    order: str
    state: str


@function_tool
def look(order: str) -> Ship:
    """Look an order up."""
    return Ship(order=order, state="shipped")


@function_tool
def picture(order: str) -> ToolOutputImage:
    """A picture of the parcel."""
    return ToolOutputImage(image_url="https://example.com/a.png")


async def test_a_tools_own_result_reaches_the_model_and_the_sdk_as_it_returned_it(
    harness: Harness,
) -> None:
    """An agent's own tool's result is what it returned, as the SDK takes it: an image the
    model sees as an image, a typed result its ``tool_use_behavior`` is handed as the object —
    and the call's record keeps its readable form."""
    seen: list[Any] = []

    def first(context: Any, results: Any) -> ToolsToFinalOutputResult:
        seen.append(results[0].output)
        return ToolsToFinalOutputResult(is_final_output=True, final_output=results[0].output)

    h = declaring(harness, look="read", picture="read")
    model = ScriptedModel([("look", {"order": "o1"})])
    typed = Agent(name="t", model=model, tools=[look], tool_use_behavior=first)
    events = [e async for e in h.wrap(typed, id="typed").stream("q", user="u")]
    assert seen == [Ship(order="o1", state="shipped")]
    assert events[-1].type is RunEventType.RUN_FINISHED
    ended = [e for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert ended and "shipped" in json.dumps(ended[0].data, default=str)  # recorded readable

    model = ScriptedModel([("picture", {"order": "o1"}), "a box"])
    shown = await h.wrap(Agent(name="p", model=model, tools=[picture]), id="pic").run("q", user="u")
    assert shown.answer == "a box"
    assert model.outputs() == [[{"type": "input_image", "image_url": "https://example.com/a.png"}]]

    model = ScriptedModel([("look", {"order": "o2"})])
    stops = Agent(name="s", model=model, tools=[look], tool_use_behavior="stop_on_first_tool")
    stopped = await h.wrap(stops, id="stop").run("q", user="u")
    assert stopped.answer == "order='o2' state='shipped'"  # the SDK's final output of it


# --------------------------------------------------------------------------- server threads
class ChatScripted(OpenAIChatCompletionsModel):
    """The SDK's chat completions model (which ignores a server-side thread), scripted."""

    def __init__(self, turns: list[Any]) -> None:
        self.scripted = ScriptedModel(turns)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return await self.scripted.get_response(*args, **kwargs)


def test_who_keeps_a_server_thread_is_the_models_route() -> None:
    """``conversation_id`` and the like are a durable thread only where the run's model keeps
    it: the SDK's chat completions model ignores them, and so does a model the gateway
    answered it serves on chat completions only."""
    named = Agent(name="a", model="gpt-x")
    adapter = OpenAIAgentsAdapter(named)
    adapter.gateway = Bifrost("http://gw.test/v1", None)
    options = {"conversation_id": "conv_1"}
    assert adapter.thread_store(named, options) == ServerThread("conversation_id")
    adapter.gateway.served("gpt-x", responses=False)  # it answered: chat completions only
    assert adapter.thread_store(named, options) is None  # no thread kept there
    chat = Agent(name="c", model=ChatScripted(["x"]))
    assert adapter.thread_store(chat, options) is None
    adapter.gateway = None  # no gateway route: the SDK's own provider keeps it
    assert adapter.thread_store(named, options) == ServerThread("conversation_id")
    theirs = RunConfig(model_provider=MultiProvider(openai_use_responses=False))
    on_chat = {**options, "run_config": theirs}  # its OpenAI provider on chat completions
    assert adapter.thread_store(named, on_chat) is None
    own = {**options, "run_config": RunConfig(model_provider=OpenAIProvider(api_key="k"))}
    assert adapter.thread_store(named, own) == ServerThread("conversation_id")  # at its word


async def test_a_server_thread_the_model_ignores_leaves_memory_the_history(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ChatScripted(["one"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="ignored")
    await agent.run("hi", user="u1", framework_options={"conversation_id": "conv_1"})
    assert memory_service.named("context")[-1].body.get("window") is True


# --------------------------------------------------------------------------- agent tools
async def test_an_agent_tools_own_run_config_gets_the_runs_day_and_not_its_steering(
    harness: Harness,
) -> None:
    """An agent tool with a ``run_config`` of its own runs its child with it and what the
    harness gives every model call (the day line, a made-up tool's error), as one without
    does; the run's own ``call_model_input_filter`` steering is the calling run's only."""
    inner = ScriptedModel(["inner"], ids="i")
    own = ScriptedModel([("made_up", {}), "own"], ids="o")
    outer = ScriptedModel([("ask", {"input": "x"}), ("ask2", {"input": "y"}), "done"])
    tools: list[Tool] = [
        Agent(name="inner", instructions="INNER", model=inner).as_tool("ask", "a"),
        Agent(name="own", instructions="OWN", model=own).as_tool(
            "ask2", "b", run_config=RunConfig(workflow_name="mine")
        ),
    ]
    target = Agent(name="outer", instructions="OUTER", model=outer, tools=tools)
    result = (
        await declaring(harness, ask="read", ask2="read").wrap(target, id="n").run("go", user="u")
    )
    assert result.status is RunStatus.SUCCESS, result.error
    day = outer.system[0].split("\n\n")[-1]  # type: ignore[union-attr]
    assert day.startswith("It is now")
    assert inner.system[0] == f"INNER\n\n{day}" and own.system[0] == f"OWN\n\n{day}"
    assert "Tool 'made_up' not found." in json.dumps(own.inputs[-1], default=str)


async def test_a_message_steered_into_the_run_is_not_given_to_an_agent_tools_child(
    harness: Harness,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    @function_tool
    async def hold() -> str:
        """Wait."""
        started.set()
        await release.wait()
        return "held"

    child = ScriptedModel([("hold", {}), "child done"], ids="c")
    parent = ScriptedModel([("ask", {"input": "x"}), "all done"])
    helper = Agent(name="helper", model=child, tools=[hold]).as_tool("ask", "a")
    target = Agent(name="outer", model=parent, tools=[helper])
    agent = declaring(harness, ask="read", hold="read").wrap(target, id="steer")
    first = asyncio.create_task(agent.run("Look it up.", user="ada", thread="t"))
    await asyncio.wait_for(started.wait(), 30)
    second = asyncio.create_task(agent.run("Also say hello.", user="ada", thread="t", mode="steer"))
    await asyncio.sleep(0.3)
    release.set()
    a, b = await asyncio.wait_for(asyncio.gather(first, second), 30)
    assert a.status is RunStatus.SUCCESS and b.answer == a.answer
    assert "Also say hello." not in json.dumps(child.inputs, default=str)
    assert "Also say hello." in said_by_user(parent.inputs[-1])


async def test_an_agent_tools_child_model_calls_are_watched_and_counted_once(
    harness: Harness,
) -> None:
    """A model call of an agent tool's child run is watched for a stall like the run's own,
    and what the child spends is counted toward the run's budget once (by its helper)."""
    from trellis import current

    watched: list[list[str]] = []

    class Watching(ScriptedModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            runtime = current()
            assert runtime is not None
            watched.append(sorted(runtime._watched))
            return await super().get_response(*args, **kwargs)

    child = Watching(["child done"], usage=Usage(input_tokens=5, output_tokens=5), ids="c")
    parent = ScriptedModel([("ask", {"input": "x"}), "done"])
    helper = Agent(name="helper", model=child).as_tool("ask", "a")
    target = Agent(name="outer", model=parent, tools=[helper])
    events = [
        e async for e in declaring(harness, ask="read").wrap(target, id="w").stream("go", user="u")
    ]
    assert watched == [["openai_agents:helper"]]
    assert events[-1].data["spent"]["tokens"] == 10  # the child's reply, once


# --------------------------------------------------------------------------- detection
def test_an_agent_subclass_of_your_own_is_an_openai_agent_and_realtime_is_refused() -> None:
    class Mine(Agent[Any]):
        pass

    assert isinstance(adapters.detect(Mine(name="x")), OpenAIAgentsAdapter)
    from agents.realtime import RealtimeAgent

    with pytest.raises(ConfigurationError, match="realtime session of its own"):
        adapters.detect(RealtimeAgent(name="r"))


# --------------------------------------------------------------------------- approvals
async def test_an_sdk_approval_allowed_for_the_conversation_sticks(harness: Harness) -> None:
    """The SDK's own ``needs_approval`` tool, approved "for this conversation"
    (``remember="run"``): its later calls run unasked — in the run (the SDK's
    ``always_approve``) and in the conversation's later runs (the harness's one question)."""
    ran: list[str] = []

    @function_tool(needs_approval=True)
    def refund(order: str) -> str:
        """Refund an order."""
        ran.append(order)
        return f"refunded {order}"

    model = ScriptedModel(
        [
            ("refund", {"order": "o1"}),
            ("refund", {"order": "o2"}),
            "both",
            ("refund", {"order": "o3"}),
            "three",
            ("refund", {"order": "o4"}),
        ]
    )
    agent = declaring(harness, refund="write").wrap(
        Agent(name="r", model=model, tools=[refund]), id="sticky"
    )
    paused = await agent.run("refund o1 and o2", user="u", thread="t")
    assert paused.interrupt is not None and ran == []
    done = await agent.resume(
        paused.interrupt.interrupt_id, "approve", reviewer="u", remember="run"
    )
    assert done.status is RunStatus.SUCCESS and ran == ["o1", "o2"], done.interrupt
    later = await agent.run("refund o3", user="u", thread="t")
    assert later.status is RunStatus.SUCCESS and ran == ["o1", "o2", "o3"], later.interrupt
    other = await agent.run("refund o3", user="u", thread="t2")  # another conversation asks
    assert other.status is RunStatus.PAUSED


# --------------------------------------------------------------------------- MCP retries
async def test_an_mcp_server_that_retries_owns_its_tries(harness: Harness) -> None:
    """A read the server tries again itself (``max_retry_attempts``) is tried once by the
    harness — the library owns the tries when it makes them; without, the harness's read
    retries apply."""
    import httpx
    from agents.mcp import MCPServer
    from mcp.types import Tool as McpTool

    class Flaky(MCPServer):
        def __init__(self, retries: int) -> None:
            super().__init__()
            self.max_retry_attempts = retries
            self.calls = 0

        @property
        def name(self) -> str:
            return "flaky"

        async def connect(self) -> None: ...

        async def cleanup(self) -> None: ...

        async def list_tools(self, run_context: Any = None, agent: Any = None) -> Any:
            return [McpTool(name="find", input_schema={"type": "object", "properties": {}})]

        async def call_tool(self, tool_name: str, arguments: Any, meta: Any = None) -> Any:
            self.calls += 1
            raise httpx.ConnectError("blip")

        async def list_prompts(self) -> Any: ...

        async def get_prompt(self, name: str, arguments: Any = None) -> Any: ...

    for retries, tried in ((2, 1), (0, 3)):
        server = Flaky(retries)
        model = ScriptedModel([("find", {}), "gave up"])
        target = Agent(name="m", model=model, mcp_servers=[server])
        result = (
            await declaring(harness, find="read")
            .wrap(target, id=f"mcp{retries}")
            .run("find", user="u")
        )
        assert result.answer == "gave up" and server.calls == tried, (retries, server.calls)


# --------------------------------------------------------------------------- upgrade
async def test_a_run_state_saved_by_sdk_0_22_resumes_on_this_release() -> None:
    """A run paused on an SDK approval under openai-agents 0.22.3 (its ``RunState`` schema
    1.17, kept as the run's checkpoint) continues on this release as the harness resumes it."""
    saved = json.loads((GOLDEN / "openai_run_state_0_22_3.json").read_text())
    assert saved["$schemaVersion"] == "1.17"
    ran: list[str] = []

    @function_tool(needs_approval=True)
    def ship(order: str) -> str:
        """Ship an order."""
        ran.append(order)
        return f"shipped {order}"

    agent = Agent(
        name="shipper", instructions="Ship.", model=ScriptedModel(["shipped"]), tools=[ship]
    )
    state = await OpenAIAgentsAdapter._input(agent, _Continue(saved, "call_1", approve=True))
    result = await Runner.run(agent, state)
    assert result.final_output == "shipped" and ran == ["o1"]


# --------------------------------------------------------------------------- sandbox memory
async def test_a_sandbox_memorys_own_model_calls_go_through_the_run(harness: Harness) -> None:
    """A sandbox ``Memory``'s extraction and consolidation models, named, are made through the
    run's model route (its provider), their usage counted toward the run's budget; the copy
    keeps the SDK's comparison of memory configs (the same names are the same models)."""
    from agents.models.interface import ModelProvider
    from agents.sandbox.capabilities.memory import Memory
    from agents.sandbox.config import MemoryGenerateConfig

    phase = ScriptedModel(["summary"], usage=Usage(input_tokens=7, output_tokens=3))

    class Provider(ModelProvider):
        def get_model(self, model_name: str | None) -> Any:
            return phase if model_name == "phase-model" else ScriptedModel([])

    memory = Memory(generate=MemoryGenerateConfig(phase_one_model="phase-model"))

    @function_tool
    async def summarise() -> str:
        """Summarise."""
        made, again = governed_capability(memory).generate, governed_capability(memory).generate
        assert made == again  # the same names are the same models, as the SDK compares them
        assert hash(made.phase_one_model) == hash(again.phase_one_model)
        reply = await made.phase_one_model.get_response(None, "x", None, [], None, [], None)
        return str(reply.output[0].content[0].text)

    model = ScriptedModel([("summarise", {}), "done"])
    agent = declaring(harness, summarise="read").wrap(
        Agent(name="m", model=model, tools=[summarise]),
        id="mem",
        framework_options={"run_config": RunConfig(model_provider=Provider())},
    )
    events = [e async for e in agent.stream("go", user="u")]
    assert phase.inputs == ["x"]
    assert events[-1].data["spent"]["tokens"] >= 10  # the phase's 7 + 3 counted
    assert memory.generate is not None and memory.generate.phase_one_model == "phase-model"


def test_a_sandbox_memory_that_generates_is_said_where_it_is_wrapped(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """Its consolidation agent's file writes are the SDK's own, outside governance: said once,
    at ``h.wrap``, with what to do; a ``Memory`` that only reads says nothing."""
    from agents.sandbox import SandboxAgent
    from agents.sandbox.capabilities import Shell
    from agents.sandbox.capabilities.memory import Memory

    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        reads = [Shell(), Memory(generate=None)]
        harness.wrap(SandboxAgent(name="reads", capabilities=reads), id="reads")
        assert "generates memories" not in caplog.text
        harness.wrap(SandboxAgent(name="learns", capabilities=[Shell(), Memory()]), id="learns")
    assert "'learns' has a Memory that generates memories" in caplog.text
    assert "Memory(generate=None)" in caplog.text


@pytest.mark.host_execution
async def test_a_run_with_tool_hooks_governs_the_sandboxs_tools_through_them(
    harness: Harness,
) -> None:
    """With tool hooks, a sandbox tool's call is the hooks' first (it is not held at the
    SDK's gate): a ``before_tool`` hook sees ``exec_command`` and may deny it."""
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    from trellis import Deny, Hooks
    from trellis.contracts import ToolCall

    seen: list[str] = []

    class NoShell(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | None:
            seen.append(call.tool)
            return Deny("no shell here")

    model = ScriptedModel([("exec_command", {"cmd": "echo hi"}), "gave up"])
    target = SandboxAgent(name="box", instructions="Work.", model=model, capabilities=[Shell()])
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    agent = harness.wrap(
        target, id="hooked", hooks=[NoShell()], framework_options={"run_config": config}
    )
    result = await agent.run("go", user="u")
    assert result.answer == "gave up" and seen == ["exec_command"]
    assert "no shell here" in str(model.outputs()[-1])


@pytest.mark.host_execution
async def test_a_sandbox_shell_runs_unasked_and_announced_as_every_frameworks_shell(
    harness: Harness,
) -> None:
    """A ``SandboxAgent``'s shell (``exec_command``) is the run's sandbox shell, as Claude
    Code's and Kimi Code's ``Bash``, Deep Agents' ``execute`` and ``sandbox_exec`` are: a
    command that reads and one that writes run without asking a person, each announced (the
    skills suite's OpenAI chats waited on a person to read a skill file)."""
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    model = ScriptedModel(
        [
            ("exec_command", {"cmd": "echo kept > f.txt"}),
            ("exec_command", {"cmd": "cat f.txt"}),
            "done",
        ]
    )
    target = SandboxAgent(name="box", instructions="Work.", model=model, capabilities=[Shell()])
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    agent = harness.wrap(target, id="shell", framework_options={"run_config": config})
    events = [e async for e in agent.stream("go", user="u")]
    assert not [e for e in events if e.type is RunEventType.INTERRUPT]
    assert events[-1].type is RunEventType.RUN_FINISHED and "kept" in str(model.outputs()[-1])
    announced = [e.data["args"]["cmd"] for e in events if e.data.get("name") == bridge.NOTICE]
    assert announced == ["echo kept > f.txt", "cat f.txt"]  # each command said on the stream


@pytest.mark.host_execution
@pytest.mark.parametrize(
    ("said", "asks"),
    [
        ({"exec_command": "irreversible"}, 2),
        ({"exec_command": Rule(None, approve_when="true")}, 1),
    ],
    ids=["catalog-irreversible", "approve-when"],
)
async def test_a_sandbox_shell_asks_where_the_administrator_said_so(
    harness: Harness, said: dict[str, Any], asks: int
) -> None:
    """The catalog's word on the shell wins over the sandbox's default: an administrator's
    ``irreversible`` asks at every call, an ``approve_when`` asks for the call it holds on. (The
    shell's own ``write``, which the harness publishes to the catalog, announces it.)"""
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    model = ScriptedModel(
        [
            ("exec_command", {"cmd": "rm -f old.txt"}),
            ("exec_command", {"cmd": "rm -f older.txt"}),
            "done",
        ]
    )
    target = SandboxAgent(name="box", instructions="Work.", model=model, capabilities=[Shell()])
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    agent = declaring(harness, **said).wrap(
        target, id="said", framework_options={"run_config": config}
    )
    result = await agent.run("go", user="u")
    asked = 0
    while result.interrupt is not None:
        assert result.interrupt.tool_call is not None
        assert result.interrupt.tool_call.tool == "exec_command"
        asked += 1
        result = await agent.resume(
            result.interrupt.interrupt_id, "approve", reviewer="u", remember="run"
        )
    assert result.status is RunStatus.SUCCESS and asked >= asks


# --------------------------------------------------------------------------- pass-throughs
@pytest.mark.parametrize("kind", ["async_sqlite", "advanced_sqlite", "sqlalchemy"])
async def test_the_sdks_session_backends_keep_the_conversation(
    harness: Harness, tmp_path: Path, kind: str
) -> None:
    if kind == "async_sqlite":
        from agents.extensions.memory import AsyncSQLiteSession

        session: Any = AsyncSQLiteSession("s", tmp_path / "a.db")
    elif kind == "advanced_sqlite":
        from agents.extensions.memory import AdvancedSQLiteSession

        session = AdvancedSQLiteSession(
            session_id="s", db_path=tmp_path / "b.db", create_tables=True
        )
    else:
        from agents.extensions.memory import SQLAlchemySession

        session = SQLAlchemySession.from_url(
            "s", url=f"sqlite+aiosqlite:///{tmp_path / 'c.db'}", create_tables=True
        )
    model = ScriptedModel(["one", "two"])
    agent = harness.wrap(Agent(name="b", model=model), id=f"backend-{kind}")
    await agent.run("first", user="u", framework_options={"session": session})
    await agent.run("second", user="u", framework_options={"session": session})
    assert said_by_user(model.inputs[1]) == ["first", "second"]


async def test_the_runs_tool_execution_config_is_the_sdks(harness: Harness) -> None:
    from agents.run_config import ToolExecutionConfig

    running: list[int] = []
    most: list[int] = []

    @function_tool
    async def slow(n: int) -> str:
        """Slow."""
        running.append(n)
        most.append(len(running))
        await asyncio.sleep(0.05)
        running.remove(n)
        return str(n)

    model = ScriptedModel([[("slow", {"n": i}) for i in range(3)], "done"])
    config = RunConfig(tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1))
    agent = declaring(harness, slow="read").wrap(
        Agent(name="c", model=model, tools=[slow]),
        id="one-at-a-time",
        framework_options={"run_config": config},
    )
    assert (await agent.run("go", user="u")).answer == "done"
    assert max(most) == 1


async def test_an_sse_server_and_a_server_managers_servers_are_governed_as_any_server(
    harness: Harness,
) -> None:
    """``MCPServerSse`` and the servers an ``MCPServerManager`` keeps are ``MCPServer``\\ s:
    the run's copy of the agent governs each (``GovernedServer``), the SDK's own object kept;
    one that is not connected is left out of the run's tool set (said once)."""
    from agents.mcp import MCPServerManager, MCPServerSse

    from trellis.harness.adapters.openai_agents import GovernedServer, governed_agents

    sse = MCPServerSse(params={"url": "http://mcp.test/sse"}, name="sse")
    manager = MCPServerManager([sse])
    target = Agent(name="m", model=ScriptedModel(["ok"]), mcp_servers=[*manager.all_servers])
    copy = governed_agents(target, added=[], agent_id="sse", nested=lambda _agent, own: own)
    [governed] = copy.mcp_servers
    assert isinstance(governed, GovernedServer)
    assert cast("Any", governed).params is sse.params  # the SDK's own state, read through
    agent = harness.wrap(target, id="sse")
    assert await agent.adapter.native(agent.target) == []


# --------------------------------------------------------------------------- sandbox
@pytest.mark.host_execution
async def test_a_sandbox_tool_that_asks_is_held_at_the_sdks_gate_in_the_same_sandbox(
    harness: Harness,
) -> None:
    """A ``SandboxAgent``'s shell the catalog says asks (``irreversible``) is held at the SDK's
    own gate: the approved call continues the SDK's paused run — the model is not asked again,
    and the file the first command wrote is there for the second."""
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    model = ScriptedModel(
        [
            ("exec_command", {"cmd": "echo kept > f.txt"}),
            ("exec_command", {"cmd": "cat f.txt"}),
            "done",
        ]
    )
    target = SandboxAgent(name="box", instructions="Work.", model=model, capabilities=[Shell()])
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    governed = declaring(harness, exec_command="irreversible")
    agent = governed.wrap(target, id="held", framework_options={"run_config": config})
    result = await agent.run("go", user="u")
    for command in ("echo kept > f.txt", "cat f.txt"):
        assert result.interrupt is not None and result.interrupt.tool_call is not None
        assert result.interrupt.tool_call.args == {"cmd": command}
        result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="u")
    assert result.status is RunStatus.SUCCESS and len(model.inputs) == 3  # never asked again
    assert "kept" in str(model.outputs()[-1])


async def test_an_encrypted_session_on_wrap_is_each_conversations_with_the_one_it_wraps(
    harness: Harness, tmp_path: Path
) -> None:
    """The SDK's ``EncryptedSession`` wraps another session: given on ``h.wrap``, both are kept
    per conversation, so bob's model never reads alice's turn."""
    pytest.importorskip("cryptography")
    from agents.extensions.memory import EncryptedSession

    inner = SQLiteSession("inner", tmp_path / "e.db")
    session = EncryptedSession("outer", inner, encryption_key="k" * 32)
    model = ScriptedModel(["noted", "nothing", "42"])
    agent = harness.wrap(
        Agent(name="a", model=model), id="e", framework_options={"session": session}
    )
    await agent.run("my secret is 42", user="alice", thread="t1")
    await agent.run("what do you know?", user="bob", thread="t1")
    await agent.run("what is my secret?", user="alice", thread="t1")
    assert said_by_user(model.inputs[1]) == ["what do you know?"]
    assert said_by_user(model.inputs[2]) == ["my secret is 42", "what is my secret?"]
    inner.close()


async def test_the_sdks_tool_output_trimmer_runs_as_the_runs_own_input_filter(
    harness: Harness,
) -> None:
    """``ToolOutputTrimmer`` is a ``call_model_input_filter`` of the run's own: it runs before
    the harness's additions, and trims the agent's own tool's old result as without it."""
    from agents.extensions.tool_output_trimmer import ToolOutputTrimmer

    @function_tool
    def dump(n: int) -> str:
        """A long result."""
        return "x" * 2000 + str(n)

    model = ScriptedModel([("dump", {"n": 1}), "first", "second", "third"])
    trimmer = ToolOutputTrimmer(recent_turns=1, max_output_chars=100, preview_chars=10)
    own = SQLiteSession("trim")
    agent = declaring(harness, dump="read").wrap(
        Agent(name="t", model=model, tools=[dump]), id="trim"
    )
    options = {"run_config": RunConfig(call_model_input_filter=trimmer), "session": own}
    assert (await agent.run("go", user="u", framework_options=options)).answer == "first"
    assert len(str(model.outputs()[0])) > 2000  # its own result, as returned
    await agent.run("again", user="u", framework_options=options)
    await agent.run("and again", user="u", framework_options=options)
    assert len(str(model.outputs()[0])) < 200  # an old turn's result, trimmed by the SDK


async def test_the_runs_handoff_history_setting_is_the_sdks(harness: Harness) -> None:
    """``RunConfig.nest_handoff_history`` reaches the SDK: the agent handed off to reads the
    earlier conversation as the SDK nests it (one summary message)."""
    model = ScriptedModel([("transfer_to_billing", {}), "billed"])
    billing = Agent(name="billing", model=model)
    triage = Agent(name="triage", model=model, handoffs=[billing])
    config = RunConfig(nest_handoff_history=True)
    result = await harness.wrap(triage, id="nest", framework_options={"run_config": config}).run(
        "bill me", user="u"
    )
    assert result.answer == "billed"
    assert "<CONVERSATION HISTORY>" in json.dumps(model.inputs[-1], default=str)


@pytest.mark.parametrize("where", ["handoff", "run_config"])
async def test_the_sdks_handoff_filter_decides_what_the_next_agent_reads(
    harness: Harness, where: str
) -> None:
    """The SDK's ``handoff_filters.remove_all_tools`` — on a ``Handoff`` object, or as the
    run's ``RunConfig.handoff_input_filter`` — runs under the harness as without it: the agent
    handed off to reads the conversation without the first agent's tool call and its result,
    while that call was governed and ran."""
    from agents import handoff
    from agents.extensions.handoff_filters import remove_all_tools

    ran: list[str] = []

    @function_tool
    def find(order: str) -> str:
        """Find an order."""
        ran.append(order)
        return f"found {order}"

    model = ScriptedModel([("find", {"order": "o1"}), ("transfer_to_billing", {}), "billed"])
    billing = Agent(name="billing", model=model)
    on_handoff = where == "handoff"
    to_billing = handoff(billing, input_filter=remove_all_tools) if on_handoff else billing
    triage = Agent(name="triage", model=model, tools=[find], handoffs=[to_billing])
    config = RunConfig() if on_handoff else RunConfig(handoff_input_filter=remove_all_tools)
    agent = declaring(harness, find="read").wrap(
        triage, id=f"filter-{where}", framework_options={"run_config": config}
    )
    result = await agent.run("bill o1", user="u")
    assert result.answer == "billed" and ran == ["o1"]
    assert "found o1" in json.dumps(model.inputs[1], default=str)  # triage read its result
    assert "found o1" not in json.dumps(model.inputs[2], default=str)  # billing does not
