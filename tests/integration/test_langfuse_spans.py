"""Every step of a run reaches Langfuse, whatever framework ran it (docs/observability.md):
each framework's own built-in tool call is an ``execute_tool`` span with the attributes of a
harness tool's (``trellis.governance.action`` as governance decided it: ``ask`` for a call a
person approved), under the attempt's span; a sub-agent the framework spawns is an agent span
its calls nest under; an approval is an ``interrupt`` event (who is asked, the question, the
call) and the answer a ``decision`` event (the choice: once, conversation, always, deny); a
failure has its status and message, redacted; the run's outcome and spend are on its span and
trace; a framework's own model replies are ``chat`` generations with their usage and cost.
In-memory exporter, scripted models: offline."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from agents import Agent, Usage, function_tool
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tests.support.adapters import CLI
from tests.support.arm import arm
from tests.support.models import ScriptedChat
from tests.support.openai_model import Items, ScriptedModel
from tests.support.planned import PlannedChatModel
from trellis import Budget, Harness, ReAct, Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness import telemetry
from trellis.harness.adapters.acp import KIMI, ACPAgent

FAKE_ACP = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")
SECRET = "sk-live-0123456789abcdef0123456789abcdef"
#: what every ``execute_tool`` span carries (docs/observability.md, "Spans")
TOOL_KEYS = {
    "gen_ai.operation.name",
    "gen_ai.tool.name",
    "gen_ai.tool.call.id",
    "gen_ai.tool.type",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "langfuse.observation.type",
    "trellis.tool.source",
    "trellis.governance.action",
}


class Tree:
    """The finished spans, by name, with their parents."""

    def __init__(self, spans: InMemorySpanExporter) -> None:
        self.spans = list(spans.get_finished_spans())
        self.by_id = {s.context.span_id: s for s in self.spans if s.context is not None}

    def named(self, name: str) -> list[ReadableSpan]:
        return [s for s in self.spans if s.name == name]

    def one(self, name: str) -> ReadableSpan:
        found = self.named(name)
        assert len(found) == 1, (name, [s.name for s in self.spans])
        return found[0]

    def parent(self, span: ReadableSpan) -> ReadableSpan:
        assert span.parent is not None, span.name
        return self.by_id[span.parent.span_id]

    def everything(self) -> str:
        """Every attribute and event of every span, as text (what a backend receives)."""
        return json.dumps(
            [
                [
                    dict(s.attributes or {}),
                    [dict(e.attributes or {}) for e in s.events],
                    s.status.description,
                ]
                for s in self.spans
            ],
            default=str,
        )


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def built_in(tree: Tree, name: str, attempt: str, action: str) -> dict[str, Any]:
    """The built-in call ``name``'s span: every tool attribute, under an attempt of
    ``attempt``, with governance's ``action``."""
    span = tree.one(f"execute_tool {name}")
    found = attrs(span)
    assert found.keys() >= TOOL_KEYS, sorted(TOOL_KEYS - found.keys())
    assert found["langfuse.observation.type"] == "tool"
    assert found["gen_ai.tool.name"] == name
    assert found["trellis.governance.action"] == action, (name, found)
    assert tree.parent(span).name == f"invoke_agent {attempt}"
    return found


def failed_span(tree: Tree, name: str, message: str) -> None:
    span = tree.one(name)
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is not None and message in span.status.description
    found = attrs(span)
    assert found["langfuse.observation.level"] == "ERROR"
    assert message in found["langfuse.observation.status_message"]


def asked_and_decided(tree: Tree, tool_name: str, choice: str) -> None:
    """The approval asked (whose question, the call) and the person's answer (the choice), as
    event observations under the attempts they happened in."""
    [interrupt] = tree.named("interrupt")
    found = attrs(interrupt)
    assert found["langfuse.observation.type"] == "event"
    assert found["trellis.interrupt.tool"] == tool_name
    assert found["trellis.interrupt.question"].startswith(f"Approve {tool_name}?")
    assert found["trellis.interrupt.reason"] == "APPROVAL"
    assert tree.parent(interrupt).name.startswith("invoke_agent ")
    paused = attrs(tree.parent(interrupt))
    assert paused["trellis.run.status"] == "PAUSED"
    [decision] = tree.named("decision")
    said = attrs(decision)
    assert said["trellis.decision.decision"] == "APPROVE"
    assert said["trellis.decision.reviewer"] == "u1"
    assert said["trellis.decision.choice"] == choice
    resumed = attrs(tree.parent(decision))
    assert resumed["trellis.resumed"] is True and resumed["trellis.attempt"] == 2
    assert resumed["trellis.decision.interrupt_id"] == found["trellis.interrupt.id"]


# --------------------------------------------------------------------------- Claude Code


def claude(tmp_path: Path, script: list[dict[str, Any]]) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
            "FAKE_CLAUDE_SESSIONS": str(tmp_path),
            "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
            "FAKE_CLAUDE_COST": "0.002",
        },
    )


async def test_claude_codes_built_ins_are_tool_spans_with_governances_action(
    harness: Harness, tmp_path: Path, spans: InMemorySpanExporter
) -> None:
    script = [
        {"builtin": "Read", "args": {"file_path": "orders.csv"}},
        {"builtin": "Write", "args": {"file_path": "notes.txt", "content": f"--token {SECRET}"}},
        {"builtin": "Grep", "args": {"pattern": "x"}, "fails": "bad pattern"},
        {"text": "done"},
    ]
    agent = harness.wrap(claude(tmp_path, script), id="coder")
    paused = await agent.run("tidy", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done
    tree = Tree(spans)
    assert built_in(tree, "Read", "coder", "run")["gen_ai.tool.call.result"] == "Read ran"
    wrote = built_in(tree, "Write", "coder", "ask")  # a person approved it
    assert wrote["gen_ai.tool.call.id"] == "toolu_1"
    built_in(tree, "Grep", "coder", "run")
    failed_span(tree, "execute_tool Grep", "bad pattern")
    asked_and_decided(tree, "Write", "once")
    # its model's replies: generations with usage and the cost the CLI reported
    generations = [attrs(s) for s in tree.spans if s.name.startswith("chat ")]
    assert generations and all(g["langfuse.observation.type"] == "generation" for g in generations)
    assert all(g["gen_ai.usage.input_tokens"] > 0 for g in generations)
    assert any(g.get("gen_ai.usage.cost") for g in generations)
    # the run's outcome and spend: on the attempt that answered, and on the trace
    [answered] = [s for s in tree.named("invoke_agent coder") if attrs(s)["trellis.attempt"] == 2]
    ended = attrs(answered)
    assert ended["trellis.run.status"] == ended["langfuse.trace.metadata.status"] == "SUCCESS"
    assert ended["langfuse.trace.metadata.spent_tokens"] > 0
    assert ended["trellis.spent.cost"] > 0
    assert SECRET not in tree.everything()


# --------------------------------------------------------------------------- Kimi Code (ACP)


async def test_kimis_tools_and_its_sub_agent_are_spans(
    harness: Harness, tmp_path: Path, spans: InMemorySpanExporter
) -> None:
    script = [
        {"builtin": "Read", "args": {"path": "orders.csv"}, "output": "o1,o2"},
        {"builtin": "Bash", "args": {"command": "rm a.txt"}, "ask": True, "output": ""},
        {"builtin": "Write", "args": {"path": "a.txt", "content": "x"}, "ask": True, "output": ""},
        {"builtin": "Grep", "args": {"pattern": "x"}, "fails": "bad pattern"},
        {
            "agent": {"description": "look", "prompt": "find tea", "subagent_type": "explore"},
            "result": "agent_id: agent-0\nstatus: completed\n\n[summary]\ntea found",
        },
        {"text": "done: {last}"},
    ]
    target = ACPAgent(
        command=(sys.executable, FAKE_ACP),
        env={
            "FAKE_ACP_SCRIPT": json.dumps(script),
            "FAKE_ACP_RECORD": str(tmp_path / "record.json"),
            "KIMI_CODE_HOME": str(tmp_path / "home"),
        },
        cwd=str(tmp_path),
        profile=KIMI,
    )
    agent = harness.wrap(target, id="kimi")
    paused = await agent.run("tidy", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "approve", reviewer="u1", remember="run"
    )
    assert done.status is RunStatus.SUCCESS, done
    tree = Tree(spans)
    built_in(tree, "Read", "kimi", "run")
    built_in(tree, "Bash", "kimi", "announce")  # a command: the deployment's, announced
    commanded(tree, "Bash", "rm a.txt", "framework")  # the fake agent ran it itself
    built_in(tree, "Write", "kimi", "ask")
    failed_span(tree, "execute_tool Grep", "bad pattern")
    built_in(tree, "Agent", "kimi", "run")
    asked_and_decided(tree, "Write", "conversation")
    helper = tree.one("invoke_agent explore")
    found = attrs(helper)
    assert found["langfuse.observation.type"] == "agent"
    assert found["trellis.parent_run_id"] == done.run_id
    assert found["trellis.run.status"] == "SUCCESS"
    assert tree.parent(helper).name == "invoke_agent kimi"


# --------------------------------------------------------------------------- Deep Agents


async def test_deep_agents_file_tools_and_its_task_sub_agent_are_spans(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    plan: list[Any] = [
        ("write_file", {"file_path": "/a.txt", "content": "hi"}),
        ("read_file", {"file_path": "/missing.txt"}),
        ("task", {"description": "look", "subagent_type": "general-purpose"}),
    ]
    graph = create_deep_agent(model=PlannedChatModel(plan=plan), system_prompt="w")
    done = await harness.wrap(graph, id="deep").run("go", user="u1")
    assert done.status is RunStatus.SUCCESS, done
    tree = Tree(spans)
    writes = tree.named("execute_tool write_file")
    lead = [s for s in writes if tree.parent(s).name == "invoke_agent deep"]
    assert len(lead) == 1 and attrs(lead[0])["trellis.governance.action"] == "run"
    assert attrs(lead[0]).keys() >= TOOL_KEYS
    reads = [
        s for s in tree.named("execute_tool read_file") if s.status.status_code is StatusCode.ERROR
    ]
    assert reads, "a failed read is an error span"
    task = tree.one("execute_tool task")
    helper = tree.one("invoke_agent general-purpose")
    assert tree.parent(helper) is task  # the sub-agent under the call that spawned it
    assert attrs(helper)["trellis.helper.kind"] == "task"
    inner = [s for s in writes if tree.parent(s) is helper]
    assert inner, "the sub-agent's own calls nest under its span"


# --------------------------------------------------------------------------- LangChain


@lc_tool
def weather(city: str) -> str:
    """Weather of a city."""
    return f"sunny in {city}"


async def test_a_langchain_tool_in_a_users_graph_is_a_span_with_its_approval(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    model = PlannedChatModel(plan=[("weather", {"city": "Oslo"})])
    agent = harness.wrap(create_agent(model, tools=[weather], system_prompt="w"), id="graph")
    paused = await agent.run("weather?", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    first = Tree(spans).one("invoke_agent graph")
    assert first.status.status_code is not StatusCode.ERROR  # a pause is no error
    done = await agent.resume(
        paused.interrupt.interrupt_id, "approve", reviewer="u1", remember="always"
    )
    assert done.status is RunStatus.SUCCESS, done
    tree = Tree(spans)
    found = built_in(tree, "weather", "graph", "ask")
    assert found["gen_ai.tool.call.result"] == "sunny in Oslo"
    asked_and_decided(tree, "weather", "always")


# --------------------------------------------------------------------------- OpenAI Agents


async def test_openai_agents_function_and_hosted_tools_are_spans(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    from openai.types.responses import ResponseFunctionWebSearch
    from openai.types.responses.response_function_web_search import ActionSearch

    @function_tool
    def stock(item: str) -> str:
        """Stock of an item."""
        return f"7 {item}"

    hosted = Items(
        ResponseFunctionWebSearch(
            id="ws_1",
            type="web_search_call",
            status="completed",
            action=ActionSearch(type="search", query="carrier delays"),
        )
    )
    usage = Usage(requests=1, input_tokens=11, output_tokens=3, total_tokens=14)
    model = ScriptedModel([hosted, ("stock", '{"item": "tea"}'), "7 tea"], usage=usage)
    agent = harness.wrap(Agent(name="shop", model=model, tools=[stock]), id="shop")
    paused = await agent.run("how much tea?", user="u1")
    assert paused.interrupt is not None  # its own tool, of no declared risk: asked
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done
    tree = Tree(spans)
    assert built_in(tree, "web_search", "shop", "run")["gen_ai.tool.call.id"] == "ws_1"
    assert built_in(tree, "stock", "shop", "ask")["gen_ai.tool.call.result"] == "7 tea"
    asked_and_decided(tree, "stock", "once")
    generations = [attrs(s) for s in tree.spans if s.name.startswith("chat ")]
    assert generations and all(g["gen_ai.usage.input_tokens"] == 11 for g in generations)


# --------------------------------------------------------------------------- outcomes


async def test_a_failed_run_says_why_on_its_span_redacted(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    async def breaks(input: str, agent: Runtime) -> str:
        raise RuntimeError(f"the vendor refused key {SECRET}")

    done = await harness.wrap(breaks, id="breaks").run("go", user="u1")
    assert done.status is RunStatus.ERROR
    tree = Tree(spans)
    failed_span(tree, "invoke_agent breaks", "the vendor refused key")
    found = attrs(tree.one("invoke_agent breaks"))
    assert found["trellis.run.status"] == found["langfuse.trace.metadata.status"] == "ERROR"
    assert found["error.type"]
    assert SECRET not in tree.everything()


@tool(side_effects="read")
def leaky(order: str) -> str:
    """Look an order up."""
    raise ValueError(f"upstream said Bearer {SECRET}")


async def test_a_tool_that_fails_is_an_error_span_with_its_message_redacted(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    model = ScriptedChat([("leaky", {"order": "o1"}), "it failed"])
    agent = arm(harness.wrap(ReAct(system_prompt="p", model=model), id="lookup"), [leaky])
    done = await agent.run("o1?", user="u1")
    assert done.status is RunStatus.SUCCESS, done
    tree = Tree(spans)
    failed_span(tree, "execute_tool leaky", "upstream said")
    assert attrs(tree.one("execute_tool leaky"))["error.type"]
    assert SECRET not in tree.everything()


async def test_the_step_limit_a_timeout_and_a_cancel_say_why(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    @tool(side_effects="read")
    def ping(n: int) -> str:
        """Ping."""
        return "pong"

    model = ScriptedChat([("ping", {"n": 1}), ("ping", {"n": 2}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="p", model=model), id="limited"), [ping])
    partial = await agent.run("go", user="u1", budget=Budget(max_tool_calls=1))
    assert partial.status is RunStatus.PARTIAL, partial
    found = attrs(Tree(spans).one("invoke_agent limited"))
    assert found["langfuse.observation.level"] == "WARNING"
    assert "max_tool_calls" in found["langfuse.observation.status_message"]
    assert attrs(Tree(spans).one("partial"))["langfuse.observation.type"] == "event"

    started = asyncio.Event()

    async def slow(input: str, agent: Runtime) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    spans.clear()
    timed = await harness.wrap(slow, id="slow").run("go", user="u1", run_timeout=0.2)
    assert timed.status is RunStatus.TIMEOUT
    tree = Tree(spans)
    span = tree.one("invoke_agent slow")
    assert span.status.status_code is StatusCode.ERROR
    assert attrs(tree.one("timeout"))["langfuse.observation.level"] == "ERROR"

    spans.clear()
    wrapped = harness.wrap(slow, id="stopped")
    started.clear()
    running = asyncio.ensure_future(wrapped.run("go", user="u1"))
    await asyncio.wait_for(started.wait(), 5)
    [record] = [s async for s in harness.runs.iterate(agent_id="stopped")]
    await wrapped.cancel(record.run_id, reason="the user changed their mind")
    cancelled = await running
    assert cancelled.status is RunStatus.CANCELLED
    found = attrs(Tree(spans).one("invoke_agent stopped"))
    assert found["trellis.run.status"] == "CANCELLED"
    assert found["langfuse.observation.level"] == "WARNING"


def test_a_choice_is_named_as_a_person_made_it() -> None:
    assert telemetry.choice("APPROVE", None) == "once"
    assert telemetry.choice("EDIT", "run") == "conversation"
    assert telemetry.choice("APPROVE", "always") == "always"
    assert telemetry.choice("REJECT", None) == "deny"
    assert telemetry.choice("APPROVE", "later") == "later"


# --------------------------------------------------------------------------- shell commands


def commanded(tree: Tree, name: str, text: str, where: str) -> dict[str, Any]:
    """A shell call's span: its command (redacted), where it ran, how long it took."""
    [span] = [
        s
        for s in tree.named(f"execute_tool {name}")
        if text in str(attrs(s).get("trellis.command.text"))
    ]
    found = attrs(span)
    assert found["trellis.command.where"] == where, found
    assert found["trellis.command.duration_ms"] >= 0
    assert SECRET not in tree.everything()
    return found


async def test_a_command_in_the_runs_sandbox_is_its_audit_record_on_the_span(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    """The harness's ``sandbox_exec`` and Deep Agents' ``execute`` on the run's sandbox."""
    from tests.support.sandbox import FakeSandboxes
    from trellis.harness.sandbox import sandbox

    provider = FakeSandboxes()
    model = ScriptedChat([("sandbox_exec", {"command": "echo hi"}), "done"])
    agent = arm(harness.wrap(ReAct("s", model), id="box"), [sandbox(provider).tools()])
    assert (await agent.run("go", user="u1")).status is RunStatus.SUCCESS
    found = commanded(Tree(spans), "sandbox_exec", "echo hi", "sandbox")
    assert found["trellis.command.exit_code"] == 0
    assert str(found["trellis.command.sandbox"]).startswith("trellis-run_")
    assert "trellis.command.limit" not in found

    spans.clear()  # past the run's time: the time limit stopped it
    model = ScriptedChat([("sandbox_exec", {"command": "sleep 30"}), "done"])
    slow = arm(harness.wrap(ReAct("s", model), id="slow-box"), [sandbox(provider).tools()])
    # time enough to reach the command on a slow runner (at 1 s a CI run timed out before it
    # started, so it had no span), far short of the command's own 30 s
    await slow.run("go", user="u1", run_timeout=5)
    stopped = commanded(Tree(spans), "sandbox_exec", "sleep 30", "sandbox")
    assert stopped["trellis.command.limit"] == "time"
    assert "trellis.command.exit_code" not in stopped

    spans.clear()
    model = ScriptedChat([("execute", {"command": "echo hi"}), "done"])
    deep = harness.wrap(ReAct("s", model, backend=sandbox(FakeSandboxes())), id="deep-box")
    assert (await deep.run("go", user="u1")).status is RunStatus.SUCCESS
    assert commanded(Tree(spans), "execute", "echo hi", "sandbox")["trellis.command.exit_code"] == 0


async def test_the_isolation_layers_audit_record_is_the_spans(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    """A command the isolation layer ran says so on the run's stream (``sandbox_command``):
    the same record is on the span of the call that ran it, a limit that stopped it with it."""
    from trellis import current

    @tool(side_effects="read")
    def isolated(command: str) -> str:
        """Run a command on this machine, isolated."""
        record = {"command": command, "exit_code": 137, "duration_ms": 12.5, "limit": "memory"}
        running = current()
        assert running is not None
        running.events.custom("sandbox_command", sandbox="host-1", uid=1000000001, **record)
        return "killed"

    model = ScriptedChat([("isolated", {"command": f"make big --key {SECRET}"}), "done"])
    agent = arm(harness.wrap(ReAct("s", model), id="host"), [isolated])
    assert (await agent.run("go", user="u1")).status is RunStatus.SUCCESS
    found = commanded(Tree(spans), "isolated", "make big", "host")
    assert found["trellis.command.exit_code"] == 137
    assert found["trellis.command.limit"] == "memory"
    assert found["langfuse.observation.status_message"] == "stopped by its memory limit"


async def test_a_frameworks_own_shell_commands_are_spans(
    harness: Harness, tmp_path: Path, spans: InMemorySpanExporter
) -> None:
    """Claude Code's ``Bash``, OpenAI Agents' local shell: the command, where it ran (the
    framework's own), how long it took."""
    from agents import LocalShellTool
    from openai.types.responses.response_output_item import LocalShellCall, LocalShellCallAction

    script = [{"builtin": "Bash", "args": {"command": "ls -la"}}, {"text": "done"}]
    claude_agent = harness.wrap(claude(tmp_path, script), id="sh-claude")
    assert (await claude_agent.run("list", user="u1")).status is RunStatus.SUCCESS
    commanded(Tree(spans), "Bash", "ls -la", "framework")

    spans.clear()
    call = LocalShellCall(
        id="ls_1",
        call_id="call_ls",
        status="completed",
        type="local_shell_call",
        action=LocalShellCallAction(type="exec", command=["ls", "build"], env={}),
    )
    model = ScriptedModel([Items(call), "listed"])  # the framework's own shell: never asked
    shell = LocalShellTool(executor=lambda request: "a.txt")
    oa = harness.wrap(Agent(name="ops", model=model, tools=[shell]), id="sh-openai")
    done = await oa.run("list", user="u1")
    assert done.status is RunStatus.SUCCESS, done
    commanded(Tree(spans), "local_shell", "ls build", "framework")


def test_a_shells_command_is_read_from_its_arguments() -> None:
    assert telemetry.command_text({"command": "ls"}) == "ls"
    assert telemetry.command_text({"action": {"command": ["ls", "-a"]}}) == "ls -a"
    assert telemetry.command_text({"action": {"commands": ["ls", "pwd"]}}) == "ls pwd"
    assert telemetry.command_text({"action": "x"}) is None
    assert telemetry.command_text("ls") is None
