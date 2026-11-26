"""ACP agents (``adapters.acp``): the real ACP SDK on both sides — the harness's client, and a
scripted agent (``tests/support/fake_acp_agent.py``) in a process of its own, speaking ACP over
stdio as ``kimi acp`` does — permissions, sessions, the harness's MCP server, sub-agents, the
usage log and the sandbox included. ``tests/integration/test_kimi.py`` runs Kimi Code itself."""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.kimi import kimi_home_of
from tests.support.memory import FakeMemoryService
from tests.support.sandbox import FakeSandboxes
from trellis import Deny, Harness, Hooks, Rewrite, current, tool
from trellis.contracts import (
    ConfigurationError,
    Option,
    RunEventType,
    RunOutcome,
    RunStatus,
    ToolCall,
)
from trellis.harness import admission, compat
from trellis.harness.adapters import acp
from trellis.harness.adapters.acp import (
    ANSWERED,
    EDITED,
    KIMI,
    KIMI_ASKED,
    KIMI_BUILTINS,
    KIMI_ENV,
    REJECTED,
    ACPAgent,
    UsageLog,
    kimi,
    native_spec,
)
from trellis.harness.agent import NARROW_FROM
from trellis.harness.bridges import acp_live, live
from trellis.harness.bridges.resumed import RESUMED_ASKED, RESUMED_CALL, RESUMED_DENIED
from trellis.harness.budget import Budget
from trellis.harness.history import ASKED_WITH
from trellis.harness.sandbox import DELIVERS_HERE, sandbox
from trellis.harness.tools import bridge

#: Claude Code and Kimi Code run their shells in a folder of this machine: the tests' deployment
#: allows it (TRELLIS_HOST_EXECUTION=allow, the harness fixtures).
pytestmark = pytest.mark.host_execution

#: What Kimi reads its swarm's concurrency from.
SWARM_LIMIT = "KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY"

FAKE = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")
refunds: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return f"refunded {order}"


@tool(side_effects="read")
def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"facts about {topic}"


def fake(tmp_path: Path, script: list[dict[str, Any]], **fields: Any) -> ACPAgent:
    """The scripted agent as Kimi: its home and its record in ``tmp_path``."""
    env = {
        "FAKE_ACP_SCRIPT": json.dumps(script),
        "FAKE_ACP_RECORD": str(tmp_path / "record.json"),
        "KIMI_CODE_HOME": str(tmp_path / "home"),
        **fields.pop("env", {}),
    }
    return ACPAgent(
        command=(sys.executable, FAKE),
        env=env,
        cwd=fields.pop("cwd", str(tmp_path)),
        profile=fields.pop("profile", KIMI),
        **fields,
    )


async def children(h: Harness, parent_run_id: str) -> list[Any]:
    found = [s async for s in h.runs.iterate(parent_run_id=parent_run_id)]
    records = [await h.runs.get(s.run_id) for s in found]
    return sorted((r for r in records if r is not None), key=lambda r: r.run_id)


def recorded(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "record.json"
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------- tools and answers
async def test_a_turn_answers_and_calls_harness_tools(harness: Harness, tmp_path: Path) -> None:
    refunds.clear()
    script = [{"tool": "lookup", "args": {"topic": "tea"}}, {"text": "found: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="lookups", without={"schedules"}), [lookup])
    result = await agent.run("tell me about tea", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "found: facts about tea"
    record = recorded(tmp_path)
    # read_file is listed from the start: the agent lists a server's tools once
    assert record["listed"] == [["lookup", "read_file", "query_result"]]
    [session] = record["sessions"]
    [server] = session["servers"]
    assert server["name"] == "trellis" and server["url"].startswith("http://127.0.0.1:")
    assert server["headers"][0]["name"] == "Authorization"
    # the harness tool needs no person: Kimi asks, the harness lets it through
    assert record["permissions"] == [{"title": "mcp__trellis__lookup", "answer": "approve_once"}]
    # what Kimi is told of its folder's outputs/ (the user's), then the user's message: a run
    # nobody is known to read has no question tool, so no note on when to ask
    assert record["prompts"] == [f"{DELIVERS_HERE}\ntell me about tea"]
    assert record["initialize"][0]["terminal"] is False  # no sandbox: no terminal of ours


async def test_the_turns_tool_calls_and_results_are_in_its_transcript(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """The agent reports each call it makes (``tool_call``) and its result (its update): the
    transcript memory keeps has each call by the harness tool's name and its result, with its
    run record, so a later chat's steps list says which tool answered (explain-08 and histx on
    Kimi, the all-adapter run, said "the record holds only the answer")."""
    script = [{"tool": "lookup", "args": {"topic": "tea"}}, {"text": "found: {last}"}]
    agent = arm(memory_harness.wrap(fake(tmp_path, script), id="lookups"), [lookup])
    assert (await agent.run("tell me about tea", user="u1")).answer == "found: facts about tea"
    await memory_harness.writes.drain()
    [batch] = memory_service.named("messages")
    sent = batch.body["messages"]
    assert [(m["role"], m["content"]) for m in sent] == [
        ("USER", "tell me about tea"),
        ("ASSISTANT", ""),
        ("TOOL", "facts about tea"),
        ("ASSISTANT", "found: facts about tea"),
    ]
    [made] = sent[1]["tool_calls"]
    assert made["name"] == "lookup" and made["arguments"] == {"topic": "tea"}
    assert sent[2]["tool_call_id"] == made["id"]
    assert sent[2]["custom_metadata"]["run_record"]["status"] == "ok"


async def test_the_harness_mcp_server_refuses_a_request_without_its_token() -> None:
    import httpx

    from trellis.harness.tools.convert import mcp as convert

    server = convert.convert([])
    async with convert.served(server) as served, httpx.AsyncClient() as client:
        refused = await client.post(served.url, json={})
        assert refused.status_code == 401


async def test_the_harness_mcp_server_says_when_a_client_listed_its_tools() -> None:
    """Only a listing of its tools by a client carrying its token counts as the client having
    them: a request without the token does not, nor a connection that listed nothing (an
    ``initialize`` alone: a Kimi whose listing then failed answers without the tools)."""
    import httpx

    from trellis.harness.tools.convert import mcp as convert

    listing = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    accept = {"Accept": "application/json, text/event-stream"}
    async with convert.served(convert.convert([])) as served, httpx.AsyncClient() as client:
        await client.post(served.url, json=listing, headers=accept)
        assert not served.reached.is_set()
        hello = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        }
        await client.post(served.url, json=hello, headers={**served.headers, **accept})
        assert not served.reached.is_set()
        listed = await client.post(served.url, json=listing, headers={**served.headers, **accept})
        assert listed.status_code == 200 and served.reached.is_set()


async def test_a_kimi_that_never_reached_the_harness_tools_is_started_again_then_fails(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Kimi whose connection to the harness's MCP server failed would answer without the
    run's tools, as if it had none (the accuracy run's first Kimi session): its process is
    started again, said on the run, and when the next one does not connect either the run
    fails saying why, rather than answering without its data."""
    monkeypatch.setattr(acp, "CONNECT_SECONDS", 0.2)
    script = [{"text": "I have no sales data"}]
    agent = arm(
        harness.wrap(fake(tmp_path, script, env={"FAKE_ACP_CONNECT": "never"}), id="cut_off"),
        [lookup],
    )
    events = [e async for e in agent.stream("how many sold?", user="u1")]
    warnings = [e.data["message"] for e in events if e.data.get("code") == "acp_tools"]
    assert warnings == ["Kimi Code did not connect to the harness's tools: started again"]
    assert len(recorded(tmp_path)["sessions"]) == 2 and "prompts" not in recorded(tmp_path)
    result = await agent.run("how many sold?", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "did not connect to the harness's tools (2 tries)" in result.error.message


async def test_a_kimi_that_connected_but_never_listed_the_harness_tools_is_not_prompted(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Kimi that reached the harness's MCP server but never had its tools (its listing
    failed after the connection: the accuracy run's long-chain-04, a new conversation in the
    middle of the run whose Kimi said the run's tools were not in its list) is not prompted
    as if it had them: a conversation before it had its tools; this one's process is started
    again, said on the run, and when the next one has no tools either the run fails."""
    monkeypatch.setattr(acp, "CONNECT_SECONDS", 0.5)
    had = [{"tool": "lookup", "args": {"topic": "tea"}}, {"text": "found: {last}"}]
    before = await arm(harness.wrap(fake(tmp_path, had), id="ok"), [lookup]).run("tea?", user="u1")
    assert before.status is RunStatus.SUCCESS and before.answer == "found: facts about tea"
    script = [{"text": "those tools aren't in my tool list"}]
    target = fake(tmp_path, script, env={"FAKE_ACP_CONNECT": "partial"})
    agent = arm(harness.wrap(target, id="partial"), [lookup])
    events = [e async for e in agent.stream("how many sold?", user="u1")]
    warnings = [e.data["message"] for e in events if e.data.get("code") == "acp_tools"]
    assert warnings == ["Kimi Code did not connect to the harness's tools: started again"]
    assert len(recorded(tmp_path)["prompts"]) == 1  # the conversation before's only
    result = await agent.run("how many sold?", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "did not connect to the harness's tools (2 tries)" in result.error.message


async def test_an_agent_that_connects_only_when_it_calls_is_not_waited_for(
    harness: Harness, tmp_path: Path
) -> None:
    """Only an agent whose profile says it connects as a session opens is held to it: another
    ACP agent may reach the server at its first call, or never in a turn with no call."""
    from dataclasses import replace

    lazy = replace(KIMI, connects=False)
    script = [{"text": "hello"}]
    target = fake(tmp_path, script, env={"FAKE_ACP_CONNECT": "never"}, profile=lazy)
    agent = arm(harness.wrap(target, id="lazy"), [lookup])
    result = await agent.run("hi", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "hello"


async def test_an_approval_stops_the_turn_and_a_resume_continues_the_session(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o2"}}, {"text": "done: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="refunds"), [refund])
    paused = await agent.run("refund o2", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and refunds == []
    assert recorded(tmp_path)["cancels"]  # the turn was stopped there
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: refunded o2"
    assert refunds == ["o2"]
    record = recorded(tmp_path)
    assert [s["how"] for s in record["sessions"]] == ["new", "resume"]
    assert record["prompts"][-1] == RESUMED_CALL.format(tool="mcp__trellis__refund")


class Died(BaseException):
    """What a worker process's death looks like to the code it ran (not an ``Exception``)."""


@tool(side_effects="read")
def dies(order: str) -> str:
    """A tool whose worker dies while it runs."""
    raise Died


async def test_a_worker_that_dies_in_a_harness_call_dies_in_the_run_not_in_its_server(
    harness: Harness, tmp_path: Path
) -> None:
    """A harness tool reaches an ACP agent through an MCP server the attempt serves: a
    ``BaseException`` that is not an error (the worker dying, ``SystemExit``) raised in its
    call ended the server's own task group, and the run went on as if the call had failed (a
    matrix cell, ``X15-kimi-…``: "the worker did not die"). It ends the attempt, as an
    in-process tool's does."""
    script = [{"tool": "dies", "args": {"order": "o5"}}, {"text": "went on: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="dies"), [dies])
    with pytest.raises(BaseException) as raised:
        await agent.run("go", user="u1")
    assert any(isinstance(e, Died) for e in _leaves(raised.value)), raised.value


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for e in error.exceptions for leaf in _leaves(e)]
    return [error]


async def test_a_denied_call_resumes_the_session_told_it_will_not_run(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o3"}}, {"text": "said: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="refunds-denied"), [refund])
    paused = await agent.run("refund o3", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and refunds == []
    told = RESUMED_DENIED.format(tool="mcp__trellis__refund")
    assert recorded(tmp_path)["prompts"][-1] == told and done.answer == f"said: {told}"


@tool(side_effects="read")
async def pick_warehouse() -> str:
    """Ask which warehouse ships."""
    runtime = current()
    assert runtime is not None
    return f"ships from {await runtime.ask('Which warehouse?')}"


async def test_a_question_a_tool_asked_resumes_the_session_naming_its_call(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"tool": "pick_warehouse", "args": {}}, {"text": "{last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="shipping"), [pick_warehouse])
    asked = await agent.run("where does it ship from?", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is None
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="Leeds", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "ships from Leeds"
    told = RESUMED_ASKED.format(tool="mcp__trellis__pick_warehouse", args="{}")
    assert recorded(tmp_path)["prompts"][-1] == told


async def test_a_harness_tool_error_reaches_the_agent(harness: Harness, tmp_path: Path) -> None:
    @tool(side_effects="read")
    def broken(why: str) -> str:
        """Fail."""
        raise ValueError(why)

    script = [{"tool": "broken", "args": {"why": "no stock"}}, {"text": "read: {last}"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="broken"), [broken])
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.SUCCESS and "no stock" in str(result.answer)


# --------------------------------------------------------------------------- its own tools
@pytest.mark.parametrize(
    ("args", "fields"),
    [
        ({"command": "ls"}, {}),
        ({"command": "cat /etc/passwd"}, {}),  # outside the folder it works in
        ({"command": "cat passwd", "cwd": "/etc"}, {}),  # Bash runs it in the folder it names
        ({"command": "rm a.txt"}, {}),
        ({"command": "python3 -c 'print(1)'"}, {"env": {"PYTHONPATH": "src"}}),  # Kimi's env
    ],
)
async def test_a_command_runs_unasked_and_announced_whatever_it_does(
    harness: Harness, tmp_path: Path, args: dict[str, Any], fields: dict[str, Any]
) -> None:
    """The shell is the deployment's (its sandbox, or a host its administrator allows): Kimi's
    question about a command is not the user's, whatever the command reads or writes — it runs,
    announced, with the arguments it was made with."""
    script = [{"builtin": "Bash", "args": args, "ask": True, "output": ""}, {"text": "done"}]
    agent = harness.wrap(fake(tmp_path, script, **fields), id="ops")
    events = [e async for e in agent.stream("x", user="u1")]
    assert events[-1].error is None and not any(e.type is RunEventType.INTERRUPT for e in events)
    assert [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"] == ["Bash"]
    assert [p["answer"] for p in recorded(tmp_path)["permissions"]] == ["approve_once"]


async def test_a_folder_the_target_names_is_read_unasked(harness: Harness, tmp_path: Path) -> None:
    kept = tmp_path / "kept"
    kept.mkdir()
    script = [
        {"builtin": "Bash", "args": {"command": f"cat {kept}/a.csv"}, "ask": True, "output": "1"},
        {"text": "done"},
    ]
    agent = harness.wrap(fake(tmp_path, script, reads=(str(kept),)), id="ops")
    assert (await agent.run("x", user="u1")).interrupt is None
    assert kimi(reads=["/srv/kept"]).reads == ("/srv/kept",)


async def test_its_own_tools_are_governed_and_recorded(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """A write the agent asks about (``Edit``) asks a person; approved, it runs once, after a
    pause. A ``Bash`` command runs without asking (the shell is the deployment's, as the reads
    run); each call it ran is recorded with its result."""
    script = [
        {"builtin": "Read", "args": {"path": "orders.csv"}, "output": "o1,o2"},
        {"builtin": "Bash", "args": {"command": "ls"}, "ask": True, "output": "a.txt"},
        {"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True, "output": ""},
        {"builtin": "Grep", "args": {"pattern": "x"}, "fails": "bad pattern"},
        {"text": "done: {last}"},
    ]
    agent = memory_harness.wrap(fake(tmp_path, script), id="ops")
    asked = await agent.run("tidy up", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.tool == "Edit"
    assert asked.interrupt.tool_call.args == {"path": "/srv/a.txt"}
    assert asked.interrupt.question == (
        "Approve Edit? Kimi Code asks before it runs this call. "
        "Allow once, allow for this conversation, always allow, or deny."
    )
    done = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: bad pattern"
    assert recorded(tmp_path)["prompts"][-1] == RESUMED_CALL.format(tool="Edit")
    await memory_harness.writes.drain()
    records = [
        (c.body["tool"], c.body["status"]) for c in memory_service.calls if c.name == "record_tool"
    ]
    assert records == [("Read", "ok"), ("Bash", "ok"), ("Edit", "ok"), ("Grep", "error")]


async def test_a_file_written_in_the_folder_the_agent_works_in_is_not_the_users_question(
    harness: Harness, tmp_path: Path
) -> None:
    """The agent asks before it writes a file (its ``Edit``): inside the folder it works in
    that is the run's own work, as a command of its shell there is — the harness answers,
    the call is announced, nobody is asked. A path that leaves the folder (its ``..``
    followed) is a person's to approve."""
    script = [
        {"builtin": "Edit", "args": {"path": "notes/a.txt"}, "ask": True, "output": "a made"},
        {"builtin": "Edit", "args": {"path": "../b.txt"}, "ask": True, "output": "b made"},
        {"text": "done: {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="writes-here")
    events = [e async for e in agent.stream("write them", user="u1")]
    [asked] = [e for e in events if e.type is RunEventType.INTERRUPT]
    assert asked.data["tool_call"]["args"] == {"path": "../b.txt"}
    # both writes are announced; only the one that leaves the folder waits for a person
    notices = [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"]
    assert notices == ["Edit", "Edit"]
    ended = [e.data for e in events if e.type is RunEventType.TOOL_CALL_END]
    assert len(ended) == 1, ended


async def test_a_call_allowed_for_the_conversation_is_kimis_approval_for_its_session(
    harness: Harness, tmp_path: Path
) -> None:
    """APR-01 (DECISIONS B4). Kimi offers "Approve for this session" on a call it asks about.
    The person is offered the harness's own choices for it — allow once, for this
    conversation, deny — and "for this conversation" answers Kimi with its session approval:
    the tool's later calls ask nobody, in this run (the harness's word for the rest of it),
    in Kimi's session (its own rule for the same command) and in the conversation's next
    run. One question for the chat."""
    script = [
        {"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True, "output": "a gone"},
        {"builtin": "Edit", "args": {"path": "/srv/b.txt"}, "ask": True, "output": "b gone"},
        {"text": "done: {last}"},
        {"stop": "end_turn"},
        {"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True, "output": "a gone again"},
        {"builtin": "Edit", "args": {"path": "/srv/c.txt"}, "ask": True, "output": "c gone"},
        {"text": "then: {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="keeps")
    asked = await agent.run("tidy up", user="u1", thread="chat")
    assert asked.interrupt is not None
    assert asked.interrupt.options == [
        Option(value="once", label="Allow once"),
        Option(value="run", label="Allow for this conversation"),
        Option(value="reject", label="Deny"),
    ]
    done = await agent.resume(
        asked.interrupt.interrupt_id, "approve", reviewer="u1", remember="run"
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "done: b gone"
    later = await agent.run("and again", user="u1", thread="chat")
    assert later.status is RunStatus.SUCCESS and later.answer == "then: c gone"
    assert [p["answer"] for p in recorded(tmp_path)["permissions"]] == [
        "cancelled",  # the run paused on the one question
        "approve_always",  # its answer, as Kimi takes it
        "approve_always",  # another command of the tool, in the same run: the harness's word
        "approved for this session",  # the same command again: Kimi's own rule
        "approve_once",  # another command in a later run: allowed for the conversation
    ]


async def test_a_call_allowed_once_is_kimis_approval_once(harness: Harness, tmp_path: Path) -> None:
    """APR-01 (DECISIONS B4). Allowed once, the call is approved once in Kimi: the tool's
    next call asks again."""
    script = [
        {"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True, "output": "a gone"},
        {"builtin": "Edit", "args": {"path": "/srv/b.txt"}, "ask": True, "output": "b gone"},
        {"text": "done: {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="once")
    asked = await agent.run("tidy up", user="u1", thread="chat")
    assert asked.interrupt is not None
    again = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert again.status is RunStatus.PAUSED and again.interrupt is not None
    assert again.interrupt.tool_call is not None
    assert again.interrupt.tool_call.args == {"path": "/srv/b.txt"}
    assert [p["answer"] for p in recorded(tmp_path)["permissions"]][:2] == [
        "cancelled",
        "approve_once",
    ]


async def test_a_rejected_call_ends_the_turn_and_the_agent_reads_why(
    harness: Harness, tmp_path: Path
) -> None:
    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
            return Deny("no shell on Fridays") if call.tool == "Bash" else None

    script = [
        {"builtin": "Bash", "args": {"command": "rm -rf /"}, "ask": True},
        {"text": "said: {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="guarded", hooks=[Guard()])
    result = await agent.run("clean", user="u1")
    assert result.status is RunStatus.SUCCESS
    told = recorded(tmp_path)["prompts"][-1]
    assert told.startswith(REJECTED.format(tool="Bash", why="")[:20])
    assert "no shell on Fridays" in told and result.answer == f"said: {told}"


async def test_an_edited_call_is_made_again_with_its_arguments_and_runs(
    harness: Harness, tmp_path: Path
) -> None:
    class Narrow(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
            if call.tool == "Edit" and call.args["path"] != "/srv/b.txt":
                return Rewrite({"path": "/srv/b.txt"})
            return None

    script = [{"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True}, {"text": "{last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="narrowed", hooks=[Narrow()])
    result = await agent.run("fix", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "Edit ran"
    record = recorded(tmp_path)
    assert record["prompts"][-1] == EDITED.format(tool="Edit", args='{"path": "/srv/b.txt"}')
    assert [p["answer"] for p in record["permissions"]] == ["reject", "approve_once"]


@pytest.mark.usefixtures("attended")
async def test_its_question_is_the_runs_question(harness: Harness, tmp_path: Path) -> None:
    """One question, sent as Kimi sends it to a client that takes forms (a form of one field
    that picks): a choice of its options, in ``ask_user``'s format."""
    script = [
        {"question": "Which colour?", "options": ["red", "blue"]},
        {"text": "picked {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="asker")
    asked = await agent.run("paint", user="u1")
    assert asked.interrupt is not None
    assert asked.interrupt.question == "Which colour?\n1. red\n2. blue\n(Pick one, or say another.)"
    assert asked.interrupt.options == ["red", "blue"] and asked.interrupt.component == "ask_user"
    assert asked.interrupt.payload == {"source": {"agent": "asker", "tool": "AskUserQuestion"}}
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="blue", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "picked blue"


@pytest.mark.usefixtures("attended")
async def test_a_single_question_form_keeps_its_options_descriptions(
    harness: Harness, tmp_path: Path
) -> None:
    """A form of one question keeps what Kimi's form says of each option: its description,
    and the one labelled "(Recommended)" recommended, first, with its description as why."""
    asked = {"question": "Which db?", "header": "DB", "options": [
        {"label": "mysql", "description": "what you run now"},
        {"label": "pg (Recommended)", "description": "Postgres, for the JSON columns"}]}  # fmt: skip
    script = [{"questions": [asked]}, {"text": "got {last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="described")
    paused = await agent.run("set it up", user="u1")
    assert paused.interrupt is not None
    assert paused.interrupt.options == [
        Option(value="pg (Recommended)", description="Postgres, for the JSON columns",
               recommended=True),
        Option(value="mysql", description="what you run now"),
    ]  # fmt: skip
    assert "2. mysql: what you run now" in paused.interrupt.question
    answer = "pg (Recommended)"
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer=answer, reviewer="u1")
    assert done.answer == 'got {"answers": {"Which db?": "pg (Recommended)"}}'


@pytest.mark.parametrize("forms", ["1", "0"], ids=["form", "permission-request"])
async def test_an_unattended_run_declines_its_question_in_its_own_words(
    harness: Harness, tmp_path: Path, forms: str
) -> None:
    """A run nobody is known to read never waits on a question: Kimi's form is declined (its
    question as a permission request skipped), and Kimi tells its model so in its own words;
    the run goes on."""
    script = [{"questions": QUESTIONS}, {"text": "got {last}"}]
    agent = harness.wrap(fake(tmp_path, script, env={"FAKE_ACP_FORMS": forms}), id=f"un-{forms}")
    done = await agent.run("set it up", user="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None
    record = recorded(tmp_path)
    if forms == "1":
        assert [form["answer"] for form in record["forms"]] == [{"action": "decline"}]
    else:
        assert [p["answer"] for p in record["permissions"]] == ["q0_skip"]
    assert done.answer == "got dismissed"


async def test_a_form_of_an_agent_with_no_profile_is_asked_as_a_servers_form(
    harness: Harness, tmp_path: Path
) -> None:
    """An agent the harness has no profile of names no question tool: its form is asked as an
    MCP server's is (``tools.elicit``), a form of one field that picks a choice of its
    values, whether a person is known to read the run or not."""
    script = [{"question": "Which colour?", "options": ["red", "blue"]}, {"text": "picked {last}"}]
    agent = harness.wrap(fake(tmp_path, script, profile=None), id="no-profile-form")
    asked = await agent.run("paint", user="u1")
    assert asked.interrupt is not None and asked.interrupt.question == "Which colour?"
    assert asked.interrupt.options == [
        Option(value="red", label="red"),
        Option(value="blue", label="blue"),
    ]
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="blue", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "picked blue"


SUPPLIERS = ["SUP-04 (Lyon Mills)", "SUP-07 (Prato Knit)"]


@pytest.mark.parametrize("forms", ["1", "0"], ids=["form", "permission-request"])
@pytest.mark.parametrize(
    ("answer", "read"),
    [
        (" sup-04  (LYON mills) ", "picked SUP-04 (Lyon Mills)"),  # its option, handed back
        ("SUP-07 (Prato Knit)", "picked SUP-07 (Prato Knit)"),
        *(
            (said, f"picked {ANSWERED.format(question='Which supplier?', answer=said)}")
            for said in ("SUP-04", "not SUP-04", "the second one", "neither")
        ),
    ],
    ids=["label-case-spacing", "exact", "part", "not", "ordinal", "neither"],
)
@pytest.mark.usefixtures("attended")
async def test_an_answer_to_its_question_is_an_option_handed_back_or_the_users_words(
    harness: Harness, tmp_path: Path, forms: str, answer: str, read: str
) -> None:
    """An answer that hands an option back (its case and spacing aside) is that option; any
    other is never refused and never read as an option: the question is dismissed and the
    agent told the person's words, for its model to read."""
    script = [{"question": "Which supplier?", "options": SUPPLIERS}, {"text": "picked {last}"}]
    env = {"FAKE_ACP_FORMS": forms}
    agent = harness.wrap(fake(tmp_path, script, env=env), id=f"words-{forms}")
    asked = await agent.run("reorder", user="u1")
    assert asked.interrupt is not None
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer=answer, reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer == read


@pytest.mark.usefixtures("attended")
async def test_several_picks_or_a_form_answered_in_words_reach_the_agent(
    harness: Harness, tmp_path: Path
) -> None:
    """Several picks handed back are those options; words to a form of several questions
    fill none of it: the agent reads them."""
    regions = [{"question": "Which regions?", "options": [{"label": "north"}, {"label": "east"}],
                "multi_select": True}]  # fmt: skip
    script = [{"questions": regions}, {"text": "got {last}"}]
    (tmp_path / "picks").mkdir()
    (tmp_path / "form").mkdir()
    agent = harness.wrap(fake(tmp_path / "picks", script), id="picks")
    asked = await agent.run("alert", user="u1")
    assert asked.interrupt is not None and asked.interrupt.multiple
    picked = [" North", "EAST "]
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer=picked, reviewer="u1")
    assert done.answer == 'got {"answers": {"Which regions?": "north, east"}}'
    asked = await agent.run("alert again", user="u2")
    assert asked.interrupt is not None
    words = ["north", "and the west"]
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer=words, reviewer="u2")
    said = ANSWERED.format(question="Which regions?", answer="north, and the west")
    assert done.answer == f"got {said}"
    script = [{"questions": QUESTIONS}, {"text": "got {last}"}]
    agent = harness.wrap(fake(tmp_path / "form", script), id="form-words")
    asked = await agent.run("set it up", user="u1")
    assert asked.interrupt is not None
    keep = "Postgres, and both clouds"
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer=keep, reviewer="u1")
    said = ANSWERED.format(question="Which db?\nWhich cloud?", answer=keep)
    assert done.status is RunStatus.SUCCESS and done.answer == f"got {said}"


QUESTIONS = [
    {
        "question": "Which db?",
        "header": "DB",
        "options": [{"label": "pg", "description": "Postgres"}, {"label": "mysql"}],
    },
    {
        "question": "Which cloud?",
        "header": "Cloud",
        "options": [{"label": "aws"}, {"label": "gcp"}],
        "multi_select": True,
    },
]


@pytest.mark.usefixtures("attended")
async def test_every_question_of_a_call_is_asked_in_one_form(
    harness: Harness, tmp_path: Path
) -> None:
    """L47. Kimi asks every question of an ``AskUserQuestion`` call in one form of a client
    that takes forms (ACP's ``elicitation/create``), and only the first of a client that does
    not. The harness takes forms: one question to the person in ``ask_user``'s format, a field
    a question — its options with their descriptions, several picks where the question takes
    several — and every answer goes back."""
    script = [{"questions": QUESTIONS}, {"text": "got {last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="asks-two")
    asked = await agent.run("set it up", user="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.question == (
        "2 questions:\n1. Which db? (pg: Postgres or mysql)\n2. Which cloud? (aws or gcp)\n"
        "(Answer each, or say what to keep and what to change.)"
    )
    assert asked.interrupt.component == "ask_user" and asked.interrupt.expects is not None
    fields = asked.interrupt.expects["properties"]
    assert fields["Which db?"]["oneOf"][0] == {
        "const": "pg",
        "title": "pg",
        "description": "Postgres",
    }
    assert fields["Which cloud?"]["type"] == "array"
    picked = {"Which db?": "pg", "Which cloud?": ["aws", "gcp"]}
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer=picked, reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer == 'got {"answers": {"Which db?": "pg", "Which cloud?": "aws, gcp"}}'
    record = recorded(tmp_path)
    assert [form["answer"]["action"] for form in record["forms"]] == ["cancel", "accept"]
    assert "your call to AskUserQuestion" in record["prompts"][-1]


@pytest.mark.usefixtures("attended")
async def test_a_question_asked_as_a_permission_request_offers_every_option_of_the_agents(
    harness: Harness, tmp_path: Path
) -> None:
    """An agent that sends no form asks its question as a permission request (Kimi's
    fallback: the first question, its options and ``Skip``): a request that offers a choice is
    the agent's question, asked with every option it offers."""
    script = [{"question": "Which colour?", "options": ["red", "blue"]}, {"text": "picked {last}"}]
    agent = harness.wrap(fake(tmp_path, script, env={"FAKE_ACP_FORMS": "0"}), id="asker-plain")
    asked = await agent.run("paint", user="u1")
    assert asked.interrupt is not None
    assert asked.interrupt.question == (
        "Which colour?\n1. red\n2. blue\n3. Skip\n(Pick one, or say another.)"
    )
    assert asked.interrupt.options == ["red", "blue", "Skip"]
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="Skip", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "picked dismissed"


@pytest.mark.parametrize(
    ("approaches", "picked", "result"),
    [
        (["Fast", "Safe"], "Safe", "Selected approach: Safe"),
        (["Fast", "Safe"], "Revise", "requested revisions"),
        ([], "Approve", "Exited plan mode."),
        ([], "Reject and Exit", "Plan rejected by user"),
    ],
    ids=["an-approach", "revise", "approve", "reject-and-exit"],
)
async def test_a_plan_is_reviewed_by_a_person_who_picks_among_every_option_the_agent_offers(
    harness: Harness, tmp_path: Path, approaches: list[str], picked: str, result: str
) -> None:
    """APR-01, L47 (DECISIONS B3). Kimi presents every plan for review (``ExitPlanMode``): its
    approaches, ``Revise`` and ``Reject and Exit``. A request that offers a choice is the
    agent's question to its user: asked once, with the plan and every option, and answered with
    the one the person picked — never the first approach by nobody's choice."""
    plan = "# Plan\n\nOption A: fast.\nOption B: safe."
    script = [{"review": {"plan": plan, "options": approaches}}, {"text": "then: {last}"}]
    agent = harness.wrap(fake(tmp_path, script, mode="plan"), id="planner")
    asked = await agent.run("plan it", user="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.question.startswith(plan)
    offered = [*(approaches or ["Approve"]), "Revise", "Reject and Exit"]
    assert asked.interrupt.options == offered and asked.interrupt.tool_call is None
    assert [p["answer"] for p in recorded(tmp_path)["permissions"]] == ["cancelled"]
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer=picked, reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert done.answer is not None and result in done.answer
    record = recorded(tmp_path)
    assert len(record["permissions"]) == 2  # asked again by the resumed session: no new question
    assert "your call to ExitPlanMode" in record["prompts"][-1]
    assert record["modes"] == ["plan"]  # still planning, and asking, when the resume continued it


@pytest.mark.usefixtures("attended")
async def test_a_question_answered_with_no_option_it_offers_is_skipped(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"question": "Which?", "options": ["a", "b"]}, {"text": "{last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="skipper")
    asked = await agent.run("x", user="u1")
    assert asked.interrupt is not None
    answer = acp.chosen([], "reject_once")
    assert answer.outcome.outcome == "cancelled"  # no option of that kind: the question dismissed


@pytest.mark.usefixtures("attended")
async def test_its_question_call_with_no_question_is_its_calls_error(
    harness: Harness, tmp_path: Path
) -> None:
    """An ``AskUserQuestion`` call that asks nothing asks nobody: the form is refused saying
    why, the agent's call fails with it, and the run goes on."""
    script = [{"questions": []}, {"text": "got {last}"}]
    agent = harness.wrap(fake(tmp_path, script), id="asks-none")
    done = await agent.run("set it up", user="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done
    assert (
        done.answer
        == "got AskUserQuestion failed: AskUserQuestion asks no question: give it one or more"
    )


# --------------------------------------------------------------------------- helpers
LONG = "rm -rf build && echo this-command-is-longer-than-the-fifty-characters-kimi-shows"


async def test_a_sub_agent_is_a_child_run_and_its_calls_are_governed_with_their_arguments(
    harness: Harness, tmp_path: Path
) -> None:
    """APR-01, L47. ACP shows no update of a sub-agent's own call, and Kimi's request for it
    names the tool and a line that cuts a command at 50 characters. The call is governed with
    what Kimi shows its own user of it (its log's ``interaction.request``); a command is the
    deployment's, so it runs unasked, and the sub-agent's run is a child run."""
    script = [
        {
            "agent": {"description": "look", "prompt": "find tea", "subagent_type": "explore"},
            "asks": {"builtin": "Bash", "args": {"command": LONG}},
            "result": "agent_id: agent-0\nstatus: completed\n\n[summary]\ntea found",
        },
        {"text": "done"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="lead")
    done = await agent.run("find tea", user="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None  # the shell: unasked
    kids = await children(harness, done.run_id)
    assert [c.metadata["native_helper"]["name"] for c in kids] == ["explore"]
    assert {c.status for c in kids} == {RunStatus.SUCCESS}


async def test_a_sub_agents_command_that_only_reads_runs_without_asking_a_person(
    harness: Harness, tmp_path: Path
) -> None:
    """APR-01, L47. With the command read whole, a sub-agent's is decided as the main agent's
    is: one that only reads asks nobody."""
    script = [
        {
            "agent": {"prompt": "look"},
            "asks": {"builtin": "Bash", "args": {"command": "ls"}},
            "result": "x",
        },
        {"text": "done"},
    ]
    done = await harness.wrap(fake(tmp_path, script), id="lead-reads").run("look", user="u1")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None
    assert [p["answer"] for p in recorded(tmp_path)["permissions"]] == ["approve_once"]


async def test_with_no_log_of_the_agent_a_sub_agents_call_is_asked_about_as_its_request_says(
    harness: Harness, tmp_path: Path
) -> None:
    """An agent the harness reads no log of: a sub-agent's call is decided at once (nothing
    is waited for) with what its request says, which is all there is."""
    import dataclasses

    script = [
        {
            "agent": {"prompt": "tidy"},
            "asks": {"builtin": "Edit", "args": {"path": "a"}},
            "result": "x",
        },
        {"text": "done"},
    ]
    plain = dataclasses.replace(KIMI, usage=None)
    asked = await harness.wrap(fake(tmp_path, script, profile=plain), id="no-log").run(
        "x", user="u1"
    )
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.args == {"request": "Requesting approval to Calling Edit"}


async def test_each_user_has_a_home_of_the_agents_own_an_overlay_of_the_teams(
    harness: Harness, tmp_path: Path
) -> None:
    """SEC (DECISIONS A2, for Kimi). Kimi is started in its user's own home: the team's login
    and setup linked, its config the team's with an ``ask`` rule after it for each ``allow``
    rule that would let a call run unasked and for Kimi's ``Skill`` (KIMI-SKILL-1) and its
    network tools (the catalog's to decide), its
    sessions its user's alone."""
    team = tmp_path / "home"
    team.mkdir()
    (team / "credentials").mkdir()
    rules = '[[permission.allow]]\ntool = "Bash"\nmatch = "rm *"\n'
    (team / "config.toml").write_text(rules)
    agent = harness.wrap(fake(tmp_path, [{"text": "ok"}]), id="homes")
    for user in ("ada", "bob"):
        assert (await agent.run("x", user=user)).status is RunStatus.SUCCESS
    homes = [Path(started["env"]["KIMI_CODE_HOME"]) for started in recorded(tmp_path)["started"]]
    assert homes == [kimi_home_of("ada"), kimi_home_of("bob")]
    for home in homes:
        assert (home / "credentials").resolve() == team / "credentials"
        # the team's config, Kimi's own [tools] taking its schedule tools off the list, then asks
        asked = (home / "config.toml").read_text().removeprefix(acp.cron_hidden(rules))
        assert asked == "".join(
            f'\n[[permission.rules]]\ndecision = "ask"\npattern = "{pattern}"\n'
            for pattern in ("Bash(rm *)", *KIMI_ASKED)
        )
        assert len(list(home.glob("sessions/*/session_*"))) == 1
    assert (team / "config.toml").read_text() == rules and not (team / "sessions").exists()


def test_kimis_schedule_tools_are_taken_off_where_the_teams_config_says_nothing_of_tools() -> None:
    hidden = acp.cron_hidden('model = "kimi-k2"\n')
    assert tomllib.loads(hidden)["tools"]["disabled"] == list(acp.KIMI_OUTLIVES_THE_TURN)
    own = '[tools]\ndisabled = ["Bash"]\n'
    assert acp.cron_hidden(own) == own  # the team's own list, as it wrote it
    assert acp.cron_hidden("not = [toml") == "not = [toml"  # nothing changed in a broken config


@pytest.mark.parametrize(
    ("given", "asked"),
    [
        (
            '[permission]\nrules = [{ decision = "allow", pattern = "Bash(rm *)" }]\n',
            "".join(
                f'\n[[permission.ask]]\npattern = "{p}"\n' for p in ("Bash(rm *)", *KIMI_ASKED)
            ),
        ),
        ('permission = { rules = [{ decision = "allow", pattern = "Bash(rm *)" }] }\n', None),
    ],
    ids=["inline-rules", "inline-permission"],
)
async def test_a_config_written_inline_is_asked_over_where_toml_lets_a_rule_follow(
    harness: Harness, tmp_path: Path, given: str, asked: str | None
) -> None:
    """APR-01. A config whose ``[permission]`` rules are inline takes Kimi's other form of an
    ``ask`` rule after them (``[[permission.ask]]``): the run goes on, Kimi asking. One whose
    whole ``permission`` is an inline table takes none: its ``allow`` rule would let a call run
    unasked, so the run is not started, and says which rules."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(given)
    done = await harness.wrap(fake(tmp_path, [{"text": "ok"}]), id="allows").run("x", user="u1")
    own = (kimi_home_of("u1") / "config.toml").read_text()
    if asked is None:
        assert done.status is RunStatus.ERROR and done.error is not None
        assert "Bash(rm *)" in done.error.message and "prompts" not in recorded(tmp_path)
        assert own == acp.cron_hidden(given)
    else:
        assert done.status is RunStatus.SUCCESS, done.error
        assert own == acp.cron_hidden(given) + asked


async def test_a_failed_sub_agent_ends_its_child_run_failed(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"agent": {"prompt": "find"}, "result": "", "fails": True},
        {"text": "gave up"},
    ]
    done = await harness.wrap(fake(tmp_path, script), id="lead2").run("find", user="u1")
    assert done.status is RunStatus.SUCCESS
    [child] = await children(harness, done.run_id)
    assert child.status is RunStatus.ERROR and child.metadata["native_helper"]["name"] == "coder"


async def test_each_swarm_member_is_a_child_run(harness: Harness, tmp_path: Path) -> None:
    """Kimi's ``AgentSwarm`` passes through as it is: each item a child run, ended as its
    member's result says (a member Kimi reports nothing of fails)."""
    result = (
        "<agent_swarm_result>\n<summary>completed: 1</summary>\n"
        '<subagent agent_id="agent-0" item="a.ts" outcome="completed">fine</subagent>\n'
        '<subagent agent_id="agent-1" item="b.ts" outcome="failed">broke</subagent>\n'
        "</agent_swarm_result>"
    )
    swarm = {
        "description": "review",
        "prompt_template": "Review {{item}}",
        "items": ["a.ts", "b.ts", "c.ts"],
    }
    script = [{"swarm": swarm, "result": result}, {"text": "reviewed"}]
    done = await harness.wrap(fake(tmp_path, script), id="swarm").run("review", user="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "reviewed"
    kids = sorted(await children(harness, done.run_id), key=lambda c: c.input)
    assert [(c.input, c.status) for c in kids] == [
        ("Review a.ts", RunStatus.SUCCESS),
        ("Review b.ts", RunStatus.ERROR),
        ("Review c.ts", RunStatus.ERROR),
    ]
    assert "helper" in native_spec("AgentSwarm", KIMI).tags


async def test_resumed_sub_agents_and_swarm_members_are_child_runs_by_their_own_names(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi goes on with a sub-agent it ran before: ``Agent(resume=<agent id>)``, and a swarm's
    ``resume_agent_ids`` (each id with the prompt it goes on with) beside its items. Each is a
    child run, named as Kimi started it (its log's ``profile.bind``: a resumed agent keeps its
    type), its task the prompt it goes on with, ended as the swarm's result says of it — a
    resumed member is ``mode="resume"``, an item Kimi escapes (``&amp;``) is still its item."""
    result = (
        "<agent_swarm_result>\n<summary>completed: 2, failed: 1</summary>\n"
        "<resume_hint>Call AgentSwarm with resume_agent_ids ...</resume_hint>\n"
        '<subagent mode="resume" agent_id="agent-0" item="a.ts" outcome="completed">'
        "went on</subagent>\n"
        '<subagent mode="resume" agent_id="agent-1" state="idle" outcome="failed" '
        'stop_reason="timed out">it timed out</subagent>\n'
        '<subagent agent_id="agent-2" item="x&amp;y.ts" outcome="completed">fine</subagent>\n'
        "</agent_swarm_result>"
    )
    swarm = {
        "description": "again",
        "resume_agent_ids": {"agent-0": "continue", "agent-1": "fix it"},
        "prompt_template": "Review {{item}}",
        "items": ["x&y.ts"],
    }
    script = [
        {"bound": {"agent": "agent-0", "profile": "explore"}},
        {"bound": {"agent": "agent-1", "profile": "plan"}},
        {"swarm": swarm, "result": result},
        {
            "agent": {"description": "more", "prompt": "carry on", "resume": "agent-0"},
            "result": "ok",
        },
        {"usage": {"inputOther": 1, "output": 1}, "agent": "agent-9"},  # a log, no profile
        {"agent": {"description": "lost", "prompt": "who?", "resume": "agent-9"}, "result": "?"},
        {"text": "reviewed"},
    ]
    done = await harness.wrap(fake(tmp_path, script), id="resumes").run("review", user="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    kids = sorted(await children(harness, done.run_id), key=lambda c: c.input)
    assert [(c.input, c.metadata["native_helper"]["name"], c.status) for c in kids] == [
        ("Review x&y.ts", "coder", RunStatus.SUCCESS),
        ("carry on", "explore", RunStatus.SUCCESS),
        ("continue", "explore", RunStatus.SUCCESS),
        ("fix it", "plan", RunStatus.ERROR),
        ("who?", "agent-9", RunStatus.SUCCESS),  # no log says its type: named by its id
    ]
    assert [c.output for c in kids[:3]] == ["fine", "ok", "went on"]
    assert kids[3].error is not None and kids[3].error.message == "it timed out"


async def test_a_swarm_runs_as_many_at_once_as_the_harness_admits_and_is_on_the_tree(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi is told the adaptive limit of child runs as its swarm's concurrency (the agent's own
    ``env`` wins); each member is on the run's task tree."""
    monkeypatch.delenv(SWARM_LIMIT, raising=False)
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(start=3, machine=None))
    result = (
        "<agent_swarm_result>\n"
        '<subagent agent_id="agent-0" item="a.ts" outcome="completed">fine</subagent>\n'
        "</agent_swarm_result>"
    )
    swarm = {"description": "review", "prompt_template": "Review {{item}}", "items": ["a.ts"]}
    script = [{"swarm": swarm, "result": result}, {"text": "reviewed"}]
    agent = harness.wrap(fake(tmp_path, script), id="swarm")
    events = [e async for e in agent.stream("review", user="u1")]
    assert recorded(tmp_path)["started"][0]["env"][SWARM_LIMIT] == "3"
    tree = [e.data for e in events if e.type is RunEventType.CUSTOM]
    tree = [d for d in tree if d["name"].startswith("helper_")]
    assert [(d["name"], d.get("status")) for d in tree] == [
        ("helper_started", None),
        ("helper_finished", "success"),
    ]
    assert tree[0]["framework"] == "acp" and tree[0]["kind"] == "subagent"
    own = harness.wrap(fake(tmp_path, [{"text": "ok"}], env={SWARM_LIMIT: "7"}), id="own")
    await own.run("x", user="u1")
    assert recorded(tmp_path)["started"][-1]["env"][SWARM_LIMIT] == "7"


async def test_kimis_scheduling_is_turned_off_whatever_its_own_environment_says(
    harness: Harness, tmp_path: Path
) -> None:
    """A schedule fires only while Kimi runs, and Kimi ends with each attempt: Kimi's own switch
    (``KIMI_DISABLE_CRON``) is set for it, an agent's own value included, as Claude's cron tools
    are disallowed whatever its options say."""
    agent = harness.wrap(fake(tmp_path, [{"text": "ok"}], env={"KIMI_DISABLE_CRON": "0"}), id="c")
    await agent.run("every Monday", user="u1")
    assert recorded(tmp_path)["started"][0]["env"]["KIMI_DISABLE_CRON"] == "1"
    assert KIMI.env == KIMI_ENV


async def test_kimi_neither_updates_itself_nor_reports_usage_whatever_its_own_environment_says(
    harness: Harness, tmp_path: Path
) -> None:
    """DECISIONS A3. On a server Kimi must stay the version that was tested (its own update
    would move it mid-deployment) and send its maker no usage of the tenants': both switches
    of Kimi's own are set, and an agent's ``env`` that asks otherwise is told why."""
    asks = {"KIMI_CODE_NO_AUTO_UPDATE": "0", "KIMI_DISABLE_TELEMETRY": "0"}
    agent = harness.wrap(fake(tmp_path, [{"text": "ok"}], env=asks), id="quiet")
    events = [e async for e in agent.stream("hi", user="u1")]
    env = recorded(tmp_path)["started"][0]["env"]
    assert (env["KIMI_CODE_NO_AUTO_UPDATE"], env["KIMI_DISABLE_TELEMETRY"]) == ("1", "1")
    said = [e.data["message"] for e in events if e.data.get("code") == "acp_env"]
    assert len(said) == 2 and all("is not used" in message for message in said)


async def test_kimis_tower_mode_is_turned_off_and_an_agent_that_asks_for_it_is_told_why(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi's tower mode (``KIMI_CODE_EXPERIMENTAL_TOWER``) starts its workers in Kimi's
    ``auto`` permission mode, which asks about none of their calls: the harness could govern
    none of them. Kimi's own switch turns it off, an agent's own value included (Kimi reads its
    environment's word over its config's) — and an agent whose ``env`` asks otherwise is told,
    for each, why it is not so (a warning on the run's stream)."""
    asked = {"KIMI_CODE_EXPERIMENTAL_TOWER": "1", "KIMI_DISABLE_CRON": "0"}
    agent = harness.wrap(fake(tmp_path, [{"text": "ok"}], env=asked), id="tower")
    events = [e async for e in agent.stream("plan missions", user="u1")]
    env = recorded(tmp_path)["started"][0]["env"]
    assert (env["KIMI_CODE_EXPERIMENTAL_TOWER"], env["KIMI_DISABLE_CRON"]) == ("0", "1")
    warned = [e.data["message"] for e in events if e.data.get("code") == "acp_env"]
    assert warned == [
        f"KIMI_DISABLE_CRON=0 in the agent's env is not used: {acp.CRON_OFF}",
        f"KIMI_CODE_EXPERIMENTAL_TOWER=1 in the agent's env is not used: {acp.TOWER_OFF}",
    ]
    quiet = harness.wrap(fake(tmp_path, [{"text": "ok"}], env={"KIMI_DISABLE_CRON": "1"}), id="q")
    events = [e async for e in quiet.stream("hi", user="u1")]
    assert not [e for e in events if e.data.get("code") == "acp_env"]  # it asks as the harness does


# --------------------------------------------------------------------------- sessions
async def test_a_session_the_agent_no_longer_holds_is_run_again_from_its_prompt(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o4"}}, {"text": "done"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="moved"), [refund])
    paused = await agent.run("refund o4", user="u1")
    assert paused.interrupt is not None
    for kept in kimi_home_of("u1").rglob("state.json"):
        kept.unlink()  # resumed on a machine whose agent never held the session
    events: list[Any] = []
    record = await harness.runs.get(paused.run_id)
    assert record is not None
    resumed, resolution = await agent._resolution(
        paused.interrupt.interrupt_id, "approve", None, "u1", tenant=record.tenant_id
    )
    done = await agent._continue(resumed, resolution, events.append)
    assert done.status is RunStatus.SUCCESS and refunds == ["o4"]
    [warning] = [e for e in events if e.data.get("code") == "acp_session"]
    assert "was not resumed" in warning.data["message"]
    # from the prompt, the journal's
    # nobody known to read the run: no question tool, no note on when to ask
    assert recorded(tmp_path)["prompts"][-1] == f"{DELIVERS_HERE}\nrefund o4"


async def test_an_agent_without_session_resume_loads_the_session(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [
        {"text": "looking"},
        {"tool": "refund", "args": {"order": "o5"}},
        {"text": "done"},
    ]
    target = fake(tmp_path, script, env={"FAKE_ACP_RESUME": "0"})
    agent = arm(harness.wrap(target, id="loader"), [refund])
    paused = await agent.run("refund o5", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    # the loaded session's history replayed while it loads is not this turn's answer
    assert done.status is RunStatus.SUCCESS and done.answer == "done"
    assert [s["how"] for s in recorded(tmp_path)["sessions"]] == ["new", "load"]


async def test_the_session_mode_is_set_and_one_that_asks_nothing_refused(
    harness: Harness, tmp_path: Path
) -> None:
    await harness.wrap(fake(tmp_path, [{"text": "ok"}], mode="plan"), id="planner").run(
        "x", user="u1"
    )
    assert recorded(tmp_path)["modes"] == ["plan"]
    (tmp_path / "plain").mkdir()
    await harness.wrap(fake(tmp_path / "plain", [{"text": "ok"}]), id="plain").run("x", user="u1")
    assert recorded(tmp_path / "plain")["modes"] == ["default"]
    with pytest.raises(ConfigurationError, match="a mode that approves on its own"):
        kimi(mode="yolo")


@pytest.mark.parametrize("starts_in", ["auto", "yolo"])
async def test_a_session_asks_whatever_the_agents_config_starts_it_in(
    harness: Harness, tmp_path: Path, starts_in: str
) -> None:
    """APR-01, L46. Kimi's ``config.toml`` can start every session in a mode that asks about
    nothing (``default_permission_mode``) while ACP says ``default``: the harness sets the mode
    that asks on every session it opens — a new one, and the one a resume continues — so the
    call still asks a person and runs once, approved."""
    script = [
        {"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True, "output": "gone"},
        {"builtin": "Edit", "args": {"path": "/srv/b.txt"}, "ask": True, "output": "gone too"},
        {"text": "done: {last}"},
    ]
    config = {"FAKE_ACP_CONFIG": json.dumps({"default_permission_mode": starts_in})}
    agent = harness.wrap(fake(tmp_path, script, env=config), id="asks")
    asked = await agent.run("tidy up", user="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.args == {"path": "/srv/a.txt"}
    again = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert again.status is RunStatus.PAUSED and again.interrupt is not None
    assert again.interrupt.tool_call is not None
    assert again.interrupt.tool_call.args == {"path": "/srv/b.txt"}
    done = await agent.resume(again.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: gone too"
    record = recorded(tmp_path)
    assert record["modes"] == ["default"] * 3  # each process's session, before its prompt
    assert "unasked" not in [p["answer"] for p in record["permissions"]]


async def test_a_session_the_agent_leaves_in_a_mode_that_asks_nothing_fails_the_run(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """APR-01, L46. Fails closed: the agent answered ``session/set_mode`` but its log says the
    session still approves on its own, so no prompt is sent and the run fails saying so."""
    script = [{"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True}, {"text": "done"}]
    env = {
        "FAKE_ACP_CONFIG": json.dumps({"default_permission_mode": "auto"}),
        "FAKE_ACP_MODE_STUCK": "1",
    }
    monkeypatch.setattr(live, "SETTLE_SECONDS", 0.3)  # the wait before a prompt is sent
    result = await harness.wrap(fake(tmp_path, script, env=env), id="stuck").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "session mode was set to 'default'" in result.error.message
    assert "permission mode 'auto'" in result.error.message
    assert "prompts" not in recorded(tmp_path)


async def test_a_mode_kimis_log_confirms_late_still_runs_and_no_prompt_goes_before_it(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """APR-01, L46. Kimi writes its log after it answers, and on a busy server long after:
    the run waits for the line as long as a prompt waits for its session (not the few
    seconds a request's line is waited for), sends no prompt meanwhile, and goes on once the
    log confirms the mode."""
    gate = tmp_path / "the-log-is-written"
    script = [{"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True}, {"text": "done"}]
    env = {
        "FAKE_ACP_CONFIG": json.dumps({"default_permission_mode": "auto"}),
        "FAKE_ACP_LOG_GATE": str(gate),
    }
    # the wait for a request's line is over before the mode's line is written
    monkeypatch.setattr(acp_live, "ATTRIBUTED_SECONDS", 0.0)
    running = asyncio.ensure_future(
        harness.wrap(fake(tmp_path, script, env=env), id="late").run("x", user="u1")
    )
    while "modes" not in recorded(tmp_path):  # noqa: ASYNC110 - the stand-in's record, a file
        await asyncio.sleep(0.01)
    assert not running.done() and "prompts" not in recorded(tmp_path)
    gate.touch()
    asked = await running
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert len(recorded(tmp_path)["prompts"]) == 1


async def test_a_session_keeps_the_plan_mode_it_is_in(harness: Harness, tmp_path: Path) -> None:
    """L47. Kimi's own plan mode is the session's, and Kimi refuses to enter the plan mode a
    session is in. One its config starts planning (``default_plan_mode``) and approving on its
    own leaves plan mode to be made to ask, and enters it again; one that plans and asks
    already — its config's, or the session a resume continues (``mode="plan"`` began it) — is
    set nothing, its plan as it was."""
    asks = {"builtin": "Edit", "args": {"path": "/srv/a.txt"}, "ask": True, "output": "gone"}
    for name in "abc":
        (tmp_path / name).mkdir()
    config = {"default_plan_mode": True, "default_permission_mode": "auto"}
    alone = fake(
        tmp_path / "a", [asks, {"text": "ok"}], env={"FAKE_ACP_CONFIG": json.dumps(config)}
    )
    asked = await harness.wrap(alone, id="alone").run("x", user="u1")
    assert asked.status is RunStatus.PAUSED
    assert recorded(tmp_path / "a")["modes"] == ["default", "plan"]
    config = {"default_plan_mode": True}
    planning = fake(tmp_path / "b", [{"text": "ok"}], env={"FAKE_ACP_CONFIG": json.dumps(config)})
    assert (await harness.wrap(planning, id="planning").run("x", user="u1")).answer == "ok"
    assert "modes" not in recorded(tmp_path / "b")
    planner = harness.wrap(fake(tmp_path / "c", [asks, {"text": "ok"}], mode="plan"), id="plans")
    asked = await planner.run("x", user="u1")
    assert asked.interrupt is not None
    done = await planner.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    assert recorded(tmp_path / "c")["modes"] == ["plan"]


async def test_an_agent_the_harness_does_not_know_keeps_the_mode_it_starts_in(
    harness: Harness, tmp_path: Path
) -> None:
    """An ACP agent with no profile names its own modes: the harness sets only the one the
    target gives (``mode=``), on the session it starts."""
    for name in "ab":
        (tmp_path / name).mkdir()
    plain = fake(tmp_path / "a", [{"text": "ok"}], profile=None)
    assert (await harness.wrap(plain, id="plain").run("x", user="u1")).answer == "ok"
    assert "modes" not in recorded(tmp_path / "a")
    given = fake(tmp_path / "b", [{"text": "ok"}], profile=None, mode="plan")
    assert (await harness.wrap(given, id="given").run("x", user="u1")).answer == "ok"
    assert recorded(tmp_path / "b")["modes"] == ["plan"]


async def test_kimis_built_in_tools_do_not_count_toward_narrowing_the_runs_own(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Kimi's built-ins are the framework's, not the run's: the tools the hints narrow from
    (``NARROW_FROM``) are the run's own. Counted with them, a run of a few tools of its own
    narrowed, and the ``trellis`` server did not list the tool a question needed."""

    def make(i: int) -> Any:
        def look(key: str) -> str:
            return f"t{i}"

        return tool(look, name=f"t{i}", side_effects="read")

    memory_service.candidates = ["t1"]
    own = NARROW_FROM - len(KIMI_BUILTINS) + 1  # with the built-ins, past the threshold
    script = [{"tool": "t0", "args": {"key": "a"}}, {"text": "{last}"}]
    agent = arm(
        memory_harness.wrap(fake(tmp_path, script), id="own"), [make(i) for i in range(own)]
    )
    result = await agent.run("look a up", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "t0"
    [listed] = recorded(tmp_path)["listed"]
    assert {f"t{i}" for i in range(own)} <= set(listed)


# --------------------------------------------------------------------------- the context
async def test_the_context_goes_before_the_prompt_marked_off_from_it(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """ACP's prompt has no place for background but its blocks: the context goes first, marked
    off as every adapter's is where it shares the user's message (``history.ASKED_WITH``).
    Unmarked, a model read the context as the user's message and the question as a note."""
    await memory_harness.wrap(fake(tmp_path, [{"text": "ok"}]), id="ctx").run(
        "reach me?", user="u1"
    )
    [prompt] = recorded(tmp_path)["prompts"]
    opened, closed = ASKED_WITH.split("{context}")
    assert prompt.startswith(f"{DELIVERS_HERE}\n{opened}{memory_service.context_text}")
    assert prompt.endswith(f"{closed}\nreach me?")
    [request] = memory_service.named("context")
    available = request.body["tools"]["available"]
    assert {"Bash", "Read", "AgentSwarm"} <= set(available)  # Kimi's own tools: its tool set


async def test_streaming_carries_the_agents_text(harness: Harness, tmp_path: Path) -> None:
    agent = harness.wrap(fake(tmp_path, [{"plan": True}, {"text": "hello there"}]), id="s")
    events = [e async for e in agent.stream("hi", user="u1")]
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert deltas == ["hello ", "there"]
    assert events[-1].type is RunEventType.RUN_FINISHED


# --------------------------------------------------------------------------- limits
async def test_usage_is_read_from_the_agents_log_and_the_budget_stops_the_turn(
    harness: Harness, tmp_path: Path
) -> None:
    """Every model request Kimi logs — its sub-agents' too — counts; past the budget the turn is
    cancelled and the run ends PARTIAL."""
    script = [
        {"usage": {"inputOther": 100, "output": 50, "inputCacheRead": 10}},
        {"usage": {"inputOther": 100, "output": 50}, "agent": "agent-0"},
        {"usage": {"garbled": True}},
        {"stall": 2.0},
        {"text": "never"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="spender", budget=Budget(max_tokens=200))
    result = await agent.run("spend", user="u1")
    assert result.status is RunStatus.PARTIAL and result.error is not None
    assert result.error.code == "BUDGET_EXCEEDED"
    assert recorded(tmp_path)["cancels"]


async def test_acps_own_usage_counts_when_the_agent_gives_it(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ACP agent the harness keeps no usage log of counts the usage ACP's answer to a prompt
    carries; Kimi's own log, read as the turn runs, is its count (the answer's is not added)."""
    from trellis.harness import budget

    counted: list[dict[str, Any]] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (counted.append(kw), account(run, **kw))
    )
    usage = {"totalTokens": 30, "inputTokens": 20, "outputTokens": 10}
    usage |= {"cachedReadTokens": 7, "cachedWriteTokens": 3}
    script = [{"prompt_usage": usage}, {"text": "ok"}]
    agent = harness.wrap(fake(tmp_path, script, profile=None), id="u")
    assert (await agent.run("x", user="u1")).status is RunStatus.SUCCESS
    said = {"model": "acp", "input_tokens": 20, "output_tokens": 10, "cached": 7, "written": 3}
    assert counted == [said]
    counted.clear()
    (tmp_path / "k").mkdir()
    kimi_like = harness.wrap(fake(tmp_path / "k", script), id="k")
    assert (await kimi_like.run("x", user="u1")).status is RunStatus.SUCCESS
    assert counted == []  # Kimi's count is its log's


async def test_a_usage_line_still_being_written_is_read_next_time(tmp_path: Path) -> None:
    folder = tmp_path / "sessions" / "wd" / "s1" / "agents" / "main"
    folder.mkdir(parents=True)
    log = folder / "wire.jsonl"
    log.write_bytes(b'{"type": "usage.record", "usage": {"output": 1}}\n{"type": "usage.re')
    usage = UsageLog(tmp_path, "s1")
    assert len(usage.records()) == 1
    with log.open("ab") as file:
        file.write(b'cord", "usage": {"output": 2}}\n[]\n')
    assert [r["usage"]["output"] for r in usage.records()] == [2]


async def test_a_stalled_step_ends_the_run(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness import runtime

    monkeypatch.setattr(runtime, "STALL_SECONDS", 0.5)
    agent = harness.wrap(fake(tmp_path, [{"stall": 30}, {"text": "late"}]), id="stalled")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.code == "MODEL_TIMEOUT"


async def test_the_run_time_limit_stops_the_agent(harness: Harness, tmp_path: Path) -> None:
    script = [{"builtin": "Bash", "args": {"command": "sleep"}, "wait": 30}, {"text": "late"}]
    agent = harness.wrap(fake(tmp_path, script), id="slow", run_timeout=1.0)
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.TIMEOUT


async def test_a_cancelled_run_stops_the_agent(harness: Harness, tmp_path: Path) -> None:
    script = [{"builtin": "Bash", "args": {"command": "sleep"}, "wait": 30}, {"text": "late"}]
    agent = harness.wrap(fake(tmp_path, script), id="cancelled")
    handle = asyncio.create_task(agent.run("x", user="u1"))
    while "sessions" not in recorded(tmp_path):  # noqa: ASYNC110 - another process says so
        await asyncio.sleep(0.05)
    [run] = [s async for s in harness.runs.iterate(agent_id="cancelled")]
    await agent.cancel(run.run_id, reason="enough")
    result = await handle
    assert result.status is RunStatus.CANCELLED


waited: list[str] = []
waiting = asyncio.Event()


@tool(side_effects="read")
async def wait_long(seconds: int) -> str:
    """Wait a number of seconds, then say so."""
    waiting.set()
    await asyncio.sleep(seconds)
    waited.append("done")
    return "waited"


async def test_a_cancelled_run_cancels_the_harness_call_under_way(
    harness: Harness, tmp_path: Path
) -> None:
    waited.clear()
    waiting.clear()
    # under the second the server is given to stop in: the call is cancelled, not waited for
    script = [{"tool": "wait_long", "args": {"seconds": 1}}, {"text": "late"}]
    agent = arm(harness.wrap(fake(tmp_path, script), id="waits"), [wait_long])
    handle = asyncio.create_task(agent.run("x", user="u1"))
    await asyncio.wait_for(waiting.wait(), 30)
    [run] = [s async for s in harness.runs.iterate(agent_id="waits")]
    await agent.cancel(run.run_id, reason="enough")
    result = await handle
    assert result.status is RunStatus.CANCELLED
    await asyncio.sleep(1.5)  # past when the call would have ended
    assert waited == []  # cut short with the run, never finished after it


# --------------------------------------------------------------------------- failures
async def test_a_process_that_ends_mid_turn_fails_the_run_with_its_stderr(
    harness: Harness, tmp_path: Path
) -> None:
    agent = harness.wrap(fake(tmp_path, [{"crash": "out of memory"}]), id="crash")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "out of memory" in result.error.message and "exit 3" in result.error.message


def wait_times_out_as_the_process_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every wait for an agent's process ends with the process gone and its pipes closed, yet
    raises ``TimeoutError``: asyncio's race when a timed wait's deadline falls in the moment
    the process ends (the wait's wake-up is lost to the timeout). The ACP SDK's shutdown then
    signals the process, and asyncio raises ``ProcessLookupError``: it is already gone."""
    real = asyncio.subprocess.Process.wait

    async def gone(self: asyncio.subprocess.Process) -> int:
        await real(self)
        raise TimeoutError

    monkeypatch.setattr(asyncio.subprocess.Process, "wait", gone)


async def test_a_process_already_gone_when_the_attempt_stops_it_keeps_the_answer(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait_times_out_as_the_process_ends(monkeypatch)
    agent = harness.wrap(fake(tmp_path, [{"text": "done"}]), id="gone")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == "done"


async def test_a_process_already_gone_when_a_failed_attempt_stops_it_keeps_the_failure(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait_times_out_as_the_process_ends(monkeypatch)
    agent = harness.wrap(fake(tmp_path, [{"crash": "out of memory"}]), id="crash")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "out of memory" in result.error.message and "exit 3" in result.error.message


@pytest.mark.parametrize("said", ["Half of it so far.", None])
async def test_a_turn_ended_at_the_step_limit_ends_partial_with_an_answer(
    harness: Harness, tmp_path: Path, said: str | None
) -> None:
    """Kimi ends a turn at its step limit with no answer, the reason only in its log. The run
    ends ``PARTIAL``, its error ``BUDGET_EXCEEDED`` naming the limit and a ``step_limit``
    warning on the stream. With what the agent said before the limit, that is its answer and
    nothing more is asked (no paid last answer). With nothing said, the run would end with no
    answer at all: the agent is asked once more, in the same session, to answer without tools
    (``budget.TURN_LIMIT``) — as ``ReAct``'s step limit asks its model."""
    from trellis.harness.budget import BUDGET_EXCEEDED, TURN_LIMIT

    script = [
        *([{"text": said}] if said else []),
        {"step_limit": 1},
        {"text": "what I found so far"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="limited")
    events = [e async for e in agent.stream("work", user="u1")]
    warned = [e.data["message"] for e in events if e.data.get("code") == "step_limit"]
    assert warned == ["stopped: the step limit (1) was reached"]
    done = events[-1]
    assert done.type is RunEventType.RUN_FINISHED and done.outcome is RunOutcome.PARTIAL
    answer = said or "what I found so far"
    assert done.data["result"] == answer
    prompts = recorded(tmp_path)["prompts"]
    assert len(prompts) == (1 if said else 2)  # asked once more only when nothing was said
    if not said:
        assert TURN_LIMIT.format(limit=1) in str(prompts[-1])
    result = await agent.run("work", user="u1")
    assert result.status is RunStatus.PARTIAL and result.answer == answer
    assert result.error is not None and result.error.code == BUDGET_EXCEEDED
    assert "step limit (1)" in result.error.message


async def test_a_turn_the_agent_could_not_finish_fails_the_run_with_the_agents_reason(
    harness: Harness, tmp_path: Path
) -> None:
    """Kimi answers a prompt whose turn failed (its provider answered with an error, past its
    own retries) ``end_turn``, like a finished one; why is in its log (``turn.ended``:
    ``failed``, with the error). The run fails with that reason — whether or not text came
    first — never ``SUCCESS`` with no answer or half of one."""
    error = {"code": "provider.api_error", "name": "APIStatusError", "details": {"statusCode": 500}}
    for n, script in enumerate(
        ([{"turn_fails": error}], [{"text": "half"}, {"turn_fails": error}])
    ):
        (tmp_path / str(n)).mkdir()
        done = await harness.wrap(fake(tmp_path / str(n), script), id=f"fails{n}").run(
            "x", user="u1"
        )
        assert done.status is RunStatus.ERROR and done.error is not None
        assert "provider.api_error" in done.error.message and "500" in done.error.message


async def test_a_kept_session_its_provider_refuses_for_good_is_left_and_the_turn_runs_again(
    harness: Harness, tmp_path: Path
) -> None:
    """A conversation's turn continues the session its agent kept. When that session's own
    request is refused and a retry will not help (Kimi Code reloads a tool call in a form its
    provider refuses: every turn of the session fails from then on), nothing of the turn has
    run: the harness leaves the session and runs the turn again in a new one, given the
    conversation as a worker that kept no session is — said as a warning, once. A refusal a
    retry may cure is the run's error, as before."""
    refused = {"code": "provider.api_error", "details": {"statusCode": 400}, "retryable": False}
    script = [{"text": "noted"}, {"stop": "end_turn"}, {"turn_fails": refused}]
    agent = harness.wrap(fake(tmp_path, script), id="refused")
    first = await agent.run("The locker code is 4242.", user="u1", thread="t1")
    assert first.status is RunStatus.SUCCESS and first.answer == "noted"
    events = [e async for e in agent.stream("What is the code?", user="u1", thread="t1")]
    done = events[-1]
    assert done.outcome is RunOutcome.SUCCESS and done.data["result"] == "noted", done
    kept = recorded(tmp_path)
    assert [s["how"] for s in kept["sessions"]] == ["new", "resume", "new"]
    assert "4242" in str(kept["prompts"][-1])  # the conversation, given to the new session
    warned = [e.data["message"] for e in events if e.data.get("code") == "session"]
    assert len(warned) == 1 and "new" in warned[0]

    (tmp_path / "may-pass").mkdir()
    passing = {"code": "provider.api_error", "details": {"statusCode": 500}, "retryable": True}
    script = [{"text": "noted"}, {"stop": "end_turn"}, {"turn_fails": passing}]
    other = harness.wrap(fake(tmp_path / "may-pass", script), id="may-pass")
    await other.run("The locker code is 4242.", user="u1", thread="t2")
    failed = await other.run("What is the code?", user="u1", thread="t2")
    assert failed.status is RunStatus.ERROR
    assert [s["how"] for s in recorded(tmp_path / "may-pass")["sessions"]] == ["new", "resume"]

    # a turn that ran a tool before its session was refused is not run again: that would
    # repeat the call
    (tmp_path / "ran").mkdir()
    script = [{"text": "noted"}, {"stop": "end_turn"}]
    script += [{"tool": "lookup", "args": {"topic": "tea"}}, {"turn_fails": refused}]
    worked = harness.wrap(fake(tmp_path / "ran", script), id="ran", without={"schedules"})
    worked = arm(worked, [lookup])
    await worked.run("The locker code is 4242.", user="u1", thread="t3")
    cut = await worked.run("Look up tea, then the code.", user="u1", thread="t3")
    assert cut.status is RunStatus.ERROR
    assert [s["how"] for s in recorded(tmp_path / "ran")["sessions"]] == ["new", "resume"]


async def test_the_agents_plan_is_an_event_of_the_run(harness: Harness, tmp_path: Path) -> None:
    """Kimi's todo list reaches an ACP client as a plan (``session/update`` ``plan``): its
    entries are the run's ``plan`` event, for a surface that shows what the agent is at."""
    agent = harness.wrap(fake(tmp_path, [{"plan": True}, {"text": "ok"}]), id="plans-its-work")
    events = [e async for e in agent.stream("x", user="u1")]
    plans = [e.data for e in events if e.data.get("name") == "plan"]
    assert plans == [
        {"name": "plan", "entries": [{"content": "look", "priority": "high", "status": "pending"}]}
    ]


async def test_a_refusal_fails_the_run(harness: Harness, tmp_path: Path) -> None:
    result = await harness.wrap(fake(tmp_path, [{"stop": "refusal"}]), id="no").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "refusal" in result.error.message


async def test_an_agent_that_needs_a_login_fails_saying_so(
    harness: Harness, tmp_path: Path
) -> None:
    target = fake(tmp_path, [{"text": "x"}], env={"FAKE_ACP_AUTH": "missing"})
    result = await harness.wrap(target, id="login").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "Authentication required" in result.error.message


@pytest.mark.parametrize("version", ["2.3.3", "2.0.9", ""])
async def test_a_kimi_outside_the_tested_range_is_refused(
    harness: Harness, tmp_path: Path, version: str
) -> None:
    target = fake(tmp_path, [{"text": "x"}], env={"FAKE_ACP_VERSION": version})
    result = await harness.wrap(target, id=f"v{version}").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    low, high = compat.BINARIES["kimi-code"]
    assert f"tested with kimi-code>={low},<{high}" in result.error.message
    assert compat.INSTALL["kimi-code"] in result.error.message


async def test_an_acp_agent_without_a_profile_runs_unchecked(
    harness: Harness, tmp_path: Path
) -> None:
    target = fake(tmp_path, [{"text": "plain"}], profile=None, env={"FAKE_ACP_VERSION": "9"})
    result = await harness.wrap(target, id="plain").run("x", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "plain"


# --------------------------------------------------------------------------- options
async def test_framework_options_add_env_and_servers_and_replace_the_rest(
    harness: Harness, tmp_path: Path
) -> None:
    server = {"name": "notes", "command": sys.executable, "args": ["-c", "pass"]}
    agent = harness.wrap(
        fake(tmp_path, [{"text": "ok"}], env={"KIMI_ONE": "1"}),
        id="opts",
        without={"schedules"},
        framework_options={"env": {"KIMI_TWO": "2"}, "mcp_servers": [server]},
    )
    await agent.run("x", user="u1", framework_options={"mode": "default"})
    record = recorded(tmp_path)
    env = record["started"][0]["env"]
    assert env["KIMI_ONE"] == "1" and env["KIMI_TWO"] == "2"
    assert [s["name"] for s in record["sessions"][0]["servers"]] == ["notes"]
    assert record["modes"] == ["default"]
    with pytest.raises(ConfigurationError, match="an ACP agent's run takes"):
        harness.wrap(fake(tmp_path, []), id="bad", framework_options={"command": ["x"]})
    with pytest.raises(ConfigurationError, match="the harness's own server"):
        harness.wrap(
            fake(tmp_path, []), id="bad2", framework_options={"mcp_servers": [{"name": "trellis"}]}
        )
    with pytest.raises(ConfigurationError, match="the command that starts it"):
        ACPAgent(command=())


async def test_the_teams_http_and_sse_servers_go_to_the_agent(
    harness: Harness, tmp_path: Path
) -> None:
    servers = (
        {"type": "http", "name": "web", "url": "http://127.0.0.1:9/mcp", "headers": []},
        {"type": "sse", "name": "old", "url": "http://127.0.0.1:9/sse", "headers": []},
    )
    await harness.wrap(
        fake(tmp_path, [{"text": "ok"}], mcp_servers=servers), id="srv", without={"schedules"}
    ).run("x", user="u1")
    assert [s["name"] for s in recorded(tmp_path)["sessions"][0]["servers"]] == ["web", "old"]


# --------------------------------------------------------------------------- the sandbox
async def test_a_sandboxed_run_works_in_the_sandbox_and_says_reads_are_not_isolated(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandboxes = FakeSandboxes()
    monkeypatch.setenv("FAKE_SECRET_TOKEN", "do-not-pass")
    outputs: list[tuple[str, Any]] = []
    observed = bridge.observed

    async def recording(runtime: Any, spec: Any, args: Any, outcome: Any, **more: Any) -> None:
        outputs.append((spec.name, outcome.output))
        await observed(runtime, spec, args, outcome, **more)

    monkeypatch.setattr(bridge, "observed", recording)
    script = [
        {"builtin": "Write", "args": {"path": "/w/notes.txt", "content": "a\nb\nc\n"}},
        {"builtin": "Read", "args": {"path": "/w/notes.txt", "line": 2, "limit": 1}},
        {"builtin": "Read", "args": {"path": "/w/notes.txt"}},
        {"builtin": "Bash", "args": {"command": "touch hello"}, "ask": True, "limit": 3},
        {"builtin": "Bash", "args": {"command": "sleep 9"}, "kill": True},
        {"text": "{last}"},
    ]
    target = fake(tmp_path, script, sandbox=sandbox(sandboxes), env={"FAKE_ACP_RAW": "1"})
    agent = harness.wrap(target, id="boxed")
    events = [e async for e in agent.stream("work", user="u1")]
    assert not [e for e in events if e.type is RunEventType.INTERRUPT]  # the sandbox's: unasked
    warned = [e for e in events if e.data.get("code") == "acp_sandbox"]
    assert warned and "reads are not isolated" in warned[0].data["message"]
    assert events[-1].error is None, events[-1].error
    [made] = sandboxes.made()
    assert made not in sandboxes.boxes  # deleted with its run
    assert any(c.startswith("mkdir -p ") for c in sandboxes.commands)
    assert "touch hello" in sandboxes.commands and "sleep 9" in sandboxes.killed
    record = recorded(tmp_path)
    assert record["initialize"][0]["terminal"] is True
    started = record["started"][0]
    assert "FAKE_SECRET_TOKEN" not in started["env"]  # no secret of the host's
    assert started["env"]["HOME"].endswith("/home")
    assert started["env"]["KIMI_CODE_HOME"].endswith("/home/.agent")
    assert started["cwd"].endswith("/work")
    ran = [t for t in record["terminal"] if "exit" in t]
    # the scripted sandbox echoes the env it was given (FAKE=1); its tail, cut to the limit
    assert ran[0] == {"exit": 0, "signal": None, "output": "E=1", "truncated": True}
    assert ran[1]["signal"] == "SIGKILL"
    assert not Path(started["cwd"]).exists()  # noqa: ASYNC240  # the run's directories go with it
    # a command run in the client's terminal is recorded with that terminal's output, whole
    # (the agent's last update of the call names the terminal and carries none); a killed
    # command said nothing
    assert [output for name, output in outputs if name == "Bash"] == [" FAKE=1", ""]


@pytest.mark.parametrize("boxed", [False, True], ids=["on-this-machine", "sandboxed"])
async def test_a_users_message_that_starts_with_a_slash_is_words_not_a_command_of_the_agents(
    harness: Harness, tmp_path: Path, boxed: bool
) -> None:
    """SEC. Over ACP a prompt whose first block starts with ``/`` is a command of the agent's,
    run with no model turn. A user's message is never one: in a sandboxed run with nothing
    before it too, where it would be the prompt's first block."""
    fields = {"sandbox": sandbox(FakeSandboxes())} if boxed else {}
    agent = harness.wrap(fake(tmp_path, [{"text": "answered"}], **fields), id="slash")
    done = await agent.run("/compact everything", user="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "answered"
    assert recorded(tmp_path)["prompts"][0].endswith("/compact everything")


async def test_without_a_sandbox_the_agents_file_requests_are_refused(tmp_path: Path) -> None:
    from acp import RequestError

    attempt = acp.Attempt(acp.ACPAdapter(), None, fake(tmp_path, []))  # type: ignore[arg-type]
    with pytest.raises(RequestError):
        await attempt.read_text_file("s", "/etc/passwd")


async def test_a_process_that_ends_before_it_answers_fails_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    target = fake(tmp_path, [], env={"FAKE_ACP_CRASH": "initialize"})
    result = await harness.wrap(target, id="dead").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "cannot start" in result.error.message and "exit 4" in result.error.message


async def test_an_error_the_agent_answers_the_prompt_with_fails_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    result = await harness.wrap(fake(tmp_path, [{"error": "bad state"}]), id="err").run(
        "x", user="u1"
    )
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "Internal error" in result.error.message


async def test_calls_made_together_and_updates_of_unknown_calls(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Reads made together are each recorded (their structured output as JSON); an update of a
    call the agent never started is passed over; the team's own MCP tool joins the tool set."""
    script = [
        {"orphan": True},
        {"parallel": ["Read", "Glob"]},
        {"builtin": "mcp__crm__find", "args": {"q": "ada"}},
        {"text": "{last}"},
    ]
    agent = memory_harness.wrap(fake(tmp_path, script), id="together")
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.SUCCESS
    await memory_harness.writes.drain()
    outputs = {
        c.body["tool"]: c.body["output"] for c in memory_service.calls if c.name == "record_tool"
    }
    assert outputs["Read"] == '{"read": "Read"}' and "Glob" in outputs
    assert "mcp__crm__find" in {s.name for s in await agent.adapter.native(agent.target)}


async def test_a_kept_result_is_read_through_the_harness_server(
    harness: Harness, tmp_path: Path
) -> None:
    script = [
        {"tool": "gone", "args": {}},
        {"tool": "read_file", "args": {"file_path": "/large_tool_results/x"}},
        {"text": "{last}"},
    ]
    result = await arm(harness.wrap(fake(tmp_path, script), id="pager"), [lookup]).run(
        "x", user="u1"
    )
    assert result.status is RunStatus.SUCCESS and "/large_tool_results/x" in str(result.answer)


async def test_an_acp_agent_harness_knows_nothing_of_runs_its_sub_agent_unmapped(
    harness: Harness, tmp_path: Path
) -> None:
    """Without a profile the harness knows no sub-agent tool: the call is the agent's own."""
    script = [{"agent": {"prompt": "go"}, "result": "went"}, {"text": "{last}"}]
    target = fake(tmp_path, script, profile=None)
    done = await harness.wrap(target, id="generic").run("x", user="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "went"
    assert await children(harness, done.run_id) == []


async def test_a_sandboxed_command_that_fails_or_outlives_the_run(
    harness: Harness, tmp_path: Path
) -> None:
    from trellis.contracts import ToolError

    sandboxes = FakeSandboxes()
    sandboxes.crash = ToolError("the sandbox is gone", source="tools")
    script = [
        {"builtin": "Read", "args": {"path": "/w/none.txt"}},
        {"builtin": "Bash", "args": {"command": "crash"}},
        {"builtin": "Bash", "args": {"command": "sleep 30"}},
        {"text": "never"},
    ]
    target = fake(
        tmp_path, script, sandbox=sandbox(sandboxes), profile=None, env={"FAKE_ACP_RAW": "1"}
    )
    result = await harness.wrap(target, id="boxed2", run_timeout=3.0).run("x", user="u1")
    assert result.status is RunStatus.TIMEOUT
    record = recorded(tmp_path)
    [crashed] = [t for t in record["terminal"] if "exit" in t]
    assert crashed["signal"] == "SIGKILL" and "the sandbox is gone" in crashed["output"]
    assert "HOME" in record["started"][0]["env"] and "KIMI_CODE_HOME" in record["started"][0]["env"]
    assert "sleep 30" in sandboxes.killed  # the command still running when the run ended


# --------------------------------------------------------------------------- skills
async def test_the_runs_skills_are_in_kimis_skills_folder_and_its_login_is_linked(
    harness: Harness, tmp_path: Path
) -> None:
    from tests.support.skills import SkillStore
    from trellis.skills import combine

    store = SkillStore()
    store.put(
        "sql-review",
        "Check every JOIN.",
        description="Reviews SQL.",
        files={"rules.md": "No SELECT *."},
    )
    usual = tmp_path / "home"
    usual.mkdir()
    (usual / "config.toml").write_text("default_model = 'k2'\n")
    (usual / "AGENTS.md").write_text("Answer in metric units.\n")
    for name in ("sql-review", "tone"):  # Kimi's own skills: one of the run's names, one not
        (usual / "skills" / name).mkdir(parents=True)
        (usual / "skills" / name / "SKILL.md").write_text(f"Kimi's own {name}.")
    target = fake(tmp_path, [{"skills": True}, {"text": "ok"}], **combine(store).kimi())
    result = await harness.wrap(target, id="skilled").run("review", user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    [seen] = recorded(tmp_path)["skills"]
    assert "Check every JOIN." in seen["files"]["sql-review/SKILL.md"]  # the run's, of the two
    assert seen["files"]["sql-review/rules.md"] == "No SELECT *."
    assert seen["names"] == ["sql-review", "tone"]  # its own beside the run's
    # its own configuration (no credentials kept here), its instructions, and its sessions:
    # kept in its user's Kimi home, where the conversation's next turn finds the one this run
    # ended in
    assert seen["linked"] == ["AGENTS.md", "config.toml", "sessions"]
    assert (usual / "skills" / "tone" / "SKILL.md").is_file()  # the run's home gone, its own kept
    assert (kimi_home_of("ada") / "sessions").is_dir() and not (usual / "sessions").exists()
    started = recorded(tmp_path)["started"][0]["env"]["KIMI_CODE_HOME"]
    gone = not Path(started).exists()  # noqa: ASYNC240
    assert started != str(usual) and gone  # the run's, removed with it


async def test_kimis_skills_folder_holds_each_tenants_and_users_own_skills(
    memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """SEC: the skills folder of a Kimi run's home holds the skills of its tenant and user."""
    from tests.support.skills import SkillStore, kept_apart
    from trellis import Settings
    from trellis.skills import combine

    memory_service.tenant = None  # a platform key: each run names its tenant
    harness = Harness(config=Settings(host_execution="allow"), memory=memory_service.client())

    async def seen(store: SkillStore, tenant: str, user: str) -> str:
        folder = tmp_path / f"{tenant}-{user}"
        await asyncio.to_thread(folder.mkdir)
        target = fake(folder, [{"skills": True}, {"text": "ok"}], **combine(store).kimi())
        result = await harness.wrap(target, id=f"k-{tenant}-{user}").run(
            "go", user=user, tenant=tenant
        )
        assert result.status is RunStatus.SUCCESS, result.error
        return recorded(folder)["skills"][0]["files"]["sql-review/SKILL.md"]

    await kept_apart(seen)
    await harness.aclose()


# --------------------------------------------------------------------------- the rest of ACP
async def test_every_other_update_and_an_extension_pass_through_a_turn(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """What ACP has beside text, plans and tool calls — a thought, a message chunk that is not
    text, the agent's commands, its mode, its usage, the session's title — and its extension
    points (a notification; a request, which a client with no extensions refuses) change
    nothing of the turn: the answer is the agent's text, the stream carries only it, the
    harness tool called after them is recorded, and the turn is written to memory."""
    script = [
        {"updates": True},
        {"tool": "lookup", "args": {"topic": "tides"}},
        {"text": "so: {last}"},
    ]
    agent = arm(memory_harness.wrap(fake(tmp_path, script), id="updates"), [lookup])
    events = [e async for e in agent.stream("what of the tides?", user="u1")]
    await memory_harness.writes.drain()
    assert events[-1].type is RunEventType.RUN_FINISHED
    assert events[-1].data["result"] == "so: facts about tides"
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(deltas) == "so: facts about tides"  # no thought, no link
    record = recorded(tmp_path)
    assert record["extensions"] == [{"refused": -32601}]  # method not found: none served
    assert "errors" not in record
    assert memory_service.context_text in record["prompts"][0]
    tools = [c.body["tool"] for c in memory_service.named("record_tool")]
    assert tools == ["lookup"]
    said = [
        (m["role"], m["content"])
        for call in memory_service.named("messages")
        for m in call.body["messages"]
        if m["role"] in ("USER", "ASSISTANT") and m["content"]
    ]
    assert said[0] == ("USER", "what of the tides?")
    assert said[-1] == ("ASSISTANT", "so: facts about tides")


async def test_an_acp_agent_the_harness_has_no_profile_of_has_its_own_calls_asked_about(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Any ACP agent (no profile: the harness knows none of its tools): a call it asks
    permission for that declares nothing is governance's question, once; one the catalog says
    is a write is asked about once, as the administrator's catalog says (its reason the catalog's);
    each resume continues the agent's session, and a call it runs without asking is recorded
    as it reports it."""
    script = [
        {"builtin": "Peek", "args": {"at": "prod"}, "output": "3 pods"},
        {"builtin": "Deploy", "args": {"to": "prod"}, "ask": True, "output": "deployed"},
        {"builtin": "Notify", "args": {"who": "ops"}, "ask": True, "output": "told ops"},
        {"text": "done: {last}"},
    ]
    memory_service.catalog = {"Notify": {"side_effects": "write"}}
    agent = memory_harness.wrap(fake(tmp_path, script, profile=None), id="generic-asks")
    first = await agent.run("ship it", user="u1")
    assert first.status is RunStatus.PAUSED and first.interrupt is not None
    assert first.interrupt.tool_call is not None
    assert (first.interrupt.tool_call.tool, first.interrupt.tool_call.args) == (
        "Deploy",
        {"to": "prod"},
    )
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="u1")
    assert second.status is RunStatus.PAUSED and second.interrupt is not None
    assert second.interrupt.tool_call is not None and second.interrupt.tool_call.tool == "Notify"
    assert "has not declared" in first.interrupt.question
    assert second.interrupt.question == (
        "Approve Notify? Notify is write. "
        "Allow once, allow for this conversation, always allow, or deny."
    )
    done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: told ops"
    record = recorded(tmp_path)
    assert [s["how"] for s in record["sessions"]] == ["new", "resume", "resume"]
    answers = [(p["title"], p["answer"]) for p in record["permissions"]]
    assert answers.count(("Deploy", "approve_once")) == 1
    assert answers.count(("Notify", "approve_once")) == 1
    await memory_harness.writes.drain()
    calls = [(c.body["tool"], c.body["status"]) for c in memory_service.named("record_tool")]
    assert calls == [("Peek", "ok"), ("Deploy", "ok"), ("Notify", "ok")]


async def test_a_call_governance_and_the_agent_both_ask_about_is_asked_once(
    harness: Harness, tmp_path: Path
) -> None:
    """A tool of the agent's own MCP server declares nothing, so governance asks about its
    call, and the agent asks permission for it too: the person is asked once, and the approved
    call runs once."""
    script = [
        {"builtin": "mcp__crm__send", "args": {"to": "ada"}, "ask": True, "output": "sent"},
        {"text": "done: {last}"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="asked-once")
    asked = await agent.run("tell ada", user="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.question.startswith("Approve mcp__crm__send?")
    done = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: sent"
    answers = [p["answer"] for p in recorded(tmp_path)["permissions"]]
    assert answers == ["cancelled", "approve_once"]  # stopped to ask, then let through
