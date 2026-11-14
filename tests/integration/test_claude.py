"""Claude Agent SDK: the real SDK drives a scripted stand-in for the Claude Code CLI
(``tests/support/fake_claude_cli.py``) through ``cli_path`` — subprocess, control protocol and
in-process MCP tool calls included."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny

from tests.support.arm import arm, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService, pushed
from trellis import Deny, Harness, Hooks, Rewrite, current, tool
from trellis.contracts import (
    Option,
    RunEventType,
    RunOutcome,
    RunStatus,
    ToolCall,
    ToolSpec,
)
from trellis.harness.adapters.claude import (
    BUILTIN_SIDE_EFFECTS,
    RESUMED_ASKED,
    RESUMED_CALL,
    RESUMED_DENIED,
    native_spec,
)
from trellis.harness.agent import NARROW_FROM
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.journal import content_key
from trellis.harness.sandbox import DELIVERS_HERE
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert.claude import _schema

#: These tests are about what a write does once it runs, not whether it asks.
pytestmark = pytest.mark.usefixtures("writes_unasked")

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
refunds: list[str] = []


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return f"refunded {order}"


def options(tmp_path: Path, script: list[dict[str, Any]], **fields: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
            "FAKE_CLAUDE_SESSIONS": str(tmp_path),
            "FAKE_CLAUDE_BUILTINS": str(tmp_path / "builtins.jsonl"),
        },
        **fields,
    )


def built_ins(tmp_path: Path) -> list[dict[str, Any]]:
    """What Claude Code's own tools ran."""
    log = tmp_path / "builtins.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def started_with(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "cli.json").read_text())


async def test_a_query_answers_and_calls_harness_tools(harness: Harness, tmp_path: Path) -> None:
    refunds.clear()
    target = options(
        tmp_path,
        [{"tool": "refund", "args": {"order": "o1"}}, {"text": "refunded o1"}],
        system_prompt="You refund.",
    )
    agent = arm(
        harness.wrap(target, id="refunds", without={"schedules"}),
        [tool(refund.fn, side_effects="write")],
    )
    result = await agent.run("refund o1", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded o1"
    assert refunds == ["o1"]
    cli = started_with(tmp_path)
    # read_file is listed from the start: the CLI lists a server's tools once
    assert cli["tools"] == ["refund", "read_file", "query_result"] and cli["allowed_tools"] is None
    # what Claude Code would ask its user, it asks the harness (its can_use_tool)
    assert cli["permission_prompt_tool"] == "stdio"
    # the team's, then what Claude is told of its folder's outputs/ (the user's)
    assert cli["system_prompt"] == f"You refund.\n\n{DELIVERS_HERE}"
    assert target.mcp_servers == {}  # the team's options are untouched


asked: list[dict[str, Any]] = []


class NoArguments:
    """An MCP server's tool that takes no argument: its schema is ``{"type": "object"}``."""

    async def resolve(self) -> list[Tool]:
        spec = ToolSpec(
            name="now",
            description="The time.",
            input_schema={"type": "object"},
            side_effects="read",
        )

        async def run(args: dict[str, Any]) -> Any:
            asked.append(args)
            return "noon"

        return [Tool(spec, run, feature="mcp")]


async def test_the_turns_tool_calls_and_results_are_in_its_transcript(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Claude reports its calls in its own messages (``ToolUseBlock``, ``ToolResultBlock``):
    the transcript memory keeps has each call by the harness tool's name and its result, with
    its run record, so a later chat's steps list says which tool answered (explain-08 and
    histx on Claude, the all-adapter run, said "the record holds only the answer")."""
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o4"}}, {"text": "refunded o4"}]
    target = options(tmp_path, script)
    agent = arm(memory_harness.wrap(target, id="refunds"), [tool(refund.fn, side_effects="write")])
    assert (await agent.run("refund o4", user="u1")).answer == "refunded o4"
    await memory_harness.writes.drain()
    [batch] = memory_service.named("messages")
    sent = batch.body["messages"]
    assert [(m["role"], m["content"]) for m in sent] == [
        ("USER", "refund o4"),
        ("ASSISTANT", ""),
        ("TOOL", "refunded o4"),
        ("ASSISTANT", "refunded o4"),
    ]
    [made] = sent[1]["tool_calls"]
    assert made["name"] == "refund" and made["arguments"] == {"order": "o4"}
    assert sent[2]["tool_call_id"] == made["id"]
    assert sent[2]["custom_metadata"]["run_record"]["status"] == "ok"


async def test_a_tool_whose_schema_declares_no_properties_takes_no_argument(
    harness: Harness, tmp_path: Path
) -> None:
    """The SDK reads a schema with no ``properties`` as a map of argument names to types:
    the tool is offered as a JSON schema of an object without arguments."""
    target = options(tmp_path, [{"tool": "now", "args": {}}, {"text": "it is noon"}])
    asked.clear()
    result = await arm(harness.wrap(target, id="clock"), [NoArguments()]).run("time?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "it is noon"
    assert asked == [{}]  # offered with no argument, called with none
    [now] = await NoArguments().resolve()
    assert _schema(now) == {"type": "object", "properties": {}}
    kept = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    assert _schema(replace(now, spec=now.spec.model_copy(update={"input_schema": kept}))) == kept


async def test_a_tool_result_over_a_mebibyte_reaches_claude_as_a_preview(
    harness: Harness, tmp_path: Path
) -> None:
    """A tool's result comes back from the CLI in one message, which the SDK reads into a 1 MiB
    buffer unless its options say more (the harness's say enough for a team's own MCP server's
    result, or a built-in's): a harness tool's large result is cut before, so even a team's own
    1 MiB buffer holds its preview."""

    @tool(side_effects="read")
    def export(rows: int) -> str:
        """Export a report."""
        return "x" * (1024 * 1024 + 1)

    script = [{"tool": "export", "args": {"rows": 3}}, {"text": "{last}"}]
    own = options(tmp_path, script, max_buffer_size=1024 * 1024)
    result = await arm(harness.wrap(own, id="small"), [export]).run("export", user="u")
    assert result.status is RunStatus.SUCCESS and "/large_tool_results/" in result.answer
    assert len(result.answer) < 25_000


async def test_an_approval_stops_the_cli_and_a_resume_continues_its_session(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    agent = arm(
        harness.wrap(
            options(tmp_path, [{"tool": "refund", "args": {"order": "o2"}}, {"text": "done"}]),
            id="refunds",
        ),
        [refund],
    )
    paused = await agent.run("refund o2", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and refunds == []
    session = started_with(tmp_path)
    assert session["resume"] is None and session["prompt"] == "refund o2"
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "done"
    assert refunds == ["o2"]
    resumed = started_with(tmp_path)
    # told which call the person answered about, by the name Claude calls it: the session
    # holds that call's result as "waiting", which is not "no result yet"
    assert resumed["resume"] is not None
    assert resumed["prompt"] == RESUMED_CALL.format(tool="mcp__trellis__refund")


async def test_a_query_that_pauses_counts_what_its_steps_spent(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paused query ends before its ``ResultMessage``: the usage its assistant messages
    carried is on the run's budget, and the query that finishes counts its result's, as
    before."""
    from trellis.harness import budget

    counted: list[dict[str, Any]] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (counted.append(kw), account(run, **kw))
    )
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o3"}}, {"text": "done"}]
    agent = arm(harness.wrap(options(tmp_path, script, model="m"), id="refunds"), [refund])
    paused = await agent.run("refund o3", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    step = {"model": "m", "input_tokens": 10, "output_tokens": 5, "cached": 0, "cost": None}
    assert counted == [step]
    counted.clear()
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    assert [(c["input_tokens"], c["output_tokens"]) for c in counted] == [(20, 10)]


async def test_a_denied_call_resumes_the_session_told_it_will_not_run(
    harness: Harness, tmp_path: Path
) -> None:
    """After a deny the call does not run: the resumed session is told so, not told to call
    again and expect its result."""
    refunds.clear()
    agent = arm(
        harness.wrap(
            options(
                tmp_path, [{"tool": "refund", "args": {"order": "o3"}}, {"text": "not refunded"}]
            ),
            id="refunds-denied",
        ),
        [refund],
    )
    paused = await agent.run("refund o3", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    finished = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and refunds == []
    assert started_with(tmp_path)["prompt"] == RESUMED_DENIED.format(tool="mcp__trellis__refund")


@tool(side_effects="read")
async def pick_warehouse() -> str:
    """Ask which warehouse ships."""
    runtime = current()
    assert runtime is not None
    return f"ships from {await runtime.ask('Which warehouse?')}"


async def test_a_question_a_tool_asked_resumes_the_session_naming_its_call(
    harness: Harness, tmp_path: Path
) -> None:
    """The session holds the call the question was asked in as waiting for a person: told
    only to repeat a call with no result yet, Claude asked the person again with another tool
    (#203). It is told which call to make again, with its arguments: a session whose
    transcript ends before that call no longer holds them."""
    script = [{"tool": "pick_warehouse", "args": {}}, {"text": "{last}"}]
    agent = arm(harness.wrap(options(tmp_path, script), id="shipping"), [pick_warehouse])
    asked = await agent.run("where does it ship from?", user="u1")
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.interrupt.tool_call is None
    done = await agent.resume(asked.interrupt.interrupt_id, "answer", answer="Leeds", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "ships from Leeds"
    told = RESUMED_ASKED.format(tool="mcp__trellis__pick_warehouse", args="{}")
    assert started_with(tmp_path)["prompt"] == told


async def test_claudes_question_to_its_user_is_the_runs_question(
    harness: Harness, tmp_path: Path
) -> None:
    """Claude's ``AskUserQuestion`` is asked as the run's (as Kimi's): each question a choice,
    several picks with ``multiSelect``, the option labelled "(Recommended)" recommended with
    its description as why; Claude reads the answers by question, as its own UI gives them."""
    questions = [
        {
            "question": "Which regions?",
            "header": "Regions",
            "multiSelect": True,
            "options": [
                {"label": "north (Recommended)", "description": "your last choice"},
                {"label": "south", "description": "slower"},
            ],
        },
        {
            "question": "Which format?",
            "header": "Format",
            "multiSelect": False,
            "options": [{"label": "CSV"}, {"label": "PDF"}],
        },
    ]
    script = [{"builtin": "AskUserQuestion", "args": {"questions": questions}}, {"text": "ok"}]
    agent = harness.wrap(options(tmp_path, script), id="asker")
    first = await agent.run("set it up", user="u1")
    assert first.interrupt is not None and first.interrupt.tool_call is None
    assert first.interrupt.multiple and first.interrupt.options[0] == Option(
        value="north (Recommended)", description="your last choice", recommended=True
    )
    picked = ["north (Recommended)", "south"]
    second = await agent.resume(
        first.interrupt.interrupt_id, "answer", answer=picked, reviewer="u1"
    )
    assert second.interrupt is not None and second.interrupt.options == ["CSV", "PDF"]
    done = await agent.resume(second.interrupt.interrupt_id, "answer", answer="CSV", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    [asked] = built_ins(tmp_path)
    assert asked["args"]["answers"] == {
        "Which regions?": "north (Recommended), south",
        "Which format?": "CSV",
    }


async def test_claudes_question_is_asked_once_though_it_words_it_anew(
    harness: Harness, tmp_path: Path
) -> None:
    """The resumed session is told to make its paused call again, and words its
    ``AskUserQuestion`` differently: it is the question the person answered, not a new one, and
    Claude reads the answer under the question that was asked."""

    def asking(question: str, *labels: str) -> dict[str, Any]:
        options = [{"label": label} for label in labels]
        return {"questions": [{"question": question, "header": "Format", "options": options}]}

    step = {
        "builtin": "AskUserQuestion",
        "args": asking("Which format?", "CSV", "PDF"),
        "again": asking("What format do you want?", "PDF", "CSV"),
    }
    agent = harness.wrap(options(tmp_path, [step, {"text": "ok"}]), id="asks-once")
    paused = await agent.run("export it", user="u1")
    assert paused.interrupt is not None and paused.interrupt.options == ["CSV", "PDF"]
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="PDF", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.interrupt
    [asked] = built_ins(tmp_path)
    assert asked["args"]["answers"] == {"Which format?": "PDF"}


async def test_built_in_tools_are_governed_and_not_run_again_after_a_pause(
    harness: Harness, tmp_path: Path
) -> None:
    """A ``Bash`` command Claude Code asks about asks a person; approved, it runs once — the
    resumed session does not run it again when the run pauses later, on a harness tool. A
    command that only reads runs without asking."""
    refunds.clear()
    script = [
        {"builtin": "Read", "args": {"file_path": "orders.csv"}},
        {"builtin": "Bash", "args": {"command": "ls"}},
        {"builtin": "Bash", "args": {"command": "rm -rf build"}},
        {"tool": "refund", "args": {"order": "o3"}},
        {"text": "done: {last}"},
    ]
    agent = arm(harness.wrap(options(tmp_path, script), id="ops"), [refund])
    asked = await agent.run("tidy up, then refund o3", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.args == {"command": "rm -rf build"}
    assert asked.interrupt.question == "Approve Bash? Claude Code asks before it runs this call."
    assert [b["tool"] for b in built_ins(tmp_path)] == ["Read", "Bash"]  # ls: not asked
    again = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u1")
    assert started_with(tmp_path)["prompt"] == RESUMED_CALL.format(tool="Bash")  # a built-in
    assert again.interrupt is not None and again.interrupt.tool_call is not None
    assert again.interrupt.tool_call.tool == "refund"
    done = await agent.resume(again.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "done: refunded o3"
    assert [b["args"] for b in built_ins(tmp_path)][1:] == [
        {"command": "ls"},
        {"command": "rm -rf build"},
    ]  # each once
    assert refunds == ["o3"]


async def test_a_read_outside_the_working_folder_asks_a_person(
    harness: Harness, tmp_path: Path
) -> None:
    """A command that only reads runs unasked inside the folder Claude Code works in; a file
    anywhere else on the machine (``/etc/passwd``) changes nothing, but asks a person."""
    script = [
        {"builtin": "Bash", "args": {"command": "ls"}},
        {"builtin": "Bash", "args": {"command": "cat /etc/passwd"}},
        {"text": "done"},
    ]
    agent = harness.wrap(options(tmp_path, script), id="ops")
    asked = await agent.run("look around", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.args == {"command": "cat /etc/passwd"}
    assert [b["args"] for b in built_ins(tmp_path)] == [{"command": "ls"}]  # ls: not asked


async def test_a_read_after_a_cd_is_placed_from_where_the_shell_may_be(
    harness: Harness, tmp_path: Path
) -> None:
    """Claude Code's shell keeps the folder a ``cd`` left it in: once it went to a folder the
    options add (``add_dirs``: readable, so the ``cd`` runs unasked), a relative read is placed
    from there too, and one that leads out of what the run may read from there asks."""
    run, kept = tmp_path / "a" / "run", tmp_path / "b" / "kept"
    run.mkdir(parents=True)
    kept.mkdir(parents=True)
    script = [
        {"builtin": "Bash", "args": {"command": f"cd {kept}"}},
        {"builtin": "Bash", "args": {"command": "cat x.csv"}},
        {"builtin": "Bash", "args": {"command": "cat ../run/x.csv"}},  # b/run, not a/run
        {"text": "done"},
    ]
    target = options(tmp_path, script, cwd=str(run), add_dirs=[str(kept)])
    asked = await harness.wrap(target, id="ops").run("look around", user="u1")
    assert asked.interrupt is not None and asked.interrupt.tool_call is not None
    assert asked.interrupt.tool_call.args == {"command": "cat ../run/x.csv"}
    assert [b["args"]["command"] for b in built_ins(tmp_path)] == [f"cd {kept}", "cat x.csv"]


async def test_a_command_claudes_own_rules_allow_runs_without_asking_a_person(
    harness: Harness, tmp_path: Path
) -> None:
    """A team's allowlist is Claude Code's own (``allowed_tools``, its settings): a command it
    lets run is not asked about by the harness either."""
    script = [{"builtin": "Bash", "args": {"command": "touch a.txt"}}, {"text": "said: {last}"}]
    agent = harness.wrap(options(tmp_path, script, allowed_tools=["Bash"]), id="allowed")
    result = await agent.run("make a.txt", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "said: Bash ran"
    assert built_ins(tmp_path) == [{"tool": "Bash", "args": {"command": "touch a.txt"}}]


async def test_a_command_a_person_rejected_is_not_run_and_claude_reads_why(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"builtin": "Bash", "args": {"command": "rm -rf build"}}, {"text": "said: {last}"}]
    agent = harness.wrap(options(tmp_path, script), id="careful")
    asked = await agent.run("clean", user="u1")
    assert asked.interrupt is not None
    done = await agent.resume(asked.interrupt.interrupt_id, "reject", reviewer="u1")
    assert done.answer == "said: Bash was not run: the approver rejected it"
    assert built_ins(tmp_path) == []


@pytest.mark.parametrize(
    "rule", [Rule(risk="irreversible"), Rule(risk="write", approve_when="true")], ids=str
)
async def test_a_team_that_wants_every_shell_command_approved_says_so_in_the_catalog(
    tmp_path: Path, rule: Rule
) -> None:
    """The tool catalog's word on ``Bash`` (its risk, or an ``approve_when``) is the team's:
    then a command that only reads asks a person too, before Claude Code is asked."""
    script = [{"builtin": "Bash", "args": {"command": "ls"}}, {"text": "said: {last}"}]
    async with with_governance(Harness(), Governance(FakeCatalog({"Bash": rule}))) as h:
        asked = await h.wrap(options(tmp_path, script), id="strict").run("look", user="u1")
        assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
        call = asked.interrupt.tool_call
        assert call is not None and (call.tool, call.args) == ("Bash", {"command": "ls"})
        assert built_ins(tmp_path) == []


async def test_the_teams_own_callback_is_asked_only_what_claude_code_asks(
    harness: Harness, tmp_path: Path
) -> None:
    """``can_use_tool`` is Claude Code's question to its user: a call its rules let run (here
    ``allowed_tools``) never reaches the team's callback, as without the harness."""
    asked: list[str] = []

    async def own(name: str, args: dict[str, Any], context: Any) -> Any:
        asked.append(name)
        return PermissionResultAllow()

    script = [
        {"builtin": "Write", "args": {"file_path": "a.txt"}},
        {"builtin": "Bash", "args": {"command": "touch b.txt"}},
        {"text": "said: {last}"},
    ]
    target = options(tmp_path, script, can_use_tool=own, allowed_tools=["Write"])
    result = await harness.wrap(target, id="own").run("work", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "said: Bash ran"
    assert asked == ["Bash"]
    assert [b["tool"] for b in built_ins(tmp_path)] == ["Write", "Bash"]


async def test_what_claude_saves_in_its_outputs_folder_is_kept_for_its_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file Claude writes in ``outputs/`` of its working folder is the user's, as a
    sandbox's ``outputs/`` is (``sandbox.delivered``): on success it moves to the file store,
    named in the journal and announced; the rest of the folder stays; Claude is told so."""
    import trellis.harness.sandbox as boxes
    from trellis import Settings

    monkeypatch.setattr(boxes, "OUTPUT_MAX_BYTES", 8)
    work = tmp_path / "work"
    (work / "outputs" / "sub").mkdir(parents=True)
    for name, text in (("report.csv", "n\n1\n"), ("big.csv", "x" * 9), ("sub/x.txt", "x")):
        (work / "outputs" / name).write_text(text)
    (work / "notes.txt").write_text("scratch")
    script = [{"text": "saved"}]
    harness = Harness(config=Settings(files_url=f"memory://claude-out-{id(tmp_path)}"))
    agent = harness.wrap(options(tmp_path, script, cwd=str(work)), id="saver")
    events = [e async for e in agent.stream("save it", user="u1")]
    assert events[-1].outcome is RunOutcome.SUCCESS
    files = [e.data for e in events if e.data.get("name") == "file"]
    assert [(f["file"], f["size"], f["source"]) for f in files] == [
        ("report.csv", 4, "claude_agent_sdk")
    ]
    warned = [e.data["message"] for e in events if e.data.get("code") == "sandbox"]
    assert warned == ["the output big.csv was not kept: 9 bytes, past the 8 kept"]
    assert not (work / "outputs" / "report.csv").exists()  # moved to the file store
    assert (work / "outputs" / "big.csv").exists() and (work / "notes.txt").exists()
    assert (work / "outputs" / "sub" / "x.txt").exists()
    assert started_with(tmp_path)["system_prompt"] == boxes.DELIVERS_HERE  # none of its own
    await harness.aclose()


async def test_a_built_in_tool_goes_through_the_hooks_then_the_teams_own_callback(
    harness: Harness, tmp_path: Path
) -> None:
    asked: list[tuple[str, dict[str, Any]]] = []

    async def own(name: str, args: dict[str, Any], context: Any) -> Any:
        asked.append((name, args))
        if name == "Write":
            return PermissionResultDeny(message="no writes on Fridays")
        return PermissionResultAllow()

    class Guard(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
            if call.tool == "Grep":
                return Deny("no searching here")
            if call.tool == "Edit":
                return Rewrite({**call.args, "file_path": "safe.txt"})
            return None

    script = [
        {"builtin": "Grep", "args": {"pattern": "x"}},
        {"builtin": "Edit", "args": {"file_path": "/etc/hosts"}},
        {"builtin": "Write", "args": {"file_path": "a.txt"}},
        {"text": "said: {last}"},
    ]
    target = options(tmp_path, script, can_use_tool=own)
    agent = harness.wrap(target, id="careful", hooks=[Guard()])
    events = [e async for e in agent.stream("work", user="u1")]
    assert events[-1].data["result"] == "said: no writes on Fridays"
    assert built_ins(tmp_path) == [{"tool": "Edit", "args": {"file_path": "safe.txt"}}]
    assert asked == [("Edit", {"file_path": "safe.txt"}), ("Write", {"file_path": "a.txt"})]
    notices = [e.data["tool"] for e in events if e.data.get("name") == "tool_notice"]
    assert notices == ["Edit", "Write"]  # writes are announced, as harness writes are


async def test_a_session_the_cli_no_longer_holds_is_run_again_from_its_prompt(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o4"}}, {"text": "done"}]
    agent = arm(
        harness.wrap(
            # files checkpointed on this machine's disk: the session is not kept in the journal
            options(tmp_path, script, enable_file_checkpointing=True),
            id="moved",
        ),
        [refund],
    )
    paused = await agent.run("refund o4", user="u1")
    assert paused.interrupt is not None
    for kept in (tmp_path / "fake-claude").glob("*.json"):
        kept.unlink()  # resumed on a machine whose CLI never held the session
    events: list[Any] = []
    record = await harness.runs.get(paused.run_id)
    assert record is not None
    resumed, resolution = await agent._resolution(
        paused.interrupt.interrupt_id, "approve", None, "u1", tenant=record.tenant_id
    )
    done = await agent._continue(resumed, resolution, events.append)
    assert done.status is RunStatus.SUCCESS and refunds == ["o4"]
    [warning] = [e for e in events if e.data.get("code") == "claude_session"]
    assert "No conversation found with session ID" in warning.data["message"]
    assert started_with(tmp_path)["prompt"] == "refund o4"  # from the prompt, against the journal


async def test_a_permission_prompt_tool_of_the_teams_own_is_refused(
    harness: Harness, tmp_path: Path
) -> None:
    target = options(tmp_path, [{"text": "x"}], permission_prompt_tool_name="mcp__ops__ask")
    result = await harness.wrap(target, id="own").run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "give your own permission check as can_use_tool" in result.error.message


async def test_the_context_goes_with_the_prompt_as_a_hook_adds_it(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    target = options(tmp_path, [{"text": "told: {context}"}], system_prompt="You help.")
    await memory_harness.wrap(target, id="helper").run("reach me how?", user="u1")
    started = started_with(tmp_path)
    assert started["system_prompt"] == f"You help.\n\n{DELIVERS_HERE}"  # the team's first
    # Claude's built-in tools are the agent's tools too, sorted: the context carries their hints
    rendered = pushed(memory_service.context_text + "\n\n## Tools\n- Agent (confidence 0.90)")
    assert started["context"] == rendered
    [asked] = memory_service.named("context")
    available = asked.body["tools"]["available"]
    assert available == sorted(available) and {"Read", "Bash", "Skill"} <= set(available)


async def test_without_memory_no_context_hook_is_registered(
    harness: Harness, tmp_path: Path
) -> None:
    target = options(tmp_path, [{"text": "told: {context}"}])
    result = await harness.wrap(target, id="helper").run("reach me how?", user="u1")
    assert result.answer == "told: " and started_with(tmp_path)["context"] == ""


def lookups(n: int) -> list[Any]:
    """``n`` read tools of the run's own, ``t0`` to ``t<n-1>``."""

    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return f"t{i}"

        return tool(lookup, name=f"t{i}", side_effects="read")

    return [make(i) for i in range(n)]


async def test_the_hints_narrow_the_tools_for_the_run(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path, narrowing: None
) -> None:
    memory_service.candidates = ["t2", "t5"]
    target = options(tmp_path, [{"text": "done"}])
    await arm(memory_harness.wrap(target, id="n", without={"schedules"}), lookups(6)).run(
        "x", user="u1"
    )
    assert started_with(tmp_path)["tools"] == [
        "t2",
        "t5",
        *MEMORY_TOOLS,
        "read_file",
        "query_result",
    ]


async def test_claudes_built_in_tools_do_not_count_toward_narrowing_the_runs_own(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Claude Code's built-ins are the framework's, not the run's: the tools the hints narrow
    from (``NARROW_FROM``) are the run's own. Counted with them, a run of a few tools of its
    own narrowed, and the one tool a question needed was not listed for the query (the
    break-and-fix suite: "find_sku isn't available to me in this session")."""
    memory_service.candidates = ["t2"]
    own = NARROW_FROM - len(BUILTIN_SIDE_EFFECTS) + 1  # with the built-ins, past the threshold
    target = options(tmp_path, [{"text": "done"}])
    await arm(memory_harness.wrap(target, id="n"), lookups(own)).run("x", user="u1")
    assert {f"t{i}" for i in range(own)} <= set(started_with(tmp_path)["tools"])
    assert "Bash" not in (started_with(tmp_path)["disallowed_tools"] or "").split(",")


async def test_streaming_carries_the_assistant_text(harness: Harness, tmp_path: Path) -> None:
    agent = harness.wrap(options(tmp_path, [{"text": "hello there"}]), id="s")
    events = [e async for e in agent.stream("hi", user="u1")]
    assert [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT] == [
        "hello there"
    ]


async def test_mcp_servers_the_team_configured_as_a_file_or_json_stay(
    harness: Harness, tmp_path: Path
) -> None:
    """``mcp_servers`` may be a path or a JSON string (the CLI's ``--mcp-config``): the harness
    adds its server beside the team's, never in place of them."""
    erp = {"type": "stdio", "command": "erp-mcp", "args": []}
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"erp": erp}}))
    script = [{"tool": "refund", "args": {"order": "o9"}}, {"text": "done"}]
    for configured in (config, str(config), json.dumps({"mcpServers": {"erp": erp}})):
        target = options(tmp_path, script, mcp_servers=configured)
        agent = arm(
            harness.wrap(target, id=f"mcp-{len(harness.agents)}"),
            [tool(refund.fn, side_effects="write")],
        )
        assert (await agent.run("refund o9", user="u1")).answer == "done"
        servers = json.loads(started_with(tmp_path)["mcp_config"])["mcpServers"]
        assert servers["erp"] == erp and servers["trellis"]["type"] == "sdk"


async def test_tools_whose_work_needs_the_cli_after_the_query_are_not_offered(
    harness: Harness, tmp_path: Path
) -> None:
    """The CLI runs for one attempt and ends with it. Claude Code's scheduled prompts
    (``CronCreate``) and wake-ups (``ScheduleWakeup``) fire only in a session that keeps
    running, so here they never would: they are disallowed, and Claude does not promise a
    schedule nothing keeps. The team's own ``disallowed_tools`` come first, kept."""
    target = options(tmp_path, [{"text": "ok"}], disallowed_tools=["WebFetch"])
    await harness.wrap(target, id="no-cron").run("run this every Monday", user="u1")
    disallowed = started_with(tmp_path)["disallowed_tools"].split(",")
    assert disallowed[0] == "WebFetch"
    assert {"CronCreate", "CronDelete", "CronList", "ScheduleWakeup"} <= set(disallowed)
    assert not {"Read", "Bash", "Task", "mcp__trellis__refund"} & set(disallowed)


async def test_workflows_and_messages_to_other_agents_are_not_offered(
    harness: Harness, tmp_path: Path
) -> None:
    """Claude Code's dynamic workflow (``Workflow``) runs its agents in the background with no
    task of their own, an agent's first call coming before anything names it; its
    ``SendMessage`` resumes a finished background subagent as a task of the call that first
    started it, or reaches another Claude session. No run of the harness can hold that work,
    so they are disallowed — with the agent-teams tools older CLIs have, whatever the team's
    own ``CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`` says (the harness sets no such switch)."""
    target = options(tmp_path, [{"text": "ok"}])
    target.env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
    await harness.wrap(target, id="no-workflow").run("fan this out to agents", user="u1")
    disallowed = set(started_with(tmp_path)["disallowed_tools"].split(","))
    assert {"Workflow", "SendMessage", "TeamCreate", "TeamDelete"} <= disallowed
    assert not {"Agent", "Task", "TaskStop", "ListAgents"} & disallowed


async def test_a_large_result_is_the_harness_s_to_keep_not_the_cli_s(
    harness: Harness, tmp_path: Path
) -> None:
    """Claude Code saves an MCP result over its own output limit (``MAX_MCP_OUTPUT_TOKENS``,
    25,000 tokens by default: about 50,000 characters of a CSV) to a file of its session and
    gives the model that path, which the harness's ``read_file`` and ``query_result`` do not
    read ("not found", the gate's Haiku run). The harness keeps a result over
    ``results.MAX_CHARS`` itself: the CLI's limit is raised above that, so the harness's
    keeping acts first; a team's own limit is kept."""
    from trellis.harness.adapters.claude import MCP_OUTPUT_TOKENS
    from trellis.harness.tools.results import MAX_CHARS

    assert MCP_OUTPUT_TOKENS * 1.5 >= MAX_CHARS  # a CSV is under 2 characters a token
    await harness.wrap(options(tmp_path, [{"text": "ok"}]), id="mcp-out").run("go", user="u1")
    assert started_with(tmp_path)["max_mcp_output_tokens"] == str(MCP_OUTPUT_TOKENS)
    own = options(tmp_path, [{"text": "ok"}])
    own.env["MAX_MCP_OUTPUT_TOKENS"] = "90000"
    await harness.wrap(own, id="mcp-own").run("go", user="u1")
    assert started_with(tmp_path)["max_mcp_output_tokens"] == "90000"


async def test_the_harness_tools_are_loaded_from_the_first_step(
    harness: Harness, tmp_path: Path
) -> None:
    """Claude Code defers an MCP server's tools behind its ``ToolSearch`` (the model sees their
    names only, and spends a step loading them) unless the server's config says ``alwaysLoad``.
    The harness's tools are the ones the run chose to offer (the hints' narrowing and
    ``tool_search`` are the harness's): its server is loaded; a team's own servers keep the
    CLI's deferral."""
    erp = {"type": "stdio", "command": "erp-mcp", "args": []}
    script = [{"tool": "refund", "args": {"order": "o2"}}, {"text": "done"}]
    target = options(tmp_path, script, mcp_servers={"erp": erp})
    agent = arm(harness.wrap(target, id="loaded"), [tool(refund.fn, side_effects="write")])
    assert (await agent.run("refund o2", user="u1")).answer == "done"
    servers = json.loads(started_with(tmp_path)["mcp_config"])["mcpServers"]
    assert servers["trellis"] == {"type": "sdk", "name": "trellis", "alwaysLoad": True}
    assert servers["erp"] == erp


async def test_at_its_turn_limit_a_query_ends_with_one_last_answer(
    harness: Harness, tmp_path: Path
) -> None:
    """A query that reaches ``max_turns`` ends ``error_max_turns``, with no answer. As a
    ``ReAct`` run at ``max_steps`` and an OpenAI Agents run at ``max_turns`` do, the run asks
    once more: its session resumed (Claude's own ``resume``), told the limit is reached
    (``budget.LIMIT``), one turn and no tool. Its answer is the run's, said as a
    ``step_limit`` warning."""
    from trellis.harness.budget import TURN_LIMIT

    refunds.clear()
    script = [
        {"tool": "refund", "args": {"order": "o1"}},
        {"tool": "refund", "args": {"order": "o2"}},
        {"text": "never"},
    ]
    target = options(tmp_path, script, max_turns=1)
    target.env["FAKE_CLAUDE_AT_LIMIT"] = "Refunded o1; o2 is left undone."
    agent = arm(harness.wrap(target, id="limited"), [tool(refund.fn, side_effects="write")])
    events = [e async for e in agent.stream("refund o1 and o2", user="u1")]
    done = events[-1]
    assert done.type is RunEventType.RUN_FINISHED and done.error is None
    said = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert said == ["Refunded o1; o2 is left undone."] and refunds == ["o1"]
    warned = [e.data["message"] for e in events if e.data.get("code") == "step_limit"]
    assert warned == ["stopped: the turn limit (1) was reached"]
    last = started_with(tmp_path)  # the last query the CLI was started with
    # the session keeps it: told it holds for this turn only, a later turn calls tools again
    assert last["prompt"] == TURN_LIMIT.format(limit=1) and last["resume"] is not None
    assert last["max_turns"] == "1" and last["builtins"] == "" and last["tools"] == []


async def test_a_query_resumed_after_a_pause_or_at_its_limit_is_charged_its_cost_once(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed session's result carries the session's cost, what it cost before the
    process that resumed it included (the bundled CLI, driven against a scripted model).
    A query that paused had no result: the attempt that resumes it is charged both. The
    one last answer at the turn limit resumes a session whose query was charged already:
    it is charged what the cost grew by."""
    from trellis.harness import budget

    costs: list[float | None] = []
    account = budget.account
    monkeypatch.setattr(
        budget, "account", lambda run, **kw: (costs.append(kw["cost"]), account(run, **kw))
    )
    refunds.clear()
    script = [{"tool": "refund", "args": {"order": "o3"}}, {"text": "done"}]
    paid = options(tmp_path, script, model="m")
    paid.env["FAKE_CLAUDE_COST"] = "0.5"
    agent = arm(harness.wrap(paid, id="refunds"), [refund])
    paused = await agent.run("refund o3", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    assert costs == [None, 1.5]  # the step that paused (no result: no cost), then all three
    costs.clear()
    steps = [{"builtin": "Read", "args": {"file_path": f"f{n}"}} for n in range(3)]
    limited = options(tmp_path, [*steps, {"text": "never"}], model="m", max_turns=2)
    limited.env["FAKE_CLAUDE_COST"] = "0.5"
    result = await harness.wrap(limited, id="limited").run("read them", user="u1")
    assert result.status is RunStatus.SUCCESS
    assert costs == [1.0, 0.5]  # two steps, then the last answer's one: not 1.0 and 1.5


async def test_a_last_query_at_the_limit_that_gives_no_answer_leaves_the_error(
    harness: Harness, tmp_path: Path
) -> None:
    target = options(tmp_path, [{"tool": "refund", "args": {"order": "o3"}}, {"text": "x"}],
                     max_turns=1)  # fmt: skip
    target.env["FAKE_CLAUDE_AT_LIMIT"] = ""
    agent = arm(harness.wrap(target, id="limited-2"), [tool(refund.fn, side_effects="write")])
    result = await agent.run("refund o3", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "maximum number of turns" in result.error.message


async def test_a_query_stopped_by_its_own_budget_keeps_its_error(
    harness: Harness, tmp_path: Path
) -> None:
    """Only the turn limit is answered once more: a query the CLI stopped at the options' own
    ``max_budget_usd`` ends with the CLI's error, as before."""
    target = options(tmp_path, [{"tool": "refund", "args": {"order": "o6"}}, {"text": "x"}],
                     max_budget_usd=0.01)  # fmt: skip
    target.env["FAKE_CLAUDE_COST"] = "0.02"
    agent = arm(harness.wrap(target, id="own-budget"), [tool(refund.fn, side_effects="write")])
    result = await agent.run("refund o6", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "maximum budget" in result.error.message


#: a result over ``tools.results.MAX_CHARS``, and where the run keeps it
LONG = "\n".join(f"row {n}" for n in range(1, 30_001))
KEPT = f"/large_tool_results/{content_key('result', LONG)}"


@tool(side_effects="read")
def dump() -> str:
    """Dump every row."""
    return LONG


async def test_a_large_result_is_previewed_naming_where_the_run_keeps_it(
    harness: Harness, tmp_path: Path
) -> None:
    script = [{"tool": "dump", "args": {}}, {"text": "{last}"}]
    agent = arm(harness.wrap(options(tmp_path, script), id="d"), [dump])
    preview = (await agent.run("dump it", user="u1")).answer
    assert f"saved in the filesystem at this path: {KEPT}" in preview and len(preview) < 25_000
    assert "row 30000" in preview and "row 100" not in preview


async def test_a_kept_result_is_read_in_pages_after_a_pause(
    harness: Harness, tmp_path: Path
) -> None:
    refunds.clear()
    script = [
        {"tool": "dump", "args": {}},
        {"tool": "refund", "args": {"order": "o1"}},
        {"tool": "read_file", "args": {"file_path": KEPT, "offset": 10, "limit": 2}},
        {"text": "{last}"},
    ]
    agent = arm(harness.wrap(options(tmp_path, script), id="dumper"), [dump, refund])
    paused = await agent.run("dump it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert refunds == ["o1"]  # the session went on: read_file read the journal's copy
    assert finished.answer == (
        f'11  row 11\n12  row 12\n…[29988 more lines: read_file(file_path="{KEPT}", offset=12)]'
    )


class DenyAll(Hooks):
    async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
        return Deny("nothing runs here")


@pytest.mark.parametrize(
    "mode",
    [
        {},
        {"allowed_tools": ["Bash", "Write"]},
        {"permission_mode": "acceptEdits"},
        {"permission_mode": "bypassPermissions"},
    ],
    ids=["default", "allowed_tools", "acceptEdits", "bypassPermissions"],
)
async def test_built_in_tools_are_governed_in_every_permission_mode(
    harness: Harness, tmp_path: Path, mode: dict[str, Any]
) -> None:
    """The CLI skips ``can_use_tool`` for allowed tools and under ``acceptEdits`` and
    ``bypassPermissions``: the harness's ``PreToolUse`` hook governs every call regardless."""
    script = [
        {"builtin": "Write", "args": {"file_path": "a.txt"}},
        {"builtin": "Bash", "args": {"command": "rm -rf /"}},
        {"text": "said: {last}"},
    ]
    agent = harness.wrap(options(tmp_path, script, **mode), id="locked", hooks=[DenyAll()])
    result = await agent.run("work", user="u1")
    assert result.answer == "said: Bash was not run: nothing runs here"
    assert built_ins(tmp_path) == []  # neither ran


async def test_calls_claude_runs_itself_are_recorded_with_their_result_and_duration(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """Its built-ins and its own MCP servers' tools run in the CLI: their ``PostToolUse`` (or
    ``PostToolUseFailure``) records each as the run's tool call — on the stream, in the memory
    service with its result or error and how long it took — while a harness tool's call is the
    bridge's, recorded once."""
    script = [
        {"builtin": "Read", "args": {"file_path": "orders.csv"}},
        {"builtin": "Grep", "args": {"pattern": "o3"}, "fails": "no such file"},
        {"tool": "lookup", "args": {"sku": "A-1"}},
        {"text": "done: {last}"},
    ]

    @tool(side_effects="read")
    def lookup(sku: str) -> str:
        """Units of a SKU."""
        return f"{sku}: 3"

    agent = arm(memory_harness.wrap(options(tmp_path, script), id="ops"), [lookup])
    events = [e async for e in agent.stream("check o3", user="u1")]
    assert events[-1].outcome is RunOutcome.SUCCESS
    await memory_harness.writes.drain()
    recorded = {c.body["tool"]: c.body for c in memory_service.calls if c.name == "record_tool"}
    assert set(recorded) == {"Read", "Grep", "lookup"}  # the harness tool once, by the bridge
    assert recorded["Read"]["status"] == "ok" and recorded["Read"]["output"] == "Read ran"
    assert recorded["Read"]["latency_ms"] >= 0
    assert recorded["Grep"]["status"] == "error" and recorded["Grep"]["output"] == "no such file"
    results = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [r["tool"] for r in results] == ["Read", "Grep", "lookup"]
    assert native_spec("mcp__crm__find").server == "crm"
    assert native_spec("Bash").side_effects == "write"  # Claude Code's own check asks first
