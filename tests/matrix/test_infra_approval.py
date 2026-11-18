"""An infrastructure tool is the platform operator's: a call of one that would need a
person's approval is denied — never asked of the run's user, nor of a role a hook names — on
every adapter and every way an infrastructure tool arrives (ADR 0008,
``docs/governance.md#who-decides-what``).

A shell (the harness's ``sandbox_exec``, Deep Agents' ``execute`` on a sandbox backend, an
OpenAI ``SandboxAgent``'s ``exec_command``, Claude Code's and Kimi Code's ``Bash``) or a network
tool (Claude's ``WebFetch``, Kimi's ``FetchURL``) runs unasked and announced by default. When
the tool catalog makes one ask (its risk declared ``write`` here), or a ``before_tool`` hook asks
about it (``Ask(assignee="role:ops")``), the call is denied — not run, the model told why, the
denial on the run's stream — while a business tool of the same run (a harness ``@tool`` write,
Claude's or Kimi's ``Write``) asks as ever: the run's user, or the role the hook names.

Offline, on the suite's stand-ins (the planned models, the scripted Claude CLI, Kimi Code on a
planned endpoint, the fake sandbox provider; an OpenAI ``SandboxAgent`` on the SDK's
Unix-local sandbox). Sharded with the generated matrix (``make matrix``).
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest

from tests.matrix.test_approval_rule import KIMI_ROW_SECONDS, KIMI_TURN_SECONDS
from tests.support.adapters import BUILDERS, CLI, SYSTEM
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.planned import FINAL, Call, PlannedChat, PlannedChatModel, PlannedModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, Settings, sandbox, tool
from trellis.contracts import Interrupt, RunEvent, RunEventType, RunStatus, ToolCall
from trellis.harness.governance.decision import ONCE
from trellis.harness.hooks import Ask, Hooks

pytestmark = pytest.mark.matrix

#: What the model reads of an infrastructure call denied rather than asked.
REFUSED: Final = "is the platform's infrastructure, and a call of it that needs approval is denied"
#: The role a deployment's hook routes an approval to, and one of its people.
OPS: Final = "role:ops"
OPERATOR: Final = "ops-1"
#: The business tool of the harness's rows: a write in the user's name.
NOTE: Final = "note"


class OpsReviews(Hooks):
    """A deployment's rule in code: the calls of ``tools`` are the operators' to approve."""

    def __init__(self, *tools: str) -> None:
        self.tools = tools

    async def before_tool(self, call: ToolCall) -> Ask | None:
        if call.tool in self.tools:
            return Ask(f"Run {call.tool}?", assignee=OPS)
        return None


# --------------------------------------------------------------------------- the rows


@dataclass
class Deployment:
    h: Harness
    memory: FakeMemoryService
    tmp: Path
    boxes: FakeSandboxes = field(default_factory=FakeSandboxes)
    servers: contextlib.AsyncExitStack = field(default_factory=contextlib.AsyncExitStack)
    #: Kimi's planned endpoint, where a Kimi row's model reads what it is told
    endpoint: Any = None

    @property
    def notes(self) -> Path:
        return self.tmp / "notes"


@dataclass(frozen=True)
class Row:
    """An adapter, the infrastructure tool its way brings, its arguments, the business call
    after it, and how to tell what ran."""

    adapter: str
    infra: str
    args: dict[str, Any]
    #: the business call's tool and arguments
    business: tuple[str, dict[str, Any]] = (NOTE, {"text": "hello"})
    #: a live Claude session (its runs carry a conversation)
    thread: str | None = None

    def called(self, tmp: Path) -> dict[str, Any]:
        """Its arguments, ``{tmp}`` the test's folder."""
        return {k: v.replace("{tmp}", str(tmp)) for k, v in self.args.items()}

    @property
    def id(self) -> str:
        return f"{self.adapter}{'_live' if self.thread else ''}-{self.infra}"


def _note(d: Deployment) -> Any:
    def note(text: str) -> str:
        """Note a line in the user's notes."""
        with d.notes.open("a") as kept:
            kept.write(f"{text}\n")
        return "noted"

    return tool(note, side_effects="write")


HARNESS_ADAPTERS: Final = tuple(BUILDERS)
ROWS: Final = [
    # the harness's own sandbox shell, on every adapter
    *(Row(a, "sandbox_exec", {"command": "echo hi"}) for a in HARNESS_ADAPTERS),
    Row("claude_agent_sdk", "sandbox_exec", {"command": "echo hi"}, thread="t1"),
    # Deep Agents' own shell on the run's sandbox (ReAct's backend, a graph of your own)
    Row("react", "execute", {"command": "echo hi"}),
    Row("deepagents", "execute", {"command": "echo hi"}),
    # an OpenAI SandboxAgent's own shell
    Row("openai_agents", "exec_command", {"cmd": "echo hi > {tmp}/ran.txt"}),
    # Claude Code's own shell and network (a query, and a conversation's live session)
    Row("claude", "Bash", {"command": "echo hi"}, business=("Write", {"file_path": "n.md"})),
    Row(
        "claude",
        "Bash",
        {"command": "echo hi"},
        business=("Write", {"file_path": "n.md"}),
        thread="t1",
    ),
    Row(
        "claude",
        "WebFetch",
        {"url": "https://example.com/x"},
        business=("Write", {"file_path": "n.md"}),
    ),
    # Kimi Code's own shell and network
    Row(
        "kimi",
        "Bash",
        {"command": "touch bash-ran.txt"},
        business=("Write", {"path": "n.md", "content": "hello"}),
    ),
    Row(
        "kimi",
        "FetchURL",
        {"url": "https://example.com/x"},
        business=("Write", {"path": "n.md", "content": "hello"}),
    ),
]


@contextlib.asynccontextmanager
async def deployed(tmp: Path) -> AsyncIterator[Deployment]:
    from tests.support.kimi import detached

    memory = FakeMemoryService()
    settings = Settings(host_execution="allow")
    async with Harness(config=settings, memory=memory.client()) as h:
        d = Deployment(h, memory, tmp)
        async with d.servers:
            yield d
    await detached()


async def built(d: Deployment, row: Row, plan: list[Call]) -> tuple[Any, list[Any]]:
    """The row's target following ``plan``, and the tools the harness runs for it."""
    if row.infra == "sandbox_exec":
        tools = [sandbox(d.boxes).tools(), _note(d)]
        return await BUILDERS[row.adapter](d.h, tools, d.tmp, plan)
    if row.infra == "execute":
        if row.adapter == "react":
            return ReAct(SYSTEM, PlannedChat(plan), backend=sandbox(d.boxes)), [_note(d)]
        from deepagents import create_deep_agent

        from tests.support.arm import native_tools

        own = await native_tools(d.h, _note(d), framework="deepagents")
        graph = create_deep_agent(
            model=PlannedChatModel(plan=plan),
            tools=own,
            system_prompt=SYSTEM,
            **sandbox(d.boxes).deep_agent(),
        )
        return graph, [_note(d)]
    if row.infra == "exec_command":
        return _sandbox_agent(d, plan), [_note(d)]
    if row.adapter == "claude":
        return _claude(d, plan), []
    return await _kimi(d, plan), []


def _sandbox_agent(d: Deployment, plan: list[Call]) -> Any:
    from agents.sandbox import SandboxAgent
    from agents.sandbox.capabilities import Shell

    return SandboxAgent(name="box", model=PlannedModel(plan), capabilities=[Shell()])


def _framework_options(row: Row) -> dict[str, Any] | None:
    if row.infra != "exec_command":
        return None
    from agents import RunConfig
    from agents.sandbox import SandboxRunConfig
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    return {"run_config": RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))}


def _claude(d: Deployment, plan: list[Call]) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions

    script: list[dict[str, Any]] = [{"builtin": name, "args": args} for name, args in plan]
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps([*script, {"text": FINAL}]),
        "FAKE_CLAUDE_BUILTINS": str(d.tmp / "builtins.jsonl"),
        "FAKE_CLAUDE_SESSIONS": str(d.tmp),
    }
    return ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env=env,
        sandbox={"enabled": True, "allowUnsandboxedCommands": False},
    )


async def _kimi(d: Deployment, plan: list[Call]) -> Any:
    from tests.support.kimi import KIMI_BIN, PlannedEndpoint, kimi_target, needs_kimi

    if KIMI_BIN is None and needs_kimi.args[0]:
        pytest.skip(needs_kimi.kwargs["reason"])
    d.endpoint = await PlannedEndpoint(plan).attached()
    return kimi_target(d.endpoint, d.tmp / f"kimi-{uuid.uuid4().hex[:6]}")


def _ran(d: Deployment, row: Row) -> Callable[[str], bool]:
    """Whether a call of a tool ran, as its effect shows."""

    def builtins() -> list[str]:
        path = d.tmp / "builtins.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        return [json.loads(line)["tool"] for line in lines]

    def ran(name: str) -> bool:
        if name == NOTE:
            return d.notes.exists()
        if name in ("sandbox_exec", "execute"):
            return d.boxes.commands == ["echo hi"]
        if name == "exec_command":
            return (d.tmp / "ran.txt").exists()
        if row.adapter == "claude":
            return name in builtins()
        if name == "FetchURL":  # nothing to see offline: its denial is on the run's stream
            return False
        written = "bash-ran.txt" if name == "Bash" else "n.md"
        return any(d.tmp.glob(f"kimi-*/kimi-work/{written}"))

    return ran


def _said(d: Deployment, events: list[RunEvent]) -> str:
    """Everything the run's stream and (Kimi's) model were told, as text."""
    told = json.dumps([e.data for e in events], default=str)
    return told + (d.endpoint.said() if d.endpoint is not None else "")


def _waiting(events: list[RunEvent]) -> Interrupt | None:
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED]
    data = finished[-1].data.get("interrupt") if finished else None
    return Interrupt.model_validate(data) if data else None


def _limited(row: Row) -> Any:
    if row.adapter != "kimi":
        return pytest.param(row, id=row.id)
    limit = KIMI_ROW_SECONDS + 4 * KIMI_TURN_SECONDS
    return pytest.param(row, id=row.id, marks=pytest.mark.timeout(limit))


# --------------------------------------------------------------------------- the tests


@pytest.mark.parametrize("row", [_limited(row) for row in ROWS])
async def test_an_infrastructure_call_the_catalog_makes_ask_is_denied_not_asked_of_the_user(
    row: Row, tmp_path: Path
) -> None:
    """The catalog makes the infrastructure tool ask: denied, the model told why, nothing run,
    the denial on the run's stream; the business call of the same run asks the run's user, who
    answers, and it runs. (Kimi's planned endpoint makes the business call first: a turn its
    denial stopped goes on with the next prompt, which the endpoint reads as the plan's end.)"""
    business, business_args = row.business
    calls = [(row.infra, row.called(tmp_path)), (business, business_args)]
    infra_first = row.adapter != "kimi"
    async with deployed(tmp_path) as d:
        d.memory.catalog[row.infra] = {"name": row.infra, "side_effects": "write"}
        d.memory.catalog[business] = {"name": business, "side_effects": "write"}
        ran = _ran(d, row)
        target, tools = await built(d, row, calls if infra_first else calls[::-1])
        agent = arm(
            d.h.wrap(target, id=f"infra-{row.adapter}", framework_options=_framework_options(row)),
            tools,
        )
        events = [e async for e in agent.stream("work", user="ada", thread=row.thread)]
        asked = _waiting(events)
        assert asked is not None and asked.tool_call is not None, (_said(d, events), events[-1])
        # the only question is the business call's, and it is the run's user's
        assert asked.tool_call.tool == business, asked
        assert asked.assignee in (None, "user:ada"), asked
        assert _refusals(events) == ([row.infra] if infra_first else [])
        done = await agent.resume(asked.interrupt_id, "answer", answer=ONCE.value, reviewer="ada")
        assert done.status is RunStatus.SUCCESS, (done.error, done.interrupt)
        assert ran(business) and not ran(row.infra)
        assert REFUSED in _said(d, events) + str(done.answer), _said(d, events)


def _refusals(events: list[RunEvent]) -> list[str]:
    return [
        e.data["tool"]
        for e in events
        if e.type is RunEventType.CUSTOM
        and e.data.get("name") == "decision"
        and e.data.get("decision") == "REFUSE"
    ]


@pytest.mark.parametrize("row", [_limited(row) for row in ROWS])
async def test_a_hooks_question_about_an_infrastructure_call_is_denied_not_routed(
    row: Row, tmp_path: Path
) -> None:
    """A ``before_tool`` hook asks a role about both calls (``Ask(assignee="role:ops")``): the
    business call waits in that role's inbox, not the user's, and runs once an operator
    approves it (``agent.resume(..., reviewer=...)``); the infrastructure call after it is
    denied all the same — no tenant's role approves the platform's infrastructure."""
    business, business_args = row.business
    async with deployed(tmp_path) as d:
        # the administrator lets the tool run unasked (a network tool: allowed out)
        d.memory.catalog[row.infra] = {"name": row.infra, "side_effects": "read"}
        d.memory.catalog[business] = {"name": business, "side_effects": "write"}
        ran = _ran(d, row)
        target, tools = await built(
            d, row, [(business, business_args), (row.infra, row.called(tmp_path))]
        )
        agent = arm(
            d.h.wrap(
                target,
                id=f"ops-{row.adapter}",
                hooks=[OpsReviews(row.infra, business)],
                framework_options=_framework_options(row),
            ),
            tools,
        )
        events = [e async for e in agent.stream("work", user="ada", thread=row.thread)]
        asked = _waiting(events)
        assert asked is not None and asked.tool_call is not None, (_said(d, events), events[-1])
        assert (asked.tool_call.tool, asked.assignee) == (business, OPS), asked
        assert [s.run_id for s in await d.h.inbox(OPS)] == [asked.run_id]
        assert await d.h.inbox("user:ada") == []
        done = await agent.resume(
            asked.interrupt_id, "answer", answer=ONCE.value, reviewer=OPERATOR
        )
        assert done.status is RunStatus.SUCCESS, (done.error, done.interrupt)
        assert ran(business) and not ran(row.infra)
        told = str(done.answer) + (d.endpoint.said() if d.endpoint is not None else "")
        assert REFUSED in told, told
