"""One approval rule, on every adapter and every way a tool arrives — as Claude Code asks.

What a tool says is its risk, and its risk is what a person is asked:

* ``read`` (``readOnlyHint: true``, ``side_effects="read"``) runs, nobody asked;
* ``write`` (``destructiveHint: false``, ``side_effects="write"``) and a tool that says nothing
  (no annotations at all, no ``side_effects``) ask: allow once, allow for this conversation,
  always allow (a memory service keeps it), deny;
* ``irreversible`` (annotations that leave it destructive by the MCP defaults,
  ``side_effects="irreversible"``) asks every call: allow once, or deny;
* the tool catalog's ``risk`` (an administrator's) wins over all of it.

A row is an adapter and a way its tools arrive (:data:`ROWS`): the harness's own (``mcp(...)``,
``@tool`` functions, the gateway's MCP listing) on every adapter, and each framework's own —
LangChain tools and ``langchain.mcp`` tools in ReAct, ``create_agent`` (on its own and with
``trellis.middleware()``), a hand-built graph, Deep Agents and a langgraph-swarm; an OpenAI
agent's ``MCPServerStdio`` and ``FunctionTool``; the Claude Agent SDK's in-process MCP server
and its built-ins (a query, and a conversation's live session); Kimi Code's MCP servers and its
built-ins. Each row walks the tools of each risk its way can say through every answer
(:data:`SCRIPT`), and the oracle (:class:`Rule`) says what each call must do: deny never runs
the body, allow once runs that call only, allow for this conversation asks no more in it (a
later run of it too; another conversation asks), always allow asks no more in the user's next
conversation, an irreversible tool is asked every call and offered neither. Every row's catalog
keeps what the harness published, as the memory service does, so the risk the catalog derives
from it is the one asked by. Offline, on the suite's stand-ins: the planned models, the
scripted Claude CLI, Kimi Code on a planned endpoint, the stand-in gateway and MCP server.

Sharded with the generated matrix (``make matrix``): every row runs in CI, not in ``make test``.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from tests.support.adapters import BUILDERS, SYSTEM
from tests.support.arm import arm
from tests.support.gateway import URL as GATEWAY_URL
from tests.support.gateway import FakeGateway
from tests.support.memory import FakeMemoryService
from tests.support.planned import FINAL, Call, PlannedChat, PlannedChatModel, PlannedModel
from trellis import Harness, ReAct, Settings, mcp, middleware, stdio, tool
from trellis.contracts import InterruptReason, RunStatus
from trellis.harness.adapters.acp import KIMI_CODE
from trellis.harness.adapters.claude import CLAUDE_CODE
from trellis.harness.governance import catalog as catalog_module
from trellis.harness.governance.decision import (
    ALWAYS,
    CONVERSATION,
    DENY,
    EVERY_TIME,
    KEPT,
    ONCE,
    SAID,
)
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.tools.bridge import FRAMEWORK_ASKS

pytestmark = pytest.mark.matrix

SERVER: Final = Path(__file__).parents[1] / "support" / "mcp_risks_server.py"
#: Each risk kind, and the tool of the stand-in server (and of each way) that says it.
TOOLS: Final = {"read": "look", "write": "note", "undeclared": "tidy", "irreversible": "purge"}
#: The kinds a tool can say, in each way's words.
ALL_KINDS: Final = tuple(TOOLS)
#: What a person is told about each risk, after the tool's name.
WHY: Final = {
    "write": "is write.",
    "undeclared": "has not declared its side effects.",
    "irreversible": "is irreversible.",
}


@pytest.fixture(autouse=True)
def catalog_read_each_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog is read again at every decision: what a run published is the next run's."""
    monkeypatch.setattr(catalog_module, "GOVERNANCE_TTL_SECONDS", -1.0)


# --------------------------------------------------------------------------- the oracle

#: One planned call: its risk kind, its ``n``, and the answer if a person is asked.
Step = tuple[str, int, str | None]
#: The runs of a row: a conversation's label and its calls.
Script = list[tuple[str, list[Step]]]

#: The conversation a later run of goes on (a row whose runs carry no conversation has none).
LATER: Final = "c1 later"
#: Every answer, in turn, for each risk: deny, once, for this conversation (asked no more in
#: it, a later run of it included), always (another conversation's first call; the next asks
#: nobody). An irreversible tool is allowed once each time. No run makes more calls than a
#: run's step limit lets it (``budget``), nor fails calls in a row past a framework's own
#: limit (a denial is a failed call).
SCRIPT: Final[Script] = [
    (
        "c0",
        [
            ("write", 1, DENY.value),
            ("write", 2, ONCE.value),
            ("undeclared", 1, DENY.value),
            ("undeclared", 2, ONCE.value),
            ("irreversible", 1, DENY.value),
            ("irreversible", 2, ONCE.value),
            ("read", 1, None),
        ],
    ),
    (
        "c1",
        [
            ("write", 3, CONVERSATION.value),
            ("undeclared", 3, CONVERSATION.value),
            ("irreversible", 3, ONCE.value),
            ("write", 4, None),
            ("undeclared", 4, None),
            ("irreversible", 4, ONCE.value),
        ],
    ),
    # a later run of the conversation: what was allowed for it stands
    (LATER, [("read", 2, None), ("write", 5, None), ("undeclared", 5, None)]),
    (
        "c2",
        [
            ("write", 6, ALWAYS.value),
            ("undeclared", 6, ALWAYS.value),
            ("irreversible", 6, ONCE.value),
        ],
    ),
    ("c3", [("write", 7, None), ("undeclared", 7, None), ("irreversible", 7, ONCE.value)]),
]
#: An administrator's risk in the catalog wins: the tool that says nothing declared ``read``
#: (it runs), the write declared ``irreversible`` (asked every call, offered once or deny).
OVERRIDES: Final = {"undeclared": "read", "write": "irreversible"}
OVERRIDDEN: Final[Script] = [
    ("c1", [("write", 1, DENY.value), ("undeclared", 1, None), ("write", 2, ONCE.value)]),
    ("c2", [("undeclared", 2, None), ("write", 3, ONCE.value)]),
]


@dataclass
class Rule:
    """What each call must do (the module's rule): whether a person is asked, with which
    choices, and whether its body runs — from what was allowed so far."""

    risks: dict[str, str]
    conversation: dict[str, set[str]] = field(default_factory=dict)
    always: set[str] = field(default_factory=set)

    def call(self, kind: str, scope: str, answer: str | None) -> tuple[tuple[Any, ...], bool]:
        """The choices a person is offered (none: not asked), and whether the body runs."""
        risk = self.risks[kind]
        if risk == "read":
            return (), True
        if risk != "irreversible" and (
            kind in self.always or kind in self.conversation.get(scope, set())
        ):
            return (), True
        offered = EVERY_TIME if risk == "irreversible" else KEPT
        assert answer is not None, f"{kind} is asked: the script names no answer"
        if answer == CONVERSATION.value:
            self.conversation.setdefault(scope, set()).add(kind)
        elif answer == ALWAYS.value:
            self.always.add(kind)
        return offered, answer != DENY.value


# --------------------------------------------------------------------------- stand-ins


class LedgerGateway(FakeGateway):
    """The stand-in gateway, listing the ledger's tools as the key's MCP tools, each with the
    annotations the server gives (none for ``tidy``), and running them (each call a line in
    ``ran``)."""

    def __init__(self, ran: Path) -> None:
        super().__init__()
        self.ran = ran

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/mcp/tool/execute":
            self.requests.append(request)
            call = json.loads(request.content)
            name = call["function"]["name"].removeprefix("ledger-")
            n = json.loads(call["function"]["arguments"])["n"]
            with self.ran.open("a") as log:
                log.write(f"{name} {n}\n")
            content = f"{name} {n} done"
            return httpx.Response(
                200, json={"role": "tool", "content": content, "tool_call_id": call.get("id")}
            )
        return await super().handle(request)

    def _rpc(self, message: dict[str, Any]) -> httpx.Response:
        if message["method"] != "tools/list":
            return super()._rpc(message)
        hints: dict[str, dict[str, bool] | None] = {
            "look": {"readOnlyHint": True},
            "note": {"destructiveHint": False},
            "tidy": None,
            "purge": {"destructiveHint": True},
        }
        schema = {"type": "object", "properties": {"n": {"type": "integer", "description": "n"}}}
        tools = [
            {"name": f"ledger-{name}", "inputSchema": schema}
            | ({} if said is None else {"annotations": said})
            for name, said in hints.items()
        ]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}})


def _wrote(ran: Path, name: str, n: int) -> str:
    with ran.open("a") as log:
        log.write(f"{name} {n}\n")
    return f"{name} {n} done"


def ledger(ran: Path) -> Any:
    """The stand-in server, reached by the harness itself (``mcp(stdio(...))``)."""
    return mcp(stdio(sys.executable, str(SERVER), env={"RAN": str(ran)}), name="ledger")


def functions(ran: Path) -> list[Any]:
    """The ledger's tools as functions: ``@tool`` with ``side_effects``, ``tidy`` without."""

    def made(name: str) -> Callable[..., str]:
        def run(n: int) -> str:
            return _wrote(ran, name, n)

        run.__name__ = name
        run.__doc__ = f"{name.capitalize()} an entry of the ledger."
        return run

    return [
        tool(made("look"), side_effects="read"),
        tool(made("note"), side_effects="write"),
        tool(made("tidy")),
        tool(made("purge"), side_effects="irreversible"),
    ]


#: What each kind's LangChain tool says in its ``metadata``: a framework's own tool (its
#: ``side_effects``), and a ``langchain.mcp`` tool (its server's annotations where that library
#: and ``langchain-mcp-adapters`` keep them; none at all for the one that says nothing).
LANGCHAIN_SAYS: Final[dict[str, dict[str, Any]]] = {
    "read": {"side_effects": "read"},
    "write": {"side_effects": "write"},
    "undeclared": {},
    "irreversible": {"side_effects": "irreversible"},
}
LANGCHAIN_MCP_SAYS: Final[dict[str, dict[str, Any]]] = {
    "read": {"mcp": {"tool": {"annotations": {"read_only_hint": True}}}},
    "write": {"mcp": {"tool": {"annotations": {"destructive_hint": False}}}},
    "undeclared": {"mcp": {"tool": {}, "server": {"name": "ledger"}}},
    "irreversible": {"mcp": {"tool": {"annotations": {"destructive_hint": True}}}},
}


def langchain_tools(ran: Path, says: dict[str, dict[str, Any]]) -> list[Any]:
    from langchain_core.tools import StructuredTool

    def made(kind: str) -> Any:
        name = TOOLS[kind]

        def run(n: int) -> str:
            return _wrote(ran, name, n)

        made_tool = StructuredTool.from_function(
            run, name=name, description=f"{name.capitalize()} an entry of the ledger."
        )
        made_tool.metadata = says[kind] or None
        return made_tool

    return [made(kind) for kind in ALL_KINDS]


def openai_function_tools(ran: Path) -> list[Any]:
    from agents import function_tool

    @function_tool
    def tidy(n: int) -> str:
        """Tidy an entry of the ledger."""
        return _wrote(ran, "tidy", n)

    return [tidy]


def claude_server(ran: Path) -> Any:
    """The ledger as an in-process MCP server of the Claude Agent SDK's
    (``create_sdk_mcp_server``), each tool with the annotations the stand-in gives."""
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool
    from mcp.types import ToolAnnotations

    said: dict[str, ToolAnnotations | None] = {
        "look": ToolAnnotations(read_only_hint=True),
        "note": ToolAnnotations(destructive_hint=False),
        "tidy": None,
        "purge": ToolAnnotations(destructive_hint=True),
    }

    def made(name: str) -> Any:
        @sdk_tool(name, f"{name.capitalize()} an entry of the ledger.", {"n": int}, said[name])
        async def run(args: dict[str, Any]) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": _wrote(ran, name, args["n"])}]}

        return run

    return create_sdk_mcp_server("ledger", tools=[made(name) for name in said])


# --------------------------------------------------------------------------- the ways


@dataclass(frozen=True)
class Way:
    """A way tools arrive: the risks its tools can say, each tool's name as a call names it
    (and the governed call is named), and each call's arguments."""

    kinds: tuple[str, ...] = ALL_KINDS
    named: Callable[[str], str] = lambda kind: TOOLS[kind]
    args: Callable[[int], dict[str, Any]] = lambda n: {"n": n}
    #: what ran, as ``"<tool> <n>"`` lines, where it is not the ``ran`` file
    ran: Callable[[Path], list[str]] | None = None
    #: the framework that runs the tools under its own permission flow: where it asks its
    #: user about a call, its question is the person's (its reason, the rule's choices)
    asker: str | None = None


def _claude_builtins_ran(tmp: Path) -> list[str]:
    path = tmp / "builtins.jsonl"
    found = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [f"{f['tool']} {f['args']['n']}" for f in found]


def _kimi_builtins_ran(tmp: Path) -> list[str]:
    work = tmp / "kimi-work"
    written = sorted(work.glob("w*.txt"), key=lambda p: p.stat().st_mtime_ns)
    return [f"Write {p.stem.removeprefix('w')}" for p in written]


WAYS: Final[dict[str, Way]] = {
    # the harness's own, on every adapter
    "harness_mcp": Way(named=lambda kind: f"ledger-{TOOLS[kind]}"),
    "tool_function": Way(),
    "gateway_mcp": Way(named=lambda kind: f"ledger-{TOOLS[kind]}"),
    # a LangChain-based framework's own tools
    "langchain_tool": Way(),
    "langchain_mcp": Way(),
    # the OpenAI Agents SDK's own
    "openai_mcp_server": Way(),
    "openai_function_tool": Way(kinds=("undeclared",)),
    # the Claude Agent SDK's own
    "claude_mcp_server": Way(named=lambda kind: f"mcp__ledger__{TOOLS[kind]}", asker=CLAUDE_CODE),
    "claude_builtin": Way(
        kinds=("read", "write"),
        named=lambda kind: {"read": "Read", "write": "Write"}[kind],
        ran=_claude_builtins_ran,
        asker=CLAUDE_CODE,
    ),
    # Kimi Code's own: ACP tells nothing of an MCP tool's annotations, so its servers' tools
    # say nothing to the harness (the catalog can say it)
    "kimi_mcp_server": Way(
        kinds=("undeclared",), named=lambda kind: f"mcp__ledger__{TOOLS[kind]}", asker=KIMI_CODE
    ),
    "kimi_builtin": Way(
        kinds=("write",),
        named=lambda kind: "Write",
        args=lambda n: {"path": f"w{n}.txt", "content": f"entry {n}"},
        ran=_kimi_builtins_ran,
        asker=KIMI_CODE,
    ),
}
HARNESS_WAYS: Final = ("harness_mcp", "tool_function", "gateway_mcp")
LANGCHAIN: Final = ("react", "langgraph", "middleware", "graph", "deepagents", "swarm")
ADAPTERS: Final[dict[str, tuple[str, ...]]] = {
    "function": HARNESS_WAYS,
    **dict.fromkeys(LANGCHAIN, (*HARNESS_WAYS, "langchain_tool", "langchain_mcp")),
    "openai_agents": (*HARNESS_WAYS, "openai_mcp_server", "openai_function_tool"),
    "claude_query": (*HARNESS_WAYS, "claude_mcp_server", "claude_builtin"),
    "claude_live": (*HARNESS_WAYS, "claude_mcp_server", "claude_builtin"),
    "kimi": (*HARNESS_WAYS, "kimi_mcp_server", "kimi_builtin"),
}


@dataclass(frozen=True)
class Row:
    adapter: str
    way: str

    @property
    def id(self) -> str:
        return f"{self.adapter}-{self.way}"

    @property
    def conversations(self) -> bool:
        """Whether its runs carry a conversation: a Claude query has none (each run its own;
        what is allowed for the conversation covers the rest of that run), a live session has."""
        return self.adapter != "claude_query"

    @property
    def turn_by_turn(self) -> bool:
        """Whether its model makes each run's calls afresh (a function follows its plan; the
        scripted Claude CLI scripts each turn of a session), rather than going on from the
        tool results the conversation holds."""
        return self.adapter in ("function", "claude_query", "claude_live", "kimi")


ROWS: Final = [Row(a, w) for a, ways in ADAPTERS.items() for w in ways]


# --------------------------------------------------------------------------- one row's runs


@dataclass
class Deployment:
    """A row's harness (the gateway on for the gateway's tools), its memory service and
    where its tools' calls are noted."""

    h: Harness
    memory: FakeMemoryService
    tmp: Path
    servers: contextlib.AsyncExitStack

    @property
    def ran(self) -> Path:
        return self.tmp / "ran"


@contextlib.asynccontextmanager
async def deployed(row: Row, tmp: Path) -> AsyncIterator[Deployment]:
    from tests.support.kimi import detached

    memory = FakeMemoryService()
    gateway: Any = False
    settings = Settings()
    if row.way == "gateway_mcp":
        gateway = LedgerGateway(tmp / "ran").gateway()
        settings = Settings(llm_api_key="vk", llm_base_url=GATEWAY_URL, llm_gateway="bifrost")
    async with (
        contextlib.AsyncExitStack() as servers,
        Harness(config=settings, memory=memory.client(), gateway=gateway) as h,
    ):
        yield Deployment(h, memory, tmp, servers)
    await detached()


async def built(d: Deployment, row: Row, plan: list[Call]) -> tuple[Any, list[Any]]:
    """The row's target following ``plan``, and the tools the harness runs for it."""
    if row.way in HARNESS_WAYS:
        sources = {
            "harness_mcp": lambda: [ledger(d.ran)],
            "tool_function": lambda: functions(d.ran),
            "gateway_mcp": list,
        }[row.way]()
        if row.adapter == "swarm":  # the run's tools, offered to each member
            return _swarm(plan, []), sources
        builder = BUILDERS["claude_agent_sdk" if row.adapter.startswith("claude") else row.adapter]
        return await builder(d.h, sources, d.tmp, plan)
    if row.way in ("langchain_tool", "langchain_mcp"):
        says = LANGCHAIN_SAYS if row.way == "langchain_tool" else LANGCHAIN_MCP_SAYS
        return _langchain(row.adapter, plan, langchain_tools(d.ran, says)), []
    if row.way.startswith("openai_"):
        return await _openai(d, row, plan), []
    if row.way.startswith("claude_"):
        return _claude(d, plan), []
    return await _kimi(d, row, plan), []


async def _openai(d: Deployment, row: Row, plan: list[Call]) -> Any:
    """An OpenAI agent calling its own: an ``MCPServerStdio`` of its own, or a
    ``FunctionTool``."""
    from agents import Agent
    from agents.mcp import MCPServerStdio

    if row.way == "openai_function_tool":
        own = openai_function_tools(d.ran)
        return Agent(name="w", instructions=SYSTEM, model=PlannedModel(plan), tools=own)
    server = MCPServerStdio(
        params={"command": sys.executable, "args": [str(SERVER)], "env": {"RAN": str(d.ran)}},
        name="ledger",
        client_session_timeout_seconds=30,
    )
    await d.servers.enter_async_context(server)
    return Agent(name="w", instructions=SYSTEM, model=PlannedModel(plan), mcp_servers=[server])


def _swarm(plan: list[Call], tools: list[Any]) -> Any:
    """A langgraph-swarm graph of two ``ReAct`` agents, as the library builds one: the ledger
    agent (active first) follows ``plan`` with ``tools`` and can hand over to the desk."""
    from langgraph.graph import StateGraph
    from langgraph_swarm import SwarmState, add_active_agent_router, create_handoff_tool

    members = {
        "ledger": ReAct(
            "You work the ledger.",
            PlannedChat(plan),
            tools=[*tools, create_handoff_tool(agent_name="desk")],
        ),
        "desk": ReAct(
            "You are the front desk.",
            PlannedChat([]),
            tools=[create_handoff_tool(agent_name="ledger")],
        ),
    }
    builder = StateGraph(SwarmState)
    for name, member in members.items():
        builder.add_node(name, member, destinations=tuple(set(members) - {name}))
    swarm = add_active_agent_router(builder, route_to=list(members), default_active_agent="ledger")
    return swarm.compile(checkpointer=RunCheckpointer())


def _langchain(adapter: str, plan: list[Call], own: list[Any]) -> Any:
    """A LangChain-based framework's agent with tools of its own."""
    from deepagents import create_deep_agent
    from langchain.agents import create_agent

    if adapter == "react":
        return ReAct(system_prompt=SYSTEM, model=PlannedChat(plan), tools=own)
    if adapter == "swarm":
        return _swarm(plan, own)
    model = PlannedChatModel(plan=plan)
    if adapter == "langgraph":
        return create_agent(model, tools=own, system_prompt=SYSTEM)
    if adapter == "middleware":
        return create_agent(
            model,
            tools=own,
            system_prompt=SYSTEM,
            middleware=middleware(),
            checkpointer=RunCheckpointer(),
        )
    if adapter == "deepagents":
        return create_deep_agent(model=model, tools=own, system_prompt=SYSTEM)
    from langchain_core.messages import SystemMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    bound = model.bind_tools(own)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke([SystemMessage(SYSTEM), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(own))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


def _claude(d: Deployment, plan: list[Call]) -> Any:
    """Claude Code (the scripted CLI) calling its own: the in-process server's tools, or its
    built-ins."""
    from claude_agent_sdk import ClaudeAgentOptions

    from tests.support.adapters import CLI

    script: list[dict[str, Any]] = []
    for name, args in plan:
        if name.startswith("mcp__ledger__"):
            script.append(
                {"tool": name.removeprefix("mcp__ledger__"), "server": "ledger", "args": args}
            )
        else:
            script.append({"builtin": name, "args": args})
    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps([*script, {"text": FINAL}]),
        "FAKE_CLAUDE_BUILTINS": str(d.tmp / "builtins.jsonl"),
    }
    return ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env=env,
        mcp_servers={"ledger": claude_server(d.ran)},
    )


async def _kimi(d: Deployment, row: Row, plan: list[Call]) -> Any:
    """Kimi Code (``kimi acp``) on a planned endpoint, calling its own: a stdio MCP server of
    its own, or its built-ins."""
    from tests.support.kimi import KIMI_BIN, PlannedEndpoint, kimi_target, needs_kimi

    if KIMI_BIN is None and needs_kimi.args[0]:
        pytest.skip(needs_kimi.kwargs["reason"])
    endpoint = await PlannedEndpoint(plan).attached()
    servers: list[dict[str, Any]] = []
    if row.way == "kimi_mcp_server":
        servers.append(
            {
                "name": "ledger",
                "command": sys.executable,
                "args": [str(SERVER)],
                "env": [{"name": "RAN", "value": str(d.ran)}],
            }
        )
    return kimi_target(endpoint, d.tmp, mcp_servers=servers)


async def walked(
    row: Row, tmp: Path, script: Script, overrides: dict[str, str] | None = None
) -> None:
    """``script``'s runs of the row's agent — a new agent each run, as a later turn is — each
    call asked (with what it is offered) or not, and its body run or not, as :class:`Rule`
    says; ``overrides``: the risks an administrator set in the catalog."""
    way = WAYS[row.way]
    rule = Rule({kind: kind for kind in ALL_KINDS} | (overrides or {}))
    async with deployed(row, tmp) as d:
        for kind, risk in (overrides or {}).items():
            if kind in way.kinds:
                name = way.named(kind)
                d.memory.catalog[name] = {"name": name, "side_effects": risk}
        said: dict[str, list[Call]] = {}
        for number, (label, planned) in enumerate(script):
            steps = [s for s in planned if s[0] in way.kinds]
            if not steps or (label == LATER and not row.conversations):
                continue
            conversation = "c1" if label == LATER else label
            thread = conversation if row.conversations else None
            calls = [(way.named(kind), way.args(n)) for kind, n, _ in steps]
            earlier = [] if row.turn_by_turn or thread is None else said.get(thread, [])
            if thread is not None:
                said[thread] = [*earlier, *calls]
            scope = conversation if row.conversations else f"run-{number}"
            plan = [*earlier, *calls]
            await _ran(
                d, row, rule, plan=plan, steps=steps, thread=thread, scope=scope, number=number
            )


async def _ran(
    d: Deployment,
    row: Row,
    rule: Rule,
    *,
    plan: list[Call],
    steps: list[Step],
    thread: str | None,
    scope: str,
    number: int,
) -> None:
    """One run of ``plan`` in ``thread``: each of ``steps`` asked, answered and run as
    ``rule`` says; then what the harness published kept in the catalog, as the service does."""
    way = WAYS[row.way]

    def ran() -> list[str]:
        if way.ran is not None:
            return way.ran(d.tmp)
        return d.ran.read_text().splitlines() if d.ran.exists() else []

    target, tools = await built(d, row, plan)
    agent = arm(d.h.wrap(target, id=f"ledger-{row.adapter}-{number}"), tools)
    before = len(ran())
    result = await agent.run("work the ledger", user="ada", thread=thread)
    expected: list[str] = []
    for kind, n, answer in steps:
        offered, runs = rule.call(kind, scope, answer)
        name = way.named(kind)
        if offered:
            _asked(result, name, rule.risks[kind], way.args(n), offered, asker=way.asker)
            assert result.interrupt is not None
            result = await agent.resume(
                result.interrupt.interrupt_id, "answer", answer=answer, reviewer="ada"
            )
        if runs:
            expected.append(f"{name.removeprefix('mcp__ledger__').removeprefix('ledger-')} {n}")
    unasked = result.interrupt and (result.interrupt.tool_call, result.interrupt.question)
    assert result.status is RunStatus.SUCCESS, f"{scope}: {unasked!r} {result.error}"
    assert ran()[before:] == expected, f"{scope}: ran {ran()[before:]}, not {expected}"
    await d.h.writes.drain()  # the always kept; what was published, stored
    for call in d.memory.named("put_catalog"):
        for entry in call.body["tools"]:
            d.memory.catalog.setdefault(entry["name"], dict(entry))


def _asked(
    result: Any,
    name: str,
    risk: str,
    args: dict[str, Any],
    offered: tuple[Any, ...],
    *,
    asker: str | None,
) -> None:
    """``result`` paused for a person on this call of ``name``, offered ``offered``: asked
    because of its risk, or — a framework's own tool — because its framework asks (``asker``)."""
    interrupt = result.interrupt
    found = interrupt and (
        interrupt.reason,
        interrupt.tool_call and (interrupt.tool_call.tool, interrupt.tool_call.args),
        [o.value for o in interrupt.options],
        interrupt.question,
    )
    why = [f"{name} {WHY[risk]}"] + ([FRAMEWORK_ASKS.format(framework=asker)] if asker else [])
    questions = [f"Approve {name}? {reason} {SAID[offered]}" for reason in why]
    expected = (InterruptReason.APPROVAL, (name, args), [o.value for o in offered])
    assert found is not None and found[:3] == expected and found[3] in questions, (
        f"asked {found!r}, not {expected!r} with one of {questions} "
        f"({result.status}, {result.error})"
    )


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
async def test_one_rule(row: Row, tmp_path: Path) -> None:
    await walked(row, tmp_path, SCRIPT)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
async def test_the_catalogs_risk_wins(row: Row, tmp_path: Path) -> None:
    await walked(row, tmp_path, OVERRIDDEN, OVERRIDES)
