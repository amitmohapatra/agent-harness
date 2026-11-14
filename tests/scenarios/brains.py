"""Every adapter, its model a *brain*: a scripted model that answers from the request it is sent,
never from a script consumed call by call, so concurrent runs, steered messages and resumed runs
of one agent each get their own answer.

A user message carries *tokens* (``tok_<chat>_<n>``, :func:`token`) and *commands*
(``do:<tool>:<json args>``, :func:`command`). Given a request, the brain finds the latest user
message holding a token, makes that message's commands in order (counting the tool results after
it), then answers::

    ANS <THE MESSAGE'S TOKEN> SEEN <EVERY OTHER TOKEN THE REQUEST HOLDS> LAST <last result>

in capitals, so an answer kept in the conversation is never read as a token again. What the
answer says is what reached the model: the earlier turns (memory's window, the framework's
session, the run store's replay), and no other chat's or user's.

The Claude Agent SDK is driven by the scripted CLI (``tests/support/fake_claude_cli.py``), whose
"model" follows a script fixed when the agent is built: it answers ``ANS {heard} CTX {context}``
(every prompt its session was given, and what the harness's hooks added) — lower case, parsed by
:func:`parse` as the same answer. Kimi Code is the real ``kimi acp`` against an endpoint the
brain answers (``tests.support.kimi.PlannedEndpoint(turns=)``).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
from agents import Agent as OpenAIAgent
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from tests.support.arm import arm, native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.models import BASE_URL, Script, ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, Runtime, middleware
from trellis.harness.agent import Agent
from trellis.harness.automatic import JournalBackend
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.tools.convert import text_of

CLI: Final = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
TOKEN: Final = re.compile(r"tok_[a-z0-9]+_[0-9]+")
COMMAND: Final = re.compile(r"do:([a-z_]+):(?=\{)")
SLOW: Final = re.compile(r"slow:([0-9.]+)")
#: ``loop:<n>:<tool>``: that many calls of a tool, each with ``{"key": "k<i>"}`` (a long task)
LOOP: Final = re.compile(r"loop:([0-9]+):([a-z_]+)")
#: ``fail:model``: the model's provider fails the message's first call (a 500)
FAIL: Final = "fail:model"
UUID: Final = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
#: what the harness tells a model whose call a reviewer edited, where the framework cannot take
#: the edited arguments in place (OpenAI Agents, ACP): make the call again with them — a model
#: told so does (``adapters.openai_agents``, ``adapters.acp``)
REDO: Final = re.compile(
    r"Call (\S+) again(?: now)? with exactly these arguments: (\{.*\})", re.DOTALL
)
ANSWER: Final = re.compile(r"ANS (\S+) SEEN (\S*) MEM (\S*) LAST (.*)", re.DOTALL)
#: the long-term memories of a memory context (user-level recall), apart from the conversation
MEMORIES: Final = re.compile(r"## Memories\n(.*?)(?=\n## |</memory_context>|$)", re.DOTALL)
SYSTEM: Final = "You help the user."
#: what the scripted Claude CLI answers (its placeholders filled in by the CLI)
CLAUDE_ANSWER: Final = "ANS {heard} CTX {context} LAST {last}"

#: every adapter the scenarios run over (``react_da``: ReAct on the Deep Agents backend)
ADAPTERS: Final = (
    "function",
    "react",
    "react_da",
    "langgraph",
    "middleware",
    "graph",
    "deepagents",
    "openai_agents",
    "claude_agent_sdk",
    "kimi",
)
#: the adapters whose framework takes a steered message at its next step (``Adapter.steers``)
STEERS: Final = frozenset(
    {"react", "react_da", "middleware", "openai_agents", "claude_agent_sdk", "deepagents"}
)


def token(chat: str, n: int) -> str:
    return f"tok_{chat}_{n}"


def command(tool: str, **args: Any) -> str:
    return f"do:{tool}:{json.dumps(args, separators=(',', ':'))}"


# --------------------------------------------------------------------------- the brain
@dataclass
class Turned:
    """What the brain decided for one request: a call, or the answer."""

    call: Any = None  # (tool, args), or a list of them made in one message
    answer: str = ""
    slow: float = 0.0
    fail: bool = False


def decide(messages: Sequence[tuple[str, str]]) -> Turned:
    """``messages``: the request as (role, text) pairs, in order (``system`` included); a tool
    result's role is ``tool``."""
    latest = -1
    for n, (role, text) in enumerate(messages):
        if role == "user" and TOKEN.search(text):
            latest = n
    if latest < 0:
        everything = " ".join(t for _, t in messages)
        return Turned(answer=f"ANS NONE SEEN {_seen(everything, None)} MEM  LAST ")
    # the message itself, not the memory context before it; its token is the first it holds
    asked = messages[latest][1].split("</memory_context>")[-1]
    mine = (TOKEN.findall(asked) or TOKEN.findall(messages[latest][1]))[0]
    asked = asked[asked.find(mine) :]
    results = [t for role, t in messages[latest + 1 :] if role == "tool"]
    commands = _commands(asked)
    for n, tool in LOOP.findall(asked):
        commands += [(tool, {"key": f"k{i}"}) for i in range(int(n))]
    task = UUID.findall(" ".join(t for _, t in messages))
    commands = [
        (name, {k: (task[-1] if v == "{task}" and task else v) for k, v in args.items()})
        for name, args in commands
    ]
    slow = SLOW.search(asked)
    if FAIL in asked and not results:
        return Turned(fail=True)
    redo = REDO.search(results[-1]) if results else None
    if redo is not None:  # a reviewer's edit to make: the call again, with those arguments
        try:
            return Turned(call=(redo[1].removeprefix("mcp__trellis__"), json.loads(redo[2])))
        except ValueError:
            pass
    done = 0
    for step in _steps(commands):  # a step: one call, or several in one message (``batch``)
        size = len(step) if isinstance(step, list) else 1
        if len(results) < done + size:
            pending = step[len(results) - done :] if isinstance(step, list) else step
            return Turned(call=pending, slow=float(slow[1]) if slow and not results else 0)
        done += size
    everything = " ".join(t for _, t in messages)
    recalled = " ".join(MEMORIES.findall(everything))
    conversation = MEMORIES.sub("", everything)
    last = results[-1] if results else ""
    return Turned(
        answer=f"ANS {mine.upper()} SEEN {_seen(conversation, mine)} MEM {_seen(recalled, mine)} LAST {last}",
        slow=float(slow[1]) if slow and not commands else 0.0,
    )


def _commands(asked: str) -> list[tuple[str, dict[str, Any]]]:
    """Each ``do:<tool>:<json>`` of a message, its JSON read whole (nested braces and all); a
    command inside another's JSON (a task's description) is that command's, not the message's."""
    decoder = json.JSONDecoder()
    found: list[tuple[str, dict[str, Any]]] = []
    spans: list[tuple[int, int]] = []
    for match in COMMAND.finditer(asked):
        if any(a <= match.start() < b for a, b in spans):
            continue
        try:
            args, end = decoder.raw_decode(asked, match.end())
        except ValueError:
            continue
        spans.append((match.start(), end))
        found.append((match[1], args))
    return found


def _steps(commands: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    """``do:batch:{"calls": [[tool, args], ...]}`` is several calls in one message."""
    return [
        [(c[0], c[1]) for c in args["calls"]] if name == "batch" else (name, args)
        for name, args in commands
    ]


def text(content: Any) -> str:
    """A message's text: a string, or the text parts of a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, list | tuple):
        return "\n".join(text(p) if not isinstance(p, str) else p for p in content)
    found = getattr(content, "text", None)
    return found if isinstance(found, str) else str(content)


def _seen(text: str, mine: str | None) -> str:
    return ",".join(sorted({t.upper() for t in TOKEN.findall(text)} - {(mine or "").upper()}))


@dataclass
class Parsed:
    mine: str
    #: the tokens of the conversation (its history) the model was given
    seen: set[str] = field(default_factory=set)
    last: str = ""
    #: the tokens of the user's long-term memories (recall across chats) it was given
    mem: set[str] = field(default_factory=set)


def parse(answer: str | None) -> Parsed:
    """An answer of the brain (or of the scripted Claude CLI) as the token it answers, the other
    tokens it saw, and the last tool result it read."""
    text = answer or ""
    found = ANSWER.search(text)
    if found:
        seen = {s.lower() for s in found[2].split(",") if s}
        mem = {s.lower() for s in found[3].split(",") if s}
        return Parsed(found[1].lower(), seen, found[4].strip(), mem)
    # Claude: ANS <every prompt heard> CTX <context> LAST <last result>
    heard, _, rest = text.partition(" CTX ")
    tokens = TOKEN.findall(heard)
    context, _, last = rest.partition(" LAST ")
    mine = tokens[-1] if tokens else "none"
    recalled = set(TOKEN.findall(" ".join(MEMORIES.findall(context)))) - {mine}
    seen = (set(tokens) | set(TOKEN.findall(MEMORIES.sub("", context)))) - {mine}
    return Parsed(mine, seen, last.strip(), recalled)


class ProviderFailure(RuntimeError):
    """What a scripted provider raises for ``fail:model`` (a 500 from its API)."""


# --------------------------------------------------------------------------- per wire format
def _openai_messages(body: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m.get("role", ""), text(m.get("content"))) for m in body.get("messages") or []]


class _BrainScript(Script):
    """The chat-completions endpoint behind ``ReAct``'s ``ChatOpenAI``; its call ids its own, as
    a model's are (a run resumed in another process has a model of its own)."""

    def __init__(self) -> None:
        super().__init__([])
        self.prefix = uuid.uuid4().hex[:8]

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        answered = super().reply(body)
        for call in answered["choices"][0]["message"].get("tool_calls") or []:
            call["id"] = f"call_{self.prefix}_{call['id'].removeprefix('call_')}"
        return answered

    def next(self, body: dict[str, Any]) -> Any:
        turned = decide(_openai_messages(body))
        offered = bool(body.get("tools")) and body.get("tool_choice") != "none"
        return (
            turned.call if turned.call and offered else turned.answer or "ANS NONE SEEN  MEM  LAST "
        )


class BrainChat(ScriptedChat):
    """A gateway model answered by the brain; a ``slow:<s>`` message stalls its first call."""

    def __init__(self) -> None:
        script = _BrainScript()

        async def handle(request: httpx.Request) -> httpx.Response:
            script.arrived.set()
            turned = decide(_openai_messages(json.loads(request.content)))
            if turned.slow:
                await asyncio.sleep(turned.slow)
            if turned.fail:
                return httpx.Response(
                    500,
                    json={
                        "error": {"message": "scripted provider failure", "type": "server_error"}
                    },
                )
            return script.handle(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        ChatOpenAI.__init__(  # type: ignore[call-arg]
            self,
            model="scripted",
            api_key="scripted",  # type: ignore[arg-type]
            base_url=BASE_URL,
            max_retries=0,
            http_async_client=client,
            script=script,  # type: ignore[call-arg]
        )

    def said(self) -> str:
        return json.dumps([r["messages"] for r in self.requests], default=str)


def _langchain_messages(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    roles = {
        HumanMessage: "user",
        AIMessage: "assistant",
        ToolMessage: "tool",
        SystemMessage: "system",
    }
    out = []
    for m in messages:
        role = next((r for kind, r in roles.items() if isinstance(m, kind)), "other")
        out.append((role, text(m.content)))
    return out


class BrainChatModel(ScriptedChatModel):
    """A LangChain chat model answered by the brain (``create_agent``, a graph, Deep Agents)."""

    turns: list[Any] = Field(default_factory=list)

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:  # type: ignore[override]
        turned = decide(_langchain_messages(messages))
        if turned.slow:
            await asyncio.sleep(turned.slow)
        if turned.fail:
            raise ProviderFailure("scripted provider failure (500)")
        return self._generate(messages, stop, **kwargs)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:  # type: ignore[override]
        turned = decide(_langchain_messages(messages))
        offered = bool(kwargs.get("tools")) and kwargs.get("tool_choice") != "none"
        if "Messages to summarize" in (text(messages[-1].content) if messages else ""):
            self.turns = ["Summary: the conversation so far."]
        elif turned.call and offered:
            self.turns = [turned.call]
        else:
            self.turns = [turned.answer or "ANS NONE SEEN  MEM  LAST "]
        return super()._generate(messages, stop, run_manager, **kwargs)

    def said(self) -> str:
        return "\n".join(text(m.content) for call in self.seen for m in call)


def _item_text(item: Any) -> tuple[str, str]:
    def get(key: str) -> Any:
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    if get("type") == "function_call_output":
        return "tool", text(get("output"))
    role = get("role") or get("type") or ""
    return str(role), text(get("content"))


class BrainModel(ScriptedModel):
    """An OpenAI Agents SDK model answered by the brain."""

    def __init__(self) -> None:
        super().__init__([], ids=f"{uuid.uuid4().hex[:8]}_")

    def _request(self, input: Any, system: str | None) -> list[tuple[str, str]]:
        items = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
        return [("system", system or ""), *(_item_text(i) for i in items)]

    def _next(self, input: Any, system: str | None) -> list[Any]:
        turned = decide(self._request(input, system))
        self.turns = [
            turned.call
            if turned.call and not self.bare
            else turned.answer or "ANS NONE SEEN  MEM  LAST "
        ]
        return super()._next(input, system)

    async def get_response(
        self, system_instructions: str | None, input: Any, *args: Any, **kwargs: Any
    ) -> Any:  # type: ignore[override]
        turned = decide(self._request(input, system_instructions))
        if turned.slow:
            await asyncio.sleep(turned.slow)
        return await super().get_response(system_instructions, input, *args, **kwargs)

    async def stream_response(  # type: ignore[override]
        self, system_instructions: str | None, input: Any, *args: Any, **kwargs: Any
    ) -> Any:
        turned = decide(self._request(input, system_instructions))
        if turned.slow:
            await asyncio.sleep(turned.slow)
        async for event in super().stream_response(system_instructions, input, *args, **kwargs):
            yield event

    def said(self) -> str:
        return json.dumps([self.system, self.inputs], default=str)


def _kimi_messages(body: dict[str, Any]) -> list[tuple[str, str]]:
    from tests.support.kimi import result_text

    return [
        (
            m.get("role", ""),
            result_text(m.get("content")) if m.get("role") == "tool" else text(m.get("content")),
        )
        for m in body.get("messages") or []
    ]


def _brain_endpoint() -> type:
    from starlette.responses import JSONResponse

    from tests.support.kimi import PlannedEndpoint

    class BrainEndpoint(PlannedEndpoint):
        """Kimi's model endpoint answered by the brain: ``slow:`` stalls, ``fail:model`` a 500."""

        async def _chat(self, request: Any) -> Any:
            turned = decide(_kimi_messages(await request.json()))
            if turned.slow:
                await asyncio.sleep(turned.slow)
            if turned.fail:
                return JSONResponse(
                    {"error": {"message": "scripted provider failure"}}, status_code=500
                )
            return await super()._chat(request)

    return BrainEndpoint


def kimi_turn(body: dict[str, Any]) -> Any:
    """``PlannedEndpoint(turns=)``: Kimi's request answered by the brain (its tools named as
    Kimi names the harness's: ``mcp__trellis__<tool>``)."""
    from tests.support.kimi import kimi_name, result_text

    turned = decide(
        [
            (
                m.get("role", ""),
                result_text(m.get("content"))
                if m.get("role") == "tool"
                else text(m.get("content")),
            )
            for m in body.get("messages") or []
        ]
    )
    if turned.call and body.get("tools"):
        if isinstance(turned.call, list):
            return [(kimi_name(n), a) for n, a in turned.call]
        name, args = turned.call
        return (kimi_name(name), args)
    return turned.answer or "ANS NONE SEEN  MEM  LAST "


# --------------------------------------------------------------------------- the targets
@dataclass
class Built:
    target: Any
    tools: list[Any]
    #: the model (or the Claude CLI's record file, or Kimi's endpoint): what it was sent
    model: Any = None


async def _function(
    h: Harness, tools: list[Any], tmp: Path, claude: list[dict[str, Any]] | None
) -> Built:
    async def follow(input: Any, agent: Runtime) -> str:
        asked = text(input)
        messages: list[tuple[str, str]] = [("system", agent.context or "")]
        for said in agent.history or ():
            messages.append((str(said.role), said.content))
        messages.append(("user", asked))
        while True:
            turned = decide(messages)
            if turned.slow:
                await asyncio.sleep(turned.slow)
            if turned.fail:
                raise ProviderFailure("scripted provider failure (500)")
            if turned.call is None:
                return turned.answer
            for name, args in turned.call if isinstance(turned.call, list) else [turned.call]:
                messages.append(("tool", text_of(await agent.tools.call(name, **args)) or ""))

    return Built(follow, tools)


async def _react(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    model = BrainChat()
    return Built(ReAct(system_prompt=SYSTEM, model=model), tools, model)


async def _react_da(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    model = BrainChat()
    return Built(ReAct(system_prompt=SYSTEM, model=model, backend=JournalBackend()), tools, model)


async def _langgraph(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    model = BrainChatModel()
    built = await native_tools(h, *tools, framework="langgraph")
    return Built(create_agent(model, tools=built, system_prompt=SYSTEM), tools, model)


async def _middleware(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    model = BrainChatModel()
    graph = create_agent(
        model, system_prompt=SYSTEM, middleware=middleware(), checkpointer=RunCheckpointer()
    )
    return Built(graph, tools, model)


async def _graph(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    model = BrainChatModel()
    built = await native_tools(h, *tools, framework="langgraph")
    bound = model.bind_tools(built)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke([SystemMessage(SYSTEM), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(built))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return Built(graph.compile(), tools, model)


async def _deepagents(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    model = BrainChatModel()
    built = await native_tools(h, *tools, framework="deepagents")
    return Built(create_deep_agent(model=model, tools=built, system_prompt=SYSTEM), tools, model)


async def _openai_agents(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    model = BrainModel()
    return Built(OpenAIAgent(name="helper", instructions=SYSTEM, model=model), tools, model)


async def _claude(
    h: Harness, tools: list[Any], tmp: Path, claude: list[dict[str, Any]] | None
) -> Built:
    """The scripted CLI: ``claude`` is the steps of every turn before the answer (tool calls,
    stalls); its answer is :data:`CLAUDE_ANSWER`."""
    record = tmp / f"cli-{uuid.uuid4().hex[:6]}.json"
    script = [*(claude or []), {"text": CLAUDE_ANSWER}]
    options = ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_RECORD": str(record),
            "FAKE_CLAUDE_SESSIONS": str(tmp / "claude-sessions"),
        },
    )
    return Built(options, tools, record)


async def _kimi(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    from tests.support.kimi import KIMI_BIN, kimi_target

    BrainEndpoint = _brain_endpoint()

    if KIMI_BIN is None:
        import pytest

        pytest.skip("the kimi program is not on PATH (see tests/scenarios/README)")
    endpoint = await BrainEndpoint(turns=kimi_turn).attached()
    return Built(kimi_target(endpoint, tmp / f"kimi-{uuid.uuid4().hex[:6]}"), tools, endpoint)


BUILDERS: Final = {
    "function": _function,
    "react": _react,
    "react_da": _react_da,
    "langgraph": _langgraph,
    "middleware": _middleware,
    "graph": _graph,
    "deepagents": _deepagents,
    "openai_agents": _openai_agents,
    "claude_agent_sdk": _claude,
    "kimi": _kimi,
}


async def wrapped(
    h: Harness,
    adapter: str,
    tools: Sequence[Any],
    tmp: Path,
    *,
    id: str,
    claude: list[dict[str, Any]] | None = None,
    **wrap: Any,
) -> tuple[Agent, Built]:
    """``adapter``'s agent, wrapped by ``h`` as ``id``, with ``tools`` (``claude``: the scripted
    CLI's steps before its answer)."""
    built = await BUILDERS[adapter](h, list(tools), tmp, claude)
    return arm(h.wrap(built.target, id=id, **wrap), built.tools), built
