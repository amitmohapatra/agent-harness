"""The arms of the head-to-head: each framework alone (A), the same agent through ``h.wrap``
(B), and our ``ReAct`` (C), with each of its features removed in turn. Every arm is built from
the same :class:`Setup` — one model on the metering proxy, the task set's instructions and its
tools (the same tool code: a native arm calls ``Tool.run`` itself, a wrapped one through the
harness) — so only the framework varies.

* **A, native** — set up as each framework's own documentation does it: LangChain's
  ``create_agent``, ``create_deep_agent``, the OpenAI Agents SDK's ``Agent`` on an
  ``OpenAIChatCompletionsModel``, ``ClaudeAgentOptions`` with an in-process MCP server, and a
  CrewAI ``Crew`` (in an interpreter of its own: CrewAI is installed in a venv of its own, and
  has no wrapped arm). Where a framework has a signature feature, an ``A+<feature>`` arm
  turns it on: LangChain's summarization, Deep Agents' planning, the OpenAI Agents SDK's
  session.
* **B, wrapped** — the A agent built with ``h.tools(...)`` (or given them on ``h.wrap``) and
  run through ``h.evaluate``: memory, the run store, governance, tool narrowing on.
* **C, ReAct** — ``ReAct(...)`` through ``h.wrap``, and ``C-<feature>`` with
  ``without={feature}`` for each of :data:`REMOVED`.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from trellis.contracts import ToolCall, ToolOutcome, ToolStatus
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import text_of

Kind = Literal["A", "B", "C"]
#: The frameworks, by the name the report uses.
FRAMEWORKS: Final = (
    "langchain",
    "deepagents",
    "openai_agents",
    "claude_agent_sdk",
    "crewai",
    "trueforge",
    "react",
)
#: The features a ``C-<feature>`` arm runs ``ReAct`` without.
REMOVED: Final = ("tool_search", "planning", "files", "code_mode", "memory")
#: What ``ReAct`` adds to its loop on a run's signal (``trellis.harness.adaptive``): ``C0`` runs
#: it without them all (``ReAct`` as main builds it), a ``C+<item>`` arm with that one alone.
ADAPTIVE: Final = (
    "token_count",
    "strict_tools",
    "native_output",
    "tool_repair",
    "stable_prefix",
    "think",
    "clear_inputs",
    "unsupported_content",
    "rubric",
)
#: The name the agent goes by where a framework asks for one.
NAME: Final = "assistant"


@dataclass(frozen=True)
class Arm:
    name: str
    framework: str
    kind: Kind
    description: str
    #: A: the signature feature turned on; C: the feature removed
    feature: str | None = None
    #: the interpreter it runs in: the harness's, or CrewAI's own venv
    venv: Literal["main", "crewai"] = "main"
    #: C: the adaptive items it runs with (:data:`ADAPTIVE`; ``None``: all of them)
    adds: tuple[str, ...] | None = None

    @property
    def wrapped(self) -> bool:
        """Whether it runs through the harness (memory and the run store on)."""
        return self.kind != "A"

    @property
    def without(self) -> set[str]:
        """C: what its ``ReAct`` runs without — its removed feature, and the adaptive items
        it does not add."""
        removed = {self.feature} if self.feature else set()
        if self.adds is None:
            return removed
        return removed | (set(ADAPTIVE) - set(self.adds))


def _arms() -> dict[str, Arm]:
    arms = [
        Arm("langchain/A", "langchain", "A", "LangChain create_agent, alone"),
        Arm(
            "langchain/A+summarization",
            "langchain",
            "A",
            "create_agent with SummarizationMiddleware",
            feature="summarization",
        ),
        Arm("deepagents/A", "deepagents", "A", "create_deep_agent, alone"),
        Arm(
            "deepagents/A+planning",
            "deepagents",
            "A",
            "create_deep_agent with TodoListMiddleware (write_todos)",
            feature="planning",
        ),
        Arm("openai_agents/A", "openai_agents", "A", "OpenAI Agents SDK Agent + Runner, alone"),
        Arm(
            "openai_agents/A+session",
            "openai_agents",
            "A",
            "the same with a SQLiteSession (compaction needs OpenAI's Responses API)",
            feature="session",
        ),
        Arm("claude_agent_sdk/A", "claude_agent_sdk", "A", "Claude Agent SDK query(), alone"),
        Arm("crewai/A", "crewai", "A", "CrewAI Agent + Task + Crew, alone", venv="crewai"),
        Arm("trueforge/A", "trueforge", "A", "TrueForge (local mode), deferred tool loading"),
        Arm(
            "trueforge/A+preload",
            "trueforge",
            "A",
            "TrueForge with every tool schema preloaded (deferred loading off)",
            feature="preload",
        ),
        Arm("langchain/B", "langchain", "B", "create_agent through h.wrap"),
        Arm("deepagents/B", "deepagents", "B", "create_deep_agent through h.wrap"),
        Arm("openai_agents/B", "openai_agents", "B", "OpenAI Agents SDK Agent through h.wrap"),
        Arm("claude_agent_sdk/B", "claude_agent_sdk", "B", "ClaudeAgentOptions through h.wrap"),
        Arm("react/C", "react", "C", "ReAct through h.wrap, every feature on"),
        *(
            Arm(f"react/C-{f}", "react", "C", f"ReAct without={{{f!r}}}", feature=f)
            for f in REMOVED
        ),
        Arm(
            "react/C0",
            "react",
            "C",
            "ReAct without its adaptive items (as main builds it)",
            adds=(),
        ),
        *(
            Arm(
                f"react/C+{item}",
                "react",
                "C",
                f"ReAct with the adaptive {item} alone",
                adds=(item,),
            )
            for item in ADAPTIVE
        ),
    ]
    return {arm.name: arm for arm in arms}


ARMS: Final[dict[str, Arm]] = _arms()


@dataclass(frozen=True)
class Setup:
    """What every arm is built from: the model on the metering proxy."""

    model: str
    #: the OpenAI-compatible base URL (the proxy's ``/v1``)
    base_url: str
    key: str
    #: the proxy's root (Anthropic's ``/anthropic`` hangs off it)
    root: str
    window: int = 8192
    max_steps: int = 6

    @classmethod
    def on(cls, root: str, model: str, key: str, *, window: int = 8192) -> Setup:
        """The setup for the proxy at ``root`` (no path)."""
        root = root.rstrip("/")
        return cls(model=model, base_url=f"{root}/v1", key=key, root=root, window=window)


# --------------------------------------------------------------------------- native tools
@dataclass
class Calls:
    """The tool calls a native run made, in order, each with its outcome: the trajectory its
    scores read (a wrapped run's is the harness's journal)."""

    made: list[tuple[ToolCall, ToolOutcome]] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [call.tool for call, _ in self.made]

    def reset(self) -> None:
        self.made = []


def invoker(tool: Tool, calls: Calls) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """The tool's own code, as a native framework calls it: the call recorded, the result as
    text, an error as text the model reads (every framework the same way)."""

    async def call(args: dict[str, Any]) -> str:
        step = len(calls.made) + 1
        try:
            output = await tool.run(dict(args))
        except Exception as exc:
            text = f"Error: {type(exc).__name__}: {exc}"
            outcome = ToolOutcome(tool=tool.name, status=ToolStatus.ERROR, output=text)
            calls.made.append((ToolCall(tool=tool.name, args=dict(args), step=step), outcome))
            return text
        calls.made.append(
            (
                ToolCall(tool=tool.name, args=dict(args), step=step),
                ToolOutcome(tool=tool.name, output=output),
            )
        )
        return text_of(output)

    return call


def schema_of(tool: Tool) -> dict[str, Any]:
    schema = dict(tool.spec.input_schema or {})
    return {**schema, "type": "object", "properties": schema.get("properties") or {}}


def as_langchain(tools: Sequence[Tool], calls: Calls) -> list[Any]:
    from langchain_core.tools import StructuredTool

    made = []
    for tool in tools:
        run = invoker(tool, calls)

        async def coroutine(_run: Any = run, **args: Any) -> str:
            return await _run(args)

        made.append(
            StructuredTool.from_function(
                coroutine=coroutine,
                name=tool.name,
                description=tool.spec.description or tool.name,
                args_schema=schema_of(tool),
            )
        )
    return made


def as_openai_agents(tools: Sequence[Tool], calls: Calls) -> list[Any]:
    from agents import FunctionTool

    made = []
    for tool in tools:
        run = invoker(tool, calls)

        async def invoke(_context: Any, arguments: str, _run: Any = run) -> str:
            try:
                args = json.loads(arguments or "{}")
            except ValueError as exc:
                return f"Error: the arguments are not JSON: {exc}"
            return await _run(args if isinstance(args, dict) else {})

        made.append(
            FunctionTool(
                name=tool.name,
                description=tool.spec.description or tool.name,
                params_json_schema=schema_of(tool),
                on_invoke_tool=invoke,
                strict_json_schema=False,
            )
        )
    return made


#: The in-process MCP server's name in a native Claude arm.
SERVER: Final = "tools"


def as_claude(tools: Sequence[Tool], calls: Calls) -> Any:
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    made = []
    for tool in tools:
        run = invoker(tool, calls)

        async def handler(args: dict[str, Any], _run: Any = run) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": await _run(dict(args))}]}

        made.append(
            sdk_tool(tool.name, tool.spec.description or tool.name, schema_of(tool))(handler)
        )
    return create_sdk_mcp_server(SERVER, tools=made)


# --------------------------------------------------------------------------- the models
def chat_model(setup: Setup) -> Any:
    """LangChain's chat model on the proxy, as LangChain's docs set one up."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=setup.model, base_url=setup.base_url, api_key=setup.key)  # type: ignore[arg-type]


def agents_model(setup: Setup) -> Any:
    """The OpenAI Agents SDK's chat-completions model on the proxy."""
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=setup.base_url, api_key=setup.key)
    return OpenAIChatCompletionsModel(model=setup.model, openai_client=client)


def claude_env(setup: Setup) -> dict[str, str]:
    """The Claude Code CLI on the proxy's Anthropic route, nothing else reached."""
    return {
        "ANTHROPIC_BASE_URL": f"{setup.root}/anthropic",
        "ANTHROPIC_API_KEY": setup.key,
        "ANTHROPIC_MODEL": setup.model,
        "ANTHROPIC_SMALL_FAST_MODEL": setup.model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
    }


# --------------------------------------------------------------------------- arm A
Runner = Callable[[str], Awaitable[str]]


def native(arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls) -> Runner:
    """Arm A: ``async (task) -> answer`` running the framework alone on ``tools``."""
    builders = {
        "langchain": _langchain,
        "deepagents": _deepagents,
        "openai_agents": _openai_agents,
        "claude_agent_sdk": _claude,
        "crewai": _crewai,
        "trueforge": _trueforge,
    }
    if arm.kind != "A" or arm.framework not in builders:
        raise ValueError(f"{arm.name} is not a native arm of this interpreter")
    return builders[arm.framework](arm, setup, system, tools, calls)


def _graph_runner(graph: Any) -> Runner:
    async def run(task: str) -> str:
        state = await graph.ainvoke({"messages": [{"role": "user", "content": task}]})
        return answer_of_messages(state.get("messages", []))

    return run


def answer_of_messages(messages: Sequence[Any]) -> str:
    """The last AI message's text (a graph's answer)."""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            content = message.content
            if isinstance(content, str):
                return content
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
    return ""


def _langchain(arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls) -> Runner:
    from langchain.agents import create_agent

    middleware: list[Any] = []
    if arm.feature == "summarization":
        from langchain.agents.middleware import SummarizationMiddleware

        # LangChain's documented setup: summarize past a token count, keep the last messages
        middleware.append(
            SummarizationMiddleware(
                model=chat_model(setup),
                trigger=("tokens", int(0.75 * setup.window)),
                keep=("messages", 20),
            )
        )
    graph = create_agent(
        chat_model(setup),
        tools=as_langchain(tools, calls),
        system_prompt=system,
        middleware=middleware,
    )
    return _graph_runner(graph)


def _deepagents(arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls) -> Runner:
    from deepagents import create_deep_agent

    middleware: list[Any] = []
    if arm.feature == "planning":
        from langchain.agents.middleware import TodoListMiddleware

        middleware.append(TodoListMiddleware())
    graph = create_deep_agent(
        model=chat_model(setup),
        tools=as_langchain(tools, calls),
        system_prompt=system,
        middleware=middleware,
    )
    return _graph_runner(graph)


def _openai_agents(
    arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls
) -> Runner:
    from agents import Agent, Runner, SQLiteSession

    agent = Agent(
        name=NAME,
        instructions=system,
        model=agents_model(setup),
        tools=as_openai_agents(tools, calls),
    )
    count = iter(range(1_000_000))

    async def run(task: str) -> str:
        session = SQLiteSession(f"run-{next(count)}") if arm.feature == "session" else None
        result = await Runner.run(agent, task, session=session)
        return str(result.final_output or "")

    return run


def _claude(arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls) -> Runner:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    options = ClaudeAgentOptions(
        system_prompt=system,
        model=setup.model,
        mcp_servers={SERVER: as_claude(tools, calls)},
        allowed_tools=[f"mcp__{SERVER}__{t.name}" for t in tools],
        env=claude_env(setup),
    )

    async def run(task: str) -> str:
        answer = ""
        async for message in query(prompt=task, options=options):
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"claude: {message.subtype}: {message.result}")
                answer = message.result or ""
        return answer

    return run


#: JSON schema types as the Python types CrewAI's pydantic tool arguments take.
TYPES: Final[dict[str, Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def arguments_model(tool: Tool) -> Any:
    """The tool's JSON schema as a pydantic model (CrewAI's ``args_schema``): each property
    typed and described, the required ones required."""
    from pydantic import Field, create_model

    schema = schema_of(tool)
    required = set(schema.get("required") or ())
    fields: dict[str, Any] = {}
    for name, prop in schema["properties"].items():
        named = prop.get("type") if isinstance(prop, dict) else None
        kind = TYPES.get(named, Any) if isinstance(named, str) else Any
        described = Field(
            ... if name in required else None,
            description=(prop.get("description") if isinstance(prop, dict) else None) or name,
        )
        fields[name] = (kind if name in required else kind | None, described)
    return create_model(f"{tool.name.replace('-', '_')}_args", **fields)


def as_crewai(tools: Sequence[Tool], calls: Calls) -> list[Any]:
    from crewai.tools import BaseTool  # pyright: ignore[reportMissingImports]

    made = []
    for tool in tools:
        run = invoker(tool, calls)

        def _run(self: Any, _call: Any = run, **args: Any) -> str:
            given = {k: v for k, v in args.items() if v is not None}
            return asyncio.run(_call(given))

        kind = type(
            f"Tool_{tool.name.replace('-', '_')}",
            (BaseTool,),
            {
                "__module__": __name__,
                "__annotations__": {"name": str, "description": str, "args_schema": type},
                "name": tool.name,
                "description": tool.spec.description or tool.name,
                "args_schema": arguments_model(tool),
                "_run": _run,
            },
        )
        made.append(kind())
    return made


def _crewai(arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls) -> Runner:
    from crewai import LLM, Agent, Crew, Task  # pyright: ignore[reportMissingImports]

    llm = LLM(model=f"openai/{setup.model}", base_url=setup.base_url, api_key=setup.key)
    crew_tools = as_crewai(tools, calls)

    def kickoff(task: str) -> str:
        agent = Agent(role=NAME, goal=system, backstory=system, tools=crew_tools, llm=llm)
        job = Task(description=task, expected_output="The answer to the request.", agent=agent)
        return str(Crew(agents=[agent], tasks=[job]).kickoff())

    async def run(task: str) -> str:
        # CrewAI's tools are synchronous: the crew runs in a thread, each tool call its own loop
        return await asyncio.to_thread(kickoff, task)

    return run


#: Processes an arm started that work for it (TrueForge's server): their CPU seconds and
#: memory count as the arm's.
HELPERS: list[int] = []
#: Numbers each MCP server a TrueForge arm serves (one per agent built: its own calls).
_SERVED = itertools.count(1)


def _trueforge(arm: Arm, setup: Setup, system: str, tools: Sequence[Tool], calls: Calls) -> Runner:
    from benchmark.cost.frameworks import trueforge as tf

    server = tf.TrueForge.current()
    if server.pid not in HELPERS:
        HELPERS.append(server.pid)
    model = server.provider(setup.base_url, setup.key, setup.model, setup.window)
    by_name = {t.name: invoker(t, calls) for t in tools}

    async def call(name: str, args: dict[str, Any]) -> str:
        run = by_name.get(name)
        return await run(args) if run else f"Error: no tool {name}"

    served = tf.Served(tf.mcp_app(tools, call))
    name = tf.resource_name(f"tools-{len(tools)}-{next(_SERVED)}")
    server.mcp_server(name, served.url, "The tools this assistant works with.")
    agent = tf.spec(model, system, name, preload=arm.feature == "preload")

    async def run(task: str) -> str:
        return await tf.turn(server.url, agent, task, seconds=3600)

    return run


# --------------------------------------------------------------------------- arms B and C
async def wrapped(
    arm: Arm, setup: Setup, h: Any, system: str, sources: Sequence[Any], *, output: Any = None
) -> tuple[Any, dict[str, Any]]:
    """Arms B and C: the target and its ``h.wrap`` options (``tools``, ``without``); a C
    arm's ``ReAct`` answers an ``output`` when the task set has one."""
    if arm.kind == "C":
        from trellis import ReAct

        target = ReAct(
            system,
            setup.model,
            output=output,
            max_steps=setup.max_steps,
            context_window=setup.window,
        )
        return target, {"tools": list(sources), "without": arm.without}
    if arm.kind != "B":
        raise ValueError(f"{arm.name} is not wrapped")
    if arm.framework in ("langchain", "deepagents"):
        tools = await h.tools(*sources, framework=arm.framework.replace("langchain", "langgraph"))
        if arm.framework == "langchain":
            from langchain.agents import create_agent

            graph = create_agent(chat_model(setup), tools=tools, system_prompt=system)
        else:
            from deepagents import create_deep_agent

            graph = create_deep_agent(model=chat_model(setup), tools=tools, system_prompt=system)
        return graph, {}
    if arm.framework == "openai_agents":
        from agents import Agent

        return Agent(name=NAME, instructions=system, model=agents_model(setup)), {
            "tools": list(sources)
        }
    if arm.framework == "claude_agent_sdk":
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(system_prompt=system, model=setup.model, env=claude_env(setup))
        return options, {"tools": list(sources)}
    raise ValueError(f"{arm.name}: no wrapped arm for {arm.framework}")


async def bounded(run: Runner, task: str, seconds: float) -> str:
    """A native run within the same time a wrapped run gets (``h.wrap(timeout=)``)."""
    return await asyncio.wait_for(run(task), seconds)
