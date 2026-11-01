"""ACP agents: an agent the harness speaks to over the Agent Client Protocol
(https://agentclientprotocol.com), the open protocol editors drive coding agents with. The target
is an :class:`ACPAgent` — the command that starts the agent's ACP server, and how to run it —
and Kimi Code (:func:`kimi`) is the one ACP agent this release configures, pins and tests.

Why ACP for Kimi: Moonshot archived the Python Kimi CLI (``kimi-cli`` 1.52.0, its final
release, runs nothing) and the Python ``kimi-agent-sdk`` that wrapped it; its successor, Kimi
Code CLI (``@moonshot-ai/kimi-code``, TypeScript), is driven programmatically through ``kimi
acp`` — ACP over stdio, through the official Python SDK (``agent-client-protocol``).

* run: one ``kimi acp`` process per attempt; ``initialize``, a session (``session/new``, or the
  session an earlier attempt kept resumed: ``session/resume``, else ``session/load``), then
  ``session/prompt``. Everything the agent does on its own — its built-in tools, its sub-agents
  and swarms, plan mode, goals, compaction — it does as it always does: the harness passes it
  through and maps what ACP shows of it;
* tools: the harness's tools reach the agent as one more MCP server (``trellis``, so
  ``mcp__trellis__<tool>``), served for the attempt on the loopback interface
  (``tools.convert.mcp``) and handed over in ``session/new``, beside the agent's own servers;
* permissions: the agent asks before the calls its own policy asks about
  (``session/request_permission``): a harness tool is let through (the bridge is its check); any
  other is decided as a harness call is — the run's hooks, governance by its risk
  (:data:`KIMI_BUILTINS`), a person when it asks (a durable pause). A rejection, and an edit,
  end the turn there and the agent is told in a prompt of its own (ACP's answer carries no
  reason and no arguments): why, or to call the tool again with the edited arguments — that
  call is approved when it comes. The agent's own questions (Kimi's ``AskUserQuestion``) are the
  run's questions (``Runtime.ask``);
* records: every call the main agent runs itself is recorded as the run's once it ends
  (``tool_call`` / ``tool_call_update``: ``tools.bridge.observed``); a sub-agent's own calls are
  not shown over ACP — only those it asks permission for, which are governed;
* helpers: each sub-agent the agent starts (Kimi's ``Agent``) and each member of a swarm
  (``AgentSwarm``) is a child run of the run (``bridges.helpers``), from the call to its result;
  how many members a swarm runs at once is the harness's adaptive limit of child runs
  (``admission.py``), as ``KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`` (the agent's own ``env``
  wins);
* pause: the turn is cancelled (``session/cancel``) and the process ends; the session is in the
  journal and the next attempt resumes it, told to call the tool again, which the journal
  answers. The agent keeps its sessions on its machine (Kimi: ``KIMI_CODE_HOME``): a resume
  elsewhere says so and runs the prompt again in a new session, against the journal;
* spending: ACP's own ``usage`` on the prompt's answer when the agent gives it and the harness
  keeps no usage log of it; Kimi gives none, and its own usage log (``usage.record``, every
  model request of every agent, sub-agents included) is read as the turn runs — tokens only
  (dollars need the gateway's log: not yet); past the budget the turn is cancelled;
* models: with a route that attributes calls (``Capability.ATTRIBUTION``: Bifrost) serving the
  harness and a ``model`` named, Kimi's model calls go through
  Bifrost's OpenAI route with the virtual key, carrying the deny-all MCP scope, the run's
  dimensions and its ``traceparent`` as ``KIMI_CODE_CUSTOM_HEADERS`` lines;
* stalls: a model step that sends nothing for ``runtime.STALL_SECONDS`` (the time a tool call
  runs not counted) ends the run with a ``ModelTimeout``;
* framework options: :class:`ACPAgent` fields — ``env`` and ``mcp_servers`` added to the
  agent's, ``cwd``, ``mode`` and ``model`` replacing them.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, cast

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness import admission, budget, compat
from trellis.harness.adapters.base import (
    Extracted,
    HistoryIn,
    Invocation,
    Narrowing,
    Output,
    ToolFormat,
    helper,
    unique,
)
from trellis.harness.adapters.claude import RESUMED, RESUMED_CALL, RESUMED_DENIED
from trellis.harness.bounded import SAID_ONCE, LRUSet
from trellis.harness.bridges.helpers import Helpers, Kind
from trellis.harness.gateways import NO_KEY, Capability, OpenAICompatible
from trellis.harness.history import ASKED_WITH, Said, Speaker
from trellis.harness.journal import Pending
from trellis.harness.prompts import GatewayHeaders
from trellis.harness.runtime import Paused, Runtime, reason_of
from trellis.harness.tools import bridge
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS, SideEffects

log = logging.getLogger("trellis.harness")

#: The harness's MCP server, as the agent names its tools (``tools.convert.mcp.SERVER``).
SERVER: Final = "trellis"
HARNESS_PREFIX: Final = f"mcp__{SERVER}__"
#: The key a turn's model steps are watched for a stall under (``Runtime.model_started``).
STEP: Final = "acp"
#: The model name an ACP agent's own usage is counted under when the target names none.
ACP_MODEL: Final = "acp"
#: How often the agent's usage log is read while a turn runs (and the budget checked).
USAGE_SECONDS: Final = 0.5
#: How long, after a turn, the agent's usage log is read until it stops growing (at most), and
#: how often.
SETTLE_SECONDS: Final = 2.0
SETTLE_STEP: Final = 0.1
#: The most a cancelled turn waits for the agent to say it stopped.
CANCEL_SECONDS: Final = 5.0
#: The last of the agent's stderr kept, for the error of a process that ended mid-turn.
STDERR_TAIL: Final = 4096
#: The most MCP tools the agent was seen calling that join its tool set (the most recent).
SEEN_KEPT: Final = 1024
#: The session modes the harness governs every call in: a mode that approves on its own
#: (Kimi's ``auto``, ``yolo``) never asks, and governance would never see the calls.
GOVERNED_MODES: Final = frozenset({"default", "plan"})
#: The framework options a run may give: :class:`ACPAgent` fields (``command`` and ``profile``
#: are the agent's own).
OPTIONS: Final = ("env", "cwd", "mcp_servers", "mode", "model")
#: The agent's answer to a turn it ended for a reason the run fails with.
FAILED_STOPS: Final = frozenset({"refusal", "max_turn_requests"})
#: What the agent is told after a call the run rejected (ACP's answer carries no reason).
REJECTED: Final = "Your call to {tool} was not run: {why} Carry on with the task without it."
#: What the agent is told after a call whose arguments a reviewer (or a hook) changed (ACP's
#: answer carries no arguments): call it again with these, and that call runs.
EDITED: Final = (
    "A reviewer changed the arguments of your call to {tool}, so it was not run. Call {tool} "
    "again now with exactly these arguments: {args}"
)
#: Why the person denied the call, after the resumed session is told they did (ACP's answer
#: carries no reason: the prompt does).
DENIED_WHY: Final = "Their reason: {why}"
#: Why a sub-agent whose call failed without saying why ended.
SUBAGENT_FAILED: Final = "the sub-agent's call failed"
#: A swarm member in Kimi's ``AgentSwarm`` result.
SWARM_MEMBER: Final = re.compile(
    r'<subagent agent_id="(?P<agent>[^"]*)" item="(?P<item>[^"]*)" '
    r'outcome="(?P<outcome>[^"]*)">(?P<text>.*?)</subagent>',
    re.DOTALL,
)


# --------------------------------------------------------------------------- Kimi Code
#: Kimi Code's built-in tools, and what each does as governance sees it (the tool catalog's
#: word overrides it): reads run, writes run and are announced, ``Bash`` asks; a sub-agent's
#: call (``Agent``, ``AgentSwarm``) only hands work on — each of its calls Kimi asks about is
#: governed on its own. Any other tool Kimi asks about is a write.
KIMI_BUILTINS: Final[dict[str, SideEffects]] = {
    "Read": "read",
    "ReadMediaFile": "read",
    "Glob": "read",
    "Grep": "read",
    "FetchURL": "read",
    "WebSearch": "read",
    "TodoList": "read",
    "TaskList": "read",
    "TaskOutput": "read",
    "WaitFor": "read",
    "Skill": "read",
    "EnterPlanMode": "read",
    "ExitPlanMode": "read",
    "GetGoal": "read",
    "CronList": "read",
    "AskUserQuestion": "read",
    "Agent": "read",
    "AgentSwarm": "read",
    "Write": "write",
    "Edit": "write",
    "TaskStop": "write",
    "CreateGoal": "write",
    "UpdateGoal": "write",
    "SetGoalBudget": "write",
    "CronCreate": "write",
    "CronDelete": "write",
    "Bash": "irreversible",
}
#: Where Kimi Code keeps its sessions, its config and its usage log, unless ``KIMI_CODE_HOME``
#: says.
KIMI_HOME: Final = "~/.kimi-code"
#: The environment Kimi Code reads its model from, and the headers it sends with each request.
KIMI_HEADERS: Final = "KIMI_CODE_CUSTOM_HEADERS"
#: The line of Kimi's usage log a model request's usage is on (``agents/<id>/wire.jsonl``).
USAGE_RECORD: Final = "usage.record"


@dataclass(frozen=True, slots=True)
class Profile:
    """What the harness knows of an ACP agent it tested: its name (the ``compat.BINARIES``
    key), its built-in tools and their risk, the tools that start sub-agents (``helpers``) and
    swarms (``swarms``), its question tool (asked as the run's questions), how a run routes its
    model through the gateway (``routed``), and where it logs its usage (``usage``)."""

    name: str
    builtins: Mapping[str, SideEffects]
    helpers: frozenset[str] = frozenset()
    swarms: frozenset[str] = frozenset()
    questions: frozenset[str] = frozenset()
    routed: Callable[[dict[str, str], Runtime, str | None], None] | None = None
    usage: Callable[[Mapping[str, str], str], UsageLog] | None = None
    #: the variable that names where the agent keeps its state (sessions, config, skills), set
    #: to a home of the run's own when the run gives it one (a sandboxed run)
    home: str | None = None
    #: the variable that caps how many members of a swarm the agent runs at once: set to the
    #: harness's adaptive limit of child runs (``admission.py``) unless the agent's own ``env``
    #: names it
    swarm_limit: str | None = None


def _kimi_routed(env: dict[str, str], runtime: Runtime, model: str | None) -> None:
    """Kimi's model calls through Bifrost when it serves the harness (a route with
    ``Capability.ATTRIBUTION``) and the agent names its
    model (its own ``KIMI_MODEL_BASE_URL`` wins): Bifrost's OpenAI route, the virtual key, and —
    on Bifrost's route — its headers as ``KIMI_CODE_CUSTOM_HEADERS`` lines beside the agent's
    own: the deny-all MCP scope, the run's dimensions and its ``traceparent``. Elsewhere Kimi
    calls the model it is configured with (said once)."""
    gateway = runtime.agent.harness.model_gateway
    bifrost = cast("OpenAICompatible", gateway) if gateway.has(Capability.ATTRIBUTION) else None
    if bifrost is not None and model and "KIMI_MODEL_BASE_URL" not in env:
        env["KIMI_MODEL_PROVIDER_TYPE"] = "openai"
        env["KIMI_MODEL_BASE_URL"] = bifrost.url
        env["KIMI_MODEL_API_KEY"] = bifrost.key or NO_KEY
        env["KIMI_MODEL_NAME"] = bifrost.routed(model)
    elif model:
        env.setdefault("KIMI_MODEL_NAME", model)
    if bifrost is None or not env.get("KIMI_MODEL_BASE_URL", "").startswith(bifrost.origin):
        _unrouted(runtime.agent_id)
        return
    ours = GatewayHeaders().now()
    lines = (env.get(KIMI_HEADERS) or "").splitlines()
    kept = [line for line in lines if line.partition(":")[0].strip().lower() not in ours]
    env[KIMI_HEADERS] = "\n".join([*kept, *(f"{k}: {v}" for k, v in ours.items())])


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent, of the most recent
def _unrouted(agent_id: str) -> None:
    log.info(
        "Kimi's model calls do not go through Bifrost (no LLM_GATEWAY=bifrost, no model= named, or "
        "the agent's own KIMI_MODEL_BASE_URL): the gateway does not attribute them to the "
        "run; their tokens are counted from Kimi's usage log",
        extra={"agent_id": agent_id},
    )


class UsageLog:
    """Kimi's usage log of one session: each model request's usage, every agent's — the main
    agent's and each sub-agent's own file (``<home>/sessions/<workspace>/<session>/agents/
    <agent>/wire.jsonl``, a ``usage.record`` line each) — read from where the attempt began
    (:meth:`start`: what an earlier attempt read is in the journal's spending)."""

    def __init__(self, home: Path, session_id: str) -> None:
        self.home = home
        self.session_id = session_id
        #: how far each file was read, in bytes
        self.read_to: dict[Path, int] = {}

    def _files(self) -> list[Path]:
        return sorted(self.home.glob(f"sessions/*/{self.session_id}/agents/*/wire.jsonl"))

    def start(self) -> None:
        for path in self._files():
            self.read_to[path] = path.stat().st_size

    def records(self) -> list[dict[str, Any]]:
        """The usage records written since the last read (a line still being written is left
        for the next)."""
        found: list[dict[str, Any]] = []
        for path in self._files():
            at = self.read_to.get(path, 0)
            with path.open("rb") as file:
                file.seek(at)
                data = file.read()
            end = data.rfind(b"\n") + 1
            self.read_to[path] = at + end
            for line in data[:end].splitlines():
                entry = _json_line(line)
                if entry.get("type") == USAGE_RECORD and isinstance(entry.get("usage"), dict):
                    found.append(entry)
        return found


def _json_line(line: bytes) -> dict[str, Any]:
    try:
        entry = json.loads(line)
    except ValueError:
        return {}
    return entry if isinstance(entry, dict) else {}


def _kimi_usage(env: Mapping[str, str], session_id: str) -> UsageLog:
    home = env.get("KIMI_CODE_HOME") or KIMI_HOME
    return UsageLog(Path(home).expanduser(), session_id)


KIMI: Final = Profile(
    name="kimi-code",
    builtins=KIMI_BUILTINS,
    helpers=frozenset({"Agent"}),
    swarms=frozenset({"AgentSwarm"}),
    questions=frozenset({"AskUserQuestion"}),
    routed=_kimi_routed,
    usage=_kimi_usage,
    home="KIMI_CODE_HOME",
    swarm_limit="KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY",
)


# --------------------------------------------------------------------------- the target
@dataclass(frozen=True)
class ACPAgent:
    """An ACP agent, as a run starts it: ``command`` (its ACP server over stdio), ``env`` (over
    this process's environment), ``cwd`` (its working directory: this process's by default),
    ``mcp_servers`` of its own (ACP's ``McpServer`` objects, as dicts: ``{"name", "command",
    "args", "env"}`` for stdio, ``{"type": "http", "name", "url", "headers"}``), ``mode`` (a
    session mode it lists that asks before acting: :data:`GOVERNED_MODES`), ``model`` (a model
    name the gateway serves, where the agent's profile routes it), ``sandbox`` (``sandbox()``:
    the agent's shell commands and file writes go to the run's sandbox, :class:`Sandboxed`),
    ``skills`` (a ``SkillSet``, ``**src.kimi()``: the run's pinned skills in the agent's own
    skills folder, :class:`RunHome`) and ``profile`` (what the harness knows of it:
    :data:`KIMI`)."""

    command: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    mcp_servers: tuple[Mapping[str, Any], ...] = ()
    mode: str | None = None
    model: str | None = None
    sandbox: Any = None
    skills: Any = None
    profile: Profile | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ConfigurationError("an ACP agent needs the command that starts it")
        _check_mode(self.mode)
        _check_servers(self.mcp_servers)


def kimi(
    *,
    model: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    mcp_servers: Sequence[Mapping[str, Any]] = (),
    mode: str | None = None,
    sandbox: Any = None,
    skills: Any = None,
    command: str = "kimi",
) -> ACPAgent:
    """Kimi Code (``@moonshot-ai/kimi-code``, its ``kimi acp`` server) as a target: ``model``
    the gateway's model name (with Bifrost; without, Kimi's own configuration — its login, or
    ``KIMI_MODEL_*`` in ``env``), ``cwd`` where it works, ``env`` over this process's,
    ``mcp_servers`` of its own beside the harness's, ``mode`` its session mode (``"default"``
    or ``"plan"``), ``sandbox`` a ``sandbox()`` its ``Bash`` commands and file writes go to
    (its reads do not: :class:`Sandboxed`), ``skills`` a ``SkillSet`` (``**src.kimi()``), its
    skills for each run, ``command`` the ``kimi`` executable."""
    return ACPAgent(
        command=(command, "acp"),
        env=dict(env or {}),
        cwd=cwd,
        mcp_servers=tuple(mcp_servers),
        mode=mode,
        model=model,
        sandbox=sandbox,
        skills=skills,
        profile=KIMI,
    )


def _check_mode(mode: str | None) -> None:
    if mode is not None and mode not in GOVERNED_MODES:
        raise ConfigurationError(
            f"session mode {mode!r}: the harness governs every call the agent asks about, and "
            f"a mode that approves on its own asks about none (use one of "
            f"{', '.join(sorted(GOVERNED_MODES))}; give a tool's risk in the tool catalog)"
        )


def _check_servers(servers: Sequence[Mapping[str, Any]]) -> None:
    for server in servers:
        if server.get("name") == SERVER:
            raise ConfigurationError(
                f"an MCP server named {SERVER!r}: the harness's own server (its tools) has "
                "that name"
            )


@dataclass(frozen=True, slots=True)
class ACPInput:
    prompt: str
    context: str | None
    #: the tool call a person answered about, when the run resumes from that pause
    answered: ToolCall | None = None
    #: the person denied that call (``REJECT``): it does not run
    denied: bool = False
    #: why they denied it, when they said
    why: str | None = None


class ACPRunError(RuntimeError):
    """The agent ended the turn as failed, or its process ended in the middle of it."""


#: The end of a turn, on the attempt's queue: the prompt's answer (or its error).
@dataclass(frozen=True, slots=True)
class _Ended:
    answer: Any = None
    error: BaseException | None = None


# --------------------------------------------------------------------------- the adapter
class ACPAdapter:
    name: ClassVar[str] = "acp"
    tool_format: ToolFormat = "mcp"
    fixed_tools: bool = False
    #: the agent lists a server's tools when its session starts: the tools offered then
    narrows: Narrowing = "run"
    #: a prompt takes one message, not a conversation: memory's window is in the context's text
    replays_history = False
    #: with memory off, the conversation's earlier turns are in the context's text too
    history_in: HistoryIn = "context"
    #: its session (the journal's) is continued by the conversation's next turn
    sessions = True

    def __init__(self) -> None:
        #: the MCP tools the agent was seen calling, the most recent kept
        self.seen: LRUSet[str] = LRUSet(maxsize=SEEN_KEPT)

    def steers(self, target: Any) -> bool:
        """None: ACP has no point between an agent's steps where a client adds a message to a
        turn under way (``session/prompt`` takes a turn's messages as it starts), so a
        steered message waits its turn."""
        return False

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        return None

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        prompt = input if isinstance(input, str) else json.dumps(input, default=str)
        return ACPInput(prompt=prompt, context=context)

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        return [item async for item in self._turns(target, native_input, run)]

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from acp.schema import AgentMessageChunk, TextContentBlock

        items: list[Any] = []
        async for item in self._turns(target, native_input, run):
            items.append(item)
            content = item.content if isinstance(item, AgentMessageChunk) else None
            if isinstance(content, TextContentBlock) and content.text:
                yield content.text
        yield Output(items)

    def extract(self, target: Any, output: Any) -> Extracted:
        from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock, ToolCallStart

        transcript: list[Said] = []
        text: list[str] = []
        for item in output:
            if isinstance(item, AgentMessageChunk) and isinstance(item.content, TextContentBlock):
                text.append(item.content.text)
            elif isinstance(item, ToolCallStart) and text:
                transcript.append(Said(Speaker.ASSISTANT, "".join(text)))
                text = []
            elif isinstance(item, PromptResponse) and item.stop_reason in FAILED_STOPS:
                raise ACPRunError(f"the agent ended the turn: {item.stop_reason}")
        answer = "".join(text)
        if answer:
            transcript.append(Said(Speaker.ASSISTANT, answer))
        return Extracted(answer=answer or None, transcript=transcript)

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        denied = resolution.decision is InterruptDecision.REJECT
        return dataclasses.replace(
            native_input,
            answered=pending.interrupt.tool_call,
            denied=denied,
            why=reason_of(resolution) if denied else None,
        )

    def check_options(self, options: Mapping[str, Any]) -> None:
        unknown = [key for key in options if key not in OPTIONS]
        if unknown:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, unknown))}: an ACP agent's run takes "
                f"{', '.join(OPTIONS)}"
            )
        _check_mode(options.get("mode"))
        _check_servers(options.get("mcp_servers") or ())

    async def native(self, target: Any) -> list[ToolSpec]:
        """The agent's built-in tools (its profile's) and the MCP tools it was seen calling
        (its own servers', listed by the agent only)."""
        profile = target.profile
        names = [*(profile.builtins if profile is not None else ()), *self.seen]
        return unique(native_spec(name, profile) for name in names)

    # ------------------------------------------------------------------ internals
    async def _turns(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The attempt's items — the agent's session updates and each prompt's answer — with
        the attempt's helpers open while it runs."""
        helpers = Helpers(run.runtime, self.name)
        error: BaseException | None = None
        try:
            async with contextlib.aclosing(self._attempt(target, native_input, run)) as items:
                async for item in items:
                    yield item
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error)

    async def _attempt(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        from acp import PROTOCOL_VERSION, spawn_agent_process
        from acp.schema import ClientCapabilities

        runtime = run.runtime
        target = given(target, runtime.framework_options)
        profile = target.profile
        own = target.sandbox is not None or target.skills is not None
        home = RunHome(target, runtime) if own else None
        boxed = Sandboxed(target, runtime, home) if home is not None and target.sandbox else None
        env = home.env() if home is not None else {**os.environ, **target.env}
        if profile is not None and profile.routed is not None:
            profile.routed(env, runtime, target.model)
        if profile is not None and profile.swarm_limit is not None:
            env.setdefault(profile.swarm_limit, str(admission.ADMISSION.limit))
        cwd = boxed.work if boxed is not None else target.cwd or os.getcwd()
        attempt = Attempt(self, runtime, target, boxed)
        capabilities = boxed.capabilities() if boxed is not None else ClientCapabilities()
        async with contextlib.AsyncExitStack() as stack:
            if home is not None:
                stack.push_async_callback(home.closed)
                await home.skills()
            servers = [_server(s) for s in target.mcp_servers]
            if run.native_tools is not None:
                servers.append(await stack.enter_async_context(_harness_server(run.native_tools)))
                attempt.served = run.native_tools
            conn, process = await stack.enter_async_context(
                spawn_agent_process(
                    cast("Any", attempt),  # an ACP ``Client``: the methods the agent calls
                    *target.command,
                    env=env,
                    cwd=cwd,
                    transport_kwargs={"limit": _buffer_limit()},
                )
            )
            attempt.bind(conn, process)
            stack.push_async_callback(attempt.stopped)  # before the connection closes
            init = await attempt.answered(
                conn.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities=capabilities)
            )
            if profile is not None:
                found = init.agent_info.version if init.agent_info is not None else ""
                compat.require_binary(profile.name, found, command=target.command[0])
            session = await attempt.session(init, cwd, servers, native_input)
            if profile is not None and profile.usage is not None:
                usage = attempt.usage = profile.usage(env, session)
                usage.start()
            async with contextlib.aclosing(attempt.turns(native_input)) as items:
                async for item in items:
                    yield item

    def observed(self, name: str) -> None:
        """A tool the agent ran itself: an MCP server's joins the tool set."""
        if name.startswith("mcp__") and not name.startswith(HARNESS_PREFIX):
            self.seen.add(name)


def _buffer_limit() -> int:
    """The most one message from the agent may be: the SDK's default (50 MiB) — a tool's
    result travels in one."""
    from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES

    return DEFAULT_STDIO_BUFFER_LIMIT_BYTES


def given(target: ACPAgent, options: Mapping[str, Any]) -> ACPAgent:
    """The target with a run's framework options: ``env`` and ``mcp_servers`` added to its
    own, the others replacing them."""
    if not options:
        return target
    changes = dict(options)
    if "env" in changes:
        changes["env"] = {**target.env, **changes["env"]}
    if "mcp_servers" in changes:
        changes["mcp_servers"] = (*target.mcp_servers, *changes["mcp_servers"])
    return dataclasses.replace(target, **changes)


def _server(config: Mapping[str, Any]) -> Any:
    """One of the agent's own MCP servers, as ACP takes it."""
    from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer

    kind = config.get("type")
    if kind == "http":
        return HttpMcpServer.model_validate(config)
    if kind == "sse":
        return SseMcpServer.model_validate(config)
    return McpServerStdio.model_validate({"args": [], "env": [], **config})


@contextlib.asynccontextmanager
async def _harness_server(server: Any) -> AsyncIterator[Any]:
    """The harness's tools as an MCP server for the attempt, as ACP hands it to the agent."""
    from acp.schema import HttpHeader, HttpMcpServer

    from trellis.harness.tools.convert import mcp as convert

    async with convert.served(server) as served:
        headers = [HttpHeader(name=k, value=v) for k, v in served.headers.items()]
        yield HttpMcpServer(type="http", name=SERVER, url=served.url, headers=headers)


def native_spec(name: str, profile: Profile | None) -> ToolSpec:
    """A tool the agent runs itself, as the harness records and publishes it: a built-in by its
    risk (its profile's), an MCP server's tool (``mcp__<server>__<tool>``) by its server."""
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return ToolSpec(name=name, source="mcp", server=parts[1])
    builtins = profile.builtins if profile is not None else {}
    spec = ToolSpec(name=name, side_effects=builtins.get(name, DEFAULT_SIDE_EFFECTS))
    hands_on = profile is not None and name in profile.helpers | profile.swarms
    return helper(spec) if hands_on else spec


# --------------------------------------------------------------------------- one attempt
@dataclass
class _Call:
    """A tool call the main agent made, as its updates told it."""

    name: str
    args: dict[str, Any]
    started: float = field(default_factory=time.time)
    running: bool = False
    #: the run decided against it (or paused on it): it did not run, and is not recorded
    refused: bool = False


class Attempt:
    """One attempt's ACP client: the agent's session updates, queued for the turn, and its
    permission requests, decided as the run's."""

    def __init__(
        self,
        adapter: ACPAdapter,
        runtime: Runtime,
        target: ACPAgent,
        boxed: Sandboxed | None = None,
    ) -> None:
        self.adapter = adapter
        #: the run's sandbox, when the agent works in one: its file and terminal requests
        self.boxed = boxed
        self.runtime = runtime
        self.target = target
        self.profile = target.profile
        self.conn: Any = None
        self.process: Any = None
        self.session_id: str | None = None
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.calls: dict[str, _Call] = {}
        self.usage: UsageLog | None = None
        #: what the agent is told next, in a prompt of its own, once the turn has stopped
        self.followup: str | None = None
        #: the call a reviewer edited: approved when the agent makes it with those arguments
        self.edited: tuple[str, dict[str, Any]] | None = None
        #: the turn was cancelled (a pause, a refusal, the budget): the agent is stopping
        self.cancelled = False
        #: the session an earlier attempt kept was resumed (not a new one)
        self.resumed = False
        #: a loaded session's history is being replayed (``session/load``): not this turn's
        self.loading = False
        self.stderr = bytearray()
        self.drain: asyncio.Task[None] | None = None
        #: the agent's requests being answered now (each in a task of the connection's)
        self.handling: set[asyncio.Task[Any]] = set()
        #: the harness's tools as the attempt serves them (``tools.convert.mcp.Tools``)
        self.served: Any = None

    def bind(self, conn: Any, process: Any) -> None:
        self.conn = conn
        self.process = process
        self.drain = asyncio.create_task(self._drained(process.stderr))

    async def _drained(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(STDERR_TAIL):
            self.stderr.extend(chunk)
            del self.stderr[:-STDERR_TAIL]

    async def answered(self, request: Any) -> Any:
        """The agent's answer to ``request``; when its process ended instead (the connection
        closed with it), why it ended."""
        try:
            return await request
        except Exception as exc:
            raise await self._failed(exc) from exc

    def why_ended(self) -> str:
        tail = self.stderr.decode(errors="replace").strip()
        code = self.process.returncode
        return f"the agent's process ended (exit {code}) in the middle of the turn" + (
            f": {tail}" if tail else ""
        )

    # ------------------------------------------------------------------ the session
    async def session(self, init: Any, cwd: str, servers: list[Any], native_input: ACPInput) -> str:
        """The session the turn runs in: the one an earlier attempt kept, resumed (else loaded),
        when the agent still holds it; else a new one (in ``mode``, when one is given)."""
        from acp import RequestError

        journal = self.runtime.replay.journal
        kept = journal.session
        caps = init.agent_capabilities.model_dump() if init.agent_capabilities else {}
        if kept is not None:
            resumable = (caps.get("session_capabilities") or {}).get("resume") is not None
            try:
                if resumable:
                    await self.answered(
                        self.conn.resume_session(session_id=kept, cwd=cwd, mcp_servers=servers)
                    )
                else:
                    self.loading = True
                    await self.answered(
                        self.conn.load_session(session_id=kept, cwd=cwd, mcp_servers=servers)
                    )
                self.loading = False
                self.session_id, self.resumed = kept, True
                return kept
            except RequestError as exc:
                self.loading = False
                self.runtime.events.warning(
                    "acp_session", f"session {kept} was not resumed ({exc}): run again"
                )
                journal.session = None
        created = await self.answered(self.conn.new_session(cwd=cwd, mcp_servers=servers))
        self.session_id = journal.session = created.session_id
        if self.target.mode is not None:
            await self.answered(
                self.conn.set_session_mode(session_id=created.session_id, mode_id=self.target.mode)
            )
        return created.session_id

    # ------------------------------------------------------------------ the turns
    async def turns(self, native_input: ACPInput) -> AsyncGenerator[Any, None]:
        """The turn's updates and its answer, then — when the run stopped it to tell the agent
        something (a rejection's reason, an edit) — the next turn's, until one ends on its own
        or the run pauses."""
        prompt = self._first(native_input)
        while prompt is not None:
            self.cancelled = False
            async with contextlib.aclosing(self._turn(prompt)) as items:
                async for item in items:
                    yield item
            prompt, self.followup = self.followup, None
            if self.runtime.pending is not None or self.runtime.budget_spent is not None:
                return

    def _first(self, native_input: ACPInput) -> list[Any]:
        from acp import text_block

        if self.resumed and self.runtime.attempt > 1:  # the run's own session, after a pause
            return [text_block(self._resumed(native_input))]
        blocks = [text_block(native_input.prompt)]
        if native_input.context:  # marked off: the question is the block after it
            blocks.insert(0, text_block(ASKED_WITH.format(context=native_input.context)))
        return blocks

    def _resumed(self, native_input: ACPInput) -> str:
        """What the resumed session is told: the call a person answered about, named as the
        agent calls it (a harness tool through the ``trellis`` server, its own by its name)."""
        call = native_input.answered
        if call is None:
            return RESUMED
        harness = call.tool in self.runtime.toolbox
        told = RESUMED_DENIED if native_input.denied else RESUMED_CALL
        said = told.format(tool=f"{HARNESS_PREFIX}{call.tool}" if harness else call.tool)
        why = native_input.why
        return f"{said} {DENIED_WHY.format(why=why)}" if why else said

    async def _turn(self, prompt: Any) -> AsyncGenerator[Any, None]:
        if isinstance(prompt, str):
            from acp import text_block

            prompt = [text_block(prompt)]
        runtime = self.runtime
        asking = asyncio.ensure_future(self.conn.prompt(session_id=self.session_id, prompt=prompt))
        asking.add_done_callback(self._ended)
        runtime.model_started(STEP)
        try:
            while True:
                try:
                    async with asyncio.timeout(USAGE_SECONDS):
                        item = await self.queue.get()
                except TimeoutError:
                    self._died()
                    await self._spent()
                    continue
                self._died()
                if isinstance(item, _Ended):
                    await self._settled()
                    if item.error is not None:
                        raise await self._failed(item.error)
                    self._answered_usage(item.answer)
                    yield item.answer
                    return
                if await self._seen(item):
                    await self._spent()  # a call ended: what its step spent
                yield item
                if runtime.pending is not None:
                    await self.cancel()  # the run paused: the agent stops here
        finally:
            runtime.model_ended(STEP)
            if not asking.done():
                asking.cancel()

    def _died(self) -> None:
        """What ended a harness call that was not an error (the worker dying), raised in the
        attempt: in the server's request it ended only the server."""
        fatal = self.served.fatal if self.served is not None else None
        if fatal is not None:
            raise fatal

    async def _failed(self, error: BaseException) -> BaseException:
        """The agent's error — or, when the connection closed (its process ended), why the
        process ended."""
        if not isinstance(error, ConnectionError):
            return error
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(CANCEL_SECONDS):
                await self.process.wait()
                assert self.drain is not None  # bound with the process
                await asyncio.shield(self.drain)  # what it said last
        return ACPRunError(self.why_ended())

    def _ended(self, asking: asyncio.Future[Any]) -> None:
        if asking.cancelled():
            return
        error = asking.exception()
        if error is not None:
            self.queue.put_nowait(_Ended(error=error))
            return
        self.queue.put_nowait(_Ended(answer=asking.result()))

    async def cancel(self) -> None:
        """Stop the turn (once): the agent answers the prompt ``cancelled``."""
        if self.cancelled:
            return
        self.cancelled = True
        with contextlib.suppress(Exception):
            async with asyncio.timeout(CANCEL_SECONDS):
                await self.conn.cancel(session_id=self.session_id)

    async def _settled(self) -> None:
        """The turn ended: the usage the agent logs last (it writes its log as it goes, a
        moment behind) read until a read finds nothing new — accounted, the turn being over."""
        if self.usage is None:
            return
        ends = asyncio.get_running_loop().time() + SETTLE_SECONDS  # a log that keeps growing
        self._accounted()
        await asyncio.sleep(SETTLE_STEP)
        while self._accounted() and asyncio.get_running_loop().time() < ends:  # noqa: ASYNC110 - a file, polled
            await asyncio.sleep(SETTLE_STEP)

    async def _spent(self) -> None:
        """The usage the agent logged since the last read, on the run's budget; the turn
        cancelled once the budget is spent."""
        runtime = self.runtime
        self._accounted()
        spent = budget.reached(runtime)
        if spent is not None and runtime.budget_spent is None:
            budget.stopped(runtime, spent)
            await self.cancel()

    def _answered_usage(self, answer: Any) -> None:
        """ACP's own usage on the prompt's answer, on the run's budget — for an agent the
        harness keeps no usage log of (its profile's ``usage``: Kimi's log counts every request,
        and is its count)."""
        usage = getattr(answer, "usage", None)
        if usage is None or self.usage is not None:
            return
        budget.account(
            self.runtime,
            model=self.target.model or ACP_MODEL,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached=usage.cached_read_tokens or 0,
        )

    def _accounted(self) -> int:
        """The usage records the agent logged since the last read, on the run's budget: how
        many."""
        found = self.usage.records() if self.usage is not None else []
        for record in found:
            usage = record["usage"]
            cached = int(usage.get("inputCacheRead") or 0)
            written = int(usage.get("inputCacheCreation") or 0)
            budget.account(
                self.runtime,
                model=self.target.model or str(record.get("model") or "kimi"),
                input_tokens=int(usage.get("inputOther") or 0) + cached + written,
                output_tokens=int(usage.get("output") or 0),
                cached=cached,
            )
        return len(found)

    # ------------------------------------------------------------------ what the agent did
    async def _seen(self, update: Any) -> bool:
        """A session update: a tool call running or ending (recorded; a sub-agent's a helper of
        the run); the stall watch follows the calls. Whether a call ended."""
        from acp.schema import ToolCallProgress

        runtime = self.runtime
        if not isinstance(update, ToolCallProgress):
            return False
        call = self.calls.get(update.tool_call_id)
        if call is None:
            return False
        if update.status == "in_progress" and not call.running:
            call.running = True
            runtime.model_ended(STEP)  # the tool's time is not the model's
            await self._helpers_started(update.tool_call_id, call)
        elif update.status in ("completed", "failed"):
            self.calls.pop(update.tool_call_id, None)
            if not any(c.running for c in self.calls.values()):
                runtime.model_started(STEP)
            await self._ran(update, call)
            return True
        return False

    async def _ran(self, update: Any, call: _Call) -> None:
        failed = update.status == "failed"
        text = _output(update)
        await self._helpers_ended(update.tool_call_id, call, text, failed=failed)
        if call.refused or call.name.startswith(HARNESS_PREFIX):
            return  # not run; or a harness tool, which the bridge records
        self.adapter.observed(call.name)
        outcome = ToolOutcome(
            tool=call.name,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=text,
            error_class="ACPToolError" if failed else None,
        )
        spec = self.runtime.native.get(call.name) or native_spec(call.name, self.profile)
        await bridge.observed(
            self.runtime, spec, call.args, outcome, ref=update.tool_call_id, started=call.started
        )

    async def _helpers_started(self, call_id: str, call: _Call) -> None:
        """A sub-agent the agent starts (its profile's ``helpers``), or each member of a swarm
        (``swarms``: one per item, its task the template with the item in it), a helper."""
        helpers, profile = self.runtime.helpers, self.profile
        if profile is None:  # an agent the harness knows no sub-agent tool of
            return
        assert helpers is not None  # the turn runs with the attempt's helpers (``_turns``)
        args = call.args
        if call.name in profile.helpers:
            task = str(args.get("prompt") or args.get("description") or "")
            name = str(args.get("subagent_type") or "coder")
            await helpers.start(call_id, kind=Kind.SUBAGENT, name=name, task=task)
        elif call.name in profile.swarms:
            template = str(args.get("prompt_template") or "{{item}}")
            name = str(args.get("subagent_type") or "coder")
            for item in args.get("items") or ():
                task = template.replace("{{item}}", str(item))
                key = f"{call_id}/{item}"
                await helpers.start(key, kind=Kind.SUBAGENT, name=name, task=task)

    async def _helpers_ended(self, call_id: str, call: _Call, text: str, *, failed: bool) -> None:
        helpers, profile = self.runtime.helpers, self.profile
        if profile is None:
            return
        assert helpers is not None
        if call.name in profile.helpers:
            if failed:
                await helpers.end(call_id, error=text or SUBAGENT_FAILED)
            else:
                await helpers.end(call_id, output=text)
        elif call.name in profile.swarms:
            members = {m["item"]: m for m in SWARM_MEMBER.finditer(text)}
            for item in call.args.get("items") or ():
                member = members.get(str(item))
                key = f"{call_id}/{item}"
                if member is not None and member["outcome"] == "completed":
                    await helpers.end(key, output=member["text"].strip())
                else:
                    why = member["text"].strip() if member is not None else text
                    await helpers.end(key, error=why or SUBAGENT_FAILED)

    # ------------------------------------------------------------------ the ACP client
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """An update, queued for the turn; a call's name and arguments kept at once (a
        permission request for it may come before the turn reads its update)."""
        from acp.schema import ToolCallProgress, ToolCallStart

        if self.loading:  # a loaded session's history, replayed: not this turn's
            return
        if isinstance(update, ToolCallStart):
            args = _args(update.raw_input, update.content)
            self.calls[update.tool_call_id] = _Call(name=update.title, args=args)
        elif isinstance(update, ToolCallProgress) and update.raw_input is not None:
            known = self.calls.get(update.tool_call_id)
            if known is not None:
                known.args = _args(update.raw_input, None)
        self.queue.put_nowait(update)

    async def stopped(self) -> None:
        """The attempt is ending, before the connection closes: its sandbox's commands stopped,
        and the agent's requests still being answered (a decision, a command it waits on)
        given up — the connection closes no answer can go out on."""
        if self.boxed is not None:
            await self.boxed.stopped()
        waiting = [t for t in self.handling if not t.done()]
        for task in waiting:
            task.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)

    def _handles(self) -> None:
        """The request being answered now: given up if the attempt ends first."""
        task = asyncio.current_task()
        assert task is not None  # the connection answers each request in a task
        self.handling.add(task)
        task.add_done_callback(self.handling.discard)

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        """A call the agent asks permission for: a harness tool's let through (the bridge
        decides it as it runs); the agent's question asked as the run's; any other decided as
        the run decides a call — let through, or the turn stopped (a pause, a rejection, an
        edit) and the agent told why in the next prompt."""
        self._handles()
        name = tool_call.title or ""
        if name.startswith(HARNESS_PREFIX):
            return _chosen(options, "allow_once")
        if self.profile is not None and name in self.profile.questions:
            return await self._question(options, tool_call)
        return await self._decided(options, name, tool_call)

    async def _decided(self, options: list[Any], name: str, tool_call: Any) -> Any:
        """A call of the agent's own, decided as the run decides a call."""
        from acp.schema import RequestPermissionResponse

        runtime = self.runtime
        known = self.calls.get(tool_call.tool_call_id)
        args = known.args if known is not None else _args(tool_call.raw_input, tool_call.content)
        if self.edited == (name, args):
            self.edited = None
            return _chosen(options, "allow_once")
        side_effects = (self.profile.builtins if self.profile else {}).get(name)
        found = runtime.native.get(name)
        spec = found or ToolSpec(name=name, side_effects=side_effects or DEFAULT_SIDE_EFFECTS)
        runtime.model_ended(STEP)
        try:
            decided = await bridge.permitted(spec, args)
        except Paused:  # the run pauses: the agent stops here, and the resume goes on
            self._refused(known)
            await self._stop()
            return RequestPermissionResponse(outcome=_cancelled())
        if isinstance(decided, ToolOutcome):
            self._refused(known)
            self.followup = REJECTED.format(tool=name, why=str(decided.output))
            await self._stop()
            return _chosen(options, "reject_once")
        if decided.args != args:
            self._refused(known)
            self.edited = (name, decided.args)
            self.followup = EDITED.format(tool=name, args=json.dumps(decided.args))
            await self._stop()
            return _chosen(options, "reject_once")
        return _chosen(options, "allow_once")

    async def _question(self, options: list[Any], tool_call: Any) -> Any:
        """The agent's question to its user (Kimi's ``AskUserQuestion``), asked as the run's: a
        choice of its options; the answer is the option of that label."""
        from acp.schema import RequestPermissionResponse

        labels = [o.name for o in options if not o.kind.startswith("reject")]
        question = _text(tool_call.content) or "The agent asks you to choose."
        try:
            answer = await self.runtime.ask(question, options=labels)
        except Paused:
            await self._stop()
            return RequestPermissionResponse(outcome=_cancelled())
        chosen = {o.name: o.option_id for o in options}[answer]  # one of the labels asked
        return RequestPermissionResponse(outcome=_selected(chosen))

    # the agent's files and terminals, in the run's sandbox (``Sandboxed``)
    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._boxed().read(path, line, limit)

    async def write_text_file(self, session_id: str, path: str, content: str, **kwargs: Any) -> Any:
        return await self._boxed().write(path, content)

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        *,
        args: list[str] | None = None,
        env: list[Any] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        given = {e.name: e.value for e in env or ()}
        return await self._boxed().create(command, args or [], given, output_byte_limit)

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return self._boxed().output(terminal_id)

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return await self._boxed().wait(terminal_id)

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return self._boxed().kill(terminal_id)

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return self._boxed().release(terminal_id)

    def _boxed(self) -> Sandboxed:
        from acp import RequestError

        self._handles()

        if self.boxed is None:  # the capability was not given: the agent asks out of turn
            raise RequestError.method_not_found("fs/terminal (this run has no sandbox)")
        return self.boxed

    def _refused(self, call: _Call | None) -> None:
        if call is not None:
            call.refused = True

    async def _stop(self) -> None:
        """Stop the turn once the agent has its answer (``session/cancel`` sent next)."""
        asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(self.cancel()))


# --------------------------------------------------------------------------- a sandboxed run
#: Where a sandboxed run's agent keeps its state and works, on this machine (per run).
SANDBOXED_ROOT: Final = "trellis-acp"
#: The variables a sandboxed agent's process gets from this one's environment (no other: no
#: credential of the host's reaches it; the model's and the gateway's come from the profile).
SANDBOXED_ENV: Final = ("PATH", "LANG", "LC_ALL", "TZ")
#: What every sandboxed run of an agent is told, once per agent in the log and on each run's
#: stream: its reads are not in the sandbox.
READS_NOT_ISOLATED: Final = (
    "the agent's shell commands and file writes run in the run's sandbox, but its own file "
    "reads and searches (Kimi's Read, Glob, Grep) read this machine's files: reads are not "
    "isolated (run the harness itself in a container to isolate them)"
)


class RunHome:
    """A home of the run's own for the agent (a sandboxed run's, or one given skills): the
    folder the agent keeps its state in (its profile's ``home``: Kimi's ``KIMI_CODE_HOME``,
    its sessions and its skills) and a working directory, both kept across the run's attempts
    on this machine and removed with the run.

    * **skills** (``ACPAgent(skills=...)``, ``**src.kimi()``): the run's pinned skills are
      written into the agent's own skills folder (``<home>/skills/<name>/SKILL.md`` and its
      files: ``skills.write_plugin``) before each attempt — the versions the run pinned, so a
      resume reads the same ones;
    * **a run that is not sandboxed** keeps this process's environment, and the agent's own
      configuration and login (``config.toml``, ``credentials``) are linked into the run's home
      from the agent's usual one;
    * **a sandboxed run** (:class:`Sandboxed`) gets an environment of :data:`SANDBOXED_ENV`
      and the agent's own ``env`` only, its ``HOME`` the run's too, nothing linked: no
      credential of this machine's reaches it (its model comes through the gateway, or from
      ``env``)."""

    #: what of the agent's usual home a run's home links to (its configuration, its login)
    LINKED: Final = ("config.toml", "credentials")

    def __init__(self, target: ACPAgent, runtime: Runtime) -> None:
        from trellis.contracts import stable_id

        self.target = target
        self.runtime = runtime
        self.root = Path(tempfile.gettempdir()) / SANDBOXED_ROOT / stable_id(runtime.run_id)
        self.home = self.root / "home"
        self.agent_home = self.home / ".agent"
        self.work = str(self.root / "work")
        self.agent_home.mkdir(parents=True, exist_ok=True)
        Path(self.work).mkdir(parents=True, exist_ok=True)

    def env(self) -> dict[str, str]:
        profile = self.target.profile
        variable = profile.home if profile is not None else None
        if self.target.sandbox is not None:
            env = {k: os.environ[k] for k in SANDBOXED_ENV if k in os.environ}
            env = {**env, **self.target.env, "HOME": str(self.home)}
        else:
            env = {**os.environ, **self.target.env}
            if variable is not None:
                self._linked(Path(env.get(variable) or KIMI_HOME).expanduser())
        if variable is not None:  # the run's, whatever the environment says
            env[variable] = str(self.agent_home)
        return env

    def _linked(self, usual: Path) -> None:
        for name in self.LINKED:
            found, link = usual / name, self.agent_home / name
            if found.exists() and not link.exists():
                link.symlink_to(found)

    async def skills(self) -> None:
        """The run's pinned skills, in the agent's skills folder (``<home>/skills``)."""
        if self.target.skills is None:
            return
        from trellis.harness import skills

        shutil.rmtree(self.agent_home / "skills", ignore_errors=True)  # an earlier attempt's
        await skills.write_plugin(self.target.skills, self.agent_home)

    async def closed(self) -> None:
        """The attempt ended: the run's home and working directory removed with the run (kept
        while it waits on a person: the next attempt resumes the agent's session from them)."""
        if self.runtime.pending is None:
            shutil.rmtree(self.root, ignore_errors=True)


class Sandboxed:
    """A run's agent working with the run's sandbox (``ACPAgent(sandbox=sandbox())``): it is
    given ACP's file and terminal capabilities, served here — ``fs/write_text_file`` and
    ``fs/read_text_file`` the sandbox's files, ``terminal/*`` a command run in it (Kimi's
    ``Bash``) — so its commands and writes happen in the sandbox (``SandboxSource.opened``: the
    run's own, kept across attempts, deleted at the run's end).

    Its reads are not isolated: Kimi Code 2.1.1 reads and searches the host's files itself
    (``Read`` looks a file up on the host first; ``Glob`` and ``Grep`` never ask the client).
    What the harness can do it does: the agent's process gets a home and a working directory of
    the run's own, both empty, and an environment of :data:`SANDBOXED_ENV` and the agent's own
    ``env`` only (:class:`RunHome`) — and each run says that its reads are not isolated
    (:data:`READS_NOT_ISOLATED`)."""

    def __init__(self, target: ACPAgent, runtime: Runtime, home: RunHome) -> None:
        self.target = target
        self.runtime = runtime
        self.source = target.sandbox
        self.work = home.work
        self.terminals: dict[str, tuple[asyncio.Task[Any], int | None]] = {}
        self._made = False
        runtime.added.append(self.source)  # its end (and its pause) find it
        _said_reads(runtime.agent_id)
        runtime.events.warning("acp_sandbox", READS_NOT_ISOLATED)

    def capabilities(self) -> Any:
        from acp.schema import ClientCapabilities, FileSystemCapabilities

        files = FileSystemCapabilities(read_text_file=True, write_text_file=True)
        return ClientCapabilities(fs=files, terminal=True)

    async def box(self, *, changes: bool = False) -> Any:
        """The run's sandbox, its copy of the agent's working directory made the first time
        (the agent names its files by that path)."""
        box = await self.source.opened(self.runtime, changes=changes)
        if not self._made:
            await box.exec(f"mkdir -p {shlex.quote(self.work)}")
            self._made = True
        return box

    async def read(self, path: str, line: int | None, limit: int | None) -> Any:
        from acp import RequestError
        from acp.schema import ReadTextFileResponse

        try:
            data = await (await self.box()).read(path)
        except ToolError as exc:  # no such file: what the agent reads of a missing one
            raise RequestError.resource_not_found(path) from exc
        text = data.decode("utf-8", errors="replace")
        if line is not None or limit is not None:
            first = max((line or 1) - 1, 0)
            lines = text.splitlines(keepends=True)
            text = "".join(lines[first : first + limit if limit is not None else None])
        return ReadTextFileResponse(content=text)

    async def write(self, path: str, content: str) -> None:
        await (await self.box(changes=True)).write(path, content.encode())

    async def create(
        self, command: str, args: list[str], env: dict[str, str], limit: int | None
    ) -> Any:
        from acp.schema import CreateTerminalResponse

        box = await self.box(changes=True)
        line = shlex.join([command, *args]) if args else command
        task = asyncio.ensure_future(box.exec(line, timeout=self.runtime.remaining(), env=env))
        ident = f"terminal-{len(self.terminals) + 1}"
        self.terminals[ident] = (task, limit)
        return CreateTerminalResponse(terminal_id=ident)

    def output(self, ident: str) -> Any:
        from acp.schema import TerminalExitStatus, TerminalOutputResponse

        task, limit = self.terminals[ident]
        if not task.done():
            return TerminalOutputResponse(output="", truncated=False)
        text, code, signal = _ran(task)
        cut = False
        if limit is not None and len(text.encode()) > limit:
            cut, text = True, text.encode()[-limit:].decode("utf-8", errors="ignore")
        status = TerminalExitStatus(exit_code=code, signal=signal)
        return TerminalOutputResponse(output=text, truncated=cut, exit_status=status)

    async def wait(self, ident: str) -> Any:
        from acp.schema import WaitForTerminalExitResponse

        task, _ = self.terminals[ident]
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(task)
        _, code, signal = _ran(task)
        return WaitForTerminalExitResponse(exit_code=code, signal=signal)

    def kill(self, ident: str) -> None:
        self.terminals[ident][0].cancel()

    def release(self, ident: str) -> None:
        task, _ = self.terminals.pop(ident)
        task.cancel()

    async def stopped(self) -> None:
        """The attempt is ending: its commands still running are stopped."""
        running = [task for task, _ in self.terminals.values()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)


def _ran(task: asyncio.Task[Any]) -> tuple[str, int | None, str | None]:
    """A finished command's output (stdout, then stderr), its exit code and the signal that
    ended it (a command out of time, or killed)."""
    if task.cancelled():
        return "", None, "SIGKILL"
    error = task.exception()
    if error is not None:
        return f"{type(error).__name__}: {error}", None, "SIGKILL"
    result = task.result()
    return result.stdout + result.stderr, result.exit_code, None


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent, of the most recent
def _said_reads(agent_id: str) -> None:
    log.warning("%s: %s", agent_id, READS_NOT_ISOLATED, extra={"agent_id": agent_id})


def _chosen(options: list[Any], kind: str) -> Any:
    """The answer choosing the first option of ``kind`` (``allow_once``, ``reject_once``);
    the agent's question cancelled when it offers none."""
    from acp.schema import RequestPermissionResponse

    for option in options:
        if option.kind == kind:
            return RequestPermissionResponse(outcome=_selected(option.option_id))
    return RequestPermissionResponse(outcome=_cancelled())


def _selected(option_id: str) -> Any:
    from acp.schema import AllowedOutcome

    return AllowedOutcome(outcome="selected", option_id=option_id)


def _cancelled() -> Any:
    from acp.schema import DeniedOutcome

    return DeniedOutcome(outcome="cancelled")


def _args(raw: Any, content: Any) -> dict[str, Any]:
    """A call's arguments: its ``rawInput``, else the JSON its content's text holds (Kimi shows
    a starting call's arguments so), else the text itself."""
    if isinstance(raw, dict):
        return raw
    text = _text(content)
    if not text:
        return {}
    try:
        found = json.loads(text)
    except ValueError:
        return {"request": text}
    return found if isinstance(found, dict) else {"request": text}


def _text(content: Any) -> str:
    """The text of a tool call's content (its ``content`` blocks' text)."""
    parts: list[str] = []
    for block in content or ():
        inner = getattr(block, "content", None)
        text = getattr(inner, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _output(update: Any) -> str:
    raw = update.raw_output
    if isinstance(raw, str):
        return raw
    if raw is not None:
        return json.dumps(raw, default=str)
    return _text(update.content)


__all__ = [
    "KIMI",
    "KIMI_BUILTINS",
    "READS_NOT_ISOLATED",
    "ACPAdapter",
    "ACPAgent",
    "ACPInput",
    "ACPRunError",
    "Profile",
    "RunHome",
    "Sandboxed",
    "UsageLog",
    "kimi",
]
