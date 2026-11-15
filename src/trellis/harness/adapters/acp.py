"""ACP agents: an agent the harness speaks to over the Agent Client Protocol
(https://agentclientprotocol.com), the open protocol editors drive coding agents with. The target
is an :class:`ACPAgent` — the command that starts the agent's ACP server, and how to run it —
and Kimi Code (:func:`kimi`) is the one ACP agent this release configures, pins and tests.

Why ACP for Kimi: Moonshot archived the Python Kimi CLI (``kimi-cli`` 1.52.0, its final
release, runs nothing) and the Python ``kimi-agent-sdk`` that wrapped it; its successor, Kimi
Code CLI (``@moonshot-ai/kimi-code``, TypeScript), is driven programmatically through ``kimi
acp`` — ACP over stdio, through the official Python SDK (``agent-client-protocol``).

* run: one ``kimi acp`` process per attempt (``bridges.acp_live.ACPSession``); ``initialize``, a
  session (``session/new``, or the session an earlier attempt kept resumed: ``session/resume``,
  else ``session/load``; forked, ``session/fork``, when another worker's process holds it),
  then ``session/prompt``. A conversation's turn that ends with work in the agent's background
  (Kimi's ``run_in_background``) leaves the process to the conversation, and the next turn is
  served by it (``bridges.acp_live``). Everything the agent does on its own — its built-in
  tools, its sub-agents and swarms, plan mode, goals, compaction — it does as it always does:
  the harness passes it through and maps what ACP shows of it, and reads its log for what ACP
  does not;
* tools: the harness's tools reach the agent as one more MCP server (``trellis``, so
  ``mcp__trellis__<tool>``), served for the process on the loopback interface
  (``tools.convert.mcp``), each call made in the run being served, and handed over in
  ``session/new``, beside the agent's own servers;
* permissions: the agent's own, then a person when it asks. It asks before the calls its own
  policy asks about (``session/request_permission``; one its rules allow, in its own
  configuration, it runs): a harness tool is let through (the bridge is its check); any other
  is decided as a harness call is — the run's hooks, governance by its risk
  (:data:`KIMI_BUILTINS`) and the tool catalog's rules — and then, as the agent asked, a
  person is asked (``bridge.asked``: a durable pause; once for a call governance asks about
  too), unless the call only reads (a read tool, or a shell command
  ``governance.shell.read_only`` reads as read-only). A rejection,
  and an edit, end the turn there and the agent is told in a prompt of its own (ACP's answer
  carries no reason and no arguments): why, or to call the tool again with the edited
  arguments — that call is approved when it comes. A request that offers a choice of the
  agent's own (Kimi's ``AskUserQuestion``, the plan its ``ExitPlanMode`` presents for review)
  is the agent's question to its user: the run's question (``Runtime.ask``), with every
  option it offers;
* records: every call the main agent runs itself is recorded as the run's once it ends
  (``tool_call`` / ``tool_call_update``: ``tools.bridge.observed``); a sub-agent's own calls are
  not shown over ACP — only those it asks permission for, which are governed;
* helpers: each sub-agent the agent starts or goes on with (Kimi's ``Agent``, its ``resume``)
  and each member of a swarm (``AgentSwarm``, its ``resume_agent_ids``) is a child run of the
  run (``bridges.helpers``), from the call to its result — one run in the background, to its
  task's end (``acp_live.Background``);
  how many members a swarm runs at once is the harness's adaptive limit of child runs
  (``admission.py``), as ``KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`` (the agent's own ``env``
  wins);
* pause: the turn is cancelled (``session/cancel``) and the process ends (kept while work in
  its background goes on); the session is in the journal and the next attempt resumes it,
  told to call the tool again, which the journal answers. The agent keeps its sessions on its
  machine (Kimi: ``KIMI_CODE_HOME``): a resume elsewhere says so and runs the prompt again in
  a new session, against the journal;
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
import contextvars
import dataclasses
import functools
import html
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import time
import tomllib
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
    stable_id,
)
from trellis.harness import admission, budget, compat
from trellis.harness.adapters.base import (
    Extracted,
    HistoryIn,
    Invocation,
    Narrowing,
    Output,
    ToolFormat,
    builtin,
    helper,
    unique,
)
from trellis.harness.adapters.claude import RESUMED, resumed_call
from trellis.harness.bounded import LRU, SAID_ONCE, LRUSet
from trellis.harness.bridges import live as shared
from trellis.harness.bridges.acp_live import (
    CANCEL_SECONDS,
    CODE,
    GONE,
    GONE_SAID,
    HARNESS_PREFIX,
    QUEUED,
    SERVER,
    USAGE_SECONDS,
    WAITS,
    ACPRunError,
    ACPSession,
    Background,
    accounted,
    chosen,
    dismissed,
    selected,
    shape,
    unfilled,
)
from trellis.harness.bridges.helpers import Helpers, Kind
from trellis.harness.bridges.live import LiveSessions, Turn
from trellis.harness.gateways import NO_KEY, Capability, OpenAICompatible
from trellis.harness.governance.shell import Shell
from trellis.harness.history import ASKED_WITH, Called, Said, Speaker
from trellis.harness.journal import Pending, call_key
from trellis.harness.prompts import GatewayHeaders
from trellis.harness.runtime import Paused, Runtime, reason_of
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import UNDECLARED, SideEffects
from trellis.harness.tools.elicit import elicited

log = logging.getLogger("trellis.harness")

#: The key a turn's model steps are watched for a stall under (``Runtime.model_started``).
STEP: Final = "acp"
#: The model name an ACP agent's own usage is counted under when the target names none.
ACP_MODEL: Final = "acp"
#: How long, after a turn, the agent's usage log is read until it stops growing (at most), and
#: how often.
SETTLE_SECONDS: Final = 2.0
SETTLE_STEP: Final = 0.1
#: The most MCP tools the agent was seen calling that join its tool set (the most recent).
SEEN_KEPT: Final = 1024
#: The most terminals of a sandboxed run whose output is kept past their release.
TERMINALS_KEPT: Final = 16
#: The session modes the harness governs every call in: a mode that approves on its own
#: (Kimi's ``auto``, ``yolo``) never asks, and governance would never see the calls.
GOVERNED_MODES: Final = frozenset({"default", "plan"})
#: Kimi's log lines of a session's modes (``agents/main/wire.jsonl``): its permission mode
#: (:data:`KIMI_ASKS`: the one that asks before a call), and its plan mode going on and off.
PERMISSION_SET: Final = "permission.set_mode"
KIMI_ASKS: Final = "manual"
#: ... and the two session modes of Kimi's ACP server that ask (:data:`GOVERNED_MODES`): the
#: one that plans, and its default.
KIMI_PLANS: Final = "plan"
KIMI_DEFAULT: Final = "default"
PLAN_MODE: Final[Mapping[str, bool]] = {
    "plan_mode.enter": True,
    "plan_mode.cancel": False,
    "plan_mode.exit": False,
}
#: The framework options a run may give: :class:`ACPAgent` fields (``command`` and ``profile``
#: are the agent's own).
OPTIONS: Final = ("env", "cwd", "mcp_servers", "mode", "model")
#: The agent's answer to a turn it ended for a reason the run fails with.
FAILED_STOPS: Final = frozenset({"refusal", "max_turn_requests"})
#: The stop reason of a turn whose last reply reached the model's output limit (Kimi retries
#: such a reply itself first), and of one that ended as the agent ended it.
MAX_TOKENS: Final = "max_tokens"
END_TURN: Final = "end_turn"
#: What the agent is told after a call the run rejected (ACP's answer carries no reason).
REJECTED: Final = "Your call to {tool} was not run: {why} Carry on with the task without it."
#: What the agent is told after a call whose arguments a reviewer (or a hook) changed (ACP's
#: answer carries no arguments): call it again with these, and that call runs.
EDITED: Final = (
    "A reviewer changed the arguments of your call to {tool}, so it was not run. Call {tool} "
    "again now with exactly these arguments: {args}"
)
#: The run event the agent's plan is (ACP's ``plan`` update: Kimi's todo list), its entries
#: as the agent tells them (``content``, ``status``, ``priority``).
PLAN: Final = "plan"
#: Why a run failed whose agent could not finish its turn: the agent's own error for it.
FAILED_TURN: Final = "the agent could not finish its turn: {why}"
#: What a person is asked when the agent offers options and says nothing with them.
CHOOSE: Final = "The agent asks you to choose."
#: Why the person denied the call, after the resumed session is told they did (ACP's answer
#: carries no reason: the prompt does).
DENIED_WHY: Final = "Their reason: {why}"
#: Why a sub-agent whose call failed without saying why ended.
SUBAGENT_FAILED: Final = "the sub-agent's call failed"
#: A swarm member in Kimi's ``AgentSwarm`` result, and each of its attributes (``agent_id``,
#: ``item`` — escaped as XML — ``outcome``; a resumed member's ``mode="resume"``; ``state``,
#: ``stop_reason``).
SWARM_MEMBER: Final = re.compile(r"<subagent(?P<attrs>[^>]*)>(?P<text>.*?)</subagent>", re.DOTALL)
SWARM_ATTR: Final = re.compile(r'(\w+)="([^"]*)"')
#: The sub-agent Kimi starts when an ``Agent`` call names none (and resumes none).
KIMI_SUBAGENT: Final = "coder"


# --------------------------------------------------------------------------- Kimi Code
#: Kimi Code's built-in tools, and what each does as governance sees it (the tool catalog's
#: word overrides it): reads run; writes, ``Bash`` among them, run and are announced, Kimi's
#: own policy asking first where it would (a team that wants every shell command approved
#: gives ``Bash`` the risk ``irreversible``, or an ``approve_when``, in the tool catalog); a
#: sub-agent's
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
    "AskUserQuestion": "read",
    "Agent": "read",
    "AgentSwarm": "read",
    "Write": "write",
    "Edit": "write",
    "TaskStop": "write",
    "CreateGoal": "write",
    "UpdateGoal": "write",
    "SetGoalBudget": "write",
    "Bash": "write",
}
#: What Kimi Code is called when a person is asked about a call it asks about.
KIMI_CODE: Final = "Kimi Code"
#: Kimi Code's built-in tools whose work needs Kimi to go on running after the turn: a scheduled
#: prompt (``CronCreate``) fires only in a session that is running then, and its list and delete
#: go with it. The harness runs ``kimi acp`` for one attempt and ends it, so a schedule would
#: never fire: Kimi's own switch turns scheduling off (:data:`KIMI_ENV`: a call answers "Cron
#: scheduling is disabled", and the model can say so), as Claude's are disallowed
#: (``claude.OUTLIVES_THE_QUERY``), and they are not of the run's tool set (no hint offers them).
#: Kimi still lists them: ACP has no way to take a built-in off (``kimi acp`` takes no agent file).
KIMI_OUTLIVES_THE_TURN: Final = ("CronCreate", "CronDelete", "CronList")
#: Why scheduling is off (:data:`KIMI_OUTLIVES_THE_TURN`).
CRON_OFF: Final = (
    "Kimi's scheduling is off: a scheduled prompt fires only while Kimi runs, and the harness "
    "ends Kimi once the run's work is done, so nothing would fire it"
)
#: Why Kimi's tower mode (``KIMI_CODE_EXPERIMENTAL_TOWER``: ``TowerInit``, ``TowerSpawn``,
#: ``TowerMerge``...) is off: ``TowerSpawn`` starts each worker in Kimi's ``auto`` permission
#: mode, which asks about none of its calls (:data:`GOVERNED_MODES`), so the harness would
#: govern none of them — a run's sub-agents work through ``Agent``, whose calls Kimi asks
#: about.
TOWER_OFF: Final = (
    "Kimi's tower mode is off: its workers run in Kimi's auto permission mode, which asks about "
    "none of their calls, so the harness could govern none of them (use Agent)"
)
#: What the harness sets in Kimi's environment, whatever the agent's own says, and why (said
#: on the run when the agent's own ``env`` asks otherwise): scheduling off (before, Kimi
#: scheduled a prompt nothing would fire, and answered as if it would: the break-and-fix gate),
#: and tower mode off (Kimi reads its environment's word over its config's).
#: Its own update and its usage reports are off too: a server's Kimi stays the version that
#: was tested, and sends its maker nothing of the tenants' use.
UPDATES_OFF: Final = (
    "Kimi does not update itself under the harness: an update would move it past the version "
    "the harness was tested with while it serves runs (install the version you want)"
)
TELEMETRY_OFF: Final = (
    "Kimi's usage reports to its maker are off under the harness: a server's runs are its "
    "tenants', and what is recorded of them is the harness's to keep"
)
KIMI_ENV: Final = {
    "KIMI_DISABLE_CRON": "1",
    "KIMI_CODE_EXPERIMENTAL_TOWER": "0",
    "KIMI_CODE_NO_AUTO_UPDATE": "1",
    "KIMI_DISABLE_TELEMETRY": "1",
}
KIMI_ENV_WHY: Final = {
    "KIMI_DISABLE_CRON": CRON_OFF,
    "KIMI_CODE_EXPERIMENTAL_TOWER": TOWER_OFF,
    "KIMI_CODE_NO_AUTO_UPDATE": UPDATES_OFF,
    "KIMI_DISABLE_TELEMETRY": TELEMETRY_OFF,
}
#: Where Kimi Code keeps its sessions, its config and its usage log, unless ``KIMI_CODE_HOME``
#: says.
KIMI_HOME: Final = "~/.kimi-code"
#: The environment Kimi Code reads its model from, and the headers it sends with each request.
KIMI_HEADERS: Final = "KIMI_CODE_CUSTOM_HEADERS"
#: The line of Kimi's usage log a model request's usage is on (``agents/<id>/wire.jsonl``).
USAGE_RECORD: Final = "usage.record"
#: The line every turn's end is on (Kimi 2.1's durable ``TurnEnded``: ``agentId``, ``turnId``,
#: ``reason`` completed, cancelled, failed or blocked), and the error code of a turn it ended
#: at its step limit (``loop_control.max_steps_per_turn``: reason failed) — over ACP a
#: finished turn with no answer. Its ``agent.turn.ended`` lines are another record.
TURN_ENDED: Final = "turn.ended"
AT_STEP_LIMIT: Final = "loop.max_steps_exceeded"
#: ... and its ``reason`` when Kimi could not finish the turn (its ``error`` says why).
TURN_FAILED: Final = "failed"
#: The line of a sub-agent's log that names the profile it was started with (its type).
PROFILE_BOUND: Final = "profile.bind"
#: The ``stopReason`` of a turn Kimi's own repeat breaker ended (``KIMI_CODE_REPEAT_BREAKER``:
#: the same call 12 times in a row, after reminders to answer): a finished turn with no answer.
REPEAT_BREAKER: Final = "repeat_breaker"


#: What an ACP agent the harness has no profile of is called when a person is asked about a
#: call it asks about.
THE_AGENT: Final = "The agent"


@dataclass(frozen=True, slots=True)
class Profile:
    """What the harness knows of an ACP agent it tested: its name (the ``compat.BINARIES``
    key), its built-in tools and their risk, the tools that start sub-agents (``helpers``) and
    swarms (``swarms``), its question tool (asked as the run's questions), how a run routes its
    model through the gateway (``routed``), where it logs its usage (``usage``), and what it
    is called when a person is asked about a call it asks about (``label``)."""

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
    #: what is set in the agent's environment whatever its own says: what a run of it cannot
    #: keep or govern, turned off (Kimi's scheduling, its tower mode) — and why each is, said
    #: on the run when the agent's own ``env`` asks otherwise (``env_why``)
    env: Mapping[str, str] = field(default_factory=dict)
    env_why: Mapping[str, str] = field(default_factory=dict)
    #: the rules of the agent's own configuration that let it run a call without asking, for
    #: an environment and a working folder: a run of an agent that holds any is not started
    #: (:data:`UNASKED`)
    allows: Callable[[Mapping[str, str], str], list[str]] | None = None
    #: the agent's home of a tenant's user (its ``home`` variable's value), for an
    #: environment, a working folder and whose it is: every run that is not sandboxed gets its
    #: user's own, never one shared with another user's (sessions, caches)
    homed: Callable[[Mapping[str, str], str, tuple[str, str]], str] | None = None
    label: str = THE_AGENT


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
        #: the main agent's step limit, when its turn ended there (:data:`AT_STEP_LIMIT`)
        self.limited: int | None = None
        #: whether the main agent's turn ended in the log (:data:`TURN_ENDED`): it is logged
        #: a moment after the turn's last usage, the step limit's reason in it
        self.ended = False
        #: whether Kimi's repeat breaker ended the main agent's latest turn (:data:`REPEAT_BREAKER`)
        self.broken = False
        #: the error Kimi could not finish the main agent's latest turn for (its provider's,
        #: past Kimi's own retries), when it ended ``failed`` anywhere but at its step limit
        self.failed: dict[str, Any] | None = None
        #: the agents whose log is not read (a background sub-agent's: its run's to count)
        self.skip: set[str] = set()

    def _files(self, agent: str = "*") -> list[Path]:
        found = self.home.glob(f"sessions/*/{self.session_id}/agents/{agent}/wire.jsonl")
        return sorted(path for path in found if path.parent.name not in self.skip)

    def profile(self, agent: str) -> str | None:
        """The profile Kimi started the sub-agent ``agent`` (its id) with: its type, which a
        resumed agent keeps (its log's ``profile.bind``); ``None`` when no log of it says."""
        for path in self._files(agent):
            for line in path.read_bytes().splitlines():
                entry = _json_line(line)
                if entry.get("type") == PROFILE_BOUND:
                    return str(entry.get("profileName") or "") or None
        return None

    def mode(self) -> tuple[str, str | None]:
        """The session mode Kimi's log says the main agent is in, as its ACP server names it
        (``plan`` while its plan mode is on, else ``default``), and the permission mode it
        approves on its own in there (``auto``, ``yolo``: it asks about nothing) — ``None``
        when it asks before a call (``manual``). Kimi's ``config.toml`` starts a session in
        either (``default_plan_mode``, ``default_permission_mode``) and its ACP server still
        answers ``default``: the log is what says. Read whole, once a session is opened."""
        planning, permission = False, KIMI_ASKS
        for path in self._files("main"):
            with path.open("rb") as file:
                for line in file:
                    entry = _json_line(line)
                    kind = entry.get("type")
                    if kind == PERMISSION_SET:
                        permission = str(entry.get("mode"))
                    elif kind in PLAN_MODE:
                        planning = PLAN_MODE[kind]
        return ("plan" if planning else "default"), (
            None if permission == KIMI_ASKS else permission
        )

    def asking_in(self, mode: str | None) -> tuple[str, tuple[str, ...]]:
        """The mode the session is to ask in — ``mode``, else the one it is in — and the
        modes to set, in order, to get it there. Kimi's ACP server refuses to enter the plan
        mode of a session already in it, leaving its permission mode as it was: so a session
        that plans and asks is set nothing (its plan as it is), and one that plans but
        approves on its own leaves plan mode first — only Kimi's ``config.toml`` starts a
        session so, its plan not begun."""
        at, alone = self.mode()
        wanted = mode or at
        if at == wanted == KIMI_PLANS:
            return wanted, (() if alone is None else (KIMI_DEFAULT, KIMI_PLANS))
        return wanted, (wanted,)

    def start(self) -> None:
        for path in self._files():
            self.read_to[path] = path.stat().st_size

    def entries(self) -> list[tuple[str, dict[str, Any]]]:
        """The lines written since the last read, each with the agent whose log it is in (a
        line still being written is left for the next)."""
        found: list[tuple[str, dict[str, Any]]] = []
        for path in self._files():
            at = self.read_to.get(path, 0)
            with path.open("rb") as file:
                file.seek(at)
                data = file.read()
            end = data.rfind(b"\n") + 1
            self.read_to[path] = at + end
            found.extend((path.parent.name, _json_line(line)) for line in data[:end].splitlines())
        return found

    def records(self) -> list[dict[str, Any]]:
        """The usage records written since the last read (and the main agent's turn's end)."""
        found: list[dict[str, Any]] = []
        for _, entry in self.entries():
            if entry.get("type") == USAGE_RECORD and isinstance(entry.get("usage"), dict):
                found.append(entry)
            elif entry.get("type") == TURN_ENDED and entry.get("agentId") == "main":
                self.ended = True
                self.broken = entry.get("stopReason") == REPEAT_BREAKER
                error = entry.get("error") or {}
                if error.get("code") == AT_STEP_LIMIT:
                    self.limited = int((error.get("details") or {}).get("maxSteps") or 0)
                elif entry.get("reason") == TURN_FAILED:
                    self.failed = error
        return found


def _answers(item: Any, answered: bool) -> bool:
    """Whether the turn has an answer after ``item``: text after its last tool call, as
    :meth:`ACPAdapter.extract` reads it."""
    from acp.schema import AgentMessageChunk, TextContentBlock, ToolCallStart

    if isinstance(item, ToolCallStart):
        return False
    if isinstance(item, AgentMessageChunk) and isinstance(item.content, TextContentBlock):
        return answered or bool(item.content.text)
    return answered


def _json_line(line: bytes) -> dict[str, Any]:
    try:
        entry = json.loads(line)
    except ValueError:
        return {}
    return entry if isinstance(entry, dict) else {}


def _kimi_usage(env: Mapping[str, str], session_id: str) -> UsageLog:
    home = env.get("KIMI_CODE_HOME") or KIMI_HOME
    return UsageLog(Path(home).expanduser(), session_id)


def _kimi_rules(env: Mapping[str, str], cwd: str) -> dict[str, list[str]]:
    """The permission rules of Kimi's own configuration, by decision (``allow``, ``ask``,
    ``deny``), each by its pattern: those of its ``config.toml`` (``[permission]``: a ``rules``
    entry with its decision; an ``allow``, ``ask`` or ``deny`` entry — a pattern, or a tool with
    what it matches) and of a project's ``.kimi-code/local.toml``, in ``cwd`` or a folder above
    it (Kimi reads its project's: every one above is looked at, which misses none). A file
    that is not TOML holds none: Kimi itself refuses to start on it."""
    home = Path(env.get("KIMI_CODE_HOME") or KIMI_HOME).expanduser()
    above = [Path(cwd), *Path(cwd).parents]
    found: dict[str, list[str]] = {"allow": [], "ask": [], "deny": []}
    for path in (home / "config.toml", *(folder / ".kimi-code" / "local.toml" for folder in above)):
        try:
            permission = tomllib.loads(path.read_text()).get("permission")
        except (OSError, tomllib.TOMLDecodeError):
            continue
        section = permission if isinstance(permission, dict) else {}
        rules = [(r.get("decision"), r) for r in section.get("rules") or ()]
        rules += [(decision, r) for decision in found for r in section.get(decision) or ()]
        for decision, rule in rules:
            tool, matches = rule.get("tool"), rule.get("match") or rule.get("pattern")
            pattern = f"{tool}({matches})" if tool and matches else str(tool or matches)
            found.setdefault(str(decision), []).append(pattern)
    return found


def kimi_allows(env: Mapping[str, str], cwd: str) -> list[str]:
    """The rules of Kimi's own configuration that let it run a call without asking, by their
    pattern: its ``allow`` rules (:func:`_kimi_rules`; since Kimi Code 2.6 they apply over
    ACP) that no ``ask`` rule of the same pattern stands before — Kimi's policy reads its
    ``ask`` rules first, so such a call is asked about."""
    rules = _kimi_rules(env, cwd)
    return [pattern for pattern in rules["allow"] if pattern not in rules["ask"]]


#: What of the team's Kimi home a user's own links to: its login, and what it set Kimi up
#: with. Its sessions, its logs and what Kimi caches are the user's own.
KIMI_SHARED: Final = ("credentials", "skills", "agents", "plugins", "AGENTS.md", "mcp.json")


def kimi_home(env: Mapping[str, str], cwd: str, owner: tuple[str, str]) -> str:
    """The Kimi home of a tenant's user (``owner``) on this worker, made for them alone under
    the XDG data folder (``$XDG_DATA_HOME/trellis/kimi``, else ``~/.local/share/trellis/kimi``;
    as Claude Code's config folder is): its sessions and what Kimi caches are theirs, never
    another user's. It is an overlay of the team's home (``env``'s ``KIMI_CODE_HOME``, else
    ``~/.kimi-code``): the team's login and setup are linked (:data:`KIMI_SHARED`), and its
    ``config.toml`` is the team's, as it is, with an ``ask`` rule after it for each ``allow``
    rule that would let a call run unasked (:func:`kimi_allows`, the project's too: Kimi reads
    ``ask`` rules first) — so Kimi asks, and governance decides. A config those rules cannot be
    added to (its ``[permission]`` rules written inline) is copied as it is: the run is then
    not started (:data:`UNASKED`)."""
    team = Path(env.get("KIMI_CODE_HOME") or KIMI_HOME).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    home = Path(base) / "trellis" / "kimi" / stable_id(*owner, prefix="u_")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in KIMI_SHARED:
        if (team / name).exists() and not (home / name).is_symlink():
            (home / name).symlink_to(team / name)
    config = team / "config.toml"
    given = config.read_text() if config.exists() else ""
    asks = "".join(
        f'\n[[permission.rules]]\ndecision = "ask"\npattern = {json.dumps(pattern)}\n'
        for pattern in kimi_allows(env, cwd)
    )
    try:
        tomllib.loads(given + asks)
    except tomllib.TOMLDecodeError:
        asks = ""
    part = home / "config.toml.part"
    part.write_text(given + asks)
    part.replace(home / "config.toml")
    return str(home)


#: Why a run of an agent whose own configuration lets it run calls without asking got no
#: prompt.
UNASKED: Final = (
    "{agent}'s own configuration lets it run calls without asking ({rules}: allow rules in "
    "its config.toml, or in a .kimi-code/local.toml of the project), and the harness governs "
    "only the calls the agent asks about. No prompt was sent. Remove those rules: under the "
    "harness the tool catalog's rules decide what runs without a person"
)

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
    env=KIMI_ENV,
    env_why=KIMI_ENV_WHY,
    allows=kimi_allows,
    homed=kimi_home,
    label=KIMI_CODE,
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
    skills folder, :class:`RunHome`), ``profile`` (what the harness knows of it:
    :data:`KIMI`) and ``reads`` (folders besides ``cwd`` its shell may read without asking a
    person: ``governance.shell``)."""

    command: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    mcp_servers: tuple[Mapping[str, Any], ...] = ()
    mode: str | None = None
    model: str | None = None
    sandbox: Any = None
    skills: Any = None
    profile: Profile | None = None
    reads: tuple[str, ...] = ()

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
    reads: Sequence[str] = (),
) -> ACPAgent:
    """Kimi Code (``@moonshot-ai/kimi-code``, its ``kimi acp`` server) as a target: ``model``
    the gateway's model name (with Bifrost; without, Kimi's own configuration — its login, or
    ``KIMI_MODEL_*`` in ``env``), ``cwd`` where it works, ``env`` over this process's,
    ``mcp_servers`` of its own beside the harness's, ``mode`` its session mode (``"default"``
    or ``"plan"``), ``sandbox`` a ``sandbox()`` its ``Bash`` commands and file writes go to
    (its reads do not: :class:`Sandboxed`), ``skills`` a ``SkillSet`` (``**src.kimi()``), its
    skills for each run, ``command`` the ``kimi`` executable, ``reads`` folders besides
    ``cwd`` a shell command may read without asking a person."""
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
        reads=tuple(reads),
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
    #: the tool call a person answered about (``Pending.call``), when the run resumes from
    #: that pause
    answered: ToolCall | None = None
    #: the person denied that call (``REJECT``): it does not run
    denied: bool = False
    #: the person answered a question that call asked while it ran (not its approval)
    asked: bool = False
    #: why they denied it, when they said
    why: str | None = None
    #: the run paused before its prompt was sent — on the question of a background
    #: sub-agent's run of its conversation (``live.Keeper.asks``): it is sent now
    unsent: bool = False
    #: that question's answer — the sub-agent's run, its interrupt, the person's resolution —
    #: for when its session is not on this server (``live.gone``)
    background_answer: tuple[str, str, InterruptResolution] | None = None


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
        #: each conversation's agent process, kept while work in its background goes on
        #: (``bridges.acp_live``)
        self.live = LiveSessions()

    async def aclose(self) -> None:
        """The agent is closing: every conversation's kept process ends."""
        await self.live.aclose()

    def steers(self, target: Any) -> bool:
        """None: ACP has no point between an agent's steps where a client adds a message to a
        turn under way (``session/prompt`` takes a turn's messages as it starts), so a
        steered message waits its turn."""
        return False

    def asks(self, target: Any) -> bool:
        """An agent whose profile names its question tool (Kimi's ``AskUserQuestion``)."""
        return target.profile is not None and bool(target.profile.questions)

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
        """The answer (the text after the last call) and the turn as said: each text, each
        call — a harness tool by its own name, not the agent's ``mcp__trellis__`` one, with its
        arguments as last told — and each call's result, so what memory keeps of the turn
        holds its steps."""
        from acp.schema import (
            AgentMessageChunk,
            PromptResponse,
            TextContentBlock,
            ToolCallProgress,
            ToolCallStart,
        )

        transcript: list[Said] = []
        text: list[str] = []
        at: dict[str, int] = {}  # each call's message in the transcript
        for item in output:
            if isinstance(item, AgentMessageChunk) and isinstance(item.content, TextContentBlock):
                text.append(item.content.text)
            elif isinstance(item, ToolCallStart):
                call = Called(
                    item.tool_call_id,
                    item.title.removeprefix(HARNESS_PREFIX),
                    _args(item.raw_input, item.content),
                )
                last = transcript[-1] if transcript and not text else None
                if last is not None and last.role is Speaker.ASSISTANT:  # made together
                    transcript[-1] = Said(Speaker.ASSISTANT, last.content, (*last.tool_calls, call))
                else:
                    transcript.append(Said(Speaker.ASSISTANT, "".join(text), (call,)))
                text = []
                at[item.tool_call_id] = len(transcript) - 1
            elif isinstance(item, ToolCallProgress) and item.tool_call_id in at:
                if item.raw_input is not None:
                    _argued(transcript, at[item.tool_call_id], item)
                if item.status in ("completed", "failed"):
                    transcript.append(
                        Said(Speaker.TOOL, _output(item), tool_call_id=item.tool_call_id)
                    )
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
        answered = shared.background_answer(pending, resolution)
        if answered is not None:
            return dataclasses.replace(native_input, unsent=True, background_answer=answered)
        denied = resolution.decision is InterruptDecision.REJECT
        return dataclasses.replace(
            native_input,
            answered=pending.call,
            denied=denied,
            asked=pending.interrupt.tool_call is None,
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

    def workdir(self, target: Any, options: Mapping[str, Any]) -> Path | None:
        """Where the agent works on this machine (its ``cwd``, else this process's); ``None``
        for a sandboxed run, whose writes go to the sandbox."""
        ran = given(target, options)
        return None if ran.sandbox is not None else Path(ran.cwd or os.getcwd())

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
        the attempt's helpers open while it runs. A background sub-agent's run has no prompt of
        its own: its attempt is its task's end (``acp_live.Background``)."""
        helpers = Helpers(run.runtime, self.name)
        error: BaseException | None = None
        found = cast("Background | None", self.live.attended.get(run.runtime.run_id))
        source = (
            self._attempt(target, native_input, run)
            if found is None
            else self._attended(run, found)
        )
        try:
            async with contextlib.aclosing(source) as items:
                async for item in items:
                    yield item
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error, paused=helpers.suspended(error))

    async def _attended(self, run: Invocation, found: Background) -> AsyncGenerator[Any, None]:
        """One attempt of a background sub-agent's run: what its sub-agent asks permission for
        decided in it (:class:`Deciding`), its harness tools its own, until its task ends."""
        runtime = run.runtime
        deciding = Deciding(runtime, found.session.profile, found.session.shell)
        turn = Turn(runtime, deciding, run.native_tools, contextvars.copy_context())
        async with contextlib.aclosing(found.served(turn)) as items:
            async for item in items:
                yield item

    async def _attempt(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        from acp.schema import (
            ClientCapabilities,
            ElicitationCapabilities,
            ElicitationFormCapabilities,
        )

        runtime = run.runtime
        target = given(target, runtime.framework_options)
        profile = target.profile
        if profile is not None and profile.homed and profile.home and target.sandbox is None:
            # its user's own home of the agent's, an overlay of the team's (``kimi_home``)
            team, owner = {**os.environ, **target.env}, (runtime.tenant, runtime.user)
            homed = profile.homed(team, target.cwd or os.getcwd(), owner)
            target = dataclasses.replace(target, env={**target.env, profile.home: homed})
        own = target.sandbox is not None or target.skills is not None
        home = RunHome(target, runtime) if own else None
        boxed = Sandboxed(target, runtime, home) if home is not None and target.sandbox else None
        env = home.env() if home is not None else {**os.environ, **target.env}
        if profile is not None and profile.routed is not None:
            profile.routed(env, runtime, target.model)
        if profile is not None and profile.swarm_limit is not None:
            env.setdefault(profile.swarm_limit, str(admission.ADMISSION.limit))
        if profile is not None:
            _overridden(profile, target.env, runtime)
            env.update(profile.env)
        cwd = boxed.work if boxed is not None else target.cwd or os.getcwd()
        attempt = Attempt(self, runtime, target, boxed)
        # where its shell works, the folders the target names, the run's own skills, where a
        # kept result is read (the name the model is told): what a command may read unasked
        reads = [str(home.agent_home / "skills")] if home is not None else []
        attempt.shell = Shell(cwd, [*target.reads, *reads, results.PREFIX], env)
        attempt.served = run.native_tools
        # a form of the agent's for its user is answered (Kimi asks every question of an
        # ``AskUserQuestion`` call in one form; over a permission request, only the first)
        forms = ElicitationCapabilities(form=ElicitationFormCapabilities())
        served = boxed.capabilities() if boxed is not None else {}
        capabilities = ClientCapabilities(elicitation=forms, **served)
        # a session another worker's kept process still holds (a background sub-agent of the
        # run at work there) is forked: the turn goes on here, in a process of its own
        journal = runtime.replay.journal
        held = journal.session is not None and await shared.held_elsewhere(runtime, self.live)
        # a conversation's process is kept past its turn while work in its background goes
        # on; not a run's whose home is its own (removed with it), nor one that forks
        keeps = shared.converses(runtime) and not own and not held and _follows(profile)
        await self._gone(runtime, native_input)
        async with contextlib.AsyncExitStack() as stack:
            if home is not None:
                stack.push_async_callback(home.closed)
                await home.skills()
            if profile is not None and profile.allows is not None:
                unasked = ", ".join(profile.allows(env, cwd))
                if unasked:  # fails closed: nothing the agent would run unasked is run
                    raise ACPRunError(UNASKED.format(agent=profile.label, rules=unasked))
            started = Started(target, env, cwd, capabilities, attempt.shell, fork=held)
            key = runtime.conversation if keeps else None
            session, attempt.resumed = await self._session(key, started, run)
            try:
                if runtime.attempt == 1 or native_input.unsent:  # its prompt is still to send
                    await session.asks(runtime)
                async with contextlib.aclosing(session.served(attempt, native_input)) as items:
                    async for item in items:
                        yield item
            finally:
                await self._kept(session, runtime)

    async def _session(
        self, key: str | None, started: Started, run: Invocation
    ) -> tuple[ACPSession, bool]:
        """The session the turn runs in: its conversation's kept process, when it was started
        with this turn's settings, holds the session the turn continues and is free (no turn of
        the agent's own under way); else a new process (``key``: a conversation's, kept past
        the turn while work in its background goes on). And whether it holds what came before
        (a kept one, a session resumed, loaded or forked)."""
        runtime, target = run.runtime, started.target
        form = shape(target, started.env, run.native_tools, KIMI_HEADERS)
        found = cast("ACPSession | None", self.live.get(key)) if key is not None else None
        if found is not None:
            why = None
            if found.shape != form:
                why = "this turn's settings are not the ones it was started with"
            elif found.session_id != runtime.replay.journal.session:
                why = "this turn does not continue the session it holds"
            elif not await found.settled():
                why = "a turn of Kimi Code's own did not end"
            if why is not None:
                await self.live.close(found.key, why)
                found = None
        resumed = found is not None
        if found is None:
            found = await self._started(key, form, started, run)
            resumed = found.continued
        shared.said(self.live, runtime, CODE)
        return found, resumed

    async def _started(
        self, key: str | None, form: str, started: Started, run: Invocation
    ) -> ACPSession:
        """A new process for the turn, its session the one the run continues (resumed, else
        loaded; forked when another worker's process holds it), or a new one."""
        from acp import PROTOCOL_VERSION

        runtime, target, env = run.runtime, started.target, started.env
        profile = target.profile
        model = target.model or ACP_MODEL
        session = ACPSession(
            key or runtime.run_id,
            form,
            self.live,
            owner=(runtime.tenant, runtime.user),
            keeps=key is not None,
            model=model,
        )
        session.profile, session.shell = profile, started.shell
        tools = run.native_tools

        def spawned(client: Any) -> Any:
            return _spawned(
                client,
                *target.command,
                env=env,
                cwd=started.cwd,
                transport_kwargs={"limit": _buffer_limit()},
                use_unstable_protocol=True,  # ``elicitation/create``: the agent's forms
            )

        def served(relay: ACPSession) -> Any:
            from trellis.harness.tools.convert import mcp

            return _harness_server(mcp.relayed(tools, relay.call))

        await session.start(spawned, served if tools is not None else None)
        try:
            init = await session.answered(
                session.conn.initialize(
                    protocol_version=PROTOCOL_VERSION, client_capabilities=started.capabilities
                )
            )
            if profile is not None:
                found = init.agent_info.version if init.agent_info is not None else ""
                compat.require_binary(profile.name, found, command=target.command[0])
            servers = [*(_server(s) for s in target.mcp_servers), *session.servers]
            opened = await session.opened(init, started.cwd, servers, runtime, fork=started.fork)
            usage = profile.usage if profile is not None else None
            await session.asking(target.mode, usage(env, opened) if usage is not None else None)
            if usage is not None:
                session.usage = usage(env, opened)
                session.usage.start()
                session.watch(usage(env, opened))
        except BaseException:
            await session.close()
            raise
        return session

    async def _kept(self, session: ACPSession, runtime: Runtime) -> None:
        """The turn ended: its conversation keeps the process while work in its background goes
        on (and there is room for it: ``live.KEPT``, over the server's agents); any other
        process ends."""
        busy = session.keeps and not session.closed and await session.busy()
        if busy and self.live.get(session.key) is session:
            return
        if busy:
            async with self.live.place(shared.among(runtime, self), owner=session.owner) as room:
                if room:
                    self.live.keep(session)
                    return
        if self.live.get(session.key) is session:
            self.live.forget(session, "nothing works in its background")
        await session.close()

    async def _gone(self, runtime: Runtime, native_input: Any) -> None:
        """The answer this turn brings to a background sub-agent's question
        (``background_answer``), when no session here holds that sub-agent (a killed
        server's): nothing continues it, so its run records the answer and ends
        (``live.gone``)."""
        answered = native_input.background_answer
        if answered is not None and answered[0] not in self.live.attended:
            await shared.gone(runtime, *answered, why=GONE, said=GONE_SAID, code=CODE)

    def observed(self, name: str) -> None:
        """A tool the agent ran itself: an MCP server's joins the tool set."""
        if name.startswith("mcp__") and not name.startswith(HARNESS_PREFIX):
            self.seen.add(name)


@contextlib.asynccontextmanager
async def _spawned(client: Any, *command: str, **options: Any) -> AsyncIterator[tuple[Any, Any]]:
    """The agent's process and its connection (the SDK's ``spawn_agent_process``), stopped by
    the SDK's own shutdown — a process already gone not an error. The shutdown waits for the
    process (``shutdown_timeout``), then signals it; when the wait's deadline falls in the
    moment the process ends, asyncio loses the wait's wake-up, and the signal raises
    ``ProcessLookupError`` (its transport is closed: the process exited, its pipes too). The
    process gone is what the shutdown is for: the attempt's own outcome stands."""
    from acp import spawn_agent_process

    spawned = spawn_agent_process(client, *command, **options)
    pair = await spawned.__aenter__()
    try:
        yield pair
    finally:  # its session ends it (``ACPSession.close``): never with an error of the turn's
        with contextlib.suppress(ProcessLookupError):
            await spawned.__aexit__(None, None, None)


def _overridden(profile: Profile, own: Mapping[str, str], runtime: Runtime) -> None:
    """The run told, for each variable the profile sets whatever the agent's own ``env`` says
    (:attr:`Profile.env`), that the agent's own value is not used, and why."""
    for name, value in profile.env.items():
        if name in own and own[name] != value:
            why = profile.env_why.get(name, "")
            said = f"{name}={own[name]} in the agent's env is not used: {why}"
            runtime.events.warning("acp_env", said)


def _buffer_limit() -> int:
    """The most one message from the agent may be: the SDK's default (50 MiB) — a tool's
    result travels in one."""
    from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES

    return DEFAULT_STDIO_BUFFER_LIMIT_BYTES


@dataclass(frozen=True, slots=True)
class Started:
    """What an attempt starts the agent's process with: the target (with the run's framework
    options), its environment, its working directory, the client's capabilities, its shell
    (``governance.shell``), and whether the session it continues is forked (another worker's
    process holds it: ``live.held_elsewhere``)."""

    target: ACPAgent
    env: dict[str, str]
    cwd: str
    capabilities: Any
    shell: Shell
    fork: bool = False


def _follows(profile: Profile | None) -> bool:
    """Whether the harness can follow the agent's background work: it keeps a log the harness
    reads (its profile's ``usage``: Kimi's), which says what ACP does not."""
    return profile is not None and profile.usage is not None


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
    risk (its profile's) and tagged ``BUILTIN``, an MCP server's tool (``mcp__<server>__<tool>``)
    by its server."""
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return ToolSpec(name=name, source="mcp", server=parts[1])
    builtins = profile.builtins if profile is not None else {}
    spec = builtin(ToolSpec(name=name, side_effects=builtins.get(name, UNDECLARED)))
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
    #: its arguments came as ``rawInput`` (final), not only as the text of its updates
    given: bool = False
    #: the client's terminal it runs in (a sandboxed run's ``Bash``): its output is the call's
    terminal: str | None = None


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
        #: the agent's process the attempt is served by (``acp_live.ACPSession``), its
        #: connection and its session's id
        self.session: ACPSession | None = None
        self.conn: Any = None
        self.session_id: str | None = None
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.calls: dict[str, _Call] = {}
        #: the agent's shell: where it works, what else the run may read, its environment
        #: (``governance.shell``); Kimi Code's starts each command in its folder (or the
        #: call's ``cwd``)
        self.shell = Shell(None)
        self.usage: UsageLog | None = None
        #: what the agent is told next, in a prompt of its own, once the turn has stopped
        self.followup: str | None = None
        #: the call a reviewer edited: approved when the agent makes it with those arguments
        self.edited: tuple[str, dict[str, Any]] | None = None
        #: the turn was cancelled (a pause, a refusal, the budget): the agent is stopping
        self.cancelled = False
        #: the session holds what came before (kept, resumed, loaded or forked)
        self.resumed = False
        #: the agent's requests being answered now (each in a task of the connection's)
        self.handling: set[asyncio.Task[Any]] = set()
        #: the harness's tools as the run has them (``tools.convert.mcp.Tools``)
        self.served: Any = None

    def bind(self, session: ACPSession) -> None:
        """The attempt is served by ``session``: its connection, its session, its usage log."""
        self.session = session
        self.conn = session.conn
        self.session_id = session.session_id
        self.usage = session.usage

    # ------------------------------------------------------------------ the turns
    async def turns(self, native_input: ACPInput) -> AsyncGenerator[Any, None]:
        """The turn's updates and its answer, then — when the run stopped it to tell the agent
        something (a rejection's reason, an edit) — the next turn's, until one ends on its own
        or the run pauses."""
        prompt = self._first(native_input)
        asked_once = False
        while prompt is not None:
            self.cancelled = False
            answered = False
            async with contextlib.aclosing(self._turn(prompt)) as items:
                async for item in items:
                    answered = _answers(item, answered)
                    yield item
            prompt, self.followup = self.followup, None
            if self.runtime.pending is not None or self.runtime.budget_spent is not None:
                return
            told = None if prompt is not None or asked_once else self._unanswered(answered)
            if told is not None:  # the turn ended with no answer: asked once more
                asked_once, prompt = True, told

    def _unanswered(self, answered: bool) -> str | None:
        """What the agent is told when its turn ended with no answer where the run asks once
        more, in the same session, said as a warning: a loop on one call — refused by the
        bridge (``bridge.stalled``, which said so) or ended by Kimi's own repeat breaker
        (:data:`REPEAT_BREAKER`, ``bridge.TURN_STALLED``). ``None``: not asked. At the agent's
        step limit (:data:`AT_STEP_LIMIT`) nothing is asked (decision D1, as Claude's turn
        limit): the run ends ``PARTIAL`` with what Kimi said, the limit as the budget it spent
        (``Runtime.budget_spent``) and a ``step_limit`` warning — no paid last answer."""
        usage, runtime = self.usage, self.runtime
        limit = usage.limited if usage is not None else None
        if limit is not None:
            usage.limited = None  # type: ignore[union-attr]  # read once
            stopped = f"the step limit ({limit}) was reached"
            log.warning("run %s stopped: %s", runtime.run_id, stopped)
            runtime.events.warning("step_limit", f"stopped: {stopped}")
            runtime.budget_spent = stopped
            return None
        if answered:
            return None
        if bridge.stalled(runtime):
            return bridge.TURN_STALLED
        if usage is not None and usage.broken:
            stopped = "stopped: the same call over and over (the agent's repeat breaker)"
            log.warning("run %s %s", runtime.run_id, stopped)
            runtime.events.warning("stalled", stopped)
            return bridge.TURN_STALLED
        return None

    def _first(self, native_input: ACPInput) -> list[Any]:
        from acp import text_block

        if self.resumed and self.runtime.attempt > 1 and not native_input.unsent:
            # the run's own session, after a pause: told which call a person answered about
            return [text_block(self._resumed(native_input))]
        from trellis.harness.sandbox import DELIVERS_HERE

        blocks = [text_block(native_input.prompt)]
        if native_input.context:  # marked off: the question is the block after it
            blocks.insert(0, text_block(ASKED_WITH.format(context=native_input.context)))
        if self.target.sandbox is None:  # its own folder's outputs/ are the user's (workdir)
            blocks.insert(0, text_block(DELIVERS_HERE))
        if len(blocks) == 1:
            # the user's words would start the prompt, where ACP reads a ``/`` as a command of
            # the agent's (Kimi's ``/compact``, ``/skill:<name>``: run with no model turn).
            # They are words for the model, as in every run whose prompt has a block before
            blocks = [text_block(f"\n{native_input.prompt}")]
        return blocks

    def _resumed(self, native_input: ACPInput) -> str:
        """What the resumed session is told: the call a person answered about, named as the
        agent calls it (a harness tool through the ``trellis`` server, its own by its name)."""
        call = native_input.answered
        if call is None:
            return RESUMED
        harness = call.tool in self.runtime.toolbox
        name = f"{HARNESS_PREFIX}{call.tool}" if harness else call.tool
        said = resumed_call(call, name, denied=native_input.denied, asked=native_input.asked)
        why = native_input.why
        return f"{said} {DENIED_WHY.format(why=why)}" if why else said

    async def _turn(self, prompt: Any) -> AsyncGenerator[Any, None]:
        """One prompt's updates and its answer. It is sent once the agent is free (no turn of
        its own under way: ``Keeper.settled``); an answer that came with no turn of the
        prompt's own — the agent queued it behind a turn of its own — fails the turn
        (``acp_live.QUEUED``)."""
        if isinstance(prompt, str):
            from acp import text_block

            prompt = [text_block(prompt)]
        runtime, session = self.runtime, self.session
        assert session is not None  # bound before its turns
        if not await session.settled():
            raise ACPRunError(QUEUED)
        own, seen = session.own_turns, False
        if self.usage is not None:
            self.usage.ended, self.usage.failed = False, None  # this turn's end is to come
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
                    await self.launched()
                    continue
                self._died()
                if isinstance(item, _Ended):
                    yield await self._answer(item, own, seen)
                    return
                seen = True
                if await self._seen(item):
                    await self._spent()  # a call ended: what its step spent
                yield item
                if runtime.pending is not None:
                    await self.cancel()  # the run paused: the agent stops here
        finally:
            runtime.model_ended(STEP)
            if not asking.done():
                asking.cancel()

    async def _answer(self, ended: _Ended, own: int, seen: bool) -> Any:
        """The prompt's answer, its turn over and its usage counted: the agent's error raised
        as the run's; and an ``end_turn`` with nothing ``seen`` of a turn fails it when the
        agent queued the prompt behind a turn of its own (``acp_live.QUEUED``; ``own``: how
        many of those had begun when the prompt was sent)."""
        session = self.session
        assert session is not None  # bound before its turns
        stop = getattr(ended.answer, "stop_reason", None)
        if stop == END_TURN and not seen and await session.queued(own):
            raise ACPRunError(QUEUED)
        await self._settled()
        await self.launched()
        if ended.error is not None:
            raise await session.failed(ended.error)
        self._answered_usage(ended.answer)
        failed = self.usage.failed if self.usage is not None else None
        if failed is not None:  # answered ``end_turn`` like a finished turn: its log says why
            details = {k: v for k, v in failed.items() if v not in (None, "", {})}
            raise ACPRunError(FAILED_TURN.format(why=json.dumps(details, default=str)))
        # past the agent's own retries: what a run with no answer says
        self.runtime.cut_off = stop == MAX_TOKENS
        return ended.answer

    async def launched(self) -> None:
        """The turn's ``Agent`` calls whose sub-agent the agent launched in the background
        (``ACPSession.launches``: its log says, ACP sends no end of such a call) are over for
        the turn: the model is waited on again once none of its calls runs."""
        assert self.session is not None  # bound before its turns
        for call_id in await self.session.launches(self.calls):
            self.calls[call_id].running = False
            if not any(call.running for call in self.calls.values()):
                self.runtime.model_started(STEP)

    def _died(self) -> None:
        """What ended a harness call that was not an error (the worker dying), raised in the
        attempt: in the server's request it ended only the server."""
        fatal = self.served.fatal if self.served is not None else None
        if fatal is not None:
            raise fatal

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
        moment behind) read until the turn's end is in it (:data:`TURN_ENDED`, with a step
        limit's reason) and a read finds nothing new — accounted, the turn being over; at most
        :data:`SETTLE_SECONDS`."""
        usage = self.usage
        if usage is None:
            return
        ends = asyncio.get_running_loop().time() + SETTLE_SECONDS  # a log that keeps growing
        self._accounted()
        await asyncio.sleep(SETTLE_STEP)
        while (self._accounted() or not usage.ended) and asyncio.get_running_loop().time() < ends:  # noqa: ASYNC110 - a file, polled
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
            accounted(self.runtime, record, self.target.model)
        return len(found)

    # ------------------------------------------------------------------ what the agent did
    async def _seen(self, update: Any) -> bool:
        """A session update: a tool call running or ending (recorded; a sub-agent's a helper of
        the run); the stall watch follows the calls. Whether a call ended."""
        from acp.schema import AgentPlanUpdate, ToolCallProgress

        runtime = self.runtime
        if isinstance(update, AgentPlanUpdate):  # the agent's plan (Kimi's todo list), as it is
            entries = [entry.model_dump(exclude_none=True) for entry in update.entries]
            runtime.events.custom(PLAN, entries=entries)
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
        if not text and call.terminal is not None and self.boxed is not None:
            # run in the client's terminal: the agent's last update names it and says nothing
            text = self.boxed.said(call.terminal)
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
        """A sub-agent the agent starts (its profile's ``helpers``) or goes on with (Kimi's
        ``resume``), or each member of a swarm (``swarms``: one per item, its task the template
        with the item in it; one per sub-agent it resumes, its task the prompt it goes on with),
        a helper — a resumed one by the name it was started with (:meth:`_named`)."""
        helpers, profile = self.runtime.helpers, self.profile
        if profile is None:  # an agent the harness knows no sub-agent tool of
            return
        assert helpers is not None  # the turn runs with the attempt's helpers (``_turns``)
        args = call.args
        if call.name in profile.helpers:
            task = str(args.get("prompt") or args.get("description") or "")
            resumed = args.get("resume")
            name = self._named(str(resumed)) if resumed else self._named(None, args)
            await helpers.start(call_id, kind=Kind.SUBAGENT, name=name, task=task)
            if args.get("run_in_background"):
                # its call returns at once and ACP tells no end of it: its end is its task's
                # (the log's: ``ACPSession.launches``), else it is cut off with the attempt
                helpers.attribute(call_id, call_id, background=True)
        elif call.name in profile.swarms:
            for key, (kind, which), task in _swarm(call_id, args):
                name = self._named(which if kind == "resume" else None, args)
                await helpers.start(key, kind=Kind.SUBAGENT, name=name, task=task)

    def _named(self, resumed: str | None, args: Mapping[str, Any] | None = None) -> str:
        """A sub-agent's name: the type its call names (Kimi's default when none); a resumed
        one's (``resumed``: its id), the type it was started with, as Kimi's log says
        (:meth:`UsageLog.profile`) — its id when no log says."""
        if resumed is None:
            return str((args or {}).get("subagent_type") or KIMI_SUBAGENT)
        found = self.usage.profile(resumed) if self.usage is not None else None
        return found or resumed

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
            members = _members(text)
            for key, which, _ in _swarm(call_id, call.args):
                member = members.get(which)
                if member is not None and member[0] == "completed":
                    await helpers.end(key, output=member[1])
                else:
                    why = member[1] if member is not None else text
                    await helpers.end(key, error=why or SUBAGENT_FAILED)

    # ------------------------------------------------------------------ the ACP client
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """An update, queued for the turn; a call's name and arguments kept at once (a
        permission request for it may come before the turn reads its update)."""
        from acp.schema import ToolCallProgress, ToolCallStart

        if isinstance(update, ToolCallStart):
            args = _args(update.raw_input, update.content)
            given = update.raw_input is not None
            self.calls[update.tool_call_id] = _Call(name=update.title, args=args, given=given)
        elif isinstance(update, ToolCallProgress):
            known = self.calls.get(update.tool_call_id)
            ran_in = next(
                (b.terminal_id for b in update.content or () if b.type == "terminal"), None
            )
            if known is not None and ran_in is not None:
                known.terminal = ran_in
            if known is not None and update.raw_input is not None:
                known.args, known.given = _args(update.raw_input, None), True
            elif known is not None and not known.given and update.raw_output is None:
                # the arguments as the model streams them (Kimi's ``tool.call.delta``: the
                # text so far), before it asks permission for the call and only then starts it
                streamed = _streamed(update.content)
                if streamed is not None:
                    known.args = streamed
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
            # its call through the harness's MCP server names no call: said here
            known = self.calls.get(tool_call.tool_call_id)
            args = known.args if known is not None else _args(tool_call.raw_input, None)
            called = call_key(name.removeprefix(HARNESS_PREFIX), args)
            self.runtime.replay.announced(called, tool_call.tool_call_id)
            return chosen(options, "allow_once")
        if _offers_a_choice(options):
            return await self._question(options, name, tool_call)
        return await self._decided(options, name, tool_call)

    async def _decided(self, options: list[Any], name: str, tool_call: Any) -> Any:
        """A call of the agent's own, decided as the run decides a call."""
        runtime = self.runtime
        known = self.calls.get(tool_call.tool_call_id)
        args = known.args if known is not None else _args(tool_call.raw_input, tool_call.content)
        if self.edited == (name, args):
            self.edited = None
            return chosen(options, "allow_once")
        spec = _spec(runtime, self.profile, name)
        runtime.model_ended(STEP)
        asks_as = self.profile.label if self.profile is not None else THE_AGENT
        try:
            request = (tool_call.tool_call_id, options)
            decided = await _permitted(spec, args, self.shell, asks_as, request)
        except Paused:  # the run pauses: the agent stops here, and the resume goes on
            self._refused(known)
            await self._stop()
            return dismissed()
        if isinstance(decided, ToolOutcome):
            self._refused(known)
            self.followup = REJECTED.format(tool=name, why=str(decided.output))
            await self._stop()
            return chosen(options, "reject_once")
        if decided.args != args:
            self._refused(known)
            self.edited = (name, decided.args)
            self.followup = EDITED.format(tool=name, args=json.dumps(decided.args))
            await self._stop()
            return chosen(options, "reject_once")
        return _let_through(runtime, name, options)

    async def _question(self, options: list[Any], name: str, tool_call: Any) -> Any:
        """The agent's question to its user, asked as the run's (:func:`_asked`); the run
        paused on it, the agent stops here."""
        args = self.calls[tool_call.tool_call_id].args  # its call's update came before it
        try:
            return await _asked(self.runtime, options, name, args, tool_call)
        except Paused:
            await self._stop()
            return dismissed()

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        """A form the agent asks its user to fill (Kimi's ``AskUserQuestion``: every question
        of the call, a field each), asked as the run's question (:func:`_filled`) on behalf of
        the call it is for; the run paused on it, the agent stops here."""
        self._handles()
        known = self.calls[mode.tool_call_id]  # its call's update came before it
        try:
            async with self.runtime.asking_within(known.name, known.args):
                return await _filled(self.runtime, message, mode)
        except Paused:
            await self._stop()
            return unfilled()

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
      resume reads the same ones — beside the agent's own skills of its usual home, each linked
      there (a name the run has a skill of is the run's);
    * **a run that is not sandboxed** keeps this process's environment, and what the agent
      keeps in its usual home for every session (:data:`LINKED`: its configuration, its login,
      its instructions, its plugins) is linked into the run's home — and so are its sessions
      (:data:`SESSIONS`), which stay in
      the usual home: the conversation's next turn resumes the session this run ends in
      (``sessions``), where a home removed with its run would have taken the session with it;
    * **a sandboxed run** (:class:`Sandboxed`) gets an environment of :data:`SANDBOXED_ENV`
      and the agent's own ``env`` only, its ``HOME`` the run's too, nothing linked: no
      credential of this machine's reaches it (its model comes through the gateway, or from
      ``env``), and its session ends with it (:meth:`closed`)."""

    #: what of the agent's usual home a run's home links to: its configuration, its login, its
    #: instructions for every session (Kimi's ``AGENTS.md``), its plugins (and their skills)
    LINKED: Final = ("config.toml", "credentials", "AGENTS.md", "plugins")
    #: the folder of its home the agent keeps its sessions in, linked too (made if missing)
    SESSIONS: Final = "sessions"

    def __init__(self, target: ACPAgent, runtime: Runtime) -> None:
        self.target = target
        self.runtime = runtime
        self.root = Path(tempfile.gettempdir()) / SANDBOXED_ROOT / stable_id(runtime.run_id)
        self.home = self.root / "home"
        self.agent_home = self.home / ".agent"
        self.work = str(self.root / "work")
        self.agent_home.mkdir(parents=True, exist_ok=True)
        Path(self.work).mkdir(parents=True, exist_ok=True)
        #: the agent's usual home, linked (a run that is not sandboxed: :meth:`env`)
        self.usual: Path | None = None

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
        self.usual = usual
        (usual / self.SESSIONS).mkdir(parents=True, exist_ok=True)
        for name in (*self.LINKED, self.SESSIONS):
            found, link = usual / name, self.agent_home / name
            if found.exists() and not link.exists():
                link.symlink_to(found)

    async def skills(self) -> None:
        """The run's pinned skills, in the agent's skills folder (``<home>/skills``), and the
        agent's own skills of its usual home linked beside them (a name the run has is the
        run's)."""
        if self.target.skills is None:
            return
        from trellis.harness import skills

        folder = self.agent_home / "skills"
        shutil.rmtree(folder, ignore_errors=True)  # an earlier attempt's
        await skills.write_plugin(self.target.skills, self.agent_home)
        own = self.usual / "skills" if self.usual is not None else None
        if own is None or not own.is_dir():
            return
        folder.mkdir(exist_ok=True)
        for found in sorted(own.iterdir()):
            link = folder / found.name
            if not link.exists():
                link.symlink_to(found)

    async def closed(self) -> None:
        """The attempt ended: the run's home and working directory removed with the run (kept
        while it waits on a person: the next attempt resumes the agent's session from them).
        A sandboxed run's sessions were in that home (nothing of it is linked), so its session
        is forgotten with it: the conversation's next turn takes its earlier turns from the
        harness, not from a session that is gone."""
        if self.runtime.pending is None:
            shutil.rmtree(self.root, ignore_errors=True)
            if self.target.sandbox is not None:
                self.runtime.replay.journal.session = None


class Sandboxed:
    """A run's agent working with the run's sandbox (``ACPAgent(sandbox=sandbox())``): it is
    given ACP's file and terminal capabilities, served here — ``fs/write_text_file`` and
    ``fs/read_text_file`` the sandbox's files, ``terminal/*`` a command run in it (Kimi's
    ``Bash``) — so its commands and writes happen in the sandbox (``SandboxSource.opened``: the
    run's own, kept across attempts, deleted at the run's end).

    Its reads are not isolated: Kimi Code reads and searches the host's files itself
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
        #: what the latest terminals the agent released said (:data:`TERMINALS_KEPT`): a
        #: call's record reads it once the call ends, after its terminal went
        self.released: LRU[str, str] = LRU(TERMINALS_KEPT)
        self._made = False
        runtime.added.append(self.source)  # its end (and its pause) find it
        _said_reads(runtime.agent_id)
        runtime.events.warning("acp_sandbox", READS_NOT_ISOLATED)

    def capabilities(self) -> dict[str, Any]:
        """What the client serves the agent for a sandboxed run (``ClientCapabilities``'
        fields): its files and its terminals."""
        from acp.schema import FileSystemCapabilities

        files = FileSystemCapabilities(read_text_file=True, write_text_file=True)
        return {"fs": files, "terminal": True}

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
        self.released[ident] = _ran(task)[0] if task.done() else ""
        task.cancel()

    def said(self, ident: str) -> str:
        """The output of the command the terminal ``ident`` ran, once the agent released it
        (it does before it ends the call that ran in it): what that call is recorded with."""
        return self.released.get(ident) or ""

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


async def _permitted(
    spec: ToolSpec, args: dict[str, Any], shell: Shell, asks_as: str, request: Any
) -> ToolCall | ToolOutcome:
    """A call the agent asked permission for: decided as the run decides a call
    (``bridge.permitted``: hooks, governance), and — since the agent asks — a person asked
    (``asks_as``: what the agent is called in the question), once for the call: not again when
    governance or a hook just asked about it. Not asked for the agent's sake when it only
    reads (a read tool, or a shell command ``shell`` reads as read-only: placed from the
    agent's working folder, or the call's own ``cwd``, and reading only what the run may read)
    or when the run changed its arguments (the agent is told to call it again so: that call is
    the one let through). ``request``: the agent's permission request — its call's id, and
    whether it offers to keep an approval for its session (:func:`_keeps`: the person may then
    allow the tool for the conversation). A pause propagates."""
    reads = spec.side_effects == "read" or shell.read_only(spec.name, args)
    framework = None if reads else asks_as
    call_id, options = request
    return await bridge.permitted(
        spec, args, framework=framework, call_id=call_id, kept=_keeps(options)
    )


def _keeps(options: Sequence[Any]) -> bool:
    """Whether the agent offers to keep an approval of the call for its session (ACP's
    ``allow_always``: Kimi's "Approve for this session")."""
    return any(option.kind == "allow_always" for option in options)


def _let_through(runtime: Runtime, name: str, options: list[Any]) -> Any:
    """The answer that lets a call of the tool ``name`` run: the agent's approval for its
    session where it offers one and the person allowed the tool for the conversation
    (``bridge.remembered``) — the agent then asks about that call no more, as it would not
    for its own user — else for this call."""
    lasting = _keeps(options) and bridge.remembered(runtime, name)
    return chosen(options, "allow_always" if lasting else "allow_once")


def _swarm(call_id: str, args: Mapping[str, Any]) -> list[tuple[str, tuple[str, str], str]]:
    """The members a swarm call starts: each sub-agent it resumes (``resume_agent_ids``, its id
    with the prompt it goes on with), then one per item (its task the template with the item in
    it) — each with its key among the attempt's helpers, what it is (``("resume", <agent id>)``,
    ``("item", <item>)``: as :func:`_members` finds it) and its task."""
    template = str(args.get("prompt_template") or "{{item}}")
    resumed = (args.get("resume_agent_ids") or {}).items()
    return [
        *((f"{call_id}@{a}", ("resume", str(a)), str(p)) for a, p in resumed),
        *(
            (f"{call_id}/{i}", ("item", str(i)), template.replace("{{item}}", str(i)))
            for i in args.get("items") or ()
        ),
    ]


def _members(text: str) -> dict[tuple[str, str], tuple[str, str]]:
    """The members a swarm's result reports, each by what it was — a resumed sub-agent by its
    id (``("resume", <agent id>)``), any other by its item (``("item", <item>)``) — with its
    outcome and what it said."""
    found: dict[tuple[str, str], tuple[str, str]] = {}
    for member in SWARM_MEMBER.finditer(text):
        attrs = {k: html.unescape(v) for k, v in SWARM_ATTR.findall(member["attrs"])}
        resumed = attrs.get("mode") == "resume"
        which = (
            ("resume", attrs.get("agent_id", "")) if resumed else ("item", attrs.get("item", ""))
        )
        found[which] = (attrs.get("outcome", ""), member["text"].strip())
    return found


class Deciding:
    """How a background sub-agent's run answers what its sub-agent asks permission for (the
    ``options`` of its attempt's ``live.Turn``): a harness tool's call is let through (the
    bridge decides it as it runs); any other is decided as the run decides a call — let
    through, refused, or (the run paused on it) :data:`acp_live.WAITS`, decided again by the
    attempt that follows the person's answer. Nothing is told the sub-agent (ACP's answer
    carries no reason, and only the main agent takes a prompt)."""

    def __init__(self, runtime: Runtime, profile: Profile | None, shell: Shell) -> None:
        self.runtime = runtime
        self.profile = profile
        self.shell = shell

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        name = tool_call.title or ""
        if name.startswith(HARNESS_PREFIX):
            return chosen(options, "allow_once")
        args = _args(tool_call.raw_input, tool_call.content)
        spec = _spec(self.runtime, self.profile, name)
        asks_as = self.profile.label if self.profile is not None else THE_AGENT
        try:
            request = (tool_call.tool_call_id, options)
            decided = await _permitted(spec, args, self.shell, asks_as, request)
        except Paused:
            return WAITS
        if isinstance(decided, ToolOutcome) or decided.args != args:
            return chosen(options, "reject_once")
        return _let_through(self.runtime, name, options)


def _offers_a_choice(options: Sequence[Any]) -> bool:
    """Whether a permission request is the agent's question to its user rather than whether a
    call may run: it offers options of its own to pick from — a question's answers (Kimi's
    ``AskUserQuestion``), a plan's approaches and what else to do with it (its
    ``ExitPlanMode``: ``Revise``, ``Reject and Exit``) — so two of them are of one kind, where
    a call's request has one way each to allow it once, to allow it from now on and to refuse
    it."""
    kinds = [option.kind for option in options]
    return len(set(kinds)) < len(kinds)


async def _asked(
    runtime: Runtime, options: Sequence[Any], name: str, args: dict[str, Any], tool_call: Any
) -> Any:
    """The agent's question to its user (:func:`_offers_a_choice`), asked as the run's: one
    question, what the agent shows with it (a question's words, a plan) and every option it
    offers, each by its name; the answer chooses the option of that name. Asked on behalf of
    the agent's call (``name``, ``args``: ``Runtime.asking_within``), which a session resumed
    after the pause is told to make again. A pause propagates."""
    question = _text(tool_call.content) or CHOOSE
    async with runtime.asking_within(name, args):
        answer = await runtime.ask(question, options=[option.name for option in options])
    picked = {option.name: option.option_id for option in options}[answer]  # one it was asked
    return selected(picked)


async def _filled(runtime: Runtime, message: str, mode: Any) -> Any:
    """A form the agent asks its user to fill (ACP's ``elicitation/create``, its form
    ``mode``), asked as the run's question the way an MCP server's form is (``tools.elicit``:
    one field that picks from values a choice, else a form with the agent's schema), and the
    person's answer as the agent takes it. A pause propagates."""
    from acp.schema import CreateElicitationResponse
    from pydantic import TypeAdapter

    schema = mode.requested_schema.model_dump(by_alias=True, exclude_none=True)
    asking = elicited(SimpleNamespace(message=message, requested_schema=schema))
    answer = await runtime.ask(asking.question, **asking.asking)
    return TypeAdapter(CreateElicitationResponse).validate_python(asking.result(answer))


def _spec(runtime: Runtime, profile: Profile | None, name: str) -> ToolSpec:
    """A call of the agent's own, as governance decides it: the run's tool of that name (the
    tool catalog's word), else by its risk (its profile's built-ins; undeclared otherwise)."""
    side_effects = (profile.builtins if profile else {}).get(name)
    return runtime.native.get(name) or ToolSpec(name=name, side_effects=side_effects or UNDECLARED)


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


def _streamed(content: Any) -> dict[str, Any] | None:
    """A call's arguments as far as they streamed (its update's text), once they are a whole
    JSON object; ``None`` while they are not."""
    try:
        found = json.loads(_text(content))
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def _text(content: Any) -> str:
    """The text of a tool call's content (its ``content`` blocks' text)."""
    parts: list[str] = []
    for block in content or ():
        inner = getattr(block, "content", None)
        text = getattr(inner, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _argued(transcript: list[Said], n: int, update: Any) -> None:
    """The call ``update`` names, in the transcript's message ``n``, with the arguments it now
    gives (an agent may tell them after the call starts)."""
    said = transcript[n]
    calls = tuple(
        Called(c.id, c.name, _args(update.raw_input, None)) if c.id == update.tool_call_id else c
        for c in said.tool_calls
    )
    transcript[n] = Said(said.role, said.content, calls)


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
    "KIMI_ENV",
    "KIMI_ENV_WHY",
    "KIMI_OUTLIVES_THE_TURN",
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
