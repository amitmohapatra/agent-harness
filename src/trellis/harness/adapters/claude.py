"""Claude Agent SDK: the target is the ``ClaudeAgentOptions`` a team already configured.

* input: the prompt; the memory context goes with every prompt as Claude Code's own
  ``UserPromptSubmit`` hook adds context (``bridges.claude.with_context``), beside the options'
  hooks;
* run: ``query(prompt, options)`` on a copy of the options that also carries the harness
  tools as one in-process MCP server (``mcp__trellis__*``) beside the team's own servers — a
  ``mcp_servers`` given as a config file or JSON text is read, since the SDK serves an
  in-process server only from a dict;
* permissions: Claude Code's own, then a person when it asks. The SDK's ``PreToolUse`` hook,
  which the CLI runs before every tool call whatever its ``permission_mode`` or
  ``allowed_tools``, lets a harness tool through (the bridge is its check) and decides any
  other (Claude Code's built-ins: ``Bash``, ``Write``...; a team's own MCP server's) as a
  harness call is — the run's hooks, then governance by its risk (:data:`BUILTIN_SIDE_EFFECTS`)
  and the tool catalog's rules — and leaves the rest to Claude Code: its permission mode, its
  rules (``allowed_tools``, its settings), and when it would ask its user, ``can_use_tool``.
  That callback answers as the harness's: the team's own ``can_use_tool`` when it has one;
  else a call whose approval a person already answered in the hook runs (one call, one
  question), a call that only reads (a read tool, or a shell command
  ``governance.shell.read_only`` reads as read-only) runs, the shell runs (its sandbox, or the
  host the deployment allows it on: ``trellis.harness.host``; announced, never the user's to
  answer), the network runs as the tool catalog let it through in the hook
  (``bridge.outbound``: refused there with no word of an administrator's), and any other
  asks a person (``bridge.asked``) with the choices of its risk, as a harness call — what
  Claude Code would keep for its session ("don't ask again"), the run keeps: for this
  conversation, or always;
* records: every call Claude runs itself (its built-ins, its own MCP servers' tools, a subagent's
  calls) is recorded as the run's tool calls once it ran — its result or its error, and how long
  it took — through the SDK's own ``PostToolUse`` / ``PostToolUseFailure`` hooks
  (``tools.bridge.observed``); the harness's own tools are recorded by the bridge;
* stalls: the query is watched for a stall — a model step that sends nothing for
  ``runtime.STALL_SECONDS`` (the time a tool call runs, from ``PreToolUse`` to its
  ``PostToolUse``, not counted) ends the run with a ``ModelTimeout``;
* spending: each query's usage and cost (its ``ResultMessage``) count toward the run's budget;
* helpers: each subagent Claude runs (``agents=``, its ``Agent`` tool) is a child run of the run
  (``bridges.helpers``), from its call to that call's result; its own tool calls go through the
  same ``PreToolUse`` hook, and what it spends is in its query's ``ResultMessage``. A dynamic
  workflow (its ``Workflow`` tool) is one such child run, in the background: its agents'
  calls are its run's by their kind (``claude_live.WORKFLOW_AGENT``) while it is the one at
  work; while several are, a conversation's session gives them a run the workflows share
  (``claude_live.SHARED``), and a run with no conversation takes them as its own — marked
  with the agent either way;
* conversations: a conversation's turn leaves its CLI process running, and the next turn is
  sent into it (``bridges.claude_live``: the SDK's ``ClaudeSDKClient``), so what Claude started
  in the background — a subagent with ``background``, a shell command — goes on between the
  turns; a turn ends at the result of its own prompt, and a turn the session injects (a
  finished task's notification) is never its answer. A subagent the CLI runs in the
  background keeps its child run open until its task ends, its calls decided and recorded
  for that run (``claude_live.Background``). A run with no conversation, a sub-agent's
  run, and a turn with no live session (another server, a restart, one closed) run as below:
  one process for the query, resuming the conversation's session;
* pause: an approval or ``trellis.current().ask`` stops consuming the query (the CLI process
  ends, once read to its turn's result, which the run is charged: the model's billed reply
  to the pause's placeholder is in it — a conversation's live one stays, its turn ended, and
  the resumed attempt is sent into it). The session the CLI kept
  (``ResultMessage.session_id``) is in the run's journal, and the next attempt resumes it
  (``resume=``): Claude goes on from where it was — its built-in tools are not run again —
  and, told which call a person answered about, calls that tool again, which the journal
  answers (a call a question was asked in runs as it was, whatever words Claude makes it
  again with: ``Replay.made``). The session is kept in the run's journal too, as the SDK's
  ``SessionStore`` (``bridges.claude.JournalSessions``): the SDK writes it back for the CLI on
  any machine. A session the CLI cannot continue (options keeping their own store, a transcript
  it refuses) is a warning, and the query runs again from its prompt against the journal;
* models: with a route that has an Anthropic route (``Capability.ANTHROPIC_ROUTE``: Bifrost)
  serving the harness, the CLI's model calls go through the gateway's Anthropic route with the
  virtual key (:func:`routed_env`; the options' own ``ANTHROPIC_BASE_URL`` wins), carrying
  the deny-all MCP scope, the run's dimensions and its ``traceparent`` as
  ``ANTHROPIC_CUSTOM_HEADERS`` beside the options' own — the CLI reads them as it starts, once
  per attempt; a conversation's live session, which serves many runs, carries the dimensions
  that hold for all of them (``telemetry.lasting_headers``: no run, no ``traceparent``);
* framework options: ``ClaudeAgentOptions`` fields, set on the run's copy of the options
  before the harness's own changes. Merged with them: ``hooks`` (the governance and context
  hooks go after theirs), ``can_use_tool`` (asked when Claude Code would ask, after the hooks
  and governance) and ``mcp_servers`` (the harness's ``trellis`` server goes beside them; a
  server of that name is refused). Refused: the fields that choose the session a query
  continues, and the permission prompt tool (:data:`OWNED`).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
    stable_id,
)
from trellis.harness import budget, children, skills
from trellis.harness.adapters.base import (
    Extracted,
    HistoryIn,
    Invocation,
    Keeps,
    Narrowing,
    Output,
    ToolFormat,
    builtin,
    helper,
    unique,
)
from trellis.harness.bounded import LRU, SAID_ONCE, LRUSet
from trellis.harness.bridges import live as shared
from trellis.harness.bridges.claude import (
    JournalSessions,
    SessionCost,
    TaskEnded,
    TaskStarted,
    after_theirs,
    caller_sent,
    deciding_matcher,
    harness_matcher,
    session_store,
    task_said,
    with_context,
)
from trellis.harness.bridges.claude_live import (
    CODE,
    GONE,
    GONE_SAID,
    WORKFLOW_AGENT,
    WORKFLOW_TOOL,
    Background,
    LiveSession,
    opened,
)
from trellis.harness.bridges.helpers import STOPPED, Helper, Helpers, Kind, helping, naming
from trellis.harness.bridges.live import LiveSessions, Turn
from trellis.harness.bridges.resumed import RESUMED, resumed_call
from trellis.harness.gateways import NO_KEY, Capability, OpenAICompatible
from trellis.harness.governance.shell import SHELLS, Shell, moves
from trellis.harness.history import Called, Said, Speaker
from trellis.harness.host import CLAUDE_SHELLS
from trellis.harness.journal import Pending, approvals_read, call_key
from trellis.harness.prompts import CUSTOM_HEADERS, NO_GATEWAY_TOOLS
from trellis.harness.runtime import Paused, Runtime, current
from trellis.harness.telemetry import gateway_headers, lasting_headers
from trellis.harness.tools import bridge
from trellis.harness.tools.ask import ask_question
from trellis.harness.tools.base import UNDECLARED, SideEffects
from trellis.harness.tools.mcp import data_of_listed, side_effects_of

log = logging.getLogger("trellis.harness")

if TYPE_CHECKING:
    from claude_agent_sdk.types import PreToolUseHookSpecificOutput, SyncHookJSONOutput

#: Claude Code's built-in tools — every one the bundled CLI (2.1.325) has, those its ``init``
#: message lists and those it offers where it is set up for them — and what each does as
#: governance sees it (the tool catalog's word overrides it). Reads run: looking things up,
#: listing, loading a deferred tool (``ToolSearch``), waiting for MCP servers to connect
#: (``WaitForMcpServers``), code intelligence (``LSP``), a card shown to the user (a
#: suggestion, ``OfferChromeSetup``, ``propose_skills``: nothing is written), the tools Claude
#: keeps its own work with (its task list, plan mode, a structured answer, a subagent's
#: hand-back), and a
#: subagent's or a workflow's call (``Agent``, ``Task``, ``Workflow``), which only hands work
#: on — its agents' calls are governed one by one. Writes run and are announced, Claude Code's
#: own permission check asking first where it would: a file changed, a shell (``Bash``,
#: ``PowerShell``, ``REPL``, ``Monitor``'s command), a worktree entered or left, a task
#: stopped, a page published, a notification, a file or feedback sent, the session ended, a
#: claude.ai design or project changed (``ClaudeDesign``, ``Projects``), a goal that keeps
#: the session working (``ProposeGoal``).
#: A team that wants every shell command approved gives ``Bash`` the risk ``irreversible``,
#: or an ``approve_when``, in the tool catalog. ``WebFetch`` and ``WebSearch`` are listed here
#: but decided by the catalog alone (:data:`CLAUDE_NETWORK`). What Claude Code asks its user
#: about although it only reads — leaving plan mode with a plan, a skill (:data:`CLAUDE_ASKS`)
#: — a person answers. Any other tool says nothing of what it does: a person approves its
#: calls (its risk in the catalog says more). A person asked about a call Claude runs picks
#: as for a harness tool's — once, for this conversation, always (with a memory service), deny;
#: once or deny for an irreversible one: Claude Code's own "don't ask again", kept by the run.
BUILTIN_SIDE_EFFECTS: Final[dict[str, SideEffects]] = {
    **dict.fromkeys(
        (
            "Read",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "Agent",
            "Task",
            "Workflow",
            "Skill",
            "ExitPlanMode",
            "EnterPlanMode",
            "TaskCreate",
            "TaskGet",
            "TaskList",
            "TaskUpdate",
            "StructuredOutput",
            "AskUserQuestion",
            "ListMcpResourcesTool",
            "ReadMcpResourceTool",
            "ReadMcpResourceDirTool",
            "ListConnectors",
            "ReadNotifications",
            "ReportFindings",
            "SearchMcpRegistry",
            "SuggestConnectors",
            "ListPlugins",
            "ListSkills",
            "SearchPlugins",
            "SearchSkills",
            "SuggestPluginInstall",
            "SuggestSkills",
            "ShowOnboardingRolePicker",
            "OfferChromeSetup",
            "propose_skills",
            "LSP",
            "ToolSearch",
            "WaitForMcpServers",
            "SubagentHandback",
            "SendUserMessage",
        ),
        "read",
    ),
    **dict.fromkeys(
        (
            "Write",
            "Edit",
            "MultiEdit",
            "NotebookEdit",
            "TaskStop",
            "Bash",
            "PowerShell",
            "REPL",
            "Monitor",
            "EnterWorktree",
            "ExitWorktree",
            "Artifact",
            "PushNotification",
            "SendFeedback",
            "SendUserFile",
            "ShareOnboardingGuide",
            "EndConversation",
            "ClaudeDesign",
            "Projects",
            "ProposeGoal",
        ),
        "write",
    ),
}
#: The tools Claude Code asks its user about although they only read: leaving plan mode with a
#: plan (``ExitPlanMode``), a skill (``Skill``). A
#: person answers when it asks, as for a write — unless the tool catalog has a rule for the
#: tool that lets it run (``tools.bridge.asked``).
CLAUDE_ASKS: Final = frozenset({"ExitPlanMode", "Skill"})
#: Claude Code's plan mode (``permission_mode="plan"``: the options', or a run's
#: ``framework_options``), held by the harness's own hook: the hook's word on a harness tool
#: is final for the CLI, so the CLI's plan mode alone would let a write of the harness's run.
#: Until Claude presents its plan (:data:`PLAN_TOOL`) and it runs — a person approved it, as
#: Claude Code asks — a call runs only if it reads (governance rates it a read, as
#: ``plan_mode.PlanMode`` does for ``ReAct``; a command ``governance.shell`` reads as
#: read-only) or writes Claude's plan in its own plans folder (:data:`PLANS`, in the run's
#: config folder); any other is refused with :data:`PLAN_WAITS` and recorded so. Approved,
#: the plan is in the run's journal (``Journal.plan_approved``): every attempt after governs
#: its calls as ever.
PLAN_MODE: Final = "plan"
PLAN_TOOL: Final = "ExitPlanMode"
PLANS: Final = "plans"
PLAN_WAITS: Final = (
    f"plan mode is on: present your plan with {PLAN_TOOL} and wait for its approval; until "
    "then only a call that reads runs"
)
#: Claude Code's built-in tools that reach the network outside the deployment: the tool
#: catalog decides them (``tools.bridge.outbound``: refused until an administrator gives them a
#: risk or an ``approve_when``), never a question to the run's user; published with no risk of
#: the harness's, so the catalog's is an administrator's.
CLAUDE_NETWORK: Final = frozenset({"WebFetch", "WebSearch"})
#: Claude Code's question to its user, asked as the run's (:func:`_answered`), and the mark it
#: ends a recommended option's label with.
QUESTION_TOOL: Final = "AskUserQuestion"
RECOMMENDED: Final = "(Recommended)"
#: What Claude Code is called when a person is asked about a call it asks about.
CLAUDE_CODE: Final = "Claude Code"
#: Claude Code's built-in tools whose work needs the CLI to go on running after the query: a
#: scheduled prompt (``CronCreate``, durable or not, fires only in a session that is running
#: then) and a wake-up (``ScheduleWakeup``). A query's own process ends with it; a
#: conversation's live session (``bridges.claude_live``) outlives the turn, but only until it
#: is idle, and the turn a schedule fires then has no run: its tool calls are refused and its
#: answer reaches nobody. So they stay disallowed (with the tools that list and delete the
#: schedules): offered, Claude promised a schedule that nothing keeps.
OUTLIVES_THE_QUERY: Final = (
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "RemoteTrigger",  # a claude.ai scheduled task: it fires with no run either
)
#: Claude Code's skills slot: the tool a skill source's plugin (``skills.for_run``) and its own
#: skill folders reach the model through. The tool hints never narrow it away, as the harness's
#: own ``list_skills`` and ``load_skill`` stay offered (``skills.told``): a run keeps its skills.
SKILLS_SLOT: Final = "Skill"
#: Claude Code's built-in tools that start agents no run of the harness can hold, disallowed
#: too. ``SendMessage`` resumes a finished background subagent as a new task, whose messages
#: carry the call that first started it (the bundled CLI, 2.1.283, driven against a scripted
#: model), or reaches another Claude session on the machine or in the cloud. The agent-teams
#: tools older CLIs have behind ``CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`` (``TeamCreate``,
#: ``TeamDelete``; 2.1.283 has none, its named agents are background subagents) are left out
#: with it, whatever the team's own environment says, and ``ListAgents`` — the agents
#: ``SendMessage`` reaches: the other Claude sessions of the worker's user, other tenants' runs
#: among them.
UNHELD_AGENTS: Final = ("SendMessage", "ListAgents", "TeamCreate", "TeamDelete")
#: The name of a dynamic workflow's child run (its script names it, which the call does not).
WORKFLOW: Final = "workflow"
#: What the record of a workflow agent's call keeps of who made it (its hook's input).
AGENT_KEYS: Final = ("agent_id", "agent_type")
#: Claude Code's own limit on an MCP tool's result (``MAX_MCP_OUTPUT_TOKENS``, 25,000 tokens by
#: default: about 50,000 characters of a CSV), raised so the harness keeps a large result first
#: (over ``results.MAX_CHARS``, 80,000 characters, read with ``read_file`` and ``query_result``):
#: a result the CLI saves to its own session's file is one neither reads, nor the journal holds
#: for a resume on another worker. A team's own limit (the options' ``env``, the process's) stays.
MCP_OUTPUT_ENV: Final = "MAX_MCP_OUTPUT_TOKENS"
MCP_OUTPUT_TOKENS: Final = 60_000
#: The most MCP tools a Claude agent was seen calling that join its tool set (the most recent).
SEEN_KEPT: Final = 1024
#: The most one message from the CLI may be (the SDK's ``max_buffer_size``: 1 MiB unless the
#: options say): a tool's result comes back from the CLI in one, and a run keeps results as large
#: as an agent-runs artifact (50 MiB: a journal over a checkpoint's size travels as one) —
#: escaped as JSON, twice that at most.
CLI_MESSAGE_BYTES: Final = 2 * 50 * 1024 * 1024
#: The hook event every tool call is governed at, and the ones that report it ran or failed.
#: Bifrost's route for Anthropic's API, under its root.
ANTHROPIC_ROUTE: Final = "anthropic"
TOOL_EVENT: Final = "PreToolUse"
RAN_EVENT: Final = "PostToolUse"
FAILED_EVENT: Final = "PostToolUseFailure"
#: How a message steered into the run reads to Claude, after a tool call's result.
STEERED_SAYS: Final = "The user adds, while you work: "
#: The key a query's model steps are watched for a stall under (``Runtime.model_started``).
STEP: Final = "claude_agent_sdk"
#: How long a paused query's own process is read for its turn's result, the model's billed
#: reply to the pause's placeholder in it (``_settled``), in seconds.
SETTLE_SECONDS: Final = 30.0
#: The result a query that reached its ``max_turns`` ends with (no answer).
MAX_TURNS: Final = "error_max_turns"
#: The ``stop_reason`` of a query's result whose last reply reached the model's output limit
#: (Claude Code first continues such a reply itself).
MAX_TOKENS: Final = "max_tokens"
#: When the SDK hands the session store a step: as soon as the CLI mirrors it.
SESSION_FLUSH: Final = "eager"
#: Claude Code's tool that runs a subagent (``agents=``): ``Agent``, named ``Task`` before.
SUBAGENT_TOOLS: Final = frozenset({"Agent", "Task"})
#: The subagent an ``Agent`` call runs when it names none.
GENERAL_PURPOSE: Final = "general-purpose"
#: Why a subagent whose call failed without saying why ended.
SUBAGENT_FAILED: Final = "the subagent's call failed"
#: The statuses a task ends with that say how (the SDK's ``TaskNotificationStatus`` and
#: ``TaskUpdatedStatus``): it completed, it failed; any other terminal one, it was stopped.
TASK_COMPLETED: Final = "completed"
TASK_FAILED: Final = "failed"
TASK_STOPPED: Final = "stopped"
#: Claude Code's config folder (its settings, sessions, auto memory: ``~/.claude`` unless this
#: says otherwise) and the switch that turns its auto memory off. Each tenant and user of a
#: worker has a folder of their own (:func:`config_folder`), so no run reads another's state or
#: the worker's own, and auto memory — which the memory service does not see — is off.
CONFIG_ENV: Final = "CLAUDE_CONFIG_DIR"
AUTO_MEMORY_ENV: Final = "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
#: Claude Code's setting for what a session does with a message another Claude session of the
#: same OS user sends it, and the value that refuses them: every run on a worker is that one
#: user's, other tenants' among them.
INBOUND: Final = {"crossSessionInbound": "refuse"}
#: What Claude Code says on its stream beside the model's replies that explains a slow, a
#: shortened or a rerouted turn — the system frame's ``subtype`` — and the code of the run's
#: warning that says it: a model call tried again, the session's context compacted, another
#: model answering, a step the session store did not take.
NOTICES: Final = {
    "api_retry": "model_retry",
    "compact_boundary": "compaction",
    "model_fallback": "model_fallback",
    "mirror_error": "session_store",
}
#: The ``ClaudeAgentOptions`` fields the harness owns: framework options cannot set them.
OWNED: Final[dict[str, str]] = {
    "resume": "the harness resumes the run's own session after a pause",
    "continue_conversation": "the harness resumes the run's own session after a pause",
    "permission_prompt_tool_name": "the harness decides Claude's tool permissions with "
    "can_use_tool (give your own check as can_use_tool)",
}
#: Why a conversation's live session is closed before a query that runs in a process of its
#: own (two processes do not hold one session).
OWN_PROCESS: Final = "this query runs in a process of its own"
#: What a call the harness's ``PreToolUse`` hook could not decide is refused with: the CLI
#: would let a call whose hook failed through to its own permission check.
UNDECIDED: Final = "The harness could not decide this call, so it did not run ({why})."


@dataclasses.dataclass(frozen=True, slots=True)
class ClaudeInput:
    prompt: str
    context: str | None
    #: the tool call a person answered about (``Pending.call``), when the run resumes from
    #: that pause
    answered: ToolCall | None = None
    #: the person denied that call (``REJECT``): it does not run
    denied: bool = False
    #: the person answered a question that call asked while it ran (not its approval)
    asked: bool = False
    #: the run paused before its prompt was sent — on the question of a background
    #: subagent's run of its conversation (``live.Keeper.asks``): it is sent now
    unsent: bool = False
    #: that question's answer — the subagent's run, its interrupt, the person's resolution —
    #: for when its session is not on this server (``live.gone``)
    background_answer: tuple[str, str, InterruptResolution] | None = None


class ClaudeRunError(RuntimeError):
    """The CLI reported the run as failed."""


class ClaudeAdapter(Keeps):
    name: ClassVar[str] = "claude_agent_sdk"
    tool_format: ToolFormat = "claude"
    fixed_tools: bool = False
    narrows: Narrowing = "run"
    #: a query takes one prompt, not a conversation: memory's window is in the context's text
    replays_history = False
    #: with memory off, the conversation's earlier turns are in the context's text too
    history_in: HistoryIn = "context"
    #: its session (the journal's) is continued by the conversation's next turn
    sessions = True

    def __init__(self) -> None:
        #: the tools the options declare (:meth:`native`), once read
        self._static: list[ToolSpec] | None = None
        #: the MCP tools the agent was seen calling, the most recent kept
        self.seen: LRUSet[str] = LRUSet(maxsize=SEEN_KEPT)
        #: each conversation's CLI process, kept between its turns (``bridges.claude_live``)
        self.live = LiveSessions()
        #: what each session cost so far, as its last result charged here carried it (the
        #: most recent): a process that resumes one reports the session's cost, not its own
        self.costs: LRU[str, SessionCost] = LRU()

    async def aclose(self) -> None:
        """The agent is closing: every conversation's live session ends."""
        await self.live.aclose()

    async def child_cancelled(self, run_id: str, reason: str | None) -> None:
        """The run ``run_id`` ended with the run that started it: when it is a background
        subagent's a live session keeps here, it stops (``LiveSessions.cancel``)."""
        self.live.cancel(run_id, reason or children.PARENT_ENDED)

    def kept_children(self, run_id: str) -> list[str]:
        return self.live.kept_for(run_id)

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        return None

    def steers(self, target: Any) -> bool:
        """Every query: a steered message is added after a tool call's result (a
        ``PostToolUse`` hook's ``additionalContext``, :func:`_with_governance`); a query that
        calls no tool takes none."""
        return True

    def asks(self, target: Any) -> bool:
        """Claude Code's own :data:`QUESTION_TOOL`, asked as the run's (:func:`_answered`)."""
        return True

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        prompt = input if isinstance(input, str) else json.dumps(input, default=str)
        return ClaudeInput(prompt=prompt, context=context)

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        return [m async for m in self._helped(target, native_input, run)]

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from claude_agent_sdk import AssistantMessage, StreamEvent, TextBlock

        messages: list[Any] = []
        streamed = False  # the message's text came as deltas (``include_partial_messages``)
        async for message in self._helped(target, native_input, run):
            messages.append(message)
            # the main agent's text: a subagent's (forwarded naming its ``Agent`` call) is
            # its own run's, as in the answer and the transcript
            if isinstance(message, StreamEvent) and message.parent_tool_use_id is None:
                if (said := _text_delta(message.event)) is not None:
                    streamed = True
                    yield said
            elif isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
                if not streamed:
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            yield block.text
                streamed = False
        yield Output(messages)

    def extract(self, target: Any, output: Any) -> Extracted:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        transcript = _transcript(output)
        answer: Any = None
        said: str | None = None  # what Claude said in the turn so far
        for message in output:
            if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
                text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
                said = text or said
            # the result of the turn's own prompt: one of a turn the session injected (a
            # finished background task's notification) is not the user's answer
            if isinstance(message, ResultMessage) and caller_sent(message):
                if message.subtype == MAX_TURNS:  # what it said before the limit: PARTIAL
                    answer = said
                elif message.is_error:
                    raise ClaudeRunError("; ".join(message.errors or []) or message.subtype)
                else:
                    answer = (
                        message.structured_output
                        if message.structured_output is not None
                        else message.result
                    )
            if isinstance(message, ResultMessage):
                said = None
        return Extracted(answer=answer, transcript=transcript)

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
        return dataclasses.replace(
            native_input,
            answered=pending.call,
            denied=resolution.decision is InterruptDecision.REJECT,
            asked=pending.interrupt.tool_call is None,
        )

    def check_options(self, options: Mapping[str, Any]) -> None:
        from claude_agent_sdk import ClaudeAgentOptions

        from trellis.harness.tools.convert.claude import SERVER

        fields = [f.name for f in dataclasses.fields(ClaudeAgentOptions)]
        unknown = [key for key in options if key not in fields]
        if unknown:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, unknown))}: no ClaudeAgentOptions field"
            )
        owned = [f"{key!r} ({OWNED[key]})" for key in options if key in OWNED]
        if owned:
            raise ConfigurationError(f"framework_options {'; '.join(owned)}")
        servers = options.get("mcp_servers")
        if isinstance(servers, Mapping) and SERVER in servers:
            raise ConfigurationError(
                f"framework_options' mcp_servers names {SERVER!r}: the harness's own server "
                "(its tools) has that name"
            )

    def workdir(self, target: Any, options: Mapping[str, Any]) -> Path | None:
        """Where the CLI works: the options' ``cwd`` (the run's own over the agent's), else
        this process's working directory."""
        return Path(options.get("cwd") or target.cwd or os.getcwd())

    async def native(self, target: Any) -> list[ToolSpec]:
        """Claude Code's built-in tools the options leave it (``tools``, else the ones
        :data:`BUILTIN_SIDE_EFFECTS` knows, less ``disallowed_tools``) and the MCP tools of its
        own servers: those an in-process server (``create_sdk_mcp_server``) lists, those
        ``allowed_tools`` names (``mcp__<server>__<tool>``), and those the agent was seen
        calling (its ``PostToolUse``: another server's tools are listed only by the CLI)."""
        if self._static is None:
            self._static = await _declared(target)
        return unique([*self._static, *(native_spec(name) for name in self.seen)])

    def observed(self, name: str) -> None:
        """A tool Claude ran itself (``PostToolUse``): an MCP server's joins the tool set."""
        if name.startswith("mcp__"):
            self.seen.add(name)

    # ------------------------------------------------------------------ internals
    async def _helped(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The query's messages (:meth:`_messages`), each subagent Claude runs a helper of the
        run (``bridges.helpers``) from its ``Agent`` call to that call's result."""
        async with helping(run.runtime, self.name) as helpers:
            # a background subagent's run has no query of its own: its task works in a session
            found = cast("Background | None", self.live.attended.get(run.runtime.run_id))
            if found is not None:  # the calls its subagent makes are this run's own
                helpers.attribute(found.task_id, None)
            source = (
                self._messages(target, native_input, run)
                if found is None
                else self._attended(target, run, found)
            )
            async with contextlib.aclosing(source) as messages:
                async for message in messages:
                    await _subagents(helpers, message)
                    yield message

    async def _attended(
        self, target: Any, run: Invocation, found: Background
    ) -> AsyncGenerator[Any, None]:
        """One attempt of a background subagent's run (``claude_live.Background``): the
        subagent's messages while its task works in its conversation's live session, then
        the result made of the task's end. The attempt's own hooks, permission check and
        harness tools — made as a query's are — decide and record the calls the CLI makes
        for the subagent, and its steps are watched for a stall as a query's."""
        runtime = run.runtime
        target = dataclasses.replace(target, **found.given) if found.given else target
        steps = Steps(runtime, self)
        options = _options(target, None, run, None, steps, kept=True)
        steps.waiting()
        try:
            async with contextlib.aclosing(found.served(_turn(run, options))) as messages:
                async for message in messages:
                    steps.waiting()
                    yield message
        finally:
            steps.closed = True
            runtime.model_ended(STEP)

    async def _messages(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The query's messages (:meth:`_attempted`). A query that reached its turn limit
        ends with Claude Code's own ``error_max_turns`` result, and Claude Code leaves it
        there: the run ends ``PARTIAL`` with what Claude said before it (``extract``), its
        limit as the budget it spent (``Runtime.budget_spent``) and a ``step_limit`` warning
        — no query more (a paid last answer), and a live session and the background work in
        it go on."""
        from claude_agent_sdk import ResultError, ResultMessage

        runtime = run.runtime
        limited = False
        try:
            async with contextlib.aclosing(self._attempted(target, native_input, run)) as messages:
                async for message in messages:
                    if isinstance(message, ResultMessage) and message.subtype == MAX_TURNS:
                        limited = True
                        limit = runtime.framework_options.get("max_turns") or target.max_turns
                        stopped = f"the turn limit ({limit}) was reached"
                        log.warning("run %s stopped: %s", runtime.run_id, stopped)
                        runtime.events.warning("step_limit", f"stopped: {stopped}")
                        runtime.budget_spent = stopped
                    elif isinstance(message, ResultMessage):
                        # past Claude Code's own recovery: what a run with no answer says
                        runtime.cut_off = message.stop_reason == MAX_TOKENS
                    _noticed(runtime, message)
                    yield message
        except ResultError:
            # the CLI exits with an error after its error result (the SDK raises it then): at
            # the turn limit that is the end the result already said
            if not limited:
                raise

    async def _attempted(
        self, target: Any, native_input: Any, run: Invocation
    ) -> AsyncGenerator[Any, None]:
        """The query's messages — the session an earlier attempt kept resumed, when there is
        one the CLI still holds; else from the prompt."""
        from claude_agent_sdk import AssistantMessage, ResultMessage

        runtime = run.runtime
        journal = runtime.replay.journal
        resumed, answered = journal.session, False
        # closed as it is left (``aclosing``): its CLI process ends here, in this loop
        async with contextlib.aclosing(self._query(target, native_input, run, resumed)) as messages:
            async for message in messages:
                if (
                    resumed is not None
                    and not answered
                    and isinstance(message, ResultMessage)
                    and message.is_error
                ):
                    # the CLI could not continue the session: the query again, from its prompt
                    why = "; ".join(message.errors or []) or message.subtype
                    runtime.events.warning(
                        "claude_session", f"session {resumed} was not resumed ({why}): run again"
                    )
                    journal.session = None
                    break
                answered = answered or isinstance(message, AssistantMessage)
                yield message
            else:
                return
        async with contextlib.aclosing(self._query(target, native_input, run, None)) as again:
            async for message in again:
                yield message

    async def _query(
        self,
        target: Any,
        native_input: Any,
        run: Invocation,
        session: str | None,
    ) -> AsyncGenerator[Any, None]:
        """One query (``session``: the one it resumes); the session it runs in kept in the
        journal as it goes."""
        from claude_agent_sdk import AssistantMessage, ResultMessage

        runtime = run.runtime
        given = runtime.framework_options
        target = dataclasses.replace(target, **given) if given else target
        # a skill source's plugin written for this run (its pinned skills), removed after it
        plugins, written = await skills.for_run(target.plugins)
        target = dataclasses.replace(target, plugins=plugins) if written else target
        steps = Steps(runtime, self)
        # a conversation's turn is served by its live session (``bridges.claude_live``); a
        # run with no conversation of its own, a
        # sub-agent's, and options whose parts are a process's alone (a plugin written for
        # the run; an in-process MCP server of the team's, whose tools run in the context of
        # the process's first turn; a budget of the options' own, which the CLI counts for a
        # process) run in a process of their own, as ever
        key = runtime.conversation if shared.converses(runtime) else None
        # a session another worker's live session still holds (a background subagent of the
        # run at work there) is forked: the turn goes on here, in a process of its own
        held = session is not None and await shared.held_elsewhere(runtime, self.live)
        kept = key is not None and not written and not _own_process(target) and not held
        context = _dated(native_input.context, run, kept=kept)
        options = _options(target, context, run, session, steps, kept=kept)
        options = dataclasses.replace(options, fork_session=options.fork_session or held)
        # a first attempt resumes only the conversation's session (``Runtime.continued``):
        # the turn's own prompt; a later one, the run's own session after a pause
        first = session is None or runtime.attempt == 1 or native_input.unsent
        prompt = native_input.prompt if first else _resumed(native_input, run)
        steps.waiting()
        spending = Spending(runtime, options, self.costs, None if held else session)
        live: LiveSession | None = None
        turn = _turn(run, options)
        await self._gone(runtime, native_input)
        try:
            if kept:
                assert key is not None
                live = await self._live(key, options, run, session, spending, unsent=first)
            elif key is not None:
                await self.live.close(key, OWN_PROCESS)
                shared.said(self.live, runtime, CODE)
            # closed as it is left (``aclosing``): a process of its own ends here, in this
            # loop; a live session's turn is interrupted when left before its result
            source = _queried(prompt, options) if live is None else live.served(turn, prompt)
            async with contextlib.aclosing(source) as messages:
                async for message in messages:
                    runtime.raise_stopped()  # a call the process's end stopped ends it here
                    steps.waiting()  # the next message is the next step's
                    found = getattr(message, "session_id", None)
                    if isinstance(found, str):
                        runtime.replay.journal.session = found
                    if isinstance(message, ResultMessage):
                        spending.ended(message, live)
                    elif isinstance(message, AssistantMessage):
                        spending.replied(message)
                    await _provided(steps, message)
                    yield message
                    if runtime.pending is not None:
                        # the run paused: stop here; a process of its own goes with it, once
                        # its turn's result says what the attempt was billed
                        if live is None:
                            await _settled(messages, steps, spending)
                        return
        finally:
            steps.closed = True
            runtime.model_ended(STEP)
            skills.removed(written)
            await spending.closed(live, turn)

    async def _gone(self, runtime: Runtime, native_input: Any) -> None:
        """The answer this turn brings to a background subagent's question
        (``background_answer``), when no live session here holds that subagent (a killed
        server's): nothing continues it, so its run records the answer and ends
        (``live.gone``)."""
        answered = native_input.background_answer
        if answered is not None and answered[0] not in self.live.attended:
            await shared.gone(runtime, *answered, why=GONE, said=GONE_SAID, code=CODE)

    async def _live(
        self,
        key: str,
        options: Any,
        run: Invocation,
        session: str | None,
        spending: Spending,
        *,
        unsent: bool,
    ) -> LiveSession | None:
        """The conversation's live session for this query: the one kept, when it was started
        with these options, holds the session the query continues and owes no earlier turn's
        result; else a new one, started as a query's own process is (``resume=``). ``None``:
        there is no room for one (``live.KEPT``). ``unsent``: the turn's own prompt is still
        to send — what the session's background runs wait to ask a person is asked first, in
        this run (``live.Keeper.asks``)."""
        runtime = run.runtime
        shape = _shape(options)
        found = cast("LiveSession | None", self.live.get(key))
        if found is not None:
            why = None
            if found.shape != shape:
                why = "this turn's options are not the ones it was started with"
            elif found.session_id != session:
                why = "this turn does not continue the session it holds"
            elif not await found.settled():
                why = "an earlier turn's result never came"
            if why is not None:
                await self.live.close(key, why)
                found = None
        if found is None:
            # the places are the server process's: every agent's sessions count (``KEPT``)
            found = await opened(
                self.live,
                key,
                shape,
                options,
                runtime.replay.journal.transcripts,
                owner=(runtime.tenant, runtime.user),
                among=shared.among(runtime, self),
            )
            if found is not None:
                # a process that resumes a session reports the session's cost: what its
                # first turn is charged counts from what the session cost before (the
                # journal's, on any worker)
                found.cost = spending.cost
        else:  # the session's running cost is the one its process has been charged from
            spending.cost = found.cost
        shared.said(self.live, runtime, CODE)
        if found is not None and unsent:
            await found.asks(runtime)
        return found


async def _settled(messages: AsyncIterator[Any], steps: Steps, spending: Spending) -> None:
    """The rest of a paused query's own process, to its turn's result, which the run is
    charged (``Spending.ended``): Claude Code sends the pause's placeholder to the model as the
    paused call's result, and the model's reply to it is billed — 51-83% of a paused attempt's
    cost with the bundled CLI on Haiku — while only a result says what the attempt cost, and
    the reply's final token count. Nothing in it is the run's: its messages are not the run's
    steps, and a call made in it is refused, ending the turn (``Steps.settling``). The CLI is
    waited on no longer than :data:`SETTLE_SECONDS` (then what its replies said is charged)."""
    from claude_agent_sdk import AssistantMessage, ResultMessage

    steps.settling = True
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(SETTLE_SECONDS):
            async for message in messages:
                if isinstance(message, ResultMessage):
                    spending.ended(message, None)
                    return
                if isinstance(message, AssistantMessage):
                    spending.replied(message)


def _dated(context: str | None, run: Invocation, *, kept: bool) -> str | None:
    """The context a turn's prompt is given: memory's — and, a turn a live session may serve
    that is offered ``schedule_task``, the day and hour it is (``schedules.now_said``): the
    process keeps the tool's wording from its first turn (the hour then), and Claude Code
    says the date once a session, again only when the day changes."""
    from trellis.harness.schedules import SCHEDULE_TOOLS, now_said

    if not kept or not any(t.name == SCHEDULE_TOOLS[0] for t in run.tools):
        return context
    now = now_said(run.runtime)
    return f"{context}\n\n{now}" if context else now


def _own_process(options: Any) -> bool:
    """Whether the options need a process for every query: they carry an in-process MCP
    server of the team's own (its tools run in the context of the process's first turn), or
    a budget of their own (``max_budget_usd`` caps what a process spends, and the SDK cannot
    give a running one its cap back: the cap a team set for a query stays a query's)."""
    from trellis.harness.tools.convert.claude import SERVER

    return options.max_budget_usd is not None or any(
        name != SERVER and isinstance(config, Mapping) and config.get("type") == "sdk"
        for name, config in configured_servers(options.mcp_servers).items()
    )


def _turn(run: Invocation, options: Any) -> Turn:
    """The query as a live session serves it: its own options, and this context."""
    from trellis.harness.tools.convert.claude import SERVER

    servers = options.mcp_servers if isinstance(options.mcp_servers, dict) else {}
    return Turn(run.runtime, options, servers.get(SERVER), contextvars.copy_context())


def _shape(options: Any) -> tuple[Any, ...]:
    """What a process started with ``options`` has for good, and a later turn must bring the
    same to be served by it: the options less what each turn brings of its own (its hooks'
    and its permission check's code, its harness tools' code, its journal, the session it
    resumes), the hooks' places, and the tools its ``trellis``
    server lists with their schemas (not their wording of the moment: ``convert.listing``)."""
    from trellis.harness.tools.convert import claude as convert

    servers = options.mcp_servers
    store = options.session_store
    journaled = isinstance(store, JournalSessions)
    fixed = dataclasses.replace(
        options,
        hooks=None,
        can_use_tool=None,
        resume=None,
        session_store=None if journaled else store,
        mcp_servers={n: c for n, c in servers.items() if n != convert.SERVER}
        if isinstance(servers, dict)
        else servers,
    )
    hooks = tuple(
        (event, tuple((m.matcher, len(m.hooks), m.timeout) for m in matchers))
        for event, matchers in sorted((options.hooks or {}).items())
    )
    own = servers.get(convert.SERVER) if isinstance(servers, dict) else None
    return (fixed, hooks, None if own is None else convert.listing(own), journaled)


def _queried(prompt: Any, options: Any) -> AsyncGenerator[Any, None]:
    """The query's messages: what ``claude_agent_sdk.query`` runs, for its caller to close.
    ``query`` iterates it with a bare ``async for``, so a caller that leaves early (a paused
    run) leaves it to be collected — after the attempt's task, where its cleanup (the CLI
    process's end) can no longer run, and the process outlives the run."""
    from claude_agent_sdk._internal.client import InternalClient

    messages = InternalClient().process_query(prompt=prompt, options=options)
    return cast("AsyncGenerator[Any, None]", messages)


async def _briefed(
    runtime: Runtime, call_id: str, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """A subagent's call (:data:`SUBAGENT_TOOLS`) as it runs: started as a helper of the run
    (``bridges.helpers``), its ``prompt`` given the memory context for its task before it."""
    helpers = runtime.helpers
    assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
    task = str(args.get("prompt") or args.get("description") or "")
    kind, called = Kind.SUBAGENT, str(args.get("subagent_type") or GENERAL_PURPOSE)
    helper = await helpers.start(call_id, kind=kind, name=called, task=task)
    return {**args, "prompt": helpers.briefed(helper, task)} if "prompt" in args else args


async def _working(helpers: Helpers, given: Mapping[str, Any]) -> Helper | None:
    """The helper a call is the work of, from its hook's input (``given``): a subagent's, by
    its task's id (``agent_id``: ``Helpers.of``); a dynamic workflow's, whose agents name their
    kind (:data:`WORKFLOW_AGENT`) — while it is the one workflow at work (``Helpers.of_kind``);
    ``None``: the run's own (several workflows at work: the call cannot be told apart)."""
    who, kind = given.get("agent_id"), str(given.get("agent_type") or "")
    if who is None:
        return None
    if kind == WORKFLOW_AGENT:
        return helpers.of_kind(Kind.WORKFLOW, str(who))
    return await helpers.of(str(who), kind)


async def _subagents(helpers: Helpers, message: Any) -> None:
    """A subagent's start (the main agent's ``Agent`` call: its ``subagent_type`` and
    ``prompt``; started already when its ``PreToolUse`` came first: :func:`_briefed`) or end
    (that call's result) in ``message``; and whose task it is (``task_started``: the task's
    id is the ``agent_id`` Claude Code names the subagent by in its calls' hooks). A
    subagent Claude Code runs in the background (``is_backgrounded``: its call's result is
    its launch) ends when its task does (``task_notification``; a ``task_updated`` to a
    terminal status may be all that comes). In a conversation's live session such a
    subagent's run is the session's to keep (``claude_live.Background``): it is no longer
    one of these helpers."""
    from claude_agent_sdk import (
        AssistantMessage,
        SystemMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    if isinstance(message, SystemMessage):
        await _tasks(helpers, message)
    elif isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
        for call in message.content:
            if isinstance(call, ToolUseBlock) and call.name in SUBAGENT_TOOLS:
                await helpers.start(
                    call.id,
                    kind=Kind.SUBAGENT,
                    name=str(call.input.get("subagent_type") or GENERAL_PURPOSE),
                    task=str(call.input.get("prompt") or call.input.get("description") or ""),
                )
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for result in message.content:
            if isinstance(result, ToolResultBlock):
                text = _result_text(result.content)
                if result.is_error:
                    await helpers.end(result.tool_use_id, error=text or SUBAGENT_FAILED)
                elif not helpers.waits(result.tool_use_id):  # not a background one's launch
                    await helpers.end(result.tool_use_id, output=text)


async def _tasks(helpers: Helpers, message: Any) -> None:
    """A task's start or end in a system ``message`` (``task_said``): whose helper its
    worker works for, and how a background helper's work ended."""
    found = task_said(message)
    if isinstance(found, TaskStarted) and found.call_id:
        helpers.attribute(found.task_id, found.call_id, background=found.background)
    elif isinstance(found, TaskEnded):
        failed = found.status != TASK_COMPLETED
        stopped = found.status not in (TASK_COMPLETED, TASK_FAILED)
        if found.final:  # its notification: the summary is its last word
            said = STOPPED if stopped else found.summary
            await helpers.settled(found.task_id, failed=failed, said=said)
        else:  # an update says only how it ended, and more may be said
            said = STOPPED if stopped else ""
            await helpers.settled(found.task_id, failed=failed, said=said, final=False)


async def _provided(steps: Steps, message: Any) -> None:
    """What the provider did inside a model step, in ``message`` — the main agent's, or a
    subagent's (its messages name the ``Agent`` call that runs it: its helper's run is whose
    record it is, ``Helpers.working``). Its reasoning (a ``thinking`` block) is on the run's
    stream as a ``reasoning`` event — the text only, a subagent's naming its run
    (``helpers.naming``): the block's signature goes nowhere, and none of it to memory or a
    log. A tool the API ran itself (``server_tool_use``: web search, code execution, the
    advisor) is a call of the run, or of the subagent's (:func:`_served`): recorded with its
    result where the stream carries one (the SDK parses the advisor's), and with none at the
    turn's result."""
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        ServerToolResultBlock,
        ServerToolUseBlock,
        ThinkingBlock,
    )

    if isinstance(message, ResultMessage):
        for ref in list(steps.served):
            await _served(steps, ref, None)
    elif isinstance(message, AssistantMessage):
        helpers = steps.runtime.helpers
        assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
        parent = message.parent_tool_use_id
        helper = None if parent is None else helpers.of_key(parent)
        for block in message.content:
            if isinstance(block, ThinkingBlock) and block.thinking:
                whose = naming(helper) or {}
                steps.runtime.events.custom("reasoning", text=block.thinking, **whose)
            elif isinstance(block, ServerToolUseBlock):
                steps.served[block.id] = (block.name, dict(block.input), time.time(), helper)
            elif isinstance(block, ServerToolResultBlock) and block.tool_use_id in steps.served:
                await _served(steps, block.tool_use_id, block.content)


async def _served(steps: Steps, ref: str, content: Mapping[str, Any] | None) -> None:
    """A tool the API ran itself for the model (its use ``ref``), recorded as the call of the
    run — or of the subagent whose step it was — (``bridge.observed``) with what it returned
    (``content``; ``None``: not on the SDK's stream): a read, and nobody's to approve — the
    provider already ran it."""
    name, args, started, helper = steps.served.pop(ref)
    failed = content is not None and "error_code" in content
    outcome = ToolOutcome(
        tool=name,
        status=ToolStatus.ERROR if failed else ToolStatus.OK,
        output=content,
        error_class="ServerToolError" if failed else None,
    )
    spec = ToolSpec(name=name, side_effects="read")
    helpers = steps.runtime.helpers
    assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
    with helpers.working(helper):
        await bridge.observed(steps.runtime, spec, args, outcome, ref=ref, started=started)


def _noticed(runtime: Runtime, message: Any) -> None:
    """What ``message`` says beside the model's replies, as a warning of the run: Claude's
    rate limit near or reached (one that is fine says nothing), a frame of
    :data:`NOTICES` (with what it carries, as Claude Code said it), a reply that failed."""
    from claude_agent_sdk import AssistantMessage, RateLimitEvent, SystemMessage

    if isinstance(message, RateLimitEvent):
        info = message.rate_limit_info
        if info.status != "allowed":
            said = f"Claude's rate limit ({info.rate_limit_type}): {info.status}"
            runtime.events.warning("rate_limit", said)
    elif isinstance(message, SystemMessage) and message.subtype in NOTICES:
        carried = {
            key: value
            for key, value in message.data.items()
            if key not in ("type", "subtype", "uuid", "session_id")
            and isinstance(value, str | int | float | bool)
        }
        said = f"Claude Code: {message.subtype} {json.dumps(carried, sort_keys=True)}"
        runtime.events.warning(NOTICES[message.subtype], said)
    elif isinstance(message, AssistantMessage) and message.error:
        runtime.events.warning("model_error", f"Claude's reply failed: {message.error}")


def _text_delta(event: Mapping[str, Any]) -> str | None:
    """The text of one raw event of the model's stream (a ``StreamEvent``'s), if it is a
    text delta: what the SDK's own stream gives as the model writes it."""
    delta = event.get("delta") if event.get("type") == "content_block_delta" else None
    if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
        return str(delta.get("text") or "")
    return None


def _transcript(messages: Sequence[Any]) -> list[Said]:
    """The query's own messages as said (its subagents' are inside their ``Agent`` call): each
    text, each call — a harness tool by its own name, not Claude's ``mcp__trellis__`` one — and
    each call's result, so what memory keeps of the turn holds its steps."""
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from trellis.harness.tools.convert.claude import bare

    out: list[Said] = []
    for message in messages:
        if getattr(message, "parent_tool_use_id", None) is not None:
            continue
        if isinstance(message, AssistantMessage):
            text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
            calls = tuple(
                Called(b.id, bare(b.name) or b.name, b.input)
                for b in message.content
                if isinstance(b, ToolUseBlock)
            )
            if text or calls:
                out.append(Said(Speaker.ASSISTANT, text, tool_calls=calls))
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            out.extend(
                Said(Speaker.TOOL, _result_text(b.content), tool_call_id=b.tool_use_id)
                for b in message.content
                if isinstance(b, ToolResultBlock)
            )
    return out


def _result_text(content: Any) -> str:
    """A tool result's text (a string, its text blocks, or the text a built-in's structured
    response holds)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        found = content.get("content", content.get("output", content.get("stdout")))
        return _result_text(found) if found is not None else json.dumps(content, default=str)
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


class Spending:
    """What one query spends, charged to its run as its results come. A result's usage is
    the main agent's own for that turn, and its cost its session's running total — the
    subagents' included, and what the session cost before the process that resumed it (the
    bundled CLI's results, driven against a scripted model). So each reply's usage is kept
    by its message id until a result counts it — the main agent's are in the result's
    usage, its subagents' are counted with it — and a result is charged what the session's
    cost grew by since the last one charged (``bridges.claude.SessionCost``, which the live
    session serving the query shares): this query's (it gives one for the turn and one for
    every turn the session ran for a background task), an earlier query's of the run (the
    journal's ``session_cost``), or the conversation's last turn's in this process
    (``ClaudeAdapter.costs``) — journaled as the query starts, so an attempt that resumes
    the session after a pause before any result is charged its own cost, on any worker."""

    __slots__ = ("cost", "costs", "helped", "journal", "options", "own", "results", "runtime")

    def __init__(
        self, runtime: Runtime, options: Any, costs: LRU[str, SessionCost], session: str | None
    ) -> None:
        self.runtime = runtime
        self.options = options
        self.costs = costs
        self.journal = runtime.replay.journal
        self.own: dict[str, dict[str, Any]] = {}
        self.helped: dict[str, dict[str, Any]] = {}
        #: how many of its results were charged
        self.results = 0
        if session is None:  # a new session: nothing was spent in it
            self.journal.session_cost = None
        known = self.journal.session_cost
        found = None if session is None or known is not None else costs.get(session)
        #: the session's running cost (the journal's, an earlier attempt's of the run; else
        #: the one its last turn in this process left): shared with the live session that
        #: serves the query (``ClaudeAdapter._live``)
        self.cost = found if found is not None else SessionCost(known or 0.0)
        if session is not None:
            # in the journal from the start: an attempt that pauses before its first result
            # leaves it to the one that resumes the session, on any worker
            self.journal.session_cost = self.cost.total

    def replied(self, message: Any) -> None:
        """An assistant message: its reply's usage (one reply — one model call — comes as a
        message a block, each with the reply's id)."""
        if message.message_id and message.usage:
            whose = self.own if message.parent_tool_use_id is None else self.helped
            whose[message.message_id] = message.usage

    def ended(self, result: Any, live: LiveSession | None) -> None:
        """A turn's result. One of a ``live`` session comes with what its turn is charged
        (``claude_live.LiveSession.charged``, which read the running cost they share)."""
        cost = result.total_cost_usd
        if live is None and cost is not None:
            cost = self.cost.grew_by(cost)
        self.journal.session_cost = self.cost.total
        self.results += 1
        if result.session_id:
            self.costs[result.session_id] = self.cost
        # one model call for each reply the CLI reported, its own and its subagents' (what
        # the provider serves); a turn no reply of the model's is in — one the CLI answered
        # itself, a slash command (``/compact``: ``num_turns`` 0) — is no model call: the
        # run, its answer empty, is one no model replied in (``pipeline._unanswered``)
        calls = len(self.own) + len(self.helped) or int(bool(result.num_turns))
        _spent(self.runtime, self.options, result, self.helped.values(), cost=cost, calls=calls)
        self.own.clear()
        self.helped.clear()

    async def closed(self, live: LiveSession | None, turn: Turn) -> None:
        """The query ended — with no result (paused, stopped), or its subagents went on after
        its last: what the replies no result counted said; or, a live session's ``turn``
        left before its result, that result when it comes at once
        (``LiveSession.outcome``)."""
        late = None if live is None or self.results else await live.outcome(turn)
        if late is not None:
            self.ended(late, live)
        else:
            self.left()

    def left(self) -> None:
        """What the replies no result counted said: a model call each."""
        replies = [*self.own.values(), *self.helped.values()]
        if replies:
            _spent(self.runtime, self.options, None, replies, calls=len(replies))


def _spent(
    runtime: Runtime,
    options: Any,
    result: Any,
    replies: Iterable[dict[str, Any]],
    *,
    calls: int,
    cost: float | None = None,
) -> None:
    """What a query spent, on the run's budget (``calls``: the model calls it counts as):
    its result's usage (the main agent's), the
    usage of its subagents' ``replies`` since the result before, and ``cost`` (what the
    result's ``total_cost_usd`` — the subagents' included — is charged for). With no result
    (a query that paused or was stopped), the usage its ``replies`` carried, which the CLI
    gives no cost for."""
    usages = [*([] if result is None else [result.usage or {}]), *replies]

    def total(key: str) -> int:
        return sum(int(u.get(key) or 0) for u in usages)

    cached, written = total("cache_read_input_tokens"), total("cache_creation_input_tokens")
    budget.reported(
        runtime,
        model=options.model,
        input_tokens=total("input_tokens") + cached + written,
        output_tokens=total("output_tokens"),
        cached=cached,
        written=written,
        cost=cost,
        **({} if calls == 1 else {"calls": calls}),
    )


def _resumed(native_input: ClaudeInput, run: Invocation) -> str:
    """What the resumed session is told: the call a person answered about, named as Claude
    calls it (a harness tool through the ``trellis`` server, a built-in by its own name)."""
    from trellis.harness.tools.convert.claude import named

    call = native_input.answered
    if call is None:
        return RESUMED
    harness = any(t.name == call.tool for t in run.tools)
    name = named(call.tool) if harness else call.tool
    return resumed_call(call, name, denied=native_input.denied, asked=native_input.asked)


class Steps:
    """One query's steps as the stall watchdog sees them (``Runtime.model_started``): Claude is
    waited on from one message to the next, except while a tool call runs (from its
    ``PreToolUse`` to its ``PostToolUse``: a tool's own time is not a stalled model); and the
    calls under way, by Claude's id, with when each began (what ``PostToolUse`` records), and
    the subagent each call a subagent makes is the work of (``helped``)."""

    __slots__ = (
        "adapter",
        "answered",
        "calls",
        "closed",
        "helped",
        "plans",
        "runtime",
        "served",
        "settling",
    )

    def __init__(self, runtime: Runtime, adapter: ClaudeAdapter) -> None:
        self.runtime = runtime
        #: the calls whose approval a person answered before they ran (governance's question,
        #: whatever made it ask: ``PreToolUse``), by Claude's id: Claude Code's own question
        #: about one (``can_use_tool``) is not asked too — one call, one question. Another
        #: question answered meanwhile (a hook's own ``ask``) is no approval of the call
        self.answered: set[str] = set()
        #: the query ended: a call of it that ends later (a live session's process goes on)
        #: watches nothing
        self.closed = False
        #: the run paused, and its query's own process is read to its result (``_settled``):
        #: a call the model makes then is refused, ending the turn
        self.settling = False
        #: the tools the API ran itself whose result has not come, by Claude's id of the use:
        #: their name, arguments, when they were seen and the subagent whose step it was
        #: (``None``: the main agent's) (:func:`_provided`)
        self.served: dict[str, tuple[str, dict[str, Any], float, Helper | None]] = {}
        #: whose query it is: the MCP tools it is seen calling join its tool set
        self.adapter = adapter
        #: the tool calls under way: their name, arguments and start (``time.time()``)
        self.calls: dict[str, tuple[str, dict[str, Any], float]] = {}
        #: the calls a subagent makes, by Claude's id: the helper whose work each is — the
        #: CLI names the subagent in the ``PreToolUse`` hook by its task's id (``agent_id``),
        #: and its permission request (``can_use_tool``) for the same call is the same helper's
        self.helped: dict[str, Helper] = {}
        #: the folder Claude keeps its plan in, when the query is in plan mode (``_options``:
        #: :data:`PLAN_MODE`); ``None``: it is not
        self.plans: Path | None = None

    async def helper_of(self, call_id: str, given: Mapping[str, Any]) -> Helper | None:
        """The helper whose work the call ``call_id`` is, from the ``PreToolUse`` hook's input
        (``given``: the CLI names a subagent's call by its task's id, ``agent_id``, which the
        task's start tied to the ``Agent`` call that started it: ``Helpers.of``, as the call's
        record finds it — two subagents of one type are each their own), kept for the call's
        permission request; ``None``: the run's own call."""
        helpers = self.runtime.helpers
        assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
        working = await _working(helpers, given)
        if working is not None:
            self.helped[call_id] = working
        return working

    def waiting(self) -> None:
        """Claude is to answer now (unless a call runs): watched for a stall."""
        if not self.calls and not self.closed:
            self.runtime.model_started(STEP)

    def started(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self.calls[call_id] = (name, args, time.time())
        self.runtime.model_ended(STEP)
        from trellis.harness.tools.convert.claude import bare

        own = bare(name)
        if own is not None:  # its MCP request names no call: said here
            self.runtime.replay.announced(call_key(own, args), call_id)

    def approved(self, tool_use_id: Any) -> None:
        """A person approved the call the CLI names ``tool_use_id`` before it ran: carried to
        Claude Code's own question about that call (:meth:`carries`), until the call ends. A
        call the CLI names no id for carries nothing: its tool's name would name every later
        call of the tool."""
        if tool_use_id:
            self.answered.add(str(tool_use_id))

    def carries(self, tool_use_id: Any) -> bool:
        """Whether a person approved the call ``tool_use_id`` already (:meth:`approved`)."""
        return bool(tool_use_id) and str(tool_use_id) in self.answered

    def ended(self, call_id: str) -> tuple[str, dict[str, Any], float] | None:
        self.helped.pop(call_id, None)
        self.answered.discard(call_id)
        found = self.calls.pop(call_id, None)
        self.waiting()
        return found


def _options(
    options: Any,
    context: str | None,
    run: Invocation,
    session: str | None,
    steps: Steps,
    *,
    kept: bool = False,
) -> Any:
    """The options one query runs with. ``kept``: a conversation's turn, which a live session
    may serve — its context hook is there whether or not this turn has a context, so every
    turn's hooks have the same places."""
    from trellis.harness.tools.convert import claude as convert

    if options.permission_prompt_tool_name:
        raise ConfigurationError(
            "the harness decides Claude's tool calls with a PreToolUse hook: give your own "
            "permission check as can_use_tool (the hook asks it after governance), not "
            "permission_prompt_tool_name"
        )
    runtime = run.runtime
    # whose the CLI's model calls are, told the gateway for as long as the process lives: a
    # query's own, its run and trace; one a live session may keep, what holds for every turn
    headers = lasting_headers(runtime) if kept else gateway_headers()
    env = dict(routed_env(options.env, runtime, headers) or options.env)
    if MCP_OUTPUT_ENV not in env and MCP_OUTPUT_ENV not in os.environ:
        env[MCP_OUTPUT_ENV] = str(MCP_OUTPUT_TOKENS)
    # the run's tenant and user's own Claude Code folder, with no auto memory (the team's own
    # word in the options' ``env`` stands)
    env.setdefault(CONFIG_ENV, config_folder(runtime.tenant, runtime.user))
    env.setdefault(AUTO_MEMORY_ENV, "1")
    if options.permission_mode == PLAN_MODE:
        steps.plans = Path(env[CONFIG_ENV]) / PLANS
    shell = _shell(options, env)
    hooks = _with_governance(options.hooks, steps, shell)
    prompt, context = _told(options.system_prompt, context)
    changes: dict[str, Any] = {
        "hooks": with_context(hooks, context or "") if context or kept else hooks,
        "system_prompt": prompt,
        # what Claude Code asks its user about, the harness answers (the team's own first)
        "can_use_tool": _asked(options.can_use_tool, steps, shell, _deployed(runtime, options)),
        "max_buffer_size": options.max_buffer_size or CLI_MESSAGE_BYTES,
        "env": env,
        "settings": _settings(options),
        # a user's text is delivered as written: an ``@path`` in it reads no file, a
        # ``/command`` runs nothing — unless the app says so for the run
        "verbatim_prompts": runtime.framework_options.get("verbatim_prompts", True),
    }
    store = session_store(options, runtime.replay.journal, runtime.agent_id)
    if store is not None:
        # each step mirrored as it is taken: the journal saved with a pause holds it
        changes["session_store"] = store
        changes["session_store_flush"] = SESSION_FLUSH
    if run.native_tools is not None:
        servers = configured_servers(options.mcp_servers)
        changes["mcp_servers"] = {**servers, convert.SERVER: run.native_tools}
    changes["disallowed_tools"] = _disallowed(runtime, options.disallowed_tools)
    if session is not None:
        changes["resume"] = session
        # the run's own session goes on whole: where the options cut the one they resumed is not
        # where it is cut (the paused turn's work would go)
        changes["resume_session_at"] = changes["resume_drops_turn"] = None
        if not options.fork_session:
            # the session the options named to start (``session_id``) is the one resumed:
            # Claude Code takes both only to fork
            changes["session_id"] = None
    return dataclasses.replace(options, **changes)


def _settings(options: Any) -> str:
    """The options' ``settings`` with other Claude sessions' messages refused
    (:data:`INBOUND`), as inline JSON: theirs as given — inline, or a file read as the CLI
    reads it (relative to its ``cwd``, with or without a BOM). Settings that cannot be read
    start no query: sent without them, the team's rules would be gone."""
    given = (options.settings or "").strip()
    try:
        if given and not (given.startswith("{") and given.endswith("}")):
            given = (Path(options.cwd or "") / given).read_text(encoding="utf-8-sig")
        return json.dumps({**json.loads(given or "{}"), **INBOUND})
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigurationError(
            f"the options' settings ({options.settings!r}) could not be read: {exc}"
        ) from exc


def config_folder(tenant: str, user: str) -> str:
    """The Claude Code config folder of ``tenant``'s ``user`` on this worker: under the XDG data
    folder (``$XDG_DATA_HOME/trellis/claude``, else ``~/.local/share/trellis/claude``), made
    for them alone. The SDK seeds a resumed session's folder from it."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    folder = Path(base) / "trellis" / "claude" / stable_id(tenant, user, prefix="u_")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    return str(folder)


def _told(given: Any, context: str | None) -> tuple[Any, str | None]:
    """The options' system prompt and the turn's context, one of them with what Claude is told
    of its folder's ``outputs/`` (``sandbox.DELIVERS_HERE``: the user's, as every run's is —
    ``sandbox.delivered``): after the team's prompt — appended to a preset (``append``), to a
    custom prompt or a text — or, where there is none to append to, with the context: no
    prompt stays none (the SDK's default), and one read from a file is left as it is (the CLI
    takes no text beside it)."""
    from trellis.harness.sandbox import DELIVERS_HERE

    def after(text: str | None) -> str:
        return f"{text}\n\n{DELIVERS_HERE}" if text else DELIVERS_HERE

    kind = given.get("type") if isinstance(given, Mapping) else None
    if kind == "preset":
        return {**given, "append": after(given.get("append"))}, context
    if kind == "custom":
        return {**given, "prompt": after(given.get("prompt"))}, context
    if isinstance(given, str) and given:
        return after(given), context
    return given, after(context)


def routed_env(
    env: Mapping[str, str], runtime: Runtime, headers: Mapping[str, str]
) -> dict[str, str] | None:
    """The CLI's environment with its model calls routed through the gateway when the route
    serving the harness has an Anthropic route (``Capability.ANTHROPIC_ROUTE``: Bifrost;
    ``None``: as it is): the gateway's Anthropic route (``ANTHROPIC_BASE_URL``, the
    virtual key as ``ANTHROPIC_API_KEY``) — unless the options' own ``env`` names a base URL
    (code over the environment) — and, on Bifrost's route, the deny-all MCP scope and
    ``headers`` as ``ANTHROPIC_CUSTOM_HEADERS`` lines beside the options' own (the CLI reads
    them once, as it starts)."""
    gateway = runtime.agent.harness.model_gateway
    if not gateway.has(Capability.ANTHROPIC_ROUTE):
        return None
    gateway = cast("OpenAICompatible", gateway)
    given = dict(env)
    route = f"{gateway.origin}/{ANTHROPIC_ROUTE}"
    if "ANTHROPIC_BASE_URL" not in given:
        given["ANTHROPIC_BASE_URL"] = route
        given["ANTHROPIC_API_KEY"] = gateway.key or NO_KEY
    if not given["ANTHROPIC_BASE_URL"].startswith(gateway.origin):
        return given if given != env else None  # their own endpoint: no Bifrost headers
    ours = {**NO_GATEWAY_TOOLS, **headers}
    lines = (given.get(CUSTOM_HEADERS) or "").splitlines()
    kept = [line for line in lines if line.partition(":")[0].strip().lower() not in ours]
    given[CUSTOM_HEADERS] = "\n".join([*kept, *(f"{k}: {v}" for k, v in ours.items())])
    return given


def _disallowed(runtime: Runtime, denied: Sequence[str] | None) -> list[str]:
    """The options' ``disallowed_tools`` with the framework's own tools the run does not offer:
    those whose work outlives the query (:data:`OUTLIVES_THE_QUERY`), those that start agents
    no run holds (:data:`UNHELD_AGENTS`), and those the tool hints narrow away (Claude's
    built-ins and MCP tools too, for the whole query; never its skills slot,
    :data:`SKILLS_SLOT`)."""
    given = list(denied or ())
    narrowed = (
        [n for n in runtime.native if n != SKILLS_SLOT and not runtime.offers(n)]
        if runtime.offered is not None
        else []
    )
    hidden = [n for n in (*OUTLIVES_THE_QUERY, *UNHELD_AGENTS, *narrowed) if n not in given]
    return [*given, *dict.fromkeys(hidden)]


def _with_governance(hooks: Mapping[str, Any] | None, steps: Steps, shell: Shell) -> dict[str, Any]:
    """The options' hooks with the governance hook (:func:`_governance`) and the record hooks
    (:func:`_recorded`) after theirs — registered after them, and answering after them
    (``after_theirs``)."""
    given = dict(hooks or {})
    given[TOOL_EVENT] = [*given.get(TOOL_EVENT, ()), _governance(steps, shell)]
    recorded = _recorded(steps, shell)
    for event in (RAN_EVENT, FAILED_EVENT):
        given[event] = [*given.get(event, ()), recorded]
    given[RAN_EVENT] = [*given[RAN_EVENT], _steering()]
    return after_theirs(given)


def _steering() -> Any:
    """The ``PostToolUse`` hook that gives Claude the messages steered into the run since its
    last tool call (``Runtime.steered``), as the context a hook adds after the call's result."""

    async def steered(given: Any, tool_use_id: str | None, hook: Any) -> SyncHookJSONOutput:
        runtime = current()
        taken = runtime.steered() if runtime is not None else []
        if not taken:
            return {}
        said = "\n\n".join(f"{STEERED_SAYS}{s.text}" for s in taken)
        return {"hookSpecificOutput": {"hookEventName": RAN_EVENT, "additionalContext": said}}

    return harness_matcher(steered)


def _governance(steps: Steps, shell: Shell) -> Any:
    """The ``PreToolUse`` hook every tool call goes through, in every permission mode
    (:func:`_decide`; in plan mode, :func:`_unplanned` first). A call it cannot decide — the
    catalog or the run store out of reach — is refused, saying why; were the refusal to fail
    too, Claude Code blocks the call (the matcher's ``on_failure``)."""

    from trellis.harness.tools.convert.claude import WAITING

    async def pre_tool_use(given: Any, tool_use_id: str | None, hook: Any) -> SyncHookJSONOutput:
        if steps.settling:  # the run paused: the model's reply to the pause takes no step
            return {**_decided("deny", WAITING), "continue_": False, "stopReason": WAITING}
        try:
            refused = await _offline(given["tool_name"])
            if refused is not None:  # the network the catalog does not allow
                return _decided("deny", refused)
            held = await _unplanned(steps, shell, given, tool_use_id)
            if held is not None:  # plan mode, and no plan approved yet
                return _decided("deny", held)
            return await _decide(steps, given, tool_use_id)
        except Exception as exc:
            log.warning("the call to %s was refused: %s", given.get("tool_name"), exc)
            steps.helped.pop(tool_use_id or "", None)
            return _decided("deny", UNDECIDED.format(why=f"{type(exc).__name__}: {exc}"))

    return deciding_matcher(pre_tool_use)


async def _offline(name: str) -> str | None:
    """What a call of a built-in that reaches the network (:data:`CLAUDE_NETWORK`) is refused
    with, before anything else decides it, when the tool catalog does not allow it
    (``bridge.outbound``); ``None`` for any other call."""
    if name not in CLAUDE_NETWORK:
        return None
    runtime = current()
    assert runtime is not None  # the CLI asks inside the run it works for
    return await bridge.outbound(_spec(runtime, name))


async def _unplanned(steps: Steps, shell: Shell, given: Any, tool_use_id: str | None) -> str | None:
    """What a call is refused with while the query is in plan mode with no plan approved
    (:data:`PLAN_MODE`), recorded as the run's rejected call: one that does more than read,
    other than presenting the plan and writing it in Claude's plans folder. ``None``: it is
    decided as ever (:func:`_decide`)."""
    from trellis.harness.adapters.base import HELPER
    from trellis.harness.governance.shell import EDITORS, file_of
    from trellis.harness.hooks import Deny, denied
    from trellis.harness.tools.convert.claude import bare

    runtime = steps.runtime
    name, args = given["tool_name"], dict(given.get("tool_input") or {})
    plans = steps.plans
    if plans is None or runtime.replay.journal.plan_approved or name == PLAN_TOOL:
        return None
    written = Path(file_of(args))
    if name in EDITORS and ".." not in written.parts and plans in written.parents:
        return None  # its plan, where Claude keeps it (written out: no `..` climbs out)
    if shell.read_only(name, args):
        return None
    tool = bare(name)
    found = runtime.toolbox.get(tool) if tool is not None else None
    spec = found.spec if found is not None else _spec(runtime, name)
    governance = runtime.agent.harness._governance(runtime.tenant)
    decision = await governance.check(
        spec.name, args, side_effects=spec.side_effects, agent_id=runtime.agent_id
    )
    if decision.risk == "read" and HELPER not in spec.tags:
        return None
    outcome = denied(ToolCall(tool=spec.name, args=args), Deny(PLAN_WAITS))
    ref = tool_use_id or str(given.get("tool_use_id") or name)
    await bridge.observed(runtime, spec, args, outcome, ref=ref, started=time.time())
    return str(outcome.output)


async def _decide(steps: Steps, given: Any, tool_use_id: str | None) -> SyncHookJSONOutput:
    """What the run decides about a tool call before Claude Code does: a harness tool runs
    (the bridge decides about it); any other is the run's decision (``bridge.permitted``:
    hooks, governance), then Claude Code's own — its permission mode and rules, and
    :func:`_asked` when it would ask — with the arguments as they were decided (a call a hook
    rewrote is asked about as rewritten). A subagent's call is let through with its prompt
    briefed (:func:`_briefed`), and a workflow's opens its child run. Before Claude takes a
    step, the run's progress is saved — its session as mirrored so far among it — so a worker
    that dies during the step continues the session from there, not from an earlier step."""
    from trellis.harness.tools.convert.claude import WAITING, bare

    runtime = current()
    assert runtime is not None  # the CLI asks inside the run it works for
    await runtime.progress(now=True)
    name, args = given["tool_name"], dict(given.get("tool_input") or {})
    call_id = tool_use_id or str(given.get("tool_use_id") or name)
    if bare(name) is not None:
        steps.started(call_id, name, args)  # the bridge records it
        return _decided("allow")
    assert runtime.helpers is not None  # the query runs with the attempt's helpers
    helper = await steps.helper_of(call_id, given)
    try:
        # a subagent's call is decided as its helper's work: an approval names the helper
        with approvals_read() as approved, runtime.helpers.working(helper):
            decided = await bridge.permitted(_spec(runtime, name), args, call_id=call_id, kept=True)
    except Paused:  # the run pauses: Claude stops here, and the resume goes on
        steps.helped.pop(call_id, None)
        return {**_decided("deny", WAITING), "continue_": False, "stopReason": WAITING}
    if isinstance(decided, ToolOutcome):
        steps.helped.pop(call_id, None)
        return _decided("deny", str(decided.output))
    if any(c.tool == name for c in approved):
        # a person answered the approval of this call, whatever made governance ask (a
        # hook's own question, answered meanwhile, approves nothing)
        steps.approved(tool_use_id or given.get("tool_use_id"))
    if name in SUBAGENT_TOOLS:
        ran = await _briefed(runtime, call_id, name, decided.args)
        steps.started(call_id, name, ran)
        return _decided("allow", updated=ran)
    if name == WORKFLOW_TOOL:  # its child run, open from here as a background subagent's
        flow = decided.args.get("script") or decided.args.get("name")
        task = str(flow or decided.args.get("scriptPath") or "")
        await runtime.helpers.start(call_id, kind=Kind.WORKFLOW, name=WORKFLOW, task=task)
    steps.started(call_id, name, decided.args)
    if decided.args != args:
        return _decided("ask", updated=decided.args)
    return {}  # Claude Code's own permission check decides


def _shell(options: Any, env: Mapping[str, str]) -> Shell:
    """The CLI's shell (``governance.shell.Shell``): where it works (its ``cwd``, else this
    process's); what else the run may read there - the folders the options add to Claude
    Code's own (``add_dirs``: it reads and works there itself, and a team names a folder of
    its own there, such as the tool results the CLI saves), the plugins it is given by a
    folder (the skills written for the run: ``skills.for_run``), where a kept result is read
    (``results.PREFIX``: the name the model is told, which it also tries in its shell); and
    the environment the SDK starts it with (this process's, under the options' ``env``)."""
    from trellis.harness.tools import results

    plugins = [p for p in options.plugins or () if isinstance(p, Mapping)]
    given = [str(p["path"]) for p in plugins if p.get("path")]
    added = [str(d) for d in options.add_dirs or ()]
    reads = [*added, *given, results.PREFIX]
    return Shell(options.cwd or os.getcwd(), reads, {**os.environ, **env})


def _deployed(runtime: Runtime, options: Any) -> bool:
    """Whether the options' ``can_use_tool`` is the deployment's (the wrapped options', or
    ``h.wrap(framework_options=)``'s), not a run's own."""
    agent = runtime.agent
    deployed = agent.framework_options.get("can_use_tool", agent.target.can_use_tool)
    return options.can_use_tool is deployed


def _asked(own: Any, steps: Steps, shell: Shell, deployed: bool) -> Any:
    """The ``can_use_tool`` the CLI calls when Claude Code would ask its user about a call (its
    permission mode and rules let it through without asking otherwise): the team's own
    callback, when it has one (a run's own, not ``deployed``, may refuse a call, never let one
    through for the person: what it lets through is answered as with none); else a call that
    only reads runs — a read tool, or a shell
    command the CLI's ``shell`` reads as read-only (``Shell.read_only``: from every folder it
    may be in, none known while another call that may move it is under way) —, so does the
    shell (:data:`CLAUDE_SHELLS`: in its sandbox, or on a host the deployment allows it on —
    the deployment's decision, announced in the hook) and the network the catalog let through
    (:data:`CLAUDE_NETWORK`), and any other asks a person (``bridge.asked``); Claude's
    question to its user is the run's (:func:`_answered`). A call not let through ends its step."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    from trellis.harness.tools.convert.claude import WAITING

    async def can_use_tool(name: str, args: dict[str, Any], context: Any) -> Any:
        this = str(getattr(context, "tool_use_id", None) or name)

        def moving() -> bool:
            under_way = (a for ref, (n, a, _) in steps.calls.items() if ref != this and n in SHELLS)
            return any(moves(a) for a in under_way)

        runtime = current()
        assert runtime is not None  # the CLI asks inside the run it works for
        spec = _spec(runtime, name)
        verdict = await own(name, args, context) if own is not None else None
        if not deployed and isinstance(verdict, PermissionResultAllow):
            verdict = None  # a run's own check refuses; the person answers what it lets by
        if verdict is not None:
            pass
        elif name == QUESTION_TOOL:
            try:
                verdict = PermissionResultAllow(updated_input=await _answered(runtime, args))
            except Paused:  # the run pauses: Claude stops here, and the resume goes on
                verdict = PermissionResultDeny(message=WAITING, interrupt=True)
        elif (
            steps.carries(getattr(context, "tool_use_id", None))
            # the shell in its sandbox, or on a host the deployment allows it on, and the
            # network as the catalog let it through: the deployment's, never the user's
            or name in CLAUDE_SHELLS
            or name in CLAUDE_NETWORK
            or (spec.side_effects == "read" and name not in CLAUDE_ASKS)
            or shell.read_only(name, args, moving=moving())
        ):  # a person approved this call (``PreToolUse``)
            verdict = PermissionResultAllow()
        else:
            helpers = runtime.helpers
            assert helpers is not None  # the query runs with the attempt's helpers
            try:
                # the same helper's work as the call's PreToolUse said (``Steps.helped``)
                with helpers.working(steps.helped.get(this)):
                    decided = await bridge.asked(spec, args, CLAUDE_CODE, call_id=this, kept=True)
            except Paused:  # the run pauses: Claude stops here, and the resume goes on
                decided = None
            if decided is None:
                verdict = PermissionResultDeny(message=WAITING, interrupt=True)
            elif isinstance(decided, ToolOutcome):
                verdict = PermissionResultDeny(message=str(decided.output))
            else:
                verdict = PermissionResultAllow(updated_input=decided.args)
        if not isinstance(verdict, PermissionResultAllow):
            steps.ended(this)
        return verdict

    return can_use_tool


async def _answered(runtime: Runtime, args: dict[str, Any]) -> dict[str, Any]:
    """``AskUserQuestion``'s input with the person's ``answers`` (by question, the label picked,
    several joined with ", "), each question asked as ``ask_user`` asks it: its options, several
    picks with ``multiSelect``, the one labelled :data:`RECOMMENDED` recommended (its
    description why), another answer in words. The call the resumed session makes again in
    other words is the one the person answered (``Replay.made``): Claude reads its questions
    with their answers."""
    answers: dict[str, str] = {}
    # the answers it reads are this call's: read again when it is made again after a pause
    async with runtime.asking_within(QUESTION_TOOL, args) as made:
        for asked in made.args.get("questions") or ():
            options = [o for o in asked.get("options") or () if o.get("label")]
            best = next((o for o in options if str(o["label"]).endswith(RECOMMENDED)), {})
            picked = await ask_question(
                asked["question"],
                [str(o["label"]) for o in options],
                multiple=bool(asked.get("multiSelect")),
                recommended=best.get("label"),
                why=best.get("description") or (RECOMMENDED.strip("()") if best else None),
            )
            answers[asked["question"]] = (
                ", ".join(picked) if isinstance(picked, list) else str(picked)
            )
    return {**made.args, "answers": answers}


def _spec(runtime: Runtime, name: str) -> ToolSpec:
    """A call's tool as governance decides it: its risk as the run's tool set has it (an
    in-process server's tool: its annotations), else a built-in's
    (:data:`BUILTIN_SIDE_EFFECTS`)."""
    return runtime.native.get(name) or ToolSpec(name=name, side_effects=_builtin_risk(name))


def _builtin_risk(name: str) -> str:
    """A built-in's risk as the harness says it (:data:`BUILTIN_SIDE_EFFECTS`), none for one
    that reaches the network (:data:`CLAUDE_NETWORK`: the catalog's, an administrator's)."""
    return UNDECLARED if name in CLAUDE_NETWORK else BUILTIN_SIDE_EFFECTS.get(name, UNDECLARED)


def _recorded(steps: Steps, shell: Shell) -> Any:
    """The ``PostToolUse`` / ``PostToolUseFailure`` hook: a call Claude ran itself (a built-in,
    its own MCP servers' tools), recorded as the run's (``tools.bridge.observed``) with its
    result or its error and its duration (from its ``PreToolUse``); a harness tool's call is
    the bridge's. Either way the stall watch goes on once no call runs. A shell command of
    Claude's own (not a subagent's: the CLI keeps their folder) is where the shell may have
    moved (``Shell.ran``)."""
    from trellis.harness.tools.convert.claude import bare

    async def post_tool_use(given: Any, tool_use_id: str | None, hook: Any) -> SyncHookJSONOutput:
        call_id = tool_use_id or str(given.get("tool_use_id") or given.get("tool_name"))
        found = steps.ended(call_id)
        name = str(given.get("tool_name") or "")
        if bare(name) is not None:
            return {}
        steps.adapter.observed(name)
        _, args, started = found or (name, dict(given.get("tool_input") or {}), time.time())
        if name in SHELLS and "agent_id" not in given:
            shell.ran(args)
        failed = given.get("hook_event_name") == FAILED_EVENT
        if name == PLAN_TOOL and not failed:  # the plan was let through: plan mode is over
            steps.runtime.replay.journal.plan_approved = True
        # a workflow agent's call says who made it (several workflows at once: the run's own)
        agent = (
            {k: given[k] for k in AGENT_KEYS} if given.get("agent_type") == WORKFLOW_AGENT else {}
        )
        outcome = ToolOutcome(
            tool=name,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=given.get("error") if failed else _result_text(given.get("tool_response")),
            error_class="ClaudeToolError" if failed else None,
            metadata={bridge.MADE_BY: agent} if agent else {},
        )
        spec = steps.runtime.native.get(name) or native_spec(name)
        helpers = steps.runtime.helpers
        assert helpers is not None  # the query runs with the attempt's helpers (``_helped``)
        # a call a subagent made is recorded under its run: the CLI names the subagent by
        # its task's id (``agent_id``), which its ``task_started`` tied to its ``Agent`` call
        working = await _working(helpers, given)
        with helpers.working(working):
            await bridge.observed(steps.runtime, spec, args, outcome, ref=call_id, started=started)
        return {}

    return harness_matcher(post_tool_use)


async def _declared(target: Any) -> list[ToolSpec]:
    """The tools ``ClaudeAgentOptions`` declare (:meth:`ClaudeAdapter.native`, less what the
    agent was seen calling). An in-process server's own listing comes first: a tool it lists
    is the one with its annotations (its risk), which ``allowed_tools`` naming it does not
    replace (:func:`unique` keeps the first)."""
    from trellis.harness.tools.convert.claude import SERVER, bare

    found: list[ToolSpec] = []
    denied = set(target.disallowed_tools or ())
    for server, config in configured_servers(target.mcp_servers).items():
        instance = config.get("instance") if isinstance(config, Mapping) else None
        if server != SERVER and config.get("type") == "sdk" and instance is not None:
            listed = await _sdk_tools(server, instance)
            found.extend(s for s in listed if s.name not in denied)
    given = target.tools if isinstance(target.tools, list) else list(BUILTIN_SIDE_EFFECTS)
    found.extend(native_spec(name) for name in given if name not in denied)
    found.extend(
        native_spec(name)
        for name in target.allowed_tools or ()
        if name.startswith("mcp__") and bare(name) is None and name not in denied
    )
    return found


async def _sdk_tools(server: str, instance: Any) -> list[ToolSpec]:
    """The tools an in-process MCP server lists (``create_sdk_mcp_server``), as Claude names
    them (``mcp__<server>__<tool>``); ``[]`` when it cannot be asked here (said once)."""
    try:
        entry = instance.get_request_handler("tools/list")
        listed = (await entry.handler(None, None)).tools
    except Exception as exc:
        _unlisted(server, type(exc).__name__)
        return []
    return [
        ToolSpec(
            name=f"mcp__{server}__{tool.name}",
            description=tool.description or tool.name,
            input_schema=dict(tool.input_schema or {"type": "object"}),
            source="mcp",
            server=server,
            side_effects=side_effects_of(tool.annotations),
            **data_of_listed(tool),
        )
        for tool in listed
    ]


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per server, of the most recent
def _unlisted(server: str, reason: str) -> None:
    log.info(
        "an in-process MCP server's tools could not be listed for the run's tool set: they "
        "join it as the agent calls them",
        extra={"server": server, "reason": reason},
    )


def native_spec(name: str) -> ToolSpec:
    """A tool Claude runs itself, as the harness records and publishes it: a built-in by its
    risk (:data:`BUILTIN_SIDE_EFFECTS`) and tagged ``BUILTIN``, an MCP server's tool
    (``mcp__<server>__<tool>``) by its server."""
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return ToolSpec(name=name, source="mcp", server=parts[1])
    risk = _builtin_risk(name)
    spec = builtin(ToolSpec(name=name, side_effects=risk))
    return helper(spec) if name in SUBAGENT_TOOLS else spec


def _decided(
    decision: Literal["allow", "deny", "ask"],
    reason: str | None = None,
    *,
    updated: dict[str, Any] | None = None,
) -> SyncHookJSONOutput:
    """A ``PreToolUse`` hook's answer."""
    said: PreToolUseHookSpecificOutput = {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
    }
    if reason is not None:
        said["permissionDecisionReason"] = reason
    if updated is not None:
        said["updatedInput"] = updated
    return {"hookSpecificOutput": said}


def configured_servers(configured: Any) -> dict[str, Any]:
    """The team's MCP servers as a dict, so the harness's in-process server (which the SDK
    serves only from a dict) goes beside them: a dict as it is; a path to, or the JSON text of,
    what the CLI's ``--mcp-config`` reads (``{"mcpServers": {...}}``) read into one."""
    if isinstance(configured, dict):
        return configured
    if not configured:
        return {}
    text = str(configured).strip()
    data = json.loads(text if text.startswith("{") else Path(text).read_text())
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ConfigurationError(f"mcp_servers {text[:80]!r} holds no mcpServers object")
    return servers
