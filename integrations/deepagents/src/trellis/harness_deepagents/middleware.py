"""``TrellisMiddleware``: the six moments, as one LangChain/Deep Agents middleware.

Deep Agents 0.7 is built on LangChain 1.4 middleware, so the four hooks the design named
(``before_agent``, ``wrap_model_call``, ``wrap_tool_call``, ``after_agent``) exist — with
``a``-prefixed async twins, which are the ones a harnessed run uses. This class implements
them and nothing else: the graph, the subagents, the skills and the todo list stay Deep
Agents'.

| Moment | Hook | What it does |
|---|---|---|
| run start | ``abefore_agent`` | opens a step on the run's event stream |
| context | ``awrap_model_call`` | renders the memory bundle into the system message |
| model | ``awrap_model_call`` | replaces the request's model with the gateway-backed one |
| tool | ``awrap_tool_call`` | policy, tool events and tool memory through ``ToolCallBridge`` |
| compaction | ``awrap_model_call`` | observes a summary the summarisation middleware produced |
| run end | ``aafter_agent`` | writes the answer, closes the step |

Compaction needs a word. LangChain's ``SummarizationMiddleware`` offers no post-summary
callback: it rewrites ``messages`` in ``before_model`` and the summary arrives as a
``HumanMessage`` carrying ``additional_kwargs={"lc_source": "summarization"}``. Reading it out
of the assembled model request is therefore the only hook there is — and it is a better place
than a second ``before_model`` would be, because it does not depend on middleware ordering.

One instance serves every run: the per-execution state (the bridge's step counter, which
summaries were already written) lives in ``runtime.state``, so a graph compiled once at
import time stays correct across concurrent runs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from trellis.contracts import ToolStatus
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.runs import RunEventType
from trellis.contracts.tool import ToolOutcome

from trellis.harness.reasoning.assembler import (
    DEFAULT_BUDGET_TOKENS,
    SUMMARY_KIND,
    ContextAssembler,
)
from trellis.harness.telemetry.tracer import Stopwatch
from trellis.harness.tools.bridge import ToolCallBridge
from trellis.harness_deepagents.binding import active_runtime

__all__ = ["FRAMEWORK", "SUMMARY_SOURCE", "TrellisMiddleware"]

#: What this adapter is called on spans, step names and observation metadata.
FRAMEWORK: Final = "deepagents"
#: The marker LangChain's ``SummarizationMiddleware`` leaves on the summary it injects.
SUMMARY_SOURCE: Final = "summarization"
#: The step a Deep Agents run appears as on the event stream.
STEP: Final = "deepagents.agent"
#: Where the per-execution pieces are kept on ``runtime.state``.
BRIDGE_KEY: Final = "deepagents.bridge"
SUMMARISED_KEY: Final = "deepagents.summarised"


class TrellisMiddleware(AgentMiddleware):
    """The harness, as middleware.

        agent = create_deep_agent(
            model=harness.deepagents.model(),
            tools=[search],
            backend=harness.deepagents.backend(),
            middleware=[harness.deepagents.middleware()],
        )

    Built by :class:`DeepAgentsHarness`; constructing it directly is supported for an
    application that assembles its own agent, and a test that passes ``runtime=`` explicitly.
    """

    def __init__(
        self,
        runtime: Any = None,
        *,
        policy: Any = None,
        model: Any = None,
        skills: Sequence[Any] = (),
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        memory_tokens: int | None = None,
        inject_context: bool = True,
        record_to_memory: bool = True,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self.policy = policy
        self.model = model
        self.skills = list(skills)
        self.budget_tokens = budget_tokens
        self.memory_tokens = memory_tokens
        self.inject_context = inject_context
        self.record_to_memory = record_to_memory

    @property
    def name(self) -> str:
        return "TrellisMiddleware"

    # ------------------------------------------------------------------ per-execution state
    @property
    def runtime(self) -> Any:
        return active_runtime(self._runtime)

    def bridge_for(self, runtime: Any) -> ToolCallBridge:
        """This execution's bridge, created once so its step numbering is per run."""
        bridge = runtime.state.get(BRIDGE_KEY)
        if bridge is None:
            bridge = ToolCallBridge(
                runtime,
                policy=self.policy,
                record_to_memory=self.record_to_memory,
                source=FRAMEWORK,
            )
            runtime.state[BRIDGE_KEY] = bridge
        return bridge

    # ------------------------------------------------------------------ run start
    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        await self.runtime.events.emit(RunEventType.STEP_STARTED, step=STEP, framework=FRAMEWORK)
        return None

    # ------------------------------------------------------------------ context + model
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        runtime = self.runtime
        await self._observe_summary(runtime, request.messages)
        overrides: dict[str, Any] = {}
        if self.model is not None:
            overrides["model"] = self.model
        system = self._system_message(runtime, request) if self.inject_context else None
        if system is not None:
            overrides["system_message"] = system
        return await handler(request.override(**overrides) if overrides else request)

    def _system_message(self, runtime: Any, request: ModelRequest) -> SystemMessage | None:
        """The agent's own system prompt with the memory bundle appended, or ``None``.

        The rendering is the core's :class:`ContextAssembler` — the same renderer, the same
        token budget and the same "cite memory ids" instruction the harness's own loop uses,
        so a fact reads identically whichever loop asked for it.
        """
        original = request.system_message.text if request.system_message is not None else ""
        rendered = ContextAssembler(
            prompt=original,
            skills=self.skills,
            budget_tokens=self.budget_tokens,
            memory_tokens=self.memory_tokens,
        ).system_prompt(runtime)
        if not rendered or rendered == original:
            return None
        return SystemMessage(content=rendered)

    # ------------------------------------------------------------------ compaction
    async def _observe_summary(self, runtime: Any, messages: Sequence[BaseMessage]) -> None:
        """Write a summarisation summary to memory, once, as the assembler would."""
        memory = runtime.memory
        if not memory.enabled:
            return
        seen: set[str] = runtime.state.setdefault(SUMMARISED_KEY, set())
        for message in messages:
            if message.additional_kwargs.get("lc_source") != SUMMARY_SOURCE:
                continue
            key = message.id or _text(message)[:200]
            if key in seen:
                continue
            seen.add(key)
            await memory.observe(
                MemoryObservation(
                    content=f"Conversation summary: {_text(message)}",
                    kind=SUMMARY_KIND,
                    # the run's own note: visible to this run and the one that spawned it
                    hints={"visibility": "RUN"},
                    metadata={
                        "source": "compaction",
                        "framework": FRAMEWORK,
                        "compaction": len(seen),
                    },
                )
            )

    # ------------------------------------------------------------------ tool call
    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Policy, events and tool memory around the tool Deep Agents is about to run.

        A denial raises ``PolicyDeniedError`` and a call the policy holds for a person raises
        ``ApprovalRequired``; both leave the graph, and the harness turns the second into a
        paused run. An approver's *rejection* comes back as a refusal the model can read and
        plan around, which is why a deny and a reject are different things.
        """
        runtime = self.runtime
        bridge = self.bridge_for(runtime)
        tool_call = request.tool_call
        asked = dict(tool_call.get("args") or {})
        call = bridge.prepare(str(tool_call.get("name") or ""), asked, call_id=tool_call.get("id"))
        call, rejected = await bridge.authorize(call)
        await bridge.opened(call)
        if rejected is not None:
            await bridge.settled(call, rejected, 0.0, str(rejected.status))
            return ToolMessage(
                content=str(rejected.output),
                tool_call_id=str(tool_call.get("id") or call.idempotency_key),
                status="error",
            )
        if call.args != asked:
            # an approver narrowed the arguments: the framework must run what was approved
            request = request.override(tool_call={**tool_call, "args": dict(call.args)})
        watch = Stopwatch()
        with bridge.tool_span(call):
            try:
                result = await handler(request)
            except Exception as exc:
                await bridge.settled(call, None, watch.ms, "error", error=exc)
                raise
        outcome = _outcome_of(result, call.tool)
        await bridge.settled(call, outcome, watch.ms, str(outcome.status))
        return result

    # ------------------------------------------------------------------ run end
    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Write the answer where the platform keeps answers, and close the step.

        The outcome itself is the core's ``MemoryObservationInterceptor``'s job, on the way
        out of ``harness.wrap``. What the core cannot see is the framework's final message,
        so the adapter observes it here, under the same memory policy the core obeys.
        """
        harness_runtime = self.runtime
        answer = _final_text(state)
        memory = harness_runtime.memory
        if answer and memory.enabled:
            policy = memory.policy
            if policy.record_messages:
                await memory.record_output(answer)
            if policy.observe_output:
                await memory.observe(
                    MemoryObservation(
                        content=answer,
                        kind=SUMMARY_KIND,
                        metadata={
                            "agent_id": harness_runtime.agent_id,
                            "framework": FRAMEWORK,
                        },
                    )
                )
        await harness_runtime.events.emit(
            RunEventType.STEP_FINISHED, step=STEP, framework=FRAMEWORK
        )
        return None


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    ).strip()


def _final_text(state: Any) -> str | None:
    """The last thing the agent said, which is the answer a person sees."""
    messages = (
        state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    )
    for message in reversed(list(messages or ())):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _text(message).strip()
            if text:
                return text
    return None


def _outcome_of(result: ToolMessage | Command[Any], tool: str) -> ToolOutcome:
    """The framework's tool result in the harness's outcome vocabulary."""
    if isinstance(result, ToolMessage):
        failed = result.status == "error"
        return ToolOutcome(
            tool=tool,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=result.content,
            error_class="ToolMessageError" if failed else None,
        )
    # a Command updates graph state rather than answering the model; there is no payload to
    # record beyond the fact that it ran
    return ToolOutcome(tool=tool, status=ToolStatus.OK, output=None, output_summary="command")
