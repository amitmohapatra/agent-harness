"""``OpenAIAgentsHarness``: the harness around an OpenAI Agents SDK agent (design §8).

    agent = Agent(
        name="inventory",
        instructions=harness.openai_agents.instructions("You answer stock questions."),
        tools=[lookup],
        tool_input_guardrails=[...],          # the adapter adds its own
    )
    run = harness.openai_agents.wrap(agent, agent_id="inventory-agent")
    result = await run("how much stock of SKU-1?", context=context)

What it does not do: own the agent, its handoffs, its output type or its guardrails. The
adapter adds a model provider, a session, run hooks and one tool guardrail pair, and
otherwise leaves ``Runner.run`` alone.

Two pause paths reach the same :class:`Interrupt`:

* a policy ``require_approval`` raises ``ApprovalRequired`` out of ``Runner.run``, and the
  harness records the pause;
* the SDK's own ``needs_approval=True`` tools *return* rather than raise — the run finishes
  with ``RunResult.interruptions`` — so :meth:`OpenAIAgentsHarness.wrap` turns the first
  interruption into the same ``ApprovalRequired``, and :meth:`apply_resolution` puts a
  person's answer back through ``RunState.approve``/``reject``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, Final

from trellis.contracts.errors import AgentPaused, HarnessError
from trellis.contracts.runs import InterruptDecision, InterruptResolution
from trellis.contracts.tool import ToolCall

from trellis.harness.interrupts.signals import ApprovalRequired
from trellis.harness_openai_agents.hooks import (
    FRAMEWORK,
    TrellisRunHooks,
    tool_input_guardrail,
    tool_output_guardrail,
)
from trellis.harness_openai_agents.models import BifrostModel, BifrostModelProvider
from trellis.harness_openai_agents.session import MemoryServiceSession

__all__ = ["PENDING_RESULT_KEY", "OpenAIAgentsHarness", "openai_agents_version"]

#: Where a paused run's ``RunResult`` waits on ``runtime.state``, so a caller that wants to
#: continue the SDK's *own* approval flow can reach the ``RunState`` it needs. The harness's
#: ``resume`` re-runs the agent instead, which is the usual path; this is for an application
#: that would rather resume the SDK in place.
PENDING_RESULT_KEY: Final = "openai_agents.pending_result"


def openai_agents_version() -> str | None:
    """The installed SDK version, for the compatibility matrix and span attributes."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("openai-agents")
    except Exception:  # pragma: no cover - the SDK is not installed
        return None


class OpenAIAgentsHarness:
    """Framework adapter. The only place this project imports the OpenAI Agents SDK."""

    name = FRAMEWORK

    def __init__(self, harness: Any) -> None:
        self.harness = harness
        self.version = openai_agents_version()

    # ------------------------------------------------------------------ capability check
    @staticmethod
    def available() -> bool:
        try:
            import agents  # noqa: F401, PLC0415 - capability probe
        except ImportError:
            return False
        return True

    def supports(self, target: object) -> bool:
        """Whether this adapter can wrap ``target``: anything with the SDK's agent shape."""
        return hasattr(target, "name") and hasattr(target, "tools")

    # ------------------------------------------------------------------ the bindings
    def model(self, *, client: Any = None, model: str | None = None) -> BifrostModel:
        """The SDK ``Model`` that answers through the harness's instrumented client."""
        return BifrostModel(client=client, model=model or self.harness.config.models.default_model)

    def model_provider(self, *, client: Any = None) -> BifrostModelProvider:
        """A provider that resolves every model name to the gateway (for ``RunConfig``)."""
        return BifrostModelProvider(
            client=client, default_model=self.harness.config.models.default_model
        )

    def session(self, *, memory: Any = None, **options: Any) -> MemoryServiceSession:
        """The conversation, kept on the Memory Service thread this run is bound to."""
        return MemoryServiceSession(memory, **options)

    def hooks(self, *, record_to_memory: bool = True) -> TrellisRunHooks:
        """Run hooks: the step events and the answer at run end."""
        return TrellisRunHooks(record_to_memory=record_to_memory)

    def guardrails(self, *, record_to_memory: bool = True) -> tuple[Any, Any]:
        """The tool guardrail pair: the policy decision in, the tool memory out."""
        policy = self.harness.policy if self.harness.policy_enabled else None
        return (
            tool_input_guardrail(policy=policy, record_to_memory=record_to_memory),
            tool_output_guardrail(),
        )

    def instructions(self, prompt: str = "", *, skills: Sequence[Any] = ()) -> Callable[..., str]:
        """Agent instructions that carry the memory bundle.

        The SDK accepts a two-argument callable for ``instructions`` and calls it per run, so
        this is where the context moment belongs: the bundle the harness fetched is rendered
        by the core's :class:`ContextAssembler` — the same renderer, budget and citation
        instruction the harness's own loop uses.
        """
        from trellis.harness.reasoning.assembler import ContextAssembler  # noqa: PLC0415
        from trellis.harness.runtime.propagation import current_runtime  # noqa: PLC0415

        def render(context: Any, agent: Any) -> str:
            runtime = current_runtime()
            if runtime is None:
                return prompt
            return ContextAssembler(prompt=prompt, skills=list(skills)).system_prompt(runtime)

        return render

    # ------------------------------------------------------------------ pauses
    @staticmethod
    def interrupts(result: Any) -> list[ToolCall]:
        """The tool calls a ``needs_approval`` tool is waiting on, as harness tool calls."""
        return [_as_tool_call(item) for item in getattr(result, "interruptions", []) or []]

    @staticmethod
    def pending_result(runtime: Any) -> Any:
        """The ``RunResult`` of a run that paused on a ``needs_approval`` tool, or ``None``.

        ``harness.resume(..., agent=wrapped)`` re-runs the agent, which is what the other
        adapters do and what a stateless surface wants. An application that would rather
        continue the SDK's own run takes this result, turns it into a ``RunState``, applies
        the person's answer with :meth:`apply_resolution`, and hands the state back to
        ``Runner.run``.
        """
        return runtime.state.get(PENDING_RESULT_KEY)

    @staticmethod
    def apply_resolution(state: Any, resolution: InterruptResolution) -> Any:
        """Put a person's answer back into the SDK's run state so the run can continue.

        ``RunState.approve``/``reject`` is the SDK's own resume mechanism; the harness
        decisions map onto it one for one. ``EDIT`` has no counterpart — the SDK approves or
        rejects a call, it does not rewrite one — so it is refused rather than silently
        approved with the arguments the approver wanted changed.
        """
        pending = list(state.get_interruptions())
        if not pending:
            return state
        item = pending[0]
        if resolution.decision is InterruptDecision.APPROVE:
            state.approve(item)
        elif resolution.decision in (InterruptDecision.REJECT, InterruptDecision.CANCEL):
            state.reject(item)
        else:
            raise ValueError(
                f"the OpenAI Agents SDK cannot apply a {resolution.decision.value} decision to a "
                "tool approval: it approves or rejects a call, it does not edit one"
            )
        return state

    # ------------------------------------------------------------------ wrapping
    def wrap(
        self,
        agent: Any,
        *,
        agent_id: str | None = None,
        session: Any = None,
        hooks: Any = None,
        run_config: Any = None,
        skills: list[Any] | None = None,
        max_turns: int | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
        """Run an SDK agent through the harness pipeline.

        The agent is given the adapter's tool guardrails (added to whatever it already has,
        never replacing them), and the run is given the harness's model provider, session and
        hooks unless the caller passed their own.
        """
        from agents import Runner  # noqa: PLC0415 - the adapter's own import
        from agents.exceptions import UserError  # noqa: PLC0415 - the adapter's own import

        name = agent_id or getattr(agent, "name", None) or "openai-agent"
        prepared = self._with_guardrails(agent)
        run_hooks = hooks if hooks is not None else self.hooks()
        run_session = session if session is not None else self.session()
        config = run_config if run_config is not None else self._run_config()

        # ``max_turns`` has an SDK default, so it is passed only when the caller set one
        limit: dict[str, Any] = {} if max_turns is None else {"max_turns": max_turns}

        async def target(payload: Any, runtime: Any) -> Any:
            try:
                result = await Runner.run(
                    prepared,
                    payload,
                    session=run_session,
                    hooks=run_hooks,
                    run_config=config,
                    **limit,
                )
            except UserError as exc:
                raise _unwrapped(exc) from exc
            waiting = self.interrupts(result)
            if waiting:
                # the SDK returns its approvals; the platform has one shape for a pause
                runtime.state[PENDING_RESULT_KEY] = result
                raise ApprovalRequired(waiting[0], reason="the tool requires approval")
            return result.final_output

        return self.harness.wrap(
            target,
            agent_id=name,
            skills=skills,
            framework=FRAMEWORK,
            framework_version=self.version,
            **options,
        )

    def agent(
        self,
        *,
        agent_id: str,
        instructions: str = "",
        tools: Sequence[Any] = (),
        model: str | None = None,
        skills: list[Any] | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Build an SDK ``Agent`` with the harness already bound, and wrap it."""
        from agents import Agent  # noqa: PLC0415 - the adapter's own import

        built = Agent(
            name=agent_id,
            instructions=self.instructions(instructions, skills=skills or ()),
            tools=list(tools),
            model=self.model(model=model),
            **kwargs,
        )
        return self.wrap(built, agent_id=agent_id, skills=skills)

    # ------------------------------------------------------------------ internals
    def _with_guardrails(self, agent: Any) -> Any:
        """A copy of the agent whose tools also carry the adapter's guardrails.

        Per *tool* rather than per agent, because that is where the SDK puts them: a tool
        guardrail is a field of the tool. Copies throughout — the caller's ``Agent`` and its
        tools are often module-level objects shared with other agents, and wrapping one must
        not attach this harness's policy to all of them.
        """
        input_guardrail, output_guardrail = self.guardrails()
        tools = []
        for tool in getattr(agent, "tools", ()) or ():
            if not hasattr(tool, "tool_input_guardrails"):
                tools.append(tool)
                continue
            instrumented = copy.copy(tool)
            instrumented.tool_input_guardrails = [
                *(tool.tool_input_guardrails or []),
                input_guardrail,
            ]
            instrumented.tool_output_guardrails = [
                *(tool.tool_output_guardrails or []),
                output_guardrail,
            ]
            tools.append(instrumented)
        clone = getattr(agent, "clone", None)
        return clone(tools=tools) if callable(clone) else agent

    def _run_config(self) -> Any:
        from agents import RunConfig  # noqa: PLC0415 - the adapter's own import

        return RunConfig(model_provider=self.model_provider(), tracing_disabled=True)


def _unwrapped(error: Exception) -> BaseException:
    """The harness signal the SDK buried, or the SDK's own error.

    The SDK wraps whatever a tool guardrail raises in ``UserError("Error running tool …")``.
    That is reasonable for the SDK — a guardrail that blows up *is* a user error — but it
    would turn a policy denial into an unclassifiable failure and a paused run into a crash.
    The harness's own signals are unwrapped so a denial stays a denial and a pause stays a
    pause, whichever framework ran the tool.
    """
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, AgentPaused | HarnessError):
            return cause
        cause = cause.__cause__
    return error


def _as_tool_call(item: Any) -> ToolCall:
    """One SDK ``ToolApprovalItem`` as the harness's tool call."""
    import json  # noqa: PLC0415 - only on the approval path

    arguments = item.arguments
    try:
        args = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
    except ValueError:
        args = {"arguments": arguments}
    return ToolCall(
        tool=item.name or "",
        args=args if isinstance(args, dict) else {"arguments": args},
        idempotency_key=item.call_id,
    )
