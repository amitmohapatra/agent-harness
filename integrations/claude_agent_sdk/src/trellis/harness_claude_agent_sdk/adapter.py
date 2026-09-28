"""``ClaudeAgentSDKHarness``: the harness around a Claude Agent SDK agent (design §8).

    run = harness.claude_agent_sdk.agent(
        agent_id="reviewer",
        system_prompt="You review pull requests.",
        allowed_tools=["Read", "Grep"],
        gateway_url="http://localhost:8091",
    )
    result = await run("what changed in src/?", context=context)

This adapter differs from the other two in one structural way, and the difference is worth
stating plainly: **the Claude Agent SDK does not call a model**. It spawns the ``claude``
CLI, which does. So there is no client to instrument and no ``Model`` to implement — the
model moment is ``ANTHROPIC_BASE_URL`` in the options' environment (see
:mod:`trellis.harness_claude_agent_sdk.gateway`), and the per-call token and cost figures
the other adapters put on a span come from the CLI's own ``ResultMessage`` instead.

Everything else binds the same way, through hooks: the bundle at ``UserPromptSubmit``, the
policy at ``PreToolUse``/``can_use_tool``, tool memory at ``PostToolUse``, compaction at
``PreCompact``, the answer at ``Stop``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Final

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

# ``HookEvent`` is the Literal union naming the hooks; the SDK keeps it in
# ``claude_agent_sdk.types`` rather than re-exporting it at the top level.
from claude_agent_sdk.types import HookEvent
from trellis.contracts.runs import RunEventType

from trellis.harness_claude_agent_sdk.gateway import gateway_env
from trellis.harness_claude_agent_sdk.hooks import (
    FRAMEWORK,
    PENDING_KEY,
    STEP,
    TrellisHooks,
)

__all__ = ["ClaudeAgentSDKHarness", "claude_agent_sdk_version"]

#: Where a deployment names the gateway when it is not passed in code.
GATEWAY_URL_VAR: Final = "BIFROST_URL"


def claude_agent_sdk_version() -> str | None:
    """The installed SDK version, for the compatibility matrix and span attributes."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("claude-agent-sdk")
    except Exception:  # pragma: no cover - the SDK is not installed
        return None


class ClaudeAgentSDKHarness:
    """Framework adapter. The only place this project imports the Claude Agent SDK."""

    name = FRAMEWORK

    def __init__(self, harness: Any) -> None:
        self.harness = harness
        self.version = claude_agent_sdk_version()

    # ------------------------------------------------------------------ capability check
    @staticmethod
    def available() -> bool:
        try:
            import claude_agent_sdk  # noqa: F401, PLC0415 - capability probe
        except ImportError:
            return False
        return True

    def supports(self, target: object) -> bool:
        """Whether this adapter can wrap ``target``: the SDK's options object."""
        return isinstance(target, ClaudeAgentOptions)

    # ------------------------------------------------------------------ the bindings
    def hooks(
        self,
        *,
        runtime: Any = None,
        skills: Sequence[Any] = (),
        record_to_memory: bool = True,
        inject_context: bool = True,
    ) -> TrellisHooks:
        """The hook implementations: context, tools, pause, compaction, run end."""
        return TrellisHooks(
            runtime,
            policy=self.harness.policy if self.harness.policy_enabled else None,
            skills=skills,
            record_to_memory=record_to_memory,
            inject_context=inject_context,
        )

    def options(
        self,
        *,
        hooks: TrellisHooks | None = None,
        gateway_url: str | None = None,
        require_token: bool = True,
        system_prompt: Any = None,
        allowed_tools: Sequence[str] = (),
        model: str | None = None,
        permission_callback: bool = True,
        **kwargs: Any,
    ) -> ClaudeAgentOptions:
        """``ClaudeAgentOptions`` with the harness's hooks and the gateway bound.

        ``env`` carries the endpoint and nothing else: the CLI inherits its credential from
        this process's environment, so no key is ever copied into an options object that
        gets logged or traced.
        """
        bound = hooks or self.hooks()
        env = {
            **gateway_env(
                gateway_url or os.environ.get(GATEWAY_URL_VAR), require_token=require_token
            ),
            **dict(kwargs.pop("env", {}) or {}),
        }
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=list(allowed_tools),
            model=model or self.harness.config.models.default_model,
            env=env,
            hooks=self.hook_matchers(bound),
            can_use_tool=bound.can_use_tool if permission_callback else None,
            **kwargs,
        )

    @staticmethod
    def hook_matchers(hooks: TrellisHooks) -> dict[HookEvent, list[HookMatcher]]:
        """The adapter's hooks in the SDK's ``{event: [HookMatcher]}`` shape."""
        matchers: dict[HookEvent, list[HookMatcher]] = {
            "UserPromptSubmit": [HookMatcher(hooks=[hooks.user_prompt_submit])],
            "PreToolUse": [HookMatcher(hooks=[hooks.pre_tool_use])],
            "PostToolUse": [HookMatcher(hooks=[hooks.post_tool_use])],
            "PreCompact": [HookMatcher(hooks=[hooks.pre_compact])],
            "Stop": [HookMatcher(hooks=[hooks.stop])],
            "SubagentStop": [HookMatcher(hooks=[hooks.subagent_stop])],
        }
        return matchers

    # ------------------------------------------------------------------ wrapping
    def wrap(
        self,
        options: ClaudeAgentOptions,
        *,
        agent_id: str,
        hooks: TrellisHooks | None = None,
        skills: list[Any] | None = None,
        **wrap_options: Any,
    ) -> Callable[..., Any]:
        """Run a Claude Agent SDK query through the harness pipeline.

        The wrapped callable takes the prompt and returns the agent's final text. A tool the
        policy held for a person stops the CLI (``PermissionResultDeny(interrupt=True)``) and
        the pause is re-raised here, so the run store, the event stream and the AG-UI surface
        see the same ``Interrupt`` a Deep Agents or OpenAI Agents pause produces.
        """
        bound = hooks or self.hooks()

        async def target(payload: Any, runtime: Any) -> Any:
            prompt = payload if isinstance(payload, str) else str(payload)
            await runtime.events.emit(RunEventType.STEP_STARTED, step=STEP, framework=FRAMEWORK)
            text = await self._query(prompt, options, bound, runtime)
            paused = runtime.state.get(PENDING_KEY)
            if paused is not None:
                raise paused
            return text

        return self.harness.wrap(
            target,
            agent_id=agent_id,
            skills=skills,
            framework=FRAMEWORK,
            framework_version=self.version,
            **wrap_options,
        )

    def agent(
        self,
        *,
        agent_id: str,
        system_prompt: Any = None,
        allowed_tools: Sequence[str] = (),
        model: str | None = None,
        gateway_url: str | None = None,
        require_token: bool = True,
        skills: list[Any] | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Build the options with the harness bound, and wrap them."""
        hooks = self.hooks(skills=skills or ())
        options = self.options(
            hooks=hooks,
            gateway_url=gateway_url,
            require_token=require_token,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            model=model,
            **kwargs,
        )
        return self.wrap(options, agent_id=agent_id, hooks=hooks, skills=skills)

    # ------------------------------------------------------------------ the query itself
    async def _query(
        self, prompt: str, options: ClaudeAgentOptions, hooks: TrellisHooks, runtime: Any
    ) -> str | None:
        """Drive one CLI query, putting its text on the harness's event stream as it arrives."""
        from claude_agent_sdk import (  # noqa: PLC0415 - the adapter's own import
            AssistantMessage,
            ResultMessage,
            TextBlock,
            query,
        )

        answer: str | None = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                text = "".join(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
                if text.strip():
                    hooks.remember_said("assistant", text)
                    await self._announce(runtime, text, message.message_id)
                    answer = text
            elif isinstance(message, ResultMessage):
                runtime.record_model_call(
                    {
                        "model": options.model,
                        "provider": FRAMEWORK,
                        "latency_ms": message.duration_api_ms,
                        "cost_usd": message.total_cost_usd,
                        "turns": message.num_turns,
                    }
                )
                if message.result:
                    answer = message.result
        return answer

    @staticmethod
    async def _announce(runtime: Any, text: str, message_id: str | None) -> None:
        """One assistant message, as the three text events every surface expects."""
        identifier = message_id or f"msg_{runtime.events.sequence}"
        await runtime.events.emit(RunEventType.TEXT_MESSAGE_START, message_id=identifier)
        await runtime.events.emit(
            RunEventType.TEXT_MESSAGE_CONTENT, message_id=identifier, delta=text
        )
        await runtime.events.emit(RunEventType.TEXT_MESSAGE_END, message_id=identifier)
