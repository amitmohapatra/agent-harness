"""The Claude Agent SDK through the harness: the six moments, without spawning the CLI.

This example is deliberately different from the other two. The Claude Agent SDK does not call
a model — it spawns the ``claude`` CLI, which does — so the parts a harness owns are all
hooks, and the honest way to show them is to call them with the payloads the CLI sends. That
needs no CLI, no network and no key.

To run a real query instead, install the ``claude`` CLI, export the gateway's virtual key as
``ANTHROPIC_AUTH_TOKEN``, and call the wrapped agent at the bottom of this file.

    pip install "trellis-harness[claude-agent-sdk]"
    python examples/claude_agent_sdk_agent.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from trellis.contracts import (
    AgentExecutionContext,
    InterruptDecision,
    InterruptResolution,
    ModelResponse,
    RunEventType,
)

from trellis.harness import AgentHarness, CallablePolicyProvider, CollectingEventSink
from trellis.harness.interrupts import interrupt_from_signal
from trellis.harness_claude_agent_sdk import BASE_URL_VAR, TOKEN_VAR

GATEWAY = os.environ.get("BIFROST_URL", "http://localhost:8091")


class ScriptedModel:
    """The harness model port. Here it only ever writes the compaction summary."""

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        return ModelResponse(
            text="The user asked about SKU-1 and was told there are three on hand.",
            model="scripted",
        )

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


def pre_tool_use(tool: str, args: dict[str, Any], call_id: str = "toolu_1") -> dict[str, Any]:
    """The payload the CLI sends before it runs a tool."""
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "s1",
        "transcript_path": "/dev/null",
        "cwd": ".",
        "tool_name": tool,
        "tool_input": args,
        "tool_use_id": call_id,
    }


async def main() -> None:
    # The CLI inherits its credential from this process; the harness never copies it into the
    # options, and never reads its value. Set to an obvious non-secret here only so the
    # example runs end to end without a real key: nothing in this example calls the CLI.
    os.environ.setdefault(TOKEN_VAR, "not-a-real-key-this-example-never-calls-the-cli")

    sink = CollectingEventSink()

    async def hold_writes(_context: Any, call: Any) -> Any:
        """Reading a file is fine; running a shell command waits for a person."""
        return True if call.tool in ("Read", "Grep") else "require_approval"

    harness = AgentHarness(
        model=ScriptedModel(),
        policy=CallablePolicyProvider(tool=hold_writes),
        event_sinks=[sink],
        defaults={"tenant_id": "acme", "user_id": "u1"},
    )
    context = AgentExecutionContext.create(
        tenant_id="acme",
        user_id="u1",
        agent_id="reviewer",
        thread_id="chat-1",
        turn_id="trn_chat-1",
    )

    # -- the model moment: an environment variable, not a client --------------------------
    options = harness.claude_agent_sdk.options(
        gateway_url=GATEWAY,
        system_prompt="You review pull requests.",
        allowed_tools=["Read", "Grep"],
    )
    print("\n--- the model moment ---")
    print(f"  {BASE_URL_VAR} = {options.env[BASE_URL_VAR]}")
    print(f"  hooks installed: {sorted(options.hooks)}")
    print(f"  the credential is NOT in the options: {TOKEN_VAR not in options.env}")

    hooks = harness.claude_agent_sdk.hooks()
    async with harness.execution(context, agent_id="reviewer") as runtime:
        # -- context: the bundle comes back as additionalContext -------------------------
        runtime.memory_context = _bundle("The reviewer prefers short summaries [mem_4]")
        submitted = await hooks.user_prompt_submit({"prompt": "what changed in src/?"})
        print("\n--- context ---")
        print(f"  additionalContext: {submitted['hookSpecificOutput']['additionalContext']!r}")

        # -- tools: allowed, then recorded ------------------------------------------------
        allowed = await hooks.pre_tool_use(pre_tool_use("Read", {"file_path": "src/app.py"}))
        print("\n--- tools ---")
        print(f"  Read: {allowed['hookSpecificOutput']['permissionDecision']}")
        await hooks.post_tool_use(
            {
                **pre_tool_use("Read", {"file_path": "src/app.py"}),
                "hook_event_name": "PostToolUse",
                "tool_response": {"content": "…"},
            }
        )

        # -- pause: a tool the policy holds for a person ----------------------------------
        held = await hooks.pre_tool_use(pre_tool_use("Bash", {"command": "ls"}, "toolu_2"))
        print(
            f"  Bash: {held['hookSpecificOutput']['permissionDecision']} "
            f"({held['hookSpecificOutput']['permissionDecisionReason']})"
        )
        stopped = await hooks.can_use_tool("Bash", {"command": "ls"}, _Context("toolu_2"))
        print(f"  the permission callback stops the run: interrupt={stopped.interrupt}")
        pending = runtime.state["claude_agent_sdk.pending_approval"]

        # -- compaction: the adapter's own summary, remembered run-scoped -----------------
        hooks.remember_said("assistant", "There are three on hand.")
        await hooks.pre_compact({"hook_event_name": "PreCompact", "trigger": "auto"})
        print("\n--- compaction ---")
        print("  PreCompact carries no summary, so the adapter wrote its own to memory")

        # -- run end ----------------------------------------------------------------------
        await hooks.stop({"hook_event_name": "Stop", "stop_hook_active": False})

    interrupt = interrupt_from_signal(pending, context)
    print("\n--- the pause, as the platform sees it ---")
    # Driving the hooks by hand, this example lets the block finish, so its run is recorded
    # SUCCESS. Through ``harness.claude_agent_sdk.wrap(...)`` the pause is re-raised and the
    # run finishes with outcome ``interrupt``, exactly like the other two adapters.
    print(f"  {interrupt.reason.value}: {interrupt.question}")
    print(f"  the call: {interrupt.tool_call.tool} {interrupt.tool_call.args}")

    resolution = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=InterruptDecision.EDIT,
        payload={"command": "ls -l"},
        reviewer="u1",
    )
    harness.resolutions.record(interrupt, resolution)
    resumed = harness.claude_agent_sdk.hooks()
    async with harness.execution(context, agent_id="reviewer"):
        output = await resumed.pre_tool_use(pre_tool_use("Bash", {"command": "ls"}, "toolu_2"))
    specific = output["hookSpecificOutput"]
    print(
        f"  the approver narrowed it: {specific['permissionDecision']} "
        f"{specific.get('updatedInput')}"
    )

    print("\n--- what a UI saw ---")
    for event in sink.for_run(context.agent_run_id):
        print(
            f"  {event.sequence:>2}  {event.type.value:<20} "
            f"{event.tool_call_id or event.step or ''}"
        )
    assert any(e.type is RunEventType.TOOL_CALL_RESULT for e in sink.for_run(context.agent_run_id))

    await harness.aclose()


def _bundle(rendered: str) -> Any:
    class Bundle:
        def __init__(self, text: str) -> None:
            self.rendered = text

    return Bundle(rendered)


class _Context:
    """The one ``ToolPermissionContext`` field the adapter reads."""

    def __init__(self, tool_use_id: str) -> None:
        self.tool_use_id = tool_use_id


if __name__ == "__main__":
    asyncio.run(main())
