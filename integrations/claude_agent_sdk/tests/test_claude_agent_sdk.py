"""Claude Agent SDK adapter: the six moments of design §8, against the installed SDK.

Nothing here spawns the ``claude`` CLI. Every binding this adapter owns is a plain async
callable over dictionaries — the payloads the CLI sends — so the hooks are driven directly,
which is both faster and a more exact test than watching a subprocess: the assertion is on
the JSON the CLI would receive back.

What that cannot cover is the CLI executing a tool, so the tool-call assertions are about the
decision and the records, not about a tool function running. The adapter README says so.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny
from trellis.contracts import (
    AgentExecutionContext,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ModelResponse,
    ModelUsage,
    PolicyDeniedError,
    RunEventType,
    RunOutcome,
)
from trellis.contracts.errors import ConfigurationError

from trellis.harness import AgentHarness, CallablePolicyProvider, CollectingEventSink
from trellis.harness.interrupts import ApprovalRequired, interrupt_from_signal
from trellis.harness_claude_agent_sdk import (
    BASE_URL_VAR,
    TOKEN_VAR,
    ClaudeAgentSDKHarness,
    TrellisHooks,
    Verdict,
    anthropic_base_url,
    claude_agent_sdk_version,
    gateway_env,
)

TENANT = "acme"
GATEWAY = "http://localhost:8091"


class Answers:
    """The harness model port, answering with one line."""

    def __init__(self, text: str = "a summary") -> None:
        self.text = text
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text=self.text, model="scripted", usage=ModelUsage(input_tokens=3, output_tokens=2)
        )

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


def turn(agent_id: str, thread_id: str) -> AgentExecutionContext:
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        user_id="u1",
        agent_id=agent_id,
        thread_id=thread_id,
        turn_id=f"trn_{thread_id}",
    )


def build(
    *,
    model: Any = None,
    policy: Any = None,
    sink: CollectingEventSink | None = None,
    memory: Any = None,
) -> AgentHarness:
    return AgentHarness(
        memory=memory,
        model=model or Answers(),
        policy=policy,
        event_sinks=[sink] if sink is not None else (),
        defaults={"tenant_id": TENANT},
        config={"telemetry": {"capture": {"inputs": True, "outputs": True}}},
    )


@pytest.fixture(autouse=True)
def _gateway_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's credential lives in the environment; the adapter only checks it is there."""
    monkeypatch.setenv(TOKEN_VAR, "test-virtual-key")


def pre_tool_use(tool: str, args: dict[str, Any], call_id: str = "toolu_1") -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "s1",
        "transcript_path": "/dev/null",
        "cwd": ".",
        "tool_name": tool,
        "tool_input": args,
        "tool_use_id": call_id,
    }


def post_tool_use(
    tool: str, args: dict[str, Any], response: Any, call_id: str = "toolu_1"
) -> dict[str, Any]:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": "s1",
        "transcript_path": "/dev/null",
        "cwd": ".",
        "tool_name": tool,
        "tool_input": args,
        "tool_response": response,
        "tool_use_id": call_id,
    }


# --------------------------------------------------------------------------- the adapter
def test_the_adapter_reports_the_installed_version_and_its_availability() -> None:
    harness = build()
    assert harness.claude_agent_sdk.version == claude_agent_sdk_version()
    assert ClaudeAgentSDKHarness.available() is True
    assert harness.claude_agent_sdk is harness.claude_agent_sdk


def test_the_public_name_resolves_through_the_core() -> None:
    from trellis.harness import ClaudeAgentSDKHarness as exported

    assert exported is ClaudeAgentSDKHarness


# --------------------------------------------------------------------------- 2. model
def test_the_options_point_the_cli_at_the_gateways_anthropic_endpoint() -> None:
    """The SDK has no model seam: the binding is an environment variable, and this is it."""
    options = build().claude_agent_sdk.options(gateway_url=GATEWAY, allowed_tools=["Read"])
    assert isinstance(options, ClaudeAgentOptions)
    assert options.env[BASE_URL_VAR] == f"{GATEWAY}/anthropic"


def test_the_anthropic_prefix_is_not_applied_twice() -> None:
    assert anthropic_base_url(f"{GATEWAY}/anthropic") == f"{GATEWAY}/anthropic"
    assert anthropic_base_url(f"{GATEWAY}/") == f"{GATEWAY}/anthropic"


def test_no_credential_is_ever_copied_into_the_options() -> None:
    """The CLI inherits the key from the environment; nothing here holds or prints one."""
    options = build().claude_agent_sdk.options(gateway_url=GATEWAY)
    assert TOKEN_VAR not in options.env
    assert "test-virtual-key" not in repr(options)
    assert "test-virtual-key" not in repr(options.env)


def test_an_unconfigured_gateway_is_refused_rather_than_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BIFROST_URL", raising=False)
    with pytest.raises(ConfigurationError, match="needs the gateway's URL"):
        build().claude_agent_sdk.options()


def test_a_missing_credential_fails_at_build_time_not_as_a_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    with pytest.raises(ConfigurationError, match=TOKEN_VAR):
        gateway_env(GATEWAY)
    # and an operator who manages the credential another way can say so
    assert gateway_env(GATEWAY, require_token=False)[BASE_URL_VAR] == f"{GATEWAY}/anthropic"


def test_every_hook_the_adapter_binds_is_installed() -> None:
    options = build().claude_agent_sdk.options(gateway_url=GATEWAY)
    assert set(options.hooks) == {
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PreCompact",
        "Stop",
        "SubagentStop",
    }
    assert options.can_use_tool is not None


# --------------------------------------------------------------------------- 1. context
async def test_the_memory_bundle_comes_back_as_additional_context() -> None:
    harness = build()
    hooks = harness.claude_agent_sdk.hooks()
    async with harness.execution(turn("ctx", "thr-ctx"), agent_id="ctx") as runtime:
        runtime.memory_context = _bundle("The customer is on the Pro plan [mem_7]")
        output = await hooks.user_prompt_submit({"prompt": "how much stock?"})
    specific = output["hookSpecificOutput"]
    assert specific["hookEventName"] == "UserPromptSubmit"
    assert "The customer is on the Pro plan [mem_7]" in specific["additionalContext"]
    assert "cite memory ids" in specific["additionalContext"]


async def test_without_a_bundle_no_context_is_invented() -> None:
    harness = build()
    hooks = harness.claude_agent_sdk.hooks()
    async with harness.execution(turn("ctx", "thr-ctx2"), agent_id="ctx"):
        assert await hooks.user_prompt_submit({"prompt": "hi"}) == {}


# --------------------------------------------------------------------------- 3. tools
async def test_an_allowed_tool_is_allowed_and_opens_on_the_run_stream() -> None:
    sink = CollectingEventSink()
    harness = build(sink=sink)
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("tool-agent", "thr-tool")
    async with harness.execution(context, agent_id="tool-agent"):
        decision = await hooks.pre_tool_use(pre_tool_use("Read", {"file_path": "/a"}))
        assert decision["hookSpecificOutput"]["permissionDecision"] == "allow"
        await hooks.post_tool_use(post_tool_use("Read", {"file_path": "/a"}, {"content": "hi"}))
    types = sink.types(context.agent_run_id)
    assert [t for t in types if t.value.startswith("TOOL_CALL")] == [
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
    ]
    result = [
        e for e in sink.for_run(context.agent_run_id) if e.type is RunEventType.TOOL_CALL_RESULT
    ]
    assert result[-1].data["status"] == "ok"


async def test_a_denied_tool_is_refused_with_a_reason() -> None:
    sink = CollectingEventSink()

    async def deny(context: Any, call: Any) -> Any:
        return "Bash is not allowed here" if call.tool == "Bash" else True

    harness = build(policy=CallablePolicyProvider(tool=deny), sink=sink)
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("deny-agent", "thr-deny")
    async with harness.execution(context, agent_id="deny-agent"):
        with pytest.raises(PolicyDeniedError, match="Bash is not allowed here"):
            await hooks.pre_tool_use(pre_tool_use("Bash", {"command": "rm -rf /"}))


async def test_a_failing_tool_is_recorded_as_an_error() -> None:
    sink = CollectingEventSink()
    harness = build(sink=sink)
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("err-agent", "thr-err")
    async with harness.execution(context, agent_id="err-agent"):
        await hooks.pre_tool_use(pre_tool_use("Read", {"file_path": "/missing"}))
        await hooks.post_tool_use(
            post_tool_use("Read", {"file_path": "/missing"}, {"is_error": True, "error": "no"})
        )
    results = [
        e for e in sink.for_run(context.agent_run_id) if e.type is RunEventType.TOOL_CALL_RESULT
    ]
    assert results[-1].data["status"] == "error"


async def test_two_seams_arriving_together_still_make_one_decision() -> None:
    """The race the memo exists to stop: both seams reaching one call before either finished.

    Caching the *result* would leave the whole policy call between the check and the write, so
    each seam would consult the policy, open the call on the stream, and be able to pause the
    run. The memo holds the task, so concurrent callers await one decision.
    """
    consulted: list[str] = []
    released = asyncio.Event()

    async def slow(context: Any, call: Any) -> Any:
        consulted.append(call.tool)
        await released.wait()
        return True

    sink = CollectingEventSink()
    harness = build(policy=CallablePolicyProvider(tool=slow), sink=sink)
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("race-agent", "thr-race")
    async with harness.execution(context, agent_id="race-agent"):
        both = asyncio.gather(
            hooks.pre_tool_use(pre_tool_use("Read", {"file_path": "/a"})),
            hooks.can_use_tool("Read", {"file_path": "/a"}, _Context("toolu_1")),
        )
        await asyncio.sleep(0)  # let both reach the policy
        released.set()
        hook_output, callback = await both
    assert consulted == ["Read"], "the policy must be consulted once for one call"
    assert hook_output["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert isinstance(callback, PermissionResultAllow)
    starts = [
        e for e in sink.for_run(context.agent_run_id) if e.type is RunEventType.TOOL_CALL_START
    ]
    assert len(starts) == 1, "one call opens on the stream once"


async def test_the_verdict_is_an_enum_not_a_pair_of_flags() -> None:
    """Three answers, three values: "paused and rejected" is not expressible."""
    assert [v.value for v in Verdict] == ["allow", "rejected", "paused"]


async def test_the_permission_callback_and_the_hook_make_one_decision() -> None:
    """Both seams exist and a deployment may have both; the call is authorized once."""
    seen: list[str] = []

    async def count(context: Any, call: Any) -> Any:
        seen.append(call.tool)
        return True

    harness = build(policy=CallablePolicyProvider(tool=count))
    hooks = harness.claude_agent_sdk.hooks()
    async with harness.execution(turn("once-agent", "thr-once"), agent_id="once-agent"):
        await hooks.pre_tool_use(pre_tool_use("Read", {"file_path": "/a"}))
        allowed = await hooks.can_use_tool("Read", {"file_path": "/a"}, _Context("toolu_1"))
    assert isinstance(allowed, PermissionResultAllow)
    assert seen == ["Read"], "the policy must be consulted once for one call"


# --------------------------------------------------------------------------- 4. pause
async def test_an_approval_pauses_the_run_and_the_answer_continues_it() -> None:
    sink = CollectingEventSink()

    async def hold(context: Any, call: Any) -> Any:
        return "require_approval" if call.tool == "Bash" else True

    harness = build(policy=CallablePolicyProvider(tool=hold), sink=sink)
    context = turn("hitl-agent", "thr-hitl")

    hooks = harness.claude_agent_sdk.hooks()
    with pytest.raises(ApprovalRequired) as paused:
        async with harness.execution(context, agent_id="hitl-agent") as runtime:
            decision = await hooks.pre_tool_use(pre_tool_use("Bash", {"command": "ls"}))
            assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
            # the permission callback stops the CLI rather than letting it plan around it
            denied = await hooks.can_use_tool("Bash", {"command": "ls"}, _Context("toolu_1"))
            assert isinstance(denied, PermissionResultDeny) and denied.interrupt is True
            raise runtime.state["claude_agent_sdk.pending_approval"]
    assert paused.value.tool_call.tool == "Bash"

    interrupt = _paused(sink, context.agent_run_id)
    assert interrupt.reason is InterruptReason.APPROVAL
    assert interrupt.tool_call is not None and interrupt.tool_call.args == {"command": "ls"}

    # the person approves; the resumed run finds the decision and the same call is allowed
    await harness.resume(
        interrupt,
        InterruptResolution(
            interrupt_id=interrupt.interrupt_id,
            run_id=interrupt.run_id,
            decision=InterruptDecision.APPROVE,
            reviewer="u1",
        ),
        context=context,
    )
    # the resumed run is the same run id, so the runtime the harness builds already carries
    # the approver's decision: nothing in the test has to hand it over
    resumed = harness.claude_agent_sdk.hooks()
    async with harness.execution(context, agent_id="hitl-agent") as runtime:
        allowed = await resumed.pre_tool_use(pre_tool_use("Bash", {"command": "ls"}))
        assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "claude_agent_sdk.pending_approval" not in runtime.state


async def test_an_approved_edit_rewrites_the_tool_input() -> None:
    """``PreToolUse`` can rewrite the arguments, so an approver may narrow a call."""

    async def hold(context: Any, call: Any) -> Any:
        if call.tool != "Bash":
            return True
        return True if call.args.get("command") == "ls -l" else "require_approval"

    harness = build(policy=CallablePolicyProvider(tool=hold))
    context = turn("edit-agent", "thr-edit")
    hooks = harness.claude_agent_sdk.hooks()
    async with harness.execution(context, agent_id="edit-agent") as runtime:
        await hooks.pre_tool_use(pre_tool_use("Bash", {"command": "ls"}))
        pending = runtime.state["claude_agent_sdk.pending_approval"]
    # the same translation the coordinator does when the pause leaves a wrapped agent
    interrupt = interrupt_from_signal(pending, context)
    harness.resolutions.record(
        interrupt,
        InterruptResolution(
            interrupt_id=interrupt.interrupt_id,
            run_id=interrupt.run_id,
            decision=InterruptDecision.EDIT,
            payload={"command": "ls -l"},
            reviewer="u1",
        ),
    )
    resumed = harness.claude_agent_sdk.hooks()
    async with harness.execution(context, agent_id="edit-agent"):
        output = await resumed.pre_tool_use(pre_tool_use("Bash", {"command": "ls"}))
    specific = output["hookSpecificOutput"]
    assert specific["permissionDecision"] == "allow"
    assert specific["updatedInput"] == {"command": "ls -l"}


# --------------------------------------------------------------------------- 5. run end
async def test_the_answer_is_written_to_memory_on_stop(harness: Any, memory: Any) -> None:
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("end-agent", "thr-end")
    async with harness.execution(context, agent_id="end-agent"):
        hooks.remember_said("assistant", "SKU-1 has 4 in stock")
        await hooks.stop({"hook_event_name": "Stop", "stop_hook_active": False})
    await harness.drain()
    assert any("SKU-1 has 4 in stock" in str(o["content"]) for o in memory.observations)


async def test_stop_closes_the_step_on_the_run_stream() -> None:
    sink = CollectingEventSink()
    harness = build(sink=sink)
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("step-agent", "thr-step")
    async with harness.execution(context, agent_id="step-agent"):
        await hooks.stop({"hook_event_name": "Stop", "stop_hook_active": False})
        await hooks.subagent_stop({"hook_event_name": "SubagentStop", "agent_type": "explore"})
    types = sink.types(context.agent_run_id)
    assert types[0] is RunEventType.RUN_STARTED
    assert types.count(RunEventType.STEP_FINISHED) == 2
    assert types[-1] is RunEventType.RUN_FINISHED
    finished = [
        e for e in sink.for_run(context.agent_run_id) if e.type is RunEventType.RUN_FINISHED
    ]
    assert finished[-1].outcome is RunOutcome.SUCCESS


# --------------------------------------------------------------------------- 6. compaction
async def test_pre_compact_writes_a_run_scoped_summary(harness: Any, memory: Any) -> None:
    """``PreCompact`` carries no summary, so the adapter makes one and says it did."""
    model = Answers("the customer asked about stock and was told there were four")
    harness.model_client = harness.wrap_model(model)
    harness.runtime_builder.model_client = harness.model_client
    hooks = harness.claude_agent_sdk.hooks()
    context = turn("compact-agent", "thr-compact")
    async with harness.execution(context, agent_id="compact-agent"):
        hooks.remember_said("user", "how much stock of SKU-1?")
        hooks.remember_said("assistant", "4 in stock")
        await hooks.pre_compact(
            {"hook_event_name": "PreCompact", "trigger": "auto", "custom_instructions": None}
        )
    await harness.drain()
    written = [o for o in memory.observations if "Conversation summary" in str(o["content"])]
    assert written and written[-1]["kind"] == "AGENT_RESULT"
    assert written[-1]["hints"]["visibility"] == "RUN"
    assert written[-1]["framework"] == "claude-agent-sdk"
    assert written[-1]["trigger"] == "auto"


async def test_compaction_with_nothing_said_writes_nothing() -> None:
    harness = build()
    hooks = harness.claude_agent_sdk.hooks()
    async with harness.execution(turn("quiet", "thr-quiet"), agent_id="quiet"):
        assert await hooks.pre_compact({"trigger": "manual"}) == {}


# --------------------------------------------------------------------------- guards
async def test_the_hooks_refuse_to_pretend_outside_a_run() -> None:
    with pytest.raises(RuntimeError, match=r"harness\.claude_agent_sdk"):
        _ = TrellisHooks().runtime


def test_the_gateway_url_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIFROST_URL", GATEWAY)
    options = build().claude_agent_sdk.options()
    assert options.env[BASE_URL_VAR] == f"{GATEWAY}/anthropic"
    assert os.environ["BIFROST_URL"] == GATEWAY


# --------------------------------------------------------------------------- helpers
def _bundle(rendered: str) -> Any:
    class Bundle:
        def __init__(self, text: str) -> None:
            self.rendered = text

    return Bundle(rendered)


def _paused(sink: CollectingEventSink, run_id: str) -> Interrupt:
    finished = [e for e in sink.for_run(run_id) if e.type is RunEventType.RUN_FINISHED]
    assert finished and finished[-1].outcome is RunOutcome.INTERRUPT
    return Interrupt.model_validate(finished[-1].data["interrupt"])


class _Context:
    """The ``ToolPermissionContext`` field the adapter reads."""

    def __init__(self, tool_use_id: str) -> None:
        self.tool_use_id = tool_use_id
