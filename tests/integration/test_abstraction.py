"""What the harness should handle so the developer never has to.

Each test here is a thing a user previously had to know, remember or repeat. They exist to
keep that knowledge inside the harness: if one of these fails, a burden has leaked back out
to the application.
"""

from __future__ import annotations

import pytest

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    MemoryPolicy,
    current_context,
)
from universal_agent_harness.contracts.errors import ConfigurationError

# --------------------------------------------------------------------------- identity


async def test_identity_declared_once_flows_into_every_execution(memory):
    """`defaults` is the place you say who you are. A context built by the application
    should not have to repeat it."""
    harness = AgentHarness(
        memory=memory,
        defaults={
            "tenant_id": "acme",
            "workspace_id": "ws-1",
            "agent_group_id": "supply-chain",
            "user_id": "u-1",
        },
        config={"memory": {"writeback": False}},
    )
    # a context that knows nothing but the conversation
    bare = AgentExecutionContext.create(tenant_id="acme", agent_id="inv", thread_id="chat-1")
    assert bare.agent_group_id is None and bare.workspace_id is None

    seen = {}

    async def agent(_payload, runtime):
        seen["context"] = runtime.context
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=bare)

    ctx = seen["context"]
    assert ctx.agent_group_id == "supply-chain"
    assert ctx.workspace_id == "ws-1"
    assert ctx.user_id == "u-1"


async def test_the_callers_own_values_are_never_overwritten(memory):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme", "workspace_id": "default-ws"},
        config={"memory": {"writeback": False}},
    )
    explicit = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", workspace_id="caller-ws"
    )
    seen = {}

    async def agent(_payload, runtime):
        seen["ws"] = runtime.context.workspace_id
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=explicit)
    assert seen["ws"] == "caller-ws"


async def test_execution_identity_is_never_back_filled(memory):
    """Ids that identify *this* run must not be inherited from defaults."""
    from universal_agent_harness.execution.context_factory import FILLABLE_FIELDS

    for field in ("agent_run_id", "request_id", "trace_id", "turn_id", "thread_id", "task_id"):
        assert field not in FILLABLE_FIELDS


# --------------------------------------------------------------------------- sharing


async def test_the_agent_group_can_be_declared_per_agent(memory, context):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    bare = AgentExecutionContext.create(tenant_id="acme", agent_id="inv", thread_id="t1")
    seen = {}

    async def agent(_payload, runtime):
        seen["group"] = runtime.context.agent_group_id
        await runtime.memory.share("a fact for the crew")
        return "ok"

    await harness.wrap(agent, agent_id="inv", agent_group="crew-7")(None, context=bare)
    assert seen["group"] == "crew-7"
    shared = [o for o in memory.observations if o["content"] == "a fact for the crew"]
    assert shared and shared[0]["hints"]["visibility"] == "AGENT_GROUP"
    assert shared[0]["scope"]["agent_group_id"] == "crew-7"


async def test_a_group_can_be_given_for_a_single_share(memory, context):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    bare = AgentExecutionContext.create(tenant_id="acme", agent_id="inv", thread_id="t1")

    async def agent(_payload, runtime):
        await runtime.memory.share("a fact", group="ad-hoc-crew")
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=bare)
    shared = [o for o in memory.observations if o["content"] == "a fact"]
    assert shared and shared[0]["scope"]["agent_group_id"] == "ad-hoc-crew"


async def test_sharing_without_any_group_says_exactly_how_to_fix_it(memory):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    bare = AgentExecutionContext.create(tenant_id="acme", agent_id="inv")

    async def agent(_payload, runtime):
        with pytest.raises(ConfigurationError) as exc:
            await runtime.memory.share("nobody to share with")
        message = str(exc.value)
        # the error carries all three ways to supply it
        assert "defaults=" in message and "agent_group=" in message and "group=" in message
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=bare)


# --------------------------------------------------------------------------- ingestion


async def test_ingesting_into_a_thread_creates_the_thread_first(memory, context, tmp_path):
    """A thread-visible document is readable only by thread participants, and a thread only
    grants that once it exists. The harness creates it rather than producing chunks nobody
    can retrieve."""
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    doc = tmp_path / "policy.txt"
    doc.write_text("reorder policy")

    async def agent(_payload, runtime):
        await runtime.memory.add_document(doc, title="Policy")
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)

    names = [name for name, _ in memory.calls]
    assert names.index("chat.create") < names.index("files.add"), (
        "the thread must be created before the document is ingested"
    )


async def test_a_document_for_a_wider_audience_does_not_touch_the_thread(
    memory, context, tmp_path
):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    doc = tmp_path / "policy.txt"
    doc.write_text("reorder policy")

    async def agent(_payload, runtime):
        await runtime.memory.add_document(doc, visibility="TENANT")
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert "chat.create" not in [name for name, _ in memory.calls]


# --------------------------------------------------------------------------- configuration


def test_a_misspelled_memory_policy_option_is_refused():
    with pytest.raises(ConfigurationError, match="unknown memory policy option"):
        MemoryPolicy().merged({"observe_outputs": False})


def test_a_correct_policy_option_changes_only_that_field():
    narrowed = MemoryPolicy().merged({"observe_output": False})
    assert narrowed.observe_output is False
    assert narrowed.observe_input is True   # everything else is untouched


async def test_nothing_but_a_tenant_is_required_to_run_an_agent():
    harness = AgentHarness(defaults={"tenant_id": "acme"})

    async def agent(question):
        return f"answer: {question}"

    result = await harness.wrap(agent, agent_id="inv")("how much stock?")
    assert result.succeeded
    assert current_context() is None      # and nothing leaks out of the execution
