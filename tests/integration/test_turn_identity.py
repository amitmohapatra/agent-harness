"""A context without a turn is a conversation, not a turn.

Reusing one context for two messages in a chat gave both the same ``agent_run_id``: the
second message's outcome overwrote the first's (the service upserts on tenant+run), both
tool trajectories merged into one run, and memory had no turn boundary to hang "what did I
just ask you?" on. Each invocation now derives its own turn — unless the caller named one,
which is how a replayed LangGraph superstep keeps its identity.
"""

from __future__ import annotations

from universal_agent_harness import AgentExecutionContext, AgentHarness


async def _ids(harness, ctx, text):
    seen = {}

    async def agent(payload, runtime):
        seen.update(
            turn=runtime.context.turn_id,
            session=runtime.context.session_id,
            run=runtime.context.agent_run_id,
            thread=runtime.context.thread_id,
        )
        return "ok"

    await harness.wrap(agent, agent_id="support")(text, context=ctx)
    return dict(seen)


async def test_two_messages_in_one_chat_are_two_turns(memory):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    chat = AgentExecutionContext.create(
        tenant_id="acme", user_id="u1", agent_id="support", thread_id="chat-A"
    )
    first = await _ids(harness, chat, "my order is late")
    second = await _ids(harness, chat, "what did I just ask?")

    assert first["thread"] == second["thread"] == "chat-A"
    assert first["session"] == second["session"], "one sitting, one session"
    assert first["turn"] != second["turn"], "two messages are two turns"
    assert first["run"] != second["run"], "...and therefore two runs"
    await harness.aclose()


async def test_a_named_turn_is_honoured(memory):
    """Replay: the caller says these are the same turn, so they are."""
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    ctx = AgentExecutionContext.create(
        tenant_id="acme",
        user_id="u1",
        agent_id="support",
        thread_id="chat-B",
        turn_id="turn-fixed",
    )
    first = await _ids(harness, ctx, "q")
    second = await _ids(harness, ctx, "q")
    assert first["turn"] == second["turn"] == "turn-fixed"
    assert first["run"] == second["run"], "a replayed turn is the same run"
    await harness.aclose()


async def test_separate_chats_are_separate_sessions(memory):
    harness = AgentHarness(
        memory=memory, defaults={"tenant_id": "acme"}, config={"memory": {"writeback": False}}
    )
    a = await _ids(
        harness,
        AgentExecutionContext.create(
            tenant_id="acme", user_id="u1", agent_id="support", thread_id="chat-C"
        ),
        "q",
    )
    b = await _ids(
        harness,
        AgentExecutionContext.create(
            tenant_id="acme", user_id="u1", agent_id="support", thread_id="chat-D"
        ),
        "q",
    )
    assert a["session"] != b["session"] and a["run"] != b["run"]
    await harness.aclose()
