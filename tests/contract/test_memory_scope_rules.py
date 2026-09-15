"""The Memory Service's scope coherence rules, encoded as harness tests.

The service validates every request's execution context and rejects the whole call if the
ids do not hang together:

    agent_run_id requires agent_id
    session_id   requires thread_id
    turn_id      requires session_id

A mocked transport accepts anything, so these rules have to be asserted here or they are
only discovered against a live service — which is exactly how the missing ``session_id``
was found. Each test below states one rule from the service's own validator
(``memory_service/domain/context.py``).
"""

from __future__ import annotations

import pytest

from universal_agent_harness import AgentExecutionContext


def emitted(context: AgentExecutionContext) -> dict:
    return context.scope_fields()


# --------------------------------------------------------------------------- rule 1


def test_agent_run_id_is_always_accompanied_by_an_agent_id():
    scope = emitted(AgentExecutionContext.create(tenant_id="acme", agent_id="inv"))
    assert scope["agent_run_id"] and scope["agent_id"]


def test_a_child_run_also_carries_its_agent_id():
    parent = AgentExecutionContext.create(tenant_id="acme", agent_id="planner")
    scope = emitted(parent.for_agent("inventory"))
    assert scope["agent_id"] == "inventory"
    assert scope["parent_agent_run_id"] == parent.agent_run_id


# --------------------------------------------------------------------------- rule 2


def test_session_id_is_never_emitted_without_a_thread():
    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", session_id="orphan-session"
    )
    assert "session_id" not in emitted(context)


# --------------------------------------------------------------------------- rule 3


def test_turn_id_gets_a_session_derived_from_the_thread():
    """An application that knows only "turn 3 of chat-42" must not have to invent a
    session: one session per thread is derived for it."""
    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", thread_id="chat-42", turn_id="turn-3"
    )
    assert context.session_id == "chat-42-session"
    scope = emitted(context)
    assert scope["session_id"] == "chat-42-session" and scope["turn_id"] == "turn-3"


def test_an_explicit_session_is_never_overwritten():
    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", thread_id="chat-42",
        session_id="sess-9", turn_id="turn-3",
    )
    assert emitted(context)["session_id"] == "sess-9"


def test_turn_id_without_a_thread_is_dropped_rather_than_refused():
    context = AgentExecutionContext.create(tenant_id="acme", agent_id="inv", turn_id="turn-3")
    scope = emitted(context)
    assert "turn_id" not in scope and "session_id" not in scope


def test_derivation_survives_copying_and_child_contexts():
    parent = AgentExecutionContext.create(
        tenant_id="acme", agent_id="planner", thread_id="chat-42", turn_id="turn-3"
    )
    child = parent.for_agent("inventory")
    assert emitted(child)["session_id"] == "chat-42-session"

    changed = parent.with_fields(turn_id="turn-4")
    assert emitted(changed)["session_id"] == "chat-42-session"


# --------------------------------------------------------------------------- the real model


@pytest.mark.parametrize(
    "fields",
    [
        {"thread_id": "chat-42", "turn_id": "turn-3"},
        {"thread_id": "chat-42"},
        {"turn_id": "turn-3"},
        {"session_id": "sess-1"},
        {"thread_id": "chat-42", "session_id": "sess-1", "turn_id": "turn-3"},
        {},
    ],
)
def test_every_emitted_scope_satisfies_all_three_rules(fields):
    """Property form: whatever an application supplies, what reaches the service is legal."""
    scope = emitted(AgentExecutionContext.create(tenant_id="acme", agent_id="inv", **fields))
    if "agent_run_id" in scope:
        assert "agent_id" in scope
    if "session_id" in scope:
        assert "thread_id" in scope
    if "turn_id" in scope:
        assert "session_id" in scope


def test_the_sdk_scope_model_accepts_what_we_emit():
    """The SDK's own Scope model is the nearest thing to the service's validator."""
    from universal_memory.models import Scope

    context = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", thread_id="chat-42", turn_id="turn-3",
        user_id="u1", work_id="wo-1",
    )
    scope = Scope(**context.scope_fields())
    assert scope.session_id == "chat-42-session"
    assert scope.turn_id == "turn-3"


# --------------------------------------------------------------------------- vocabulary


def test_observation_kinds_match_the_services_enum():
    """These are the service's ``ObservationKind`` values. A kind outside this set is a 422
    from the service — which a mocked transport happily accepts, so it is asserted here."""
    from universal_agent_harness import OBSERVATION_KINDS

    assert {
        "MESSAGE", "FILE", "AGENT_RESULT", "TOOL_RESULT",
        "DECISION", "FEEDBACK", "EVENT", "IMPORT",
    } == OBSERVATION_KINDS


def test_an_unknown_observation_kind_fails_before_the_wire():
    from universal_agent_harness import MemoryObservation

    with pytest.raises(ValueError, match="unknown observation kind"):
        MemoryObservation(content="x", kind="CLAIM")


async def test_the_automatic_writeback_only_uses_valid_kinds(harness, memory, context):
    """The harness's own observations must be in the vocabulary — this is the regression
    test for input/claim observations being written with invented kinds."""
    from universal_agent_harness import OBSERVATION_KINDS, AgentResult, Claim

    async def agent(payload):
        return AgentResult.ok(
            "an answer", claims=[Claim(claim_id="c1", text="a claim")]
        )

    await harness.wrap(agent, agent_id="inv")("a question", context=context)

    kinds = {o["kind"] for o in memory.observations}
    assert kinds, "the automatic path should have written something"
    assert kinds <= OBSERVATION_KINDS, f"invented kinds: {sorted(kinds - OBSERVATION_KINDS)}"
