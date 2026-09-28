"""Every way a run can pause becomes one Interrupt; every policy answer has one meaning;
the registry keeps pauses and answers per tenant and run."""

from __future__ import annotations

import pytest
from trellis.contracts import (
    AgentExecutionContext,
    AgentPaused,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ToolCall,
)

from trellis.harness.interrupts import (
    ANSWER,
    ApprovalRequired,
    ResolutionRegistry,
    interrupt_from_signal,
)
from trellis.harness.policy.outcome import PolicyOutcome, normalize

CTX = AgentExecutionContext.create(
    tenant_id="acme", user_id="u1", agent_id="ref", thread_id="thr_1", workspace_id="ws1"
)


def test_agent_paused_and_approvals_become_interrupts() -> None:
    plain = interrupt_from_signal(AgentPaused("Which region?", expects={"type": "string"}), CTX)
    assert plain.question == "Which region?" and plain.reason is InterruptReason.QUESTION
    assert plain.run_id == CTX.agent_run_id and plain.tenant_id == "acme"
    assert plain.expects == {"type": "string"}
    call = ToolCall(tool="billing.refund", args={"amount": 240}, idempotency_key="k1")
    approval = interrupt_from_signal(ApprovalRequired(call, reason="over the limit"), CTX)
    assert approval.reason is InterruptReason.APPROVAL and approval.tool_call == call
    assert "billing.refund" in approval.question and "over the limit" in approval.question
    assert approval.payload == {"reason": "over the limit"}  # the call itself is tool_call


def test_a_framework_interrupt_is_read_structurally() -> None:
    """LangGraph puts interrupt(value) in the exception args as objects with .value. Read
    by shape, not by importing langgraph — which this package must not depend on."""

    class Value:
        def __init__(self, value):
            self.value = value

    class GraphInterrupt(Exception):
        pass

    one = interrupt_from_signal(GraphInterrupt(Value("Ship it?")), CTX)
    assert one.question == "Ship it?" and one.payload is None
    structured = interrupt_from_signal(
        GraphInterrupt(Value({"question": "Ship it?", "order": 9})), CTX
    )
    assert structured.question == "GraphInterrupt" and structured.payload == {
        "question": {"question": "Ship it?", "order": 9}
    }
    several = interrupt_from_signal(GraphInterrupt(Value("first?"), Value("second?")), CTX)
    assert several.payload == {"question": ["first?", "second?"]}
    bare = interrupt_from_signal(GraphInterrupt("suspended"), CTX)
    assert bare.question == "GraphInterrupt" and bare.payload is None


def test_a_signal_that_says_nothing_or_misbehaves_still_pauses_cleanly() -> None:
    """A pause is already the delicate path — the run is suspended and waiting on a person.
    Reading the question must never turn it into a crash, and must not invent one."""

    class NodeInterrupt(Exception):
        pass

    class Hostile(Exception):
        def awaiting(self):
            raise RuntimeError("no")

    class Odd(Exception):
        def awaiting(self):
            return "not a dict"

    class Said(Exception):
        def awaiting(self):
            return {"question": "Proceed?", "expects": {"type": "boolean"}}

    for signal in (NodeInterrupt(), Hostile(), Odd()):
        interrupt = interrupt_from_signal(signal, CTX)
        assert interrupt.question == type(signal).__name__ and interrupt.payload is None
    assert interrupt_from_signal(Said(), CTX).question == "Proceed?"


@pytest.mark.parametrize(
    ("decision", "outcome", "reason"),
    [
        (True, PolicyOutcome.ALLOW, None),
        (False, PolicyOutcome.DENY, None),
        (None, PolicyOutcome.DENY, None),
        ("tool is not allowed", PolicyOutcome.DENY, "tool is not allowed"),
        ("require_approval", PolicyOutcome.REQUIRE_APPROVAL, None),
        (PolicyOutcome.REQUIRE_APPROVAL, PolicyOutcome.REQUIRE_APPROVAL, None),
        (("require_approval", "over the limit"), PolicyOutcome.REQUIRE_APPROVAL, "over the limit"),
        (42, PolicyOutcome.DENY, "42"),
    ],
)
def test_policy_answers_normalise(decision, outcome, reason) -> None:
    assert normalize(decision) == (outcome, reason)


def _interrupts() -> tuple[Interrupt, Interrupt]:
    call = ToolCall(tool="t", idempotency_key="k1")
    approval = Interrupt(
        tenant_id="acme",
        run_id="run_1",
        reason=InterruptReason.APPROVAL,
        question="ok?",
        tool_call=call,
    )
    question = Interrupt(tenant_id="acme", run_id="run_1", question="which?")
    return approval, question


def test_the_registry_hands_answers_to_the_run_once_keyed_by_tenant() -> None:
    registry = ResolutionRegistry()
    approval, question = _interrupts()
    approve = InterruptResolution(
        interrupt_id=approval.interrupt_id, run_id="run_1", decision=InterruptDecision.APPROVE
    )
    answer = InterruptResolution(
        interrupt_id=question.interrupt_id,
        run_id="run_1",
        decision=InterruptDecision.ANSWER,
        answer="eu",
    )
    registry.record(approval, approve)
    registry.record(question, answer)
    assert registry.pending("acme", "run_1") and not registry.pending("acme", "run_2")
    assert not registry.pending("globex", "run_1")  # a run id alone names nothing
    found = registry.for_run("acme", "run_1")
    assert set(found) == {"k1", ANSWER}
    assert found["k1"].resolution == approve and found["k1"].interrupt == approval
    assert found[ANSWER].answer == "eu" and found[ANSWER].decision is InterruptDecision.ANSWER
    assert registry.for_run("acme", "run_1") == {} and not registry.pending("acme", "run_1")


def test_an_announced_pause_is_claimed_only_by_its_own_tenant_user_and_thread() -> None:
    registry = ResolutionRegistry(capacity=2)
    approval, question = _interrupts()
    registry.announce(question, CTX)
    assert [i.interrupt_id for i in registry.announced("acme")] == [question.interrupt_id]
    assert registry.announced("globex") == []
    refused = [
        registry.claim(question.interrupt_id, tenant_id="globex", user_id="u1", workspace_id="ws1"),
        registry.claim(question.interrupt_id, tenant_id="acme", user_id="u2", workspace_id="ws1"),
        registry.claim(question.interrupt_id, tenant_id="acme", user_id="u1", workspace_id=None),
        registry.claim(
            question.interrupt_id, tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="x"
        ),
        registry.claim("int_nope", tenant_id="acme", user_id="u1", workspace_id="ws1"),
    ]
    assert refused == [None] * 5 and registry.announced("acme")  # still waiting
    claimed = registry.claim(
        question.interrupt_id, tenant_id="acme", user_id="u1", workspace_id="ws1", thread_id="thr_1"
    )
    assert claimed is not None and claimed[0] == question and claimed[1] == CTX
    assert (
        registry.claim(question.interrupt_id, tenant_id="acme", user_id="u1", workspace_id="ws1")
        is None
    )
    for n in range(3):  # bounded: the oldest announcement is forgotten past the capacity
        registry.announce(Interrupt(tenant_id="acme", run_id=f"run_{n}", question="q"), CTX)
    assert len(registry.announced("acme")) == 2
    registry.announce(approval, CTX)
    assert registry.claim(approval.interrupt_id, tenant_id="acme", user_id="u1", workspace_id="ws1")
