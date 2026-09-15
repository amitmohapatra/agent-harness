"""Property-based checks for the invariants that must hold for *any* input (§83).

Hypothesis is used where a property is genuinely universal: idempotency keys must be
stable and collision-resistant, redaction must never emit a known secret, and context
derivation must always preserve lineage.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from universal_agent_harness import AgentExecutionContext
from universal_agent_harness.telemetry.redaction import DefaultRedactor

ids = st.text(min_size=1, max_size=40).filter(lambda s: s.strip())
SETTINGS = settings(max_examples=150, deadline=None)


@given(tenant=ids, thread=ids, turn=ids, agent=ids, part=st.text(max_size=60))
@SETTINGS
def test_idempotency_keys_are_deterministic(tenant, thread, turn, agent, part):
    def key() -> str:
        ctx = AgentExecutionContext.create(
            tenant_id=tenant, agent_id=agent, thread_id=thread, turn_id=turn,
            agent_run_id="run_fixed",
        )
        return ctx.idempotency_key("obs", part)

    assert key() == key()


@given(a=st.text(max_size=40), b=st.text(max_size=40))
@SETTINGS
def test_different_content_gives_different_keys(a, b):
    assume(a != b)
    ctx = AgentExecutionContext.create(
        tenant_id="acme", agent_id="inv", thread_id="t", turn_id="turn", agent_run_id="run"
    )
    assert ctx.idempotency_key("obs", a) != ctx.idempotency_key("obs", b)


@given(agent=ids, group=st.one_of(st.none(), ids))
@SETTINGS
def test_children_always_preserve_lineage(agent, group):
    parent = AgentExecutionContext.create(
        tenant_id="acme", agent_id="planner", thread_id="t", user_id="u", agent_group_id=group
    )
    child = parent.for_agent(agent)
    assert child.tenant_id == parent.tenant_id
    assert child.trace_id == parent.trace_id
    assert child.thread_id == parent.thread_id
    assert child.user_id == parent.user_id
    assert child.parent_agent_run_id == parent.agent_run_id
    assert child.agent_run_id != parent.agent_run_id


@given(
    secret=st.text(min_size=8, max_size=40).filter(lambda s: s.strip() and "@" not in s),
    key=st.sampled_from(["api_key", "authorization", "password", "secret", "x-api-key",
                         "access_token", "cookie"]),
)
@SETTINGS
def test_named_secrets_are_never_emitted(secret, key):
    out = DefaultRedactor().redact_attributes({key: secret})
    assert secret not in str(out)


@given(value=st.integers(min_value=0, max_value=10**9))
@SETTINGS
def test_numeric_metric_attributes_always_survive(value):
    attributes = {
        "gen_ai.usage.input_tokens": value,
        "gen_ai.usage.output_tokens": value,
        "duration_ms": float(value),
    }
    assert DefaultRedactor().redact_attributes(attributes) == attributes


@given(text=st.text(max_size=500))
@SETTINGS
def test_redaction_never_raises(text):
    redactor = DefaultRedactor()
    redactor.redact_attributes({"note": text})
    redactor.redact_input({"nested": [text, {"deeper": text}]})
    redactor.redact_output(text)
