"""The judge against the real things it depends on.

Two levels, both opt-in by what is running rather than by an exported variable:

* ``-m live`` — the **grounded stage** against a running Memory Service. This is the half of
  the judge that decides most verdicts and it is the half a fake would flatter: our idea of
  what ``POST /v1/verify`` returns is exactly the thing that has been wrong before.
* ``TRELLIS_JUDGE_SMOKE=1`` — a **20-sample judged smoke** through Bifrost on a budgeted
  virtual key, which reports the measured cost per judged sample into
  ``judge-smoke-results.json``. Opt-in because it is the only test here that spends money.

    make dev-up                                            # in the memory checkout
    pytest tests/e2e/test_live_judge.py -m live -q

    TRELLIS_JUDGE_SMOKE=1 BIFROST_URL=http://localhost:8091/v1 \
        BIFROST_API_KEY=$MEMORY__MODELS__LLM__API_KEY \
        pytest tests/e2e/test_live_judge.py -q -s -k smoke
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest
from trellis.contracts.evaluation import JudgeMethod

from trellis.harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResponse,
    BifrostModelClient,
    GroundedJudge,
)
from trellis.harness.config.settings import JudgeConfig
from trellis.harness.evaluation.grounding import decide

MEMORY_URL = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")
MEMORY_API_KEY = os.environ.get("MEMORY_API_KEY", "dev-key")
TENANT = os.environ.get("MEMORY_TENANT", "acme")

BIFROST_URL = os.environ.get("BIFROST_URL", "http://localhost:8091/v1")
BIFROST_API_KEY = os.environ.get("BIFROST_API_KEY") or os.environ.get("MEMORY_LLM_API_KEY")
JUDGE_MODEL = os.environ.get("TRELLIS_JUDGE_MODEL", "openrouter/openai/gpt-4.1-nano")
SMOKE_SAMPLES = int(os.environ.get("TRELLIS_JUDGE_SMOKE_SAMPLES", "20"))
#: The budgeted virtual key's id, so the smoke can report what the gateway actually charged
#: rather than what each response happened to quote. The key's *token* is never read here.
JUDGE_VK_ID = os.environ.get("TRELLIS_JUDGE_VK_ID")
RESULTS = Path(__file__).resolve().parents[2] / "judge-smoke-results.json"


def _reachable(url: str, path: str = "/health/live") -> bool:
    """Ask the service. An env-var gate turns "it is down" and "I forgot to export it" into
    the same green run."""
    import httpx

    try:
        return httpx.get(f"{url}{path}", timeout=5).status_code < 500
    except Exception:
        return False


LIVE = _reachable(MEMORY_URL)


# ------------------------------------------------------------------ the grounded stage, live


@pytest.fixture
async def live_client():
    from trellis.memory import MemoryClient

    client = MemoryClient(MEMORY_URL, api_key=MEMORY_API_KEY, timeout=120.0)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def live_context(live_client):
    from tests.support import onboard

    run = uuid.uuid4().hex[:8]
    await onboard(live_client, TENANT, workspace_id=f"ws-{run}", users=["judge-user"])
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        agent_id="judged-agent",
        user_id="judge-user",
        thread_id=f"judge-thread-{run}",
        turn_id=f"turn-{run}",
        workspace_id=f"ws-{run}",
        agent_group_id=f"crew-{run}",
    )


class NeverCalled:
    """Fails the test if the grounded stage escalated when it should not have."""

    async def structured(self, request, /, schema, **kwargs):
        raise AssertionError("grounded_only must never reach a model")


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason=f"no Memory Service at {MEMORY_URL} — `make dev-up`")
async def test_the_grounded_stage_verifies_the_answer_against_the_runs_own_bundle(
    live_client, live_context, spans
) -> None:
    """One turn, memory on, judge on and ``grounded_only``: the service's ``/v1/verify`` is
    asked about the answer, with the evidence *this run* was given, and nothing reaches a
    model. Whether the report turns out decisive depends on the service's own NLI — so the
    assertion is about the call and the rule, not about a number we would then be tempted to
    pin."""
    from tests.support import RecordingMemoryClient

    recording = RecordingMemoryClient(live_client)
    judge = GroundedJudge(
        model=NeverCalled(),
        config=JudgeConfig(enabled=True, sample_rate=1.0, grounded_only=True, threshold=0.5),
    )
    harness = AgentHarness(
        memory=recording,
        judge=judge,
        defaults={"tenant_id": TENANT},
        config={"memory": {"writeback": False}, "timeouts": {"memory_seconds": 120.0}},
    )
    try:

        @harness.agent(agent_id="judged-agent")
        async def agent(state, runtime):
            assert runtime.memory.enabled
            await runtime.memory.remember(
                "SKU-1 is reordered from Castor Supply below 10 days of cover.",
                memory_type="SEMANTIC",
                lifetime="LONG_TERM",
                visibility="USER",
            )
            return AgentResponse.ok("SKU-1 is reordered from Castor Supply.")

        result = await agent({"query": "who supplies SKU-1?"}, context=live_context)
        assert result.succeeded, result.error
        await harness.drain()
    finally:
        await harness.aclose()

    verifies = recording.of("verify")
    assert verifies, "the grounded stage must ask the service, not guess"
    asked = verifies[-1]
    assert asked["answer"] == "SKU-1 is reordered from Castor Supply."
    assert "bundle" in asked, "the run's own evidence, so the check costs no extra retrieval"

    report = await recording.bind(**live_context.scope_fields()).verify(
        "SKU-1 is reordered from Castor Supply.", query="who supplies SKU-1?"
    )
    assert report is not None
    decision = decide(report)
    assert decision.reason in ("grounded", "contradicted", "borderline", "unsupported", "no_claims")
    if decision.verdict is not None:
        assert decision.verdict.method is JudgeMethod.GROUNDED
        assert decision.verdict.cost_usd == 0.0, "the deterministic stage spends nothing"
    print(f"\n[live] grounded decision: {decision.reason}")


# ------------------------------------------------------------------ the budgeted smoke


SMOKE = os.environ.get("TRELLIS_JUDGE_SMOKE") == "1"
GATEWAY_UP = _reachable(BIFROST_URL.rsplit("/v1", 1)[0], "/v1/models")

#: Twenty answers to judge: ten defensible against their evidence and ten not, so the run
#: reports a number that would notice a judge saying "1.0" to everything.
SAMPLES: list[tuple[str, str, str, bool]] = [
    (
        "Which supplier fills SKU-1?",
        "Castor Supply fills SKU-1.",
        "SKU-1 is reordered from Castor Supply below 10 days of cover.",
        True,
    ),
    (
        "What service level is safety stock held at?",
        "Safety stock is held at a 95 percent service level.",
        "Reorder policy v3: safety stock is held at a 95 percent service level.",
        True,
    ),
    (
        "How many units of SKU-1 are on hand?",
        "There are 95 units at EU-1.",
        "Stock check for SKU-1 returned 95 units at EU-1.",
        True,
    ),
    (
        "When was the refund issued?",
        "The refund was issued on Tuesday.",
        "Refund R-77 was issued on Tuesday the 12th.",
        True,
    ),
    (
        "Who approved purchase order 4471?",
        "Dana approved purchase order 4471.",
        "PO-4471 was approved by Dana on the 3rd.",
        True,
    ),
    (
        "What is the digest cadence the planner wants?",
        "The planner wants a weekly digest.",
        "The planner prefers weekly digests.",
        True,
    ),
    (
        "Which warehouse holds the EU stock?",
        "EU-1 holds the EU stock.",
        "EU stock is held at warehouse EU-1.",
        True,
    ),
    (
        "What is the reorder trigger?",
        "Reordering is triggered below 10 days of cover.",
        "SKU-1 is reordered from Castor Supply below 10 days of cover.",
        True,
    ),
    (
        "Is the reorder policy current?",
        "The current reorder policy is version 3.",
        "Reorder policy v3 supersedes v2 as of the 1st.",
        True,
    ),
    (
        "How long does Castor Supply take?",
        "Castor Supply delivers in 6 days.",
        "Castor Supply's quoted lead time is 6 days.",
        True,
    ),
    (
        "Which supplier fills SKU-1?",
        "Pollux Parts fills SKU-1 and always has.",
        "SKU-1 is reordered from Castor Supply below 10 days of cover.",
        False,
    ),
    (
        "What service level is safety stock held at?",
        "Safety stock is held at a 99.9 percent service level.",
        "Reorder policy v3: safety stock is held at a 95 percent service level.",
        False,
    ),
    (
        "How many units of SKU-1 are on hand?",
        "There are 950 units at EU-1, so no action is needed.",
        "Stock check for SKU-1 returned 95 units at EU-1.",
        False,
    ),
    (
        "When was the refund issued?",
        "No refund has been issued.",
        "Refund R-77 was issued on Tuesday the 12th.",
        False,
    ),
    (
        "Who approved purchase order 4471?",
        "Nobody has approved purchase order 4471 yet.",
        "PO-4471 was approved by Dana on the 3rd.",
        False,
    ),
    (
        "What is the digest cadence the planner wants?",
        "The planner wants hourly digests.",
        "The planner prefers weekly digests.",
        False,
    ),
    (
        "Which warehouse holds the EU stock?",
        "The EU stock is held at US-3.",
        "EU stock is held at warehouse EU-1.",
        False,
    ),
    (
        "What is the reorder trigger?",
        "Reordering is triggered below 60 days of cover.",
        "SKU-1 is reordered from Castor Supply below 10 days of cover.",
        False,
    ),
    (
        "Is the reorder policy current?",
        "Version 2 of the reorder policy is the current one.",
        "Reorder policy v3 supersedes v2 as of the 1st.",
        False,
    ),
    (
        "How long does Castor Supply take?",
        "Castor Supply delivers same day.",
        "Castor Supply's quoted lead time is 6 days.",
        False,
    ),
]


class Evidence:
    """One sample's evidence, shaped like a bundle for the rubric to read."""

    def __init__(self, text: str) -> None:
        self.rendered = text


@pytest.mark.live_llm
@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not (SMOKE and GATEWAY_UP and BIFROST_API_KEY),
    reason=(
        "the judged smoke spends money: set TRELLIS_JUDGE_SMOKE=1, BIFROST_URL and "
        "BIFROST_API_KEY (a budgeted virtual key)"
    ),
)
async def test_judged_smoke_reports_its_cost_per_sample() -> None:
    """Twenty real judgements on a budgeted key, with the cost written down.

    The grounded stage is deliberately bypassed here (no verifier is bound), because the
    number this run exists to produce is *what the LLM stage costs* — the stage the budget is
    for. In production the grounded stage settles most of these for nothing.

    The cost is taken from the **gateway's own meter** when the virtual key's id is known
    (``TRELLIS_JUDGE_VK_ID``): not every upstream returns a price in ``usage``, and a smoke
    whose cost line came back zero because nobody quoted one would be worse than no cost line
    at all. The per-response figures are kept alongside, labelled for what they are.
    """
    before = await _metered_usage()
    client = BifrostModelClient(BIFROST_URL, model=JUDGE_MODEL, api_key=BIFROST_API_KEY)
    judge = GroundedJudge(
        model=client,
        config=JudgeConfig(enabled=True, sample_rate=1.0, model=JUDGE_MODEL, threshold=0.5),
    )
    samples = SAMPLES[:SMOKE_SAMPLES]
    scored: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for index, (question, answer, evidence, defensible) in enumerate(samples):
            bound = judge.bound(question=question, bundle=Evidence(evidence))
            verdict = await bound.judge(_event(f"smoke_{index}"), response=AgentResponse.ok(answer))
            scored.append(
                {
                    "question": question,
                    "defensible": defensible,
                    "score": None if verdict is None else float(verdict.score),
                    "label": None if verdict is None else verdict.label,
                    "method": None if verdict is None else verdict.method.value,
                    "cost_usd": None if verdict is None else verdict.cost_usd,
                    "tokens": None if verdict is None else verdict.metadata.get("tokens"),
                }
            )
    finally:
        await client.aclose()
    elapsed = time.perf_counter() - started

    after = await _metered_usage()

    judged = [s for s in scored if s["score"] is not None]
    assert judged, "a smoke that judged nothing proves nothing"
    quoted = sum(float(s["cost_usd"] or 0.0) for s in judged)
    tokens = [int(s["tokens"] or 0) for s in judged]
    metered = None if before is None or after is None else round(after - before, 8)
    total = metered if metered is not None else quoted
    payload = {
        "model": JUDGE_MODEL,
        "gateway": BIFROST_URL,
        "samples": len(samples),
        "judged": len(judged),
        "abstained": len(scored) - len(judged),
        #: What the gateway's governance meter charged the virtual key: the authoritative
        #: number, and the one the phase budget is spent in.
        "cost_source": "gateway_meter" if metered is not None else "response_usage",
        "total_cost_usd": round(total, 8),
        "cost_per_judged_sample_usd": round(total / len(judged), 8),
        #: What the responses themselves quoted. Zero means nobody quoted a price, not free.
        "quoted_cost_usd": round(quoted, 8),
        "virtual_key_usage_before_usd": before,
        "virtual_key_usage_after_usd": after,
        "mean_tokens_per_judged_sample": round(sum(tokens) / len(judged), 1) if tokens else None,
        "mean_score_defensible": _mean(judged, defensible=True),
        "mean_score_not_defensible": _mean(judged, defensible=False),
        "elapsed_seconds": round(elapsed, 2),
        "items": scored,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[smoke] {json.dumps({k: v for k, v in payload.items() if k != 'items'}, indent=2)}")

    assert total < 0.20, f"a 20-sample smoke must not cost {total:.4f} USD"
    high, low = payload["mean_score_defensible"], payload["mean_score_not_defensible"]
    if high is not None and low is not None:
        assert high > low, (
            "a judge that scores defensible and indefensible answers the same is not judging: "
            f"{high} vs {low}"
        )


async def _metered_usage() -> float | None:
    """What the gateway's governance meter has charged this virtual key so far.

    ``None`` when no key id was given or the governance API does not answer — the smoke still
    runs and says which number it reported. Nothing about the key itself is read or logged,
    only its running total.
    """
    if not JUDGE_VK_ID:
        return None
    import httpx

    root = BIFROST_URL.rsplit("/v1", 1)[0]
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            response = await http.get(f"{root}/api/governance/virtual-keys/{JUDGE_VK_ID}")
        response.raise_for_status()
        body = response.json()
        budgets = (body.get("virtual_key") or body).get("budgets") or []
        return float(budgets[0]["current_usage"]) if budgets else None
    except Exception:
        return None


def _mean(judged: list[dict[str, object]], *, defensible: bool) -> float | None:
    scores = [float(s["score"]) for s in judged if s["defensible"] is defensible]  # type: ignore[arg-type]
    return round(sum(scores) / len(scores), 4) if scores else None


def _event(run_id: str):
    from trellis.contracts.events import AgentEvalEvent

    return AgentEvalEvent(agent_id="smoke-agent", agent_run_id=run_id, tenant_id=TENANT)
