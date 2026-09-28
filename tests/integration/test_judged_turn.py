"""A judged turn, through a real harness (design §11).

The claim being tested is narrow and load-bearing: **a user waiting for an answer never waits
for an opinion about it**. Everything else here follows from that — the judging happens on the
writeback queue, a saturated queue drops the judgement rather than the memory write, and the
verdict lands in three places (a score on the trace, an ``agent.judge`` span, a feedback record
with ``source="judge"``) only after the response has already gone back.

The memory-backed tests use the real Memory Service through the ``harness`` fixture, so the
judge's feedback record is really written; the rest need no service at all.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.support import span_names
from tests.support_gateway import FakeGateway
from trellis.contracts.errors import ConfigurationError
from trellis.contracts.evaluation import JudgeMethod, JudgeVerdict

from trellis.harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResponse,
    BifrostModelClient,
    GroundedJudge,
)
from trellis.harness.config.settings import JudgeConfig, RunsEngine
from trellis.harness.evaluation.rubric import VERDICT_SCHEMA, RubricPrompt
from trellis.harness.interceptors.judge import METHOD_SCORE_NAME, SCORE_NAME

TENANT = "acme"


class RecordingProvider:
    """An ``EvaluationProvider`` that keeps what was scored (Langfuse's shape, no Langfuse)."""

    def __init__(self) -> None:
        self.scores: list[tuple[str, object, dict]] = []

    async def score(self, name, value, /, **metadata) -> None:
        self.scores.append((name, value, metadata))

    async def submit_dataset_item(self, dataset, item) -> None:  # pragma: no cover - unused here
        return None

    async def submit_feedback(self, feedback) -> None:  # pragma: no cover - unused here
        return None


class SlowJudge:
    """Scores, eventually. The point is that the turn does not wait for it."""

    def __init__(self, *, delay: float = 0.05, score: float = 0.9) -> None:
        self.delay = delay
        self.score = score
        self.calls: list[str] = []
        self.finished = False

    async def judge(self, event, /, *, response=None):
        self.calls.append(event.agent_run_id)
        await asyncio.sleep(self.delay)
        self.finished = True
        return JudgeVerdict(
            score=self.score,
            method=JudgeMethod.LLM,
            label="defensible",
            rationale="every claim is in the evidence",
            model="openrouter/openai/gpt-4.1-nano",
            cost_usd=0.00007,
        )


def agent_returning(text: str):
    async def agent(payload, runtime) -> AgentResponse:
        return AgentResponse.ok(text)

    return agent


# --------------------------------------------------------------------- the asynchronous path


async def test_a_judged_turn_returns_before_the_verdict_exists(memory, context, spans) -> None:
    judge = SlowJudge(delay=0.1)
    provider = RecordingProvider()
    harness = AgentHarness(
        memory=memory,
        judge=judge,
        evaluation_provider=provider,
        defaults={"tenant_id": TENANT},
        config={"judge": {"threshold": 0.5}},
    )
    try:
        result = await harness.wrap(agent_returning("issued on Tuesday"), agent_id="refunds")(
            None, context=context
        )
        assert result.status.ok and result.data == "issued on Tuesday"
        assert not judge.finished, "the answer went back before the opinion about it existed"

        await harness.drain()
        assert judge.finished

        numeric = next(s for s in provider.scores if s[0] == SCORE_NAME)
        assert numeric[1] == pytest.approx(0.9)
        assert numeric[2]["trace_id"] == context.trace_id
        assert numeric[2]["comment"] == "every claim is in the evidence"
        assert any(s[0] == METHOD_SCORE_NAME and s[1] == "llm" for s in provider.scores)

        assert "agent.judge" in span_names(spans)

        submitted = memory.of("feedback.submit")[-1]
        assert submitted["source"] == "judge"
        assert submitted["verdict"] == "confirm", "0.9 is above the threshold"
        assert submitted["score"] == pytest.approx(0.9)
        assert submitted["reviewer"] == "judge:openrouter/openai/gpt-4.1-nano"
        assert submitted["metadata"]["method"] == "llm"
    finally:
        await harness.aclose()


async def test_a_verdict_below_the_threshold_rejects_the_answer(memory, context) -> None:
    harness = AgentHarness(
        memory=memory,
        judge=SlowJudge(delay=0.0, score=0.2),
        defaults={"tenant_id": TENANT},
        config={"judge": {"threshold": 0.5}},
    )
    try:
        await harness.wrap(agent_returning("probably Thursday"), agent_id="refunds")(
            None, context=context
        )
        await harness.drain()
        assert memory.of("feedback.submit")[-1]["verdict"] == "reject"
    finally:
        await harness.aclose()


async def test_the_judge_span_carries_what_a_plain_otlp_backend_needs(context, spans) -> None:
    """Langfuse gets a score; everyone else gets a span with the same numbers on it."""
    harness = AgentHarness(judge=SlowJudge(delay=0.0), defaults={"tenant_id": TENANT})
    try:
        await harness.wrap(agent_returning("issued"), agent_id="refunds")(None, context=context)
        await harness.drain()
    finally:
        await harness.aclose()

    span = next(s for s in spans.get_finished_spans() if s.name == "agent.judge")
    assert span.attributes["judge.score"] == pytest.approx(0.9)
    assert span.attributes["judge.method"] == "llm"
    assert span.attributes["judge.label"] == "defensible"
    assert span.attributes["judge.cost_usd"] == pytest.approx(0.00007)
    assert span.attributes["agent.id"] == "refunds"


async def test_a_failed_turn_is_not_judged(context) -> None:
    """A failed turn has no answer to score; its failure is already the record."""
    judge = SlowJudge(delay=0.0)

    async def falls_over(payload, runtime):
        raise RuntimeError("upstream 500")

    harness = AgentHarness(
        judge=judge,
        defaults={"tenant_id": TENANT},
        error_mode="return",
        config={"retries": {"enabled": False}},
    )
    try:
        result = await harness.wrap(falls_over, agent_id="refunds")(None, context=context)
        assert not result.status.ok
        await harness.drain()
        assert judge.calls == []
    finally:
        await harness.aclose()


# --------------------------------------------------------------------- sampling and budget


def fresh_context(agent_id: str, run: str) -> AgentExecutionContext:
    return AgentExecutionContext.create(
        tenant_id=TENANT, agent_id=agent_id, user_id="u1", thread_id=f"t-{run}", turn_id=run
    )


async def test_per_agent_sampling_decides_before_any_work_is_done(spans) -> None:
    """``judge.agents["quiet"].sample_rate = 0`` judges nothing for quiet while the default
    rate still judges loud — and the unsampled run does not even bind the judge."""
    config = JudgeConfig(
        enabled=True, sample_rate=1.0, grounded_only=True, agents={"quiet": {"sample_rate": 0.0}}
    )
    verified: list[str] = []

    class CountingJudge(GroundedJudge):
        async def judge(self, event, /, *, response=None):
            verified.append(event.agent_id)
            return await super().judge(event, response=response)

    harness = AgentHarness(judge=CountingJudge(config=config), defaults={"tenant_id": TENANT})
    try:
        for agent_id in ("quiet", "loud"):
            await harness.wrap(agent_returning("an answer"), agent_id=agent_id)(
                None, context=fresh_context(agent_id, f"run-{agent_id}")
            )
        await harness.drain()
    finally:
        await harness.aclose()

    assert verified == ["loud"], "an unsampled agent costs a hash, not a judgement"


async def test_a_saturated_queue_drops_the_judgement_not_the_memory_write(context) -> None:
    """A sampled judgement is optional; a memory write is not."""
    judge = SlowJudge(delay=0.0)
    harness = AgentHarness(judge=judge, defaults={"tenant_id": TENANT})
    try:
        harness.writeback.max_pending = 0
        assert harness.writeback.saturated

        result = await harness.wrap(agent_returning("issued"), agent_id="refunds")(
            None, context=context
        )
        assert result.status.ok, "the turn still answers"
        await harness.drain()
        assert judge.calls == [], "the judgement was dropped rather than run inline"
    finally:
        await harness.aclose()


def test_the_judge_is_off_until_a_deployment_turns_it_on() -> None:
    """Nothing starts spending because the package was upgraded."""
    assert AgentHarness(defaults={"tenant_id": TENANT}).judge is None
    configured = AgentHarness(
        defaults={"tenant_id": TENANT}, config={"judge": {"enabled": True, "grounded_only": True}}
    )
    assert isinstance(configured.judge, GroundedJudge)
    assert configured.judge.model is None, "grounded_only never reaches for a model"


def test_a_configured_judge_uses_the_harnesss_own_gateway_client() -> None:
    harness = AgentHarness(
        model=BifrostModelClient("http://gateway.test", model="m", api_key="vk"),
        defaults={"tenant_id": TENANT},
        config={"judge": {"enabled": True, "model": "openrouter/openai/gpt-4.1-nano"}},
    )
    assert harness.judge is not None
    assert harness.judge.model is harness.model_client, "one gateway client, one budget"
    assert harness.judge.config.model == "openrouter/openai/gpt-4.1-nano"


# --------------------------------------------------------------------- the rubric on the wire


async def test_a_bifrost_rubric_id_reaches_the_gateway_as_its_headers() -> None:
    """The rubric lives in Bifrost and is injected by id (design §3). A harness that pasted the
    text instead would be judging by whatever was deployed, not by the versioned prompt."""
    verdict = {"score": 0.8, "label": "defensible", "rationale": "supported by the policy"}
    with FakeGateway([{"choices": [{"message": {"content": json.dumps(verdict)}}]}]) as gateway:
        client = BifrostModelClient(gateway.url, model="m", api_key="vk-test")
        judge = GroundedJudge(
            model=client,
            config=JudgeConfig(
                enabled=True,
                sample_rate=1.0,
                model="openrouter/openai/gpt-4.1-nano",
                rubric_prompt_id="prompt_judge_v2",
                rubric_prompt_version="3",
            ),
        ).bound(verifier=None, bundle={"rendered": "the refund policy"}, question="when?")
        scored = await judge.judge(
            _event("refunds", "run_h1"), response=AgentResponse.ok("issued on Tuesday")
        )
        await client.aclose()

    assert scored is not None and scored.method is JudgeMethod.LLM
    assert scored.metadata["rubric"] == "bifrost:prompt_judge_v2@3"
    headers = gateway.headers[0]
    assert headers["x-bf-prompt-id"] == "prompt_judge_v2"
    assert headers["x-bf-prompt-version"] == "3"
    body = gateway.requests[0]
    assert "_options" not in body, "the options are headers, not body fields"
    assert body["response_format"]["json_schema"]["schema"] == VERDICT_SCHEMA
    assert "Score from 0 to 1" not in body["messages"][0]["content"], "the gateway holds it"


async def test_a_non_numeric_prompt_version_travels_as_the_documented_header() -> None:
    """Bifrost numbers its prompt versions, so the typed field takes a number; a deployment
    that labels them ("stable") must still select one rather than have it dropped."""
    with FakeGateway([{"choices": [{"message": {"content": "{}"}}]}]) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        await client.structured(
            RubricPrompt(prompt_id="p1", version="stable").request(
                model="m", question="q", answer="a", evidence="e", unsettled="borderline"
            ),
            schema=VERDICT_SCHEMA,
        )
        await client.aclose()
    assert gateway.headers[0]["x-bf-prompt-id"] == "p1"
    assert gateway.headers[0]["x-bf-prompt-version"] == "stable"


async def test_a_request_with_no_prompt_id_sends_no_prompt_headers() -> None:
    with FakeGateway([{"choices": [{"message": {"content": "hello"}}]}]) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        await client.invoke("hello?")
        await client.aclose()
    assert "x-bf-prompt-id" not in gateway.headers[0]


def _event(agent_id: str, run_id: str):
    from trellis.contracts.events import AgentEvalEvent

    return AgentEvalEvent(agent_id=agent_id, agent_run_id=run_id, tenant_id=TENANT)


# --------------------------------------------------------------------- runs.engine dispatch


def test_runs_engine_selects_the_adapter() -> None:
    """Both satisfy the same port, so this is the whole difference between recording to
    agent-runs and recording to Temporal."""
    http = AgentHarness(
        defaults={"tenant_id": TENANT},
        config={"runs": {"url": "http://runs.test", "api_key": "k"}},
    )
    assert type(http.runs).__name__ == "RunStoreClient"

    temporal = AgentHarness(
        defaults={"tenant_id": TENANT},
        config={
            "runs": {
                "engine": "temporal",
                "temporal": {"target": "localhost:7233", "task_queue": "trellis-runs"},
            }
        },
    )
    assert type(temporal.runs).__name__ == "TemporalRunStore"
    assert temporal.runs.task_queue == "trellis-runs"
    assert not temporal.runs.connection.connected, "connecting is lazy: a harness is built sync"


def test_temporal_without_a_target_records_nothing_rather_than_guessing() -> None:
    unconfigured = AgentHarness(
        defaults={"tenant_id": TENANT}, config={"runs": {"engine": "temporal"}}
    )
    assert unconfigured.runs.name == "noop"


def test_an_unknown_engine_is_a_startup_error_not_an_unrecorded_run() -> None:
    from pydantic import ValidationError

    from trellis.harness.config.settings import HarnessConfig

    assert RunsEngine("temporal") is RunsEngine.TEMPORAL
    with pytest.raises(ValidationError):
        HarnessConfig.load({"runs": {"engine": "temporl"}})


def test_selecting_temporal_without_the_extra_names_the_install_command(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "trellis.harness_temporal", None)
    with pytest.raises(ConfigurationError, match=r"trellis-harness\[temporal\]"):
        AgentHarness(
            defaults={"tenant_id": TENANT},
            config={
                "runs": {
                    "engine": "temporal",
                    "temporal": {"target": "localhost:7233", "task_queue": "q"},
                }
            },
        )
