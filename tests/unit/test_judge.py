"""The online judge: grounded first, budgeted, and never a guess.

Nothing here spends a cent — the model client raises if anyone calls it on the grounded path,
and the LLM path is a scripted client. That is the point: the judge decides *whether* to spend
before it spends, and that decision is what these tests are about.
"""

from __future__ import annotations

import pytest
from trellis.contracts.evaluation import JudgeMethod
from trellis.contracts.events import AgentEvalEvent
from trellis.contracts.messages import AgentResponse, AgentStatus
from trellis.contracts.model import ModelResponse, ModelUsage
from trellis.contracts.ports import Judge

from trellis.harness import Claim
from trellis.harness.config.settings import JudgeConfig
from trellis.harness.evaluation.answers import answer_text
from trellis.harness.evaluation.budget import JudgeBudget
from trellis.harness.evaluation.grounding import decide
from trellis.harness.evaluation.judge import (
    MAX_LABEL_CHARS,
    MAX_RATIONALE_CHARS,
    GroundedJudge,
    ScopedJudge,
)
from trellis.harness.evaluation.rubric import BUILTIN_RUBRIC, RubricPrompt


class Report:
    """A grounding report, shaped like the Memory Service's."""

    def __init__(self, *, supported=0, unsupported=0, contradicted=0, borderline=0) -> None:
        self.supported = supported
        self.unsupported = unsupported
        self.contradicted = contradicted
        self.borderline = borderline
        self.nli_provider = "e5"
        self.per_claim_hallucination_rate = 0.0 if not unsupported else 0.5
        self.judge_consulted = 0


class Verifier:
    def __init__(self, report=None, *, fails: bool = False) -> None:
        self.report = report
        self.fails = fails
        self.calls: list[tuple[str, dict]] = []

    async def verify(self, answer: str, /, **options):
        self.calls.append((answer, options))
        if self.fails:
            raise RuntimeError("verify is down")
        return self.report


class NeverCalled:
    """A model client that fails the test if the judge spends a token on it."""

    async def structured(self, request, /, schema, **kwargs):
        raise AssertionError("the grounded stage settled it; no model should have been called")


class Scripted:
    def __init__(self, payload: dict, *, usage: ModelUsage | None = None) -> None:
        self.payload = payload
        self.usage = usage
        self.requests: list = []

    async def structured(self, request, /, schema, **kwargs):
        self.requests.append(request)
        return ModelResponse(
            text=None, data=self.payload, model="openrouter/openai/gpt-4.1-nano", usage=self.usage
        )


def event(agent_id: str = "ref", run_id: str = "run_1") -> AgentEvalEvent:
    return AgentEvalEvent(agent_id=agent_id, agent_run_id=run_id, tenant_id="acme")


ANSWER = AgentResponse(status=AgentStatus.SUCCESS, data="the refund was issued on Tuesday")
ALWAYS = JudgeConfig(enabled=True, sample_rate=1.0)


def test_the_judge_satisfies_the_port_and_can_be_bound_to_a_run() -> None:
    judge = GroundedJudge(config=ALWAYS)
    assert isinstance(judge, Judge)
    assert isinstance(judge, ScopedJudge)
    bound = judge.bound(verifier=Verifier(), bundle=object(), question="why?")
    assert bound is not judge
    assert bound.budget is judge.budget, "ceilings are per agent, not per bound copy"
    assert judge.verifier is None, "binding must not reach back into the shared judge"


def test_binding_keeps_the_judge_a_deployment_installed() -> None:
    """A subclass is how a deployment changes what judging means; a bound copy that had
    silently become the base class would score with something nobody chose."""

    class HouseJudge(GroundedJudge):
        name = "house"

    bound = HouseJudge(config=ALWAYS).bound(verifier=Verifier(), bundle={})
    assert isinstance(bound, HouseJudge) and bound.name == "house"


async def test_a_settled_grounding_report_costs_nothing() -> None:
    verifier = Verifier(Report(supported=3))
    judge = GroundedJudge(model=NeverCalled(), config=ALWAYS).bound(
        verifier=verifier, bundle={"rendered": "evidence"}
    )
    verdict = await judge.judge(event(), response=ANSWER)
    assert verdict is not None
    assert verdict.method is JudgeMethod.GROUNDED
    assert verdict.score == 1.0
    assert verdict.cost_usd == 0.0
    assert verifier.calls[0][1]["bundle"] == {"rendered": "evidence"}, "the run's own bundle"


async def test_a_contradiction_is_decided_without_a_model() -> None:
    judge = GroundedJudge(model=NeverCalled(), config=ALWAYS).bound(
        verifier=Verifier(Report(supported=2, contradicted=1)), bundle={}
    )
    verdict = await judge.judge(event(), response=ANSWER)
    assert verdict is not None and verdict.label == "contradicted"
    assert verdict.score == pytest.approx(2 / 3)


async def test_only_what_the_classifier_could_not_settle_reaches_the_model() -> None:
    model = Scripted(
        {"score": 0.7, "label": "partly", "rationale": "the date is not in the evidence"},
        usage=ModelUsage(input_tokens=600, output_tokens=20, cost_usd=0.00008),
    )
    judge = GroundedJudge(model=model, config=ALWAYS).bound(
        verifier=Verifier(Report(supported=2, borderline=1)),
        bundle={"rendered": "refund policy"},
        question="when was the refund issued?",
    )
    verdict = await judge.judge(event(), response=ANSWER)
    assert verdict is not None
    assert verdict.method is JudgeMethod.LLM
    assert verdict.score == 0.7
    assert verdict.cost_usd == 0.00008
    assert verdict.metadata["escalated"] == "borderline"
    assert verdict.metadata["rubric"] == "builtin:grounded-answer"
    sent = model.requests[0]
    assert "when was the refund issued?" in sent.messages[0]["content"]
    assert judge.budget.spent("ref") == pytest.approx(0.00008)


async def test_grounded_only_never_calls_a_model_even_when_undecided() -> None:
    judge = GroundedJudge(
        model=NeverCalled(), config=JudgeConfig(enabled=True, sample_rate=1.0, grounded_only=True)
    ).bound(verifier=Verifier(Report(supported=1, unsupported=1)), bundle={})
    assert await judge.judge(event(), response=ANSWER) is None


async def test_a_verifier_that_is_down_escalates_rather_than_scoring_blind() -> None:
    model = Scripted({"score": 0.5, "label": "unclear", "rationale": "no evidence available"})
    judge = GroundedJudge(model=model, config=ALWAYS).bound(
        verifier=Verifier(fails=True), bundle={}
    )
    verdict = await judge.judge(event(), response=ANSWER)
    assert verdict is not None and verdict.method is JudgeMethod.LLM
    assert verdict.metadata["escalated"] == "no_report"


async def test_the_judge_abstains_instead_of_guessing() -> None:
    judge = GroundedJudge(model=NeverCalled(), config=ALWAYS)
    assert await judge.judge(event(), response=AgentResponse.ok(None)) is None, "no answer"
    assert await judge.judge(event(), response=None) is None, "no response at all"
    unbound = GroundedJudge(model=None, config=ALWAYS)
    assert await unbound.judge(event(), response=ANSWER) is None, "no verifier, no model"


async def test_an_unparsable_verdict_is_a_missing_score_not_a_crash() -> None:
    class Nonsense:
        async def structured(self, request, /, schema, **kwargs):
            return ModelResponse(text="no", data=None)

    judge = GroundedJudge(model=Nonsense(), config=ALWAYS).bound(
        verifier=Verifier(Report(supported=1, borderline=1)), bundle={}
    )
    assert await judge.judge(event(), response=ANSWER) is None


async def test_a_model_that_raises_is_a_missing_score_not_a_failed_run() -> None:
    class Broken:
        async def structured(self, request, /, schema, **kwargs):
            raise RuntimeError("gateway down")

    judge = GroundedJudge(model=Broken(), config=ALWAYS).bound(
        verifier=Verifier(Report(supported=1, borderline=1)), bundle={}
    )
    assert await judge.judge(event(), response=ANSWER) is None


async def test_sampling_is_per_agent_and_deterministic_in_the_run_id() -> None:
    config = JudgeConfig(enabled=True, sample_rate=1.0, agents={"quiet": {"sample_rate": 0.0}})
    judge = GroundedJudge(model=NeverCalled(), config=config).bound(
        verifier=Verifier(Report(supported=1)), bundle={}
    )
    assert await judge.judge(event("quiet"), response=ANSWER) is None
    assert await judge.judge(event("loud"), response=ANSWER) is not None
    half = JudgeBudget(JudgeConfig(enabled=True, sample_rate=0.5))
    first = [half.admit("a", f"run_{i}").admitted for i in range(50)]
    second = [half.admit("a", f"run_{i}").admitted for i in range(50)]
    assert first == second, "the same run must always land on the same side"
    assert 0 < sum(first) < 50, "a 0.5 rate that admits everything or nothing is not sampling"


async def test_the_hourly_count_and_the_spend_are_both_ceilings() -> None:
    clock = [0.0]
    budget = JudgeBudget(
        JudgeConfig(enabled=True, sample_rate=1.0, max_per_hour=2, max_usd_per_hour=0.001),
        clock=lambda: clock[0],
    )
    assert budget.reserve("a", "r1").admitted
    assert budget.reserve("a", "r2").admitted
    assert budget.judged("a") == 2
    assert budget.reserve("a", "r3").reason == "over_max_per_hour"
    clock[0] = 3601.0
    assert budget.judged("a") == 0, "the window rolls"
    assert budget.reserve("a", "r3").admitted
    budget.spend("a", 0.002)
    assert budget.reserve("a", "r4").reason == "over_budget"
    clock[0] = 7300.0
    assert budget.reserve("a", "r4").admitted


def test_the_count_ceiling_holds_across_concurrent_judgements() -> None:
    """The one that asking-then-recording gets wrong.

    Judging is asynchronous, so between a judge's own admission check and the verdict it
    records there are awaits. If the slot is only taken at the end, every judgement that
    started in that gap saw the same free count — a queue of 256 walks straight through a
    ceiling of 60. Taking the slot at the decision is what closes it.
    """
    budget = JudgeBudget(JudgeConfig(enabled=True, sample_rate=1.0, max_per_hour=3))
    started = [budget.reserve("a", f"r{i}").admitted for i in range(10)]
    assert sum(started) == 3, "ten runs arriving at once must not all be admitted"
    assert budget.judged("a") == 3

    asked = JudgeBudget(JudgeConfig(enabled=True, sample_rate=1.0, max_per_hour=3))
    assert all(asked.admit("a", f"r{i}").admitted for i in range(10)), (
        "admit() is a question and takes nothing: that is why reserve() exists"
    )
    assert asked.judged("a") == 0


def test_a_negative_quoted_cost_cannot_move_the_ceiling_backwards() -> None:
    """The spend accumulates a number the gateway supplies. A negative one would make the
    ceiling recede instead of approach, which is a budget that loosens under load."""
    budget = JudgeBudget(JudgeConfig(enabled=True, sample_rate=1.0, max_usd_per_hour=0.01))
    budget.spend("a", 0.009)
    budget.spend("a", -5.0)
    assert budget.spent("a") == pytest.approx(0.009)
    budget.spend("a", 0.002)
    assert budget.reserve("a", "r1").reason == "over_budget"


def test_a_disabled_judge_admits_nothing() -> None:
    budget = JudgeBudget(JudgeConfig(enabled=False, sample_rate=1.0))
    assert budget.admit("a", "r1").reason == "disabled"
    assert budget.reserve("a", "r1").reason == "disabled"
    assert budget.judged("a") == 0, "a refused reservation takes no slot"


async def test_a_run_with_no_answer_is_not_charged_a_slot() -> None:
    """Abstaining because there is nothing to score happens before the reservation: a turn
    that returned no text must not consume the hour's judging budget."""
    judge = GroundedJudge(model=NeverCalled(), config=JudgeConfig(enabled=True, sample_rate=1.0))
    assert await judge.judge(event(), response=AgentResponse.ok(None)) is None
    assert judge.budget.judged("ref") == 0


async def test_the_models_own_strings_are_bounded_before_they_travel() -> None:
    """A label and a rationale go onto a span, a trace score and a feedback row. The model
    chooses them, so it does not get to choose how long they are."""
    model = Scripted({"score": 0.6, "label": "x" * 500, "rationale": "y" * 5000})
    judge = GroundedJudge(model=model, config=ALWAYS).bound(
        verifier=Verifier(Report(supported=1, borderline=1)), bundle={}
    )
    verdict = await judge.judge(event(), response=ANSWER)
    assert verdict is not None
    assert verdict.label is not None and len(verdict.label) == MAX_LABEL_CHARS
    assert verdict.rationale is not None and len(verdict.rationale) == MAX_RATIONALE_CHARS


def test_the_grounded_rule_is_stated_once_and_holds_at_its_edges() -> None:
    assert decide(None).reason == "no_report"
    assert decide(Report()).reason == "no_claims"
    assert decide(Report(supported=1, unsupported=1)).reason == "unsupported"
    assert decide(Report(supported=1, borderline=1)).reason == "borderline"
    assert decide(Report(supported=1)).decisive
    assert decide(Report(contradicted=1, borderline=9)).decisive, "a contradiction settles it"


def test_a_bifrost_rubric_sends_its_id_and_never_its_text() -> None:
    rubric = RubricPrompt(prompt_id="prompt_judge", version="3")
    request = rubric.request(
        model="m", question="q", answer="a", evidence="e", unsettled="borderline"
    )
    assert request.prompt_id == "prompt_judge" and request.prompt_version == "3"
    assert rubric.source == "bifrost:prompt_judge@3"
    body = request.messages[0]["content"]
    assert "Score from 0 to 1" not in body, "the stored prompt is the rubric; do not paste it"
    assert "ANSWER:\na" in body and "UNSETTLED CLAIMS:\nborderline" in body


def test_a_stored_rubric_without_a_pinned_version_names_the_id_alone() -> None:
    """Bifrost serves the latest version when none is pinned, so the verdict must not claim
    one; ``bifrost:<id>@None`` would be a version somebody went looking for."""
    rubric = RubricPrompt(prompt_id="prompt_judge")
    assert rubric.source == "bifrost:prompt_judge"
    request = rubric.request(model="m", question=None, answer="a", evidence="", unsettled="")
    assert request.prompt_id == "prompt_judge" and request.prompt_version is None


def test_what_counts_as_the_answer_is_one_function_for_memory_and_the_judge() -> None:
    """``answer_text`` is shared on purpose: two spellings would let the judge score something
    memory never recorded. So the claims branch is covered here rather than only through a
    memory-backed turn."""
    assert answer_text(AgentResponse.ok("  issued on Tuesday  ")) == "issued on Tuesday"
    assert answer_text(AgentResponse.ok("   ")) is None, "whitespace is not an answer"
    assert answer_text(AgentResponse.ok(None)) is None
    assert answer_text(AgentResponse.ok({"refund": True})) is None, (
        "structured data is not flattened into prose: an agent that wants it judged returns claims"
    )
    claimed = AgentResponse.ok(
        {"refund": True},
        claims=[
            Claim(claim_id="c1", text="the refund was issued"),
            Claim(claim_id="c2", text="on Tuesday"),
        ],
    )
    assert answer_text(claimed) == "the refund was issued\non Tuesday"


def test_the_builtin_rubric_is_used_and_named_when_no_prompt_id_is_configured() -> None:
    request = RubricPrompt().request(
        model="m", question=None, answer="a", evidence="", unsettled=""
    )
    body = request.messages[0]["content"]
    assert request.prompt_id is None
    assert BUILTIN_RUBRIC.splitlines()[0] in body
    assert "(not recorded)" in body and "(none retrieved)" in body
