"""The online judge (design §11): sampled, asynchronous, grounded first.

    judge = GroundedJudge(model=BifrostModelClient(gateway, api_key=budgeted_key),
                          config=harness.config.judge)
    harness = AgentHarness(memory=memory, judge=judge)

What it is: an implementation of the contracts ``Judge`` port that scores a *finished* run.
Two stages, in this order and never the other one:

1. **grounded** — the Memory Service's ``/v1/verify`` against the very bundle the run was
   given. Deterministic, reproducible, and free of the judge's budget.
2. **LLM** — only for what stage 1 could not settle, through ``BifrostModelClient`` with a
   rubric versioned in the gateway.

A judge that cannot decide **abstains** (returns ``None``). Abstaining is a first-class
answer: an unsampled run, an over-budget hour, an answer with no text in it and a memory
service that is down all produce no score rather than a guessed one.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trellis.contracts.evaluation import JudgeMethod, JudgeVerdict
from trellis.contracts.events import AgentEvalEvent
from trellis.contracts.messages import AgentResponse

from trellis.harness.config.settings import JudgeConfig
from trellis.harness.evaluation import grounding
from trellis.harness.evaluation.answers import answer_text
from trellis.harness.evaluation.budget import JudgeBudget
from trellis.harness.evaluation.rubric import VERDICT_SCHEMA, RubricPrompt
from trellis.harness.runtime.logging import get_logger

log = get_logger("trellis.harness.judge")

#: How much of the retrieved evidence a rubric is shown. A judge that reads a whole bundle
#: costs more in prompt tokens than the turn it is judging.
MAX_EVIDENCE_CHARS = 4000
#: The same bound for the answer: a judge grades what the user saw, not an essay.
MAX_ANSWER_CHARS = 4000
#: Bounds on what the *model* sends back. Its label and rationale travel onward into a span
#: attribute, a trace score and a feedback row, so neither may be as long as the model likes.
MAX_LABEL_CHARS = 64
MAX_RATIONALE_CHARS = 1000


@runtime_checkable
class GroundingVerifier(Protocol):
    """Whatever can answer "is this answer supported?" for one run's scope. The harness's
    ``MemoryRuntime`` is one; a test's scripted verifier is another."""

    async def verify(self, answer: str, /, **options: Any) -> Any: ...


@runtime_checkable
class ScopedJudge(Protocol):
    """A judge that is bound to one run before it scores it.

    The ``Judge`` port takes an event and a response, and an event carries references, not a
    memory scope — deliberately, so enabling evaluation never widens what the harness holds.
    Binding is how a grounded judge gets the run's own evidence without that changing.
    """

    def bound(self, *, verifier: Any = None, bundle: Any = None, question: str | None = None): ...


class GroundedJudge:
    """Implements :class:`trellis.contracts.ports.Judge`. Safe to share; ``bound`` copies."""

    name = "grounded"

    def __init__(
        self,
        *,
        model: Any = None,
        config: JudgeConfig | None = None,
        rubric: RubricPrompt | None = None,
        budget: JudgeBudget | None = None,
        verifier: GroundingVerifier | None = None,
        bundle: Any = None,
        question: str | None = None,
    ) -> None:
        self.config = config or JudgeConfig()
        self.model = model
        self.rubric = rubric or RubricPrompt(
            prompt_id=self.config.rubric_prompt_id, version=self.config.rubric_prompt_version
        )
        #: Shared across bound copies on purpose: the ceilings are per agent, not per run.
        self.budget = budget or JudgeBudget(self.config)
        self.verifier = verifier
        self.bundle = bundle
        self.question = question

    # ------------------------------------------------------------------ binding
    def bound(
        self,
        *,
        verifier: GroundingVerifier | None = None,
        bundle: Any = None,
        question: str | None = None,
    ) -> GroundedJudge:
        """This judge, looking at one run's memory scope. Cheap: shares config and budget.

        ``type(self)`` rather than ``GroundedJudge``: binding is the last thing that happens
        before a run is scored, and a bound copy that had quietly become the base class would
        discard whatever a subclass overrode — the judge that scored would not be the judge the
        deployment installed.
        """
        clone = type(self).__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.verifier = verifier if verifier is not None else self.verifier
        clone.bundle = bundle if bundle is not None else self.bundle
        clone.question = question if question is not None else self.question
        return clone

    # ------------------------------------------------------------------ the port
    async def judge(
        self, event: AgentEvalEvent, /, *, response: AgentResponse | None = None
    ) -> JudgeVerdict | None:
        """Score the finished run, or abstain. Never raises into the caller: a judge that
        fails is a missing score, not a failed run."""
        answer = answer_text(response) if response is not None else None
        if not answer:
            # Nothing to score, so nothing to charge for: checked before the slot is taken.
            return None
        admission = self.budget.reserve(event.agent_id, event.agent_run_id)
        if not admission:
            log.debug("judge.skipped", agent_id=event.agent_id, reason=admission.reason)
            return None
        answer = answer[:MAX_ANSWER_CHARS]
        try:
            verdict = await self._decide(answer)
        except Exception as exc:
            log.warning("judge.failed", agent_id=event.agent_id, error=str(exc))
            return None
        if verdict is None:
            return None
        self.budget.spend(event.agent_id, verdict.cost_usd or 0.0)
        return verdict

    def admits(self, event: AgentEvalEvent) -> bool:
        """Whether this run *would* be judged, taking nothing. The interceptor's early-out, so
        an unsampled run costs a hash and not a binding; :meth:`judge` is what reserves."""
        admission = self.budget.admit(event.agent_id, event.agent_run_id)
        if not admission:
            log.debug("judge.skipped", agent_id=event.agent_id, reason=admission.reason)
        return bool(admission)

    # ------------------------------------------------------------------ stages
    async def _decide(self, answer: str) -> JudgeVerdict | None:
        decision = grounding.decide(await self._report(answer))
        if decision.verdict is not None:
            return decision.verdict
        if self.config.grounded_only or self.model is None:
            # Nothing spent and nothing claimed: the classifier could not settle it and this
            # deployment does not buy a second opinion.
            return None
        return await self._ask_model(answer, unsettled=decision.reason)

    async def _report(self, answer: str) -> Any:
        """The grounding report for this answer, against the run's own bundle when there is
        one (free: the evidence is already assembled) and a fresh retrieval otherwise."""
        verifier = self.verifier
        if verifier is None:
            return None
        options: dict[str, Any] = {"bundle": self.bundle} if self.bundle is not None else {}
        if not options and self.question:
            options["query"] = self.question
        if not options:
            return None
        try:
            return await verifier.verify(answer, **options)
        except Exception as exc:
            log.warning("judge.verify_failed", error=str(exc))
            return None

    async def _ask_model(self, answer: str, *, unsettled: str) -> JudgeVerdict | None:
        request = self.rubric.request(
            model=self.config.model,
            question=self.question,
            answer=answer,
            evidence=_evidence(self.bundle),
            unsettled=unsettled,
        )
        result = await self.model.structured(request, schema=VERDICT_SCHEMA)
        data = result.data if isinstance(result.data, dict) else None
        if data is None:
            log.warning("judge.unparsable_verdict", model=self.config.model)
            return None
        score = data.get("score")
        if not isinstance(score, int | float):
            return None
        usage = result.usage
        return JudgeVerdict(
            score=max(0.0, min(1.0, float(score))),
            method=JudgeMethod.LLM,
            label=str(data.get("label") or "judged")[:MAX_LABEL_CHARS],
            rationale=str(data.get("rationale") or "")[:MAX_RATIONALE_CHARS] or None,
            model=result.model or self.config.model,
            cost_usd=usage.cost_usd if usage is not None else None,
            metadata={
                "rubric": self.rubric.source,
                "escalated": unsettled,
                **({"tokens": usage.tokens} if usage is not None and usage.tokens else {}),
            },
        )


def _evidence(bundle: Any) -> str:
    """The evidence the run was given, as the rubric sees it: the bundle's own rendering,
    truncated. Truncated rather than summarised — a summary of the evidence would be one more
    model call, judged against itself."""
    if bundle is None:
        return ""
    rendered = getattr(bundle, "rendered", None)
    text = rendered if isinstance(rendered, str) else str(bundle)
    return text[:MAX_EVIDENCE_CHARS]


__all__ = [
    "MAX_ANSWER_CHARS",
    "MAX_EVIDENCE_CHARS",
    "MAX_LABEL_CHARS",
    "MAX_RATIONALE_CHARS",
    "GroundedJudge",
    "GroundingVerifier",
    "ScopedJudge",
]
