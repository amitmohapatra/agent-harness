"""The judge: sampled, asynchronous, grounded first. Implements the contracts ``Judge`` port.

Stages, in this order and never another:

1. **reference** — offline, when the item has an ``expected_output`` and the answer equals it
   (normalised), the score is 1.0 with no model call;
2. **grounded** — the memory service's ``/v1/verify`` against the context the run was given.
   Deterministic and free. When the service's report says its own LLM judge was consulted,
   the harness does not pay for a second opinion: an undecided report then abstains;
3. **LLM** — only for what the stages above could not settle, through Bifrost, with the
   rubric (evidence, reference answer, and the unsettled claims).

A judge that cannot decide abstains (``None``): an unsampled run, an over-budget hour, no
text, a service that is down. Abstaining is never reported as a zero.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Final, Protocol

from trellis.contracts import AgentEvalEvent, AgentResponse, JudgeMethod, JudgeVerdict
from trellis.eval import grounding, rubric
from trellis.eval.budget import JudgeBudget

log = logging.getLogger("trellis.judge")

#: The rubric's model, through Bifrost.
JUDGE_MODEL: Final = "openrouter/openai/gpt-4.1-nano"
#: Output budget of one judgement (reasoning models spend some of it thinking).
JUDGE_MAX_TOKENS: Final = 1024
MAX_LABEL_CHARS: Final = 64
MAX_RATIONALE_CHARS: Final = 1000
_SPACE = re.compile(r"\s+")


class ChatModel(Protocol):
    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]: ...


class Verifier(Protocol):
    """Whatever answers "is this answer supported by that bundle?" (a run's memory scope)."""

    async def verify(self, answer: str, bundle: Any) -> Any: ...


class GroundedJudge:
    """The judge. Share one per process: the budget is per agent, not per run."""

    name = "grounded"

    def __init__(self, *, budget: JudgeBudget, model: ChatModel | None = None) -> None:
        self.budget = budget
        self.model = model

    def admits(self, agent_id: str, run_id: str) -> bool:
        """Would this run be judged? The cheap early-out; :meth:`verdict` reserves."""
        return bool(self.budget.admit(agent_id, run_id))

    async def judge(
        self, event: AgentEvalEvent, /, *, response: AgentResponse | None = None
    ) -> JudgeVerdict | None:
        """The contracts port: score a finished run from its event and response."""
        answer = response.data if response is not None and isinstance(response.data, str) else None
        if not answer:
            return None
        return await self.verdict(
            event,
            question=event.metadata.get("question"),
            answer=answer,
            expected=event.metadata.get("expected_output"),
            evidence="\n".join(event.metadata.get("evidence") or []),
        )

    async def verdict(
        self,
        event: AgentEvalEvent,
        *,
        question: str | None,
        answer: str,
        verifier: Verifier | None = None,
        bundle: Any = None,
        evidence: str = "",
        expected: Any = None,
    ) -> JudgeVerdict | None:
        """Score one answer, or abstain. Never raises: a judge that fails is a missing score."""
        if not answer.strip():
            return None
        admission = self.budget.reserve(event.agent_id, event.agent_run_id)
        if not admission:
            return None
        try:
            verdict = await self._decide(question, answer, verifier, bundle, evidence, expected)
        except Exception as exc:
            log.warning("judge failed for run %s: %s", event.agent_run_id, exc)
            return None
        if verdict is not None:
            self.budget.spend(event.agent_id, verdict.cost_usd or 0.0)
        return verdict

    async def _decide(  # noqa: PLR0917 - one private call site
        self,
        question: str | None,
        answer: str,
        verifier: Verifier | None,
        bundle: Any,
        evidence: str,
        expected: Any,
    ) -> JudgeVerdict | None:
        if expected is not None and _normal(answer) == _normal(expected):
            return JudgeVerdict(
                score=1.0,
                method=JudgeMethod.GROUNDED,
                label="matches_reference",
                rationale="the answer is the reference answer",
                cost_usd=0.0,
            )
        report = None
        if verifier is not None and bundle is not None:
            try:
                report = await verifier.verify(answer, bundle)
            except Exception as exc:
                log.warning("grounding check unavailable: %s", exc)
        decision = grounding.decide(report)
        if decision.verdict is not None:
            return decision.verdict
        if getattr(report, "judge_consulted", 0) or self.model is None:
            return None
        return await self._ask_model(question, answer, evidence, expected, decision.reason)

    async def _ask_model(
        self, question: str | None, answer: str, evidence: str, expected: Any, unsettled: str
    ) -> JudgeVerdict | None:
        assert self.model is not None
        reply = await self.model.complete(
            rubric.messages(
                question=question,
                answer=answer,
                evidence=evidence,
                expected=expected,
                unsettled=unsettled,
            ),
            model=JUDGE_MODEL,
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            response_format=rubric.response_format(),
        )
        data = _parsed(reply)
        score = data.get("score") if data else None
        if not isinstance(score, int | float):
            log.warning("the judge model answered without a score")
            return None
        return JudgeVerdict(
            score=max(0.0, min(1.0, float(score))),
            method=JudgeMethod.LLM,
            label=str(data.get("label") or "judged")[:MAX_LABEL_CHARS],  # type: ignore[union-attr]
            rationale=str(data.get("rationale") or "")[:MAX_RATIONALE_CHARS] or None,  # type: ignore[union-attr]
            model=str(reply.get("model") or JUDGE_MODEL),
            metadata={"escalated": unsettled, "reference": expected is not None},
        )


def _parsed(reply: dict[str, Any]) -> dict[str, Any] | None:
    try:
        content = reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _normal(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return _SPACE.sub(" ", text).strip().casefold()
