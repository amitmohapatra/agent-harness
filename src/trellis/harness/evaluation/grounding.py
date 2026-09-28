"""The deterministic half of the judge: citation validation and NLI, before any model.

The Memory Service already answers "is this answer supported by the evidence?" claim by claim
(``POST /v1/verify``: citations resolved, NLI, a contradiction scan). That answer is free, it
is reproducible, and it is about the evidence this very run was given. So it runs first, and
the LLM judge only ever sees what it could not settle — which is the whole cost argument in
design §11, stated as code.

What counts as settled is deliberately conservative. "No unsupported and no borderline claims"
is a score of 1.0 and nobody needs a model to confirm it. A contradiction is decisive the
other way: a model asked to re-litigate evidence that contradicts the answer is being asked to
overrule a measurement. Everything in between — an unsupported claim with no contradiction, a
borderline one — is exactly the case a rubric was written for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trellis.contracts.evaluation import JudgeMethod, JudgeVerdict


@dataclass(frozen=True, slots=True)
class GroundedDecision:
    """What the grounded stage decided. ``verdict is None`` means "ask the model"."""

    verdict: JudgeVerdict | None
    reason: str

    @property
    def decisive(self) -> bool:
        return self.verdict is not None


#: Nothing to judge: the report has no claims, so it says nothing about the answer.
EMPTY = GroundedDecision(None, "no_claims")
#: The service was not asked or could not answer.
UNAVAILABLE = GroundedDecision(None, "no_report")


def decide(report: Any) -> GroundedDecision:
    """The grounded verdict for one ``GroundingReport``, or ``None`` to escalate.

    Takes the report structurally rather than by type: the harness treats memory bundles as
    opaque everywhere else, and a judge is not the place to start importing the SDK's models.
    """
    if report is None:
        return UNAVAILABLE
    supported = _count(report, "supported")
    unsupported = _count(report, "unsupported")
    contradicted = _count(report, "contradicted")
    borderline = _count(report, "borderline")
    total = supported + unsupported + contradicted + borderline
    if total <= 0:
        return EMPTY
    metadata = {
        "supported": supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "borderline": borderline,
        "nli_provider": getattr(report, "nli_provider", "") or None,
    }
    if contradicted > 0:
        return GroundedDecision(
            _verdict(supported / total, "contradicted", metadata, report, total), "contradicted"
        )
    if unsupported == 0 and borderline == 0:
        return GroundedDecision(_verdict(1.0, "grounded", metadata, report, total), "grounded")
    return GroundedDecision(None, "borderline" if borderline else "unsupported")


def _verdict(
    score: float, label: str, metadata: dict[str, Any], report: Any, total: int
) -> JudgeVerdict:
    rate = getattr(report, "per_claim_hallucination_rate", None)
    return JudgeVerdict(
        score=max(0.0, min(1.0, score)),
        method=JudgeMethod.GROUNDED,
        label=label,
        rationale=_rationale(label, metadata, total),
        # The grounded stage spends nothing of the judge's budget: the service's own NLI is
        # what decided. A report that *did* consult the service's LLM says so in its own
        # metadata, and that spend is the service's, on the service's key.
        cost_usd=0.0,
        metadata={
            **{k: v for k, v in metadata.items() if v is not None},
            **({"per_claim_hallucination_rate": rate} if rate is not None else {}),
            "judge_consulted": getattr(report, "judge_consulted", 0),
        },
    )


def _rationale(label: str, counts: dict[str, Any], total: int) -> str:
    if label == "contradicted":
        return (
            f"{counts['contradicted']} of {total} claims are contradicted by the evidence "
            "the run was given"
        )
    return f"every claim ({total}) is supported by the evidence the run was given"


def _count(report: Any, field: str) -> int:
    value = getattr(report, field, 0)
    return value if isinstance(value, int) else 0


__all__ = ["EMPTY", "UNAVAILABLE", "GroundedDecision", "decide"]
