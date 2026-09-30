"""The rubric the LLM stage of the judge reads.

Short, and about grounding: the grounded stage has already decided everything a classifier
can, so what reaches a model is a borderline or unsupported claim — or an offline item whose
reference answer the answer must be compared against.
"""

from __future__ import annotations

import json
from typing import Any, Final

#: The judge's answer shape (asked for as a strict JSON schema).
VERDICT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "label", "rationale"],
    "additionalProperties": False,
}

RUBRIC: Final = """You are grading one agent answer for groundedness and correctness.

Evidence the agent was given:
{evidence}

Reference answer (what a correct answer says; "(none)" when there is no reference):
{expected}

The question:
{question}

The answer:
{answer}

Automatic checking could not settle these claims:
{unsettled}

Score from 0 to 1:
1.0  every claim follows from the evidence, agrees with the reference, and answers the question
0.7  answered and defensible, with a claim the evidence only partly supports
0.4  partly answered, a claim the evidence does not support, or disagreement with the reference
0.0  contradicted by the evidence or the reference, or not an answer to the question

Answer with JSON only: {{"score": <number>, "label": "<one or two words>", \
"rationale": "<one sentence naming the claim that decided it>"}}"""

#: What the rubric may be shown of each part, so a judgement never costs more than the turn.
MAX_EVIDENCE_CHARS: Final = 4000
MAX_ANSWER_CHARS: Final = 4000


def messages(
    *,
    question: str | None,
    answer: str,
    evidence: str,
    expected: Any,
    unsettled: str,
) -> list[dict[str, Any]]:
    content = RUBRIC.format(
        question=question or "(not recorded)",
        answer=answer[:MAX_ANSWER_CHARS],
        evidence=evidence[:MAX_EVIDENCE_CHARS] or "(none retrieved)",
        expected=_text(expected),
        unsettled=unsettled or "(none)",
    )
    return [{"role": "user", "content": content}]


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True},
    }


def _text(value: Any) -> str:
    if value is None:
        return "(none)"
    return value if isinstance(value, str) else json.dumps(value, default=str)
