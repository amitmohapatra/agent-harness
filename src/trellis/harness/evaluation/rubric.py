"""The rubric the LLM judge reads, and where it comes from.

Prompts have one home and it is Bifrost (design §3, decision 3): the gateway injects a stored,
versioned prompt by id, and the trace records which version answered. So when
``judge.rubric_prompt_id`` is set, the harness sends the id and *not* the text — the rubric is
editable by the people who own it, without a deploy, and two deployments on the same prompt id
are judging by the same rubric by construction.

The built-in rubric is the fallback for a deployment that has not put one in the gateway yet.
It is deliberately short and deliberately about grounding, because the grounded stage has
already decided everything a classifier can decide: what reaches a model is a borderline or
unsupported claim, and the question is whether the answer is *defensible*, not whether it is
pleasant.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from trellis.contracts.model import ModelRequest

#: The judge's answer shape. ``strict`` so the gateway asks the provider for exactly this.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "label", "rationale"],
    "additionalProperties": False,
}

BUILTIN_RUBRIC = """You are grading one agent answer for groundedness and usefulness.

Evidence the agent was given:
{evidence}

The question:
{question}

The answer:
{answer}

Automatic checking could not settle these claims:
{unsettled}

Score from 0 to 1:
1.0  every claim follows from the evidence and the question is answered
0.7  answered and defensible, with a claim the evidence only partly supports
0.4  partly answered, or a claim the evidence does not support
0.0  contradicted by the evidence, or not an answer to the question

Answer with JSON only: {{"score": <number>, "label": "<one or two words>", \
"rationale": "<one sentence naming the claim that decided it>"}}"""


class RubricPrompt(BaseModel):
    """Which rubric judged, and where it came from.

    ``prompt_id`` wins: with one set, the template is never sent and the gateway's stored
    version is what the model reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "grounded-answer"
    prompt_id: str | None = None
    version: str | None = None
    template: str = BUILTIN_RUBRIC

    @property
    def source(self) -> str:
        """What the verdict records, so a score can be traced back to the rubric that gave it."""
        if self.prompt_id:
            return f"bifrost:{self.prompt_id}" + (f"@{self.version}" if self.version else "")
        return f"builtin:{self.name}"

    def request(
        self,
        *,
        model: str,
        question: str | None,
        answer: str,
        evidence: str,
        unsettled: str,
        max_tokens: int = 300,
    ) -> ModelRequest:
        """The model call this rubric means.

        With a ``prompt_id`` the gateway prepends the stored rubric and the harness sends only
        the material to grade — which is also why the material is labelled: a stored prompt
        cannot interpolate what it never saw.
        """
        filled = {
            "question": question or "(not recorded)",
            "answer": answer,
            "evidence": evidence or "(none retrieved)",
            "unsettled": unsettled or "(none)",
        }
        content = _material(filled) if self.prompt_id else self.template.format(**filled)
        return ModelRequest(
            model=model,
            prompt_id=self.prompt_id,
            prompt_version=self.version,
            messages=[{"role": "user", "content": content}],
            params={"temperature": 0.0, "max_tokens": max_tokens},
            metadata={"rubric": self.source},
        )


def _material(filled: dict[str, str]) -> str:
    return (
        f"QUESTION:\n{filled['question']}\n\n"
        f"ANSWER:\n{filled['answer']}\n\n"
        f"EVIDENCE:\n{filled['evidence']}\n\n"
        f"UNSETTLED CLAIMS:\n{filled['unsettled']}"
    )


__all__ = ["BUILTIN_RUBRIC", "VERDICT_SCHEMA", "RubricPrompt"]
