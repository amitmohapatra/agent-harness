"""What a turn's *answer* is, as one function.

Two things need it and must agree: the memory writeback (what gets observed as the agent's
output) and the judge (what gets scored). Two spellings of "the answer" would let a judge
score something memory never recorded.
"""

from __future__ import annotations

from trellis.contracts.messages import AgentResponse


def answer_text(result: AgentResponse) -> str | None:
    """The agent's answer as text, or ``None`` when it did not produce one.

    Text data is the answer verbatim; a structured result is deliberately *not* flattened
    into prose — an agent that wants its structure judged or remembered returns claims.
    """
    if isinstance(result.data, str) and result.data.strip():
        return result.data.strip()
    if result.claims:
        return "\n".join(c.text for c in result.claims)
    return None


__all__ = ["answer_text"]
