"""``ContextAssembler``: what goes into the model each step, and how it stays bounded
(design §5).

The assembler owns the prompt budget. It renders the system prompt from its parts (the
prompt, the skills the agent may use, the memory bundle with citations) under a token
budget, and when the conversation outgrows the budget it compacts the older turns into a
summary written by the model and remembered as an observation, so the summary survives the
process. Token counts are estimated (four characters per token) so no tokenizer is needed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.model import ModelRequest

CHARS_PER_TOKEN: Final = 4
DEFAULT_BUDGET_TOKENS: Final = 8000
#: The fraction of the budget the conversation may occupy before compaction.
COMPACTION_THRESHOLD: Final = 0.75
#: How many of the most recent turns compaction always keeps verbatim.
KEEP_RECENT_TURNS: Final = 4
SUMMARY_KIND: Final = "AGENT_RESULT"
#: Request metadata marking a model call the harness makes for itself (a compaction); the
#: model client announces no message for it.
INTERNAL: Final = "internal"
#: How the loop labels a tool result turn; compaction recognises it by this.
OBSERVATION_PREFIX: Final = "Observation"
SUMMARY_PROMPT: Final = (
    "Summarise the conversation so far in at most 200 words, keeping every fact, decision, "
    "identifier and open question a later step needs. Reply with the summary only."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def estimate_turns(turns: Sequence[dict[str, Any]]) -> int:
    return sum(estimate_tokens(str(turn.get("content") or "")) for turn in turns)


@dataclass
class ContextAssembler:
    """Builds the system prompt and keeps the conversation under budget."""

    prompt: str
    skills: Sequence[Any] = ()
    budget_tokens: int = DEFAULT_BUDGET_TOKENS
    memory_tokens: int | None = None
    compaction_threshold: float = COMPACTION_THRESHOLD
    keep_recent: int = KEEP_RECENT_TURNS
    compactions: int = field(default=0, init=False)

    # ------------------------------------------------------------------ the system prompt
    def system_prompt(self, runtime: Any) -> str:
        parts = [self.prompt.strip()]
        skills = self._skills()
        if skills:
            parts.append("Skills you may use:\n" + "\n".join(f"- {s}" for s in skills))
        memory = self._memory(runtime)
        if memory:
            parts.append(memory)
        return "\n\n".join(p for p in parts if p)

    def _skills(self) -> list[str]:
        lines: list[str] = []
        for skill in self.skills:
            name = getattr(skill, "skill_id", None) or getattr(skill, "name", None) or str(skill)
            description = getattr(skill, "description", None)
            lines.append(f"{name}: {description}" if description else str(name))
        return lines

    def _memory(self, runtime: Any) -> str:
        """The bundle the interceptor fetched, rendered under the memory share of the budget."""
        bundle = getattr(runtime, "memory_context", None)
        if bundle is None:
            return ""
        rendered = getattr(bundle, "rendered", None) or getattr(bundle, "text", None)
        if not rendered:
            facts = getattr(runtime, "state", {}).get("memory_facts") or {}
            rendered = facts.get("rendered") if isinstance(facts, dict) else None
        if not rendered:
            return ""
        limit = (self.memory_tokens or self.budget_tokens // 2) * CHARS_PER_TOKEN
        text = str(rendered)
        if len(text) > limit:
            text = text[:limit] + "\n…"
        return "What is remembered (cite memory ids when you rely on them):\n" + text

    # ------------------------------------------------------------------ compaction
    def over_budget(self, turns: Sequence[dict[str, Any]]) -> bool:
        return estimate_turns(turns) > self.budget_tokens * self.compaction_threshold

    async def compact(
        self, runtime: Any, turns: list[dict[str, Any]], *, model: str | None = None
    ) -> list[dict[str, Any]]:
        """Replace the older turns with a summary the model writes; the summary is also
        remembered as an observation, so it outlives the process."""
        if len(turns) <= self.keep_recent + 1:
            return turns
        head, older, recent = turns[0], turns[1 : -self.keep_recent], turns[-self.keep_recent :]
        if not older:
            return turns
        if head.get("role") != "system":
            raise ValueError("compaction keeps the system turn first; the first turn is not one")
        memory = getattr(runtime, "memory", None)
        policy = getattr(memory, "policy", None)
        # tool observations reach the summary only when the memory policy lets tool
        # results be observed at all; otherwise the summary keeps the conversation only
        include_observations = bool(policy is not None and policy.observe_tool_results)
        transcript = "\n".join(
            f"{t.get('role')}: {t.get('content')}"
            for t in older
            if include_observations or not str(t.get("content", "")).startswith(OBSERVATION_PREFIX)
        )
        tracer = getattr(runtime, "tracer", None)
        span = (
            tracer.span("agent.compact", attributes={"compaction": self.compactions + 1})
            if tracer is not None
            else contextlib.nullcontext()
        )
        with span:
            response = await runtime.model.invoke(
                ModelRequest(
                    messages=[
                        {"role": "system", "content": SUMMARY_PROMPT},
                        {"role": "user", "content": transcript},
                    ],
                    model=model,
                    # the harness's own call: no surface shows the summary as an answer
                    metadata={INTERNAL: "compaction"},
                )
            )
        summary = (response.text or "").strip() or transcript[: self.budget_tokens]
        self.compactions += 1
        if memory is not None and getattr(memory, "enabled", False):
            await memory.observe(
                MemoryObservation(
                    content=f"Conversation summary: {summary}",
                    kind=SUMMARY_KIND,
                    # the run's own note: visible to this run and the one that spawned it
                    hints={"visibility": "RUN"},
                    metadata={"source": "compaction", "compaction": self.compactions},
                )
            )
        return [
            head,
            {"role": "assistant", "content": f"Summary of the conversation so far: {summary}"},
            *recent,
        ]
