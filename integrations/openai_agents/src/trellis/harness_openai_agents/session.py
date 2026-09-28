"""``MemoryServiceSession``: the OpenAI Agents SDK's conversation, kept by the Memory Service.

The SDK's ``Session`` protocol is four methods — ``get_items``, ``add_items``, ``pop_item``,
``clear_session`` — and its shipped implementations keep the conversation in SQLite, in
OpenAI's hosted conversations, or in a database the application runs. None of them knows the
platform's boundary, so a conversation stored that way is invisible to the memory the rest of
the platform reasons over.

This session is the same four methods over the Memory Service thread the run is already
bound to (design §3: threads and messages live there, and nowhere else). It also carries the
compaction moment, because the SDK has none: ``OpenAIResponsesCompactionAwareSession`` exists
but is OpenAI's hosted Responses compaction, which a gateway does not proxy.

``pop_item`` is the one method that cannot be honest. The service's thread is an audited
record with no per-message delete, so a message that was written cannot be quietly unwritten;
rather than return ``None`` (which means "the conversation is empty") this raises when there
*is* something to pop, and says what to do instead.
"""

from __future__ import annotations

from typing import Any, Final

from agents.memory.session import SessionABC
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.model import ModelRequest

from trellis.harness.reasoning.assembler import (
    INTERNAL,
    KEEP_RECENT_TURNS,
    SUMMARY_KIND,
    SUMMARY_PROMPT,
)
from trellis.harness.runtime.propagation import require_runtime

__all__ = ["MemoryServiceSession"]

HINT: Final = "run the agent through harness.openai_agents.wrap(...) / .agent(...)"
#: How many messages one ``get_items`` asks the service for when the caller names no limit.
DEFAULT_WINDOW: Final = 50
#: Where a run's compaction summary is kept, so ``get_items`` returns it after a compaction.
SUMMARY_KEY: Final = "openai_agents.summary"
#: The role a compaction summary is replayed to the model under.
SUMMARY_ROLE: Final = "user"
FRAMEWORK: Final = "openai-agents"


class MemoryServiceSession(SessionABC):
    """The run's conversation, in the platform's own store.

        result = await Runner.run(agent, "hello", session=harness.openai_agents.session())

    With no ``memory`` the running execution's ``runtime.memory`` answers, so one session
    object can be handed to every run.
    """

    def __init__(
        self,
        memory: Any = None,
        *,
        session_id: str | None = None,
        keep_recent: int = KEEP_RECENT_TURNS,
    ) -> None:
        self._memory = memory
        self.keep_recent = keep_recent
        #: The SDK asks for this; the platform's answer is the thread the run is bound to.
        self.session_id = session_id or "trellis-thread"

    @property
    def memory(self) -> Any:
        return self._memory if self._memory is not None else require_runtime(hint=HINT).memory

    # ------------------------------------------------------------------ the protocol
    async def get_items(self, limit: int | None = None) -> list[Any]:
        """What was said in this thread, oldest first, as SDK input items."""
        memory = self.memory
        if not memory.enabled:
            return []
        messages = await memory.history(limit=limit or DEFAULT_WINDOW)
        items = [item for message in messages if (item := _as_item(message)) is not None]
        summary = self._summary()
        if summary is None:
            return items
        # after a compaction the model sees the summary, then the recent turns verbatim —
        # the same shape ``ContextAssembler.compact`` leaves behind
        return [
            {"role": SUMMARY_ROLE, "content": f"Summary of the conversation so far: {summary}"},
            *items[-self.keep_recent :],
        ]

    async def add_items(self, items: list[Any]) -> None:
        """Record this turn's messages on the thread."""
        memory = self.memory
        if not memory.enabled:
            return
        for item in items:
            role, text = _role_and_text(item)
            if not text:
                continue
            if role == "assistant":
                await memory.record_output(text)
            elif role in ("user", "human"):
                await memory.record_input(text)

    async def pop_item(self) -> Any | None:
        """Refuses when there is something to pop; ``None`` only when there is nothing.

        Removing the last message is not a thing the Memory Service offers: the thread is an
        audited record, and "forget" is a deliberate, audited operation on a memory rather
        than a silent rewrite of what was said. Returning ``None`` here would claim the
        conversation is empty, so an application undoing a turn would keep the message and
        never know.
        """
        memory = self.memory
        if not memory.enabled:
            return None
        if not await memory.history(limit=1):
            return None
        raise NotImplementedError(
            "the Memory Service thread has no per-message delete: pop_item() cannot remove "
            "what was recorded. Use clear_session() to end the thread, or keep the undo in "
            "your application's own state (see the adapter README)"
        )

    async def clear_session(self) -> None:
        """End the thread this run is bound to."""
        memory = self.memory
        if not memory.enabled:
            return
        await memory.chat.delete_thread()

    # ------------------------------------------------------------------ compaction
    async def compact(self, *, model: str | None = None) -> str | None:
        """Summarise the conversation and remember the summary (design §5).

        The SDK has no summarisation hook, so compaction is the session's job. The summary is
        written by the model through the harness's own model port and observed as a
        run-scoped memory, exactly as ``ContextAssembler.compact`` does — so it outlives the
        process and reads identically whichever loop produced it.
        """
        runtime = require_runtime(hint=HINT)
        memory = runtime.memory
        if not memory.enabled:
            return None
        messages = await memory.history(limit=DEFAULT_WINDOW)
        older = messages[: -self.keep_recent] if self.keep_recent else messages
        if not older:
            return None
        transcript = "\n".join(
            f"{role}: {text}"
            for role, text in (_role_and_text(_as_item(m) or {}) for m in older)
            if text
        )
        if not transcript:
            return None
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
        summary = (response.text or "").strip()
        if not summary:
            return None
        runtime.state[SUMMARY_KEY] = summary
        await memory.observe(
            MemoryObservation(
                content=f"Conversation summary: {summary}",
                kind=SUMMARY_KIND,
                # the run's own note: visible to this run and the one that spawned it
                hints={"visibility": "RUN"},
                metadata={"source": "compaction", "framework": FRAMEWORK},
            )
        )
        return summary

    def _summary(self) -> str | None:
        if self._memory is not None:
            return None
        runtime = require_runtime(hint=HINT)
        value = runtime.state.get(SUMMARY_KEY)
        return value if isinstance(value, str) else None


def _as_item(message: Any) -> dict[str, Any] | None:
    """One Memory Service message as an SDK input item."""
    role = _attr(message, "role")
    content = _attr(message, "content")
    if not role or not isinstance(content, str):
        return None
    return {"role": str(role).lower(), "content": content}


def _role_and_text(item: Any) -> tuple[str, str]:
    """An SDK input item's role and plain text, whatever shape it arrived in."""
    raw = item if isinstance(item, dict) else getattr(item, "model_dump", dict)()
    role = str(raw.get("role") or "").lower()
    content = raw.get("content")
    if isinstance(content, str):
        return role, content
    if isinstance(content, list):
        parts = []
        for part in content:
            block = part if isinstance(part, dict) else getattr(part, "model_dump", dict)()
            parts.append(str(block.get("text") or ""))
        return role, "".join(parts)
    return role, ""


def _attr(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
