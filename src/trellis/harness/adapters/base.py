"""The adapter contract: four functions per framework, nothing else.

* ``prepare_input(target, input, context)`` — the framework's input, with the pushed memory
  context as a system message;
* ``invoke(target, native_input, run)`` / ``stream(...)`` — run it (the stream yields text
  deltas, then an :class:`Output` with what ``invoke`` would have returned);
* ``extract(target, output)`` — the answer, the transcript, and the framework's own pause;
* ``resume_input(target, native_input, pending, resolution)`` — what continues a paused run.

``run`` carries the per-run tools (already converted by ``tools.convert`` for the format the
adapter names) and the runtime. Adapters never wrap models, never re-implement a loop, and
use only their framework's public API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from trellis.contracts import InterruptResolution
from trellis.harness.journal import Pending
from trellis.harness.runtime import Runtime
from trellis.harness.tools.base import Tool

ToolFormat = Literal["langchain", "openai_agents", "claude", "openai_chat", "none"]
Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Output:
    """The last item of a stream: what ``invoke`` returns."""

    value: Any


@dataclass(frozen=True, slots=True)
class NativePause:
    """A pause the framework itself reported: LangGraph's ``interrupt`` (its value is our
    interrupt when ``ask`` raised it), or an OpenAI Agents ``needs_approval`` tool."""

    value: Any
    native_id: str | None = None
    #: the framework's serialised run, when it resumes from one
    state: dict[str, Any] | None = None
    #: the tool call waiting for approval, when that is what the framework paused on
    tool: str | None = None
    args: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Extracted:
    answer: Any
    transcript: list[tuple[Role, str]] = field(default_factory=list)
    pause: NativePause | None = None


@dataclass(slots=True)
class Invocation:
    """One attempt, as an adapter sees it."""

    runtime: Runtime
    tools: Sequence[Tool]
    #: ``tools`` in the adapter's own format (``None`` when there are none)
    native_tools: Any = None


class Adapter(Protocol):
    name: ClassVar[str]
    tool_format: ClassVar[ToolFormat]
    #: the target's tools are fixed when it is built (a compiled graph): ``tools=`` is refused
    #: at wrap time, and the harness tools come from ``h.tools(...)`` instead
    fixed_tools: ClassVar[bool]

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any: ...

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any: ...

    def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]: ...

    def extract(self, target: Any, output: Any) -> Extracted: ...

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any: ...


def query_of(input: Any) -> str:
    """The text a memory lookup uses: the input itself, the last user message, or the
    first text field of a structured input."""
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        messages = input.get("messages")
        if isinstance(messages, list):
            return query_of(messages)
        for key in ("query", "question", "input", "prompt", "text", "message"):
            value = input.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(input, list):
        for message in reversed(input):
            role = (
                message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
            )
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if role in ("user", "human") and isinstance(content, str):
                return content
    return ""
