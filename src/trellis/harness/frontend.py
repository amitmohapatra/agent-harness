"""What a run's front end gives it — AG-UI's ``RunAgentInput`` ``tools``, ``context`` and
``state`` (a CopilotKit app's frontend actions, readables and shared state) — kept with the run
(``RunStart.metadata["frontend"]``), so every attempt has it, a resume on another worker too.

* **tools**: offered to the model beside the agent's own (one of the agent's by the same name
  wins). The front end runs a call, not the harness: the call pauses the run through the one
  pause (``Runtime.interrupt``), its ``tool_call`` the call — whose ``idempotency_key`` is the
  id its ``TOOL_CALL_*`` events carry — and the result the front end sends back (a resume) is
  the call's output, which the model reads as any tool's. Governed as a read, unless the
  tenant's catalog says otherwise: it runs in the person's own app, under their eyes.
* **context**: the ``{description, value}`` entries, which the model reads in the pushed
  context with the state, fitted to :data:`SHARE` of its token budget (the memory context gets
  the rest).
* **state**: the front end's shared state. The run's code reads and sets it
  (``Runtime.state``); each change is streamed as ``STATE_DELTA`` (a JSON Patch, RFC 6902), or
  ``STATE_SNAPSHOT`` where there was no object to patch (``Runtime.state``).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from trellis.contracts import InterruptReason, ToolCall, ToolSpec
from trellis.harness.asking import answer_of
from trellis.harness.journal import content_key
from trellis.harness.runtime import current, principal
from trellis.harness.tools.base import Tool

#: The key of ``RunStart.metadata`` the front end's part is kept under.
FRONTEND: Final = "frontend"
#: The key of a front-end tool's interrupt ``payload`` naming the tool (what tells a surface
#: the front end, not a person, answers it).
FRONTEND_TOOL: Final = "frontend_tool"
#: The most of the pushed context's token budget the front end's context and state take.
SHARE: Final = 0.5
#: Characters per token, to fit text to a token budget without a tokenizer (the usual
#: estimate for English and JSON).
CHARS_PER_TOKEN: Final = 4
#: What the model reads before the front end's context and state.
HEADING: Final = "From the app you work in:"
#: What the model reads where entries did not fit.
LEFT_OUT: Final = "({n} more left out: over the context budget)"
#: What ends a state cut to fit.
CUT: Final = " (cut: over the context budget)"
#: What the person's app is asked when the model calls one of its tools.
ASKED: Final = "Run {tool} in the app and send back its result"


@dataclass(frozen=True, slots=True)
class Frontend:
    """The front end's part of a run: its tools (each ``{name, description, parameters}``, the
    parameters a JSON Schema), its context entries (``{description, value}``) and its state."""

    tools: tuple[dict[str, Any], ...] = ()
    context: tuple[dict[str, Any], ...] = ()
    state: Any = None
    #: whether the front end gave a state (``None`` is a state too)
    stated: bool = field(default=False)

    @classmethod
    def of(cls, metadata: Mapping[str, Any]) -> Frontend | None:
        """The part kept with a run's record, or ``None``."""
        kept = metadata.get(FRONTEND)
        if not isinstance(kept, Mapping):
            return None
        return cls(
            tools=tuple(kept.get("tools") or ()),
            context=tuple(kept.get("context") or ()),
            state=kept.get("state"),
            stated="state" in kept,
        )

    def kept(self) -> dict[str, Any]:
        """What the run's record keeps (JSON)."""
        kept: dict[str, Any] = {}
        if self.tools:
            kept["tools"] = [dict(t) for t in self.tools]
        if self.context:
            kept["context"] = [dict(c) for c in self.context]
        if self.stated:
            kept["state"] = self.state
        return kept

    def offered(self, taken: Collection[str]) -> tuple[list[Tool], list[str]]:
        """Its tools as the run's (:func:`frontend_tool`), and the names left out because one
        of the agent's tools has them."""
        tools = [frontend_tool(t) for t in self.tools if t["name"] not in taken]
        return tools, [t["name"] for t in self.tools if t["name"] in taken]


def frontend_tool(declared: Mapping[str, Any]) -> Tool:
    """A tool the front end declared (AG-UI ``Tool``): calling it asks the front end to run it.
    Its answer is what the call returns (a cancelled answer cancels the run)."""
    name = str(declared["name"])
    spec = ToolSpec(
        name=name,
        description=str(declared.get("description") or ""),
        input_schema=declared.get("parameters") or {"type": "object", "properties": {}},
        side_effects="read",
    )

    async def run(args: dict[str, Any]) -> Any:
        runtime = current()
        assert runtime is not None, "a front-end tool is called in a run"
        call = ToolCall(tool=name, args=args, task=runtime.task, idempotency_key=runtime.call_id)
        resolution = await runtime.interrupt(
            content_key("frontend", name, args),
            reason=InterruptReason.QUESTION,
            question=ASKED.format(tool=name),
            ui="form",
            tool_call=call,
            payload={FRONTEND_TOOL: name},
            assignee=principal(runtime.user),
        )
        return answer_of(resolution)

    return Tool(spec, run)


def rendered(frontend: Frontend | None, state: Any, budget: int) -> tuple[str | None, int]:
    """The front end's context entries and its ``state`` as the model reads them, in at most
    ``budget`` tokens — entries whole, in order, until one does not fit (the rest said left
    out), the state last, cut where it does not fit — and the tokens it takes."""
    shown = frontend is not None and frontend.stated and state not in (None, {}, [])
    if frontend is None or not (frontend.context or shown):
        return None, 0
    entries = list(frontend.context)
    # every line ends with a newline; room is kept for saying what was left out
    room = budget * CHARS_PER_TOKEN - len(HEADING) - len(LEFT_OUT.format(n=len(entries))) - 1
    lines: list[str] = []
    for n, entry in enumerate(entries):
        line = f"- {entry.get('description', '')}: {entry.get('value', '')}"
        if len(line) + 1 > room:
            lines.append(LEFT_OUT.format(n=len(entries) - n))
            room = 0
            break
        lines.append(line)
        room -= len(line) + 1
    if shown and room > len(CUT) + 1:
        said = "- the app's shared state (JSON): " + json.dumps(state, default=str)
        fits = len(said) + 1 <= room
        lines.append(said if fits else said[: room - len(CUT) - 1] + CUT)
    text = "\n".join([HEADING, *lines])
    return text, -(-len(text) // CHARS_PER_TOKEN)
