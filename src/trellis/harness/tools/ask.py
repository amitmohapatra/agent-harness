"""``ask_user``: the model asks the person it works for one question — free text, a choice
among ``options``, a ``date``, a ``date_range`` or a ``number`` — through the run's own pause
(``Runtime.ask``), and reads the answer as the call's result.

Opt-in (``ReAct(tools=[ask_user()])``, ``h.wrap(target, tools=[ask_user()])``): whether a
model should ask or guess is being measured (the improvement backlog's clarifying questions).

What a person sees follows from the arguments, for every surface:

* an app's own control: the interrupt names the ``ask_user`` component, with ``props``
  ``{"kind", "options"}`` — a dropdown, a date or range picker, a number field;
* any other surface: the control the interrupt's ``ui`` names (a choice, a form whose
  ``expects`` is the answer's JSON Schema, ``ui_schema`` its widget hints);
* plain chat: the question as text, saying what kind of answer it takes (:data:`SAID`). A
  typed answer fits too: a date, a range and a number may each be answered in words.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from trellis.contracts import ConfigurationError
from trellis.harness.runtime import current
from trellis.harness.tools.sources import FunctionTool, tool

#: What kind of answer a question takes.
Kind = Literal["text", "date", "date_range", "number"]
#: The tool's name, and the component an app renders its question with.
ASK_USER: Final = "ask_user"
_DATE: Final = {"type": "string", "format": "date"}
#: The answer's JSON Schema for each kind: the structured value, or — so plain chat can
#: answer — text.
EXPECTS: Final[dict[str, dict[str, Any]]] = {
    "text": {"type": "string"},
    "date": _DATE,
    "date_range": {
        "type": ["object", "string"],
        "properties": {"start": _DATE, "end": _DATE},
        "required": ["start", "end"],
    },
    "number": {"type": ["number", "string"]},
}
#: The widget hints for each kind (react-jsonschema-form's ``uiSchema``).
WIDGETS: Final[dict[str, dict[str, Any]]] = {
    "date": {"ui:widget": "date"},
    "date_range": {"start": {"ui:widget": "date"}, "end": {"ui:widget": "date"}},
    "number": {"ui:widget": "updown"},
}
#: What plain chat reads after the question, for each kind of answer.
SAID: Final = {
    "options": " (one of: {options})",
    "date": " (a date, YYYY-MM-DD)",
    "date_range": " (a date range: its start and end, YYYY-MM-DD)",
    "number": " (a number)",
}


async def _ask_user(question: str, options: list[str] | None = None, kind: Kind = "text") -> Any:
    """Ask the person you work for one short question and wait for the answer: only when you
    need a value they did not give and no tool can find it. Give options when there are a few
    likely answers; say the kind of answer when it is a date, a date range or a number.

    Args:
        question: The question, naming what is missing.
        options: The likely answers to choose from, when there are a few.
        kind: The kind of answer: text, date, date_range or number.
    """
    runtime = current()
    if runtime is None:
        raise ConfigurationError("ask_user asks a person: it runs inside a harness run")
    picks = list(dict.fromkeys(options or []))
    hint = SAID["options"].format(options=", ".join(picks)) if picks else SAID.get(kind, "")
    return await runtime.ask(
        question.rstrip() + hint,
        options=picks or None,
        expects=None if picks else EXPECTS[kind],
        ui_schema=None if picks else WIDGETS.get(kind),
        component=ASK_USER,
        props={"kind": kind, "options": picks},
    )


def ask_user() -> FunctionTool:
    """The ``ask_user`` tool (module docstring), governed as a read: it changes nothing."""
    return tool(_ask_user, name=ASK_USER, side_effects="read")


__all__ = ["ASK_USER", "ask_user"]
