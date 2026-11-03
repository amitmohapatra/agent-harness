"""``ask_user``: the model asks the person it works for one question — free text, a choice
among ``options`` (several picks with ``multiple``, one of them ``recommended`` with ``why``),
a ``date``, a ``date_range`` or a ``number`` — through the run's own pause (``Runtime.ask``),
and reads the answer as the call's result. When to ask at all is the harness's note with the
memory context (``agent.ASKING``); the model decides each question's shape.

Opt-in (``ReAct(tools=[ask_user()])``, ``h.wrap(target, tools=[ask_user()])``).

What a person sees follows from the arguments, for every surface:

* an app's own control: the interrupt names the ``ask_user`` component, with ``props``
  ``{"kind", "options"}`` — a dropdown, a date or range picker, a number field;
* any other surface: the control the interrupt's ``ui`` names (a choice, a form whose
  ``expects`` is the answer's JSON Schema, ``ui_schema`` its widget hints);
* plain chat: the question as text, saying what kind of answer it takes (:data:`SAID`): the
  options numbered, the recommended one marked with why. A typed answer fits too: a date, a
  range and a number may each be answered in words, and a choice by another answer ("Other").
"""

from __future__ import annotations

from typing import Any, Final, Literal

from trellis.contracts import ConfigurationError, Option
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
#: A choice's answer: an option's value, or another answer in words ("Other"); with
#: ``multiple``, a list of them.
CHOSEN: Final[dict[bool, dict[str, Any]]] = {
    False: EXPECTS["text"],
    True: {"type": ["array", "string"], "items": {"type": "string"}},
}
#: The widget hints for each kind (react-jsonschema-form's ``uiSchema``).
WIDGETS: Final[dict[str, dict[str, Any]]] = {
    "date": {"ui:widget": "date"},
    "date_range": {"start": {"ui:widget": "date"}, "end": {"ui:widget": "date"}},
    "number": {"ui:widget": "updown"},
}
#: What plain chat reads after the question, for each kind of answer.
SAID: Final = {
    "options": "\n{options}\n(Pick one, or say another.)",
    "multiple": "\n{options}\n(Pick one or more, or say another.)",
    "recommended": " (Recommended: {why})",
    "date": " (a date, YYYY-MM-DD)",
    "date_range": " (a date range: its start and end, YYYY-MM-DD)",
    "number": " (a number)",
}


async def ask_question(
    question: str,
    options: list[str] | None = None,
    *,
    kind: Kind = "text",
    multiple: bool = False,
    recommended: str | None = None,
    why: str | None = None,
) -> Any:
    """Ask the person you work for one short question and wait for the answer.

    Args:
        question: The question, naming what is missing.
        options: Only when the answer is one of a few known choices (2-4); none for an open
            question.
        kind: The kind of answer: text, date, date_range or number.
        multiple: Only when the choices can be combined.
        recommended: The option you recommend, only for a real reason (say it in why).
        why: Why you recommend it, in a few words.
    """
    runtime = current()
    if runtime is None:
        raise ConfigurationError("ask_user asks a person: it runs inside a harness run")
    picks = list(dict.fromkeys(options or []))
    best = recommended if recommended in picks and why else None
    choices = [
        Option(value=p, description=why, recommended=True) if p == best else p for p in picks
    ]
    numbered = "\n".join(
        f"{n}. {p}" + (SAID["recommended"].format(why=why) if p == best else "")
        for n, p in enumerate(picks, 1)
    )
    said = SAID["multiple" if multiple else "options"].format(options=numbered)
    return await runtime.ask(
        question.rstrip() + (said if picks else SAID.get(kind, "")),
        options=choices or None,
        multiple=bool(picks) and multiple,
        expects=CHOSEN[bool(multiple)] if picks else EXPECTS[kind],
        ui_schema=None if picks else WIDGETS.get(kind),
        component=ASK_USER,
        props={"kind": kind, "options": picks},
    )


def ask_user() -> FunctionTool:
    """The ``ask_user`` tool (module docstring), governed as a read: it changes nothing."""
    return tool(ask_question, name=ASK_USER, side_effects="read")


__all__ = ["ASK_USER", "ask_question", "ask_user"]
