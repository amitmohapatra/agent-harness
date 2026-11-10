"""``ask_user``: the model asks the person it works for one question — free text, a choice
among ``options`` (several picks with ``multiple``, one of them ``recommended`` with ``why``),
a ``date``, a ``date_range`` or a ``number``, or several such values at once as one form
(``fields``: each a :class:`Field`, its ``default`` a remembered value shown filled in) —
through the run's own pause (``Runtime.ask``), and reads the answer as the call's result.
When to ask at all is the harness's note with the memory context (``agent.ASKING``); the model
decides each question's shape.

A run a person is known to read (``Runtime.attended``: a chat front end's) is offered it by
itself (:func:`offered`; off: ``without=["ask"]``); any other run only when its agent declares
it (``ReAct(tools=[ask_user()])``), since a pause there may have nobody to answer it.

What a person sees follows from the arguments, for every surface:

* an app's own control: the interrupt names the ``ask_user`` component, with ``props``
  ``{"kind", "options"}`` — a dropdown, a date or range picker, a number field;
* any other surface: the control the interrupt's ``ui`` names (a choice, a form whose
  ``expects`` is the answer's JSON Schema, ``ui_schema`` its widget hints);
* plain chat: the question as text, saying what kind of answer it takes (:data:`SAID`): the
  options numbered, the recommended one first (on every surface, wherever the model listed
  it), marked with why. A typed answer fits too: a date, a range and a number may each be
  answered in words, and a choice by another answer ("Other").

A form is the same three: the ``ask_user`` component with ``props`` ``{"kind": "form",
"fields": [...]}``; a form whose ``expects`` is an object of the fields (each its kind's
schema, its ``title``, ``default`` and ``description``) with ``ui_schema`` their order and
widgets; and in plain chat the fields numbered, each with the value filled in and where it is
from, answered in words ("keep all", "change the quantity to 40"). Its answer is one object —
what the person gave over what was filled in — or their words.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel
from pydantic import Field as PydanticField

from trellis.contracts import ConfigurationError, Option, ToolError
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import Tool
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
#: What plain chat reads after the question, for each kind of answer, and of a form's field
#: (``{said}``: its kind or its options; ``default``, ``why``: the value filled in and where
#: it is from).
SAID: Final = {
    "options": "\n{options}\n(Pick one, or say another.)",
    "multiple": "\n{options}\n(Pick one or more, or say another.)",
    "recommended": " (Recommended: {why})",
    "date": " (a date, YYYY-MM-DD)",
    "date_range": " (a date range: its start and end, YYYY-MM-DD)",
    "number": " (a number)",
    "choice": " ({options})",
    "default": ": {default}",
    "why": " ({why})",
    "optional": ", if any",
    "form": "\n{fields}\n(Answer each, or say what to keep and what to change.)",
}
#: The ``props`` kind of a question with several fields.
FORM: Final = "form"


class Field(BaseModel):
    """One value asked for: every question is one field, or several in one form."""

    name: str = PydanticField(description="The value's key in the answer.")
    label: str = ""
    kind: Kind = "text"
    options: list[str] = PydanticField(default_factory=list)
    multiple: bool = False
    default: Any = PydanticField(default=None, description="The value shown filled in.")
    why: str = PydanticField(default="", description="Where the default is from, and when.")
    required: bool = True

    @property
    def picks(self) -> list[str]:
        """Its options, each once, the recommended one first: the one to pick is the one
        read first, on every surface (a single question's choice, a form's field)."""
        given = list(dict.fromkeys(self.options))
        best = self.best
        return given if best is None else [best, *(p for p in given if p != best)]

    @property
    def best(self) -> str | None:
        """The option recommended: the default, when it is one of them and says why."""
        return self.default if self.default in self.options and self.why else None

    @property
    def expects(self) -> dict[str, Any]:
        """The answer's JSON Schema: a choice's option (or another answer in words), else
        the kind's."""
        return CHOSEN[self.multiple] if self.picks else EXPECTS[self.kind]

    def in_form(self) -> dict[str, Any]:
        """Its property in a form's schema: its options to pick from, its label, the value
        filled in and where that is from."""
        given: dict[str, Any] = {"type": "string", "enum": self.picks} if self.picks else {}
        if self.picks and self.multiple:
            given = {"type": "array", "items": given, "uniqueItems": True}
        about = {"title": self.label or self.name, "default": self.default,
                 "description": self.why or None}  # fmt: skip
        return {**(given or self.expects), **{k: v for k, v in about.items() if v is not None}}

    def shown(self) -> dict[str, Any]:
        """What an app's own control gets of it (``props``): what was said of it."""
        said = {"label": self.label, "options": self.picks, "multiple": self.multiple,
                "default": self.default, "why": self.why}  # fmt: skip
        given = {k: v for k, v in said.items() if v not in (None, "", [], False)}
        return {"name": self.name, "kind": self.kind, **given, "required": self.required}

    def line(self, label: str = "") -> str:
        """What plain chat reads of it after ``label`` (a form's field: its own): the kind of
        answer or the options, the value filled in and where it is from."""
        said = SAID.get(self.kind, "")
        if self.picks:
            said = SAID["choice"].format(options=" or ".join(self.picks))
        if self.default is not None:
            filled = self.default
            several = ", ".join(map(str, filled)) if isinstance(filled, list) else filled
            said += SAID["default"].format(default=several)
            said += SAID["why"].format(why=self.why) if self.why else ""
        return f"{label}{said}" + ("" if self.required else SAID["optional"])


async def ask_question(
    question: str,
    options: list[str] | None = None,
    *,
    kind: Kind = "text",
    multiple: bool = False,
    recommended: str | None = None,
    why: str | None = None,
    fields: list[Field] | None = None,
) -> Any:
    """Ask the person you work for one short question and wait for the answer.

    Args:
        question: The question, naming what is missing.
        options: Only when the answer is one of a few known choices (2-4); none for an open
            question.
        kind: The kind of answer: text, date, date_range or number.
        multiple: Only when the choices can be combined.
        recommended: The option you recommend, or the value you remember (shown filled in);
            only for a real reason (say it in why).
        why: Why you recommend it, or where the value is from, in a few words.
        fields: Only when you need several values at once, or the person is to confirm values
            you remember: one entry a value, its default the remembered value. They answer
            all in one go.
    """
    runtime = current()
    if runtime is None:
        raise ConfigurationError("ask_user asks a person: it runs inside a harness run")
    asked = fields or [
        Field(
            name="",
            kind=kind,
            options=options or [],
            multiple=multiple,
            default=recommended,
            why=why or "",
        )
    ]
    if len(asked) > 1:
        return await _form(runtime, question, asked)
    one = asked[0]
    picks, best = one.picks, one.best
    choices = [
        Option(value=p, description=one.why, recommended=True) if p == best else p for p in picks
    ]
    numbered = "\n".join(
        f"{n}. {p}" + (SAID["recommended"].format(why=one.why) if p == best else "")
        for n, p in enumerate(picks, 1)
    )
    said = SAID["multiple" if one.multiple else "options"].format(options=numbered)
    # a value that is no option (a number, a date): shown filled in, with where it is from
    filled = {} if picks or one.default is None else {"default": one.default}
    about = {**filled, "why": one.why} if filled and one.why else filled
    return await runtime.ask(
        question.rstrip() + (said if picks else one.line()),
        options=choices or None,
        multiple=bool(picks) and one.multiple,
        expects={**one.expects, **filled},
        ui_schema=None if picks else WIDGETS.get(one.kind),
        component=ASK_USER,
        props={"kind": one.kind, "options": picks, **about},
    )


async def _form(runtime: Runtime, question: str, fields: list[Field]) -> Any:
    """Several values asked at once: one form (its schema an object of the fields, or the
    answer in words), answered in one go — what the person gave over what was filled in."""
    names = [f.name for f in fields]
    if len(set(names)) < len(names) or not all(names):
        raise ToolError("ask_user: each of a form's fields is named once", source="tools")
    lines = "\n".join(f"{n}. {f.line(f.label or f.name)}" for n, f in enumerate(fields, 1))
    widgets = {f.name: WIDGETS[f.kind] for f in fields if not f.picks and f.kind in WIDGETS}
    answer = await runtime.ask(
        question.rstrip() + SAID["form"].format(fields=lines),
        expects={
            "type": ["object", "string"],
            "properties": {f.name: f.in_form() for f in fields},
            # a field filled in is answered already: its default stands unless changed
            "required": [f.name for f in fields if f.required and f.default is None],
        },
        ui_schema={"ui:order": names, **widgets},
        component=ASK_USER,
        props={"kind": FORM, "fields": [f.shown() for f in fields]},
    )
    if not isinstance(answer, dict):
        return answer  # in words: the model reads what to keep and what to change
    filled = {f.name: f.default for f in fields if f.default is not None}
    return {**filled, **answer}


def ask_user() -> FunctionTool:
    """The ``ask_user`` tool (module docstring), governed as a read: it changes nothing."""
    return tool(ask_question, name=ASK_USER, side_effects="read")


#: What ``ask_user`` says before its own description, and of its ``kind``, in a run whose app
#: declared tools: they are the app's controls, and a value one of them asks for is asked
#: with it (the app's date picker, not ``kind="date"``). Measured on the asking set's
#: ``frontend`` cases (Haiku): the app's date picker used in 1 of 8 runs without it, in 13 of
#: 14 with it; said of the tool alone, in 9 of 23.
APPS: Final = "Not for a value one of the app's own controls asks for ({tools}): call that control."


def offered(apps: Sequence[str] = ()) -> Tool:
    """``ask_user`` as the harness offers it by itself (``Agent.tools_for``): the same tool,
    part of the ``ask`` feature (``without=["ask"]`` turns it off), naming the app's own
    tools (``apps``) as the controls to call for what they ask (:data:`APPS`)."""
    made = ask_user().tool
    spec = made.spec
    if apps:  # said where the choice is made: of the tool, and of the kind of answer
        said = APPS.format(tools=", ".join(apps))
        schema = copy.deepcopy(dict(spec.input_schema or {}))
        kind = schema["properties"]["kind"]
        kind["description"] = f"{kind['description']} {said}"
        spec = spec.model_copy(
            update={"description": f"{said} {spec.description}", "input_schema": schema}
        )
    return dataclasses.replace(made, spec=spec, feature="ask")


__all__ = ["ASK_USER", "ask_question", "ask_user", "offered"]
