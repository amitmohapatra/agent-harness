"""A question an MCP server asks in the middle of a call (MCP elicitation) as the run's own
``ask``: what the person is shown, and the answer the server gets back.

The server's form (``requestedSchema``: flat, each field a string, a number, a boolean or a
pick from values) becomes what ``trellis.current().ask`` takes, so every surface shows it as it
shows any question (AG-UI's interrupt, an inbox, A2A):

* one field that picks from values (``enum``, or ``oneOf``/``anyOf`` of ``const`` with a
  ``title``; an ``array`` of them picks several) is a choice — a dropdown — whose answer is the
  value (the values) the server gets as that field;
* any other form is a form with the server's schema as ``expects`` (a ``format: date`` field a
  date picker where the surface renders JSON Schema), whose answer is the content;
* a link to open (``mode: url``) is a choice of :data:`OPENED` or :data:`DECLINED`.

A rejected question is the server's ``decline``; a cancelled run its ``cancel``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from trellis.contracts import Option

#: The choice a link to open is: the person opened it, or declined to.
OPENED: Final = Option(value="accept", label="Opened it")
DECLINED: Final = Option(value="decline", label="Declined")


@dataclass(frozen=True)
class Elicited:
    """One server question as ``ask`` takes it (:attr:`asking`), and how its answer goes back
    (:meth:`content`). ``picks`` names the one field a choice answers."""

    question: str
    options: list[Option | str] = field(default_factory=list)
    multiple: bool = False
    expects: dict[str, Any] | None = None
    picks: str | None = None
    link: bool = False

    @property
    def asking(self) -> dict[str, Any]:
        """``ask``'s arguments."""
        return {
            "options": self.options,
            "multiple": self.multiple,
            "expects": self.expects,
        }

    def result(self, answer: Any) -> dict[str, Any]:
        """The server's ``ElicitResult`` for the person's answer (``False``: rejected)."""
        if answer is False:
            return {"action": "decline"}
        if self.link:
            return {"action": answer}
        content = {self.picks: answer} if self.picks is not None else answer
        return {"action": "accept", "content": content}


def elicited(params: Any) -> Elicited:
    """What the server asks (the SDK's ``ElicitRequestFormParams`` or
    ``ElicitRequestURLParams``), as the person is asked it."""
    if getattr(params, "mode", None) == "url":
        return Elicited(f"{params.message}\n{params.url}", options=[OPENED, DECLINED], link=True)
    schema = dict(params.requested_schema)
    properties = schema.get("properties") or {}
    if len(properties) == 1:
        [(name, wanted)] = properties.items()
        picked = _choice(wanted)
        if picked is not None:
            options, multiple = picked
            return Elicited(params.message, options=options, multiple=multiple, picks=name)
    return Elicited(params.message, expects=schema)


def _choice(wanted: Mapping[str, Any]) -> tuple[list[Option | str], bool] | None:
    """A field's values to pick from (a value with no title as it is), and whether it picks
    several (an ``array`` of them); ``None`` for a field that does not pick."""
    items = wanted.get("items") if wanted.get("type") == "array" else None
    several = isinstance(items, Mapping)
    source: Mapping[str, Any] = items if several else wanted
    titled = source.get("oneOf") or source.get("anyOf")
    if titled:
        return [_option(c["const"], c.get("title")) for c in titled], several
    values = source.get("enum")
    if not values:
        return None
    names = wanted.get("enumNames") or []  # the form before titled values (2025-06-18)
    labels = [*names, *[None] * len(values)]
    return [_option(v, n) for v, n in zip(values, labels, strict=False)], several


def _option(value: Any, title: str | None) -> Option | str:
    return Option(value=str(value), label=title) if title else str(value)
