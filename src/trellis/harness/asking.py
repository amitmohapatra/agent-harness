"""What a person is asked: one :class:`Question`, built from ``ask``'s arguments — the
``Interrupt`` it becomes and how its answer reads back.

``trellis.current().ask(...)`` (Way 1, every adapter) is ``Question(...)`` paused on in the
run; Way 2 builds the same ``Question`` and pauses its own run with it::

    question = Question("Which plans?", options=[Option(value="a", label="Plan A"), "b"],
                        multiple=True)
    await runs.pause(question.interrupt(tenant="acme", run_id=run.run_id), checkpoint=...)
    ...                                    # the person answers (agent-runs checks the answer)
    plans = question.answer(record.last_resolution)          # ["a", "b"]

Everything goes on the contracts' ``Interrupt`` as it is: ``options`` (strings or
``Option``\\ s), ``multiple``, ``expects`` (or ``form=``, a pydantic model whose JSON Schema is
``expects`` and which the answer is read back into), ``ui_schema``, and the asker's own screen
(``component``, with its ``props``). What the person sees follows from what is asked
(:attr:`Question.ui`). Whether an answer fits is ``trellis.runs.answers.answer_problem``, the
check agent-runs makes: there is no second copy here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel
from pydantic import ValidationError as PydanticError

from trellis.contracts import (
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    Option,
)
from trellis.harness.journal import content_key
from trellis.runs.answers import schema_problem


class RunCancelled(Exception):
    """A person cancelled the run while answering it."""


@dataclass(frozen=True)
class Question:
    """One question for a person, checked when it is made (``ConfigurationError``, saying why:
    an ``expects`` that is not a JSON Schema, ``form`` and ``expects`` both given, ``props``
    with no ``component``, options with the same value...).

    ``options`` a choice (``multiple``: several picks, the answer a list of values),
    ``table`` a table, ``diff=(before, after)`` a diff, otherwise a form (``expects`` its JSON
    Schema, or ``form`` a pydantic model); a table or diff with ``expects`` is a review.
    ``ui_schema`` gives the form's widget hints (react-jsonschema-form's ``uiSchema``);
    ``component`` names your own screen, rendered with ``props`` where a surface has it."""

    question: str
    expects: dict[str, Any] | None = None
    form: type[BaseModel] | None = None
    table: Sequence[dict[str, Any]] | None = None
    diff: tuple[Any, Any] | None = None
    options: Sequence[Option | str] = ()
    multiple: bool = False
    ui_schema: dict[str, Any] | None = None
    component: str | None = None
    props: dict[str, Any] | None = None
    assignee: str | None = None
    deadline: datetime | None = None
    escalate_to: str | None = None
    #: what the asker's ``expects`` is, ``form`` given or not
    schema: dict[str, Any] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.form is not None and self.expects is not None:
            raise ConfigurationError(
                f"cannot ask {self.question!r}: give form= (its schema is expects) or "
                "expects=, not both"
            )
        schema = self.form.model_json_schema() if self.form is not None else self.expects
        if problem := schema_problem(schema):
            raise ConfigurationError(f"cannot ask {self.question!r}: {problem}")
        object.__setattr__(self, "schema", schema)
        try:  # the contract's own rules, where the question is written
            self.interrupt(tenant="-", run_id="-")
        except PydanticError as exc:
            reasons = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
            raise ConfigurationError(f"cannot ask {self.question!r}: {reasons}") from exc

    @classmethod
    def described(cls, value: Any) -> Question:
        """The question a LangGraph graph's own ``interrupt(value)`` asks: ``value`` itself, or
        — a dict — its ``question`` with the same fields ``ask`` takes (:data:`DESCRIBED`:
        ``options``, ``multiple``, ``expects``, ``ui_schema``, ``component``, ``props``,
        ``assignee``), refused as ``ask`` refuses them (``ConfigurationError``)."""
        if not isinstance(value, Mapping):
            return cls(str(value))
        given = {name: value[name] for name in DESCRIBED if value.get(name) is not None}
        options = given.pop("options", [])
        if not isinstance(options, list):
            raise ConfigurationError(f"the graph's interrupt options are not a list: {options!r}")
        try:
            given["options"] = [
                o if isinstance(o, str) else Option.model_validate(o) for o in options
            ]
        except PydanticError as exc:
            raise ConfigurationError(
                f"the graph's interrupt has an option that is not one: {exc}"
            ) from exc
        return cls(str(value.get("question") or value), **given)

    # ------------------------------------------------------------------ the interrupt
    @property
    def ui(self) -> str:
        """The control a surface renders (the fallback of ``component``): ``choice`` for
        options, ``diff``, ``table``, else ``form``."""
        if self.options:
            return "choice"
        if self.diff is not None:
            return "diff"
        return "table" if self.table is not None else "form"

    @property
    def reason(self) -> InterruptReason:
        if self.options:
            return InterruptReason.CHOICE
        if self.ui in ("diff", "table") and self.schema is not None:
            return InterruptReason.REVIEW
        return InterruptReason.QUESTION

    @property
    def payload(self) -> dict[str, Any] | None:
        """What a table or a diff shows (the run stores a large one as a run artifact)."""
        if self.table is not None:
            return {"table": list(self.table)}
        if self.diff is not None:
            return {"diff": {"before": self.diff[0], "after": self.diff[1]}}
        return None

    @property
    def key(self) -> str:
        """The journal's key for it: the same question (text, kind, options, and what else
        it asks with) is the same entry, in order."""
        parts: list[Any] = [self.question, self.ui, [_jsonable(o) for o in self.options]]
        extras = {
            name: value
            for name, value in (
                ("multiple", self.multiple),
                ("expects", self.schema if self.form is not None else None),
                ("component", self.component),
            )
            if value
        }
        return content_key("ask", *parts, *([extras] if extras else []))

    def fields(self) -> dict[str, Any]:
        """The ``Interrupt``'s fields but its ids and its payload."""
        found: dict[str, Any] = {
            "reason": self.reason,
            "question": self.question,
            "ui": self.ui,
            "expects": self.schema,
            "options": list(self.options),
            "multiple": self.multiple,
            "ui_schema": self.ui_schema,
            "component": self.component,
            "props": self.props,
            "assignee": self.assignee,
            "deadline": self.deadline,
            "escalate_to": self.escalate_to,
        }
        return found

    def interrupt(self, *, tenant: str, run_id: str, interrupt_id: str | None = None) -> Interrupt:
        """The ``Interrupt`` a run of ``tenant`` pauses on (Way 2: ``runs.pause``), its payload
        inline (upload a large one yourself: ``runs.artifacts.upload``, then ``payload_ref``)."""
        ids = {"interrupt_id": interrupt_id} if interrupt_id is not None else {}
        return Interrupt(
            tenant_id=tenant, run_id=run_id, payload=self.payload, **ids, **self.fields()
        )

    # ------------------------------------------------------------------ the answer
    def answer(self, resolution: InterruptResolution) -> Any:
        """What the person answered: the answer (a list of values for ``multiple``), the
        corrected value of a review, read into ``form`` when there is one. A cancel raises
        :class:`RunCancelled`."""
        value = answer_of(resolution)
        if self.form is None or value is None or isinstance(value, bool):
            return value
        try:
            return self.form.model_validate(value)
        except PydanticError as exc:
            raise ConfigurationError(
                f"the answer to {self.question!r} is not a {self.form.__name__}: {exc}"
            ) from exc


#: What of a graph's ``interrupt(value)`` dict goes on the interrupt, as ``ask`` takes it.
DESCRIBED: Final = ("options", "multiple", "expects", "ui_schema", "component", "props", "assignee")


def answer_of(resolution: InterruptResolution) -> Any:
    """A decision as the asker reads it: the answer; ``True``/``False`` for approve/reject;
    the edited value for an edit; a cancel raises :class:`RunCancelled`."""
    decision = resolution.decision
    if decision is InterruptDecision.CANCEL:
        raise RunCancelled(f"cancelled by {resolution.reviewer or 'the reviewer'}")
    if decision is InterruptDecision.APPROVE:
        return True
    if decision is InterruptDecision.REJECT:
        return False
    if decision is InterruptDecision.EDIT:
        return resolution.payload
    return resolution.answer


def _jsonable(option: Option | str) -> Any:
    return option if isinstance(option, str) else option.model_dump(mode="json", exclude_none=True)


__all__ = ["Question", "RunCancelled", "answer_of"]
