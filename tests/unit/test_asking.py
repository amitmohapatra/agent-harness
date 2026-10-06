"""``Question`` (what ``ask`` builds) and ``Reviewer`` (scripted answers): the interrupt a question
is, its journal key, its answer read back; and what a script entry means."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, field_validator

from trellis.contracts import (
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    Option,
    RunStatus,
    ToolCall,
)
from trellis.harness.asking import Question, RunCancelled, answer_of
from trellis.harness.journal import content_key
from trellis.harness.result import Result
from trellis.testing import ANY, Decide, Reviewer


class Size(BaseModel):
    size: int

    @field_validator("size")
    @classmethod
    def _even(cls, value: int) -> int:
        if value % 2:
            raise ValueError("sizes are even")
        return value


def resolution(decision: str, answer: Any = None, payload: Any = None) -> InterruptResolution:
    return InterruptResolution(
        interrupt_id="run_1.1.1",
        run_id="run_1",
        decision=InterruptDecision(decision),
        answer=answer,
        payload=payload,
    )


def test_what_is_asked_decides_the_control_and_the_reason() -> None:
    assert (Question("Which?", options=["a"]).ui, Question("Which?", options=["a"]).reason) == (
        "choice",
        InterruptReason.CHOICE,
    )
    review = Question("Fix?", diff=("a", "b"), expects={"type": "string"})
    assert (review.ui, review.reason, review.payload) == (
        "diff",
        InterruptReason.REVIEW,
        {"diff": {"before": "a", "after": "b"}},
    )
    rows = Question("Rows?", table=[{"a": 1}])
    assert (rows.ui, rows.reason, rows.payload) == (
        "table",
        InterruptReason.QUESTION,
        {"table": [{"a": 1}]},
    )
    plain = Question("Why?")
    assert (plain.ui, plain.reason, plain.payload, plain.schema) == (
        "form",
        InterruptReason.QUESTION,
        None,
        None,
    )


def test_a_plain_question_keeps_the_journal_key_it_always_had() -> None:
    old = content_key("ask", "Which?", "choice", ["a", "b"])
    assert Question("Which?", options=["a", "b"]).key == old
    assert Question("Which?", options=["a", "b"], multiple=True).key != old
    labelled = Question("Which?", options=[Option(value="a", label="A"), "b"])
    assert labelled.key == content_key(
        "ask", "Which?", "choice", [{"value": "a", "label": "A"}, "b"]
    )
    assert Question("Size?", form=Size).key != Question("Size?").key


def test_way_2_builds_the_same_interrupt() -> None:
    question = Question(
        "Which plans?",
        options=[Option(value="a", label="Plan A"), "b"],
        multiple=True,
        component="plan-picker",
        props={"customer": "acme"},
        assignee="role:sales",
    )
    interrupt = question.interrupt(tenant="acme", run_id="run_1", interrupt_id="run_1.1.1")
    assert isinstance(interrupt, Interrupt)
    assert (interrupt.interrupt_id, interrupt.multiple, interrupt.component) == (
        "run_1.1.1",
        True,
        "plan-picker",
    )
    assert interrupt.option_values == ["a", "b"]
    assert question.interrupt(tenant="acme", run_id="run_1").interrupt_id.startswith("int_")


def test_an_answer_reads_back_as_asked() -> None:
    sized = Question("Size?", form=Size)
    assert sized.answer(resolution("ANSWER", {"size": 4})) == Size(size=4)
    assert sized.answer(resolution("ANSWER")) is None
    assert sized.answer(resolution("REJECT")) is False
    with pytest.raises(ConfigurationError, match=r"(?s)is not a Size: .*sizes are even"):
        sized.answer(resolution("ANSWER", {"size": 3}))
    assert answer_of(resolution("APPROVE")) is True
    assert answer_of(resolution("EDIT", payload={"x": 1})) == {"x": 1}
    with pytest.raises(RunCancelled, match="cancelled by the reviewer"):
        answer_of(resolution("CANCEL"))


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"form": Size, "expects": {"type": "string"}}, "not both"),
        ({"expects": {"type": 5}}, "not a valid JSON Schema"),
        ({"props": {"a": 1}}, "props are a component's"),
        ({"options": ["a", "a"]}, "distinct values"),
        ({"multiple": True}, "multiple needs options"),
        ({"escalate_to": "role:x"}, "escalate_to needs a deadline"),
    ],
)
def test_a_question_that_cannot_be_asked_is_refused_saying_why(
    fields: dict[str, Any], why: str
) -> None:
    with pytest.raises(ConfigurationError, match=why):
        Question("Which?", **fields)


# --------------------------------------------------------------------------- Reviewer


def approval(tool: str = "refund", **fields: Any) -> Interrupt:
    return Interrupt(
        interrupt_id="run_1.1.1",
        tenant_id="t",
        run_id="run_1",
        reason=InterruptReason.APPROVAL,
        question=f"Approve {tool}?",
        tool_call=ToolCall(tool=tool, args={"amount": 5}),
        **fields,
    )


def question(text: str = "Which?", **fields: Any) -> Interrupt:
    return Interrupt(
        interrupt_id="run_1.1.1", tenant_id="t", run_id="run_1", question=text, **fields
    )


def test_a_script_entry_is_found_by_tool_component_question_then_any() -> None:
    reviewer = Reviewer(
        {
            "refund": "approve",
            "plan-picker": ["a"],
            "Which?": "b",
            ANY: Decide("cancel"),
        },
        name="qa",
    )
    assert reviewer.decide(approval()) == ("refund", Decide("approve"))
    assert reviewer.decide(question("Pick", component="plan-picker"))[1] == Decide("answer", ["a"])
    assert reviewer.decide(question()) == ("Which?", Decide("answer", "b"))
    assert reviewer.decide(question("Else?"))[0] == ANY
    given = reviewer.resolution(approval())
    assert (given.decision, given.reviewer) == (InterruptDecision.APPROVE, "qa")
    with pytest.raises(LookupError, match=r"no scripted answer for 'Else\?'"):
        Reviewer({}).decide(question("Else?"))


def test_what_a_script_entry_means_for_an_approval() -> None:
    def meant(entry: Any) -> Decide:
        return Reviewer({"refund": entry}).decide(approval())[1]

    assert meant(True) == Decide("approve") and meant(False) == Decide("reject")
    assert meant("REJECT") == Decide("reject")
    assert meant({"amount": 1}) == Decide("edit", {"amount": 1})
    assert meant(lambda i: i.tool_call.args["amount"] < 10) == Decide("approve")
    assert meant(Decide("approve", remember="run")).remember == "run"
    with pytest.raises(ValueError, match="an approval is answered"):
        meant("maybe")
    edited = Reviewer({"refund": {"amount": 1}}).resolution(approval())
    assert (edited.decision, edited.payload, edited.answer) == (
        InterruptDecision.EDIT,
        {"amount": 1},
        None,
    )


async def test_settle_answers_until_the_run_is_done_or_queued_and_stops_a_loop() -> None:
    class Looping:
        def __init__(self, status: RunStatus) -> None:
            self.status = status
            self.resumed: list[tuple[str, Any]] = []

        async def resume(self, interrupt_id: str, decision: Any, **kw: Any) -> Result:
            self.resumed.append((interrupt_id, decision))
            return Result(run_id="run_1", status=self.status, interrupt=question())

    paused = Result(run_id="run_1", status=RunStatus.PAUSED, interrupt=question())
    queued = Looping(RunStatus.QUEUED)
    done = await Reviewer({"Which?": "a"}).settle(queued, paused)  # type: ignore[arg-type]
    assert done.status is RunStatus.QUEUED and queued.resumed == [
        ("run_1.1.1", InterruptDecision.ANSWER)
    ]
    with pytest.raises(RuntimeError, match="asked more than 50 times"):
        await Reviewer({"Which?": "a"}).settle(Looping(RunStatus.PAUSED), paused)  # type: ignore[arg-type]


def test_a_graphs_interrupt_value_is_the_question_ask_would_build() -> None:
    assert Question.described("Title?") == Question("Title?")
    assert Question.described({"topic": "x"}).question == "{'topic': 'x'}"
    picked = Question.described({"question": "Which?", "options": ["a", {"value": "b"}]})
    assert picked.options == ["a", Option(value="b")] and picked.ui == "choice"
