from __future__ import annotations

from trellis.contracts import Interrupt, InterruptDecision, InterruptResolution
from trellis.harness.journal import Journal, Pending, Replay, content_key


def resolution(answer: str) -> InterruptResolution:
    return InterruptResolution(
        interrupt_id="r.1.1", run_id="r", decision=InterruptDecision.ANSWER, answer=answer
    )


def test_content_keys_are_stable_and_order_insensitive_for_arguments() -> None:
    assert content_key("call", "t", {"a": 1, "b": 2}) == content_key("call", "t", {"b": 2, "a": 1})
    assert content_key("call", "t", {"a": 1}) != content_key("call", "t", {"a": 2})


def test_the_nth_occurrence_of_a_call_gets_the_nth_recorded_output() -> None:
    journal = Journal()
    first = Replay(journal)
    first.record_call("k", 1, tool="t")
    first.record_call("k", 2, tool="t")
    again = Replay(journal)
    assert again.call("k") == (True, 1)
    assert again.call("k") == (True, 2)
    assert again.call("k") == (False, None)


def test_an_answer_is_filed_under_the_pending_question_and_survives_a_round_trip() -> None:
    interrupt = Interrupt(tenant_id="t", run_id="r", question="Which?")
    journal = Journal(pending=Pending(key="q", interrupt=interrupt))
    journal.answered(resolution("blue"))
    assert journal.pending is None
    restored = Journal.of(journal.dump())
    answered = Replay(restored).answer("q")
    assert answered is not None and answered.answer == "blue"


def test_a_run_without_a_checkpoint_has_an_empty_journal() -> None:
    assert Journal.of(None) == Journal()


def test_an_answer_with_no_open_question_files_nothing() -> None:
    journal = Journal()
    journal.answered(resolution("blue"))
    assert journal.answers == {} and journal.pending is None
