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


def test_a_call_started_and_never_ended_is_interrupted_for_the_next_attempt() -> None:
    journal = Journal()
    first = Replay(journal)
    first.start("k")
    first.record_call("k", "done")  # the first occurrence ended
    first.start("k")  # the second was running when the worker died
    again = Replay(journal)
    assert again.call("k") == (True, "done") and again.interrupted("k")
    again.unstart("k")  # it failed (or asked) instead: it runs again, as never started
    assert not again.interrupted("k") and journal.started == {"k": 1}
    again.record_call("k", "done again")
    last = Replay(journal)
    assert last.call("k") == (True, "done") and last.call("k") == (True, "done again")
    assert not last.interrupted("k")
    lone = Journal()
    Replay(lone).start("j")
    Replay(lone).unstart("j")
    assert lone.started == {}


# --------------------------------------------------------------------------- steps
def stepped(journal: Journal, step: str | None, *, resumed: str | None = None) -> Replay:
    """An attempt's cursor whose calls are made in ``step`` of a graph that resumed from
    ``resumed``."""
    replay = Replay(journal)
    replay.where = lambda: step
    replay.resumed = resumed
    return replay


def test_a_record_of_a_step_before_the_resumed_checkpoint_is_not_a_later_calls() -> None:
    journal = Journal()
    stepped(journal, "s1").record_call("k", "first", tool="t", stepped=True)
    assert journal.steps == {"c:k": ["s1"]}
    # the step under way when the run stopped runs again: its call is replayed
    assert stepped(journal, "s1", resumed="s1").call("k") == (True, "first")
    # a later step's identical call is a new one: the earlier step's record is passed over
    later = stepped(journal, "s3", resumed="s2")
    assert later.occurrence("k") == 1 and later.call("k") == (False, None)
    later.record_call("k", "second", tool="t", stepped=True)
    assert journal.calls["k"] == ["first", "second"] and journal.steps["c:k"] == ["s1", "s3"]
    # outside a graph's step, or in a graph that resumed from no checkpoint, nothing is passed
    assert stepped(journal, None, resumed="s2").call("k") == (True, "first")
    assert stepped(journal, "s3").call("k") == (True, "first")
    assert Replay(journal).call("k") == (True, "first")


def test_a_record_made_after_the_resumed_checkpoint_is_replayed_and_becomes_this_steps() -> None:
    journal = Journal()
    # made after the checkpoint the run resumed from, in a step whose checkpoint was not saved
    stepped(journal, "s5").record_call("k", "done", stepped=True)
    again = stepped(journal, "s6", resumed="s4")
    assert again.call("k") == (True, "done")
    assert journal.steps["c:k"] == ["s6"]  # the step that read it
    again.passed("s6")
    assert journal.calls == {} and journal.steps == {}


def test_a_step_passed_drops_its_records_and_moves_the_cursor_and_the_started_back() -> None:
    journal = Journal()
    replay = stepped(journal, "s1")
    replay.start("k")
    replay.record_call("k", "a", tool="t", stepped=True)
    replay.record_call("pin", "v1")  # the run's (a stored prompt's version): kept
    replay.where = lambda: "s2"
    replay.start("k")  # under way in the next step
    replay.passed(None)  # a first checkpoint follows none
    assert journal.calls["k"] == ["a"]
    replay.passed("s1")
    assert journal.calls == {"pin": ["v1"]} and journal.steps == {}
    assert journal.started == {"k": 1} and replay.occurrence("k") == 0
    assert replay.interrupted("k")  # the call under way: still known to be
    replay.record_call("k", "b", tool="t", stepped=True)
    replay.passed("s2")
    assert "k" not in journal.calls and journal.started == {}
    assert journal.used == ["t"]


def test_a_step_passed_drops_only_what_leads_its_records() -> None:
    journal = Journal()
    Replay(journal).record_call("k", "unmarked", stepped=True)  # made outside a graph's step
    stepped(journal, "s1").record_call("k", "marked", stepped=True)
    replay = stepped(journal, "s9")
    replay.passed("s9")
    assert journal.calls["k"] == ["unmarked", "marked"]
    assert journal.steps["c:k"] == [None, "s1"]
    later = stepped(journal, "s2")
    later.record_call("j", "x", stepped=True)
    later.record_call("j", "y", stepped=True)
    stepped(journal, "s3").record_call("j", "z", stepped=True)
    later.passed("s2")  # the step's records go, a later step's stay
    assert journal.calls["j"] == ["z"] and journal.steps["c:j"] == ["s3"]


def test_an_answer_read_in_an_earlier_step_is_not_a_later_questions() -> None:
    journal = Journal(answers={"q": [resolution("yes").model_dump(mode="json")]})
    first = stepped(journal, "s1").answer("q")
    assert first is not None and first.answer == "yes"
    assert journal.steps == {"a:q": ["s1"]}
    journal.answers["q"].append(resolution("no").model_dump(mode="json"))  # filed later
    later = stepped(journal, "s3", resumed="s2").answer("q")
    assert later is not None and later.answer == "no"  # the earlier step's is passed over
    assert journal.steps == {"a:q": ["s1", "s3"]}
    journal.answers["q"].append(resolution("later").model_dump(mode="json"))  # not read yet
    stepped(journal, "s4").passed("s3")
    assert len(journal.answers["q"]) == 1 and journal.steps == {}
    unread = Journal(answers={"q": [resolution("kept").model_dump(mode="json")]})
    stepped(unread, None).answer("q")  # read outside a graph's step: unmarked
    stepped(unread, "s1").passed("s9")
    assert unread.steps == {} and len(unread.answers["q"]) == 1


def test_the_trajectory_keeps_a_long_output_as_the_stream_previews_it() -> None:
    from trellis.contracts import ToolCall, ToolOutcome, ToolStatus
    from trellis.harness.events import PREVIEW_CHARS

    journal = Journal()
    replay = Replay(journal)
    call = ToolCall(tool="t", args={"a": 1}, task="the run's task")
    replay.record_step(call, ToolOutcome(tool="t", status=ToolStatus.OK, output="x" * 5000))
    replay.record_step(call, ToolOutcome(tool="t", status=ToolStatus.OK, output={"n": 1}))
    (kept, long), (_, short) = journal.trajectory
    assert kept.task is None and kept.args == {"a": 1}
    assert long.output == "x" * PREVIEW_CHARS + "…" and short.output == {"n": 1}
