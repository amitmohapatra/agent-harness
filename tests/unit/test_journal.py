from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from trellis.contracts import (
    HarnessError,
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    RunStart,
    RunStatus,
)
from trellis.harness import agent as agent_module
from trellis.harness import journal as journal_module
from trellis.harness.journal import (
    Journal,
    Made,
    Pending,
    Replay,
    apply,
    content_key,
    delta,
)
from trellis.harness.runs import LocalRuns
from trellis.runs.payloads import SEGMENTS


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
    restored = Journal.model_validate(journal.dump())
    answered = Replay(restored).answer("q")
    assert answered is not None and answered.answer == "blue"


FOUND = content_key("call", "find", {"q": "a"})


def paused_asking(journal: Journal, **replay: Any) -> Made:
    """An attempt that finds, then asks a person inside its second call: paused, and answered."""
    attempt = Replay(journal, **replay)
    found = attempt.made("find", {"q": "a"})
    attempt.record_call(FOUND, "found", tool="find")
    attempt.ended(found)
    asking = attempt.made("ask", {"question": "Which day?"})
    interrupt = Interrupt(tenant_id="t", run_id="r", question="Which day?")
    journal.pending = Pending(key="q", interrupt=interrupt, inside=asking)
    journal.answered(resolution("Friday"))
    return asking


def test_a_call_made_again_in_other_words_runs_as_the_call_the_pause_came_from() -> None:
    """A run again from its input makes every call again: the one at the paused call's place,
    of its tool, is it — in this attempt and, the journal read back, in every later one."""
    journal = Journal()
    paused = paused_asking(journal)
    assert (paused.at, journal.place, journal.again) == (1, 1, [paused])
    for kept in (journal, Journal.model_validate(journal.dump())):
        again = Replay(kept)
        assert again.made("find", {"q": "a"}).args == {"q": "a"}
        assert again.made("ask", {"question": "What day?"}) == paused
        assert again.made("ask", {"question": "Which room?"}).args == {"question": "Which room?"}


def test_only_a_call_of_its_tool_the_journal_does_not_hold_is_the_paused_call() -> None:
    journal = Journal()
    paused_asking(journal)
    first = Replay(journal)  # the paused call's tool, at another place: a new call
    assert first.made("ask", {"question": "What day?"}).args == {"question": "What day?"}
    other = Replay(journal)  # another tool at its place: itself
    other.made("find", {"q": "a"})
    assert other.made("find", {"q": "b"}).args == {"q": "b"}
    held = Replay(journal)  # a call the journal holds a record of, made at its place: itself
    held.record_call(content_key("call", "ask", {"question": "Which room?"}), "Blue", tool="ask")
    held = Replay(journal)
    held.made("find", {"q": "a"})
    assert held.made("ask", {"question": "Which room?"}).args == {"question": "Which room?"}


def test_a_session_that_goes_on_makes_the_paused_call_next_and_no_earlier_one() -> None:
    journal = Journal(session="s1")
    paused = paused_asking(journal)
    again = Replay(journal)  # the session holds the first call: the next is the paused one
    made = again.made("ask", {"question": "What day?"})
    assert made == paused
    again.ended(made)
    assert again.made("ask", {"question": "Which room?"}).at == 2
    # an attempt after it (a crash): the session holds the paused call too
    assert Replay(journal).made("ask", {"question": "Which room?"}).at == 2


def test_a_session_that_could_not_be_continued_runs_again_from_its_first_call() -> None:
    """The agent's session is on another machine (Kimi resumed elsewhere): a new one, the
    prompt again — and every call again, each at its place."""
    journal = Journal(session="s1")
    paused = paused_asking(journal)
    again = Replay(journal)
    journal.session = "s2"
    assert again.made("find", {"q": "a"}).at == 0
    assert again.made("ask", {"question": "What day?"}) == paused


def test_a_run_continued_from_its_frameworks_state_goes_on_after_the_calls_made() -> None:
    """An OpenAI Agents ``RunState`` approved: the SDK runs the call it held, the next after
    the calls made — not the run's first, where a call was paused in before."""
    journal = Journal()
    first = Replay(journal)
    asking = first.made("ask", {"question": "Which day?"})
    interrupt = Interrupt(tenant_id="t", run_id="r", question="Which day?")
    journal.pending = Pending(key="q", interrupt=interrupt, inside=asking)
    journal.answered(resolution("Friday"))
    second = Replay(journal)
    second.ended(second.made("ask", {"question": "What day?"}))
    approved = Replay(journal, goes_on=True).made("ask", {"question": "Which room?"})
    assert (approved.at, approved.args) == (1, {"question": "Which room?"})


async def test_a_run_without_a_checkpoint_has_an_empty_journal(tmp_path: Path) -> None:
    assert await Journal.read(None, LocalRuns(payload_root=tmp_path).artifacts, tenant="a") == (
        Journal()
    )


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


# --------------------------------------------------------------------------- segments
JOURNAL_SHAPES: list[tuple[Any, Any]] = [
    (None, {"calls": {"k": [1]}}),
    ({"calls": {"k": [1]}}, {"calls": {"k": [1, 2], "j": ["x"]}}),
    ({"a": [1, 2], "b": {"c": 1}}, {"a": [9], "b": {}}),
    ({"a": 1, "gone": 2}, {"a": 1}),
    ({"graph": {"v": 1}}, {"graph": None}),
    ([1, 2], [1, 2, 3]),
    ("x", "y"),
]


@pytest.mark.parametrize(("old", "new"), JOURNAL_SHAPES)
def test_a_delta_applied_to_the_old_value_is_the_new_one(old: Any, new: Any) -> None:
    node = delta(old, new)
    assert node is not None and apply(old, node) == new
    assert delta(new, new) is None


def test_a_journal_that_grew_sends_only_what_it_added() -> None:
    old = {"calls": {f"k{n}": [{"rows": list(range(500))}] for n in range(20)}}
    new = {"calls": {**old["calls"], "k20": [{"rows": [1]}]}}
    assert delta(old, new) == {"$d": {"calls": {"$d": {"k20": {"$r": [{"rows": [1]}]}}}}}


async def _run(runs: LocalRuns) -> str:
    record = await runs.start(
        RunStart(run_id="run_seg", tenant_id="acme", agent_id="a", user_id="u", input="x"),
        queue=False,
    )
    return record.run_id


async def _saved(journal: Journal, runs: LocalRuns, run_id: str) -> dict[str, Any] | None:
    save = await journal.save(runs.artifacts, run_id, worker_id=None, tenant="acme")
    if save is None:
        return None
    journal.saved(save)
    return save.checkpoint


async def test_saves_write_what_changed_nothing_when_nothing_did_and_read_back_whole(
    tmp_path: Path,
) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal, replay = Journal(), None
    replay = Replay(journal)
    checkpoints = []
    for n in range(5):
        replay.record_call(f"k{n}", {"rows": list(range(n * 100))}, tool="t")
        journal.results[f"/large_tool_results/r{n}"] = "a,b\n" + "1,2\n" * n
        checkpoints.append(await _saved(journal, runs, run_id))
        assert await _saved(journal, runs, run_id) is None, "unchanged: nothing is written"
    files = await asyncio.to_thread(lambda: list(tmp_path.rglob("*.json.zst")))
    assert len(files) == 5, "one segment per save that changed something"
    last = checkpoints[-1]
    assert last is not None and len(last[SEGMENTS]) == 5
    resumed = await Journal.read(last, runs.artifacts, tenant="acme")
    assert resumed.dump() == journal.dump()
    assert resumed.results["/large_tool_results/r4"] == "a,b\n" + "1,2\n" * 4
    replay = Replay(resumed)
    replay.record_call("k5", "more", tool="t")
    after = await _saved(resumed, runs, run_id)
    assert after is not None and after[SEGMENTS][:5] == last[SEGMENTS], "it adds on"


async def test_a_save_the_store_did_not_take_is_sent_again_with_the_next(tmp_path: Path) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal = Journal()
    Replay(journal).record_call("a", 1, tool="t")
    lost = await journal.save(runs.artifacts, run_id, worker_id=None, tenant="acme")
    assert lost is not None  # never marked saved: the heartbeat carrying it was refused
    Replay(journal).record_call("b", 2, tool="t")
    checkpoint = await _saved(journal, runs, run_id)
    assert checkpoint is not None and len(checkpoint[SEGMENTS]) == 1
    resumed = await Journal.read(checkpoint, runs.artifacts, tenant="acme")
    assert resumed.calls == {"a": [1], "b": [2]}


async def test_past_the_most_segments_a_save_holds_the_whole_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(journal_module, "MAX_SEGMENTS", 3)
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal, checkpoint = Journal(), None
    sizes = []
    for n in range(7):
        Replay(journal).record_call(f"k{n}", n, tool="t")
        checkpoint = await _saved(journal, runs, run_id)
        assert checkpoint is not None
        sizes.append(len(checkpoint[SEGMENTS]))
    assert sizes == [1, 2, 3, 1, 2, 3, 1]
    resumed = await Journal.read(checkpoint, runs.artifacts, tenant="acme")
    assert resumed.calls == {f"k{n}": [n] for n in range(7)}


async def test_a_resume_reads_its_segments_in_one_request_and_applies_them_in_order(
    tmp_path: Path,
) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal, checkpoint = Journal(), None
    for n in range(30):
        Replay(journal).record_call("k", n, tool="t")  # the order matters
        checkpoint = await _saved(journal, runs, run_id)
    asked: list[int] = []
    original = runs.artifacts.get_payloads

    async def counted(refs: Any, *, tenant: str | None = None) -> list[bytes]:
        asked.append(len(refs))
        return await original(refs, tenant=tenant)

    runs.artifacts.get_payloads = counted  # type: ignore[method-assign]
    resumed = await Journal.read(checkpoint, runs.artifacts, tenant="acme")
    assert resumed.calls == {"k": list(range(30))}
    assert asked == [30 % journal_module.MAX_SEGMENTS or journal_module.MAX_SEGMENTS]


async def test_another_tenants_segment_is_not_read(tmp_path: Path) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal = Journal()
    Replay(journal).record_call("k", 1, tool="t")
    checkpoint = await _saved(journal, runs, run_id)
    with pytest.raises(HarnessError, match="is gone"):
        await Journal.read(checkpoint, runs.artifacts, tenant="globex")


@pytest.mark.parametrize(
    "checkpoint", [{"calls": {"k": [1]}}, {"journal_ref": {"artifact_id": "art_1"}}]
)
async def test_a_checkpoint_that_lists_no_segments_is_refused(
    checkpoint: dict[str, Any], tmp_path: Path
) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    with pytest.raises(HarnessError, match="lists no journal segments"):
        await Journal.read(checkpoint, runs.artifacts, tenant="acme")


async def test_a_pause_with_nothing_new_lists_the_segments_it_has(tmp_path: Path) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal = Journal()
    Replay(journal).record_call("k", 1, tool="t")
    first = await journal.checkpoint(runs.artifacts, run_id, worker_id=None, tenant="acme")
    again = await journal.checkpoint(runs.artifacts, run_id, worker_id=None, tenant="acme")
    assert again == first and len(first[SEGMENTS]) == 1


async def test_a_forgotten_runs_segments_are_deleted_with_it(tmp_path: Path) -> None:
    runs = LocalRuns(payload_root=tmp_path, keep_finished=0)
    run_id = await _run(runs)
    journal = Journal()
    Replay(journal).record_call("k", 1, tool="t")
    await _saved(journal, runs, run_id)
    files = lambda: list(tmp_path.rglob("*.json.zst"))  # noqa: E731
    assert len(await asyncio.to_thread(files)) == 1
    await runs.finish(run_id, RunStatus.SUCCESS, output="done")
    assert await asyncio.to_thread(files) == [], "forgotten with the run"


async def test_a_cancel_that_cannot_read_the_journal_leaves_the_sandbox_to_the_reaper(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal = Journal(sandbox={"provider": "fake", "id": "sbx"})
    checkpoint = await journal.checkpoint(runs.artifacts, run_id, worker_id=None, tenant="acme")
    held = await runs.get(run_id)
    assert held is not None
    record = held.model_copy(update={"checkpoint": checkpoint})
    assert await agent_module._sandbox_of(runs, record) == {"provider": "fake", "id": "sbx"}
    shutil.rmtree(tmp_path)
    assert await agent_module._sandbox_of(runs, record) is None
    assert "left to the reaper" in caplog.text


def approval(decision: str) -> dict[str, Any]:
    return InterruptResolution(
        interrupt_id="r.1.1", run_id="r", decision=InterruptDecision(decision.upper())
    ).model_dump(mode="json")


def test_a_session_that_goes_on_reads_past_what_the_calls_before_read() -> None:
    """The same call twice in a session the framework keeps: the second paused, answered; the
    attempt that goes on makes only the second again, which reads its own answer and runs —
    not the first call's record and answer."""
    journal = Journal(session="s1")
    first = Replay(journal)
    with first.calling("k"):
        assert first.answer("a") is None  # the first call asks: the run pauses
        first.paused()
    journal.answers["a"] = [approval("approve")]
    second = Replay(journal)
    with second.calling("k"):  # the first call made again, answered: it runs
        assert not second.call("k")[0] and second.answer("a") is not None
        second.record_call("k", "ran once", tool="t", stepped=True)
    with second.calling("k"):  # the second asks: its own answer is not there yet
        assert not second.call("k")[0] and second.answer("a") is None
        second.paused()
    assert journal.cursors == {"c:k": 1, "a:a": 1}
    journal.answers["a"].append(approval("reject"))
    third = Replay(journal)  # the session holds the first call: only the second again
    with third.calling("k"):
        assert third.call("k") == (False, None)
        answered = third.answer("a")
    assert answered is not None and answered.decision is InterruptDecision.REJECT
    assert third.cursor() == {"c:k": 1, "a:a": 2}
    journal.session = None  # run again from its input: nothing of the cursors kept
    fourth = Replay(journal)
    fourth.paused()
    assert journal.cursors == {}


def test_a_replayed_call_reads_past_its_answer_so_the_next_reads_its_own() -> None:
    """Run again from its input: the first call replays from its record and the second, the
    same call, reads the second answer — the first's was its own."""
    journal = Journal(answers={"a": [approval("approve"), approval("reject")]})
    first = Replay(journal)
    with first.calling("k"):
        assert first.answer("a") is not None
        first.record_call("k", "ran", tool="t", stepped=True)
    again = Replay(journal)
    with again.calling("k"):
        assert again.call("k") == (True, "ran")
    with again.calling("k"):
        assert again.call("k")[0] is False
        later = again.answer("a")
    assert later is not None and later.decision is InterruptDecision.REJECT
    # a record of a call that read nothing besides keeps its place among those that did
    with again.calling("k"):
        again.record_call("k", "unasked", tool="t", stepped=True)
    assert journal.reads == {"k": [{"a:a": 1}, {}]}
    other = Replay(journal)
    other.record_call("k", "outside a call", stepped=True)  # read nothing: its place kept
    assert journal.reads["k"] == [{"a:a": 1}, {}, {}]


def test_a_subagent_begun_again_reads_its_calls_again() -> None:
    """A subagent paused in its work is started again where the session goes on: all its
    reads in the attempt are put back (``since``: where it started), so its first call
    replays from its record and its answer is read past."""
    journal = Journal(session="s1", answers={"a": [approval("approve")]})
    first = Replay(journal)
    since = first.cursor()
    with first.calling("k"):
        assert first.answer("a") is not None
        first.record_call("k", "ran", tool="t", stepped=True)
    with first.calling("j"):
        first.paused(since=since)
    assert journal.cursors == {}
    again = Replay(journal)
    with again.calling("k"):
        assert again.call("k") == (True, "ran")
    assert again.cursor() == {"c:k": 1, "a:a": 1}


def test_a_step_passed_drops_the_reads_of_its_records() -> None:
    journal = Journal(answers={"a": [approval("approve")]})
    replay = stepped(journal, "s1")
    with replay.calling("k"):
        replay.answer("a")
        replay.record_call("k", "x", tool="t", stepped=True)
    later = stepped(journal, "s2")
    with later.calling("k"):
        later.record_call("k", "y", tool="t", stepped=True)
    later.passed("s1")
    assert journal.calls["k"] == ["y"] and journal.reads == {"k": [{}]}
    later.passed("s2")
    assert "k" not in journal.calls and "k" not in journal.reads
