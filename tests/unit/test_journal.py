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
    SPENDING,
    Journal,
    Made,
    Pending,
    Replay,
    apply,
    content_key,
    delta,
    spending,
    spent_of,
)
from trellis.harness.runs import LocalRuns
from trellis.runs.payloads import SEGMENTS, PayloadRef


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


def test_a_read_its_framework_ran_is_answered_from_the_journal_until_a_write() -> None:
    """The replay rule, by call identity: a read a framework ran itself and reported is
    answered with its record when made again — until a write of the run, which drops the
    reads journaled before it (each runs again) and keeps every write's own record (a write
    runs once). A read journaled after the write is answered again; the journal round-trips
    with what it keeps."""
    journal = Journal()
    first = Replay(journal)
    for _ in range(2):  # the same read twice: two records of one call
        first.record_call("read-notes", "old")
        first.read("read-notes")
    first.record_call("list-files", "a b")  # a call that says nothing of what it is: kept
    assert journal.reread == ["read-notes"]
    first.wrote()
    first.record_call("write-notes", "written")
    first.record_call("read-other", "x")
    first.read("read-other")

    again = Replay(Journal.model_validate(journal.model_dump(mode="json")))
    assert again.call("read-notes") == (False, None)  # before the write: it runs again
    assert again.call("write-notes") == (True, "written")  # the write ran once
    assert again.call("list-files") == (True, "a b")
    assert again.call("read-other") == (True, "x")  # journaled since the write
    assert again.journal.reread == ["read-other"]


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
    from trellis.harness.events import preview

    journal = Journal()
    replay = Replay(journal)
    call = ToolCall(tool="t", args={"a": 1}, task="the run's task")
    replay.record_step(call, ToolOutcome(tool="t", status=ToolStatus.OK, output="x" * 5000))
    replay.record_step(call, ToolOutcome(tool="t", status=ToolStatus.OK, output={"n": 1}))
    (kept, long), (_, short) = journal.trajectory
    assert kept.task is None and kept.args == {"a": 1}
    assert long.output == "x" * 2000 + "…" and short.output == {"n": 1}  # 2000 characters
    exact = ["y" * 1996]  # its repr: 2000 characters, kept whole and as it is
    assert preview("z" * 2000) == "z" * 2000 and preview(exact) == exact
    assert preview(["y" * 1997]) == repr(["y" * 1997])[:2000] + "…"


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


def test_a_rejection_is_the_call_it_answered_where_the_session_goes_on() -> None:
    """A rejection filed for the call its framework named ``c1``: that call, made again under
    its id or still waiting, keeps it; another call — ``c2``, the same arguments — is asked,
    and reads its own answer after it; run again from its input, ids are not compared."""
    journal = Journal(session="s1")
    first = Replay(journal)
    with first.calling("k", ident="c1"):
        assert first.answer("a") is None
    asked = Interrupt(tenant_id="t", run_id="r", question="Approve?")
    journal.pending = Pending(key="a", interrupt=asked, ident="c1")
    journal.answered(
        InterruptResolution(interrupt_id="r.1.1", run_id="r", decision=InterruptDecision.REJECT)
    )
    assert journal.answered_by == {"a": ["c1"]}
    same = Replay(journal)
    with same.calling("k", ident="c1"):
        assert same.answered("a")
        kept = same.answer("a")
    assert kept is not None and kept.decision is InterruptDecision.REJECT
    other = Replay(journal)
    with other.calling("k", ident="c2"):
        assert not other.answered("a") and other.answer("a") is None
    journal.answers["a"].append(approval("approve"))  # c2's, filed with no id
    with other.calling("k", ident="c2"):
        later = other.answer("a")
    assert later is not None and later.decision is InterruptDecision.APPROVE
    journal.session = None  # run again from its input: the rejection is the first call's
    again = Replay(journal)
    with again.calling("k", ident="c9"):
        rerun = again.answer("a")
    assert rerun is not None and rerun.decision is InterruptDecision.REJECT


def test_a_call_its_framework_said_it_makes_is_claimed_once_in_order() -> None:
    replay = Replay(Journal())
    replay.announced("k", "c1")
    replay.announced("k", "c2")
    assert [replay.claimed("k"), replay.claimed("k"), replay.claimed("k")] == ["c1", "c2", None]


def test_a_step_passed_drops_whose_answers_they_were() -> None:
    journal = Journal(answers={"a": [approval("reject")]}, answered_by={"a": ["c1"]})
    stepped(journal, "s1").answer("a")
    later = stepped(journal, "s2")
    later.passed("s1")
    assert "a" not in journal.answers and "a" not in journal.answered_by


async def test_progress_after_reads_is_saved_at_most_every_20_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker's run saves its journal after calls that only read at most every 20 s (after
    a write at once): the progress a crash keeps, without a save per step."""
    from types import SimpleNamespace

    from trellis.harness import runtime as runtime_module
    from trellis.harness.runtime import Runtime

    clock = [1000.0]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock[0])
    saves: list[float] = []

    class Changing:  # a journal that always has something new to save
        async def save(self, *args: Any, **kwargs: Any) -> Any:
            saves.append(clock[0])
            return SimpleNamespace(checkpoint={}, dumped={})

        def saved(self, save: Any) -> None:
            return None

    async def heartbeat(*args: Any, **kwargs: Any) -> None:
        return None

    run: Any = SimpleNamespace(
        worker_id="w",
        lease_seconds=60.0,
        run_id="run_1",
        tenant="t",
        _saved_at=None,
        replay=SimpleNamespace(journal=Changing()),
        agent=SimpleNamespace(
            harness=SimpleNamespace(runs=SimpleNamespace(artifacts=None, heartbeat=heartbeat))
        ),
    )
    for step in (0.0, 19.5, 0.5, 5.0):  # 19.5 s on: not yet; 20 s on: saved
        clock[0] += step
        await Runtime._saved(run, now=False)
    await Runtime._saved(run, now=True)  # after a write: at once
    assert saves == [1000.0, 1020.0, 1025.0]


def test_a_field_a_newer_server_wrote_is_kept_and_saved_again() -> None:
    """A rolling upgrade: a journal a newer server wrote, with a field this one does not know,
    is read, and its dump — what this server saves — still holds it as it was."""
    journal = Journal.model_validate({"used": ["refund"], "from_the_future": {"kept": [1, 2]}})
    journal.used.append("lookup")  # this server's own work, saved with it
    dumped = journal.dump()
    assert dumped["from_the_future"] == {"kept": [1, 2]}
    assert dumped["used"] == ["refund", "lookup"]


def test_in_a_step_made_again_an_answer_filed_for_another_call_is_not_this_ones() -> None:
    """One model step's identical calls, made again after a pause (a graph's step: ``here``):
    the answer filed under another call's id is that call's — a call of another id with no
    answer of its own finds none, whichever of the calls looks first."""
    journal = Journal()
    journal.answers["k"] = [resolution("yes").model_dump(mode="json")]
    journal.answered_by["k"] = ["call_a"]
    again = Replay(journal)
    again.where = lambda: "step-1"
    with again.calling("k", ident="call_b"):
        assert again.answer("k") is None
    with again.calling("k", ident="call_a"):
        found = again.answer("k")
    assert found is not None and found.answer == "yes"


def test_an_attempt_that_goes_on_takes_the_record_made_for_its_call_by_id() -> None:
    """An attempt that goes on from the framework's own state (its serialised run): a call
    takes the record filed under its own id, wherever it stands among identical calls' — not
    the first one in order."""
    journal = Journal()
    journal.calls["k"] = [{"output": "the first's"}, {"output": "the second's"}]
    journal.called_by["k"] = ["call_a", "call_b"]
    goes_on = Replay(journal, goes_on=True)
    with goes_on.calling("k", ident="call_b"):
        assert goes_on.occurrence("k") == 1


async def test_spending_kept_beside_the_segments_is_read_back_and_saved_next(
    tmp_path: Path,
) -> None:
    """A spend-only save lists the segments as they are, with what the run and its
    sub-agents' runs spent since beside them: read back over the segments' (a sub-agent's run
    the segments do not hold yet has a journal of its spending alone), and the next save of
    the journal keeps it."""
    runs = LocalRuns(payload_root=tmp_path)
    run_id = await _run(runs)
    journal = Journal(spent={"tokens": 10}, children={"kid": Journal(spent={"tokens": 4})})
    assert journal.spend_checkpoint(spending(journal.spent, journal.children)) is None  # unsaved
    saved = await _saved(journal, runs, run_id)
    assert saved is not None
    journal.spent["tokens"] = 30
    journal.children["kid"].spent["tokens"] = 9
    journal.children["kid"].children["grandkid"] = Journal(spent={"tokens": 5})
    journal.children["new"] = Journal(spent={"tokens": 7})
    tree = spending(journal.spent, journal.children)
    checkpoint = journal.spend_checkpoint(tree)
    assert checkpoint is not None and checkpoint[SEGMENTS] == saved[SEGMENTS]
    resumed = await Journal.read(checkpoint, runs.artifacts, tenant="acme")
    assert spending(resumed.spent, resumed.children) == tree
    assert spending(*spent_of(resumed.dump())) == tree
    again = await _saved(resumed, runs, run_id)  # the segments hold it from now on
    assert again is not None and SPENDING not in again
    reread = await Journal.read(again, runs.artifacts, tenant="acme")
    assert spending(reread.spent, reread.children) == tree


def _worker_run(heartbeat: Any) -> Any:
    from types import SimpleNamespace

    from trellis.contracts import AgentExecutionContext
    from trellis.harness.events import RunEvents
    from trellis.harness.identity import Identity
    from trellis.harness.runtime import Runtime

    runs = SimpleNamespace(heartbeat=heartbeat)
    agent: Any = SimpleNamespace(without=frozenset(), harness=SimpleNamespace(runs=runs))
    identity = Identity(tenant="t", agent_id="a", run_id="run_1", user="u")
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="a"))
    journal = Journal()
    journal._segments = (PayloadRef(ref="r", size=1, sha256="0" * 64),)
    runtime = Runtime(identity=identity, agent=agent, events=events, replay=Replay(journal))
    runtime.worker_id, runtime.lease_seconds = "w", 60
    return runtime


async def test_a_reply_accounted_on_a_workers_run_is_saved_once_at_a_time() -> None:
    """Each reply accounted on a worker's run sends what the run spent with a heartbeat,
    the segments listed unchanged; replies accounted while one is under way go with the
    next; a reply of a sub-agent's run is saved by the run it works inside; an in-process
    run sends nothing."""
    from trellis.harness import budget

    sent: list[dict[str, Any]] = []
    going = asyncio.Event()

    async def heartbeat(run_id: str, worker_id: str, **kw: Any) -> None:
        sent.append(kw["checkpoint"][SPENDING]["spent"])
        await going.wait()

    run = _worker_run(heartbeat)
    child = _worker_run(heartbeat)
    child.parent, child.worker_id = run, None
    budget.account(child, model="m", input_tokens=1, output_tokens=1)
    await asyncio.sleep(0)
    budget.account(run, model="m", input_tokens=2, output_tokens=2)
    budget.account(run, model="m", input_tokens=3, output_tokens=3)
    going.set()
    await run.spent_saved()
    assert [s["tokens"] for s in sent] == [2, 12]
    await run.spent_saved()  # nothing under way
    run.worker_id = None
    budget.account(run, model="m", input_tokens=1, output_tokens=0)
    assert run._spending.done() and len(sent) == 2


async def test_a_reply_accounted_in_a_thread_is_saved_on_the_runs_loop() -> None:
    sent: list[Any] = []

    async def heartbeat(run_id: str, worker_id: str, **kw: Any) -> None:
        sent.append(kw["checkpoint"])

    run = _worker_run(heartbeat)
    run.running_in = asyncio.current_task()
    await asyncio.to_thread(run.spent_more)
    await asyncio.sleep(0)
    await run.spent_saved()
    assert len(sent) == 1


async def test_a_spending_save_refused_is_not_tried_again_and_a_lost_lease_is_said() -> None:
    from trellis.harness import budget
    from trellis.runs import LeaseLostError

    async def lost(run_id: str, worker_id: str, **kw: Any) -> None:
        raise LeaseLostError("gone", code="LEASE_LOST", status=409)

    async def down(run_id: str, worker_id: str, **kw: Any) -> None:
        raise RuntimeError("agent-runs is down")

    run = _worker_run(lost)
    budget.account(run, model="m", input_tokens=1, output_tokens=1)
    await run.spent_saved()
    assert run.lease_lost and run._spent_sent is None
    run = _worker_run(down)
    budget.account(run, model="m", input_tokens=1, output_tokens=1)
    await run.spent_saved()
    assert not run.lease_lost and run._spent_sent is None


async def test_a_spending_save_under_way_when_the_attempt_ends_is_not_made() -> None:
    from trellis.harness import budget

    async def heartbeat(run_id: str, worker_id: str, **kw: Any) -> None:
        await asyncio.sleep(5)

    run = _worker_run(heartbeat)
    budget.account(run, model="m", input_tokens=1, output_tokens=1)
    await asyncio.sleep(0)
    run.unwatched()
    with pytest.raises(asyncio.CancelledError):
        await run._spending


async def test_a_reply_accounted_before_the_journals_first_save_saves_it_whole() -> None:
    """With no segments yet there are none to list beside the spending: the journal is saved
    whole (a progress save), and no spend-only heartbeat is sent."""
    from trellis.harness import budget

    sent: list[Any] = []
    saves: list[bool] = []

    async def heartbeat(run_id: str, worker_id: str, **kw: Any) -> None:
        sent.append(kw)

    async def saved(*, now: bool) -> None:
        saves.append(now)

    run = _worker_run(heartbeat)
    run.replay.journal._segments = ()
    run._saved = saved
    budget.account(run, model="m", input_tokens=1, output_tokens=1)
    await run.spent_saved()
    assert saves == [True] and sent == []
