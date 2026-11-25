"""A conversation's turn on a worker that kept no session of it resumes the one its last turn
ended in, as the run store keeps it past that turn's end (``pipeline._last_turns_session``,
``pipeline._history_kept``): only the last turn's, only a session whose transcript is kept,
and a run store that cannot be read is a warning. A session the CLI cannot continue gives the
turn the conversation's turns instead (``adapters.claude._with_earlier``)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from trellis.contracts import RunStatus
from trellis.harness import conversation, pipeline
from trellis.harness.adapters.claude import ClaudeInput, _with_earlier
from trellis.harness.history import Said, Speaker
from trellis.harness.journal import Journal

ASKED = datetime(2026, 11, 25, 12, tzinfo=UTC)
TRANSCRIPT = [{"uuid": "e1", "prompt": "The locker code is 4242."}]


def _turn(run_id: str, **changes: Any) -> SimpleNamespace:
    turn = {
        "run_id": run_id,
        "status": RunStatus.SUCCESS,
        "parent_run_id": None,
        "user_id": "ada",
        "created_at": ASKED - timedelta(minutes=1),
        "checkpoint": {"kept": Journal(session="s1", transcripts={"s1": TRANSCRIPT})},
    }
    return SimpleNamespace(**(turn | changes))


class _Runs:
    """A run store's listing (newest first) and reads; ``down``: every listing fails."""

    def __init__(self, *turns: SimpleNamespace, down: bool = False) -> None:
        self.turns, self.down, self.artifacts = turns, down, None

    async def iterate(self, **_: Any) -> Any:
        if self.down:
            raise RuntimeError("the run store is down")
        for turn in self.turns:
            yield SimpleNamespace(run_id=turn.run_id, status=turn.status)

    async def get(self, run_id: str, *, tenant: str) -> Any:
        return next((t for t in self.turns if t.run_id == run_id and t.status), None)


@pytest.fixture(autouse=True)
def _journal_as_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    async def read(cls: Any, checkpoint: Any, artifacts: Any, *, tenant: str) -> Journal:
        return checkpoint["kept"]

    monkeypatch.setattr(Journal, "read", classmethod(read))


async def _resumed(runs: _Runs) -> Journal | None:
    agent: Any = SimpleNamespace(harness=SimpleNamespace(runs=runs))
    identity: Any = SimpleNamespace(
        tenant="t", thread="th", agent_id="a", run_id="this", user="ada"
    )
    journal = Journal()
    found = await pipeline._last_turns_session(agent, identity, ASKED, journal)
    return journal if found else None


async def test_the_last_turn_before_this_one_is_the_one_resumed() -> None:
    passed_over = [
        _turn("this"),  # the turn itself
        _turn("waiting", status=RunStatus.QUEUED),
        _turn("gone", status=None),  # listed, no longer read
        _turn("child", parent_run_id="run_0"),
        _turn("bob", user_id="bob"),
        _turn("later", created_at=ASKED + timedelta(seconds=1)),
    ]
    older = _turn("older", checkpoint={"kept": Journal(session="s0", transcripts={"s0": []})})
    journal = await _resumed(_Runs(*passed_over, _turn("last"), older))
    assert journal is not None and journal.session == "s1"
    assert journal.transcripts == {"s1": TRANSCRIPT}


@pytest.mark.parametrize(
    "last",
    [
        _turn("last", status=RunStatus.PAUSED),  # still waiting for a person
        _turn("last", checkpoint=None),  # kept none: refused, or an older server's
        _turn("last", checkpoint={"kept": Journal()}),  # a journal with no session
        _turn("last", checkpoint={"kept": Journal(session="s1")}),  # nor its transcript
    ],
    ids=["paused", "no-checkpoint", "no-session", "no-transcript"],
)
async def test_a_last_turn_that_kept_no_session_resumes_none_older(last: Any) -> None:
    older = _turn("older")  # a session that never saw the last turn
    assert await _resumed(_Runs(last, older)) is None


async def test_a_conversation_with_no_earlier_turn_resumes_none() -> None:
    assert await _resumed(_Runs(_turn("this"))) is None


async def test_a_run_store_that_cannot_be_read_is_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        assert await _resumed(_Runs(down=True)) is None
    assert "the last turn's session: the run store is down" in caplog.text


def test_a_kept_session_is_its_own_transcripts_and_its_subagents() -> None:
    journal = Journal(
        session="s1",
        transcripts={"s1": TRANSCRIPT, "s1/subagents/agent-1": [{"uuid": "x"}], "s0": []},
        used=["stock"],
    )
    kept = pipeline._session_only(journal)
    assert kept.session == "s1" and sorted(kept.transcripts) == ["s1", "s1/subagents/agent-1"]
    assert kept.used == []  # nothing of the run's calls


class _Events:
    def __init__(self) -> None:
        self.warned: list[tuple[str, str]] = []

    def warning(self, code: str, message: str) -> None:
        self.warned.append((code, message))


async def test_a_session_the_cli_could_not_continue_gives_the_turns_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [Said(Speaker.USER, "The locker code is 4242."), Said(Speaker.ASSISTANT, "Noted.")]

    async def earlier(runtime: Any, **_: Any) -> list[Said]:
        if runtime.down:
            raise RuntimeError("the run store is down")
        return turns if runtime.said else []

    monkeypatch.setattr(conversation, "earlier", earlier)

    def runtime(*, said: bool = True, down: bool = False) -> Any:
        return SimpleNamespace(continued=True, said=said, down=down, events=_Events())

    told = conversation.rendered(turns)
    plain = ClaudeInput(prompt="What is the code?", context=None)
    dated = ClaudeInput(prompt="What is the code?", context="It is now Wednesday.")
    given = runtime()
    assert (await _with_earlier(plain, given)).context == told and given.continued is False
    assert (await _with_earlier(dated, runtime())).context == f"{told}\n\nIt is now Wednesday."
    assert await _with_earlier(dated, runtime(said=False)) == dated  # nothing said before
    down = runtime(down=True)
    assert await _with_earlier(dated, down) == dated
    assert down.events.warned == [
        ("earlier_turns", "the conversation's earlier turns: the run store is down")
    ]
