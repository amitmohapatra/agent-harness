"""What a conversation carries to its next turn, kept in the file store within
``pipeline.CARRIED_BYTES`` (``pipeline._fitted``): its files moved out, the largest first, as
far as needed — each kept whole as an object of its own, named by its content, and named in the
turn's file (``pipeline.CARRIED_REF``) — the rest inline; a state past the bound without its
files is refused."""

from __future__ import annotations

import hashlib
import json

import pytest

from trellis.harness import pipeline
from trellis.harness.pipeline import CARRIED_CUT, CARRIED_REF


def test_a_state_within_the_bound_is_kept_whole_in_the_turns_file() -> None:
    state = {"todos": ["ship"], CARRIED_CUT: {"/a.md": "a" * 10}}
    data, moved = pipeline._fitted(state)
    assert json.loads(data) == state and moved == {}


def test_files_move_out_largest_first_until_the_state_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"/big.md": "b" * 3000, "/mid.md": "m" * 2000, "/small.md": "s" * 10}
    state = {"todos": ["ship"], CARRIED_CUT: files}
    bound = len(json.dumps(state)) - 4000  # more than the largest file must go
    monkeypatch.setattr(pipeline, "CARRIED_BYTES", bound)
    data, moved = pipeline._fitted(state)
    kept = json.loads(data)
    assert kept["todos"] == ["ship"] and kept[CARRIED_CUT]["/small.md"] == "s" * 10
    named = {path: kept[CARRIED_CUT][path][CARRIED_REF] for path in ("/big.md", "/mid.md")}
    for path, name in named.items():  # each whole, as an object named by its content
        assert json.loads(moved[name]) == files[path]
        assert name == hashlib.sha256(moved[name]).hexdigest() + ".json"
    assert set(moved) == set(named.values()) and len(data) <= bound


def test_a_file_unchanged_from_turn_to_turn_is_the_same_object() -> None:
    """Named by its content: a file no turn changed is the object it was, never sent again."""
    entry = {"content": ["x" * 50], "modified_at": "2026-11-25T00:00:00Z"}
    first = pipeline._moved_out(entry)
    assert first == pipeline._moved_out(dict(entry))
    assert first != pipeline._moved_out({**entry, "content": ["y" * 50]})


def test_a_state_past_the_bound_with_every_file_moved_out_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"todos": ["x" * 500], CARRIED_CUT: {"/a.md": "a" * 50, "/b.md": "b" * 40}}
    monkeypatch.setattr(pipeline, "CARRIED_BYTES", 100)
    with pytest.raises(ValueError, match="over the 100 bytes kept"):
        pipeline._fitted(state)


def test_a_deleted_file_is_never_moved_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deletion marker (Deep Agents' ``None``) stays a marker in the turn's file, however
    long its path: the files around it move out."""
    gone = "/" + "g" * 600 + ".md"  # a marker larger than the next file: passed over
    files = {"/big.md": "b" * 3000, gone: None, "/s.md": "s" * 500}
    state = {"todos": ["ship"], CARRIED_CUT: files}
    monkeypatch.setattr(pipeline, "CARRIED_BYTES", len(json.dumps(state)) - 3200)
    data, moved = pipeline._fitted(state)
    kept = json.loads(data)[CARRIED_CUT]
    assert kept[gone] is None
    assert set(kept["/big.md"]) == set(kept["/s.md"]) == {CARRIED_REF} and len(moved) == 2
