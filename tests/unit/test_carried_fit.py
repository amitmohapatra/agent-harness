"""What a conversation carries to its next turn, kept in the file store within
``pipeline.CARRIED_BYTES`` (``pipeline._fitted``): its files left out, the largest first, as far
as needed — the rest whole; a state past the bound without its files is refused."""

from __future__ import annotations

import json

import pytest

from trellis.harness import pipeline
from trellis.harness.pipeline import CARRIED_CUT


def test_files_are_left_out_largest_first_until_the_state_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"/big.md": "b" * 300, "/mid.md": "m" * 200, "/small.md": "s" * 10}
    state = {"todos": ["ship"], CARRIED_CUT: files}
    bound = len(json.dumps(state)) - 400  # more than the largest file must go
    monkeypatch.setattr(pipeline, "CARRIED_BYTES", bound)
    data, dropped = pipeline._fitted(state)
    assert dropped == ["/big.md", "/mid.md"]  # the largest, then the next: enough
    kept = json.loads(data)
    assert kept["todos"] == ["ship"] and kept[CARRIED_CUT] == {"/small.md": "s" * 10}
    assert len(data) <= bound


def test_a_state_past_the_bound_with_every_file_left_out_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"todos": ["x" * 500], CARRIED_CUT: {"/a.md": "a" * 50, "/b.md": "b" * 40}}
    monkeypatch.setattr(pipeline, "CARRIED_BYTES", 100)
    with pytest.raises(ValueError, match="over the 100 bytes kept"):
        pipeline._fitted(state)
