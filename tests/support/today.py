"""What a model's instructions read under the harness: the day it is, after them."""

from __future__ import annotations

from trellis.harness import snapshots


def dated(instructions: str = "") -> str:
    """``instructions`` as the model is sent them: the day it is after them
    (``snapshots.today``), in UTC — a test's run has no app that says its user's zone."""
    day = snapshots.said(snapshots.now(), None)
    return "\n\n".join(part for part in (instructions, day) if part)
