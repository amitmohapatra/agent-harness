"""An ACP agent's conversation whose background work would keep its process, on a server
with no place for one more kept process (``live.KEPT``, over every agent's sessions): the
turn runs as it always did, its process ends with it, and its background work ends with
the process — said, never left ``RUNNING``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.test_acp import children, fake
from tests.integration.test_acp_background import END, LOOK, ended, kept, say, started
from trellis import Harness
from trellis.contracts import RunStatus
from trellis.harness.bridges import live as shared


async def test_with_no_place_to_keep_the_process_its_background_work_ends_with_it(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shared, "KEPT", 0)  # every place taken, none idle to give its own
    script = [
        {"agent": LOOK, "background": {"after": 30, "result": "found tea"}},
        {"text": "started it"},
        END,
        {"text": "anything else?"},
    ]
    agent = harness.wrap(fake(tmp_path, script), id="bg-no-room")
    first = await say(agent, "look for tea in the background")
    assert first.status is RunStatus.SUCCESS and first.answer == "started it", first.error
    assert kept(agent) is None  # not kept: no place
    [child] = await children(harness, first.run_id)
    done = await ended(harness, child.run_id)
    assert done.status is RunStatus.ERROR and done.error is not None  # it ended with it
    second = await say(agent, "what else?")
    assert second.status is RunStatus.SUCCESS
    assert started(tmp_path) == 2  # the next turn had a process of its own
