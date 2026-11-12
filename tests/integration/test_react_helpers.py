"""Handing a task to a helper asks no one: a ``task`` call only hands the work on, and every
call the helper then makes is governed on its own (``adapters.base.helper``). Generated: every
kind of backend ``ReAct`` takes as a Deep Agents capability crossed with the helper asked (Deep
Agents' general-purpose one, a sub-agent of the agent's own) and the risk of the tool the helper
calls. In each cell the ``task`` call itself pauses for no approval, a helper's read runs, and
a helper's write that cannot be undone still asks — for that tool, by its name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import FilesystemBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox, tool
from trellis.contracts import RunStatus
from trellis.harness.adapters.langgraph import TASK_TOOL
from trellis.harness.automatic import JournalBackend

GENERAL = "general-purpose"
OWN = "counter"


def backends(tmp_path: Path) -> dict[str, Any]:
    """Every kind of backend ``ReAct`` takes, by name."""
    return {
        "state": JournalBackend(),
        "disk": FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        "store": StoreBackend(namespace=lambda _: ("helped",), store=InMemoryStore()),
        "sandbox": sandbox(FakeSandboxes()),
    }


@pytest.mark.parametrize("risk", ["read", "irreversible"])
@pytest.mark.parametrize("helper", [GENERAL, OWN])
@pytest.mark.parametrize("kind", ["state", "disk", "store", "sandbox"])
async def test_handing_a_task_on_asks_no_one_and_the_helpers_calls_keep_their_own_risk(
    harness: Harness, tmp_path: Path, kind: str, helper: str, risk: str
) -> None:
    done: list[str] = []

    @tool(side_effects=risk)
    def act(what: str) -> str:
        """Does one thing."""
        done.append(what)
        return f"did {what}"

    hand = (TASK_TOOL, {"description": "do it", "subagent_type": helper})
    work = [("act", {"what": "it"}), "it is done"]
    if helper == GENERAL:  # Deep Agents' own helper runs on the agent's model
        model = ScriptedChatModel(turns=[hand, *work, "done"])
        given: dict[str, Any] = {}
    else:
        model = ScriptedChatModel(turns=[hand, "done"])
        mine = {"name": OWN, "description": "Does it.", "model": ScriptedChatModel(turns=work)}
        given = {"subagents": [mine]}
    target = ReAct("s", model, tools=[act], backend=backends(tmp_path)[kind], **given)
    agent = harness.wrap(target, id=f"hands-{kind}-{helper}-{risk}")
    result = await agent.run("do it", user="u")
    if risk == "read":
        assert result.interrupt is None, result.interrupt  # no one was asked anything
    else:  # asked once, for the helper's own call: never for the hand-over
        assert result.status is RunStatus.PAUSED and result.interrupt is not None, result.error
        asked = result.interrupt.tool_call
        assert asked is not None and asked.tool == "act", result.interrupt
        assert done == []
        result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="r")
        assert result.interrupt is None, result.interrupt
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    assert done == ["it"]
