"""Forget by chat: "forget my address". The memory service lets an agent forget only what it
wrote itself; what the user said is theirs (``users_own_statement``). The harness asks a person
the way it asks for any call (the approval path), and forgets the statement as the user once
it is approved; a rejection is what the model reads."""

from __future__ import annotations

import pytest

from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct
from trellis.contracts import RunStatus

ADDRESS = "I live at 12 Baker Street."


def _said(memory_service: FakeMemoryService, user: str = "u1") -> str:
    """What ``user`` told the assistant, as the service keeps it: their own statement."""
    memory_id = "mem_address"
    memory_service.memories[memory_id] = {
        "content": ADDRESS,
        "memory_type": "SEMANTIC",
        "visibility": "PRIVATE",
        "user": user,
        "current": True,
        "created_at": "2026-10-01T09:00:00Z",
        "supersedes": None,
        "superseded_by": None,
    }
    return memory_id


async def test_forgetting_what_the_user_said_asks_and_forgets_it_as_the_user(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_id = _said(memory_service)
    model = ScriptedChat([("memory_forget", {"id": memory_id}), "Done: I forgot your address."])
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=model), id="forget")
    first = await agent.run("forget my address", user="u1")
    # the agent's own call is refused by the service: the user's words are asked about
    assert first.status is RunStatus.PAUSED and first.interrupt is not None, first.error
    assert memory_id in first.interrupt.question
    assert first.interrupt.tool_call is not None
    assert first.interrupt.tool_call.tool == "memory_forget"
    assert memory_service.memories[memory_id]["current"]  # nothing forgotten yet
    done = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS, done.error
    assert not memory_service.memories[memory_id]["current"]
    [forgot] = memory_service.named("forget")
    assert forgot.path["memory_id"] == memory_id
    assert forgot.scope.get("user_id") == "u1" and not forgot.scope.get("agent_id")  # as the user
    told = str(model.requests[-1]["messages"][-1]["content"])
    assert '"forgotten": true' in told


async def test_a_rejected_forget_keeps_the_statement_and_the_model_is_told(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_id = _said(memory_service)
    model = ScriptedChat([("memory_forget", {"id": memory_id}), "I kept it."])
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=model), id="forget")
    first = await agent.run("forget my address", user="u1")
    assert first.interrupt is not None, first.error
    done = await agent.resume(
        first.interrupt.interrupt_id, "reject", reviewer="u1", answer="keep it for invoices"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    assert memory_service.memories[memory_id]["current"]
    assert memory_service.named("forget") == []
    told = str(model.requests[-1]["messages"][-1]["content"])
    assert '"forgotten": false' in told and "keep it for invoices" in told


async def test_what_the_agent_wrote_is_forgotten_at_once(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("memory_forget", {"id": "mem_note"}), "Forgotten."])
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=model), id="forget")
    result = await agent.run("drop your note", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    [call] = memory_service.named("call_agent_tool")
    assert call.path["name"] == "memory_forget"
    assert memory_service.named("forget") == []


@pytest.mark.parametrize("reason", ["not_owner", None])
async def test_another_refusal_is_the_models_to_read(
    memory_harness: Harness, memory_service: FakeMemoryService, reason: str | None
) -> None:
    """Only the user's own statement is asked about: any other refusal is an error as before."""
    memory_service.forget_refusal = reason or "plain"
    model = ScriptedChat([("memory_forget", {"id": "mem_x"}), "Could not."])
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=model), id="forget")
    result = await agent.run("forget x", user="u1")
    assert result.status is RunStatus.SUCCESS, result.error
    assert memory_service.named("forget") == []
    assert "Approve" not in str(model.requests[-1]["messages"])


async def test_the_tool_says_what_it_does_with_the_users_words(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat(["ok"])
    await memory_harness.wrap(ReAct(system_prompt="s", model=model), id="forget").run(
        "hi", user="u1"
    )
    [forget] = [t for t in model.requests[0]["tools"] if t["function"]["name"] == "memory_forget"]
    assert "approve" in forget["function"]["description"]
