"""``ReAct`` keeping the conversation within the model's window, with the native middleware
sized from it: older results cleared from the request past half the window (``read_file``
reads one again by its call's id), older turns summarized near its end (the history saved where ``read_file``
reads it), and a resume that asks neither the model nor the summary again."""

from __future__ import annotations

import re
from typing import Any

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import Script, ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.automatic import JournalBackend
from trellis.harness.middleware import ASKED_WITH
from trellis.harness.react import CLEARED


@tool(side_effects="irreversible")
def publish(name: str) -> str:
    """Publish a report."""
    return f"published {name}"


def contents(model: ScriptedChat, request: int) -> list[Any]:
    return [m.get("content") for m in model.requests[request]["messages"]]


class Summarizing(Script):
    """A script that answers a request to summarize with a summary, outside its turns."""

    def __init__(self, turns: list[Any]) -> None:
        super().__init__(turns)
        self.summaries = 0

    def next(self, body: dict[str, Any]) -> Any:
        if "Messages to summarize" in str(body["messages"][-1].get("content")):
            self.summaries += 1
            return "SESSION INTENT: take notes. NEXT STEPS: publish them."
        return super().next(body)


async def test_older_results_are_cleared_past_half_the_window_and_read_back(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="read")
    def page(n: int) -> str:
        """A page of a book."""
        return f"page {n} " + "w" * 2000

    script: list[Any] = [("page", {"n": n}) for n in range(1, 7)]
    script += [
        ("read_file", {"file_path": "/large_tool_results/call_1"}),
        ("publish", {"name": "book"}),
        "done",
    ]
    model = ScriptedChat(script=Summarizing(script))
    target = ReAct(system_prompt="s", model=model, context_window=6000)
    agent = arm(memory_harness.wrap(target, id="book"), [page, publish])
    paused = await agent.run("read the book", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    cleared = [n for n in range(len(model.requests)) if CLEARED in contents(model, n)]
    first = cleared[0]
    assert first > 1  # only once the results grew past half the window
    results = [c for c in contents(model, first) if str(c).startswith(("page", "[result"))]
    assert results[0] == CLEARED
    assert all(r.startswith("page") for r in results[-3:])  # the last three stay
    offered = [{t["function"]["name"] for t in r["tools"]} for r in model.requests]
    # the placeholder's tool, offered once a result was cleared
    assert ["read_file" in o for o in offered[: first + 1]] == [False] * first + [True]
    read = [n for n, r in enumerate(model.requests) if "read_file" in str(r["messages"][-2])]
    assert contents(model, read[0])[-1] == "1  page 1 " + "w" * 2000  # read back whole
    asked = len(model.requests)
    memory_service.context_text = "remembered " * 400  # the context grows before the resume
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done", done.error
    assert len(model.requests) == asked + 1  # only the step after the approval


async def test_older_turns_are_summarized_and_their_history_kept_to_read(
    memory_harness: Harness,
) -> None:
    @tool(side_effects="read")
    def note(text: str) -> str:
        """Take a note."""
        return "noted"

    script: list[Any] = [("note", {"text": f"{n} " + "n" * 2400}) for n in range(5)]
    summarizing = Summarizing([*script, ("publish", {"name": "notes"}), "done"])
    model = ScriptedChat(script=summarizing)
    # Deep Agents' summarization: a ReAct on Deep Agents (a backend given)
    target = ReAct(system_prompt="s", model=model, context_window=4000, backend=JournalBackend())
    agent = arm(memory_harness.wrap(target, id="notes"), [note, publish])
    paused = await agent.run("take five notes", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    summaries = summarizing.summaries
    assert summaries >= 1
    after = next(
        r["messages"] for r in model.requests if "has been summarized" in str(r["messages"])
    )
    summary = next(m for m in after if "has been summarized" in str(m.get("content")))
    assert summary["role"] == "user"
    assert "/conversation_history/" in summary["content"]  # the full history, saved to read
    assert "SESSION INTENT: take notes." in summary["content"]
    # the task, and the memory context pushed with it, stay ahead of the summary
    assert after[1]["content"] == [
        {"type": "text", "text": ASKED_WITH.format(context="The user prefers email.")},
        {"type": "text", "text": "take five notes"},
    ]
    assert after[2] is summary
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done", done.error
    assert summarizing.summaries == summaries  # the resume continued from the summary


async def test_a_message_list_without_a_user_message_still_runs(harness: Harness) -> None:
    model = ScriptedChat(["fine"])
    target = ReAct(system_prompt="s", model=model, context_window=1000)
    result = await harness.wrap(target, id="nouser").run(
        [{"role": "assistant", "content": "hello"}], user="u"
    )
    assert result.answer == "fine"


async def test_every_placeholder_names_a_read_tool_the_request_offers(harness: Harness) -> None:
    """One clearing pass, LangChain's ``ClearToolUsesEdit``; ``FitWindow`` only cuts. Whatever
    a request's tool results say to read the rest with is a tool that request offers
    (duplication audit A2, B2)."""

    @tool(side_effects="read")
    def page(n: int) -> str:
        """A page of a book."""
        return f"page {n} " + "w" * 30_000

    model = ScriptedChat([("page", {"n": 1}), ("page", {"n": 2}), "done"])
    target = ReAct(system_prompt="s", model=model, context_window=6000)
    result = await arm(harness.wrap(target, id="fit"), [page]).run("read it", user="u")
    assert result.answer == "done", result.error
    named: set[str] = set()
    for sent in model.requests:
        offered = {t["function"]["name"] for t in sent["tools"]}
        for m in sent["messages"]:
            if m["role"] != "tool":
                continue
            for reader in re.findall(r"(\w+)\((?:id|file_path)=", str(m["content"])):
                named.add(reader)
                assert reader in offered, (reader, sorted(offered))
    assert named == {"read_file"}  # one read-back tool
