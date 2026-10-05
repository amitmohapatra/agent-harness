"""``ReAct`` keeping the conversation within the model's window: a large result keeps its head
and tail and is read back with ``read_result``; older results are cleared past half the window,
older turns compacted into one summary past three quarters; what was decided, and the summary,
replayed on a resume however the memory context changed."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.adapters.react import COMPACT, SUMMARY

TEXT = "".join(f"{n:04d}" for n in range(75))  # 300 characters, every position readable


@tool(side_effects="read")
def report(name: str) -> str:
    """A long report."""
    return TEXT


@tool(side_effects="irreversible")
def publish(name: str) -> str:
    """Publish a report."""
    return f"published {name}"


def contents(model: ScriptedChat, request: int) -> list[Any]:
    return [m.get("content") for m in model.requests[request]["messages"]]


def offered(model: ScriptedChat, request: int) -> list[str]:
    return [t["function"]["name"] for t in model.requests[request].get("tools", [])]


async def test_a_cut_result_is_read_back_in_parts(harness: Harness) -> None:
    model = ScriptedChat(
        [
            ("report", {"name": "q3"}),
            ("read_result", {"id": "call_1", "offset": 50, "limit": 100}),
            ("read_result", {"id": "call_1", "offset": 280, "limit": 1000}),
            [("read_result", {"id": "call_9"}), ("read_result", {"offset": 1})],
            "done",
        ]
    )
    target = ReAct(system="s", model=model, max_result_chars=100)
    result = await harness.wrap(target, id="reader", tools=[report]).run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert offered(model, 0) == ["report"]
    assert offered(model, 1) == ["read_result", "report"]  # offered once a result was cut
    assert contents(model, 2)[-1] == (
        TEXT[50:150] + '\n…[150 more characters: read_result(id="call_1", offset=150)]'
    )
    assert contents(model, 3)[-1] == TEXT[280:300]  # the limit is the longest result shown
    assert contents(model, 4)[-2:] == [
        "there is no result 'call_9' to read",
        "read_result was not run: missing required argument(s): id",
    ]


async def test_a_resumed_run_names_the_same_artifact_and_keeps_it_once(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[bytes] = []
    upload = harness.runs.artifacts.upload

    async def counted(run_id: str, data: bytes, **kw: Any) -> Any:
        uploads.append(data)
        return await upload(run_id, data, **kw)

    monkeypatch.setattr(harness.runs.artifacts, "upload", counted)
    model = ScriptedChat([("report", {"name": "q3"}), ("publish", {"name": "q3"}), "done"])
    target = ReAct(system="s", model=model, max_result_chars=100)
    agent = harness.wrap(target, id="kept", tools=[report, publish])
    paused = await agent.run("x", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done"
    assert len(uploads) == 1 and len(model.requests) == 3  # nothing kept or asked twice


class Summarizing(ScriptedChat):
    """A scripted model that answers a request to compact with ``summary`` (none: no text),
    outside its script."""

    def __init__(self, turns: list[Any], summary: str | None = "Task: notes. Next: more.") -> None:
        super().__init__(turns)
        self.summary = summary

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        if messages[-1] != {"role": "user", "content": COMPACT}:
            return await super().complete(messages, **body)
        self.requests.append({"messages": [dict(m) for m in messages], **body})
        message = {"role": "assistant", "content": self.summary}
        return {"choices": [{"message": message}]}


class Windowed(Summarizing):
    """A model object saying how many tokens it reads."""

    context_window = 4400


async def test_older_results_are_cleared_past_half_the_window_and_read_back(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="read")
    def page(n: int) -> str:
        """A page of a book."""
        return f"page {n} " + "w" * 2000

    script: list[Any] = [("page", {"n": n}) for n in range(1, 6)]
    script += [("read_result", {"id": "call_1"}), ("publish", {"name": "book"}), "done"]
    model = Windowed(script)
    agent = memory_harness.wrap(ReAct(system="s", model=model), id="book", tools=[page, publish])
    paused = await agent.run("read the book", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    cleared = [
        n
        for n, request in enumerate(model.requests)
        if any(str(c).startswith("[result cleared") for c in contents(model, n))
    ]
    first = cleared[0]
    assert first > 1  # only once the results grew past half the window
    results = [c for c in contents(model, first) if str(c).startswith(("page", "[result"))]
    assert results[0] == (
        '[result cleared to keep the context small: read_result(id="call_1") reads it again]'
    )
    assert all(r.startswith("page") for r in results[-3:])  # the last three stay
    assert "read_result" in offered(model, first)
    assert contents(model, 6)[-1].startswith("page 1 ")  # read back in full
    # the memory context grows a thousand tokens before the resume (enough to have cleared
    # and compacted earlier): the steps before the pause are replayed as they were
    asked = len(model.requests)
    memory_service.context_text = "remembered " * 400
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done", done.error
    steps = [r for r in model.requests[asked:] if r["messages"][-1]["content"] != COMPACT]
    assert [r["messages"][-1]["content"] for r in steps] == ["published book"]


async def test_older_turns_are_compacted_into_one_summary_replayed_on_resume(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="read")
    def note(text: str) -> str:
        """Take a note."""
        return "noted"

    script: list[Any] = [("note", {"text": f"{n} " + "n" * 2400}) for n in range(4)]
    model = Summarizing([*script, ("publish", {"name": "notes"}), "done"])
    target = ReAct(system="s", model=model, context_window=4000)
    agent = memory_harness.wrap(target, id="notes", tools=[note, publish])
    paused = await agent.run("take five notes", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    [compacting] = [r for r in model.requests if r["messages"][-1]["content"] == COMPACT]
    assert model.requests.index(compacting) == 3  # past three quarters, once
    assert compacting["tool_choice"] == "none" and "response_format" not in compacting
    after = model.requests[model.requests.index(compacting) + 1]["messages"]
    assert [m["role"] for m in after[:3]] == ["system", "user", "user"]
    assert after[1]["content"] == "take five notes"
    assert after[2]["content"] == f"{SUMMARY}\n\nTask: notes. Next: more."
    # the recent turns are kept whole: a call, then its result
    assert after[3]["role"] == "assistant" and after[3]["tool_calls"]
    assert after[4]["tool_call_id"] == after[3]["tool_calls"][0]["id"]
    assert len(after) < len(compacting["messages"])
    asked = len(model.requests)
    memory_service.context_text = "remembered " * 400  # the context grows before the resume
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done", done.error
    # neither a step nor the summary asked again: only the step after the approval (fitted
    # anew to the larger context: compacted again first)
    again = model.requests[asked:]
    assert [r["messages"][-1]["content"] for r in again][-2:] == [COMPACT, "published notes"]
    assert all(r["messages"] != compacting["messages"] for r in again)


async def test_no_summary_compacts_nothing_and_a_single_turn_is_never_compacted(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    @tool(side_effects="read")
    def note(text: str) -> str:
        """Take a note."""
        return "noted"

    long = "n" * 600
    script: list[Any] = [("note", {"text": f"{n} {long}"}) for n in range(5)]
    model = Summarizing([*script, "done"], summary=None)
    target = ReAct(system="s", model=model, context_window=1000)
    result = await harness.wrap(target, id="nosum", tools=[note]).run("x", user="u")
    assert result.answer == "done"
    assert "no summary was written; nothing compacted" in caplog.text
    assert [m["role"] for m in model.requests[-1]["messages"][:3]] == [
        "system",
        "user",
        "assistant",
    ]

    tiny = Summarizing([("note", {"text": long}), "done"])
    small = ReAct(system="s", model=tiny, context_window=100)
    assert (await harness.wrap(small, id="tiny", tools=[note]).run("x", user="u")).answer == "done"
    assert len(tiny.requests) == 2  # one turn is all there is: nothing to compact


async def test_a_message_list_without_a_user_message_still_fits(harness: Harness) -> None:
    model = ScriptedChat(["fine"])
    target = ReAct(system="s", model=model, context_window=10)
    agent = harness.wrap(target, id="nouser")
    result = await agent.run([{"role": "assistant", "content": "hello " * 50}], user="u")
    assert result.answer == "fine"
    assert json.dumps(model.requests[0]["messages"]).count("hello") == 50
