"""``ask_user`` and the activity line over AG-UI: what a chat UI renders (a picker for a
date range, a dropdown for options, a number field), what plain chat reads (the question with
the kind of answer it takes), the answer the model reads; and the run's tool calls grouped as
one "used N tools · ran K commands" line."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.support.arm import arm
from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Runtime, Settings, ask_user, tool
from trellis.contracts import ConfigurationError
from trellis.harness.agui.sse import decode
from trellis.harness.tools.ask import EXPECTS

PATH = "/agui"


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h = Harness(config=Settings())
    yield h
    await h.aclose()


async def post(http: httpx.AsyncClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    response = await http.post(f"{PATH}/run", json={"threadId": "t1", "messages": [], **body})
    assert response.status_code == 200, response.text
    return [e for _, e in decode(response.text)]


def served(agent: Any) -> httpx.AsyncClient:
    app = FastAPI()
    agent.serve_chat(app, identity=lambda request: "u1")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agui")


def asking(harness: Harness, args: dict[str, Any], name: str) -> Any:
    model = PlannedChat([("ask_user", args)])
    return arm(harness.wrap(ReAct(system_prompt="Help.", model=model), id=name), [ask_user()])


@pytest.mark.parametrize(
    ("kind", "answer"),
    [
        ("date_range", {"start": "2026-11-02", "end": "2026-11-06"}),
        ("date_range", "the first week of November"),  # plain chat answers in words
        ("date", "2026-11-05"),
        ("number", 3),
        ("number", "three"),
    ],
)
async def test_a_structured_question_renders_and_reads_back(
    harness: Harness, kind: str, answer: Any
) -> None:
    agent = asking(harness, {"question": "When should it arrive?", "kind": kind}, f"ask-{kind}")
    said = [{"id": "m", "role": "user", "content": "Ship my order"}]
    async with served(agent) as http:
        paused = (await post(http, {"messages": said}))[-1]
        [entry] = paused["outcome"]["interrupts"]
        assert entry["reason"] == "question"
        assert entry["message"].startswith("When should it arrive? (a ")  # what plain chat reads
        assert entry["metadata"]["component"] == "ask_user"
        assert entry["metadata"]["props"] == {"kind": kind, "options": []}
        assert entry["responseSchema"] == EXPECTS[kind]  # a structured value, or text
        resume = [{"interruptId": entry["id"], "payload": answer}]
        done = (await post(http, {"resume": resume}))[-1]
    assert done["outcome"] == {"type": "success"}
    assert str(answer if not isinstance(answer, dict) else "2026-11-06") in done["result"]


async def test_options_are_a_choice_plain_chat_reads_numbered(harness: Harness) -> None:
    """A choice with no reason to recommend one: the options as they are (plain strings on the
    wire), numbered in plain chat, and another answer in words is the "Other"."""
    args = {"question": "Which size?", "options": ["S", "L", "S"]}
    agent = asking(harness, args, "ask-size")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "Order a shirt"}]
        paused = (await post(http, {"messages": said}))[-1]
        [entry] = paused["outcome"]["interrupts"]
        assert entry["reason"] == "choice"
        assert entry["message"] == "Which size?\n1. S\n2. L\n(Pick one, or say another.)"
        assert entry["metadata"]["options"] == ["S", "L"]
        assert entry["metadata"]["props"] == {"kind": "text", "options": ["S", "L"]}
        other = [{"interruptId": entry["id"], "payload": "M, if you have it"}]
        done = (await post(http, {"resume": other}))[-1]
    assert done["result"] == "Done. M, if you have it"


async def test_a_recommended_option_says_why_and_several_may_be_picked(
    harness: Harness,
) -> None:
    args = {
        "question": "Which regions?",
        "options": ["north", "south", "east"],
        "multiple": True,
        "recommended": "north",
        "why": "your choice last time",
    }
    agent = asking(harness, args, "ask-regions")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "Set the alert"}]
        [entry] = (await post(http, {"messages": said}))[-1]["outcome"]["interrupts"]
        assert entry["message"] == (
            "Which regions?\n1. north (Recommended: your choice last time)\n2. south\n3. east"
            "\n(Pick one or more, or say another.)"
        )
        assert entry["metadata"]["options"] == [
            {"value": "north", "description": "your choice last time", "recommended": True},
            "south",
            "east",
        ]
        assert entry["metadata"]["multiple"] is True
        picked = [{"interruptId": entry["id"], "payload": ["north", "east"]}]
        done = (await post(http, {"resume": picked}))[-1]
    assert done["result"] == 'Done. ["north", "east"]'


async def test_the_recommended_option_comes_first_wherever_the_model_listed_it(
    harness: Harness,
) -> None:
    """The option to pick is the one a person reads first: in the choice a UI renders, in the
    app's own control and in the numbered text."""
    args = {"question": "Which region?", "options": ["north", "south", "west"],
            "recommended": "west", "why": "least stock"}  # fmt: skip
    agent = asking(harness, args, "ask-first")
    said = [{"id": "m", "role": "user", "content": "Send the pallet"}]
    async with served(agent) as http:
        [entry] = (await post(http, {"messages": said}))[-1]["outcome"]["interrupts"]
    assert entry["metadata"]["options"] == [
        {"value": "west", "description": "least stock", "recommended": True},
        "north",
        "south",
    ]
    assert entry["metadata"]["props"]["options"] == ["west", "north", "south"]
    assert entry["message"] == (
        "Which region?\n1. west (Recommended: least stock)\n2. north\n3. south"
        "\n(Pick one, or say another.)"
    )


@pytest.mark.parametrize(
    "args",
    [
        {"recommended": "L"},  # no reason given: nothing is recommended
        {"recommended": "XL", "why": "it fits most"},  # not one of the options
    ],
)
async def test_no_option_is_recommended_without_a_reason_among_the_options(
    harness: Harness, args: dict[str, Any]
) -> None:
    agent = asking(harness, {"question": "Which size?", "options": ["S", "L"], **args}, "ask-n")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "Order a shirt"}]
        [entry] = (await post(http, {"messages": said}))[-1]["outcome"]["interrupts"]
    assert entry["metadata"]["options"] == ["S", "L"] and "Recommended" not in entry["message"]


async def test_ask_user_runs_only_in_a_run() -> None:
    with pytest.raises(ConfigurationError, match="inside a harness run"):
        await ask_user()("When?")


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"about {topic}"


@tool(name="execute", side_effects="read")
async def run_command(command: str) -> str:
    """Run a command."""
    return f"ran {command}"


async def test_the_runs_tool_calls_are_one_activity_line(harness: Harness) -> None:
    async def busy(input: Any, agent: Runtime) -> str:
        await agent.tools.call("lookup", topic="a")
        await agent.tools.call("lookup", topic="b")
        await agent.tools.call("execute", command="ls")
        return "done"

    agent = arm(harness.wrap(busy, id="busy"), [lookup, run_command])
    async with served(agent) as http:
        events = await post(http, {"messages": [{"id": "m", "role": "user", "content": "go"}]})
    lines = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert [e["content"]["summary"] for e in lines] == [
        "used 1 tool",
        "used 2 tools",
        "used 2 tools · ran 1 command",
    ]
    assert {e["messageId"] for e in lines} == {f"{events[0]['runId']}:activity"}
    assert all(e["activityType"] == "tools" and e["replace"] is True for e in lines)
    assert lines[-1]["content"] == {
        "summary": "used 2 tools · ran 1 command",
        "tools": 2,
        "commands": 1,
    }
    # each line follows the call it counts
    results = [i for i, e in enumerate(events) if e["type"] == "TOOL_CALL_RESULT"]
    assert [events[i + 1]["type"] for i in results] == ["ACTIVITY_SNAPSHOT"] * 3


async def test_a_call_replayed_on_resume_is_counted_once(harness: Harness) -> None:
    async def twice(input: Any, agent: Runtime) -> str:
        await agent.tools.call("lookup", topic="a")
        size = await agent.ask("Size?", options=["S", "L"])
        return f"size {size}"

    agent = arm(harness.wrap(twice, id="replays"), [lookup])
    async with served(agent) as http:
        first = await post(http, {"messages": [{"id": "m", "role": "user", "content": "go"}]})
        entry = first[-1]["outcome"]["interrupts"][0]
        resumed = await post(http, {"resume": [{"interruptId": entry["id"], "payload": "S"}]})
    counted = [
        e["content"]["tools"] for e in [*first, *resumed] if e["type"] == "ACTIVITY_SNAPSHOT"
    ]
    assert counted == [1, 1]  # the replayed lookup has the same id: the line says 1 again
