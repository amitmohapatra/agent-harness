"""AG-UI parity with a CopilotKit-style client: its frontend tools, its context, its shared
state and the message's mode, driven by a fake AG-UI client over a real FastAPI app."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.harness.agui.sse import decode

PATH = "/agui"
PICK = {
    "name": "pick_color",
    "description": "Ask the person to pick a color in the app's color picker.",
    "parameters": {
        "type": "object",
        "properties": {"hint": {"type": "string"}},
        "required": ["hint"],
    },
}
CONTEXT = [{"description": "the page the person is on", "value": "Checkout, order 91"}]


class FakeClient:
    """An AG-UI client as CopilotKit drives one: it keeps the thread's messages and state,
    declares its tools, applies state patches, and answers a frontend tool call with a
    ``tool`` message on the next run."""

    def __init__(self, http: httpx.AsyncClient, *, tools: list[dict[str, Any]]) -> None:
        self.http, self.tools = http, tools
        self.messages: list[dict[str, Any]] = []
        self.state: Any = {"cart": ["tea"]}

    async def send(self, text: str, **extra: Any) -> list[dict[str, Any]]:
        self.messages.append({"id": f"u{len(self.messages)}", "role": "user", "content": text})
        return await self.run(**extra)

    async def run(self, **extra: Any) -> list[dict[str, Any]]:
        body = {
            "threadId": "t1",
            "messages": self.messages,
            "tools": self.tools,
            "context": CONTEXT,
            "state": self.state,
            **extra,
        }
        response = await self.http.post(f"{PATH}/run", json=body)
        assert response.status_code == 200, response.text
        events = [e for _, e in decode(response.text)]
        for event in events:
            if event["type"] == "STATE_SNAPSHOT":
                self.state = event["snapshot"]
            elif event["type"] == "STATE_DELTA":
                import jsonpatch

                self.state = jsonpatch.apply_patch(self.state, event["delta"])
        return events

    async def answer_tool(self, call_id: str, result: str) -> list[dict[str, Any]]:
        self.messages.append(
            {
                "id": f"r{len(self.messages)}",
                "role": "tool",
                "toolCallId": call_id,
                "content": result,
            }
        )
        return await self.run()


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"about {topic}"


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h = Harness(config=Settings())
    yield h
    await h.aclose()


def served(agent: Any) -> httpx.AsyncClient:
    app = FastAPI()
    agent.serve_chat(app, identity=lambda request: "u1")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agui")


def of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == kind]


async def test_a_frontend_tool_is_offered_streamed_paused_and_answered_by_a_tool_message(
    harness: Harness,
) -> None:
    model = PlannedChat([("pick_color", {"hint": "warm"})])
    agent = harness.wrap(ReAct(system_prompt="Help.", model=model), id="painter")
    async with served(agent) as http:
        client = FakeClient(http, tools=[PICK])
        events = await client.send("Paint the wall")

        offered = [t["function"]["name"] for t in model.script.requests[0]["tools"]]
        assert "pick_color" in offered
        [start] = of(events, "TOOL_CALL_START")
        assert start["toolCallName"] == "pick_color"
        call_id = start["toolCallId"]
        [args] = of(events, "TOOL_CALL_ARGS")
        assert json.loads(args["delta"]) == {"hint": "warm"}
        assert of(events, "TOOL_CALL_END") and not of(events, "TOOL_CALL_RESULT")
        last = events[-1]
        assert last["type"] == "RUN_FINISHED" and last["outcome"]["type"] == "interrupt"
        [entry] = last["outcome"]["interrupts"]
        assert entry["reason"] == "frontend_tool" and entry["toolCallId"] == call_id
        assert entry["metadata"]["tool_call"] == {
            "tool": "pick_color",
            "args": {"hint": "warm"},
            "task": "Paint the wall",
            "idempotency_key": call_id,
        }

        resumed = await client.answer_tool(call_id, "red")
        [result] = of(resumed, "TOOL_CALL_RESULT")
        assert result["toolCallId"] == call_id and result["content"] == "red"
        assert resumed[-1]["outcome"] == {"type": "success"}
        assert resumed[-1]["result"] == "Done. red"  # the model read the app's result


async def test_a_frontend_tool_is_answered_by_a_resume_entry_too(harness: Harness) -> None:
    async def follow(input: Any, agent: Runtime) -> str:
        picked = await agent.tools.call("pick_color", hint="cool")
        return f"painted {picked}"

    agent = harness.wrap(follow, id="by-resume")
    async with served(agent) as http:
        client = FakeClient(http, tools=[PICK])
        paused = (await client.send("Paint"))[-1]
        [entry] = paused["outcome"]["interrupts"]
        resume = [{"interruptId": entry["id"], "payload": "blue"}]
        done = await client.run(resume=resume)
        assert done[-1]["result"] == "painted blue"


async def test_a_tool_result_nothing_waits_on_is_refused(harness: Harness) -> None:
    async def echo(input: Any, agent: Runtime) -> str:
        return f"echo {input}"

    agent = harness.wrap(echo, id="nothing-waits")
    async with served(agent) as http:
        client = FakeClient(http, tools=[PICK])
        await client.send("hi")
        client.messages.append({"id": "r", "role": "tool", "toolCallId": "call_x", "content": "x"})
        response = await http.post(
            f"{PATH}/run", json={"threadId": "t1", "messages": client.messages}
        )
        assert response.status_code == 409
        assert "call_x" in response.json()["detail"]


async def test_a_tool_result_answers_only_a_frontend_tools_pause(harness: Harness) -> None:
    async def asking(input: Any, agent: Runtime) -> str:
        return str(await agent.ask("Which size?", options=["S", "L"]))

    agent = harness.wrap(asking, id="asks-a-person")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "buy"}]
        first = await http.post(
            f"{PATH}/run", json={"threadId": "t1", "messages": said, "tools": [PICK]}
        )
        assert [e for _, e in decode(first.text)][-1]["outcome"]["type"] == "interrupt"
        result = {"id": "r", "role": "tool", "toolCallId": "call_1", "content": "L"}
        response = await http.post(
            f"{PATH}/run", json={"threadId": "t1", "messages": [*said, result]}
        )
    assert response.status_code == 409  # a person's question is answered by a resume entry


async def test_the_agents_own_tool_wins_over_a_frontend_tool_named_alike(
    harness: Harness,
) -> None:
    from tests.support.arm import arm

    model = PlannedChat([("lookup", {"topic": "tides"})])
    agent = arm(harness.wrap(ReAct(system_prompt="Help.", model=model), id="clash"), [lookup])
    clash = {"name": "lookup", "description": "the app's own lookup"}
    async with served(agent) as http:
        events = await FakeClient(http, tools=[clash]).send("Look up tides")
    assert events[-1]["result"] == "Done. about tides"  # the server's tool ran
    warned = [e["value"] for e in of(events, "CUSTOM") if e["name"] == "warning"]
    assert any(w["code"] == "frontend_tools" and "lookup" in w["message"] for w in warned)


async def test_the_context_and_the_state_reach_the_model(harness: Harness) -> None:
    model = PlannedChat([])
    agent = harness.wrap(ReAct(system_prompt="Help.", model=model), id="reader")
    async with served(agent) as http:
        await FakeClient(http, tools=[]).send("What is in my cart?")
    told = json.dumps(model.script.requests[0]["messages"])
    assert "the page the person is on: Checkout, order 91" in told
    assert 'shared state (JSON): {\\"cart\\": [\\"tea\\"]}' in told


async def test_a_state_the_run_changes_streams_as_a_patch_the_client_applies(
    harness: Harness,
) -> None:
    async def shop(input: Any, agent: Runtime) -> str:
        assert agent.state == {"cart": ["tea"]}  # the request's state, as state
        agent.state = {"cart": ["tea"]}  # the same: no change to stream
        agent.state = {**agent.state, "cart": [*agent.state["cart"], "milk"]}
        return "added milk"

    agent = harness.wrap(shop, id="shop")
    async with served(agent) as http:
        client = FakeClient(http, tools=[])
        events = await client.send("Add milk")
    [delta] = of(events, "STATE_DELTA")
    assert delta["delta"] == [{"op": "add", "path": "/cart/1", "value": "milk"}]
    assert client.state == {"cart": ["tea", "milk"]}


async def test_a_state_set_where_there_was_none_is_a_snapshot(harness: Harness) -> None:
    async def start(input: Any, agent: Runtime) -> str:
        agent.state = {"step": 1}
        return "ok"

    agent = harness.wrap(start, id="fresh")
    async with served(agent) as http:
        response = await http.post(
            f"{PATH}/run",
            json={"threadId": "t1", "messages": [{"role": "user", "content": "go"}]},
        )
    events = [e for _, e in decode(response.text)]
    assert [e["snapshot"] for e in of(events, "STATE_SNAPSHOT")] == [{"step": 1}]


async def test_a_resume_reads_the_state_the_client_holds(harness: Harness) -> None:
    seen: list[Any] = []

    async def shop(input: Any, agent: Runtime) -> str:
        seen.append(agent.state)
        picked = await agent.tools.call("pick_color", hint="any")
        return f"{picked} with {agent.state['cart']}"

    agent = harness.wrap(shop, id="stateful")
    async with served(agent) as http:
        client = FakeClient(http, tools=[PICK])
        paused = (await client.send("Go"))[-1]
        call_id = paused["outcome"]["interrupts"][0]["toolCallId"]
        client.state = {"cart": ["tea", "jam"]}  # the person changed it meanwhile
        done = await client.answer_tool(call_id, "green")
    assert seen == [{"cart": ["tea"]}, {"cart": ["tea", "jam"]}]
    assert done[-1]["result"] == "green with ['tea', 'jam']"


async def test_the_mode_comes_in_forwarded_props(harness: Harness) -> None:
    release = asyncio.Event()

    async def slow(input: Any, agent: Runtime) -> str:
        if input == "first":
            await release.wait()
        return f"answered {input}"

    agent = harness.wrap(slow, id="busy")
    async with served(agent) as http:
        first = asyncio.create_task(
            http.post(f"{PATH}/run", json={"threadId": "t1", "messages": _said("first")})
        )
        await asyncio.sleep(0.2)
        second = await http.post(
            f"{PATH}/run",
            json={
                "threadId": "t1",
                "messages": _said("second"),
                "forwardedProps": {"mode": "interrupt"},
            },
        )
        cancelled = [e for _, e in decode((await first).text)][-1]
        assert cancelled["outcome"] == {"type": "cancelled"}  # the newer message took over
        assert [e for _, e in decode(second.text)][-1]["result"] == "answered second"
        refused = await http.post(
            f"{PATH}/run",
            json={"threadId": "t1", "messages": _said("x"), "forwardedProps": {"mode": "now"}},
        )
        assert refused.status_code == 422
        release.set()


def _said(text: str) -> list[dict[str, Any]]:
    return [{"id": "m", "role": "user", "content": text}]


def test_the_context_and_state_are_fitted_to_their_budget() -> None:
    from trellis.harness import frontend
    from trellis.harness.frontend import Frontend

    entries = tuple({"description": f"field {n}", "value": "v" * 30} for n in range(20))
    text, tokens = frontend.rendered(Frontend(context=entries), None, budget=60)
    assert text is not None and tokens <= 60
    assert "field 0: " in text and "more left out: over the context budget" in text
    assert text.count("field ") < 20
    big = {"rows": ["x" * 50] * 40}
    text, tokens = frontend.rendered(Frontend(state=big, stated=True), big, budget=100)
    assert text is not None and tokens <= 100 and text.endswith("(cut: over the context budget)")
    assert frontend.rendered(Frontend(state={}, stated=True), {}, budget=100) == (None, 0)
