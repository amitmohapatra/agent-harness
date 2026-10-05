"""An answer must fit its question before a paused run continues — checked by the harness for
every run, kept in process or in agent-runs: a question's schema when it is asked, an answer
or a corrected value against ``expects`` (else ``options``), and a reviewer's edited arguments
against the tool's own input schema. A refusal is a ``ConfigurationError`` saying why, raised
before anything is sent or recorded; the run keeps waiting."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from trellis import Harness, Runtime, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.agent import Agent

refunded: list[dict[str, Any]] = []


@tool(side_effects="irreversible")
def refund(order: str, amount: float) -> str:
    """Refund an order."""
    refunded.append({"order": order, "amount": amount})
    return f"refunded {amount} on {order}"


async def asking(input: str, agent: Runtime) -> Any:
    if input == "size":
        return await agent.ask("Which size?", options=["S", "L"])
    if input == "count":
        return await agent.ask("How many?", expects={"type": "integer", "minimum": 1})
    if input == "malformed":
        return await agent.ask("How many?", expects={"type": "integr"})
    return await agent.tools.call("refund", order="o1", amount=500)


@pytest.fixture(autouse=True)
def _reset() -> None:
    refunded.clear()


async def paused_on(agent: Agent, input: str) -> str:
    paused = await agent.run(input, user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    return paused.interrupt.interrupt_id


async def still_waiting(harness: Harness, interrupt_id: str) -> None:
    record = await harness.runs.get(interrupt_id.rsplit(".", 2)[0])
    assert record is not None and record.status is RunStatus.PAUSED
    assert record.awaiting is not None and record.awaiting.interrupt_id == interrupt_id


async def test_a_malformed_schema_is_refused_where_it_is_asked(harness: Harness) -> None:
    failed = await harness.wrap(asking, id="asking").run("malformed", user="u1")
    assert failed.status is RunStatus.ERROR and failed.error is not None
    assert failed.error.code == "CONFIGURATION_ERROR"
    assert "cannot ask 'How many?': expects is not a valid JSON Schema" in failed.error.message


async def test_an_answer_that_does_not_fit_is_refused_and_the_run_keeps_waiting(
    harness: Harness,
) -> None:
    agent = harness.wrap(asking, id="asking")
    counted = await paused_on(agent, "count")
    with pytest.raises(ConfigurationError, match=r"not an answer to .*'five' is not of type"):
        await agent.resume(counted, "answer", answer="five", reviewer="u1")
    with pytest.raises(ConfigurationError, match="0 is less than the minimum of 1"):
        await agent.resume(counted, "answer", answer=0, reviewer="u1")
    await still_waiting(harness, counted)
    assert (await agent.resume(counted, "answer", answer=3, reviewer="u1")).answer == 3

    sized = await paused_on(agent, "size")
    with pytest.raises(ConfigurationError, match=r"'M' is not one of the options \['S', 'L'\]"):
        await agent.resume(sized, "answer", answer="M", reviewer="u1")
    await still_waiting(harness, sized)
    assert (await agent.resume(sized, "answer", answer="L", reviewer="u1")).answer == "L"


async def test_an_edit_that_does_not_fit_the_tool_is_refused_saying_which_argument(
    harness: Harness,
) -> None:
    agent = harness.wrap(asking, id="asking", tools=[refund])
    asked = await paused_on(agent, "refund")
    with pytest.raises(ConfigurationError, match="do not fit refund: amount must be of type"):
        await agent.resume(asked, "edit", answer={"order": "o1", "amount": "50"}, reviewer="u1")
    with pytest.raises(ConfigurationError, match=r"missing required argument\(s\): amount"):
        await agent.resume(asked, "edit", answer={"order": "o1"}, reviewer="u1")
    await still_waiting(harness, asked)
    assert refunded == []
    done = await agent.resume(asked, "edit", answer={"order": "o1", "amount": 50}, reviewer="u1")
    assert done.answer == "refunded 50.0 on o1" and refunded == [{"order": "o1", "amount": 50}]


async def test_an_edit_of_a_tool_the_toolbox_cannot_list_now_is_left_to_the_tool(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = harness.wrap(asking, id="asking", tools=[refund])
    asked = await paused_on(agent, "refund")

    async def down() -> list[Any]:
        raise httpx.ConnectError("the MCP server is down")

    monkeypatch.setattr(agent._toolbox(await harness.tenant()), "tools", down)
    record, resolution = await agent._resolution(
        asked, "edit", {"order": "o1", "amount": "lots"}, "u1", tenant=await harness.tenant()
    )
    assert resolution.payload == {"order": "o1", "amount": "lots"}  # not checked here
    monkeypatch.undo()
    done = await agent._continue(record, resolution)
    assert done.answer is not None and "refund failed" in done.answer  # the tool checked it
    assert refunded == []


async def test_a_surface_says_why_an_answer_was_refused(harness: Harness) -> None:
    app = FastAPI()
    agent = harness.wrap(asking, id="asking")
    agent.serve_chat(app, identity=lambda request: "u1")
    paused = await agent.run("size", user="u1", thread="t1")
    assert paused.interrupt is not None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://agui") as http:
        response = await http.post(
            "/agui/run",
            json={
                "threadId": "t1",
                "messages": [],
                "resume": [{"interruptId": paused.interrupt.interrupt_id, "payload": "M"}],
            },
        )
    assert response.status_code == 409
    assert "is not one of the options" in response.json()["detail"]
