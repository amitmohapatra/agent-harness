"""What leaves the process is redacted, once, where it leaves: a tool call's arguments and
output on the run's event stream (``agent.stream``, AG-UI, A2A task updates and push bodies)
and in the memory service's tool records. The tool and the model get the values as they are."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.helpers import new_message, new_text_part
from a2a.types import (
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskPushNotificationConfig,
)
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict

from tests.integration.test_a2a_server import URL, asgi, caller, connect
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEventType
from trellis.harness.a2a import push
from trellis.harness.agui.sse import decode
from trellis.harness.events import LOG, RunEvents
from trellis.harness.identity import Identity
from trellis.harness.redaction import DEFAULT as REDACTOR

EMAIL = "ada@example.com"
SECRET = "sk-live0123456789abcdefghij"
ARGS = {"email": EMAIL, "api_key": SECRET}
SHOWN_ARGS = {"email": "[email]", "api_key": "[redacted]"}
SHOWN_OUTPUT = {"sent_to": "[email]", "receipt": "[redacted]"}
received: list[dict[str, str]] = []


@tool(side_effects="write")
def notify(email: str, api_key: str) -> dict[str, str]:
    """Notify a customer."""
    received.append({"email": email, "api_key": api_key})
    return {"sent_to": email, "receipt": api_key}


async def notifying(input: Any, agent: Runtime) -> str:
    await agent.tools.call("notify", **ARGS)
    agent.log("notified", card_number="4111 1111 1111 1111", customer=EMAIL)
    return "sent"


@pytest.fixture(autouse=True)
def _reset() -> None:
    received.clear()


def leaked(text: str) -> bool:
    return EMAIL in text or SECRET in text or "4111" in text


async def test_the_event_stream_carries_tool_calls_redacted(harness: Harness) -> None:
    agent = harness.wrap(notifying, id="notifier", tools=[notify])
    events = [e async for e in agent.stream("go", user="u")]
    data = {e.type: e.data for e in events if e.tool_call_id is not None}
    assert data[RunEventType.TOOL_CALL_ARGS]["args"] == SHOWN_ARGS
    assert data[RunEventType.TOOL_CALL_RESULT]["output"] == SHOWN_OUTPUT
    [logged] = [e.data for e in events if e.data.get("name") == LOG]
    assert logged["card_number"] == "[redacted]" and logged["customer"] == "[email]"
    assert received == [ARGS]  # the tool itself got the values as they are


async def test_the_model_reads_the_tool_output_as_it_is(harness: Harness) -> None:
    model = ScriptedChat([("notify", ARGS), "done"])
    agent = harness.wrap(ReAct(system="You notify.", model=model), id="react", tools=[notify])
    events = [e async for e in agent.stream("go", user="u")]
    told = model.requests[1]["messages"][-1]
    assert told["role"] == "tool" and SECRET in told["content"] and EMAIL in told["content"]
    assert not leaked(json.dumps([e.model_dump(mode="json") for e in events]))


def test_a_value_is_redacted_once_and_a_long_output_is_a_preview() -> None:
    identity = Identity(tenant="t", user="u", agent_id="a", run_id="r", thread="r")
    events, seen = RunEvents(identity.context()), []
    events.tool(RunEventType.TOOL_CALL_RESULT, "c1", output="x" * 5000)
    events.listen(seen.append)
    long = {"text": "y" * 3000, "token": "abc"}
    events.tool(RunEventType.TOOL_CALL_ARGS, "c1", args=long)
    events.tool(RunEventType.TOOL_CALL_RESULT, "c1", output=[{"n": i} for i in range(500)])
    args, output = seen[0].data["args"], seen[1].data["output"]
    assert args == REDACTOR.redact_input(long) and args["token"] == "[redacted]"
    assert isinstance(output, str) and output.endswith("…") and len(output) == 2001
    assert events.sequence == 2  # nothing was built while nobody listened


async def test_an_agui_stream_carries_tool_calls_redacted() -> None:
    async with Harness(config=Settings()) as harness:
        app = FastAPI()
        harness.wrap(notifying, id="chat", tools=[notify]).serve_chat(
            app, identity=lambda request: "u1"
        )
        async with asgi(app) as http:
            response = await http.post(
                "/agui/run",
                json={"threadId": "t1", "messages": [{"id": "m", "role": "user", "content": "go"}]},
            )
    events = {e["type"]: e for _, e in decode(response.text)}
    assert json.loads(events["TOOL_CALL_ARGS"]["delta"]) == SHOWN_ARGS
    assert json.loads(events["TOOL_CALL_RESULT"]["content"]) == SHOWN_OUTPUT
    assert not leaked(response.text) and received == [ARGS]


@pytest.fixture
async def a2a_pushes(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[bytes]]:
    """The push bodies the A2A server delivers (every target is taken as public)."""
    bodies: list[bytes] = []

    async def public(url: str) -> None:
        return None

    async def deliver(
        self: push.PushNotifier, url: str, token: str, task: str, body: bytes
    ) -> bool:
        bodies.append(body)
        return True

    monkeypatch.setattr(push, "check_addresses", public)
    monkeypatch.setattr(push.PushNotifier, "deliver", deliver)
    yield bodies


async def test_a2a_task_updates_and_pushes_carry_tool_calls_redacted(
    a2a_pushes: list[bytes],
) -> None:
    async with Harness(config=Settings()) as harness:
        app = FastAPI()
        harness.wrap(notifying, id="greeter", tools=[notify]).serve_a2a(app, URL)
        async with asgi(app) as http:
            client = await connect(http)
            hook = TaskPushNotificationConfig(url="https://hooks.example.com/a2a", token="tok")
            request = SendMessageRequest(
                message=new_message([new_text_part("go")], role=Role.ROLE_USER),
                configuration=SendMessageConfiguration(task_push_notification_config=hook),
            )
            responses = [r async for r in client.send_message(request, context=caller())]
    updates = json.dumps([MessageToDict(r) for r in responses])
    assert '"[redacted]"' in updates and '"[email]"' in updates
    assert not leaked(updates) and received == [ARGS]
    assert a2a_pushes and not any(leaked(body.decode()) for body in a2a_pushes)
    assert any(b"[redacted]" in body for body in a2a_pushes)


async def test_the_memory_services_tool_records_are_redacted(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    spooled: list[dict[str, Any] | None] = []
    submit = memory_harness.writes.submit

    async def keep(label: str, work: Any, **kw: Any) -> None:
        if label == "memory.record_tool":
            spooled.append(kw.get("record"))
        await submit(label, work, **kw)

    monkeypatch.setattr(memory_harness.writes, "submit", keep)
    agent = memory_harness.wrap(notifying, id="notifier", tools=[notify])
    assert (await agent.run("go", user="u")).answer == "sent"
    await memory_harness.writes.drain()
    [call] = [c for c in memory_service.named("record_tool") if c.body["tool"] == "notify"]
    assert call.body["args"] == SHOWN_ARGS and call.body["output"] == SHOWN_OUTPUT
    [record] = spooled  # what the spool keeps if the write cannot be delivered
    assert record is not None and not leaked(json.dumps(record))
    assert received == [ARGS]
