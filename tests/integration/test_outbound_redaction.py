"""What leaves the process, and who masks it: traces mask credentials (``redaction.py``);
everything else carries values as they are — the run's events, its answer, memory, A2A, the
run's record — and what a user is shown passes the ``on_output`` hooks (``Hooks.on_output``):
the answer ``agent.run`` returns and every event of ``agent.stream`` and AG-UI. The tool and
the model always get the values as they are."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from a2a.helpers import new_message, new_text_part
from a2a.types import (
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskPushNotificationConfig,
)
from agents import Agent as OpenAIAgent
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict
from langchain.agents import create_agent
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.integration.test_a2a_server import URL, asgi, caller, connect
from tests.integration.test_claude import options
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, Hooks, Output, OutputKind, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEventType
from trellis.harness import telemetry
from trellis.harness.a2a import push
from trellis.harness.agui.sse import decode
from trellis.harness.events import LOG, RunEvents
from trellis.harness.hooks import WITHHELD, Chain
from trellis.harness.identity import Identity

EMAIL = "ada@example.com"
SECRET = "sk-live0123456789abcdefghij"
CARD = "4111 1111 1111 1111"
ARGS = {"email": EMAIL, "api_key": SECRET}
OUTPUT = {"sent_to": EMAIL, "receipt": SECRET}
received: list[dict[str, str]] = []


@tool(side_effects="write")
def notify(email: str, api_key: str) -> dict[str, str]:
    """Notify a customer."""
    received.append({"email": email, "api_key": api_key})
    return {"sent_to": email, "receipt": api_key}


async def notifying(input: Any, agent: Runtime) -> str:
    await agent.tools.call("notify", **ARGS)
    agent.log("notified", card_number=CARD, customer=EMAIL)
    return f"sent to {EMAIL}"


def masked(text: str) -> str:
    return text.replace(EMAIL, "[email]").replace(SECRET, "[key]").replace(CARD, "[card]")


class Mask(Hooks):
    """A team's own masking of what its users see."""

    def on_output(self, item: Output) -> Output:
        if item.kind is OutputKind.ANSWER:
            return item.replace(masked(item.value))
        event = item.value
        data = json.loads(masked(json.dumps(event.data)))
        return item.replace(event.model_copy(update={"data": data}))


class Boom(Hooks):
    def on_output(self, item: Output) -> Output:
        raise RuntimeError("the masking service is down")


@pytest.fixture(autouse=True)
def _reset() -> None:
    received.clear()


def leaked(text: str) -> bool:
    """Whether a value is there as it is: the values themselves, never a fragment of one (a
    random id — a run's, an event's — holds any four digits now and then)."""
    return EMAIL in text or SECRET in text or CARD in text


def dumped(events: list[Any]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in events])


async def test_the_event_stream_and_the_answer_carry_values_as_they_are(harness: Harness) -> None:
    agent = arm(harness.wrap(notifying, id="notifier"), [notify])
    events = [e async for e in agent.stream("go", user="u")]
    data = {e.type: e.data for e in events if e.tool_call_id is not None}
    assert data[RunEventType.TOOL_CALL_ARGS]["args"] == ARGS
    assert data[RunEventType.TOOL_CALL_RESULT]["output"] == OUTPUT
    [logged] = [e.data for e in events if e.data.get("name") == LOG]
    assert logged["card_number"] == CARD and logged["customer"] == EMAIL
    assert (await agent.run("go", user="u")).answer == f"sent to {EMAIL}"
    assert agent.hooks.shows == []  # no on_output hook: nothing sent is wrapped
    assert received == [ARGS, ARGS]


async def test_an_on_output_hook_masks_the_answer_and_the_stream_not_the_record() -> None:
    async with Harness(config=Settings()) as harness:
        agent = arm(harness.wrap(notifying, id="notifier", hooks=[Mask()]), [notify])
        events = [e async for e in agent.stream("go", user="u")]
        result = await agent.run("go", user="u")
        record = await harness.runs.get(result.run_id, tenant=await harness._tenant())
    assert not leaked(dumped(events)) and "[key]" in dumped(events)
    assert result.answer == "sent to [email]"
    assert record is not None and record.output == f"sent to {EMAIL}"  # the run's own record
    assert received == [ARGS, ARGS]  # the tool got the values as they are


async def test_the_model_reads_the_tool_output_as_it_is(harness: Harness) -> None:
    model = ScriptedChat([("notify", ARGS), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="You notify.", model=model), id="react"), [notify])
    [e async for e in agent.stream("go", user="u")]
    told = model.requests[1]["messages"][-1]
    assert told["role"] == "tool" and SECRET in told["content"] and EMAIL in told["content"]


async def test_traces_mask_credentials_only(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``chat`` span's input and output hold the model's tool calls, whose arguments are JSON
    text, and the tool results as the model read them: the key masked, the address kept
    (personal data is the team's to mask, in its own hooks)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    model = ScriptedChat([("notify", ARGS), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="You notify.", model=model), id="react"), [notify])
    assert (await agent.run("go", user="u")).answer == "done"
    chats = [s for s in exporter.get_finished_spans() if s.name.startswith("chat")]
    assert len(chats) == 2
    shown = json.dumps([dict(s.attributes or {}) for s in chats])
    assert SECRET not in shown and "[redacted]" in shown and EMAIL in shown
    assert received == [ARGS]


def test_a_long_output_is_a_preview_and_nothing_is_built_unheard() -> None:
    identity = Identity(tenant="t", user="u", agent_id="a", run_id="r", thread="r")
    events, seen = RunEvents(identity.context()), []
    events.tool(RunEventType.TOOL_CALL_RESULT, "c1", output="x" * 5000)
    events.listen(seen.append)
    long = {"text": "y" * 3000, "token": "abc"}
    events.tool(RunEventType.TOOL_CALL_ARGS, "c1", args=long)
    events.tool(RunEventType.TOOL_CALL_RESULT, "c1", output=[{"n": i} for i in range(500)])
    args, output = seen[0].data["args"], seen[1].data["output"]
    assert args == long
    assert isinstance(output, str) and output.endswith("…") and len(output) == 2001
    assert events.sequence == 2  # nothing was built while nobody listened


async def chat(hooks: list[Hooks]) -> str:
    async with Harness(config=Settings()) as harness:
        app = FastAPI()
        arm(harness.wrap(notifying, id="chat", hooks=hooks), [notify]).serve_chat(
            app, identity=lambda request: "u1"
        )
        async with asgi(app) as http:
            response = await http.post(
                "/agui/run",
                json={"threadId": "t1", "messages": [{"id": "m", "role": "user", "content": "go"}]},
            )
    return response.text


async def test_an_agui_stream_is_what_the_on_output_hooks_leave() -> None:
    assert leaked(await chat([]))
    shown = await chat([Mask()])
    events = {e["type"]: e for _, e in decode(shown)}
    assert json.loads(events["TOOL_CALL_ARGS"]["delta"]) == {"email": "[email]", "api_key": "[key]"}
    assert json.loads(events["TOOL_CALL_RESULT"]["content"])["receipt"] == "[key]"
    assert not leaked(shown) and received == [ARGS, ARGS]


async def test_a_raising_on_output_hook_withholds_what_it_was_given(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with Harness(config=Settings()) as harness:
        agent = arm(harness.wrap(notifying, id="boom", hooks=[Boom()]), [notify])
        with caplog.at_level(logging.ERROR, logger="trellis.hooks"):
            events = [e async for e in agent.stream("go", user="u")]
            result = await agent.run("go", user="u")
    assert result.answer == WITHHELD and not leaked(dumped(events))
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.data["result"] == WITHHELD
    assert any("on_output of Boom failed" in r.getMessage() for r in caplog.records)
    shown = await chat([Boom()])
    assert not leaked(shown) and "RUN_FINISHED" in shown


def test_the_harness_hooks_show_first_then_the_agents() -> None:
    order: list[str] = []

    class Named(Hooks):
        def __init__(self, name: str) -> None:
            self.name = name

        def on_output(self, item: Output) -> Output:
            order.append(self.name)
            return item.replace(f"{item.value}+{self.name}")

    chain = Chain([Named("harness"), Hooks(), Named("agent")])
    assert chain.shown(Output(OutputKind.ANSWER, "a")).value == "a+harness+agent"
    assert order == ["harness", "agent"] and len(chain.shows) == 2


@pytest.mark.parametrize("framework", ["react", "langgraph", "openai_agents", "claude"])
async def test_every_framework_shows_its_answer_through_the_hooks(
    framework: str, tmp_path: Path
) -> None:
    said = f"write to {EMAIL}"
    target = {
        "react": lambda: ReAct(system_prompt="s", model=ScriptedChat([said])),
        "langgraph": lambda: create_agent(ScriptedChat([said]), tools=[]),
        "openai_agents": lambda: OpenAIAgent(name="o", model=ScriptedModel([said])),
        "claude": lambda: options(tmp_path, [{"text": said}]),
    }[framework]()
    async with Harness(config=Settings()) as harness:
        agent = harness.wrap(target, id=f"show-{framework}", hooks=[Mask()])
        result = await agent.run("hi", user="u")
    assert result.answer == "write to [email]"


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


async def test_a2a_task_updates_and_pushes_carry_values_as_they_are(
    a2a_pushes: list[bytes],
) -> None:
    async with Harness(config=Settings()) as harness:
        app = FastAPI()
        arm(harness.wrap(notifying, id="greeter", hooks=[Mask()]), [notify]).serve_a2a(app, URL)
        async with asgi(app) as http:
            client = await connect(http)
            hook = TaskPushNotificationConfig(url="https://hooks.example.com/a2a", token="tok")
            request = SendMessageRequest(
                message=new_message([new_text_part("go")], role=Role.ROLE_USER),
                configuration=SendMessageConfiguration(task_push_notification_config=hook),
            )
            responses = [r async for r in client.send_message(request, context=caller())]
    updates = json.dumps([MessageToDict(r) for r in responses])
    assert SECRET in updates and EMAIL in updates and received == [ARGS]
    assert a2a_pushes and any(SECRET in body.decode() for body in a2a_pushes)


async def test_memory_gets_tool_records_and_the_transcript_as_they_are(
    memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    spooled: list[dict[str, Any] | None] = []
    async with Harness(config=Settings(), memory=memory_service.client()) as memory_harness:
        submit = memory_harness.writes.submit

        async def keep(label: str, work: Any, **kw: Any) -> None:
            if label == "memory.record_tool":
                spooled.append(kw.get("record"))
            await submit(label, work, **kw)

        monkeypatch.setattr(memory_harness.writes, "submit", keep)
        agent = arm(memory_harness.wrap(notifying, id="notifier", hooks=[Mask()]), [notify])
        assert (await agent.run("go", user="u")).answer == "sent to [email]"
        await memory_harness.writes.drain()
    [call] = [c for c in memory_service.named("record_tool") if c.body["tool"] == "notify"]
    assert call.body["args"] == ARGS and call.body["output"] == OUTPUT
    [record] = spooled  # what the spool keeps if the write cannot be delivered
    assert record is not None and leaked(json.dumps(record))
    transcript = json.dumps([c.body for c in memory_service.named("messages")])
    assert EMAIL in transcript and received == [ARGS]


def test_a_withheld_event_keeps_its_shape_and_none_of_its_values() -> None:
    identity = Identity(tenant="t", user="u", agent_id="a", run_id="r", thread="r")
    events, seen = RunEvents(identity.context()), []
    events.listen(seen.append)
    events.text(RunEventType.TEXT_MESSAGE_CONTENT, "m1", delta=f"mail {EMAIL}")
    events.tool(RunEventType.TOOL_CALL_START, "c1", tool="notify", args=ARGS)
    item = Output(OutputKind.EVENT, seen[0])
    assert Hooks().on_output(item) is item  # the default sends it as it is
    text, call = (Output(OutputKind.EVENT, e).withheld().value for e in seen)
    assert text.data == {"withheld": WITHHELD, "delta": WITHHELD}
    assert call.data == {"withheld": WITHHELD, "tool": "notify"}
