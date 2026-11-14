"""``GET {path}/threads/{thread_id}/events``: every run of a chat's thread on one stream, the
runs its schedules fire included, live and on reconnect (``Last-Event-ID``). One poller per
process serves every open thread; a scheduled run with nothing to report is not delivered."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.types import Message

from trellis import Harness, Runtime, Settings
from trellis.harness.agui import threads as threads_module
from trellis.harness.agui.sse import decode
from trellis.harness.agui.threads import Threads
from trellis.harness.harness import LOCAL_TENANT
from trellis.harness.schedules import NO_REPORT

THREAD = "t1"
ROUTE = "/agui/threads/{thread_id}/events"


class Desk:
    """Schedules what it is told in chat; a fired run answers what ``fired`` says next."""

    def __init__(self) -> None:
        self.fired: deque[str] = deque()

    async def __call__(self, input: str, agent: Runtime) -> str:
        if input.startswith("Scheduled task:"):
            return self.fired.popleft()
        args = {"task": "check stock", "when": "0 8 * * 1", "condition": "under 5"}
        await agent.tools.call("schedule_task", **args)
        return "scheduled"


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


class Chat:
    def __init__(self) -> None:
        self.harness = Harness(config=Settings())
        self.desk = Desk()
        self.app = FastAPI()
        self.agent = self.harness.wrap(self.desk, id="desk")
        self.agent.serve_chat(self.app, identity=user_of)
        self.threads = Threads(self.agent)

    async def schedule(self, user: str = "u1", thread: str = THREAD) -> str:
        await self.agent.run("watch stock", user=user, thread=thread)
        schedules = self.harness.runs.schedules
        page = await schedules.list(on_behalf_of=user)
        return page.items[0].schedule_id

    async def fire(self, schedule_id: str, answer: str) -> str:
        self.desk.fired.append(answer)
        fired = await self.harness.runs.schedules.fire(schedule_id)
        assert await self.harness.worker([self.agent]).run_once() is True
        return fired.run_id

    def stream(self, user: str = "u1", last: str | None = None) -> AsyncGenerator[str]:
        return self.threads.stream(LOCAL_TENANT, user, THREAD, last)


async def events(stream: AsyncIterator[str], count: int) -> list[tuple[str, dict[str, Any]]]:
    """The next ``count`` events of an open stream (SSE ``id`` kept as text)."""
    out: list[tuple[str, dict[str, Any]]] = []
    async with asyncio.timeout(5):
        async for chunk in stream:
            ident = chunk.split("\n", 1)[0].removeprefix("id: ")
            [(_, event)] = decode(chunk.replace(f"id: {ident}", "id: 0"))
            out.append((ident, event))
            if len(out) == count:
                return out
    return out


def texts(found: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [e["delta"] for _, e in found if e["type"] == "TEXT_MESSAGE_CONTENT"]


@pytest.fixture
async def chat(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Chat]:
    monkeypatch.setattr(threads_module, "POLL_SECONDS", 0.01)
    made = Chat()
    yield made
    await made.harness.aclose()


async def test_a_scheduled_reply_arrives_on_the_open_stream_and_no_report_does_not(
    chat: Chat,
) -> None:
    sid = await chat.schedule()
    stream = chat.stream()
    # the chat's own turn, replayed on connect: its reply ends in RUN_FINISHED
    replayed = await events(stream, 4)
    assert texts(replayed) == ["scheduled"] and replayed[-1][1]["type"] == "RUN_FINISHED"
    quiet = await chat.fire(sid, "NO_REPORT")
    told = await chat.fire(sid, "Only 3 left.")
    live = await events(stream, 4)
    assert texts(live) == ["Only 3 left."]
    assert {e.get("runId") for _, e in live} == {told} and quiet != told
    assert live[-1][1]["type"] == "RUN_FINISHED" and live[-1][1]["result"] == "Only 3 left."
    assert live[0][0].startswith(f"{told}:")
    await stream.aclose()


async def test_a_reconnect_replays_what_came_after_the_last_event_it_saw(chat: Chat) -> None:
    sid = await chat.schedule()
    first = await events(chat.stream(), 4)
    told = await chat.fire(sid, "Only 2 left.")
    again = await events(chat.stream(last=first[-1][0]), 4)
    assert texts(again) == ["Only 2 left."] and {e.get("runId") for _, e in again} == {told}
    by_query = await events(chat.stream(last=f"{told}:0"), 3)
    assert texts(by_query) == ["Only 2 left."]  # the events of that run after its first


async def test_another_users_thread_named_alike_hears_nothing_of_it(chat: Chat) -> None:
    sid = await chat.schedule(user="u1")
    await chat.schedule(user="u2")
    theirs = chat.stream(user="u2")
    assert texts(await events(theirs, 4)) == ["scheduled"]  # u2's own turn only
    await chat.fire(sid, "Only 1 left.")
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await events(theirs, 1)


async def test_one_poller_and_one_listing_per_tick_serve_every_open_thread(
    chat: Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = await chat.schedule()
    listings: list[dict[str, Any]] = []
    iterate = chat.harness.runs.iterate

    def counted(**kwargs: Any) -> Any:
        listings.append(kwargs)
        return iterate(**kwargs)

    monkeypatch.setattr(chat.harness.runs, "iterate", counted)
    streams = [chat.stream() for _ in range(3)]
    for stream in streams:
        await events(stream, 4)  # each replayed the thread
    replays = len(listings)
    connects = [k for k in listings if k.get("thread_id") == THREAD]
    assert len(connects) == 3 and all(k["limit"] == 20 for k in connects)  # its latest 20 runs
    await chat.fire(sid, "Only 4 left.")
    for stream in streams:
        assert texts(await events(stream, 4)) == ["Only 4 left."]
    threads = chat.threads
    assert threads.polling is not None and not threads.polling.done()
    ticks = [k for k in listings[replays:] if "thread_id" not in k and "status" not in k]
    assert ticks and all(k.get("max_pages") for k in ticks)  # one tenant listing per tick
    # no listing per stream (the run's own turn order reads its thread, as any run does)
    assert not [k for k in listings[replays:] if k.get("limit") == 20]
    for stream in streams:
        await stream.aclose()
    await asyncio.sleep(0.05)
    assert threads.polling is None  # nobody listening: no polling


async def test_the_poller_looks_once_a_second(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    sleep = asyncio.sleep

    async def quick(seconds: float) -> None:  # each wait the poller asks for, said and shortened
        waits.append(seconds)
        await sleep(0)

    chat = Chat()
    sid = await chat.schedule()
    stream = chat.stream()
    await events(stream, 4)
    monkeypatch.setattr(threads_module.asyncio, "sleep", quick)
    await chat.fire(sid, "Only 2 left.")
    assert texts(await events(stream, 4)) == ["Only 2 left."]
    monkeypatch.undo()
    await stream.aclose()
    await chat.harness.aclose()
    assert 1.0 in waits  # one look per second


async def test_the_route_needs_a_caller_and_is_in_the_openapi_document() -> None:
    chat = Chat()
    transport = httpx.ASGITransport(app=chat.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://agui") as http:
        refused = await http.get(f"/agui/threads/{THREAD}/events")
    assert refused.status_code == 401
    route = chat.app.openapi()["paths"][ROUTE]["get"]
    assert list(route["responses"]["200"]["content"]) == ["text/event-stream"]
    await chat.harness.aclose()


async def test_a_run_followed_live_is_held_until_it_ends(chat: Chat) -> None:
    sid = await chat.schedule()
    stream = chat.stream()
    await events(stream, 4)
    reading = asyncio.create_task(events(stream, 4))  # the stream follows each run as it runs
    for answer in ("NO_REPORT", "Only 5 left."):  # the first, quiet, is left out
        chat.desk.fired.append(answer)
        fired = await chat.harness.runs.schedules.fire(sid)
        async with asyncio.timeout(5):
            while fired.run_id not in chat.threads._seen:  # noqa: ASYNC110 - the poller's
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert await chat.harness.worker([chat.agent]).run_once() is True
    said = await reading
    assert texts(said) == ["Only 5 left."] and {e.get("runId") for _, e in said} == {fired.run_id}
    await stream.aclose()


async def test_the_poller_survives_a_failed_look(
    chat: Chat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = await chat.schedule()
    looked = chat.threads._look
    failures = [RuntimeError("store down")]

    async def flaky(tenant: str) -> None:
        if failures:
            raise failures.pop()
        await looked(tenant)

    monkeypatch.setattr(chat.threads, "_look", flaky)
    stream = chat.stream()
    await events(stream, 4)
    # a run in another thread is no one's here
    await chat.agent.run("elsewhere", user="u1", thread="t9")
    told = await chat.fire(sid, "Only 7 left.")
    assert {e.get("runId") for _, e in await events(stream, 4)} == {told}
    assert "no look at" in caplog.text
    await stream.aclose()
    await Threads(chat.agent)._look(LOCAL_TENANT)  # nothing open: every run passed over


async def test_from_agent_runs_log_a_scheduled_runs_events_are_held_and_a_quiet_one_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trellis.contracts import RunEventType, RunRecord, RunStatus
    from trellis.harness.schedules import TASK

    async with Harness(config=Settings()) as h:

        async def echo(input: str, agent: Runtime) -> str:
            return input

        agent = h.wrap(echo, id="echo")
        logged = {
            text: [e async for e in agent.stream(text, user="u1")] for text in ("hi", NO_REPORT)
        }
        folded = logged["hi"][0].model_copy(update={"type": RunEventType.INTERRUPT})
        logged["hi"].insert(1, folded)  # the log's own interrupt: RUN_FINISHED carries it

        class Log:
            async def stream_events(self, run_id: str, *, after: int, tenant: str) -> Any:
                for n, event in enumerate(logged[run_id]):
                    yield type("Entry", (), {"position": n, "event": event})()

        monkeypatch.setattr(threads_module, "event_log", lambda store: Log())
        threads = Threads(agent)

        def record(text: str, *, fired: bool) -> RunRecord:
            meta = {TASK: {"task": "t"}, "schedule_id": "sch_1"} if fired else {}
            return RunRecord(
                run_id=text, tenant_id="t", agent_id="echo", status=RunStatus.SUCCESS, metadata=meta
            )

        async def chunks(text: str, *, fired: bool) -> list[str]:
            return [c async for c in threads._run(record(text, fired=fired), -1)]

        shown = await chunks("hi", fired=True)
        assert shown and shown[-1].startswith("id: hi:")
        assert await chunks(NO_REPORT, fired=True) == []
        assert len(await chunks(NO_REPORT, fired=False)) == len(shown)
        # in process, a run followed as it runs: its text is said from its record at the end
        said = logged["hi"][0].model_copy(update={"type": RunEventType.TEXT_MESSAGE_CONTENT})

        async def live(run_id: str, *, tenant: str) -> Any:
            for event in [said, *logged["hi"]]:
                yield event

        monkeypatch.setattr(threads_module, "event_log", lambda store: None)
        monkeypatch.setattr(agent, "events", live)
        running = record("hi", fired=False).model_copy(update={"status": RunStatus.RUNNING})
        followed = [c async for c in threads._run(running, -1)]
        assert '"TEXT_MESSAGE_CONTENT"' not in "".join(followed)


def test_a_reply_from_a_record_and_the_last_event_id() -> None:
    from trellis.contracts import AgentError, ErrorCategory, RunRecord, RunStatus
    from trellis.harness.agui.events import AGUIEvent, AGUIEventType

    def reply(status: RunStatus, output: Any = None, **more: Any) -> list[str]:
        made = RunRecord(
            run_id="r", tenant_id="t", agent_id="a", status=status, output=output, **more
        )
        return [e.type.value for e in threads_module._reply(made, None)]

    assert reply(RunStatus.SUCCESS, {"n": 1})[1:] == [
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert reply(RunStatus.SUCCESS, NO_REPORT) == ["RUN_FINISHED"]
    assert reply(RunStatus.CANCELLED) == ["RUN_FINISHED"]
    failed = AgentError(code="X", category=ErrorCategory.DEPENDENCY, message="broke")
    assert reply(RunStatus.ERROR, error=failed) == ["RUN_ERROR"]
    assert reply(RunStatus.TIMEOUT) == ["RUN_ERROR"]
    assert threads_module._ends(AGUIEvent(type=AGUIEventType.RUN_ERROR))
    for last, found in [
        (None, (None, -1)),
        ("r", (None, -1)),
        ("r:x", (None, -1)),
        ("a:b:3", ("a:b", 3)),
    ]:
        assert threads_module._position(last) == found


async def test_the_route_streams_the_callers_thread_until_the_client_goes() -> None:
    chat = Chat()
    await chat.schedule()
    got: list[Any] = []
    gone = asyncio.Event()

    async def receive() -> Message:
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            got.append(message["status"])
        elif message.get("body"):
            got.append(message["body"].decode())
            gone.set()

    path = f"/agui/threads/{THREAD}/events"
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"x-user", b"u1")],
        "server": ("agui", 80),
        "client": ("test", 1),
    }
    async with asyncio.timeout(5):
        await chat.app(scope, receive, send)
    assert got[0] == 200 and '"TEXT_MESSAGE_START"' in got[1]
    await chat.harness.aclose()
