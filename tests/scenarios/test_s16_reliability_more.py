"""More failure and recovery (REL, CONV-04, CONV-10, WF-07):

* a server killed while a turn waits for a person (``PAUSED``): the answer given on another
  server continues it, and the approved call runs once;
* a server killed while it continues a turn after the answer (resuming): the run ends
  ``holder_lost``, never stuck, and the call approved before the kill is not run again;
* ``interrupt`` sent to another server than the one working the turn (limitation L1, fixed by
  chat-order): the working turn is cancelled there;
* a library's own retries inside a workflow (LangGraph's ``RetryPolicy`` on a node whose read
  fails once): the read runs exactly twice, whoever retried it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import RetryPolicy

from tests.scenarios.brains import command, parse, token
from tests.scenarios.conftest import harness
from tests.scenarios.kit import TOOLS, Served, calling, eventually, free_port, record
from tests.support.arm import arm, native_tools
from trellis import Harness, tool
from trellis.contracts import RunStatus
from trellis.harness.agui.sse import decode

pytestmark = pytest.mark.timeout(900)
SERVED = ["react", "langgraph", "openai_agents", "claude_agent_sdk", "kimi", "function"]


class Chat:
    """One user's chat with one agent over AG-UI, on whichever server ``port`` names."""

    def __init__(self, agent_id: str, user: str, thread: str) -> None:
        self.agent_id, self.thread = agent_id, thread
        self.client = httpx.AsyncClient(headers={"x-user": user}, timeout=300)

    async def post(self, port: int, **body: Any) -> list[tuple[int, dict[str, Any]]]:
        sent = await self.client.post(
            f"http://127.0.0.1:{port}/agui/{self.agent_id}/run",
            json={"threadId": self.thread, **body},
        )
        assert sent.status_code == 200, sent.text
        return decode(sent.text)

    async def say(self, port: int, text: str, **more: Any) -> list[tuple[int, dict[str, Any]]]:
        message = {"id": text[:16], "role": "user", "content": text}
        return await self.post(port, messages=[message], **more)


def _spec(adapter: str, agent_id: str, claude: list[dict[str, Any]]) -> dict[str, Any]:
    agents = [{"adapter": adapter, "id": agent_id, "claude": claude}]
    return {"mode": "agui", "hold": 6, "agents": agents}


@pytest.mark.plan("REL", "CONV-10", "APR-05")
@pytest.mark.parametrize("adapter", SERVED)
async def test_a_server_killed_while_a_turn_waits_for_a_person(
    adapter: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    agent_id = f"s16p-{adapter}-{uid}"
    pay = {"amount": 8, "to": f"p-{uid}"}
    spec = _spec(adapter, agent_id, [{"tool": "transfer", "args": pay}])
    chat = Chat(agent_id, f"ada-{uid}", f"t-{uid}")
    one = await Served({**spec, "port": free_port()}, tmp_path, "one").started()
    events = await chat.say(one.port, f"{token(uid, 1)} {command('transfer', **pay)}")
    paused = events[-1][1]
    assert paused["outcome"]["type"] == "interrupt", paused
    one.kill()  # the server that asked is gone; the question is agent-runs'
    two = await Served({**spec, "port": free_port()}, tmp_path, "two").started()
    try:
        answer = {"interruptId": paused["outcome"]["interrupts"][0]["id"], "payload": True}
        done = (await chat.post(two.port, resume=[answer]))[-1][1]
    finally:
        two.stop()
    assert done.get("outcome") == {"type": "success"}, done
    assert parse(done["result"]).mine == token(uid, 1), done
    lines = book.read_text().splitlines() if book.exists() else []
    assert [x.split(":{")[0] for x in lines] == [f"transfer:8:p-{uid}"], lines  # once


@pytest.mark.plan("REL", "CONV-10", "LR-05")
@pytest.mark.parametrize("adapter", ["react", "langgraph", "openai_agents", "function"])
async def test_a_server_killed_while_it_continues_an_answered_turn(
    adapter: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    agent_id = f"s16r-{adapter}-{uid}"
    pay = {"amount": 8, "to": f"r-{uid}"}
    chat = Chat(agent_id, f"ada-{uid}", f"t-{uid}")
    one = await Served(
        {**_spec(adapter, agent_id, []), "port": free_port()}, tmp_path, "one"
    ).started()
    steps = f"{command('transfer', **pay)} {command('wait', seconds=120)}"
    events = await chat.say(one.port, f"{token(uid, 1)} {steps}")
    paused = events[-1][1]
    run_id = events[0][1]["runId"]
    answer = {"interruptId": paused["outcome"]["interrupts"][0]["id"], "payload": True}
    resuming = asyncio.create_task(chat.post(one.port, resume=[answer]))
    async with harness() as h:
        await calling(h, run_id)  # approved, the payment made, now in its slow call
        one.kill()
        resuming.cancel()

        async def ended() -> bool:
            return (await record(h, run_id)).status is not RunStatus.RUNNING

        assert await eventually(ended, within=90), await record(h, run_id)
        last = await record(h, run_id)
        assert last.status is RunStatus.ERROR and last.error is not None, last
        assert last.error.code == "holder_lost", last.error
    lines = book.read_text().splitlines() if book.exists() else []
    assert [x.split(":{")[0] for x in lines] == [f"transfer:8:r-{uid}"], lines  # once, not again


@pytest.mark.plan("CONV-04")
async def test_an_interrupt_sent_to_another_server(uid: str, tmp_path: Path) -> None:
    agent_id = f"s16i-{uid}"
    spec = _spec("react", agent_id, [])
    chat = Chat(agent_id, f"ada-{uid}", f"t-{uid}")
    one = await Served({**spec, "port": free_port()}, tmp_path, "one").started()
    two = await Served({**spec, "port": free_port()}, tmp_path, "two").started()
    try:
        first = asyncio.create_task(
            chat.say(
                one.port, f"{token(uid, 1)} {command('wait', seconds=20)}", runId=f"run_{uid}first"
            )
        )
        async with harness() as h:
            await calling(h, f"run_{uid}first")
            mode = {"forwardedProps": {"mode": "interrupt"}}
            second = (await chat.say(two.port, f"{token(uid, 2)} instead, this", **mode))[-1][1]
            assert second.get("outcome") == {"type": "success"}, second
            await first
            assert (await record(h, f"run_{uid}first")).status is RunStatus.CANCELLED
    finally:
        one.stop()
        two.stop()


FLAKY: dict[str, int] = {}


@tool(side_effects="read")
def flaky(key: str) -> str:
    """A read whose service fails the first time it is asked about a key."""
    FLAKY[key] = FLAKY.get(key, 0) + 1
    if FLAKY[key] == 1:
        raise ConnectionError("the service hung up")
    return f"value-of-{key}"


@pytest.mark.plan("WF-07")
@pytest.mark.xfail(
    strict=True,
    reason="SCN-11 (langchain-fixes): a failed harness tool answers its error as the result, so a "
    "node's RetryPolicy never sees a failure and the read is not tried again by either layer",
)
async def test_a_nodes_retry_policy_runs_a_failed_read_exactly_twice(h: Harness, uid: str) -> None:
    tools = {t.name: t for t in await native_tools(h, flaky, framework="langgraph")}

    async def fetch(state: MessagesState) -> dict[str, Any]:
        got = await tools["flaky"].ainvoke({"key": uid})
        return {"messages": [AIMessage(f"ANS NONE SEEN  MEM  LAST {got}")]}

    graph = StateGraph(MessagesState)
    graph.add_node(
        "fetch",
        fetch,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=Exception),
    )
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", END)
    agent = arm(h.wrap(graph.compile(), id=f"s16w-{uid}"), [flaky, *TOOLS])
    done = await agent.run("fetch it", user=f"ada-{uid}", thread=f"t-{uid}")
    assert done.status is RunStatus.SUCCESS and f"value-of-{uid}" in done.answer, (
        done.status,
        done.error,
    )
    assert FLAKY[uid] == 2, FLAKY[uid]  # failed once, tried once more: by one layer, not both
