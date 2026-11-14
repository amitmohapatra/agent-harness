"""Faults at the boundaries (REL): a full disk under the file store and under the run store, a
server whose clock runs ahead, a network cut between a server and agent-runs, a webhook delivered
twice, and a provider that answers 429 or "context too long". Each fault is switched by the test
(a flag file the server process watches, ``tests/scenarios/faults.py``; a 64 KiB tmpfs for the
file store), never timed: the run ends with a status a person can read, nothing is run twice and
nothing is left ``RUNNING``."""

from __future__ import annotations

import asyncio
import http.server
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.scenarios.brains import command, parse, token, wrapped
from tests.scenarios.conftest import harness
from tests.scenarios.kit import TOOLS, Chat, Served, calling, eventually, free_port, record
from tests.support.invariants import Invariants
from trellis import Harness, Runtime, current, tool
from trellis.contracts import RunStatus
from trellis.runs import RunsClient, WebhookEvent, parse_delivery, verify_signature
from trellis.runs.webhooks import SIGNATURE_HEADER

pytestmark = [pytest.mark.timeout(600), pytest.mark.plan("REL")]


@pytest.fixture
def full_disk(tmp_path: Path) -> Iterator[Path]:
    """A folder on a file system of 64 KiB: writing more than that is ``ENOSPC``, for real."""
    folder = tmp_path / "small-disk"
    folder.mkdir()
    made = subprocess.run(
        ["mount", "-t", "tmpfs", "-o", "size=64k", "tmpfs", str(folder)],
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip(f"cannot mount a small file system here: {made.stderr.decode()[:120]}")
    try:
        yield folder
    finally:
        subprocess.run(["umount", str(folder)], capture_output=True, check=False)


@tool(side_effects="read")
async def big_report(kib: int) -> str:
    """Make a report of that many KiB for the user."""
    run = current()
    assert run is not None
    await run.save_file("report.bin", b"x" * (kib * 1024), source="big_report")
    return f"made a report of {kib} KiB"


async def test_a_full_disk_under_the_file_store(full_disk: Path, uid: str, tmp_path: Path) -> None:
    """The file does not fit: the call fails with why (the model reads it), the run ends, and a
    file that fits is kept afterwards — nothing half-written stands in its way."""
    async with harness(files_url=f"file://{full_disk}") as h:
        agent, _ = await wrapped(h, "react", [*TOOLS, big_report], tmp_path, id=f"s18f-{uid}")
        user, thread = f"ada-{uid}", f"t-{uid}"
        big = await agent.run(
            f"{token(uid, 1)} {command('big_report', kib=512)}", user=user, thread=thread
        )
        assert big.status in (RunStatus.SUCCESS, RunStatus.ERROR), big
        said = big.answer or (big.error.message if big.error else "")
        assert "space" in said.lower() or "big_report failed" in said, said  # told why
        small = await agent.run(
            f"{token(uid, 2)} {command('big_report', kib=8)}", user=user, thread=thread
        )
        assert small.status is RunStatus.SUCCESS and "made a report of 8 KiB" in small.answer, small


def _served(adapter: str, agent_id: str, **faults: Any) -> dict[str, Any]:
    agents = [{"adapter": adapter, "id": agent_id}]
    return {"mode": "agui", "hold": 6, "port": free_port(), "agents": agents, "faults": faults}


async def test_a_full_disk_under_the_run_store(
    uid: str, tmp_path: Path, invariants: Invariants
) -> None:
    """agent-runs refuses the run's data (events, payloads: 507) while its disk is full: the turn
    still ends for its user, and once there is room the chat goes on."""
    invariants.allow("INV-7", "a 507 injected for agent-runs' writes: its document lists none")
    invariants.allow("INV-10", "the events agent-runs refused are not in its log")
    invariants.allow("INV-2", "the events agent-runs refused are not in its log")
    flag = tmp_path / "runs-full"
    agent_id = f"s18r-{uid}"
    served = await Served(
        _served("react", agent_id, runs_full=str(flag)), tmp_path, "one"
    ).started()
    chat = Chat(agent_id, f"ada-{uid}", f"t-{uid}")
    try:
        flag.write_text("full")
        during = await chat.say(served.port, f"{token(uid, 1)} {command('lookup', key='k')}")
        ending = during[-1][1]
        assert ending["type"] in ("RUN_FINISHED", "RUN_ERROR"), ending  # it ended, for its user
        flag.unlink()
        after = (await chat.say(served.port, f"{token(uid, 2)} and now?"))[-1][1]
        assert after.get("outcome") == {"type": "success"}, after
        assert parse(after["result"]).mine == token(uid, 2)
    finally:
        served.stop()


async def test_a_server_whose_clock_runs_ahead(uid: str, tmp_path: Path) -> None:
    """Server two's clock is ten minutes ahead: one chat's turns alternating between the servers
    are answered in order with the turns before them, and a turn longer than its hold on the
    skewed server is held to its end (a hold is a length of time, agent-runs' clock its judge)."""
    agent_id = f"s18c-{uid}"
    one = await Served(_served("react", agent_id), tmp_path, "one").started()
    two = await Served(_served("react", agent_id, skew=600), tmp_path, "two").started()
    chat = Chat(agent_id, f"ada-{uid}", f"t-{uid}")
    try:
        seen: list[str] = []
        for n, port in enumerate((one.port, two.port, one.port, two.port), start=1):
            slow = (
                f" {command('wait', seconds=15)}" if n == 4 else ""
            )  # 2.5 holds, on the skewed one
            done = (await chat.say(port, f"{token(uid, n)} turn {n}{slow}"))[-1][1]
            assert done.get("outcome") == {"type": "success"}, (n, done)
            said = parse(done["result"])
            assert said.mine == token(uid, n) and set(seen) <= said.seen, (n, said)
            seen.append(token(uid, n))
    finally:
        one.stop()
        two.stop()


@pytest.mark.xfail(
    strict=True,
    reason="SCN-13: scn-fixes (a run agent-runs ended holder_lost goes on where it works: "
    "harness.py Harness._lost stops only a CANCELLED one)",
)
async def test_a_network_cut_between_a_server_and_agent_runs(
    uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invariants: Invariants
) -> None:
    """The server working a turn loses agent-runs: its hold lapses and agent-runs ends the run
    (``holder_lost``). When the network is back the server does not carry the turn on (the write
    after its slow step never runs), the run stays ended, and the chat takes the next message."""
    invariants.allow("INV-2", "SCN-13: what the server did after its run was ended is in no log")
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    monkeypatch.setenv("TRELLIS_TEST_WRITES_UNASKED", "1")
    flag = tmp_path / "cut"
    agent_id = f"s18n-{uid}"
    served = await Served(
        _served("react", agent_id, partition=str(flag)), tmp_path, "one"
    ).started()
    chat = Chat(agent_id, f"ada-{uid}", f"t-{uid}")
    run_id = f"run_{uid}cut"
    try:
        steps = f"{command('wait', seconds=45)} {command('note', text=f'after {uid}')}"
        first = asyncio.create_task(chat.say(served.port, f"{token(uid, 1)} {steps}", runId=run_id))
        async with harness() as h:
            await calling(h, run_id)  # in its slow step
            flag.write_text("cut")

            async def lost() -> bool:
                return (await record(h, run_id)).status is RunStatus.ERROR

            assert await eventually(lost, within=60), await record(h, run_id)
            flag.unlink()
            await first  # the server's own attempt is over: told its run was ended
            again = (await chat.say(served.port, f"{token(uid, 2)} still there?"))[-1][1]
            assert again.get("outcome") == {"type": "success"}, again
            ended = await record(h, run_id)
            assert ended.status is RunStatus.ERROR and ended.error is not None, ended
            assert ended.error.code == "holder_lost", ended.error
    finally:
        served.stop()
    lines = book.read_text().splitlines() if book.exists() else []
    assert f"note:after {uid}" not in lines, ("the turn went on after its run was ended", lines)


class _Receiver(http.server.BaseHTTPRequestHandler):
    """A webhook receiver of the application's own that loses its first answer to each delivery
    (a 500 after the body was read: agent-runs cannot know it arrived, and sends it again)."""

    got: ClassVar[list[tuple[str | None, bytes]]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        first = all(seen != body for _, seen in self.got)
        self.got.append((self.headers.get(SIGNATURE_HEADER), body))
        self.send_response(500 if first else 204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


async def test_a_webhook_delivered_twice_is_one_event(h: Harness, uid: str) -> None:
    """A delivery whose answer was lost is sent again: the same ``event_id``, signed again, so an
    application that keeps the ids it handled handles the run's end once."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert isinstance(h.runs, RunsClient)
    url = f"http://127.0.0.1:{server.server_address[1]}/hook"
    hook = await h.runs.webhooks.create(url, [WebhookEvent.FINISHED])
    try:

        async def echo(input: Any, agent: Runtime) -> str:
            return str(input)

        done = await h.wrap(echo, id=f"s18w-{uid}").run("hello", user=f"ada-{uid}")

        def mine() -> list[Any]:
            signed = [
                parse_delivery(body)
                for signature, body in list(_Receiver.got)
                if verify_signature(hook.secret, signature, body)
            ]
            return [d for d in signed if d.data.run.run_id == done.run_id]

        async def twice() -> bool:
            return len(mine()) >= 2

        assert await eventually(twice, within=180), f"delivered {len(mine())} time(s), not again"
        assert len({d.event_id for d in mine()}) == 1, mine()  # two deliveries, one event
    finally:
        await h.runs.webhooks.delete(hook.webhook_id)
        server.shutdown()


@pytest.mark.parametrize("adapter", ["react", "react_da"])
@pytest.mark.parametrize(
    ("fault", "says"),
    [("429", ("rate", "limit")), ("overflow", ("context", "length"))],
)
async def test_a_provider_that_refuses_the_request(  # noqa: PLR0917 - pytest's fixtures
    h: Harness, adapter: str, fault: str, says: tuple[str, str], uid: str, tmp_path: Path
) -> None:
    """The provider answers 429, or that the request is longer than the model's context: the run
    ends ``ERROR`` with an error that says which (a person, or a calling agent, reads it), and
    the chat takes the next message."""
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s18m-{adapter}-{fault}-{uid}")
    user, thread = f"ada-{uid}", f"t-{uid}"
    done = await agent.run(f"{token(uid, 1)} fail:{fault}", user=user, thread=thread)
    assert done.status is RunStatus.ERROR and done.error is not None, (done.status, done.answer)
    told = f"{done.error.code} {done.error.category} {done.error.message}".lower()
    assert all(word in told for word in says), done.error
    assert done.error.retryable is (fault == "429"), done.error
    after = await agent.run(f"{token(uid, 2)} and now?", user=user, thread=thread)
    assert after.status is RunStatus.SUCCESS, after.error
