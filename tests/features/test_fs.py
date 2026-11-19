"""FS against the real services: the files a run keeps for its user.

* a file a run saves (``Runtime.save_file``) is in the file store under the run's tenant, user
  and conversation, announced on its stream, and served by AG-UI to that user alone — from
  another replica that shares the store (a 404 for anyone else, and for a run that kept no
  such file);
* ``query_result(save_as="csv")`` over a large result keeps the rows it selected as a file, and
  a later run of the same user, in another process, is told of it and reads it back;
* a run's sandbox on a real Docker daemon: a file written and a command run in a container of
  the run's own, labelled with it while it works and deleted at its end (in process:
  tests/integration/test_sandbox*.py and the matrix F73).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.features.conftest import harness
from tests.features.kit import armed, big_report
from tests.features.test_surf import Served, identity
from trellis import Runtime, sandbox
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.journal import content_key
from trellis.harness.sandbox import DockerSandbox
from trellis.harness.sandbox.base import RUN
from trellis.harness.tools.convert import text_of

pytestmark = pytest.mark.timeout(300)


@pytest.mark.plan("FS", "X13", "SURF")
async def test_a_saved_file_is_served_to_its_user_alone_from_another_replica(
    tmp_path: Path, uid: str
) -> None:
    store = tmp_path / "files"
    user, agent_id = f"kit-{uid}", f"saver-{uid}"
    data = b"sku,qty\nA-1,3\nB-2,7\n"

    async def export(input: Any, agent: Runtime) -> str:
        kept = await agent.save_file("stock.csv", data, source="export", content_type="text/csv")
        return f"saved {kept.name}"

    async with harness(files_url=str(store)) as h1, harness(files_url=str(store)) as h2:
        apps = []
        for h in (h1, h2):
            app = FastAPI()
            h.wrap(export, id=agent_id).serve_chat(app)
            apps.append(app)
        async with (
            Served(apps[0]) as one,
            Served(apps[1]) as two,
            httpx.AsyncClient(timeout=60, follow_redirects=False) as c,
        ):
            run_id = f"run_{uid}files"
            started = await c.post(
                f"{one.url}/agui/run",
                headers=identity(user),
                json={
                    "threadId": "t",
                    "runId": run_id,
                    "messages": [{"id": "m1", "role": "user", "content": "export"}],
                },
            )
            assert started.status_code == 200, started.text
            assert "stock.csv" in started.text  # the file event on the stream
            got = await c.get(
                f"{two.url}/agui/runs/{run_id}/files/stock.csv", headers=identity(user)
            )
            assert got.status_code == 200 and got.content == data, got.status_code
            other = await c.get(
                f"{two.url}/agui/runs/{run_id}/files/stock.csv", headers=identity(f"lev-{uid}")
            )
            assert other.status_code == 404  # only the run's own user
            missing = await c.get(
                f"{two.url}/agui/runs/{run_id}/files/other.csv", headers=identity(user)
            )
            assert missing.status_code == 404
    kept = list(store.rglob("stock.csv"))
    assert len(kept) == 1 and user in str(kept[0]) and "feat" in str(kept[0]), kept
    assert kept[0].read_bytes() == data


@pytest.mark.plan("FS", "HAR-LARGE", "F26q")
async def test_query_result_save_as_keeps_the_rows_and_a_later_run_reads_them_back(
    tmp_path: Path, uid: str
) -> None:
    store, user, agent_id = tmp_path / "files", f"mo-{uid}", f"saveas-{uid}"
    rows = 20_000
    text = "\n".join(f"row {i:06d} sku-{i % 97:03d} qty {i % 13}" for i in range(rows))
    path = f"/large_tool_results/{content_key('result', text)}"
    sql = "SELECT line FROM result WHERE line LIKE '%sku-007%' ORDER BY line"
    plan = [
        ("big_report", {"rows": rows}),
        ("query_result", {"file_path": path, "sql": sql, "save_as": "csv"}),
    ]
    async with harness(files_url=str(store)) as h:
        agent, _ = await armed("react", h, tmp_path, plan, id=agent_id, tools=[big_report])
        events = [e async for e in agent.stream("Export sku-007.", user=user, thread="t")]
    assert events[-1].type is RunEventType.RUN_FINISHED
    files = [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == "file"
    ]
    assert files, [e.data for e in events if e.type is RunEventType.CUSTOM]
    saved = list(store.rglob("*.csv"))
    assert len(saved) == 1, saved
    body = saved[0].read_text().splitlines()
    expected = sorted(line for line in text.splitlines() if "sku-007" in line)
    assert body[1:] == expected and len(expected) == rows // 97 + (1 if rows % 97 > 7 else 0)
    name = saved[0].name
    path = f"/files/{events[0].run_id}/{name}"  # how a later run of the chat names it
    # a later run, another process: told of the kept file, reads it back
    async with harness(files_url=str(store)) as h:
        plan2 = [("query_result", {"file_path": path, "sql": "SELECT count(*) AS n FROM result"})]
        agent, built = await armed("react", h, tmp_path, plan2, id=agent_id, tools=[big_report])
        # memory off for this turn: a planned model counts the chat's earlier tool results
        # (memory's window replays them) as its own steps; the run store's history is the
        # questions and answers alone
        done = await agent.run(
            "How many rows did I export?", user=user, thread="t", without=["memory"]
        )
    assert done.status is RunStatus.SUCCESS, done
    assert path in built.said(), built.said()[:2000]  # told of it
    assert f"n\n{len(expected)}" in str(done.answer), done.answer
    # another user is never told of it
    async with harness(files_url=str(store)) as h:
        agent, built = await armed("react", h, tmp_path, [], id=agent_id, tools=[big_report])
        await agent.run("What did I export?", user=f"ned-{uid}", thread="t")
    assert name not in built.said()


@pytest.mark.plan("FS", "F73")
async def test_a_runs_commands_run_in_a_container_of_its_own_deleted_at_its_end(uid: str) -> None:
    """A run's sandbox on a real Docker daemon: the file it writes is there for the command
    that follows, the command's output is the tool's result, the container is labelled with
    the run while it works and gone once the run ended."""
    provider = DockerSandbox()
    box = sandbox(provider)
    while_running: list[list[str]] = []

    async def analyse(input: Any, agent: Runtime) -> str:
        await agent.tools.use(box.tools())
        await agent.tools.call("sandbox_write", path="units.csv", content="sku,qty\nA-1,3\nB-2,7\n")
        ran = await agent.tools.call(
            "sandbox_exec",
            command="python3 -c \"print(sum(int(l.split(',')[1]) for l in list(open('units.csv'))[1:]))\"",
        )
        mine = await provider.labelled({RUN: agent.run_id})
        while_running.append([ref.id for ref in mine])
        return text_of(ran)

    async with harness() as h:
        agent = h.wrap(analyse, id=f"analyst-{uid}")
        result = await agent.run("How many units?", user=f"sam-{uid}", thread="t")
    assert result.status is RunStatus.SUCCESS, result
    assert json.loads(result.answer)["stdout"].strip() == "10", result.answer
    assert [len(found) for found in while_running] == [1], while_running
    assert await provider.labelled({RUN: result.run_id}) == []
