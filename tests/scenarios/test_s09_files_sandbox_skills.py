"""Scenario 9 — files, sandbox and skills under concurrency (FS).

* Two users at once, on one thread id, each have a file made for them by a run (``save_file``):
  each downloads their own over AG-UI, and the other's is a 404.
* A document uploaded for a user is what that user's run draws on; another user attaching it is
  refused (404) and nothing runs.
* Each run has a sandbox of its own: two users writing and reading the same path at once read
  only their own (the provider is the in-process stand-in, ``tests/support/sandbox.py``: this
  machine has no Docker daemon).
* A skill (``skills.directory``) is read by runs of every engine that has a skills slot, at once.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.scenarios.brains import ADAPTERS, SYSTEM, BrainChat, command, parse, token, wrapped
from tests.scenarios.kit import TOOLS
from tests.support.arm import arm
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, current, sandbox, skills, tool
from trellis.contracts import RunStatus
from trellis.harness.agui.sse import decode

pytestmark = [pytest.mark.timeout(900), pytest.mark.plan("FS")]


@tool(side_effects="read")
async def make_report(name: str, body: str) -> str:
    """Make a report file for the user."""
    run = current()
    assert run is not None
    await run.save_file(name, body.encode(), source="make_report", content_type="text/plain")
    return f"made {name}"


def _app(agent: object) -> FastAPI:
    app = FastAPI()

    def user_of(request: Request) -> str:
        return request.headers["x-user"]

    agent.serve_chat(app, identity=user_of)  # type: ignore[attr-defined]
    return app


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_files_made_for_two_users_at_once(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    claude = [
        {"tool": "make_report", "args": {"name": "report.txt", "body": f"for-the-user {uid}"}}
    ]
    agent, _ = await wrapped(
        h, adapter, [*TOOLS, make_report], tmp_path, id=f"s9f-{adapter}-{uid}", claude=claude
    )
    users = {"a": f"alice-{uid}", "b": f"bob-{uid}"}
    thread = f"shared-{uid}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(agent)), base_url="http://agui", timeout=300
    ) as c:

        async def make(who: str) -> tuple[str, list]:
            make = command("make_report", name="report.txt", body=f"for-the-user {uid}")
            sent = await c.post(
                "/agui/run",
                headers={"x-user": users[who]},
                json={
                    "threadId": thread,
                    "messages": [
                        {
                            "id": f"m-{who}",
                            "role": "user",
                            "content": f"{token(who + uid[:4], 1)} {make}",
                        }
                    ],
                },
            )
            events = decode(sent.text)
            assert events[-1][1].get("outcome") == {"type": "success"}, events[-1]
            return events[0][1]["runId"], events

        (run_a, _), (run_b, _) = await asyncio.gather(make("a"), make("b"))
        own = await c.get(
            f"/agui/runs/{run_a}/files/report.txt",
            headers={"x-user": users["a"]},
            follow_redirects=True,
        )
        assert own.status_code == 200 and own.content == f"for-the-user {uid}".encode(), (
            own.status_code
        )
        theirs = await c.get(f"/agui/runs/{run_a}/files/report.txt", headers={"x-user": users["b"]})
        assert theirs.status_code == 404, theirs.status_code
        mine = await c.get(
            f"/agui/runs/{run_b}/files/report.txt",
            headers={"x-user": users["b"]},
            follow_redirects=True,
        )
        assert mine.status_code == 200


async def test_an_uploaded_document_is_its_users_only(h: Harness, uid: str, tmp_path: Path) -> None:
    doc_path = tmp_path / "policy.txt"
    doc_path.write_text(f"The refund window for {uid} is 47 days. {token('d' + uid[:4], 1)}\n")
    doc = await h.add_document(doc_path, user=f"alice-{uid}", wait=180)
    agent, built = await wrapped(h, "react", TOOLS, tmp_path, id=f"s9d-{uid}")
    done = await agent.run(
        f"{token(uid, 1)} what is the refund window?",
        user=f"alice-{uid}",
        thread=f"t-{uid}",
        document_ids=[doc.document_id],
    )
    assert done.status is RunStatus.SUCCESS, done.error
    assert "47 days" in built.model.said(), "the document did not reach the run"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(agent)), base_url="http://agui"
    ) as c:
        part = {"type": "document", "source": {"type": "file", "value": doc.document_id}}
        refused = await c.post(
            "/agui/run",
            headers={"x-user": f"bob-{uid}"},
            json={
                "threadId": f"t-{uid}",
                "messages": [
                    {
                        "id": "m1",
                        "role": "user",
                        "content": [{"type": "text", "text": f"{token(uid, 2)} read it"}, part],
                    }
                ],
            },
        )
        assert refused.status_code == 404, refused.text


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_sandbox_per_run_two_users_at_once(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    if adapter == "claude_agent_sdk":
        pytest.skip("the scripted Claude CLI's script is fixed per agent: it cannot write each user's own text")
    provider = FakeSandboxes()
    agent, _ = await wrapped(
        h, adapter, [*TOOLS, sandbox(provider).tools()], tmp_path, id=f"s9s-{adapter}-{uid}"
    )

    async def go(who: str) -> str:
        steps = " ".join(
            [
                command("sandbox_write", path="notes.txt", content=f"secret of {who}"),
                command("wait", seconds=1),
                command("sandbox_read", path="notes.txt"),
            ]
        )
        done = await agent.run(
            f"{token(who + uid[:4], 1)} {steps}", user=f"{who}-{uid}", thread=f"t-{uid}"
        )
        assert done.status is RunStatus.SUCCESS, (who, done.error)
        return parse(done.answer).last

    read_a, read_b = await asyncio.gather(go("alice"), go("bob"))
    assert "secret of alice" in read_a and "bob" not in read_a, read_a
    assert "secret of bob" in read_b and "alice" not in read_b, read_b
    made = [i for kind, i in provider.calls if kind == "create"]
    deleted = [i for kind, i in provider.calls if kind == "delete"]
    # one sandbox each, both deleted at their runs' end
    assert len(set(made)) == 2 and set(made) <= set(deleted), provider.calls


async def test_a_skill_read_by_runs_at_once(h: Harness, uid: str, tmp_path: Path) -> None:
    folder = tmp_path / "skills" / "refunds"
    folder.mkdir(parents=True)
    folder.joinpath("SKILL.md").write_text(
        f"---\nname: refunds\ndescription: How we refund.\n---\nAlways refund within 47 days ({uid}).\n"
    )
    src = skills.directory(str(tmp_path / "skills"))
    react = arm(h.wrap(ReAct(SYSTEM, BrainChat(), skills=src), id=f"s9k-react-{uid}"), TOOLS)
    read = command("read_file", file_path="/skills/refunds/SKILL.md")
    results = await asyncio.gather(
        *(
            react.run(
                f"{token(f'k{n}' + uid[:4], 1)} {read}", user=f"u{n}-{uid}", thread=f"t-{uid}"
            )
            for n in range(3)
        )
    )
    for done in results:
        assert done.status is RunStatus.SUCCESS, done.error
        assert "47 days" in parse(done.answer).last, done.answer
