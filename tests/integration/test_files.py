"""The files runs keep for their users (``trellis.harness.files``): one fsspec store where
``TRELLIS_FILES_URL`` says (else the XDG data folder), each file under its run's tenant, user,
conversation and run; announced on the run's stream and named in its journal; served over AG-UI
to the run's own user only — a short-lived signed link where the store signs them (a cloud's,
here fsspec's in-memory filesystem signing as GCS and S3 do), else streamed."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import fsspec
import httpx
import pytest
from fastapi import FastAPI, Request
from fsspec.implementations.memory import MemoryFileSystem

from trellis import Harness, Runtime, Settings
from trellis.contracts import ToolError
from trellis.harness.agui.sse import decode
from trellis.harness.files import LINK_SECONDS, FileStore, scope
from trellis.harness.journal import Journal
from trellis.harness.settings import default_files_url, sourced

PATH = "/agui"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class SigningMemory(MemoryFileSystem):
    """fsspec's in-memory filesystem, signing links as a cloud's does (``gcsfs``, ``s3fs``)."""

    protocol = ("signedmem",)
    signed: ClassVar[list[tuple[str, int]]] = []

    def sign(self, path: str, expiration: int = 100, **kwargs: Any) -> str:
        self.signed.append((path, expiration))
        return f"https://storage.example/{path.lstrip('/')}?expires={expiration}"


fsspec.register_implementation("signedmem", SigningMemory, clobber=True)


def kept_files(root: Path) -> list[Path]:
    """The ``sales.csv`` files kept below ``root`` (a test's look at the local store)."""
    return list(root.rglob("sales.csv"))


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


async def report(input: Any, agent: Runtime) -> Any:
    stored = await agent.save_file("sales.csv", b"region,total\nnorth,42\n", source="export_table")
    await agent.save_file("sales.csv", b"region,total\nnorth,43\n", source="export_table")
    return stored.model_dump()


@pytest.fixture
async def local(tmp_path: Path) -> AsyncIterator[tuple[Harness, httpx.AsyncClient]]:
    harness = Harness(config=Settings(files_url=tmp_path.as_uri()))
    app = FastAPI()
    harness.wrap(report, id="files").serve_chat(app, identity=user_of)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
    ) as client:
        yield harness, client
    await harness.aclose()


async def ran(client: httpx.AsyncClient, thread: str = "t1") -> tuple[str, list[dict[str, Any]]]:
    response = await client.post(
        f"{PATH}/run",
        json={"threadId": thread, "messages": [{"id": "m1", "role": "user", "content": "go"}]},
    )
    events = [e for _, e in decode(response.text)]
    return events[-1]["runId"], events


async def test_a_file_is_kept_in_its_runs_scope_announced_and_streamed_to_its_user(
    local: tuple[Harness, httpx.AsyncClient], tmp_path: Path
) -> None:
    harness, client = local
    run_id, events = await ran(client)
    announced = [e["value"] for e in events if e.get("name") == "file"]
    assert announced[-1] == {
        "file": "sales.csv",
        "size": 22,
        "content_type": "text/csv",
        "source": "export_table",
    }
    record = await harness.runs.get(run_id, tenant="default")
    assert record is not None and record.output["size"] == 22
    kept = kept_files(tmp_path)
    assert len(kept) == 1 and kept[0].read_bytes() == b"region,total\nnorth,43\n"  # replaced
    assert kept[0].relative_to(tmp_path).parts[:2] == ("default", "u1")
    assert kept[0].parent.name == run_id
    served = await client.get(f"{PATH}/runs/{run_id}/files/sales.csv")
    assert served.status_code == 200 and served.content == b"region,total\nnorth,43\n"
    assert served.headers["content-type"].startswith("text/csv")
    assert served.headers["content-disposition"] == 'attachment; filename="sales.csv"'
    for missing in ("other.csv", "..%2Fsecret", "a%20b.csv"):
        assert (await client.get(f"{PATH}/runs/{run_id}/files/{missing}")).status_code == 404


async def test_another_user_cannot_read_a_runs_file_and_alike_names_never_meet(
    local: tuple[Harness, httpx.AsyncClient], tmp_path: Path
) -> None:
    _, client = local
    run_id, _ = await ran(client)
    other = {"x-user": "u2"}
    assert (
        await client.get(f"{PATH}/runs/{run_id}/files/sales.csv", headers=other)
    ).status_code == 404
    client.headers["x-user"] = "u2"
    theirs, _ = await ran(client)  # the same thread name, the same file name
    paths = sorted(p.relative_to(tmp_path).parts[1] for p in kept_files(tmp_path))
    assert paths == ["u1", "u2"]
    assert (await client.get(f"{PATH}/runs/{theirs}/files/sales.csv")).status_code == 200
    assert (await client.get(f"{PATH}/runs/nope/files/sales.csv")).status_code == 404


async def test_a_store_that_signs_links_is_a_short_lived_redirect() -> None:
    bucket = f"signedmem://bucket-{uuid.uuid4().hex}/trellis"
    harness = Harness(config=Settings(files_url=bucket))
    app = FastAPI()
    harness.wrap(report, id="files").serve_chat(app, identity=user_of)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
    ) as client:
        run_id, _ = await ran(client)
        served = await client.get(f"{PATH}/runs/{run_id}/files/sales.csv")
    await harness.aclose()
    assert served.status_code == 307
    assert served.headers["location"].startswith("https://storage.example/")
    assert SigningMemory.signed[-1][1] == LINK_SECONDS


async def test_the_journal_names_each_file_once_with_what_made_it() -> None:
    harness = Harness(config=Settings(files_url=f"memory://files-{uuid.uuid4().hex}"))
    seen: dict[str, Any] = {}

    async def work(input: Any, agent: Runtime) -> str:
        await agent.save_file("a.xlsx", b"x", source="execute")
        await agent.save_file("a.xlsx", b"xy", source="execute")
        await agent.save_file(
            "b.pdf", b"%PDF", source="sandbox_exec", content_type="application/pdf"
        )
        seen["journal"] = Journal.model_validate(agent.replay.journal.dump())
        seen["folder"] = agent.files_folder()
        return "ok"

    result = await harness.wrap(work, id="j").run("x", user="u")
    await harness.aclose()
    assert result.answer == "ok", result.error
    assert seen["journal"].files == [
        {"name": "a.xlsx", "size": 2, "content_type": XLSX, "source": "execute"},
        {"name": "b.pdf", "size": 4, "content_type": "application/pdf", "source": "sandbox_exec"},
    ]
    assert seen["folder"] == f"default/u/{result.run_id}/{result.run_id}"


async def test_a_name_that_is_not_one_path_part_is_refused_and_parts_stay_in_scope() -> None:
    store = FileStore(f"memory://names-{uuid.uuid4().hex}")
    for bad in ("../x.csv", "a/b.csv", "", ".hidden", "x" * 200, "a b.csv"):
        with pytest.raises(ToolError, match="is not a file name"):
            await store.put("f", bad, b"1", source="t")
    folder = scope("t1", "../../etc", "c", "r")
    assert folder.split("/")[0] == "t1" and ".." not in folder and len(folder.split("/")) == 4
    assert await store.size(folder, "x.csv") is None
    await store.put(folder, "x.csv", b"12345", source="t")
    assert await store.size(folder, "x.csv") == 5
    assert b"".join([c async for c in store.chunks(folder, "x.csv")]) == b"12345"
    assert await store.link(folder, "x.csv") is None  # not a store that signs: streamed


async def test_the_store_is_trellis_files_url_else_the_xdg_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    async with Harness(config=Settings()) as harness:
        assert harness.files.url == (tmp_path / "trellis" / "files").as_uri()
    env = {"XDG_DATA_HOME": str(tmp_path)}
    detected = (tmp_path / "trellis" / "files").as_uri()
    assert default_files_url(env) == detected
    assert "files_url" not in sourced(env)  # the harness takes the default: nothing set
    given = {**env, "TRELLIS_FILES_URL": "s3://bucket/files"}
    assert sourced(given)["files_url"] == ("s3://bucket/files", "TRELLIS_FILES_URL")
    assert Settings.from_env(given).files_url == "s3://bucket/files"
    assert default_files_url({}).endswith("/.local/share/trellis/files")


def test_a_cloud_store_whose_package_is_missing_fails_at_start_naming_it() -> None:
    with pytest.raises(ImportError, match="adlfs"):
        Harness(config=Settings(files_url="az://container/files"))
