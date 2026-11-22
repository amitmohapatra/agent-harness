"""``LocalFolders``: a run's working folder on this machine, with no isolation — what the
harness gives an agent that has no shell of its own (``sandbox.local``). A command runs with
``sh -c`` as the service's own user, in the run's folder; the run's kept results, named at
``/large_tool_results``, are files of that folder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trellis.contracts import ToolError
from trellis.harness.sandbox import SandboxLost, SandboxRef, SandboxSpec
from trellis.harness.sandbox.base import RUN, TENANT
from trellis.harness.sandbox.isolation import READ_MAX_BYTES, RESULTS_PATH
from trellis.harness.sandbox.local import RESULTS, ROOT, LocalFolders, default_root


def ref(name: str = "trellis-run-1", **labels: str) -> SandboxRef:
    return SandboxRef(provider="local", id=name, labels={RUN: "run-1", TENANT: "acme", **labels})


@pytest.fixture
def folders(tmp_path: Path) -> LocalFolders:
    return LocalFolders(tmp_path / "runs")


def test_the_default_root_is_under_the_systems_temporary_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    assert default_root() == tmp_path / ROOT
    assert LocalFolders().root == tmp_path / ROOT and LocalFolders.name == "local"


async def test_a_command_runs_in_the_runs_own_folder_as_the_service(folders: LocalFolders) -> None:
    box = await folders.create(ref(), SandboxSpec(files={"data/in.txt": "seven", "b.bin": b"\x00"}))
    ran = await box.exec(
        'pwd; cat data/in.txt; id -u; printf "%s" "$GIVEN"; echo oops >&2', env={"GIVEN": "x"}
    )
    assert ran.exit_code == 0 and ran.stderr == "oops\n" and ran.duration_ms is not None
    assert ran.stdout == f"{box.home}\nseven{os.geteuid()}\nx"
    assert box.home == folders.root / "trellis-run-1" and box.ref == ref()
    assert await box.read("b.bin") == b"\x00"
    assert (await box.exec("exit 3")).exit_code == 3
    assert (await box.exec("kill -9 $$")).exit_code == 137  # 128 + the signal, as a shell says it


async def test_a_folder_made_again_keeps_its_files_and_takes_no_new_ones(
    folders: LocalFolders,
) -> None:
    box = await folders.create(ref(), SandboxSpec(files={"a.txt": "first"}))
    await box.write("deep/er/b.txt", b"made")
    again = await folders.create(ref(), SandboxSpec(files={"a.txt": "second"}))
    assert await again.read("a.txt") == b"first" and await again.read("deep/er/b.txt") == b"made"
    bare = await folders.create(ref("trellis-run-2"))  # no spec: no files but its own two
    assert sorted(p.name for p in bare.home.iterdir()) == [RESULTS, "outputs"]


async def test_a_command_past_its_time_is_killed_with_every_process_of_it(
    folders: LocalFolders,
) -> None:
    box = await folders.create(ref())
    with pytest.raises(TimeoutError):
        await box.exec("(sleep 30; echo late > late.txt) & sleep 30", timeout=0.3)
    listed = (await box.exec("sleep 0.5; ls")).stdout
    assert listed == "outputs\n"  # its child wrote nothing: it is gone


async def test_a_kept_result_is_a_file_of_the_folder_named_as_the_model_names_it(
    folders: LocalFolders,
) -> None:
    box = await folders.create(ref())
    await box.write(f"{RESULTS_PATH}/rows.csv", b"sku,units\nA,7\n")
    assert (await box.exec(f"ls {box.home}/{RESULTS}")).stdout == "rows.csv\n"  # there, alone
    assert await box.read(f"{RESULTS_PATH}/rows.csv") == b"sku,units\nA,7\n"
    counted = await box.exec(f"tail -n +2 {RESULTS_PATH}/rows.csv | wc -l")
    assert counted.stdout.strip() == "1"


async def test_a_file_that_cannot_be_read_says_why(
    folders: LocalFolders, monkeypatch: pytest.MonkeyPatch
) -> None:
    box = await folders.create(ref(), SandboxSpec(files={"big.txt": "12345"}))
    with pytest.raises(ToolError, match=r"no file missing\.txt"):
        await box.read("missing.txt")
    with pytest.raises(ToolError, match="is not a file"):
        await box.read(RESULTS)
    monkeypatch.setattr("trellis.harness.sandbox.local.READ_MAX_BYTES", 4)
    with pytest.raises(ToolError, match="more than the 4 bytes"):
        await box.read("big.txt")
    assert READ_MAX_BYTES > 4


async def test_folders_are_found_by_their_labels_and_removed_with_their_run(
    folders: LocalFolders,
) -> None:
    assert await folders.labelled({TENANT: "acme"}) == []  # no folder yet: no root either
    one = await folders.create(ref())
    await folders.create(ref("trellis-run-2", **{RUN: "run-2"}))
    await folders.create(ref("trellis-run-3", **{TENANT: "other"}))
    found = await folders.labelled({TENANT: "acme"})
    assert [f.id for f in found] == ["trellis-run-1", "trellis-run-2"]
    assert found[0] == ref() and await folders.labelled({RUN: "run-2", TENANT: "other"}) == []
    assert (await folders.attach(ref())).home == one.home
    await folders.delete(ref())
    await folders.delete(ref())  # gone already: nothing to do
    assert not one.home.exists()
    assert [f.id for f in await folders.labelled({TENANT: "acme"})] == ["trellis-run-2"]
    with pytest.raises(SandboxLost, match="trellis-run-1 is gone"):
        await folders.attach(ref())
    with pytest.raises(SandboxLost):
        await one.exec("ls")


async def test_a_name_never_leaves_the_root(folders: LocalFolders) -> None:
    box = await folders.create(ref("../../outside"))
    assert box.home == folders.root / "outside"
