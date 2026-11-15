"""What a conversation carries of a run's working files (``sandbox.carry.chosen``): the newest
first while they fit the total, none past the largest file carried, the rest left out — and
what doctor says a shell command could reach (``doctor.commands_see``)."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from trellis.harness import doctor
from trellis.harness.sandbox import carry
from trellis.harness.sandbox.carry import FILE_BYTES, TOTAL_BYTES, chosen


def test_the_newest_files_are_carried_while_they_fit_and_none_past_the_largest() -> None:
    half = TOTAL_BYTES // 2
    listing = "\n".join(
        [
            f"300 {half} ./newest.csv",
            f"200 {half} ./middle.csv",
            f"100 {half} ./oldest.csv",
            f"400 {FILE_BYTES + 1} ./huge.bin",
            "250 10 ./small.txt",
            "not a line of the listing",
            "260 10 no-dot-slash",
        ]
    )
    kept, left = chosen(listing)
    assert kept == ["newest.csv", "small.txt"]  # the oldest go first
    assert left == ["huge.bin", "middle.csv", "oldest.csv"]


def test_the_listing_skips_the_outputs_and_the_caches() -> None:
    for name in (".tmp", "__pycache__", ".cache", ".results"):
        assert f"-name '{name}'" in carry.FILES
    assert "-path ./outputs" in carry.FILES


def test_doctor_fails_a_docker_socket_any_user_can_reach_and_warns_of_readable_secrets(
    tmp_path: Path,
) -> None:
    sock = tmp_path / "docker.sock"
    sock.write_text("")
    os.chmod(sock, 0o666)
    secrets = tmp_path / "vault-secrets"
    (secrets / "db").mkdir(parents=True)
    (secrets / "db" / "password").write_text("x")
    (secrets / "db" / "password").chmod(0o644)
    (secrets / "key").write_text("y")
    (secrets / "key").chmod(0o600)
    mounts = tmp_path / "mounts"
    mounts.write_text(f"tmpfs {secrets} tmpfs rw 0 0\nproc /proc proc rw 0 0\n\n")
    check, warned = doctor.commands_see({}, mounts=str(mounts), sockets=[str(sock)])
    assert not check.ok and str(sock) in check.said
    assert len(warned) == 1 and str(secrets / "db" / "password") in warned[0]
    os.chmod(sock, 0o660)
    closed, _ = doctor.commands_see(
        {"DOCKER_HOST": f"unix://{sock}"}, mounts=str(tmp_path / "none"), sockets=[]
    )
    assert closed.ok and stat.S_IMODE(sock.stat().st_mode) == 0o660


class Box:
    """A workspace that answers each command as told (``answers``: by its first word)."""

    def __init__(self, answers: dict[str, tuple[int, str, str]], files: dict[str, bytes]) -> None:
        self.answers = answers
        self.files = files
        self.ran: list[str] = []

    async def exec(self, command: str, **_: object) -> object:
        from trellis.harness.sandbox.base import ExecResult

        self.ran.append(command)
        code, out, err = self.answers.get(command.split(maxsplit=1)[0], (0, "", ""))
        return ExecResult(exit_code=code, stdout=out, stderr=err, duration_ms=1)

    async def read(self, path: str) -> bytes:
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data


def box_of(answers: dict[str, tuple[int, str, str]], files: dict[str, bytes]) -> Any:
    """A :class:`Box`, as the ``Sandbox`` the functions take."""
    return Box(answers, files)


def runtime_of(tmp_path: Path, user: str | None = "u1") -> object:
    from types import SimpleNamespace

    from trellis.harness.files import FileStore

    warned: list[str] = []
    files = FileStore(f"file://{tmp_path}/store")
    return SimpleNamespace(
        tenant="t",
        user=user,
        conversation="c",
        run_id="r",
        parent=None,
        agent_id="a",
        restored=None,
        agent=SimpleNamespace(harness=SimpleNamespace(files=files)),
        events=SimpleNamespace(warning=lambda code, said: warned.append(said)),
        warned=warned,
    )


async def test_what_cannot_be_carried_is_a_warning_and_the_run_goes_on(tmp_path: Path) -> None:
    rt: Any = runtime_of(tmp_path)
    await carry.kept(rt, box_of({"find": (0, "", "")}, {}))  # nothing to carry: nothing kept
    assert await rt.agent.harness.files.read(carry.folder(rt), carry.ARCHIVE) is None
    failing = box_of({"find": (0, "1 3 ./a.txt\n", ""), "tar": (2, "", "tar: no space")}, {})
    await carry.kept(rt, failing)
    assert rt.warned == ["the working files were not kept: tar: no space"]
    big = f"1 {carry.FILE_BYTES + 1} ./huge.bin\n2 3 ./a.txt\n"
    box = box_of({"find": (0, big, "")}, {carry.STAGED: b"archive"})
    await carry.kept(rt, box)
    assert "not carried to the conversation's next turn: huge.bin" in rt.warned[-1]
    # the next run: its archive unpacked, what was left out said too
    again: Any = runtime_of(tmp_path)
    await carry.restored(again, box_of({"tar": (0, "", "")}, {}))
    said = carry.told(again)
    assert said is not None and "a.txt" in said and "huge.bin" in said
    assert carry.told(again) is None and carry.told(None) is None
    broken: Any = runtime_of(tmp_path)
    await carry.restored(broken, box_of({"tar": (2, "", "tar: bad archive")}, {}))
    assert broken.warned == ["the working files were not restored: tar: bad archive"]
    nobody: Any = runtime_of(tmp_path, user=None)  # no user's conversation: nothing carried
    await carry.restored(nobody, box_of({}, {}))
    await carry.kept(nobody, box_of({}, {}))
    assert nobody.restored is None
