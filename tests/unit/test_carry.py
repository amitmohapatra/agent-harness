"""What a conversation carries of a run's working files (``sandbox.carry.chosen``): the newest
first while they fit the total, none past the largest file carried, the rest left out — and
what doctor says a shell command could reach (``doctor.commands_see``)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

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
