"""``trellis doctor``: the effective settings with their sources, secrets redacted; each service
reached with its key; the spool and the Docker daemon checked; suggestions, never changes.
And where the Docker daemon is, found as docker-py finds it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from trellis.harness import doctor
from trellis.harness.sandbox.docker import DOCKER_SOCKET, DockerSandbox, docker_host
from trellis.harness.settings import Settings

ENV = {
    "BIFROST_URL": "http://gw.test/v1",
    "BIFROST_VIRTUAL_KEY": "vk-secret-1234",
    "MEMORY_URL": "http://mem.test",
    "TRELLIS_API_KEY": "trk_secret_5678",
    "RUNS_URL": "http://runs.test",
    "TRELLIS_SPOOL_DIR": "",
    "PROMPTS_DIR": "./prompts",
}


def test_the_settings_are_shown_with_their_sources_and_secrets_redacted() -> None:
    rows = {name: (value, source) for name, value, source in doctor.effective(ENV)}
    assert rows["bifrost_url"] == ("http://gw.test/v1", "BIFROST_URL")
    assert rows["bifrost_virtual_key"] == ("set (...1234)", "BIFROST_VIRTUAL_KEY")
    assert rows["api_key"][0] == "set (...5678)" and "secret" not in rows["api_key"][0]
    assert rows["eval_sample"] == ("0.1", "default")
    assert rows["spool_dir"] == ("unset", "default")
    assert doctor.shown("api_key", "short") == "set"
    assert doctor.shown("otlp_headers", {"authorization": "Basic x"}) == "set"


@respx.mock
async def test_each_service_is_reached_with_its_key(tmp_path: Path) -> None:
    respx.get("http://gw.test/v1/models").mock(return_value=httpx.Response(200, json={}))
    respx.get("http://mem.test/v1/keys/self").mock(return_value=httpx.Response(401))
    respx.get("http://runs.test/v1/runs/doctor-check").mock(return_value=httpx.Response(404))
    settings = Settings.from_env({**ENV, "TRELLIS_SPOOL_DIR": str(tmp_path)})
    found = {c.name: c for c in await doctor.checks(settings)}
    assert found["gateway"].ok and found["runs"].ok and found["spool"].ok
    assert not found["memory"].ok and "refused the key (401)" in found["memory"].said


@respx.mock
async def test_a_service_down_or_failing_is_a_failed_check(tmp_path: Path) -> None:
    respx.get("http://gw.test/v1/models").mock(side_effect=httpx.ConnectError("down"))
    respx.get("http://mem.test/v1/keys/self").mock(return_value=httpx.Response(500))
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    settings = Settings.from_env(
        {**ENV, "RUNS_URL": "", "TRELLIS_SPOOL_DIR": str(blocked / "spool")}
    )
    found = {c.name: c for c in await doctor.checks(settings)}
    assert "cannot be reached (ConnectError)" in found["gateway"].said
    assert found["memory"].said.endswith("answered 500") and "runs" not in found
    assert not found["spool"].ok


@respx.mock
async def test_the_docker_daemon_is_pinged_when_sandboxes_are_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://docker.test:2375")
    respx.get("http://docker.test:2375/v1.43/_ping").mock(return_value=httpx.Response(200))
    [check] = await doctor.checks(Settings(sandbox="docker"))
    assert check.ok and check.said == "tcp://docker.test:2375 answered 200"
    respx.get("http://docker.test:2375/v1.43/_ping").mock(side_effect=httpx.ConnectError("x"))
    [check] = await doctor.checks(Settings(sandbox="docker"))
    assert not check.ok


def test_suggestions_never_apply_anything() -> None:
    settings = Settings.from_env(ENV)
    said = doctor.suggestions(settings, ENV)
    assert any(s.startswith("PROMPTS_DIR is deprecated") for s in said)
    assert any("network_config.max_retries" in s for s in said)
    assert any("TRELLIS_SPOOL_DIR" in s for s in said)
    assert any("service.version" in s for s in said)
    assert doctor.suggestions(Settings(spool_dir="/s", agent_version="1"), {}) == []
    lone = Settings(memory_url="http://m", spool_dir="/s", agent_version="1")
    assert doctor.suggestions(lone, {}) == [
        "runs are kept in this process: set RUNS_URL for durable runs and workers"
    ]


@respx.mock
def test_main_prints_the_report_and_says_whether_everything_passed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert doctor.main([], {}) == 0  # nothing configured: nothing to fail
    out = capsys.readouterr().out
    assert out.startswith("Settings (value, and where it came from):") and "Checks:\n  none" in out
    respx.get("http://gw.test/v1/models").mock(return_value=httpx.Response(403))
    assert doctor.main([], {"BIFROST_URL": "http://gw.test/v1"}) == 1


def test_the_cli_takes_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["trellis"])
    assert doctor.cli() == 2
    monkeypatch.setattr("sys.argv", ["trellis", "doctor"])
    monkeypatch.setattr(doctor, "main", lambda argv: 0)
    assert doctor.cli() == 0


# --------------------------------------------------------------------------- the daemon
def test_the_daemon_is_found_as_docker_py_finds_it(tmp_path: Path) -> None:
    assert docker_host({"DOCKER_HOST": "tcp://h:2375"}) == "tcp://h:2375"
    config = tmp_path / ".docker"
    assert docker_host({"DOCKER_CONFIG": str(config)}) == f"unix://{DOCKER_SOCKET}"
    meta = config / "contexts" / "meta" / hashlib.sha256(b"remote").hexdigest()
    meta.mkdir(parents=True)
    (meta / "meta.json").write_text(json.dumps({"Endpoints": {"docker": {"Host": "ssh://u@box"}}}))
    (config / "config.json").write_text(json.dumps({"currentContext": "remote"}))
    assert docker_host({"DOCKER_CONFIG": str(config)}) == "ssh://u@box"
    assert docker_host({"DOCKER_CONFIG": str(config), "DOCKER_CONTEXT": "default"}) == (
        f"unix://{DOCKER_SOCKET}"
    )
    assert DockerSandbox(socket="/tmp/d.sock").host == "unix:///tmp/d.sock"
    assert DockerSandbox(host="tcp://x:1").host == "tcp://x:1"


def test_a_context_without_a_host_is_the_local_socket(tmp_path: Path) -> None:
    config = tmp_path / ".docker"
    meta = config / "contexts" / "meta" / hashlib.sha256(b"odd").hexdigest()
    meta.mkdir(parents=True)
    (meta / "meta.json").write_text(json.dumps({"Endpoints": {"docker": {"Host": ""}}}))
    (config / "config.json").write_text(json.dumps({"currentContext": "odd"}))
    assert docker_host({"DOCKER_CONFIG": str(config)}) == f"unix://{DOCKER_SOCKET}"
