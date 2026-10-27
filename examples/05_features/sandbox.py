"""Code run away from the host: ``ReAct(backend=sandbox())`` — Deep Agents' ``execute`` and
file tools work in the run's own sandbox (``create_deep_agent(**sandbox().deep_agent())`` the
same for a Deep Agent of your own, ``sandbox().tools()`` the same sandbox as tools for an engine
with none) — then the same provider without a harness (Way 2): ``governed`` around a sandbox's
``exec``.

``sandbox()`` alone is a Docker container with no network (the daemon ``DOCKER_HOST`` names), of
``sandbox(image=)`` (else ``python:3.12-slim``). Without ``DOCKER_HOST`` this
example plugs in a provider of its own — temporary directories of this machine (no isolation: an
example of the interface only). Either way the sandbox lives as its run does: made at the first
command, named in the run's journal, deleted at its end; ``execute`` is governed and recorded as
any tool.

    DOCKER_HOST=unix:///var/run/docker.sock python -m examples.05_features.sandbox
    python -m examples.05_features.sandbox
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples._support.offline import react_model

from trellis import Harness, ReAct, sandbox
from trellis.contracts import RunEventType, ToolError
from trellis.harness.governance import Governance, governed
from trellis.harness.sandbox import (
    DockerSandbox,
    ExecResult,
    SandboxLost,
    SandboxProvider,
    SandboxRef,
    SandboxSpec,
)

SCRIPT = "import csv\nrows = list(csv.DictReader(open('sales.csv')))\n" + (
    "print(sum(int(r['units']) for r in rows), 'units')\n"
)


class TempDirs:
    """Sandboxes as temporary directories of this machine — NO isolation: an example of the
    interface (``trellis.harness.sandbox.base``), never a place for code nobody reviewed."""

    name = "tempdir"

    def __init__(self) -> None:
        self.root = Path(tempfile.gettempdir()) / "trellis-sandboxes"
        self.root.mkdir(exist_ok=True)

    async def create(self, ref: SandboxRef, spec: SandboxSpec) -> Directory:
        home = self.root / ref.id
        if not home.exists():  # one that exists is adopted (an attempt after a crash)
            home.mkdir()
            (home / ".labels.json").write_text(json.dumps(ref.labels))
            for path, content in spec.files.items():
                (home / path).write_bytes(content.encode() if isinstance(content, str) else content)
        return Directory(ref, home)

    async def attach(self, ref: SandboxRef) -> Directory:
        if not (self.root / ref.id).exists():
            raise SandboxLost(f"the sandbox {ref.id} is gone", source="tools")
        return Directory(ref, self.root / ref.id)

    async def delete(self, ref: SandboxRef) -> None:
        shutil.rmtree(self.root / ref.id, ignore_errors=True)

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        found = []
        for home in self.root.iterdir():
            own = json.loads((home / ".labels.json").read_text())
            if all(own.get(k) == v for k, v in labels.items()):
                found.append(SandboxRef(provider=self.name, id=home.name, labels=own))
        return found


class Directory:
    def __init__(self, ref: SandboxRef, home: Path) -> None:
        self.ref = ref
        self.home = home

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's, killed past it
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.home,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # its own process group: killed with what it started
        )
        try:
            async with asyncio.timeout(timeout):
                out, err = await process.communicate()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            raise
        code = process.returncode or 0
        return ExecResult(exit_code=code, stdout=out.decode(), stderr=err.decode())

    async def read(self, path: str) -> bytes:
        if not (self.home / path).is_file():
            raise ToolError(f"there is no file {path} in the sandbox", source="tools")
        return (self.home / path).read_bytes()

    async def write(self, path: str, data: bytes) -> None:
        (self.home / path).parent.mkdir(parents=True, exist_ok=True)
        (self.home / path).write_bytes(data)


async def main() -> None:
    docker = bool(os.environ.get("DOCKER_HOST"))
    provider: SandboxProvider = DockerSandbox() if docker else TempDirs()
    spec = SandboxSpec(files={"sales.csv": "region,units\nnorth,12\nsouth,30\n"})
    async with Harness() as h:
        model = react_model(
            [
                ("execute", {"command": f"cat > total.py << 'EOF'\n{SCRIPT}EOF"}),
                ("execute", {"command": "python3 total.py"}),
                "We sold 42 units.",
            ]
        )
        analyst = ReAct(
            system_prompt="You analyse data: write a Python script in your sandbox, run it.",
            model=model,
            backend=sandbox(provider, spec=spec, call_timeout=60),
        )
        agent = h.wrap(analyst, id="analyst")
        async for event in agent.stream("How many units did we sell?", user="ada"):
            if event.type is RunEventType.TOOL_CALL_RESULT:
                print(event.data["tool"], "->", event.data["output"])
            if event.type is RunEventType.RUN_FINISHED:
                print(event.outcome, event.data.get("result"))
        print("sandboxes left:", await provider.labelled({"trellis.agent_id": "analyst"}))

    # Way 2: no harness — the provider and governance around a command
    box = await provider.create(SandboxRef(provider=provider.name, id="my-own"), spec)

    async def sandbox_exec(command: str) -> dict[str, Any]:
        return (await box.exec(command, timeout=30)).model_dump()

    run = governed(sandbox_exec, Governance(), on_ask=lambda decision: True)
    print(await run(command="wc -l sales.csv"))
    await provider.delete(box.ref)


if __name__ == "__main__":
    asyncio.run(main())
