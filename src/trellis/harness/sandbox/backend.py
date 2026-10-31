"""The run's sandbox as a Deep Agents backend — ``ReAct(backend=sandbox())``, or
``create_deep_agent(**sandbox().deep_agent())`` — where the file tools and ``execute`` work.

    agent = h.wrap(ReAct("You analyse data.", "gpt-4o", backend=sandbox()), id="analyst")

Deep Agents' ``BaseSandbox`` gives the tools — ``ls``, ``read_file``, ``write_file``,
``edit_file``, ``delete``, ``glob``, ``grep`` as commands it runs, and ``execute`` — from three
operations, which :class:`SandboxBackend` runs in the run's sandbox (``SandboxSource``, the one
the ``sandbox()`` tools use): a command, a file written, a file read. So the sandbox lives as
its run does, with nothing to call — made at the first call and named in the run's journal,
attached again by a resume on any worker, snapshotted and paused while the run waits for a
person, deleted when the run ends, reaped when its process died (``trellis.harness.sandbox``).
Every call is bounded by the ``call_timeout`` given and what is left of the run's time, and the
tools are governed: ``execute`` and the file tools that change files write (the catalog may
say more), the others read (``ReAct`` declares them; ``middleware.OUTSIDE_STATE``).

The harness runs graphs asynchronously: the synchronous operations raise.
"""

from __future__ import annotations

from typing import Any, Final

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from trellis.contracts import ConfigurationError, ToolError
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS

#: Said by a synchronous operation (a graph run with ``invoke``): the harness runs ``ainvoke``.
ASYNC_ONLY: Final = "the sandbox backend runs in a harness run's graph, asynchronously (ainvoke)"


class SandboxBackend(BaseSandbox):
    """Deep Agents' ``BaseSandbox`` on the run's sandbox (``source``: ``SandboxSource``, its
    provider and spec), each command at most ``call_timeout`` seconds."""

    def __init__(self, source: Any, *, call_timeout: float = REMOTE_TIMEOUT_SECONDS) -> None:
        if call_timeout <= 0:
            raise ConfigurationError("call_timeout is a number of seconds over 0")
        self.source = source
        self.call_timeout = call_timeout

    @property
    def id(self) -> str:
        runtime = current()
        recorded = runtime.replay.journal.sandbox if runtime is not None else None
        return str((recorded or {}).get("id") or "trellis-sandbox")

    # ------------------------------------------------------------------ the three operations
    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109 - the command's, killed past it
    ) -> ExecuteResponse:
        runtime = _running(self)
        box = await self.source.opened(runtime, changes=True)
        await self.source.with_results(runtime, box)  # the results the run keeps, there too
        bounds = [s for s in (timeout, self.call_timeout, runtime.remaining()) if s is not None]
        result = await box.exec(command, timeout=min(bounds))
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return ExecuteResponse(output=output, exit_code=result.exit_code)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        runtime = _running(self)
        box = await self.source.opened(runtime, changes=True)
        done = []
        for path, data in files:
            try:
                await box.write(path, data)
            except ToolError as exc:
                done.append(FileUploadResponse(path=path, error=_error_of(exc)))
            else:
                done.append(FileUploadResponse(path=path))
        return done

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        runtime = _running(self)
        box = await self.source.opened(runtime)
        found = []
        for path in paths:
            try:
                content = await box.read(path)
            except ToolError as exc:
                found.append(FileDownloadResponse(path=path, error=_error_of(exc)))
            else:
                found.append(FileDownloadResponse(path=path, content=content))
        return found

    # ------------------------------------------------------------------ synchronous: none
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise ConfigurationError(ASYNC_ONLY)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise ConfigurationError(ASYNC_ONLY)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise ConfigurationError(ASYNC_ONLY)


def _running(backend: SandboxBackend) -> Runtime:
    """The run calling the backend, with the backend's sandbox among its sources (so its end,
    and its pause, find it: ``sandbox.ended``, ``sandbox.paused``)."""
    runtime = current()
    if runtime is None:
        raise ConfigurationError(
            "the sandbox backend works inside a harness run: wrap the agent (h.wrap) and run it"
        )
    if backend.source not in runtime.added:
        runtime.added.append(backend.source)
    return runtime


def _error_of(exc: ToolError) -> FileOperationError:
    """A sandbox's file error as Deep Agents names it."""
    text = str(exc).lower()
    if "no file" in text or "not find" in text or "no such" in text:
        return "file_not_found"
    if "directory" in text:
        return "is_directory"
    if "permission" in text or "denied" in text:
        return "permission_denied"
    return "invalid_path"


__all__ = ["SandboxBackend"]
