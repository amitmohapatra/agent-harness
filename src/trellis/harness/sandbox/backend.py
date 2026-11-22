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

:class:`CodeBackend` is ``ReAct``'s backend when it is given none (``sandbox.default_files``):
its files the graph's state (``JournalBackend``'s), and ``execute`` a command in the working
folder the harness gives the run's agent by itself (``sandbox.code_for``: on this machine, with
no isolation), described in a few words and left out where the deployment switched commands
off (:class:`CodeCard`).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final

from deepagents.backends.protocol import (
    ExecuteOffloadResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from trellis.contracts import ConfigurationError, ToolError
from trellis.harness.automatic import JournalBackend
from trellis.harness.runtime import Runtime, current
from trellis.harness.sandbox.base import audited, hint, recorded
from trellis.harness.sandbox.local import Folder
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS
from trellis.harness.tools.results import EXECUTE

#: Deep Agents' tool that writes a whole file.
WRITE_FILE: Final = "write_file"
#: Said by a synchronous operation (a graph run with ``invoke``): the harness runs ``ainvoke``.
ASYNC_ONLY: Final = "the sandbox backend runs in a harness run's graph, asynchronously (ainvoke)"
#: What a command on :class:`CodeBackend` answers in a run with no shell (never offered there).
NO_CODE: Final = "Error: this run has no shell (without=['code'], or switched off)."


class SandboxBackend(BaseSandbox):
    """Deep Agents' ``BaseSandbox`` on the run's sandbox (``source``: ``SandboxSource``, its
    provider and spec), each command at most ``call_timeout`` seconds."""

    #: the run's own sandbox: a file written here is its scratch work, announced rather than
    #: asked (``middleware.in_sandbox``), as its commands are
    trellis_sandbox = True

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
        result = await audited(box, command, timeout=min(bounds))
        why = hint(result.exit_code, result.stderr, self.source.spec, result.limit)
        output = "\n".join(part for part in (result.stdout, result.stderr, why) if part)
        return ExecuteResponse(output=output, exit_code=result.exit_code)

    async def aexecute_with_offload(
        self,
        command: str,
        capture_path: str,
        *,
        max_inline_bytes: int,
        max_capture_bytes: int | None = None,
        timeout: int | None = None,  # noqa: ASYNC109 - the command's, killed past it
    ) -> ExecuteOffloadResult:
        """The model's own command: Deep Agents' ``execute`` tool runs it here, and its file
        tools' commands in :meth:`aexecute`. Told with its result what the user gets of what
        it wrote (``SandboxSource.outputs_said``) — which a file tool's parsed output must
        not carry."""
        runtime = _running(self)
        box = await self.source.opened(runtime, changes=True)
        await self.source.seen(runtime, box)
        ran = await super().aexecute_with_offload(
            command,
            capture_path,
            max_inline_bytes=max_inline_bytes,
            max_capture_bytes=max_capture_bytes,
            timeout=timeout,
        )
        await _told(self.source, runtime, box, command, ran.response)
        return ran

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


class CodeBackend(JournalBackend, SandboxBackendProtocol):
    """``ReAct``'s backend when it is given none (``sandbox.default_files``): its files the
    graph's state, as ``JournalBackend``'s (the lean graph and its long-task files unchanged),
    and Deep Agents' ``execute`` a command in the run's working folder — the one the harness
    gives the run's agent by itself (``Agent.code``, ``sandbox.code_of``), made at its first
    command and told what the user gets of the files it wrote —, run there by Deep Agents' own
    ``LocalShellBackend`` (:func:`_own_shell`): the library's shell from end to end. Where the
    deployment switched commands off, ``execute`` is not offered (:class:`CodeCard`); a call
    there all the same answers :data:`NO_CODE`."""

    #: what ``LangGraphAdapter.code`` reads (``adapters.base.DEFAULT_CODE``), with no import
    trellis_default_code = True

    @property
    def id(self) -> str:
        return "trellis-code"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise ConfigurationError(ASYNC_ONLY)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109 - the command's, killed past it
    ) -> ExecuteResponse:
        from trellis.harness.sandbox import code_of  # noqa: PLC0415 - it imports this

        runtime = current()
        code = code_of(runtime)
        if runtime is None or code is None:
            return ExecuteResponse(output=NO_CODE, exit_code=126)
        box = await code.opened(runtime, changes=True)
        await code.seen(runtime, box)
        if isinstance(box, Folder):  # the harness's own default: the library's own shell
            await code.with_results(runtime, box)  # the results the run keeps, there too
            bounds = [s for s in (timeout, code.call_timeout, runtime.remaining()) if s is not None]
            ran = await _own_shell(box, command, min(bounds))
        else:  # a sandbox given as the default (a test's, a deployment's later): its own way
            ran = await SandboxBackend(code, call_timeout=code.call_timeout).aexecute(
                command, timeout=timeout
            )
        return await _told(code, runtime, box, command, ran)


async def _own_shell(box: Folder, command: str, seconds: float) -> ExecuteResponse:
    """``command`` run by Deep Agents' own ``LocalShellBackend`` in the run's folder — the
    library's shell, as the library runs it: on this machine, with the service's environment,
    its output and time limit the library's —, and audited as every command is
    (``base.recorded``)."""
    from deepagents.backends import LocalShellBackend  # noqa: PLC0415 - only where it runs

    shell = LocalShellBackend(root_dir=box.home, virtual_mode=True, inherit_env=True)
    started = time.monotonic()
    ran = await asyncio.to_thread(
        shell.execute, box.placed(command), timeout=max(1, math.ceil(seconds))
    )
    recorded(box, command, ran.exit_code, started, None)
    return ran


async def _told(
    source: Any, runtime: Runtime, box: Any, command: str, response: ExecuteResponse
) -> ExecuteResponse:
    """``response`` to the model's own ``command`` told what the user gets of what it wrote
    (``SandboxSource.outputs_said``), after its output."""
    said = await source.outputs_said(runtime, box, command)
    if said:
        response.output = "\n".join(part for part in (response.output, said) if part)
    return response


class CodeCard(AgentMiddleware):
    """``ReAct`` on :class:`CodeBackend`: Deep Agents' ``execute`` described in a few words
    (``sandbox.CODE``, in place of its own), and left out where the run has none
    (``without=["code"]``, a deployment that says ``deny``: ``sandbox.code_of``) — in the
    graph and in each of its helpers."""

    #: given to the graph's helpers too (``react._carded``)
    helpers = True

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        from trellis.harness.sandbox import CODE, code_of  # noqa: PLC0415 - it imports this

        coded = code_of(current()) is not None
        tools = [_described(t, CODE) for t in request.tools if coded or not _executes(t)]
        return await handler(request.override(tools=tools))


def _executes(tool: Any) -> bool:
    return isinstance(tool, BaseTool) and tool.name == EXECUTE


def _described(tool: Any, description: str) -> Any:
    """``tool`` described so, when it is ``execute``; any other as it is."""
    return tool.model_copy(update={"description": description}) if _executes(tool) else tool


class SandboxCard(AgentMiddleware):
    """``ReAct(backend=sandbox())``: Deep Agents' ``execute`` described with where the user's
    files go (``sandbox.OUTPUTS``) and the sandbox's environment card (``SandboxSource.card``:
    its Python and installed distributions), so the model's first script uses what is there.
    Deep Agents' ``custom_tool_descriptions`` is fixed when the graph is built, before the card
    can be read; this sets it on each request."""

    def __init__(self, source: Any) -> None:
        super().__init__()
        self.source = source

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        from trellis.harness.sandbox import DELIVERS, STARTS  # noqa: PLC0415 - imports this

        card = await self.source.card()  # "": not read (no daemon), only where outputs go
        said = f"\n\n{DELIVERS} {STARTS}" + (f" {card}" if card else "")
        tools = [
            t.model_copy(update={"description": t.description + said})
            if isinstance(t, BaseTool) and t.name == EXECUTE
            else t
            for t in request.tools
        ]
        return await handler(request.override(tools=tools))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """A file the model wrote with ``write_file`` is told, as its command is
        (``SandboxBackend.aexecute_with_offload``), what the user gets of it: nothing, unless
        it is directly in the kept folder."""
        runtime = current()
        if runtime is None or request.tool_call["name"] != WRITE_FILE:
            return await handler(request)
        box = await self.source.opened(runtime, changes=True)
        await self.source.seen(runtime, box)
        result = await handler(request)
        if not isinstance(result, ToolMessage) or result.status == "error":
            return result
        path = str(request.tool_call["args"].get("file_path", ""))
        said = await self.source.written_said(runtime, box, path)
        return (
            result.model_copy(update={"content": f"{result.content}\n{said}"}) if said else result
        )


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


__all__ = [
    "CodeBackend",
    "CodeCard",
    "JournalBackend",
    "SandboxBackend",
    "SandboxCard",
]
