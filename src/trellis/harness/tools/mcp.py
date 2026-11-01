"""MCP servers of an agent's own, as ``ReAct(tools=[...])`` entries — one entry type for every
MCP server the harness reaches itself:

* ``mcp("https://host/mcp")`` — a server over HTTP (streamable), ``headers=`` for its auth;
* ``mcp(stdio("npx", "@mcp/files"))`` — a local server the harness starts and talks to over
  its stdin and stdout.

The gateway's MCP tools are every run's: the ones the Bifrost virtual key allows (Bifrost's
admin chooses them, per key), Code Mode included where it fits (``tools.toolbox``).

A server the harness reaches itself (HTTP, stdio) is reached with the official MCP SDK
(``mcp.Client``, the ``mcp`` extra): one connection per server and event loop, opened at the
first listing and kept — a stdio server is one process — and opened again after it breaks.
Its tools are named ``<server>-<tool>`` (``server`` from ``name=``, else the name the server
gives itself), their side effects come from their annotations (``readOnlyHint``: ``read``;
``destructiveHint``: ``irreversible``; otherwise ``write``; the catalog has the last word), and
every call goes through the bridge as any other: governance, the journal, retries of reads,
paging, the run's records.

A server that asks the person in the middle of a call (MCP elicitation: the client says it can
be asked, so a server asks only then) pauses the run as ``trellis.current().ask`` does
(``tools.elicit``: one field of a few values a choice, else a form with the server's schema);
the call is made again when the run resumes, and the answer goes back to the server. On the
protocol of 2026-07-28 the question comes back in the call's result, with its call; before it,
the question is a request of the server's own, which says no call: it is the call's that is under
way on the connection, and a question while several are is refused (:data:`WHOSE`)."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
from trellis.contracts.errors import is_pause_signal
from trellis.harness.asking import RunCancelled
from trellis.harness.runtime import current
from trellis.harness.tools import resources
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS, SideEffects, Tool, data_of
from trellis.harness.tools.elicit import elicited
from trellis.harness.tools.pages import MAX_PAGES

if TYPE_CHECKING:
    from mcp import Client

log = logging.getLogger("trellis.tools")

#: An HTTP server's address.
URL: Final = re.compile(r"^https?://", re.IGNORECASE)
#: What a server that asks while several of its calls are under way is told (before
#: 2026-07-28 its question names no call).
WHOSE: Final = (
    "several calls to this server are under way, and the question names none of them: "
    "ask again when one call is"
)
#: What a server is told when the person answers later: the run pauses, and the call is made
#: again, with the answer, when it resumes.
LATER: Final = "the person answers later: the call is made again then"


@dataclass(frozen=True)
class Stdio:
    """A local MCP server: the command the harness starts, its arguments, environment and
    working directory."""

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None


def stdio(
    command: str, *args: str, env: Mapping[str, str] | None = None, cwd: str | None = None
) -> Stdio:
    """A local MCP server started as ``command *args`` (``env`` merged over the SDK's safe
    defaults, ``cwd`` its folder), for ``mcp(stdio(...))``."""
    if not command:
        raise ConfigurationError("stdio(...) needs the command that starts the server")
    return Stdio(command, tuple(args), None if env is None else dict(env), cwd)


def server_name(text: str) -> str:
    """A server's name as tool names carry it: lower case, ``_`` for anything else."""
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")


def side_effects_of(annotations: Any) -> SideEffects:
    """An MCP tool's side effects from its server's annotations — the MCP SDK's or, through the
    gateway, bifrost-sdk's ``ToolAnnotations``: hints, not guarantees (the catalog has the last
    word)."""
    if _hint(annotations, "read_only_hint", "readOnlyHint"):
        return "read"
    if _hint(annotations, "destructive_hint", "destructiveHint"):
        return "irreversible"
    return "write"


def data_of_listed(listed: Any) -> dict[str, Any]:
    """An MCP tool's declared freshness (``tools.base.data_of``): in its ``_meta`` (MCP's place
    for what the protocol does not name: ``{"freshness": "live"}``, ``{"maxAge": 60}``), else
    in its annotations where the SDK keeps keys it does not know; a value of another kind is
    not read."""
    return data_of(getattr(listed, "meta", None), strict=False) or data_of(
        getattr(listed, "annotations", None), strict=False
    )


def _hint(annotations: Any, attribute: str, key: str) -> bool:
    """An annotation's hint: an SDK object's attribute, or a mapping's key (the wire's name)."""
    if isinstance(annotations, Mapping):
        return bool(annotations.get(key))
    return bool(getattr(annotations, attribute, None))


class MCPSource:
    """One ``mcp(...)`` entry (see the module)."""

    def __init__(
        self,
        target: str | Stdio,
        *,
        name: str | None = None,
        timeout: float | None = REMOTE_TIMEOUT_SECONDS,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.target = target
        self.timeout = timeout
        self.headers = dict(headers or {})
        transport = next((t for t in TRANSPORTS if t.accepts(target)), None)
        if transport is None:
            raise ConfigurationError(
                f"mcp({target!r}): give a server's URL (https://...), or stdio(command, ...) for "
                "a local server (the gateway's MCP tools are the ones the virtual key allows: "
                "every run has them)"
            )
        self.transport = transport
        self.server = None if name is None else server_name(name)
        if name is not None and not self.server:
            raise ConfigurationError(f"mcp(name={name!r}) has no letter or digit to name tools")
        #: each listed tool's annotations, as the catalog takes them (``tools.toolbox``)
        self.annotations: dict[str, dict[str, Any]] = {}
        self._connection = _Connection(self._client)
        #: the calls under way, which a server's question is for
        self._calls: set[_Call] = set()

    def __repr__(self) -> str:
        return f"mcp({self.target!r})"

    @property
    def kind(self) -> McpKind:
        return self.transport.kind

    async def resolve(self) -> list[Tool]:
        """The server's tools, listed through the connection (every page)."""
        client = await self._connection.client()
        server = self.server or self._own_name(client)
        listed: list[Any] = []
        cursor = None
        while True:
            page = await client.list_tools(cursor=cursor)
            listed.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                break
        self.server = server
        self.annotations = {
            f"{server}-{t.name}": t.annotations.model_dump(by_alias=True, exclude_none=True)
            for t in listed
            if t.annotations is not None
        }
        tools = [self._tool(server, t) for t in listed]
        has = getattr(client.server_capabilities, "resources", None) is not None
        if has and resources.READ not in {t.name for t in listed}:
            reader = await self._reader(client, server)
            if reader is not None:
                tools.append(reader)
                self.annotations[reader.name] = {"readOnlyHint": True, "idempotentHint": True}
        return tools

    async def _reader(self, client: Client, server: str) -> Tool | None:
        """The read tool of a server that has resources (``tools.resources``): its description
        from the first page of each listing; ``None`` when the server lists none."""
        listed = await client.list_resources()
        templates = await client.list_resource_templates()
        if not listed.resources and not templates.resource_templates:
            return None
        more = bool(listed.next_cursor or templates.next_cursor)
        spec = ToolSpec(
            name=f"{server}-{resources.READ}",
            description=resources.described(
                server, listed.resources, templates.resource_templates, more=more
            ),
            input_schema=resources.SCHEMA,
            source="mcp",
            server=server,
            side_effects="read",
            idempotent=True,
        )
        return Tool(spec, self._read, timeout=self.timeout, feature="mcp")

    async def _read(self, args: dict[str, Any]) -> Any:
        """A resource's text by its URI; every resource and template, as rows, without one."""
        uri = args.get("uri")
        if uri:
            return resources.text_of(await self._request(lambda c: c.read_resource(uri)))
        return await self._request(self._every)

    async def _every(self, client: Client) -> list[dict[str, Any]]:
        """Every resource and template the server lists (every page, up to ``MAX_PAGES``)."""
        rows: list[dict[str, Any]] = []
        for listing, field_name in (
            (client.list_resources, "resources"),
            (client.list_resource_templates, "resource_templates"),
        ):
            cursor = None
            for _ in range(MAX_PAGES):
                page = await listing(cursor=cursor)
                rows.extend(resources.row(item) for item in getattr(page, field_name))
                cursor = page.next_cursor
                if not cursor:
                    break
        return rows

    def _tool(self, server: str, listed: Any) -> Tool:
        annotations = listed.annotations
        spec = ToolSpec(
            name=f"{server}-{listed.name}",
            description=listed.description or listed.title or listed.name,
            input_schema=listed.input_schema or {"type": "object"},
            output_schema=getattr(listed, "output_schema", None),
            source="mcp",
            server=server,
            side_effects=side_effects_of(annotations),
            idempotent=bool(annotations is not None and annotations.idempotent_hint),
            **data_of_listed(listed),
        )

        async def run(args: dict[str, Any]) -> Any:
            return await self._call(listed.name, args)

        return Tool(spec, run, timeout=self.timeout, feature="mcp")

    async def _call(self, name: str, args: dict[str, Any]) -> Any:
        result = await self._request(lambda client: client.call_tool(name, args))
        if result.is_error:
            said = [c.text for c in result.content if getattr(c, "type", None) == "text"]
            raise ToolError("\n".join(said) or f"{name} failed")
        return answer_of(result)

    async def _request(self, made: Callable[[Client], Awaitable[Any]]) -> Any:
        """One request through the connection, the questions the server asks in it answered
        (:class:`_Call`): an error the server answers is the model's to read, a connection
        that broke is opened again at the next request (and a read is tried again)."""
        from mcp import MCPError  # noqa: PLC0415 - the mcp extra
        from mcp.types import CONNECTION_CLOSED  # noqa: PLC0415

        client = await self._connection.client()
        call = _Call()
        self._calls.add(call)
        try:
            return await call.answering(made(client))
        except Exception as exc:
            if is_pause_signal(exc) or isinstance(exc, RunCancelled):
                raise  # the server asked, and the person answers later
            if isinstance(exc, MCPError) and exc.code != CONNECTION_CLOSED:
                raise ToolError(f"{self.server}: {exc}") from exc  # the server's answer
            self._connection.broken()  # opened again at the next call
            raise ToolError(
                f"the MCP server {self.server} cannot be reached: {exc}", retryable=True
            ) from exc
        finally:
            self._calls.discard(call)

    def _own_name(self, client: Client) -> str:
        info = client.server_info
        named = server_name(info.name) if info is not None and info.name else ""
        return named or self.transport.fallback_name(self.target) or UNNAMED

    def _client(self) -> Client:
        return self.transport.client(self)

    async def _elicited(self, context: Any, params: Any) -> Any:
        """The SDK's elicitation callback: the server's question, handed to the call it is
        for and answered there (:meth:`_Call.answering`)."""
        from mcp.types import INVALID_REQUEST, ErrorData  # noqa: PLC0415 - the mcp extra

        call = _CALL.get() or (next(iter(self._calls)) if len(self._calls) == 1 else None)
        if call is None:
            return ErrorData(code=INVALID_REQUEST, message=WHOSE)
        answered: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        call.asked.put_nowait((params, answered))
        return await answered

    async def aclose(self) -> None:
        """Close the connection (a stdio server's process ends with it)."""
        await self._connection.aclose()


class McpKind(StrEnum):
    HTTP = "http"
    STDIO = "stdio"


@dataclass(frozen=True)
class Transport:
    """One kind of ``mcp(...)`` target: what it accepts, the SDK client a server of it is
    reached with, and a name for a server that gives itself none."""

    kind: McpKind
    accepts: Callable[[Any], bool]
    client: Callable[[MCPSource], Client]
    fallback_name: Callable[[Any], str]


def _sdk() -> Any:
    try:
        import mcp  # noqa: PLC0415 - the mcp extra
        import mcp.client.stdio  # noqa: PLC0415
        import mcp.client.streamable_http  # noqa: PLC0415
        import mcp.shared._httpx_utils  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - the mcp extra installs it
        raise ConfigurationError(
            "mcp(...) needs the mcp extra: pip install 'trellis-harness[mcp]'"
        ) from exc
    return mcp


def _http_client(source: MCPSource) -> Client:
    sdk = _sdk()
    if not source.headers:
        return sdk.Client(source.target, elicitation_callback=source._elicited)
    http = sdk.shared._httpx_utils.create_mcp_http_client(headers=source.headers)
    source._connection.closing.append(http)
    transport = sdk.client.streamable_http.streamable_http_client(source.target, http_client=http)
    return sdk.Client(transport, elicitation_callback=source._elicited)


def _stdio_client(source: MCPSource) -> Client:
    sdk = _sdk()
    target = source.target
    assert isinstance(target, Stdio)
    params = sdk.client.stdio.StdioServerParameters(
        command=target.command,
        args=list(target.args),
        env=None if target.env is None else dict(target.env),
        cwd=target.cwd,
    )
    return sdk.Client(params, elicitation_callback=source._elicited)


def _host_name(target: Any) -> str:
    return server_name((urlsplit(target).hostname or "").split(".")[0])


def _command_name(target: Any) -> str:
    return server_name(PurePath(target.args[-1] if target.args else target.command).stem)


#: The kinds of ``mcp(...)`` target, registered here and only here, tried in this order.
TRANSPORTS: Final[tuple[Transport, ...]] = (
    Transport(McpKind.STDIO, lambda t: isinstance(t, Stdio), _stdio_client, _command_name),
    Transport(
        McpKind.HTTP, lambda t: isinstance(t, str) and bool(URL.match(t)), _http_client, _host_name
    ),
)
#: A server that gives no name, at an address with none to take.
UNNAMED: Final = "mcp"


def answer_of(result: Any) -> Any:
    """A call's answer: its structured content when it has one (a value the server wrapped as
    ``{"result": ...}`` unwrapped), else its text, with any other content after it."""
    structured = result.structured_content
    if structured is not None:
        if isinstance(structured, dict) and list(structured) == ["result"]:
            return structured["result"]
        return structured
    text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
    others = [c.model_dump(exclude_none=True) for c in result.content if c.type != "text"]
    return text if not others else {"text": text, "content": others}


def mcp(
    target: str | Stdio,
    *,
    name: str | None = None,
    call_timeout: float | None = REMOTE_TIMEOUT_SECONDS,
    headers: Mapping[str, str] | None = None,
) -> MCPSource:
    """An MCP server's tools, as a ``ReAct(tools=[...])`` entry: ``target`` is its URL, or a
    local server (:func:`stdio`). ``name`` names its tools (``<name>-<tool>``; else the
    server's own name), ``call_timeout`` is the most one call may take, ``headers`` go with
    every HTTP request (its auth)."""
    return MCPSource(target, name=name, timeout=call_timeout, headers=headers)


#: The call under way in this task: the SDK asks a server's question (2026-07-28) in a task of
#: the call's own, which sees it.
_CALL: ContextVar[_Call | None] = ContextVar("trellis_mcp_call", default=None)


@dataclass(eq=False)
class _Call:
    """One call under way, and the questions its server asks in it."""

    asked: asyncio.Queue[tuple[Any, asyncio.Future[Any]]] = field(default_factory=asyncio.Queue)

    async def answering(self, made: Awaitable[Any]) -> Any:
        """The call's result, each question the server asks in it answered here, in the run's
        task: the person is asked (``ask``), and when they answer later the server is told
        :data:`LATER`, the call stops and the pause goes on to the bridge, which makes the
        call again when the run resumes (its answer then in the journal)."""
        token = _CALL.set(self)
        try:
            task = asyncio.ensure_future(made)
        finally:
            _CALL.reset(token)
        try:
            while True:
                question = asyncio.ensure_future(self.asked.get())
                await asyncio.wait({task, question}, return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    question.cancel()
                    return task.result()
                await _answer(*question.result())
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def _answer(params: Any, answered: asyncio.Future[Any]) -> None:
    """Ask the person the server's question; their answer, or :data:`LATER`, to the server."""
    from mcp.types import INVALID_REQUEST, ElicitResult, ErrorData  # noqa: PLC0415

    asking = elicited(params)
    runtime = current()
    assert runtime is not None, "a harness tool runs in a run (the bridge)"
    try:
        answer = await runtime.ask(asking.question, **asking.asking)
    except BaseException:
        answered.set_result(ErrorData(code=INVALID_REQUEST, message=LATER))
        raise
    answered.set_result(ElicitResult.model_validate(asking.result(answer)))


@dataclass
class _Connection:
    """One MCP client held open in a task of its own (the SDK's client is entered and left in
    one task), used from any task of the same event loop; opened again in another loop, or
    after :meth:`broken`."""

    make: Any
    closing: list[Any] = field(default_factory=list)
    _client: Any = None
    _loop: asyncio.AbstractEventLoop | None = None
    _task: asyncio.Task[None] | None = None
    _stop: asyncio.Event | None = None
    _lock: asyncio.Lock | None = None

    async def client(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop, self._lock, self._client = loop, asyncio.Lock(), None
        assert self._lock is not None
        async with self._lock:
            if self._client is None or self._task is None or self._task.done():
                self._client = await self._open(loop)
        return self._client

    async def _open(self, loop: asyncio.AbstractEventLoop) -> Any:
        ready: asyncio.Future[Any] = loop.create_future()
        stop = self._stop = asyncio.Event()
        client = self.make()
        extra = list(self.closing)
        self.closing.clear()

        async def hold() -> None:
            try:
                async with client as opened:
                    ready.set_result(opened)
                    await stop.wait()
            except BaseException as exc:  # the server's failure, said to whoever waits
                if not ready.done():
                    ready.set_exception(exc)
                else:
                    log.info("an MCP connection closed: %s", exc)
            finally:
                for http in extra:
                    await http.aclose()

        self._task = loop.create_task(hold())
        return await ready

    def broken(self) -> None:
        """The connection failed: the next call opens another."""
        self._client = None
        if self._stop is not None:
            self._stop.set()

    async def aclose(self) -> None:
        task, stop = self._task, self._stop
        self._client = self._task = None
        if stop is not None:
            stop.set()
        if task is not None and task.get_loop() is asyncio.get_running_loop():
            await asyncio.gather(task, return_exceptions=True)
