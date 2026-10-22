"""The tools ``tools=[...]`` accepts — the ones this process runs itself: ``tool(fn)`` (a bare
function in the list is ``tool(fn)``), ``a2a(url)`` (a remote agent) and ``openapi(spec)``; a
wrapped agent in the list is a sub-agent (``trellis.harness.subagents``), and ``mcp(...)`` an
MCP server's tools (``tools.mcp``) — without one, the MCP tools the agent's Bifrost virtual key
allows, loaded automatically (``tools.toolbox``).

Each resolves to :class:`~trellis.harness.tools.base.Tool`\\ s once per agent. A local
function says what it does (``side_effects``) and an OpenAPI operation is judged by its method;
the tool catalog may override either (``trellis.harness.governance``). Each takes a
``timeout``: the most one call may take (an OpenAPI operation and an A2A exchange have one by
default, the same: :data:`~trellis.harness.tools.base.REMOTE_TIMEOUT_SECONDS`).
"""

from __future__ import annotations

import functools
import inspect
import re
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, overload

import httpx
from pydantic import BaseModel, ConfigDict, create_model

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
from trellis.harness.runtime import current
from trellis.harness.tools.base import (
    DEFAULT_SIDE_EFFECTS,
    REMOTE_TIMEOUT_SECONDS,
    SideEffect,
    SideEffects,
    Source,
    Tool,
    invoked,
)

#: What an OpenAPI method does, as a risk tier.
METHOD_SIDE_EFFECTS: Final[dict[str, SideEffects]] = {
    "get": "read",
    "head": "read",
    "options": "read",
    "post": "write",
    "put": "write",
    "patch": "write",
    "delete": "irreversible",
}
#: The statuses an OpenAPI operation answers that may pass on their own: a call that only
#: reads is tried again after one (``tools.base.retried``).
RETRYABLE_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- local functions


class FunctionTool:
    """A Python function as a tool. Still callable as the function it wraps."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
        idempotent: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.fn = fn
        self.model = _arguments_model(fn)
        said, params = documented(inspect.getdoc(fn) or "")
        self.spec = ToolSpec(
            name=name or fn.__name__,
            description=description or said,
            input_schema=_described(self.model.model_json_schema(), params),
            source="local",
            side_effects=side_effects,
            idempotent=idempotent,
        )
        self.tool = Tool(self.spec, self._run, timeout=timeout)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    async def resolve(self) -> list[Tool]:
        return [self.tool]

    async def _run(self, args: dict[str, Any]) -> Any:
        validated = self.model.model_validate(args)
        values = {name: getattr(validated, name) for name in type(validated).model_fields}
        return await invoked(self.fn, **values)


@overload
def tool(
    fn: Callable[..., Any],
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
    idempotent: bool = False,
    timeout: float | None = None,
) -> FunctionTool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
    idempotent: bool = False,
    timeout: float | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...
def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
    idempotent: bool = False,
    timeout: float | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """A function as a tool: ``tool(fn)``, ``@tool`` or ``@tool(side_effects="irreversible")``.

    The schema comes from the signature (pydantic validates the model's arguments), each
    parameter described by the docstring's ``Args:`` (or ``:param x:``) unless an
    ``Annotated[..., Field(description=...)]`` says; the description is the docstring without
    its sections (``Args:``, ``Returns:``...), every other paragraph kept — "use it when / not
    when" included. ``idempotent``: a call repeated with the
    same idempotency key (``trellis.current().idempotency_key``) has its effect once, so it is
    retried like a read and run again after a crash. ``timeout``: the most one call may take,
    in seconds (a sync function runs in a worker thread, which cannot be stopped: its result
    is dropped). A result that comes from outside the run (a person, another system) is an
    ``ask`` in the function: ``return await trellis.current().ask(question, expects=...)``.
    """
    made = functools.partial(
        FunctionTool,
        name=name,
        description=description,
        side_effects=side_effects,
        idempotent=idempotent,
        timeout=timeout,
    )
    return made(fn) if fn is not None else made


#: A docstring's section headers (Google style): its ``Args`` describe the parameters, and none
#: of them is the tool's description.
_SECTION: Final = re.compile(
    r"^(Args|Arguments|Parameters|Params|Keyword Args|Returns?|Yields?|Raises|Examples?|"
    r"Notes?|Todo|See Also|Attributes):\s*$"
)
_ARG_SECTIONS: Final = frozenset({"Args", "Arguments", "Parameters", "Params", "Keyword Args"})
#: An argument's entry in an ``Args`` section (``name (type): what it is``), and a Sphinx field.
_ARG: Final = re.compile(r"^\s+\**(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_PARAM: Final = re.compile(r"^:param\s+(?:[^:]*\s)?(\w+):\s*(.*)$")
_FIELD: Final = re.compile(r"^:(\w+)(\s[^:]*)?:")


def documented(doc: str) -> tuple[str, dict[str, str]]:
    """A function's docstring as a tool's: its description — every paragraph but the sections
    (``Args:``, ``Returns:``, ``Raises:``... and Sphinx ``:param x:`` fields), so a "use it
    when / not when" paragraph reaches the model — and each parameter's description, from
    ``Args:`` (Google style) or ``:param x:`` (Sphinx)."""
    kept: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None  # the section the line is in
    indent: int | None = None  # its entries' indent
    current: str | None = None  # the parameter a line goes on describing
    for line in doc.splitlines():
        stripped = line.strip()
        header = _SECTION.match(line)
        if header:
            section, indent, current = header.group(1), None, None
            continue
        if section is not None and (not stripped or line[0].isspace()):
            if stripped and section in _ARG_SECTIONS:
                width = len(line) - len(line.lstrip())
                indent = width if indent is None else indent
                entry = _ARG.match(line) if width == indent else None
                if entry:
                    current = str(entry.group(1))
                    params[current] = entry.group(2).strip()
                elif current is not None:
                    params[current] = f"{params[current]} {stripped}".strip()
            continue
        section = None
        field = _FIELD.match(line)
        if field:
            param = _PARAM.match(line)
            current = param.group(1) if param else None
            if param:
                params[param.group(1)] = param.group(2).strip()
            continue
        if current is not None and stripped and line[0].isspace():
            params[current] = f"{params[current]} {stripped}".strip()  # a field goes on
            continue
        current = None
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), params


def _described(schema: dict[str, Any], params: Mapping[str, str]) -> dict[str, Any]:
    """The arguments' schema with each documented parameter's description, where the
    signature gave none (an ``Annotated[..., Field(description=...)]`` wins)."""
    properties = schema.get("properties") or {}
    for name, text in params.items():
        found = properties.get(name)
        if isinstance(found, dict) and text and "description" not in found:
            found["description"] = text
    return schema


def _arguments_model(fn: Callable[..., Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    try:  # the annotations as the function's module means them (``from __future__ import
        # annotations`` leaves them strings, which pydantic cannot resolve from here)
        hints = typing.get_type_hints(fn, include_extras=True)
    except (NameError, TypeError, AttributeError):  # unresolvable: the annotations as written
        hints = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(param.name, param.annotation)
        annotation = Any if annotation is inspect.Parameter.empty else annotation
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)
    return create_model(  # type: ignore[call-overload]
        f"{fn.__name__}_arguments", __config__=ConfigDict(extra="forbid"), **fields
    )


# --------------------------------------------------------------------------- A2A


class A2ASource:
    """A remote A2A agent as one tool: a message in, its answer out."""

    def __init__(self, url: str, name: str | None, timeout: float) -> None:
        self.url = url
        self.name = name
        self.timeout = timeout

    async def resolve(self) -> list[Tool]:
        from trellis.harness.a2a.client import remote_agent_tool  # noqa: PLC0415

        return [await remote_agent_tool(self.url, name=self.name, timeout=self.timeout)]


def a2a(url: str, *, name: str | None = None, timeout: float = REMOTE_TIMEOUT_SECONDS) -> A2ASource:
    """The A2A agent whose card is at ``url`` (its base URL), as a tool. ``timeout``: the most
    one exchange may take, in seconds."""
    return A2ASource(url, name, timeout)


# --------------------------------------------------------------------------- OpenAPI


class OpenAPISource:
    """Operations of an OpenAPI 3 document, each a tool named by its ``operationId``."""

    def __init__(
        self,
        spec: str | Mapping[str, Any],
        *,
        only: Sequence[str] | None,
        base_url: str | None,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> None:
        self.spec = spec
        self.only = None if only is None else frozenset(only)
        self.base_url = base_url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def resolve(self) -> list[Tool]:
        document = await self._document()
        base = self.base_url or _server_url(document)
        if not base:
            raise ValueError("the OpenAPI document names no server; pass base_url=")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=base, headers=self.headers, timeout=self.timeout
            )
        client = self._client
        tools: list[Tool] = []
        for path, item in (document.get("paths") or {}).items():
            for method, operation in item.items():
                if method not in METHOD_SIDE_EFFECTS or not isinstance(operation, dict):
                    continue
                name = operation.get("operationId")
                if not name or (self.only is not None and name not in self.only):
                    continue
                tools.append(_operation_tool(client, path, method, operation, self.timeout))
        return tools

    async def _document(self) -> Mapping[str, Any]:
        if isinstance(self.spec, Mapping):
            return self.spec
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.spec, headers=self.headers)
            response.raise_for_status()
            return response.json()


def openapi(
    spec: str | Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
    base_url: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = REMOTE_TIMEOUT_SECONDS,
) -> OpenAPISource:
    """The operations of an OpenAPI 3 document (a URL or the parsed document) as tools.
    ``timeout``: the most one operation may take, in seconds."""
    return OpenAPISource(
        spec,
        only=None if only is None else list(only),
        base_url=base_url,
        headers=headers,
        timeout=timeout,
    )


def _server_url(document: Mapping[str, Any]) -> str | None:
    servers = document.get("servers") or []
    return servers[0].get("url") if servers and isinstance(servers[0], dict) else None


def _operation_tool(
    client: httpx.AsyncClient,
    path: str,
    method: str,
    operation: Mapping[str, Any],
    timeout: float,
) -> Tool:
    """Path and query parameters and a JSON body, flattened into one argument object. A call
    that does more than read sends the call's idempotency key as ``Idempotency-Key``; an
    answer that may pass (:data:`RETRYABLE_STATUSES`) is an error that says so."""
    parameters = [p for p in operation.get("parameters") or [] if isinstance(p, dict)]
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in parameters:
        properties[param["name"]] = param.get("schema") or {"type": "string"}
        if param.get("required") or param.get("in") == "path":
            required.append(param["name"])
    body = (
        ((operation.get("requestBody") or {}).get("content") or {}).get("application/json") or {}
    ).get("schema")
    if body is not None:
        properties["body"] = body
        if (operation.get("requestBody") or {}).get("required"):
            required.append("body")
    located = {p["name"]: p.get("in", "query") for p in parameters}
    side_effects = METHOD_SIDE_EFFECTS[method]

    async def run(args: dict[str, Any]) -> Any:
        url = path.format(**{k: v for k, v in args.items() if located.get(k) == "path"})
        query = {k: v for k, v in args.items() if located.get(k) == "query"}
        runtime = current()
        key = runtime.idempotency_key if runtime is not None else None
        headers = {"Idempotency-Key": key} if key and side_effects != SideEffect.READ else None
        response = await client.request(
            method.upper(), url, params=query, json=args.get("body"), headers=headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            retryable = response.status_code in RETRYABLE_STATUSES
            raise ToolError(str(exc), source="tools", retryable=retryable) from exc
        return response.json() if response.content else None

    return Tool(
        ToolSpec(
            name=str(operation["operationId"]),
            description=str(operation.get("summary") or operation.get("description") or ""),
            input_schema={"type": "object", "properties": properties, "required": required},
            source="openapi",
            side_effects=side_effects,
        ),
        run,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- normalising


@dataclass(frozen=True)
class EntryKind:
    """One kind of ``tools=[...]`` entry: what it is (``accepts``) and the source it becomes
    (``source``)."""

    name: str
    accepts: Callable[[Any], bool]
    source: Callable[[Any], Source]


def _is_agent(item: Any) -> bool:
    from trellis.harness.agent import Agent  # noqa: PLC0415 - the agent imports the sources

    return isinstance(item, Agent)


def _sub_agent(item: Any) -> Source:
    from trellis.harness.subagents import SubAgent  # noqa: PLC0415

    return SubAgent(item, name=None, description=None, side_effects=None)


#: The kinds of ``tools=[...]`` entry, registered here and only here, tried in this order: a
#: wrapped agent (a sub-agent), anything that resolves to tools itself (``tool``, ``mcp``,
#: ``a2a``, ``openapi``, ``sandbox``, ``skills``, a source of yours), a bare function.
ENTRY_KINDS: Final[tuple[EntryKind, ...]] = (
    EntryKind("agent", _is_agent, _sub_agent),
    EntryKind("source", lambda item: hasattr(item, "resolve"), lambda item: item),
    EntryKind("function", callable, FunctionTool),
)


def as_source(item: Source | Callable[..., Any] | Any) -> Source:
    """A ``tools=[...]`` entry as a source, by the first :data:`ENTRY_KINDS` that accepts it."""
    for kind in ENTRY_KINDS:
        if kind.accepts(item):
            return kind.source(item)
    raise ConfigurationError(
        f"not a tool: {item!r} — give a function, tool(...), mcp(...), a2a(...), openapi(...), "
        "sandbox(), skills(...) or a wrapped agent"
    )
