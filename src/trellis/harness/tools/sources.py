"""The tools ``tools=[...]`` accepts — the ones this process runs itself: ``tool(fn)`` (a bare
function in the list is ``tool(fn)``), ``a2a(url)`` (a remote agent) and ``openapi(spec)``.
MCP tools are not listed here: they are whatever the agent's Bifrost virtual key allows,
loaded automatically (``tools.toolbox``).

Each resolves to :class:`~trellis.harness.tools.base.Tool`\\ s once per agent. A local
function says what it does (``side_effects``) and an OpenAPI operation is judged by its method;
the tool catalog may override either (``trellis.harness.governance``).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final, overload

import httpx
from pydantic import BaseModel, ConfigDict, create_model

from trellis.contracts import ToolSpec
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS, SideEffects, Source, Tool

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
#: How long an OpenAPI operation may take.
OPENAPI_TIMEOUT_SECONDS: Final = 30.0


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
    ) -> None:
        self.fn = fn
        self.model = _arguments_model(fn)
        self.spec = ToolSpec(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").split("\n\n")[0],
            input_schema=self.model.model_json_schema(),
            source="local",
            side_effects=side_effects,
        )
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    async def resolve(self) -> list[Tool]:
        return [Tool(self.spec, self._run)]

    async def _run(self, args: dict[str, Any]) -> Any:
        validated = self.model.model_validate(args)
        values = {name: getattr(validated, name) for name in type(validated).model_fields}
        result = self.fn(**values)
        return await result if inspect.isawaitable(result) else result


@overload
def tool(
    fn: Callable[..., Any],
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
) -> FunctionTool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...
def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """A function as a tool: ``tool(fn)``, ``@tool`` or ``@tool(side_effects="irreversible")``.

    The schema comes from the signature (pydantic validates the model's arguments), the
    description from the docstring's first paragraph.
    """
    if fn is not None:
        return FunctionTool(fn, name=name, description=description, side_effects=side_effects)
    return lambda f: FunctionTool(f, name=name, description=description, side_effects=side_effects)


def _arguments_model(fn: Callable[..., Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)
    return create_model(  # type: ignore[call-overload]
        f"{fn.__name__}_arguments", __config__=ConfigDict(extra="forbid"), **fields
    )


# --------------------------------------------------------------------------- A2A


class A2ASource:
    """A remote A2A agent as one tool: a message in, its answer out."""

    def __init__(self, url: str, name: str | None) -> None:
        self.url = url
        self.name = name

    async def resolve(self) -> list[Tool]:
        from trellis.harness.a2a.client import remote_agent_tool  # noqa: PLC0415

        return [await remote_agent_tool(self.url, name=self.name)]


def a2a(url: str, *, name: str | None = None) -> A2ASource:
    """The A2A agent whose card is at ``url`` (its base URL), as a tool."""
    return A2ASource(url, name)


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
    ) -> None:
        self.spec = spec
        self.only = None if only is None else frozenset(only)
        self.base_url = base_url
        self.headers = dict(headers or {})
        self._client: httpx.AsyncClient | None = None

    async def resolve(self) -> list[Tool]:
        document = await self._document()
        base = self.base_url or _server_url(document)
        if not base:
            raise ValueError("the OpenAPI document names no server; pass base_url=")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=base, headers=self.headers, timeout=OPENAPI_TIMEOUT_SECONDS
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
                tools.append(_operation_tool(client, path, method, operation))
        return tools

    async def _document(self) -> Mapping[str, Any]:
        if isinstance(self.spec, Mapping):
            return self.spec
        async with httpx.AsyncClient(timeout=OPENAPI_TIMEOUT_SECONDS) as client:
            response = await client.get(self.spec, headers=self.headers)
            response.raise_for_status()
            return response.json()


def openapi(
    spec: str | Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
    base_url: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> OpenAPISource:
    """The operations of an OpenAPI 3 document (a URL or the parsed document) as tools."""
    return OpenAPISource(
        spec, only=None if only is None else list(only), base_url=base_url, headers=headers
    )


def _server_url(document: Mapping[str, Any]) -> str | None:
    servers = document.get("servers") or []
    return servers[0].get("url") if servers and isinstance(servers[0], dict) else None


def _operation_tool(
    client: httpx.AsyncClient, path: str, method: str, operation: Mapping[str, Any]
) -> Tool:
    """Path and query parameters and a JSON body, flattened into one argument object."""
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

    async def run(args: dict[str, Any]) -> Any:
        url = path.format(**{k: v for k, v in args.items() if located.get(k) == "path"})
        query = {k: v for k, v in args.items() if located.get(k) == "query"}
        response = await client.request(method.upper(), url, params=query, json=args.get("body"))
        response.raise_for_status()
        return response.json() if response.content else None

    return Tool(
        ToolSpec(
            name=str(operation["operationId"]),
            description=str(operation.get("summary") or operation.get("description") or ""),
            input_schema={"type": "object", "properties": properties, "required": required},
            source="openapi",
            side_effects=METHOD_SIDE_EFFECTS[method],
        ),
        run,
    )


# --------------------------------------------------------------------------- normalising


def as_source(item: Source | Callable[..., Any]) -> Source:
    """A ``tools=[...]`` entry as a source: a bare function becomes ``tool(fn)``."""
    if hasattr(item, "resolve"):
        return item  # type: ignore[return-value]
    if callable(item):
        return tool(item)
    raise TypeError(f"not a tool source: {item!r}")
