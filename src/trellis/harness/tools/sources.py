"""The tools ``tools=[...]`` accepts — the ones this process runs itself: ``tool(fn)`` (a bare
function in the list is ``tool(fn)``), ``a2a(url)`` (a remote agent) and ``openapi(spec)``.
MCP tools are not listed here: they are whatever the agent's Bifrost virtual key allows,
loaded automatically (``tools.toolbox``).

Each resolves to :class:`~trellis.harness.tools.base.Tool`\\ s once per agent. A local
function says what it does (``side_effects``) and an OpenAPI operation is judged by its method;
the tool catalog may override either (``trellis.harness.governance``). Each takes a
``timeout``: the most one call may take (an OpenAPI operation and an A2A exchange have one by
default, the same: :data:`~trellis.harness.tools.base.REMOTE_TIMEOUT_SECONDS`).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final, overload

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

from trellis.contracts import ToolCall, ToolError, ToolSpec
from trellis.harness.hooks import Approval
from trellis.harness.runtime import current
from trellis.harness.tools.base import (
    DEFAULT_SIDE_EFFECTS,
    REMOTE_TIMEOUT_SECONDS,
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
        approval: Approval | None = None,
        external: bool = False,
    ) -> None:
        self.fn = fn
        self.model = _arguments_model(fn)
        self.external = external
        #: an external tool's result schema: its return annotation's (``None``: any value)
        self.result_schema = _result_schema(fn) if external else None
        self.spec = ToolSpec(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").split("\n\n")[0],
            input_schema=self.model.model_json_schema(),
            source="local",
            side_effects=side_effects,
            idempotent=idempotent,
        )
        self.tool = Tool(self.spec, self._run, timeout=timeout, approval=approval)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    async def resolve(self) -> list[Tool]:
        return [self.tool]

    async def _run(self, args: dict[str, Any]) -> Any:
        validated = self.model.model_validate(args)
        values = {name: getattr(validated, name) for name in type(validated).model_fields}
        if self.external:
            return await self._outside(values)
        return await invoked(self.fn, **values)

    async def _outside(self, args: dict[str, Any]) -> Any:
        """An external tool's call: the run pauses with it, and the result given from
        outside is what the call returns (the function itself never runs)."""
        runtime = current()
        if runtime is None:
            raise ToolError(
                f"{self.spec.name} is an external tool: its result is given to a paused run "
                "(agent.resume(run_id, result=...)), so it is called inside a Harness run",
                source="tools",
            )
        call = ToolCall(
            tool=self.spec.name,
            args=args,
            task=runtime.task,
            idempotency_key=runtime.idempotency_key,
        )
        result = await runtime.external(call, self.result_schema)
        if result is False:
            raise ToolError(
                f"{self.spec.name} was not done: its result was refused", source="tools"
            )
        return result


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
    approval: Approval | None = None,
    external: bool = False,
) -> FunctionTool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    side_effects: SideEffects = DEFAULT_SIDE_EFFECTS,
    idempotent: bool = False,
    timeout: float | None = None,
    approval: Approval | None = None,
    external: bool = False,
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
    approval: Approval | None = None,
    external: bool = False,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """A function as a tool: ``tool(fn)``, ``@tool`` or ``@tool(side_effects="irreversible")``.

    The schema comes from the signature (pydantic validates the model's arguments), the
    description from the docstring's first paragraph. ``idempotent``: a call repeated with the
    same idempotency key (``trellis.current().idempotency_key``) has its effect once, so it is
    retried like a read and run again after a crash. ``timeout``: the most one call may take,
    in seconds (a sync function runs in a worker thread, which cannot be stopped: its result
    is dropped).

    ``approval``: your rule for each call, ``fn(args)`` (sync or async) returning ``None``
    (governance decides, as for any tool), ``True`` (approved: it runs without asking) or
    ``Ask(question, assignee=, component=, props=)`` (a person approves it first, asked that);
    its answer is journaled. ``external``: the call is done outside the run (a person, another
    system): the run pauses with it and the result given from outside
    (``agent.resume(run_id, result=...)``) is what the model reads; the function is never
    run, its return annotation is the result's schema.
    """
    made = functools.partial(
        FunctionTool,
        name=name,
        description=description,
        side_effects=side_effects,
        idempotent=idempotent,
        timeout=timeout,
        approval=approval,
        external=external,
    )
    return made(fn) if fn is not None else made


def _result_schema(fn: Callable[..., Any]) -> dict[str, Any] | None:
    """The JSON Schema of what ``fn`` returns, from its return annotation (``None`` without
    one, or for ``None``/``Any``)."""
    annotation = inspect.signature(fn, eval_str=True).return_annotation
    if annotation in (inspect.Signature.empty, None, type(None), Any):
        return None
    return TypeAdapter(annotation).json_schema()


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
        headers = {"Idempotency-Key": key} if key and side_effects != "read" else None
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


def as_source(item: Source | Callable[..., Any]) -> Source:
    """A ``tools=[...]`` entry as a source: a bare function becomes ``tool(fn)``."""
    if hasattr(item, "resolve"):
        return item  # type: ignore[return-value]
    if callable(item):
        return tool(item)
    raise TypeError(f"not a tool source: {item!r}")
