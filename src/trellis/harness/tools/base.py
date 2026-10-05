"""A tool as the harness holds it: what it is (a contracts ``ToolSpec``) and how to run it.

Every source — a local function, an MCP tool the Bifrost virtual key allows, an A2A agent, an
OpenAPI operation, the memory service's agent tools — resolves to :class:`Tool`\\ s. The native
converters (``tools.convert``) wrap a ``Tool`` in the framework's own tool type, and every
call goes through the bridge (governance, approval, journal, recording) before ``run``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from trellis.contracts import ToolSpec

SideEffects = Literal["read", "write", "irreversible"]

#: What an unknown tool is assumed to do: something, but nothing a person must approve.
DEFAULT_SIDE_EFFECTS: SideEffects = "write"

Runner = Callable[[dict[str, Any]], Awaitable[Any]]

#: A JSON schema ``type`` and the Python values that are one (a bool is not a number).
JSON_TYPES: Final[dict[str, tuple[type, ...]]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    run: Runner = field(repr=False)
    #: Bifrost Code Mode meta-tool: its nested calls are recorded from the gateway's log.
    code_mode: bool = False

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def side_effects(self) -> str:
        return self.spec.side_effects


class Source(Protocol):
    """Something ``tools=[...]`` accepts. Resolved again each time the toolbox lists its
    definitions (``toolbox.TOOLS_TTL_SECONDS``)."""

    async def resolve(self) -> list[Tool]: ...


def arguments_problem(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Why ``args`` do not fit a tool's input ``schema``, or ``None``: a light check — required
    fields, the basic types of the declared ones, unknown fields where none are allowed; the
    tool itself validates the rest (a local tool through pydantic). What a model's call is
    checked with (``ReAct``), and a reviewer's edited arguments (``Agent.resume``)."""
    missing = [name for name in schema.get("required") or [] if name not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    properties: dict[str, Any] = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = [name for name in args if name not in properties]
        if unknown:
            return f"unknown argument(s): {', '.join(unknown)}"
    for name, value in args.items():
        expected = (properties.get(name) or {}).get("type")
        allowed = JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if allowed is None:
            continue
        wrong_bool = isinstance(value, bool) and expected != "boolean"
        if wrong_bool or not isinstance(value, allowed):
            return f"{name} must be of type {expected}"
    return None
