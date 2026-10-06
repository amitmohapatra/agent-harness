"""The Bifrost gateway, as the harness uses it: MCP tools (normal, Code Mode, and through a
Virtual MCP), their execution log, chat completions for the ``ReAct`` target and the LLM judge,
and the gateway's repositories of stored prompts and Agent Skills.

Agent Mode (the gateway running tools itself) is never used: every call comes back to the
harness so governance and approval sit in front of it. A completion carries the gateway's
deny-all MCP scope (bifrost-sdk), so the gateway adds no MCP tools to it; the Code Mode
meta-tools are offered under the harness's own names (:data:`CODE_MODE_TOOLS`), because the
gateway runs a call named as one of its own meta-tools itself, inside the completion; and a
tool the gateway would run itself (an MCP client's ``tools_to_auto_execute``) is not offered
(``tools.toolbox``).
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from bifrost_sdk import Bifrost, MCPLog, Options, ToolDef
from bifrost_sdk.admin import Admin, Skill

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
from trellis.harness.fresh import Fresh
from trellis.harness.identity import identity_headers
from trellis.harness.runtime import current
from trellis.harness.tools.base import Tool

#: Page size when reading the MCP execution log back.
LOG_PAGE: Final = 500
#: The gateway writes its MCP log a few seconds behind: a script's nested calls are read
#: every :data:`LOG_POLL_SECONDS` until two reads agree, for at most :data:`LOG_SETTLE_SECONDS`.
LOG_POLL_SECONDS: Final = 2.0
LOG_SETTLE_SECONDS: Final = 20.0
#: How long a stored prompt's id and latest version, and a skill's served version, are kept
#: before they are read again; while the gateway cannot be reached the last ones read stand
#: (read again after :data:`REPOSITORY_RETRY_SECONDS`).
REPOSITORY_TTL_SECONDS: Final = 300.0
REPOSITORY_RETRY_SECONDS: Final = 30.0
#: What separates a prompt's or a skill's name from the version it is pinned to.
PINNED: Final = "@"

#: Bifrost's Code Mode meta-tools, under the harness's names. The gateway publishes no schema
#: for them; these are the arguments its executor checks for. Under its own names (camelCase)
#: the gateway's agent loop runs a call itself when the completion declares one, out of the
#: harness's sight: under these, every call comes back to the harness and goes through the
#: bridge — journaled, governed, recorded — which runs it as the gateway's (:data:`GATEWAY_NAMES`).
CODE_MODE_TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="list_tool_files",
        description="List the virtual declaration files of the MCP servers available to code.",
        input_schema={"type": "object", "properties": {}},
        source="mcp",
        side_effects="read",
    ),
    ToolSpec(
        name="read_tool_file",
        description=(
            "Read a declaration file list_tool_files names (optionally a line range): the "
            "server's tools and their parameters."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "fileName": {"type": "string"},
                "startLine": {"type": "integer"},
                "endLine": {"type": "integer"},
            },
            "required": ["fileName"],
        },
        source="mcp",
        side_effects="read",
    ),
    ToolSpec(
        name="get_tool_docs",
        description="The documentation of one tool of one server.",
        input_schema={
            "type": "object",
            "properties": {"server": {"type": "string"}, "tool": {"type": "string"}},
            "required": ["server", "tool"],
        },
        source="mcp",
        side_effects="read",
    ),
    ToolSpec(
        name="execute_tool_code",
        description=(
            "Run a Starlark (Python-like) script that calls the servers' tools as "
            "server.tool(param=value); returns what it prints."
        ),
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        source="mcp",
        side_effects="read",
    ),
)
#: The gateway's own name of each Code Mode meta-tool.
GATEWAY_NAMES: Final = {
    "list_tool_files": "listToolFiles",
    "read_tool_file": "readToolFile",
    "get_tool_docs": "getToolDocs",
    "execute_tool_code": "executeToolCode",
}


def pinned(ref: str) -> tuple[str, str | None]:
    """``"name@version"`` as its name and version (``None`` when it names none)."""
    name, _, version = ref.partition(PINNED)
    if not name or (PINNED in ref and not version):
        raise ConfigurationError(f"{ref!r} is not a name, or name{PINNED}version")
    return name, version or None


def prompt_ref(ref: str) -> tuple[str, int | None]:
    """A stored prompt's name and the version it is pinned to: ``"triage"``, ``"triage@3"``."""
    name, version = pinned(ref)
    if version is not None and not (version.isdigit() and int(version) >= 1):
        raise ConfigurationError(f"{ref!r}: a prompt's version is a number from 1")
    return name, None if version is None else int(version)


@dataclass(frozen=True, slots=True)
class PromptPin:
    """A stored prompt as a model call selects it: its id and one committed version."""

    name: str
    id: str
    version: int

    def options(self) -> Options:
        return Options(prompt_id=self.id, prompt_version=self.version)

    def attributes(self) -> dict[str, Any]:
        """What a ``chat`` span says about it (the gateway's log does not record it)."""
        return {
            "trellis.prompt.name": self.name,
            "trellis.prompt.id": self.id,
            "trellis.prompt.version": self.version,
        }


class Gateway:
    """One Bifrost gateway (``BIFROST_URL`` is its OpenAI-compatible ``/v1`` base)."""

    def __init__(
        self,
        url: str,
        virtual_key: str | None,
        *,
        client: Bifrost | None = None,
        admin: Admin | None = None,
    ) -> None:
        self.client = client or Bifrost(url, api_key=virtual_key)
        #: the gateway's management API, asked with the virtual key (the stored prompts, the
        #: skills, the MCP clients' Agent Mode lists), as the MCP log is read
        self.admin = admin or Admin(url, token=virtual_key)
        self._prompts: dict[str, Fresh[PromptPin]] = {}
        self._served: dict[str, Fresh[Skill]] = {}
        #: skills as a given version reads (immutable once published), by name and version
        self._versions: dict[tuple[str, str], Skill] = {}

    async def tools(self, slug: str | None = None) -> list[ToolDef]:
        """Every MCP tool the virtual key allows: the gateway's own `/mcp` listing asked with
        the key (never `/api`, which admin auth closes to it), Code Mode clients included —
        or, with ``slug``, the tools of that one Virtual MCP (``/mcp/<slug>``)."""
        return await self.client.tools(slug=slug)

    async def auto_executed(self) -> dict[str, frozenset[str]]:
        """Each MCP client's ``tools_to_auto_execute`` (Agent Mode: tools the gateway runs
        itself), by client name; ``*`` is every tool of the client."""
        return {
            c.config.name: frozenset(c.config.tools_to_auto_execute)
            for c in await self.client.mcp.clients()
        }

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        clients: Sequence[str],
        slug: str | None = None,
        parent_request_id: str | None = None,
    ) -> Any:
        """Run one call (through the Virtual MCP ``slug``, else scoped to ``clients``); the
        tool's result, or :class:`ToolError` when the tool failed. Inside a run the request
        waits at most what is left of the call's time, carries the call's idempotency key as
        its id (the gateway's log shows it; the MCP protocol gives a server no field for it)
        and says who the run is for: the trusted identity header, which the gateway forwards to
        a server whose ``allowed_extra_headers`` name it, and the MCP session id, which keys
        the gateway's per-user credentials when the request has no virtual key."""
        runtime = current()
        call = {
            "id": (runtime.idempotency_key if runtime else None)
            or f"call_{runtime.run_id if runtime else 'direct'}_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        options = Options(
            mcp_clients=None if slug is not None else list(clients),
            parent_request_id=parent_request_id,
            mcp_session_id=f"{runtime.tenant}:{runtime.user}" if runtime else None,
            extra=identity_headers(runtime.tenant, runtime.user) if runtime else {},
        )
        timeout = runtime.remaining() if runtime else None
        turn = await self.client.execute_tool(call, options=options, timeout=timeout, slug=slug)
        content = turn.get("content")
        if turn.get("is_error") or turn.get("isError"):
            raise ToolError(str(content), source=f"mcp.{name}")
        if isinstance(content, str):
            try:
                return json.loads(content)
            except ValueError:
                return content
        return content

    async def code_mode_calls(self, parent_request_id: str, since: datetime) -> list[MCPLog]:
        """The nested calls the Code Mode scripts of one run made, from the gateway's log,
        once it has caught up (two reads agree, or :data:`LOG_SETTLE_SECONDS` passed)."""
        deadline = time.monotonic() + LOG_SETTLE_SECONDS
        seen: list[MCPLog] | None = None
        while True:
            found = await self.client.mcp_logs(since, LOG_PAGE, parent_request_id=parent_request_id)
            settled = seen is not None and found and [e.id for e in found] == [e.id for e in seen]
            if settled or time.monotonic() >= deadline:
                return found
            seen = found
            await asyncio.sleep(LOG_POLL_SECONDS)

    async def complete(
        self, messages: list[dict[str, Any]], *, prompt: PromptPin | None = None, **body: Any
    ) -> dict[str, Any]:
        """A chat completion (never with the gateway's MCP tools: bifrost-sdk sends the
        deny-all scope), with the stored ``prompt`` prepended by the gateway; inside a run,
        each request waits at most what is left of the model call's time (the SDK's retries
        stay within it: the caller bounds them)."""
        runtime = current()
        timeout = runtime.remaining() if runtime else None
        options = prompt.options() if prompt is not None else None
        return await self.client.complete(messages, timeout=timeout, options=options, **body)

    # ------------------------------------------------------------------ repositories
    async def prompt(self, ref: str) -> PromptPin:
        """The stored prompt ``ref`` names (``"name"`` or ``"name@version"``): its id, and the
        version pinned — else its latest committed one, read again every
        :data:`REPOSITORY_TTL_SECONDS`. ``ConfigurationError`` for a prompt that does not
        exist, has no committed version, or not the one named."""
        found = self._prompts.get(ref)
        if found is None:
            found = self._prompts[ref] = Fresh(
                functools.partial(self._prompt, ref),
                what=f"the stored prompt {ref!r}",
                ttl=REPOSITORY_TTL_SECONDS,
                retry=REPOSITORY_RETRY_SECONDS,
                fatal=(ConfigurationError,),
            )
        return await found.get()

    async def _prompt(self, ref: str) -> PromptPin:
        name, version = prompt_ref(ref)
        try:
            found = await self.admin.prompts.find(name)
        except ValueError as exc:  # two prompts with that name
            raise ConfigurationError(f"the gateway's prompt {name!r}: {exc}") from exc
        latest = found.latest_version if found is not None else None
        if found is None or latest is None:
            raise ConfigurationError(f"the gateway has no committed prompt named {name!r}")
        if version is not None and version > latest.number:
            raise ConfigurationError(
                f"the gateway's prompt {name!r} has no version {version} (its latest: "
                f"{latest.number})"
            )
        return PromptPin(name=name, id=found.id, version=version or latest.number)

    async def skill(self, name: str, version: str | None = None) -> Skill:
        """The skill ``name`` as ``version`` reads — its ``SKILL.md`` body and file list —
        else as its served version (read again every :data:`REPOSITORY_TTL_SECONDS`; the last
        one read stands while the gateway is down). ``LookupError`` for no such skill."""
        if version is None:
            found = self._served.get(name)
            if found is None:
                found = self._served[name] = Fresh(
                    functools.partial(self._skill, name, None),
                    what=f"the skill {name!r}",
                    ttl=REPOSITORY_TTL_SECONDS,
                    retry=REPOSITORY_RETRY_SECONDS,
                )
            return await found.get()
        known = self._versions.get((name, version))
        if known is None:
            known = self._versions[name, version] = await self._skill(name, version)
        return known

    async def _skill(self, name: str, version: str | None) -> Skill:
        found = await self.admin.skills.find(name, version)
        if found is None:
            raise LookupError(f"the gateway has no skill named {name!r}")
        return found

    async def served(self, name: str) -> str | None:
        """The version of the skill ``name`` the gateway serves now — read now, not kept: the
        gateway serves a file only of that version (``None``: no such skill)."""
        found = [s for s in await self.admin.skills.list(search=name) if s.name == name]
        return found[0].version if found else None

    async def skill_file(self, name: str, path: str) -> bytes:
        """A file of the skill's served version (the only one the gateway serves files of)."""
        return await self.admin.skills.read_file(name, path)

    async def aclose(self) -> None:
        await asyncio.gather(self.client.aclose(), self.admin.aclose())


def code_mode_tools(gateway: Gateway, servers: Sequence[str]) -> list[Tool]:
    """The meta-tools, under the harness's names, scoped to ``servers``. Scripts run under the
    run's id, so their nested calls can be read back from the log and recorded."""

    async def run(name: str, args: dict[str, Any]) -> Any:
        runtime = current()
        return await gateway.execute(
            GATEWAY_NAMES[name],
            args,
            clients=servers,
            parent_request_id=runtime.run_id if runtime is not None else None,
        )

    return [
        Tool(spec, functools.partial(run, spec.name), feature="code_mode")
        for spec in CODE_MODE_TOOLS
    ]
