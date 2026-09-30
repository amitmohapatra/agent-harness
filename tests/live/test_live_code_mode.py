"""Bifrost Code Mode through the harness: a source of three read-only Code Mode servers is
given to the agent as the meta-tools, a Starlark script calls a server's tool, and its nested
call is read back from the gateway's MCP log into the memory service's tool records."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_gateway, needs_memory
from tests.live.support import eventually, memory_scope
from trellis import Runtime, mcp
from trellis.contracts import RunStatus
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS

pytestmark = [pytest.mark.live, needs_gateway, needs_memory]


async def test_a_large_read_only_source_runs_as_code_mode(wikis: list[str]) -> None:
    suffix = uuid.uuid4().hex[:8]
    user = f"live-user-{suffix}"
    nested = f"{wikis[1]}-read_wiki_structure"
    script = f'r = {wikis[1]}.read_wiki_structure(repoName="facebook/react")\nprint(str(r)[:120])'

    async def researcher(input: str, agent: Runtime) -> Any:
        files = await agent.tools.call("listToolFiles")
        assert f"{wikis[1]}.pyi" in str(files)
        return await agent.tools.call("executeToolCode", code=script)

    async with live_harness() as h:
        agent = h.wrap(
            researcher, id=f"live-research-{suffix}", tools=[mcp(*wikis)], memory="read_write"
        )
        scope = memory_scope(h, user=user, agent_id=agent.id)
        # the operator's catalog says these servers only read (the harness never guesses)
        names = [
            f"{w}-{t}"
            for w in wikis
            for t in ("ask_wiki_question", "read_wiki_contents", "read_wiki_structure")
        ]
        await scope.advanced.tools.put_catalog(
            [{"name": n, "side_effects": "read", "source": "mcp"} for n in names]
        )
        assert [t.name for t in await h.resolve(agent.sources, tenant=h.settings.tenant)] == [
            s.name for s in CODE_MODE_TOOLS
        ]

        result = await agent.run("What does the React wiki cover?", user=user)
        assert result.status is RunStatus.SUCCESS, result.error
        assert "React" in str(result.answer)
        await h.writes.drain()  # includes the import from the gateway's log, once it settles
        assert h.writes.failed == 0

        async def imported() -> bool:
            entries = await scope.advanced.tools.catalog(names=[nested])
            return bool(entries) and entries[0].stats.calls >= 1

        assert await eventually(imported, within=30)
