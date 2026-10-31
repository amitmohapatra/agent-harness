"""A sandbox configured on ``ReAct`` (``backend=sandbox()``: Deep Agents' file tools and
``execute`` on the run's sandbox) duplicates nothing the harness does without one: one tool per
name and purpose offered, a large result stored once (Deep Agents' eviction to the sandbox, or
the run's journal where the graph has no ``read_file`` of its own), and Code Mode's meta-tools
apart from the sandbox's ``execute``."""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox, tool
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS
from trellis.harness.sandbox import SandboxRef
from trellis.harness.tools import results

ROWS = 30_000  # about 110k characters: past Deep Agents' 20,000-token eviction and MAX_CHARS


@tool(side_effects="read")
def sales() -> str:
    """Every sales row."""
    return "sku,units\n" + "\n".join(f"s{n},{n % 7}" for n in range(ROWS))


def offered(model: ScriptedChatModel) -> list[list[str]]:
    return [sorted(r["tools"]) for r in model.requests]


class Kept(FakeSandboxes):
    """Sandboxes whose files stay readable after the run deletes its box."""

    def __init__(self) -> None:
        super().__init__()
        self.gone: dict[str, Any] = {}

    async def delete(self, ref: SandboxRef) -> None:
        self.gone[ref.id] = self.boxes.get(ref.id)
        await super().delete(ref)


@pytest.fixture
def kept_by_harness(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The results the harness keeps in the run's journal (``tools.results.keep``)."""
    seen: list[str] = []
    real = results.keep

    def spy(text: str, *, tool: str | None = None) -> str | None:
        path = real(text, tool=tool)
        if path is not None:
            seen.append(path)
        return path

    monkeypatch.setattr(results, "keep", spy)
    return seen


async def test_a_sandbox_backend_offers_each_tool_once_and_stores_a_large_result_once(
    harness: Harness, kept_by_harness: list[str]
) -> None:
    provider = Kept()
    model = ScriptedChatModel(turns=[("sales", {}), ("execute", {"command": "echo 3"}), "3"])
    target = ReAct("s", model, backend=sandbox(provider), context_window=128_000)
    agent = arm(harness.wrap(target, id="analyst"), [sales])
    result = await agent.run("how many units?", user="u")
    assert result.answer == "3", result.error
    for names in offered(model):
        assert len(names) == len(set(names)), names  # no two tools of one name
        assert "read_result" not in names and "sandbox_read" not in names
        assert "execute" in names and names.count("read_file") == 1
        # the evicted file is the sandbox's: code reads it there, no SQL tool beside execute
        assert "query_result" not in names
    # Deep Agents evicted the result to the sandbox: there once, by its call's id ...
    [box] = provider.gone.values()
    saved = [p for p in box.files if p.startswith(results.PREFIX)]
    assert len(saved) == 1, saved
    told = [m for m in model.seen[1] if getattr(m, "type", "") == "tool"][-1].text
    assert saved[0] in told and told.startswith(results.EVICTED)
    # ... and the harness kept no second copy in the run's journal
    assert kept_by_harness == []


async def test_without_a_backend_the_journal_keeps_it_once_and_no_sandbox_is_made(
    harness: Harness, kept_by_harness: list[str]
) -> None:
    model = ScriptedChatModel(turns=[("sales", {}), "done"])
    agent = arm(harness.wrap(ReAct("s", model, context_window=128_000), id="plain"), [sales])
    result = await agent.run("how many units?", user="u")
    assert result.answer == "done", result.error
    assert len(set(kept_by_harness)) == 1 and kept_by_harness[0].startswith(results.PREFIX)
    names = offered(model)[-1]
    assert names.count("read_file") == 1 and "query_result" in names
    assert "execute" not in names


def test_code_modes_meta_tools_are_apart_from_the_sandboxs_execute() -> None:
    """Code Mode runs code that calls MCP tools at the gateway; the sandbox runs commands over
    files: no tool of one has a name of the other's."""
    meta = {t.name for t in CODE_MODE_TOOLS}
    assert meta.isdisjoint({"execute", "read_file", "write_file", "ls", "grep", "glob"})
    assert meta.isdisjoint({"sandbox_exec", "sandbox_read", "sandbox_write", "query_result"})
