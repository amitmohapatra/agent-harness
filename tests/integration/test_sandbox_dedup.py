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


def run_boxes(provider: Kept) -> list[Any]:
    """The run's sandboxes, not the one the environment card's probe made
    (``SandboxProvider.card``)."""
    return [b for ref, b in provider.gone.items() if not ref.startswith("trellis-probe-")]


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


async def test_a_sandbox_backend_offers_each_tool_once_and_the_run_keeps_a_large_result(
    harness: Harness, kept_by_harness: list[str]
) -> None:
    """With a sandbox a large result has one owner, the run, as without one: kept in its
    journal, its stub naming the table ``query_result`` computes over (offered from then on),
    and written into the sandbox as a file for code (``SandboxSource.with_results``) — once
    there, Deep Agents evicting nothing beside it. Left to Deep Agents, the file was the
    sandbox's alone, where the harness could not read it: no ``query_result``, no table named,
    and a count done in code came out 105 where every query said 104."""
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
    assert "query_result" in offered(model)[1]  # once the run keeps the result
    [path] = set(kept_by_harness)
    told = [m for m in model.seen[1] if getattr(m, "type", "") == "tool"][-1].text
    assert told.startswith(results.EVICTED) and path in told
    assert "Its rows are the table sales (columns: " in told  # the table the query reads
    assert results.CODE.format(path=path, tool="execute") in told  # and the file, for code
    # in the sandbox once, at that path: no copy of Deep Agents' beside it
    [box] = run_boxes(provider)
    assert [p for p in box.files if p.startswith(results.PREFIX)] == [path]


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


async def test_a_result_over_the_window_is_offloaded_once_where_code_reads_it(
    harness: Harness, kept_by_harness: list[str]
) -> None:
    """Below Deep Agents' eviction but over the window, a result is Deep Agents' to make fit
    (its summarization's overflow path offloads it whole to the backend): ``FitWindow`` does
    not also cut it and keep it in the run's journal, which the sandbox would then hold a
    second time. One copy, the whole result, in the sandbox where code reads it."""
    # a log, not a table: a table is also kept for query_result (``results.tabled``)
    rows = "\n".join(f"event {n}: stock moved by {n % 5}" for n in range(1_500))

    @tool(side_effects="read")
    def stock() -> str:
        """Every stock event."""
        return rows

    provider = Kept()
    model = ScriptedChatModel(turns=[("stock", {}), ("execute", {"command": "echo 2"}), "2"])
    target = ReAct("s", model, backend=sandbox(provider), context_window=6_000)
    agent = arm(harness.wrap(target, id="fit"), [stock])
    result = await agent.run("how many units?", user="u")
    assert result.answer == "2", result.error
    assert kept_by_harness == []  # no harness copy
    [box] = run_boxes(provider)
    saved = {p: v for p, v in box.files.items() if p.startswith(results.PREFIX)}
    assert list(saved) == [f"{results.PREFIX}/call_1_0"]  # once, by its call's id
    assert saved[f"{results.PREFIX}/call_1_0"].decode() == rows  # whole
