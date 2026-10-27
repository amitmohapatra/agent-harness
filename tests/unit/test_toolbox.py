"""The toolbox: the MCP tools the virtual key allows, with their side effects from their
annotations, Code Mode where governance says it is safe, and everything published to the
catalog."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from bifrost_sdk import ToolAnnotations

from tests.support.catalog import FakeCatalog
from trellis import tool
from trellis.contracts import ConfigurationError
from trellis.harness import fresh
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.tools import toolbox
from trellis.harness.tools.toolbox import Toolbox, side_effects_of
from trellis.harness.writes import Writes

READ = ToolAnnotations.model_validate({"readOnlyHint": True})
DESTRUCTIVE = ToolAnnotations.model_validate({"destructiveHint": True})


class Def:
    def __init__(
        self,
        name: str,
        client: str,
        *,
        code_mode: bool = True,
        annotations: ToolAnnotations | None = READ,
    ) -> None:
        self.name, self.client, self.description, self.parameters = name, client, f"{name} tool", {}
        self.code_mode = code_mode
        self.annotations = annotations


class Gateway:
    def __init__(
        self,
        defs: list[Def],
        *,
        auto: dict[str, frozenset[str]] | None = None,
        refuses: bool = False,
    ) -> None:
        self.defs = defs
        #: each client's tools_to_auto_execute, unless the management API ``refuses``
        self.auto = auto or {}
        self.refuses = refuses
        self.executed: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []
        self.listed = 0

    async def tools(self) -> list[Def]:
        self.listed += 1
        return self.defs

    async def auto_executed(self) -> dict[str, frozenset[str]]:
        if self.refuses:
            raise PermissionError("admin auth closes /api to a virtual key")
        return self.auto

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        clients: Any,
        parent_request_id: Any = None,
    ) -> Any:
        self.executed.append((name, args, tuple(clients)))
        return {"ok": name}


def governance(catalog: FakeCatalog | None, writes: Writes) -> Governance:
    """Governance over ``catalog``, publishing through ``writes`` as the harness does."""

    async def submit(entries: list[dict[str, object]], send: Any) -> None:
        await writes.submit("memory.tool_catalog", send)

    return Governance(catalog, submit=submit)


def box(
    defs: list[Def],
    catalog: FakeCatalog | None = None,
    sources: list[Any] | None = None,
    *,
    gateway: Gateway | None = None,
) -> tuple[Toolbox, Gateway, Writes]:
    gateway, writes = gateway or Gateway(defs), Writes()
    made = Toolbox(
        sources or [],
        gateway=gateway,  # type: ignore[arg-type]
        governance=governance(catalog, writes),
    )
    return made, gateway, writes


async def resolved(
    defs: list[Def],
    catalog: FakeCatalog | None = None,
    sources: list[Any] | None = None,
    *,
    gateway: Gateway | None = None,
) -> tuple[dict[str, Any], Gateway, Writes]:
    made, gateway, writes = box(defs, catalog, sources, gateway=gateway)
    tools = await made.tools()
    await writes.drain()
    return {t.name: t for t in tools}, gateway, writes


async def test_mcp_tools_have_the_side_effects_of_their_annotations() -> None:
    tools, gateway, _ = await resolved(
        [
            Def("erp-get_stock", "erp"),
            Def("erp-create_po", "erp", annotations=None),
            Def("erp-delete_po", "erp", annotations=DESTRUCTIVE),
        ]
    )
    assert {n: t.side_effects for n, t in tools.items()} == {
        "erp-get_stock": "read",
        "erp-create_po": "write",
        "erp-delete_po": "irreversible",
    }
    assert await tools["erp-get_stock"].run({"sku": "a"}) == {"ok": "erp-get_stock"}
    assert gateway.executed == [("erp-get_stock", {"sku": "a"}, ("erp",))]


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        (None, "write"),
        ({"read_only_hint": True}, "read"),
        ({"read_only_hint": True, "destructive_hint": True}, "read"),
        ({"destructive_hint": True}, "irreversible"),
        ({"destructive_hint": False, "idempotent_hint": True}, "write"),
    ],
)
def test_annotations_give_an_mcp_tool_its_side_effects(hints: dict | None, expected: str) -> None:
    annotations = None
    if hints is not None:
        annotations = SimpleNamespace(read_only_hint=None, destructive_hint=None)
        for name, value in hints.items():
            setattr(annotations, name, value)
    assert side_effects_of(annotations) == expected


async def test_every_tool_is_published_once_mcp_with_annotations_local_with_side_effects() -> None:
    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return order

    catalog = FakeCatalog({"erp-get_stock": Rule(risk="write")})
    gateway, writes = Gateway([Def("erp-get_stock", "erp")]), Writes()
    governs = governance(catalog, writes)
    for _ in range(2):  # a second toolbox in the tenant publishes nothing new
        made = Toolbox([refund], gateway=gateway, governance=governs)  # type: ignore[arg-type]
        await made.tools()
    await writes.drain()
    [entries] = catalog.published
    by_name = {e["name"]: e for e in entries}
    assert by_name["refund"]["side_effects"] == "irreversible"
    mcp_entry = by_name["erp-get_stock"]
    assert mcp_entry["annotations"] == {"readOnlyHint": True}
    assert "side_effects" not in mcp_entry  # the catalog's own (an admin's) stays


async def test_many_read_only_code_mode_tools_become_the_meta_tools() -> None:
    tools, _, _ = await resolved([Def(f"wiki-t{i}", "wiki") for i in range(20)])
    assert list(tools) == [s.name for s in CODE_MODE_TOOLS]
    assert all(t.feature == "code_mode" for t in tools.values())


async def test_three_read_only_servers_go_to_code_mode_scoped_to_them_others_stay_normal() -> None:
    defs = [
        Def("a-x", "a"),
        Def("b-y", "b"),
        Def("c-z", "c"),
        Def("erp-create_po", "erp", annotations=None),
    ]
    tools, gateway, _ = await resolved(defs)
    assert "erp-create_po" in tools and tools["erp-create_po"].feature != "code_mode"
    await tools["execute_tool_code"].run({"code": "print(1)"})  # the harness's name...
    # ...run as the gateway's meta-tool
    assert gateway.executed == [("executeToolCode", {"code": "print(1)"}, ("a", "b", "c"))]


async def test_a_tool_the_gateway_would_run_itself_is_not_offered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    defs = [Def("erp-get", "erp"), Def("erp-pay", "erp"), Def("crm-a", "crm"), Def("crm-b", "crm")]
    auto = {"erp": frozenset({"pay"}), "crm": frozenset({"*"})}
    tools, _, _ = await resolved(defs, gateway=Gateway(defs, auto=auto))
    assert list(tools) == ["erp-get"]
    assert "erp-pay is not offered" in caplog.text and "tools_to_auto_execute" in caplog.text


async def test_agent_mode_lists_that_cannot_be_read_leave_every_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="trellis.tools")
    defs = [Def("erp-get", "erp")]
    tools, _, _ = await resolved(defs, gateway=Gateway(defs, refuses=True))
    assert list(tools) == ["erp-get"]
    assert "were not checked" in caplog.text


@pytest.mark.parametrize(
    "defs",
    [
        [Def(f"wiki-t{i}", "wiki", code_mode=False) for i in range(20)],
        [Def(f"wiki-t{i}", "wiki", annotations=None if i == 0 else READ) for i in range(20)],
        [Def("a-x", "a"), Def("b-y", "b")],
    ],
    ids=["not-code-mode-clients", "one-write-tool", "too-few"],
)
async def test_code_mode_only_when_every_tool_of_those_servers_reads(defs: list[Def]) -> None:
    tools, _, _ = await resolved(defs)
    assert not any(t.feature == "code_mode" for t in tools.values())
    assert len(tools) == len(defs)


@pytest.mark.parametrize(
    "rule",
    [Rule(risk="read", approve_when="true"), Rule(risk="write")],
    ids=["an-approval-rule", "the-catalogs-risk"],
)
async def test_governance_keeps_a_server_out_of_code_mode(rule: Rule) -> None:
    catalog = FakeCatalog({"wiki-t0": rule})
    tools, _, _ = await resolved([Def(f"wiki-t{i}", "wiki") for i in range(20)], catalog)
    assert not any(t.feature == "code_mode" for t in tools.values())


async def test_two_tools_of_one_name_are_refused() -> None:
    def dup(x: int) -> int:
        return x

    with pytest.raises(ConfigurationError, match="erp-get_stock"):
        await resolved([Def("erp-get_stock", "erp")], sources=[tool(dup, name="erp-get_stock")])


# --------------------------------------------------------------------------- freshness


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(fresh, "_now", clock)
    return clock


async def test_definitions_are_listed_again_after_their_ttl(clock: Clock) -> None:
    made, gateway, writes = box([Def("erp-get", "erp")], FakeCatalog())
    first = await made.tools()
    clock.now += toolbox.TOOLS_TTL_SECONDS / 2
    assert await made.tools() == first and gateway.listed == 1
    clock.now += toolbox.TOOLS_TTL_SECONDS
    await made.tools()
    assert gateway.listed == 2
    await writes.aclose()


async def test_a_rule_that_changes_changes_the_code_mode_choice(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness.governance import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "_now", clock)
    catalog = FakeCatalog()
    made, gateway, writes = box([Def(f"wiki-t{i}", "wiki") for i in range(20)], catalog)
    assert all(t.feature == "code_mode" for t in await made.tools())
    catalog.rules["wiki-t3"] = Rule(risk="read", approve_when="true")
    catalog.version = 2
    clock.now += catalog_module.GOVERNANCE_TTL_SECONDS + 1
    assert not any(t.feature == "code_mode" for t in await made.tools())
    assert gateway.listed == 1  # the definitions stood: only the rules were read again
    await writes.aclose()


async def test_concurrent_cold_runs_share_one_listing() -> None:
    catalog = FakeCatalog()
    made, gateway, writes = box([Def("erp-get", "erp")], catalog)
    results = await asyncio.gather(*(made.tools() for _ in range(10)))
    assert all(r == results[0] for r in results)
    assert gateway.listed == 1 and len(catalog.asked) == 1
    await writes.aclose()


async def test_a_failed_publish_is_sent_again_at_the_next_listing(clock: Clock) -> None:
    catalog = FakeCatalog()
    catalog.refuse_publish = 1
    made, _, writes = box([Def("erp-get", "erp")], catalog)
    await made.tools()
    await writes.drain()
    assert catalog.published == []  # refused: not remembered as published
    clock.now += toolbox.TOOLS_TTL_SECONDS + 1
    await made.tools()
    await writes.drain()
    assert [e["name"] for e in catalog.published[0]] == ["erp-get"]
    clock.now += toolbox.TOOLS_TTL_SECONDS + 1
    await made.tools()
    await writes.drain()
    assert len(catalog.published) == 1  # stored: not sent again
    await writes.aclose()


async def test_an_empty_toolbox_asks_the_catalog_nothing() -> None:
    catalog = FakeCatalog()
    made, _, writes = box([], catalog)
    assert await made.tools() == []
    assert catalog.asked == []
    await writes.aclose()


async def test_a_listing_that_fails_keeps_the_last_good_one(clock: Clock) -> None:
    made, gateway, writes = box([Def("erp-get", "erp")], FakeCatalog())
    first = await made.tools()
    down = Gateway([])

    async def unreachable() -> list[Def]:
        down.listed += 1
        raise ConnectionError("the gateway is down")

    gateway.tools = unreachable  # type: ignore[method-assign]
    clock.now += toolbox.TOOLS_TTL_SECONDS + 1
    assert [t.name for t in await made.tools()] == [t.name for t in first]
    clock.now += toolbox.TOOLS_RETRY_SECONDS / 2
    await made.tools()
    assert down.listed == 1  # not asked again before the retry interval
    clock.now += toolbox.TOOLS_RETRY_SECONDS
    await made.tools()
    assert down.listed == 2
    await writes.aclose()


async def test_a_toolbox_never_listed_raises_when_listing_fails() -> None:
    made, gateway, writes = box([])

    async def unreachable() -> list[Def]:
        raise ConnectionError("the gateway is down")

    gateway.tools = unreachable  # type: ignore[method-assign]
    with pytest.raises(ConnectionError, match="down"):
        await made.tools()
    await writes.aclose()


async def test_an_mcp_tool_its_server_says_is_idempotent_is_idempotent() -> None:
    hinted = ToolAnnotations.model_validate({"idempotentHint": True})
    made, _, writes = box([Def("erp-put", "erp", annotations=hinted), Def("erp-post", "erp")])
    assert {t.name: t.spec.idempotent for t in await made.tools()} == {
        "erp-put": True,
        "erp-post": False,
    }
    await writes.aclose()


# --------------------------------------------------------------------------- hints' readiness


async def test_the_tools_behind_code_mode_are_named_for_the_hints() -> None:
    """B17: under Code Mode the servers' tools are reached through the meta-tools; the hints
    are asked about them too, so a learned procedure of them can be offered."""
    defs = [Def(f"wiki-t{i}", "wiki") for i in range(20)] + [Def("erp-a", "e", annotations=None)]
    made, _, writes = box(defs)
    assert made.hidden() == [], "nothing built yet"
    await made.tools()
    assert made.hidden() == [f"wiki-t{i}" for i in range(20)]
    await made.tools(code_mode=False)
    assert made.hidden(code_mode=False) == []
    await writes.aclose()


async def test_the_first_hints_wait_until_the_published_tools_are_indexed(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B13: a tenant's first run asked for hints before its tools were indexed (all zeros,
    the alphabet offered). The first time only, and at most a few seconds, the toolbox waits."""
    monkeypatch.setattr(toolbox, "INDEX_POLL_SECONDS", 0.0)
    catalog = FakeCatalog()
    catalog.unindexed = 2
    made, _, writes = box([Def("erp-get", "erp")], catalog)
    await made.indexed()  # never listed: nothing published to wait for
    assert catalog.index_asked == []
    await made.tools()
    await writes.drain()
    await made.indexed()
    assert catalog.index_asked == [["erp-get"]] * 3
    await made.indexed()  # the first time only
    assert len(catalog.index_asked) == 3

    late = FakeCatalog()
    late.unindexed = 10**6
    slow, _, slow_writes = box([Def("erp-get", "erp")], late)
    await slow.tools()
    await slow_writes.drain()
    with caplog.at_level("INFO", logger="trellis.tools"):
        await slow.indexed(within=0.0)
    assert "not indexed" in caplog.text

    down = FakeCatalog()
    down.index_fails = True
    failing, _, down_writes = box([Def("erp-get", "erp")], down)
    await failing.tools()
    await down_writes.drain()
    with caplog.at_level("INFO", logger="trellis.tools"):
        await failing.indexed()
    assert "could not be read" in caplog.text
    for w in (writes, slow_writes, down_writes):
        await w.aclose()
