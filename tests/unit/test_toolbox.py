"""The toolbox: the MCP tools the virtual key allows, tiered by annotations and the catalog,
Code Mode where it is safe, and everything published to the catalog."""

from __future__ import annotations

from typing import Any

import pytest
from bifrost_sdk import ToolAnnotations

from trellis import tool
from trellis.contracts import ConfigurationError
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS
from trellis.harness.clients.memory import Governance
from trellis.harness.tools import toolbox
from trellis.harness.tools.policy import Tier, tier
from trellis.harness.tools.toolbox import Published, Toolbox
from trellis.harness.writes import Writes
from trellis.memory.errors import ValidationError

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
    def __init__(self, defs: list[Def]) -> None:
        self.defs = defs
        self.executed: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []

    async def tools(self) -> list[Def]:
        return self.defs

    async def execute(
        self, name: str, args: dict[str, Any], *, clients: Any, parent_request_id: Any = None
    ) -> Any:
        self.executed.append((name, args, tuple(clients)))
        return {"ok": name}


class Catalog:
    """The memory service's catalog as the toolbox reads it: conditional on an ETag (the
    entries' version), or down."""

    def __init__(self, entries: dict[str, Governance] | None = None) -> None:
        self.entries = entries or {}
        self.asked: list[list[str]] = []
        self.etags_sent: list[str | None] = []
        self.published: list[list[dict[str, Any]]] = []
        self.version = 1
        self.down = False
        self.refuse_publish = 0

    async def catalog(
        self, names: list[str], *, etag: str | None = None
    ) -> tuple[dict[str, Governance] | None, str | None]:
        self.asked.append(list(names))
        self.etags_sent.append(etag)
        if self.down:
            raise ConnectionError("memory is down")
        current = f'"v{self.version}"'
        if etag == current:
            return None, current
        return {n: e for n, e in self.entries.items() if n in names}, current

    def record(self, op: str, **args: Any) -> dict[str, Any]:
        return {"op": op, "scope": {}, "args": args}

    async def publish_catalog(self, entries: list[dict[str, Any]]) -> None:
        if self.refuse_publish:
            self.refuse_publish -= 1
            raise ValidationError("not stored", retryable=False)  # not retried by the writes
        self.published.append(list(entries))


async def resolved(
    defs: list[Def], catalog: Catalog | None = None, sources: list[Any] | None = None
) -> tuple[dict[str, Any], Gateway, Writes]:
    gateway, writes = Gateway(defs), Writes()
    tools = await toolbox.resolve(
        sources or [],
        gateway=gateway,  # type: ignore[arg-type]
        catalog=catalog,  # type: ignore[arg-type]
        writes=writes,
        published=Published(),
    )
    await writes.drain()
    return {t.name: t for t in tools}, gateway, writes


async def test_mcp_tools_are_tiered_by_their_annotations() -> None:
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


async def test_the_catalog_overrides_the_tier_and_sets_the_approval_rule() -> None:
    catalog = Catalog(
        {
            "erp-get_stock": Governance(risk="irreversible"),
            "erp-create_po": Governance(risk="write", approve_when="amount > 10000"),
        }
    )
    tools, _, _ = await resolved(
        [Def("erp-get_stock", "erp"), Def("erp-create_po", "erp", annotations=None)], catalog
    )
    assert tools["erp-get_stock"].side_effects == "irreversible"
    po = tools["erp-create_po"]
    assert tier(po, {"amount": 20000})[0] is Tier.ASK
    assert tier(po, {"amount": 5})[0] is Tier.NOTIFY
    assert catalog.asked == [["erp-get_stock", "erp-create_po"]]


async def test_every_tool_is_published_once_mcp_with_annotations_local_with_side_effects() -> None:
    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return order

    catalog = Catalog({"erp-get_stock": Governance(risk="write")})
    gateway, writes, published = Gateway([Def("erp-get_stock", "erp")]), Writes(), Published()
    for _ in range(2):  # a second resolve publishes nothing new
        await toolbox.resolve(
            [refund],
            gateway=gateway,  # type: ignore[arg-type]
            catalog=catalog,  # type: ignore[arg-type]
            writes=writes,
            published=published,
        )
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
    assert all(t.code_mode for t in tools.values())


async def test_three_read_only_servers_go_to_code_mode_scoped_to_them_others_stay_normal() -> None:
    defs = [
        Def("a-x", "a"),
        Def("b-y", "b"),
        Def("c-z", "c"),
        Def("erp-create_po", "erp", annotations=None),
    ]
    tools, gateway, _ = await resolved(defs)
    assert "erp-create_po" in tools and not tools["erp-create_po"].code_mode
    await tools["executeToolCode"].run({"code": "print(1)"})
    assert gateway.executed == [("executeToolCode", {"code": "print(1)"}, ("a", "b", "c"))]


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
    assert not any(t.code_mode for t in tools.values())
    assert len(tools) == len(defs)


async def test_an_approval_rule_keeps_a_server_out_of_code_mode() -> None:
    catalog = Catalog({"wiki-t0": Governance(risk="read", approve_when="true")})
    tools, _, _ = await resolved([Def(f"wiki-t{i}", "wiki") for i in range(20)], catalog)
    assert not any(t.code_mode for t in tools.values())


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
    monkeypatch.setattr(toolbox, "_now", clock)
    return clock


def box(defs: list[Def], catalog: Catalog | None) -> tuple[Toolbox, Gateway, Writes]:
    gateway, writes = Gateway(defs), Writes()
    made = Toolbox(
        [],
        gateway=gateway,  # type: ignore[arg-type]
        catalog=catalog,  # type: ignore[arg-type]
        writes=writes,
        published=Published(),
    )
    return made, gateway, writes


class CountingGateway(Gateway):
    def __init__(self, defs: list[Def]) -> None:
        super().__init__(defs)
        self.listed = 0

    async def tools(self) -> list[Def]:
        self.listed += 1
        return self.defs


async def test_governance_is_read_again_within_seconds_conditionally(clock: Clock) -> None:
    catalog = Catalog({"erp-create_po": Governance(risk="write")})
    made, _, writes = box([Def("erp-create_po", "erp", annotations=None)], catalog)
    first = await made.tools()
    assert first[0].approve_when is None
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS / 2
    assert await made.tools() == first  # fresh: not asked again
    assert len(catalog.asked) == 1
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS
    assert await made.tools() == first  # asked with its ETag: 304, the same tools
    assert catalog.etags_sent == [None, '"v1"']
    # an administrator adds a rule: the next read (seconds later, not minutes) has it
    catalog.entries["erp-create_po"] = Governance(risk="write", approve_when="amount > 10")
    catalog.version = 2
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS + 1
    [po] = await made.tools()
    assert po.approve_when == "amount > 10"
    await writes.aclose()


async def test_definitions_are_listed_again_after_their_own_longer_ttl(clock: Clock) -> None:
    catalog = Catalog()
    gateway, writes = CountingGateway([Def("erp-get", "erp")]), Writes()
    made = Toolbox(
        [],
        gateway=gateway,  # type: ignore[arg-type]
        catalog=catalog,  # type: ignore[arg-type]
        writes=writes,
        published=Published(),
    )
    await made.tools()
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS + 1
    await made.tools()
    assert gateway.listed == 1 and len(catalog.asked) == 2  # governance only
    clock.now += toolbox.TOOLS_TTL_SECONDS
    await made.tools()
    assert gateway.listed == 2
    assert catalog.etags_sent[-1] is None  # a new listing: a full read
    await writes.aclose()


async def test_concurrent_cold_runs_share_one_read() -> None:
    catalog = Catalog()
    gateway, writes = CountingGateway([Def("erp-get", "erp")]), Writes()
    made = Toolbox(
        [],
        gateway=gateway,  # type: ignore[arg-type]
        catalog=catalog,  # type: ignore[arg-type]
        writes=writes,
        published=Published(),
    )
    import asyncio

    results = await asyncio.gather(*(made.tools() for _ in range(10)))
    assert all(r == results[0] for r in results)
    assert gateway.listed == 1 and len(catalog.asked) == 1
    await writes.aclose()


async def test_an_unreadable_catalog_makes_every_tool_that_does_more_than_read_ask(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    @tool(side_effects="write")
    def note(text: str) -> str:
        """Write a note."""
        return text

    catalog = Catalog()
    catalog.down = True
    gateway, writes = (
        Gateway([Def("erp-get", "erp"), Def("erp-po", "erp", annotations=None)]),
        Writes(),
    )
    made = Toolbox(
        [note],
        gateway=gateway,  # type: ignore[arg-type]
        catalog=catalog,  # type: ignore[arg-type]
        writes=writes,
        published=Published(),
    )
    with caplog.at_level("WARNING", logger="trellis.tools"):
        tools = {t.name: t for t in await made.tools()}
        clock.now += toolbox.GOVERNANCE_TTL_SECONDS + 1
        await made.tools()  # still down: asked again, warned once
    assert caplog.text.count("the tool catalog could not be read") == 1
    assert len(catalog.asked) == 2
    assert tools["erp-get"].approve_when is None and tier(tools["erp-get"], {})[0] is Tier.AUTO
    for name in ("erp-po", "note"):
        chosen, why = tier(tools[name], {})
        assert chosen is Tier.ASK and "could not be read" in why
    catalog.down = False
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS + 1
    with caplog.at_level("INFO", logger="trellis.tools"):
        tools = {t.name: t for t in await made.tools()}
    assert "can be read again" in caplog.text
    assert tier(tools["erp-po"], {})[0] is Tier.NOTIFY  # back to its own tier
    await writes.aclose()


async def test_governance_read_earlier_stands_for_a_while_when_the_catalog_goes_down(
    clock: Clock,
) -> None:
    catalog = Catalog({"erp-po": Governance(risk="write", approve_when="amount > 10")})
    made, _, writes = box([Def("erp-po", "erp", annotations=None)], catalog)
    await made.tools()
    catalog.down = True
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS + 1
    [still] = await made.tools()
    assert still.approve_when == "amount > 10"  # the rule read 31 s ago stands
    clock.now += toolbox.GOVERNANCE_STALE_SECONDS
    [unread] = await made.tools()
    assert tier(unread, {"amount": 1})[0] is Tier.ASK  # too old to trust: it asks
    catalog.down = False
    clock.now += toolbox.GOVERNANCE_TTL_SECONDS + 1
    [back] = await made.tools()
    assert back.approve_when == "amount > 10" and catalog.etags_sent[-1] is None
    await writes.aclose()


async def test_a_failed_publish_is_sent_again_at_the_next_listing(clock: Clock) -> None:
    catalog = Catalog()
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
    catalog = Catalog()
    made, _, writes = box([], catalog)
    assert await made.tools() == []
    assert catalog.asked == []
    await writes.aclose()
