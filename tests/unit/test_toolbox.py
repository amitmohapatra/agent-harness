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
    def __init__(self, entries: dict[str, Governance] | None = None) -> None:
        self.entries = entries or {}
        self.asked: list[list[str]] = []
        self.published: list[list[dict[str, Any]]] = []

    async def catalog(self, names: list[str]) -> dict[str, Governance]:
        self.asked.append(list(names))
        return {n: e for n, e in self.entries.items() if n in names}

    def record(self, op: str, **args: Any) -> dict[str, Any]:
        return {"op": op, "scope": {}, "args": args}

    async def publish_catalog(self, entries: list[dict[str, Any]]) -> None:
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
        published=set(),
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
    gateway, writes, published = Gateway([Def("erp-get_stock", "erp")]), Writes(), set()
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
