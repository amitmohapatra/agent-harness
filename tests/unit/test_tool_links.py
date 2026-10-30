"""Tool chains: which tool's output feeds which tool's input, read from the schemas the tools
publish (a function's return annotation, an MCP tool's ``outputSchema``) — data, not rules —
and said in the descriptions the model reads, whatever the framework."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel

from tests.unit.test_mcp_source import FakeClient, connected
from tests.unit.test_toolbox import resolved
from trellis import tool
from trellis.contracts import ToolSpec
from trellis.harness.tools.links import Link, chained, links, related
from trellis.harness.tools.mcp import mcp, stdio


class Sku(TypedDict):
    sku_id: str
    name: str


class Stock(BaseModel):
    sku_id: str
    on_hand: int
    supplier_id: str


class LeadTime(BaseModel):
    supplier_id: str
    lead_time_days: int


@tool(side_effects="read")
def find_sku(name: str) -> Sku:
    """Find a SKU by its product name.

    Args:
        name: The product's name, such as "blue mug".
    """
    return {"sku_id": "SKU-7", "name": name}


@tool(side_effects="read")
def get_stock(sku_id: str) -> Stock:
    """The units of a SKU on hand, and who supplies it.

    Args:
        sku_id: The SKU's id, such as SKU-7.
    """
    return Stock(sku_id=sku_id, on_hand=4, supplier_id="SUP-2")


@tool(side_effects="read")
def supplier_lead_time(supplier_id: str) -> LeadTime:
    """How many days a supplier takes to deliver.

    Args:
        supplier_id: The supplier's id, such as SUP-2.
    """
    return LeadTime(supplier_id=supplier_id, lead_time_days=9)


@tool(side_effects="read")
def reorder_quantity(on_hand: int, lead_time_days: int) -> int:
    """The units to reorder so stock lasts the supplier's lead time.

    Args:
        on_hand: Units on hand now.
        lead_time_days: Days the supplier takes to deliver.
    """
    return max(0, 3 * lead_time_days - on_hand)


@tool(side_effects="read")
def search_notes(name: str) -> dict[str, Any]:
    """Search the notes by a name."""
    return {"name": name}


def test_a_functions_return_annotation_is_its_output_schema() -> None:
    found, stock = find_sku.spec.output_schema, get_stock.spec.output_schema
    assert found is not None and stock is not None
    assert set(found["properties"]) == {"sku_id", "name"}
    assert set(stock["properties"]) == {"sku_id", "on_hand", "supplier_id"}
    assert reorder_quantity.spec.output_schema == {"type": "integer"}

    @tool
    def untyped(x: int):  # type: ignore[no-untyped-def]
        return x

    assert untyped.spec.output_schema is None  # nothing declared: nothing assumed


def test_a_three_hop_chain_is_read_from_the_schemas() -> None:
    specs = [t.spec for t in (reorder_quantity, supplier_lead_time, get_stock, find_sku)]
    found = links(specs)
    assert Link("find_sku", "sku_id", "get_stock", "sku_id") in found
    assert Link("get_stock", "supplier_id", "supplier_lead_time", "supplier_id") in found
    assert Link("get_stock", "on_hand", "reorder_quantity", "on_hand") in found
    assert Link("supplier_lead_time", "lead_time_days", "reorder_quantity", "lead_time_days") in (
        found
    )
    # a tool feeds no argument of its own, and a generic field ("name") links nothing
    assert all(link.source != link.target for link in found)
    assert not any(link.field == "name" for link in found)


def test_an_id_field_feeds_the_argument_named_for_the_tools_noun() -> None:
    """MCP servers often answer ``{"id": ...}``: ``find_sku``'s id is a ``sku_id``."""
    finder = ToolSpec(
        name="find_sku",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    stock = ToolSpec(
        name="get_stock",
        input_schema={"type": "object", "properties": {"sku_id": {"type": "string"}}},
    )
    order = ToolSpec(
        name="get_order",
        input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
    )
    assert links([finder, stock, order]) == [Link("find_sku", "id", "get_stock", "sku_id")]


def test_fields_of_another_type_or_in_a_list_of_records() -> None:
    lister = ToolSpec(
        name="list_skus",
        output_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"sku_id": {"type": "string"}}},
                }
            },
        },
    )
    by_number = ToolSpec(
        name="stock_by_number",
        input_schema={"type": "object", "properties": {"sku_id": {"type": "integer"}}},
    )
    by_text = ToolSpec(
        name="stock_by_text",
        input_schema={"type": "object", "properties": {"sku_id": {"type": "string"}}},
    )
    assert links([lister, by_number, by_text]) == [
        Link("list_skus", "items[].sku_id", "stock_by_text", "sku_id")
    ]


def test_each_tool_says_its_links_in_its_description() -> None:
    specs = [t.spec for t in (find_sku, get_stock, supplier_lead_time, search_notes)]
    described = {s.name: s.description for s in chained(specs)}
    assert "find_sku → sku_id → get_stock" in described["get_stock"]
    assert "find_sku → sku_id → get_stock" in described["find_sku"]
    assert "get_stock → supplier_id → supplier_lead_time" in described["get_stock"]
    assert described["search_notes"] == search_notes.spec.description  # no link: unchanged
    again = {s.name: s.description for s in chained(chained(specs))}
    assert again == described  # said once


def test_a_candidates_chain_is_the_tools_it_feeds_and_those_it_needs() -> None:
    specs = [t.spec for t in (find_sku, get_stock, supplier_lead_time, reorder_quantity)]
    required = {s.name: (s.input_schema or {}).get("required", []) for s in specs}
    found = links(specs)
    assert related({"find_sku"}, found, required) == {
        "get_stock",
        "supplier_lead_time",
        "reorder_quantity",
    }
    # its required supplier_id comes from get_stock, whose sku_id comes from find_sku
    assert related({"supplier_lead_time"}, found, required) == {
        "get_stock",
        "find_sku",
        "reorder_quantity",
    }
    assert related({"find_sku"}, [], required) == set()


async def test_the_toolbox_offers_every_tool_with_its_links() -> None:
    tools, _, _ = await resolved([], sources=[find_sku, get_stock, search_notes])
    assert "find_sku → sku_id → get_stock" in tools["get_stock"].spec.description
    assert tools["search_notes"].spec.description == search_notes.spec.description
    assert await tools["get_stock"].run({"sku_id": "SKU-7"}) == Stock(
        sku_id="SKU-7", on_hand=4, supplier_id="SUP-2"
    )


async def test_an_mcp_tools_output_schema_is_kept() -> None:
    schema = {"type": "object", "properties": {"sku_id": {"type": "string"}}}

    class Listing(FakeClient):
        async def list_tools(self, cursor: Any = None) -> Any:
            from types import SimpleNamespace

            listed = SimpleNamespace(
                name="find_sku",
                description="Find a SKU.",
                title=None,
                input_schema=None,
                output_schema=schema,
                annotations=None,
            )
            return SimpleNamespace(tools=[listed], next_cursor=None)

    tools = await connected(mcp(stdio("srv")), Listing()).resolve()
    assert tools[0].spec.output_schema == schema


class Opaque:
    """A type pydantic has no schema for."""


def test_an_output_schema_only_where_a_type_says_one() -> None:
    from trellis.harness.tools.sources import output_schema

    def unresolved() -> Nowhere:  # type: ignore[name-defined]  # noqa: F821
        return None

    def opaque() -> Opaque:
        return Opaque()

    def nothing() -> None:
        return None

    assert output_schema(unresolved) is None  # a name it cannot resolve: nothing assumed
    assert output_schema(opaque) is None
    assert output_schema(nothing) is None


class Supplier(BaseModel):
    id: str
    country: str | None = None


class Part(BaseModel):
    part_id: str
    supplier: Supplier | None = None


def test_nested_records_references_and_nullable_types() -> None:
    """A pydantic model's nested record (``$defs`` and ``$ref``, ``anyOf`` with null): its
    ``id`` is named by the record (``supplier.id`` is a ``supplier_id``); a type list with
    null is its other type."""

    @tool
    def get_part(part_id: str) -> Part:
        """A part."""
        return Part(part_id=part_id)

    lead = ToolSpec(
        name="lead_time",
        input_schema={
            "type": "object",
            "properties": {"supplier_id": {"type": ["string", "null"]}},
            "required": ["supplier_id"],
        },
    )
    parts = ToolSpec(
        name="parts_by_number",
        input_schema={"type": "object", "properties": {"part_id": {"$ref": "#/$defs/Nowhere"}}},
    )
    found = links([get_part.spec, lead, parts])
    assert Link("get_part", "supplier.id", "lead_time", "supplier_id") in found
    assert Link("get_part", "part_id", "parts_by_number", "part_id") in found
    # parts_by_number's part_id is optional: get_part is not needed before it
    assert related({"parts_by_number"}, found, {"parts_by_number": []}) == set()
    assert related({"lead_time"}, found, {"lead_time": ["supplier_id"]}) == {"get_part"}
    assert Link("get_part", "supplier.id", "lead_time", "supplier_id").said() == (
        "get_part → id as supplier_id → lead_time"
    )


def test_a_number_feeds_a_number_and_nothing_feeds_from_no_schema() -> None:
    counted = ToolSpec(
        name="count_units",
        output_schema={"type": "object", "properties": {"units": {"type": "integer"}}},
    )
    priced = ToolSpec(
        name="price_units",
        input_schema={"type": "object", "properties": {"units": {"type": "number"}}},
    )
    assert links([counted, priced]) == [Link("count_units", "units", "price_units", "units")]
    assert links([ToolSpec(name="a"), ToolSpec(name="b")]) == []


def test_a_schema_deeper_than_its_bound_is_read_to_the_bound() -> None:
    schema: dict[str, Any] = {"type": "object", "properties": {"leaf_id": {"type": "string"}}}
    for level in range(8):
        schema = {"type": "object", "properties": {f"level{level}": schema}}
    deep = ToolSpec(name="deep", output_schema=schema)
    takes = ToolSpec(name="takes", input_schema={"type": "object", "properties": {"leaf_id": {}}})
    assert links([deep, takes]) == []  # past the bound: not read


async def test_the_toolbox_keeps_each_chain_for_the_hints() -> None:
    from tests.unit.test_toolbox import box

    made, _, writes = box([], sources=[find_sku, get_stock, search_notes])
    assert made.related({"find_sku"}) == set()  # nothing built yet
    await made.tools()
    await writes.drain()
    assert made.related({"find_sku"}) == {"get_stock"}
    assert made.related({"get_stock"}) == {"find_sku"}  # its required sku_id
    assert made.related({"search_notes"}) == set()
