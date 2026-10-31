"""Values that change (the SKU scenario's stock): a tool says how long what it returns stays
current, as data beside its other metadata (``tool(freshness="live")``, ``max_age=``, a
LangChain tool's ``metadata``, an MCP tool's annotations); every tool result the conversation
replays carries when it was fetched; and a live tool's earlier result past its max age is
marked out of date in the replay and in the context — the tool stays offered, and the model
fetches the value again instead of answering from the old one."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool

from tests.integration.test_memory_pipeline import many
from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import ConfigurationError, RunStatus, ToolSpec
from trellis.harness import snapshots
from trellis.harness.identity import conversation_key
from trellis.harness.middleware import spec_of
from trellis.harness.tools.base import MAX_AGE, _duration, data_of, max_age
from trellis.harness.tools.mcp import MCPSource

THREAD = "skus"
#: where the memory service keeps the conversation: the tenant's, the user's and the thread's
KEPT = conversation_key("acme", "u1", THREAD)
#: when the first turn fetched the stock
FETCHED = datetime(2026, 10, 30, 9, 0, 0, tzinfo=UTC)
STOCK = {"SKU-00123": [40, 7]}  # what each fetch of the SKU's stock answers, in turn


def stock_tool(**declared: Any) -> Any:
    answers = list(STOCK["SKU-00123"])

    @tool(side_effects="read", **declared)
    def get_stock(sku: str) -> int:
        """Units of a SKU in stock now."""
        calls.append(sku)
        return answers.pop(0)

    calls: list[str] = []
    get_stock.calls = calls  # type: ignore[attr-defined]
    return get_stock


def at(when: datetime) -> Any:
    return lambda: when


def replayed(request: dict[str, Any]) -> list[str]:
    """The tool results a request replays from earlier turns."""
    return [m["content"] for m in request["messages"] if m["role"] == "tool"]


def context(request: dict[str, Any]) -> str:
    """The memory context as sent: the first part of the latest question (audit K1)."""
    asked = [m for m in request["messages"] if m["role"] == "user"][-1]["content"]
    return asked[0]["text"] if isinstance(asked, list) else ""


# --------------------------------------------------------------------------- declared
def test_a_tool_declares_how_long_its_values_stay_current() -> None:
    live = tool(lambda sku: 1, name="live_stock", freshness="live", side_effects="read")
    aged = tool(lambda sku: 1, name="price", max_age=300, side_effects="read")
    plain = tool(lambda sku: 1, name="catalog", side_effects="read")
    static = tool(lambda sku: 1, name="country", freshness="static", side_effects="read")
    assert max_age(live.spec) == 0 and max_age(aged.spec) == 300 and max_age(static.spec) is None
    assert max_age(plain.spec) == 0  # live by default: a tool's data changes
    # the model reads it in the tool's description; a tool that says nothing is as it was
    assert "call it again for the current value" in live.tool.spec.description
    assert "5 minutes" in aged.tool.spec.description
    assert plain.tool.spec.description == plain.spec.description
    with pytest.raises(ConfigurationError):
        tool(lambda: 1, name="bad", freshness="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        tool(lambda: 1, name="bad", max_age=-1)


def test_a_framework_tool_and_an_mcp_tool_declare_it_beside_their_side_effects() -> None:
    @lc_tool
    def quote(symbol: str) -> float:
        """A stock quote."""
        return 1.0

    quote.metadata = {"side_effects": "read", "freshness": "live"}
    assert max_age(spec_of(quote)) == 0
    source = MCPSource("https://example.test/mcp", name="inv")
    from mcp.types import Tool as Listed
    from mcp.types import ToolAnnotations

    listed = Listed.model_validate(
        {"name": "stock", "description": "Stock.", "inputSchema": {"type": "object"}}
    )
    assert max_age(source._tool("inv", listed).spec) == 0  # said nothing: live
    listed = listed.model_copy(
        update={
            "annotations": ToolAnnotations.model_validate({"readOnlyHint": True}),
            "meta": {"maxAge": 60},  # MCP's _meta: what the protocol does not name
        }
    )
    made = source._tool("inv", listed).spec
    assert max_age(made) == 60 and made.side_effects == "read"


# --------------------------------------------------------------------------- replayed
async def test_a_replayed_result_says_which_tool_fetched_it_and_when(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_stock = stock_tool()
    model = ScriptedChat(
        [("get_stock", {"sku": "SKU-00123"}), "40 units.", "From get_stock, at 09:00 UTC."]
    )
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="inv"), [get_stock])
    monkeypatch.setattr(snapshots, "now", at(FETCHED))
    await agent.run("Stock of SKU-00123?", user="u1", thread=THREAD)
    monkeypatch.setattr(snapshots, "now", at(FETCHED + timedelta(minutes=5)))
    result = await agent.run("Where did that number come from?", user="u1", thread=THREAD)
    assert result.status is RunStatus.SUCCESS, result.error
    [told] = replayed(model.requests[-1])
    stored = [m for m in memory_service.threads[KEPT] if m["role"] == "TOOL"]
    # kept so: any later turn reads which tool fetched it and when
    assert stored[0]["content"] == "[snapshot of get_stock from 2026-10-30T09:00:00Z]\n40"
    assert told.startswith("[snapshot of get_stock from 2026-10-30T09:00:00Z: out of date")
    assert "out of date" in context(model.requests[-1])  # live by default: call it again


#: (what the tool declares, how much later the second turn is, whether the first value is stale)
CASES = {
    "live": ({"freshness": "live"}, timedelta(seconds=30), True),
    "past-max-age": ({"max_age": 600}, timedelta(hours=2), True),
    "within-max-age": ({"max_age": 600}, timedelta(minutes=5), False),
    "live-by-default": ({}, timedelta(seconds=1), True),
    "static": ({"freshness": "static"}, timedelta(days=30), False),
}


@pytest.mark.parametrize("case", list(CASES))
async def test_a_live_value_past_its_max_age_is_fetched_again(
    memory_harness: Harness, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    declared, later, stale = CASES[case]
    get_stock = stock_tool(**declared)
    model = ScriptedChat(
        [
            ("get_stock", {"sku": "SKU-00123"}),
            "40 units.",
            ("get_stock", {"sku": "SKU-00123"}),
            "7 units.",
        ]
    )
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="inv"), [get_stock])
    monkeypatch.setattr(snapshots, "now", at(FETCHED))
    await agent.run("Stock of SKU-00123?", user="u1", thread=THREAD)
    monkeypatch.setattr(snapshots, "now", at(FETCHED + later))
    result = await agent.run("And the current stock of SKU-00123?", user="u1", thread=THREAD)
    assert result.status is RunStatus.SUCCESS and result.answer == "7 units.", result.error
    asked = model.requests[2]  # the second turn's first request
    [told] = replayed(asked)
    assert told.startswith("[snapshot of get_stock from 2026-10-30T09:00:00Z")
    assert ("out of date" in told) is stale
    assert ("get_stock" in context(asked) and "out of date" in context(asked)) is stale
    assert "get_stock" in [t["function"]["name"] for t in asked["tools"]]  # still offered
    assert get_stock.calls == ["SKU-00123", "SKU-00123"]


class Annotated:
    """An SDK object keeping keys it does not know as pydantic extras."""

    def __init__(self, **extra: Any) -> None:
        self.model_extra = extra


class Slotted:
    __slots__ = ("x",)


@pytest.mark.parametrize(
    ("declared", "found"),
    [
        (None, {}),
        ({}, {}),
        ({"freshness": "live"}, {MAX_AGE: 0.0}),
        ({"max_age": 5}, {MAX_AGE: 5.0}),
        ({"maxAge": 7.5, "readOnlyHint": True}, {MAX_AGE: 7.5}),
        ({"freshness": None, "max_age": None}, {}),
        ({"freshness": "static", "paged": True}, {"freshness": "static", "paged": True}),
        (Annotated(maxAge=60), {MAX_AGE: 60.0}),
        (
            type("Attrs", (), {"__init__": lambda self: setattr(self, "freshness", "live")})(),
            {MAX_AGE: 0.0},
        ),
        (Slotted(), {}),
    ],
)
def test_freshness_is_read_from_a_mapping_attributes_or_extras(declared: Any, found: Any) -> None:
    assert data_of(declared) == found


@pytest.mark.parametrize("declared", [{"freshness": "hourly"}, {"max_age": -1}, {"max_age": True}])
def test_a_freshness_of_another_kind_is_refused_or_from_a_server_not_read(declared: Any) -> None:
    with pytest.raises(ConfigurationError, match="quote"):
        data_of(declared, name="quote")
    assert data_of(declared, strict=False) == {}


def test_a_spec_that_says_nothing_is_live_and_a_static_one_never_ages() -> None:
    assert max_age(ToolSpec(name="t")) == 0  # live by default
    assert max_age(ToolSpec(name="t", max_age=True)) == 0  # type: ignore[call-arg]
    assert max_age(ToolSpec(name="t", freshness="static")) is None  # type: ignore[call-arg]
    assert max_age(ToolSpec(name="t", max_age=3)) == 3  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (1, "1 second"),
        (45, "45 seconds"),
        (60, "1 minute"),
        (90, "90 seconds"),
        (300, "5 minutes"),
        (3_600, "1 hour"),
        (7_200, "2 hours"),
        (172_800, "2 days"),
    ],
)
def test_an_age_is_said_as_a_person_says_it(seconds: float, said: str) -> None:
    assert _duration(seconds) == said


async def test_a_live_tool_whose_value_is_out_of_date_stays_offered_when_hints_narrow(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool hints narrow the second turn to another tool: the live tool whose earlier value
    is out of date is offered all the same — the model can fetch it again."""
    get_stock = stock_tool(freshness="live")

    @tool(side_effects="read")
    def catalog(sku: str) -> str:
        """A SKU's catalogue entry."""
        return "a blue mug"

    model = ScriptedChat([("get_stock", {"sku": "SKU-00123"}), "40 units.", "fetching"])
    target = ReAct(system_prompt="s", model=model)
    # five tools and more: the memory service hints which fit (``HINTS_MIN_TOOLS``)
    agent = arm(memory_harness.wrap(target, id="inv-hinted"), [get_stock, catalog, *many(3)])
    monkeypatch.setattr(snapshots, "now", at(FETCHED))
    await agent.run("Stock of SKU-00123?", user="u1", thread=THREAD)
    memory_service.candidates = ["catalog"]  # the hints now point elsewhere
    monkeypatch.setattr(snapshots, "now", at(FETCHED + timedelta(hours=1)))
    await agent.run("What is SKU-00123, and its stock now?", user="u1", thread=THREAD)
    offered = [t["function"]["name"] for t in model.requests[-1]["tools"]]
    assert "catalog" in offered and "get_stock" in offered
    assert "t0" not in offered  # narrowed: the hinted tool, and the live one out of date
