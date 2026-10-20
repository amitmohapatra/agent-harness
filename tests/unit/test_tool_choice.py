"""What a model chooses its tools by, whatever their source: a function's docstring split into
its description and its parameters' (the "use it when" paragraphs kept), MCP definitions kept
as the server gave them in every framework's format, names kept apart, and the check of a tool
set (``tools.lint``) — and the toolbox logging what the check found."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from tests.unit.test_toolbox import READ, Def, resolved
from trellis import tool
from trellis.contracts import ConfigurationError, ToolSpec
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS
from trellis.harness.sandbox import sandbox
from trellis.harness.skills import Skills
from trellis.harness.tools.convert import claude, openai_agents
from trellis.harness.tools.convert import langchain as lc
from trellis.harness.tools.lint import MANY_TOOLS, MOST_TOOLS, Finding, lint
from trellis.harness.tools.sources import documented


# --------------------------------------------------------------------------- docstrings
def test_a_google_docstring_is_a_description_and_the_parameters_descriptions() -> None:
    doc = """Lost sales by style.

    Use for stock-outs. Not for excess stock: use excess_stock.

    Args:
        store (str): The store's id,
            such as S12.
        *rest: Ignored.

    Returns:
        A list.

    The last paragraph stays."""
    import inspect

    text, params = documented(inspect.cleandoc(doc))
    assert text == (
        "Lost sales by style.\n\nUse for stock-outs. Not for excess stock: use excess_stock."
        "\n\nThe last paragraph stays."
    )
    assert params == {"store": "The store's id, such as S12.", "rest": "Ignored."}


def test_sphinx_fields_and_a_docstring_without_sections() -> None:
    text, params = documented(
        "Do it.\n\n:param x: the x\n    and more\n:type x: int\n:returns: y\nAfter."
    )
    assert (text, params) == ("Do it.\n\nAfter.", {"x": "the x and more"})
    assert documented("") == ("", {})
    assert documented("One line.") == ("One line.", {})
    # an Args section whose first line is not an entry, then text that goes on
    assert documented("A.\nArgs:\n    free text\n    b: B.\nNext.") == ("A.\nNext.", {"b": "B."})


def test_a_function_tool_keeps_every_paragraph_and_describes_its_parameters() -> None:
    @tool(side_effects="read")
    def lost_sales(
        store: str, weeks: Annotated[int, Field(description="Weeks back, 1 to 52.")] = 4
    ) -> list[Any]:
        """Lost sales by style.

        Use for stock-outs, not for an overview.

        Args:
            store: The store's id.
            weeks: Overridden by the annotation.
        """
        return []

    assert lost_sales.spec.description == (
        "Lost sales by style.\n\nUse for stock-outs, not for an overview."
    )
    properties = lost_sales.spec.input_schema["properties"]  # type: ignore[index]
    assert properties["store"]["description"] == "The store's id."
    assert properties["weeks"]["description"] == "Weeks back, 1 to 52."
    assert tool(lambda: 1, name="given", description="Said here.").spec.description == "Said here."


def test_annotations_resolve_where_the_function_is_and_as_written_otherwise() -> None:
    def later(x: int) -> Missing:  # type: ignore[name-defined]  # noqa: F821
        """A return type nobody can resolve: the parameters as written still work."""
        return x

    made = tool(later)
    assert made.spec.input_schema["properties"]["x"]["type"] == "integer"  # type: ignore[index]


# --------------------------------------------------------------------------- every format
SCHEMA = {
    "type": "object",
    "properties": {
        "store": {"type": "string", "description": "The store's id."},
        "metric": {"type": "string", "enum": ["units", "revenue"], "description": "Which."},
        "filters": {
            "type": "object",
            "description": "Narrow it down.",
            "properties": {"style": {"type": "string", "description": "A style id."}},
        },
    },
    "required": ["store"],
}


async def test_mcp_definitions_reach_every_framework_as_the_server_gave_them() -> None:
    d = Def("erp-lost_sales", "erp")
    d.description, d.parameters = "Lost sales by style. Not for an overview.", SCHEMA
    tools, _, _ = await resolved([d])
    found = tools["erp-lost_sales"]
    assert (found.spec.description, found.spec.input_schema) == (d.description, SCHEMA)
    [made] = lc.convert([found])
    sent = convert_to_openai_tool(made)["function"]
    assert (sent["name"], sent["description"], sent["parameters"]) == (
        "erp-lost_sales",
        d.description,
        SCHEMA,
    )
    agents = openai_agents.convert([found])[0]
    assert (agents.description, agents.params_json_schema) == (d.description, SCHEMA)
    assert claude._schema(found)["properties"] == SCHEMA["properties"]


async def test_the_same_tool_on_two_servers_is_two_names_and_a_clash_is_refused() -> None:
    tools, _, _ = await resolved([Def("erp-lookup", "erp"), Def("crm-lookup", "crm")])
    assert sorted(tools) == ["crm-lookup", "erp-lookup"]
    with pytest.raises(ConfigurationError, match="two tools are named erp-lookup"):
        await resolved([Def("erp-lookup", "erp")], sources=[tool(lambda: 1, name="erp-lookup")])


# --------------------------------------------------------------------------- lint
def spec(
    name: str, description: str = "Reads the lost sales of one store by style.", **kw: Any
) -> ToolSpec:
    return ToolSpec(name=name, description=description, **kw)


def codes(found: list[Finding]) -> list[tuple[str | None, str]]:
    return [(f.tool, f.code) for f in found]


def test_a_clean_tool_set_has_no_findings() -> None:
    assert (
        lint(
            [
                spec("lost_sales", input_schema=SCHEMA),
                spec("app_help", "Where in the app to do something."),
            ]
        )
        == []
    )
    assert lint([]) == []


def test_each_tool_is_checked_for_what_a_model_chooses_by() -> None:
    found = lint(
        [
            spec("bad name!"),
            spec("empty", ""),
            spec("terse", "Gets data."),
            spec("args", input_schema={"properties": {"a": {"type": "string"}, "b": {}}}),
        ]
    )
    assert codes(found) == [
        ("bad name!", "invalid_name"),
        ("empty", "no_description"),
        ("terse", "short_description"),
        ("args", "undescribed_parameter"),
        ("args", "same_description"),
    ]
    assert "a, b" in found[3].message
    assert str(found[0]).startswith("bad name!: ")
    assert lint([spec("x" * 65)])[0].code == "invalid_name"


def test_pairs_a_model_cannot_tell_apart() -> None:
    assert codes(
        lint([spec("lost_sales"), spec("Lost-Sales", "Something else entirely, said well.")])
    ) == [("Lost-Sales", "similar_names")]
    overview = "A store's dashboard: sales, stock, excess stock and lost sales at a glance."
    alike = "A store's dashboard: sales, stock, excess stock and lost sales for the week."
    assert codes(lint([spec("overview", overview), spec("dashboard", alike)])) == [
        ("dashboard", "overlapping_descriptions")
    ]
    assert (
        lint(
            [
                spec("a", "Lost sales by style, one store."),
                spec("b", "Excess stock by style, per region."),
            ]
        )
        == []
    )


def test_too_many_tools_at_once() -> None:
    many = [
        spec(f"t{n}", f"Report number {n} of the ledger, its totals and its lines.")
        for n in range(MANY_TOOLS)
    ]
    # ...described alike: only the set's finding is asserted here
    assert lint(many)[-1].code == "many_tools"
    assert "many_tools" not in [f.code for f in lint(many[:-1])]
    most = [spec(f"t{n}") for n in range(MOST_TOOLS + 1)]
    found = lint(most)
    assert found[-1] == Finding(None, "too_many_tools", found[-1].message)
    assert str(found[-1]) == found[-1].message


async def test_the_harness_own_tools_pass_the_check() -> None:
    skills = await Skills(["triage"]).resolve()
    sandboxed = await sandbox().resolve()
    assert lint([*CODE_MODE_TOOLS, *(t.spec for t in [*skills, *sandboxed])]) == []


async def test_the_toolbox_logs_what_the_check_found_when_its_tools_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    terse = Def("erp-get", "erp", annotations=READ)
    with caplog.at_level(logging.WARNING, logger="trellis.tools"):
        await resolved([terse], sources=[tool(lambda: 1, name="mine", description="")])
    said = [r.getMessage() for r in caplog.records if "tool choice" in r.getMessage()]
    assert said == [
        "tool choice: mine: no description: say what it is for, and when",
        "tool choice: erp-get: the description is 2 words: say what it is for, and when to "
        "use it rather than a similar tool",
    ]
