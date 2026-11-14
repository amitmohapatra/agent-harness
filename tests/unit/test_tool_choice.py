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
from trellis.harness.tools.convert import claude, openai_agents
from trellis.harness.tools.convert import langchain as lc
from trellis.harness.tools.lint import Finding, lint
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


def test_each_name_google_and_numpy_give_the_arguments_section_is_read() -> None:
    for header in ("Args", "Arguments", "Parameters", "Params", "Keyword Args"):
        assert documented(f"Do it.\n\n{header}:\n    x: the x") == ("Do it.", {"x": "the x"})


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
    assert (
        tool(
            lambda: 1, name="given", description="Said here.", side_effects="read"
        ).spec.description
        == "Said here."
    )


def test_annotations_resolve_where_the_function_is_and_as_written_otherwise() -> None:
    def later(x: int) -> Missing:  # type: ignore[name-defined]  # noqa: F821
        """A return type nobody can resolve: the parameters as written still work."""
        return x

    made = tool(later, side_effects="read")
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
        await resolved(
            [Def("erp-lookup", "erp")],
            sources=[tool(lambda: 1, name="erp-lookup", side_effects="read")],
        )


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


def test_two_tools_sharing_three_quarters_of_their_words_are_said_alike() -> None:
    same = spec("a", "Reads the ledger totals.")
    alike = spec("b", "Reads the ledger lines.")  # 2 shared of 4 in all with a: 50%
    three = spec("c", "Reads ledger totals now.")  # 3 shared of 4 in all with a: 75%
    assert "overlapping_descriptions" in [f.code for f in lint([same, three])]
    assert "overlapping_descriptions" not in [f.code for f in lint([same, alike])]


def test_too_many_tools_at_once() -> None:
    many = [
        spec(f"t{n}", f"Report number {n} of the ledger, its totals and its lines.")
        for n in range(20)  # 20 at once: OpenAI's guide suggests fewer
    ]
    # ...described alike: only the set's finding is asserted here
    assert lint(many)[-1].code == "many_tools"
    assert "many_tools" not in [f.code for f in lint(many[:-1])]
    most = [spec(f"t{n}") for n in range(129)]  # over 128: the most the OpenAI API takes
    assert "too_many_tools" not in [f.code for f in lint(most[:-1])]
    found = lint(most)
    assert found[-1] == Finding(None, "too_many_tools", found[-1].message)
    assert str(found[-1]) == found[-1].message


async def test_the_harness_own_tools_pass_the_check() -> None:
    from trellis.harness.schedules import SPECS

    sandboxed = await sandbox().tools().resolve()
    assert lint([*CODE_MODE_TOOLS, *(t.spec for t in sandboxed)]) == []
    assert lint(SPECS) == []  # the schedule tools: each parameter described

    def typed(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            name: (
                schema.get("type"),
                typed(schema["items"].get("properties", {}))
                if schema.get("type") == "array" and schema.get("items", {}).get("type") == "object"
                else schema.get("items", {}).get("type"),
            )
            for name, schema in properties.items()
        }

    shapes = {spec.name: typed((spec.input_schema or {})["properties"]) for spec in SPECS}
    task = {"task": ("string", None), "when": ("string", None), "condition": ("string", None)}
    assert shapes == {
        "schedule_task": {
            **task,
            "once": ("boolean", None),
            "in_minutes": ("number", None),
            "choices": ("array", task),
            "timezone": ("string", None),
            "tools": ("array", "string"),
        },
        "list_schedules": {},
        "update_schedule": {
            "id": ("string", None),
            "ids": ("array", "string"),
            **task,
            "enabled": ("boolean", None),
        },
        "cancel_schedule": {"id": ("string", None), "ids": ("array", "string")},
    }
    described = {
        (spec.name, name)
        for spec in SPECS
        for name, schema in (spec.input_schema or {})["properties"].items()
        if schema.get("description")
    }
    assert {("update_schedule", "id"), ("update_schedule", "when")} <= described
    choice = SPECS[0].input_schema["properties"]["choices"]["items"]["properties"]  # type: ignore[index]
    assert all(schema.get("description") for schema in choice.values())


def _undescribed(schema: dict[str, Any], at: str = "") -> list[str]:
    """Each parameter of ``schema`` with no description, nested ones (an array's objects')
    included, by its path."""
    found = []
    for name, field in (schema.get("properties") or {}).items():
        if not field.get("description"):
            found.append(f"{at}{name}")
        items = field.get("items") or {}
        if items.get("type") == "object":
            found += _undescribed(items, f"{at}{name}[].")
    return found


async def test_every_parameter_of_every_tool_the_harness_offers_is_described(tmp_path: Any) -> None:
    """The harness's own tools - schedules, Code Mode, the sandbox, skills, asking, the kept
    results' readers, a sub-agent, a remote agent, an MCP server's resources - describe each
    parameter, so a model knows what to give (the tool set check's ``undescribed_parameter``)."""
    from trellis.harness import skills, subagents
    from trellis.harness.a2a import client as a2a_client
    from trellis.harness.schedules import SPECS
    from trellis.harness.tools import query, resources
    from trellis.harness.tools import results as results_module
    from trellis.harness.tools.ask import ask_user

    specs = [*SPECS, *CODE_MODE_TOOLS]
    specs += [t.spec for t in await sandbox().tools().resolve()]
    specs += [t.spec for t in await skills.directory(tmp_path).tools().resolve()]
    specs += [ask_user().spec]
    schemas = {spec.name: spec.input_schema or {} for spec in specs}
    schemas |= {
        "read_file": results_module.READ_FILE_SCHEMA,
        "query_result": query.QUERY_RESULT_SCHEMA,
        "query_result (saving)": query.QUERY_RESULT_SAVING_SCHEMA,
        "a sub-agent": subagents.INPUT,
        "a remote agent": a2a_client.SCHEMA,
        "read_resource": resources.SCHEMA,
    }
    missing = {name: found for name, schema in schemas.items() if (found := _undescribed(schema))}
    assert missing == {}
    assert len(schemas) >= 20


async def test_the_toolbox_logs_what_the_check_found_when_its_tools_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    terse = Def("erp-get", "erp", annotations=READ)
    with caplog.at_level(logging.WARNING, logger="trellis.tools"):
        await resolved(
            [terse], sources=[tool(lambda: 1, name="mine", description="", side_effects="read")]
        )
    said = [r.getMessage() for r in caplog.records if "tool choice" in r.getMessage()]
    assert said == [
        "tool choice: mine: no description: say what it is for, and when",
        "tool choice: erp-get: the description is 2 words: say what it is for, and when to "
        "use it rather than a similar tool",
    ]
