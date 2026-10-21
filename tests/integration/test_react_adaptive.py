"""What ``ReAct`` adds to its loop on a run's signal (``trellis.harness.adaptive``): each piece
on its signal (positive), off without it (negative), at its edges, and off with
``without={<feature>}`` — and together."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from tests.support.models import Script, ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness import adaptive, telemetry
from trellis.harness.react import CLEARED
from trellis.harness.telemetry import LoopEvent


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    return 7


@tool(side_effects="write")
def reorder(sku: str, units: int) -> str:
    """Reorder units of a SKU.

    Args:
        sku: The SKU, such as a-1.
        units: How many units.
    """
    return f"ordered {units} of {sku}"


class Counted:
    """The loop events a test's runs counted."""

    def __init__(self) -> None:
        self.events: list[LoopEvent] = []

    def loop(self, event: LoopEvent) -> None:
        self.events.append(event)


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> Counted:
    found = Counted()
    monkeypatch.setattr(telemetry.metrics, "loop", found.loop)
    return found


def tools_of(model: ScriptedChat, request: int) -> list[str]:
    return [t["function"]["name"] for t in model.requests[request].get("tools") or []]


def broken(name: str, arguments: str, call: str = "c1") -> dict[str, Any]:
    """A model message whose one call's arguments are ``arguments`` as written."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


# --------------------------------------------------------------------------- tool repair
async def test_arguments_cut_short_are_repaired_instead_of_asked_again(
    harness: Harness, counted: Counted
) -> None:
    model = ScriptedChat([broken("stock", '{"sku": "a-1"'), "7 units"])
    agent = harness.wrap(ReAct(system="s", model=model), id="repair", tools=[stock])
    result = await agent.run("a-1?", user="u")
    assert result.answer == "7 units" and len(model.requests) == 2
    sent = model.requests[1]["messages"]
    assert json.loads(sent[-2]["tool_calls"][0]["function"]["arguments"]) == {"sku": "a-1"}
    assert sent[-1] == {"content": "7", "role": "tool", "tool_call_id": "c1"}
    assert counted.events.count(LoopEvent.REPAIRED) == 1
    assert counted.events.count(LoopEvent.MALFORMED_ARGS) == 1


async def test_a_misspelled_tool_name_is_matched_to_the_one_tool_it_means(
    harness: Harness, counted: Counted
) -> None:
    model = ScriptedChat([("Stock", {"sku": "a-1"}), "7 units"])
    agent = harness.wrap(ReAct(system="s", model=model), id="named", tools=[stock])
    assert (await agent.run("a-1?", user="u")).answer == "7 units"
    assert model.requests[1]["messages"][-1]["content"] == "7"
    assert counted.events == [LoopEvent.REPAIRED]


async def test_without_tool_repair_a_cut_short_call_is_told_and_asked_again(
    harness: Harness,
) -> None:
    model = ScriptedChat([broken("stock", '{"sku": "a-1"'), ("stock", {"sku": "a-1"}), "7"])
    agent = harness.wrap(
        ReAct(system="s", model=model), id="norepair", tools=[stock], without={"tool_repair"}
    )
    assert (await agent.run("a-1?", user="u")).answer == "7"
    assert model.requests[1]["messages"][-1]["content"].startswith("stock was not run")


@pytest.mark.parametrize(
    "arguments",
    [
        "{sku:",  # json_repair makes the value up (empty): no repair
        '{"units": 3',  # repaired, but the required sku is missing
        "[1, 2",  # not an object
    ],
)
async def test_a_repair_that_does_not_fit_the_schema_is_no_repair(
    harness: Harness, counted: Counted, arguments: str
) -> None:
    model = ScriptedChat([broken("stock", arguments), ("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="unfit", tools=[stock])
    assert (await agent.run("a?", user="u")).answer == "7"
    assert model.requests[1]["messages"][-1]["content"].startswith("stock was not run")
    assert LoopEvent.REPAIRED not in counted.events


def test_a_name_two_tools_can_mean_or_none_is_left_as_it_is() -> None:
    message = AIMessage(content="", tool_calls=[{"name": "a_b", "args": {}, "id": "1"}])
    tools: list[Any] = [
        {"type": "function", "function": {"name": "A-B", "parameters": {}}},
        {"type": "function", "function": {"name": "a-B", "parameters": {}}},
    ]
    assert adaptive.repaired(message, tools) is message
    unknown = AIMessage(content="", tool_calls=[{"name": "zzz", "args": {}, "id": "1"}])
    assert adaptive.repaired(unknown, tools) is unknown


def test_an_unnamed_or_idless_broken_call_is_left_as_it_is() -> None:
    tools: list[Any] = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    for call in (
        {"name": None, "args": "{}", "id": "1", "error": None, "type": "invalid_tool_call"},
        {"name": "t", "args": "{}", "id": None, "error": None, "type": "invalid_tool_call"},
    ):
        message = AIMessage(content="", invalid_tool_calls=[call])  # type: ignore[list-item]
        assert adaptive.repaired(message, tools) is message


async def test_a_reply_without_calls_passes_untouched() -> None:
    reply = ModelResponse(result=[HumanMessage(content="x")])

    async def handler(request: Any) -> ModelResponse[Any]:
        return reply

    request = ModelRequest(model=None, messages=[], tools=[])  # type: ignore[arg-type]
    assert await adaptive.RepairToolCalls().awrap_model_call(request, handler) is reply


# --------------------------------------------------------------------------- strict tools
async def test_strict_schemas_from_the_first_malformed_call_on(
    harness: Harness, counted: Counted
) -> None:
    model = ScriptedChat([broken("stock", "{sku:"), ("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="strict", tools=[stock])
    assert (await agent.run("a?", user="u")).answer == "7"
    first, *later = model.requests
    assert all("strict" not in t["function"] for t in first["tools"])
    assert all(t["function"].get("strict") is True for r in later for t in r["tools"])
    assert LoopEvent.STRICT_ON in counted.events


async def test_arguments_that_do_not_fit_turn_strict_schemas_on(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"units": 1}), ("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="unfit", tools=[stock])
    assert (await agent.run("a?", user="u")).answer == "7"
    assert model.requests[1]["messages"][-1]["content"].startswith("stock was not run")
    assert all(t["function"].get("strict") is True for t in model.requests[1]["tools"])


async def test_a_model_that_never_errs_never_gets_strict_schemas(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), ("stock", {"sku": "b"}), "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="clean", tools=[stock])
    assert (await agent.run("a?", user="u")).answer == "7"
    assert all("strict" not in t["function"] for r in model.requests for t in r["tools"])


async def test_without_strict_tools_the_schemas_stay_as_they_were(
    harness: Harness, counted: Counted
) -> None:
    model = ScriptedChat([broken("stock", "{sku:"), ("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(
        ReAct(system="s", model=model), id="lax", tools=[stock], without={"strict_tools"}
    )
    assert (await agent.run("a?", user="u")).answer == "7"
    assert all("strict" not in t["function"] for r in model.requests for t in r["tools"])
    assert LoopEvent.STRICT_ON not in counted.events


# --------------------------------------------------------------------------- native output
class Units(BaseModel):
    """Units in stock."""

    sku: str
    units: int


async def test_output_is_the_providers_structured_answer_on_a_model_that_has_it(
    harness: Harness,
) -> None:
    model = ScriptedChat(
        [("stock", {"sku": "a-1"}), '{"sku": "a-1", "units": 7}'],
        profile={"structured_output": True},
    )
    agent = harness.wrap(ReAct(system="s", model=model, output=Units), id="native", tools=[stock])
    result = await agent.run("a-1?", user="u")
    assert result.answer == Units(sku="a-1", units=7), result.error
    first = model.requests[0]
    assert first["response_format"]["json_schema"]["strict"] is True
    assert "Units" not in tools_of(model, 0) and "tool_choice" not in first
    assert len(model.requests) == 2  # no answer tool's turn


async def test_a_model_without_structured_output_answers_through_the_answer_tool(
    harness: Harness,
) -> None:
    model = ScriptedChat([("stock", {"sku": "a-1"}), ("Units", {"sku": "a-1", "units": 7})])
    agent = harness.wrap(ReAct(system="s", model=model, output=Units), id="tool", tools=[stock])
    assert (await agent.run("a-1?", user="u")).answer == Units(sku="a-1", units=7)
    assert "Units" in tools_of(model, 0) and "response_format" not in model.requests[0]


async def test_without_native_output_langchain_chooses_as_it_always_did(
    harness: Harness,
) -> None:
    model = ScriptedChat(
        [("stock", {"sku": "a-1"}), '{"sku": "a-1", "units": 7}'],
        profile={"structured_output": True},
    )
    agent = harness.wrap(
        ReAct(system="s", model=model, output=Units),
        id="off",
        tools=[stock],
        without={"native_output"},
    )
    assert (await agent.run("a-1?", user="u")).answer == Units(sku="a-1", units=7)
    # LangChain's own choice for a profile with structured output: not strict
    assert model.requests[0]["response_format"]["json_schema"]["strict"] is False


def test_native_output_is_for_an_output_on_a_model_that_has_structured_output() -> None:
    plain = ScriptedChat([])
    assert adaptive.native_output(None, plain, gateway=True) is None
    assert adaptive.native_output(Units, plain, gateway=False) is None
    assert adaptive.native_output(Units, plain, gateway=True) is not None
    profiled = ScriptedChat([], profile={"structured_output": True})
    made = adaptive.native_output(Units, profiled, gateway=False)
    assert made is not None and made.strategy.schema_spec.strict is True


# --------------------------------------------------------------------------- think
RULE = "Remembered rule from the user: never reorder more than 5 units."


async def test_the_think_tool_is_offered_where_a_rule_may_forbid_a_change(
    harness: Harness,
) -> None:
    model = ScriptedChat([("think", {"thought": "6 > 5: the rule forbids it"}), "No: 5 at most."])
    agent = harness.wrap(ReAct(system="s", model=model), id="think", tools=[stock, reorder])
    asked = [{"role": "system", "content": RULE}, {"role": "user", "content": "Reorder 6 a-1."}]
    result = await agent.run(asked, user="u")
    assert result.answer == "No: 5 at most."
    assert "think" in tools_of(model, 0)
    assert model.requests[1]["messages"][-1]["content"] == adaptive.THOUGHT


@pytest.mark.parametrize(
    ("asked", "tools"),
    [
        ("Reorder 6 a-1.", [stock, reorder]),  # no context that may hold a rule
        ([{"role": "system", "content": RULE}, {"role": "user", "content": "a-1?"}], [stock]),
    ],
)
async def test_no_think_tool_without_a_rule_or_without_a_change(
    harness: Harness, asked: Any, tools: list[Any]
) -> None:
    model = ScriptedChat(["fine"])
    agent = harness.wrap(ReAct(system="s", model=model), id="nothink", tools=tools)
    await agent.run(asked, user="u")
    assert "think" not in tools_of(model, 0)


async def test_without_think_no_think_tool(harness: Harness) -> None:
    model = ScriptedChat(["fine"])
    agent = harness.wrap(
        ReAct(system="s", model=model), id="off", tools=[stock, reorder], without={"think"}
    )
    asked = [{"role": "system", "content": RULE}, {"role": "user", "content": "Reorder 6 a-1."}]
    await agent.run(asked, user="u")
    assert "think" not in tools_of(model, 0)


async def test_a_model_that_reasons_gets_no_think_tool() -> None:
    seen: list[list[str]] = []

    async def handler(request: Any) -> ModelResponse[Any]:
        seen.append([adaptive._named(t) for t in request.tools])
        return ModelResponse(result=[AIMessage(content="x")])

    think = adaptive.think_tool()
    request = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[SystemMessage(content=RULE), HumanMessage(content="x")],
        tools=[think],
    )
    await adaptive.ThinkTool(reasons=True).awrap_model_call(request, handler)
    assert seen == [[]]
    assert await think.ainvoke({"thought": "t"}) == adaptive.THOUGHT


# --------------------------------------------------------------------------- the token count
def _bulky(n: int) -> Any:
    def report(store: str) -> str:
        """A report.

        Args:
            store: The store's id.
        """
        return "ok"

    return tool(report, name=f"report_{n}", description="A long description. " * 60)


def test_the_count_holds_the_instructions_and_the_tool_definitions() -> None:
    counter = adaptive.CalibratedCounter()
    messages = [HumanMessage(content="hello")]
    tools: list[Any] = [{"type": "function", "function": {"name": "t", "description": "d" * 400}}]
    alone = counter(messages)
    assert counter(messages, system=SystemMessage(content="s" * 400)) >= alone + 100
    assert counter(messages, tools=tools) >= alone + 100


def test_the_count_is_scaled_by_the_usage_the_model_reported() -> None:
    counter = adaptive.CalibratedCounter()
    said = AIMessage(
        content="x" * 40,
        response_metadata={"model_provider": "openai"},
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )
    messages = [HumanMessage(content="y" * 40), said]
    assert counter(messages) > adaptive.count_tokens_approximately(messages)


def test_a_tool_set_is_made_into_text_once_and_at_most_toolsets_are_kept() -> None:
    counter = adaptive.CalibratedCounter(toolsets=2)
    sets = [
        [{"type": "function", "function": {"name": f"t{n}", "description": "d"}}] for n in range(3)
    ]
    first = counter._definitions(sets[0])
    assert counter._definitions(sets[0]) is first  # kept
    counter._definitions(sets[1])
    counter._definitions(sets[0])  # the most recent again
    counter._definitions(sets[2])  # past the bound: the least recent (sets[1]) goes
    assert list(counter._texts) == [("t0",), ("t2",)]


async def test_results_are_cleared_once_the_tools_and_messages_fill_half_the_window(
    harness: Harness,
) -> None:
    tools = [stock, *(_bulky(n) for n in range(4))]
    script: list[Any] = [("stock", {"sku": f"s{n}"}) for n in range(5)] + ["done"]
    model = ScriptedChat(script)
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=4000), id="count", tools=tools
    )
    assert (await agent.run("count", user="u")).answer == "done"
    cleared = [r for r in model.requests if CLEARED in json.dumps(r["messages"])]
    assert cleared  # the definitions alone are past half the window


async def test_without_token_count_the_messages_alone_are_counted(harness: Harness) -> None:
    tools = [stock, *(_bulky(n) for n in range(4))]
    script: list[Any] = [("stock", {"sku": f"s{n}"}) for n in range(5)] + ["done"]
    model = ScriptedChat(script)
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=4000),
        id="nocount",
        tools=tools,
        without={"token_count"},
    )
    assert (await agent.run("count", user="u")).answer == "done"
    assert not [r for r in model.requests if CLEARED in json.dumps(r["messages"])]


# --------------------------------------------------------------------------- clear inputs
@tool(side_effects="read")
def chapter(n: int) -> str:
    """A chapter of a book.

    Args:
        n: The chapter's number.
    """
    return f"chapter {n} " + "w" * 12_000


#: What a long task writes to its scratch file first: large enough that clearing its call's
#: arguments frees what the clearing must.
BODY = "notes " * 3000


def _reading(model_turns: list[Any]) -> list[Any]:
    """Notes written first, then six chapters read, then the answer."""
    script: list[Any] = [("write_file", {"file_path": "/notes.md", "content": BODY})]
    return [*script, *(("chapter", {"n": n}) for n in range(1, 7)), *model_turns]


async def test_with_the_scratch_files_offered_cleared_calls_lose_their_arguments(
    harness: Harness,
) -> None:
    model = ScriptedChat(_reading(["done"]))
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=40_000), id="inputs", tools=[chapter]
    )
    assert (await agent.run("read the book", user="u")).answer == "done"
    last = json.dumps(model.requests[-1]["messages"])
    assert CLEARED in last and BODY not in last  # the write's body went with its result


async def test_without_clear_inputs_cleared_calls_keep_their_arguments(harness: Harness) -> None:
    model = ScriptedChat(_reading(["done"]))
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=40_000),
        id="keep",
        tools=[chapter],
        without={"clear_inputs"},
    )
    assert (await agent.run("read the book", user="u")).answer == "done"
    last = json.dumps(model.requests[-1]["messages"])
    assert CLEARED in last and BODY.strip() in last


# --------------------------------------------------------------------------- stable prefix
class Cached(Script):
    """A script whose replies say the prompt cache held part of the prompt."""

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        found = super().reply(body)
        found["usage"]["prompt_tokens_details"] = {"cached_tokens": 8}
        return found


LONG = ("write_todos", "ls", "write_file", "edit_file")


async def test_once_the_cache_works_the_long_task_tools_go_from_the_first_call(
    harness: Harness,
) -> None:
    model = ScriptedChat(script=Cached(["one", "two"]))
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=32_000), id="prefix", tools=[stock]
    )
    await agent.run("first", user="u")
    assert not set(LONG) & set(tools_of(model, 0))
    await agent.run("second", user="u")
    assert set(LONG) <= set(tools_of(model, 1))


@pytest.mark.parametrize(
    ("script", "window", "without"),
    [
        (Script(["one", "two"]), 32_000, set()),  # no cache reported
        (Cached(["one", "two"]), 8_000, set()),  # no room for the tools
        (Cached(["one", "two"]), 32_000, {"stable_prefix"}),
    ],
)
async def test_no_tools_from_the_first_call_without_a_cache_room_or_the_feature(
    harness: Harness, script: Script, window: int, without: set[str]
) -> None:
    model = ScriptedChat(script=script)
    agent = harness.wrap(
        ReAct(system="s", model=model, context_window=window),
        id="noprefix",
        tools=[stock],
        without=without,  # type: ignore[arg-type]
    )
    await agent.run("first", user="u")
    await agent.run("second", user="u")
    assert not set(LONG) & set(tools_of(model, 1))


# --------------------------------------------------------------------------- unsupported content
PIXEL = "data:image/png;base64,iVBORw0KGgo="
PHOTO: Any = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": PIXEL}},
        ],
    }
]


async def test_an_image_the_model_cannot_take_is_replaced_by_a_notice(harness: Harness) -> None:
    model = ScriptedChat(["a photo"], profile={"image_inputs": False})
    agent = harness.wrap(ReAct(system="s", model=model), id="image")
    assert (await agent.run(PHOTO, user="u")).answer == "a photo"
    sent = json.dumps(model.requests[0]["messages"])
    assert PIXEL not in sent and "does not support image" in sent


@pytest.mark.parametrize(
    ("profile", "without"), [({}, set()), ({"image_inputs": False}, {"unsupported_content"})]
)
async def test_an_image_goes_as_it_is_to_a_model_that_takes_it_or_without_the_feature(
    harness: Harness, profile: dict[str, Any], without: set[str]
) -> None:
    model = ScriptedChat(["a photo"], profile=profile)
    agent = harness.wrap(ReAct(system="s", model=model), id="asis", without=without)  # type: ignore[arg-type]
    assert (await agent.run(PHOTO, user="u")).answer == "a photo"
    assert PIXEL in json.dumps(model.requests[0]["messages"])


# --------------------------------------------------------------------------- rubric
def _graded(result: str, gap: str | None = None) -> tuple[str, dict[str, Any]]:
    criterion: dict[str, Any] = {"name": "names the units", "passed": gap is None}
    if gap is not None:
        criterion["gap"] = gap
    return ("GraderResponse", {"result": result, "explanation": result, "criteria": [criterion]})


async def test_an_answer_is_graded_against_the_runs_rubric_and_revised(harness: Harness) -> None:
    model = ScriptedChat(
        [
            "some",
            _graded("needs_revision", "say how many"),
            "7 units",
            _graded("satisfied"),
        ]
    )
    agent = harness.wrap(ReAct(system="s", model=model), id="rubric", tools=[stock])
    asked = {"messages": [{"role": "user", "content": "a-1?"}], "rubric": "Names the units."}
    result = await agent.run(asked, user="u")
    assert result.answer == "7 units", result.error
    revised = model.requests[2]["messages"][-1]["content"]
    assert "say how many" in revised


async def test_no_grading_without_a_rubric_or_without_the_feature(harness: Harness) -> None:
    model = ScriptedChat(["some", "more"])
    plain = harness.wrap(ReAct(system="s", model=model), id="plain", tools=[stock])
    assert (await plain.run("a-1?", user="u")).answer == "some"
    off = harness.wrap(ReAct(system="s", model=model), id="off", tools=[stock], without={"rubric"})
    asked = {"messages": [{"role": "user", "content": "a-1?"}], "rubric": "Names the units."}
    assert (await off.run(asked, user="u")).answer == "more"
    assert len(model.requests) == 2


# --------------------------------------------------------------------------- together
async def test_react_adaptive_turns_every_piece_off(harness: Harness, counted: Counted) -> None:
    model = ScriptedChat([broken("Stock", '{"sku": "a-1"'), ("stock", {"sku": "a-1"}), "7"])
    agent = harness.wrap(
        ReAct(system="s", model=model),
        id="none",
        tools=[stock, reorder],
        without={"react_adaptive"},
    )
    asked = [{"role": "system", "content": RULE}, {"role": "user", "content": "a-1?"}]
    assert (await agent.run(asked, user="u")).answer == "7"
    assert "think" not in tools_of(model, 0)
    assert all("strict" not in t["function"] for r in model.requests for t in r["tools"])
    assert LoopEvent.REPAIRED not in counted.events and LoopEvent.STRICT_ON not in counted.events


async def test_the_pieces_work_together_in_one_run(harness: Harness, counted: Counted) -> None:
    model = ScriptedChat(
        [
            broken("stock", "{sku:"),  # unrepairable: strict schemas from here on
            broken("STOCK", '{"sku": "a-1"', call="c2"),  # repaired: name and arguments
            ("think", {"thought": "a read: no rule applies"}),
            "7 units",
        ]
    )
    agent = harness.wrap(ReAct(system="s", model=model), id="all", tools=[stock, reorder])
    asked = [{"role": "system", "content": RULE}, {"role": "user", "content": "a-1?"}]
    result = await agent.run(asked, user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "7 units", result.error
    assert "think" in tools_of(model, 0)
    assert all(t["function"].get("strict") is True for t in model.requests[1]["tools"])
    assert model.requests[2]["messages"][-1]["content"] == "7"
    assert counted.events.count(LoopEvent.REPAIRED) == 1
    assert counted.events.count(LoopEvent.STRICT_ON) == 1


def test_an_edited_request_still_has_its_messages() -> None:
    # ToolMessage answers that are not about arguments do not count as malformed
    assert not adaptive._malformed(ToolMessage(content="boom", tool_call_id="1", status="error"))
    assert adaptive._malformed(
        ToolMessage(content="x was not run: y. " + adaptive.FIX_ARGUMENTS, tool_call_id="1",
                    status="error")
    )  # fmt: skip
