"""``ask_user`` and the activity line over AG-UI: what a chat UI renders (a picker for a
date range, a dropdown for options, a number field), what plain chat reads (the question with
the kind of answer it takes), the answer the model reads; and the run's tool calls grouped as
one "used N tools · ran K commands" line."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.support.arm import arm
from tests.support.planned import PlannedChat
from trellis import Harness, Hooks, ReAct, Runtime, Settings, ask_user, tool
from trellis.contracts import ConfigurationError
from trellis.harness.agui.sse import decode
from trellis.harness.tools.ask import EXPECTS

PATH = "/agui"
SAID = [{"id": "m", "role": "user", "content": "Order a shirt"}]


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h = Harness(config=Settings())
    yield h
    await h.aclose()


async def post(http: httpx.AsyncClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    response = await http.post(f"{PATH}/run", json={"threadId": "t1", "messages": [], **body})
    assert response.status_code == 200, response.text
    return [e for _, e in decode(response.text)]


def served(agent: Any) -> httpx.AsyncClient:
    app = FastAPI()
    agent.serve_chat(app, identity=lambda request: "u1")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agui")


def asking(harness: Harness, args: dict[str, Any], name: str) -> Any:
    model = PlannedChat([("ask_user", args)])
    return arm(harness.wrap(ReAct(system_prompt="Help.", model=model), id=name), [ask_user()])


@pytest.mark.parametrize(
    ("kind", "answer"),
    [
        ("date_range", {"start": "2026-11-02", "end": "2026-11-06"}),
        ("date_range", "the first week of November"),  # plain chat answers in words
        ("date", "2026-11-05"),
        ("number", 3),
        ("number", "three"),
    ],
)
async def test_a_structured_question_renders_and_reads_back(
    harness: Harness, kind: str, answer: Any
) -> None:
    agent = asking(harness, {"question": "When should it arrive?", "kind": kind}, f"ask-{kind}")
    said = [{"id": "m", "role": "user", "content": "Ship my order"}]
    async with served(agent) as http:
        paused = (await post(http, {"messages": said}))[-1]
        [entry] = paused["outcome"]["interrupts"]
        assert entry["reason"] == "question"
        assert entry["message"].startswith("When should it arrive? (a ")  # what plain chat reads
        assert entry["metadata"]["component"] == "ask_user"
        assert entry["metadata"]["props"] == {"kind": kind, "options": []}
        assert entry["responseSchema"] == EXPECTS[kind]  # a structured value, or text
        widgets = {
            "date": {"ui:widget": "date"},
            "date_range": {"start": {"ui:widget": "date"}, "end": {"ui:widget": "date"}},
            "number": {"ui:widget": "updown"},
        }
        assert entry["metadata"]["ui_schema"] == widgets[kind]  # the form's widget hints
        resume = [{"interruptId": entry["id"], "payload": answer}]
        done = (await post(http, {"resume": resume}))[-1]
    assert done["outcome"] == {"type": "success"}
    assert str(answer if not isinstance(answer, dict) else "2026-11-06") in done["result"]


async def test_options_are_a_choice_plain_chat_reads_numbered(harness: Harness) -> None:
    """A choice with no reason to recommend one: the options as they are (plain strings on the
    wire), numbered in plain chat, and another answer in words is the "Other"."""
    args = {"question": "Which size?", "options": ["S", "L", "S"]}
    agent = asking(harness, args, "ask-size")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "Order a shirt"}]
        paused = (await post(http, {"messages": said}))[-1]
        [entry] = paused["outcome"]["interrupts"]
        assert entry["reason"] == "choice"
        assert entry["message"] == "Which size?\n1. S\n2. L\n(Pick one, or say another.)"
        assert entry["metadata"]["options"] == ["S", "L"]
        assert entry["metadata"]["props"] == {"kind": "text", "options": ["S", "L"]}
        other = [{"interruptId": entry["id"], "payload": "M, if you have it"}]
        done = (await post(http, {"resume": other}))[-1]
    assert done["result"] == "Done. M, if you have it"


async def test_a_recommended_option_says_why_and_several_may_be_picked(
    harness: Harness,
) -> None:
    args = {
        "question": "Which regions?",
        "options": ["north", "south", "east"],
        "multiple": True,
        "recommended": "north",
        "why": "your choice last time",
    }
    agent = asking(harness, args, "ask-regions")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "Set the alert"}]
        [entry] = (await post(http, {"messages": said}))[-1]["outcome"]["interrupts"]
        assert entry["message"] == (
            "Which regions?\n1. north (Recommended: your choice last time)\n2. south\n3. east"
            "\n(Pick one or more, or say another.)"
        )
        assert entry["metadata"]["options"] == [
            {"value": "north", "description": "your choice last time", "recommended": True},
            "south",
            "east",
        ]
        assert entry["metadata"]["multiple"] is True
        picked = [{"interruptId": entry["id"], "payload": ["north", "east"]}]
        done = (await post(http, {"resume": picked}))[-1]
    assert done["result"] == 'Done. ["north", "east"]'


async def test_the_recommended_option_comes_first_wherever_the_model_listed_it(
    harness: Harness,
) -> None:
    """The option to pick is the one a person reads first: in the choice a UI renders, in the
    app's own control and in the numbered text."""
    args = {"question": "Which region?", "options": ["north", "south", "west"],
            "recommended": "west", "why": "least stock"}  # fmt: skip
    agent = asking(harness, args, "ask-first")
    said = [{"id": "m", "role": "user", "content": "Send the pallet"}]
    async with served(agent) as http:
        [entry] = (await post(http, {"messages": said}))[-1]["outcome"]["interrupts"]
    assert entry["metadata"]["options"] == [
        {"value": "west", "description": "least stock", "recommended": True},
        "north",
        "south",
    ]
    assert entry["metadata"]["props"]["options"] == ["west", "north", "south"]
    assert entry["message"] == (
        "Which region?\n1. west (Recommended: least stock)\n2. north\n3. south"
        "\n(Pick one, or say another.)"
    )


@pytest.mark.parametrize(
    "args",
    [
        {"recommended": "L"},  # no reason given: nothing is recommended
        {"recommended": "XL", "why": "it fits most"},  # not one of the options
    ],
)
async def test_no_option_is_recommended_without_a_reason_among_the_options(
    harness: Harness, args: dict[str, Any]
) -> None:
    agent = asking(harness, {"question": "Which size?", "options": ["S", "L"], **args}, "ask-n")
    async with served(agent) as http:
        said = [{"id": "m", "role": "user", "content": "Order a shirt"}]
        [entry] = (await post(http, {"messages": said}))[-1]["outcome"]["interrupts"]
    assert entry["metadata"]["options"] == ["S", "L"] and "Recommended" not in entry["message"]


async def test_ask_user_runs_only_in_a_run() -> None:
    with pytest.raises(ConfigurationError, match="inside a harness run"):
        await ask_user()("When?")


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"about {topic}"


@tool(name="execute", side_effects="read")
async def run_command(command: str) -> str:
    """Run a command."""
    return f"ran {command}"


async def test_the_runs_tool_calls_are_one_activity_line(harness: Harness) -> None:
    async def busy(input: Any, agent: Runtime) -> str:
        await agent.tools.call("lookup", topic="a")
        await agent.tools.call("lookup", topic="b")
        await agent.tools.call("execute", command="ls")
        return "done"

    agent = arm(harness.wrap(busy, id="busy"), [lookup, run_command])
    async with served(agent) as http:
        events = await post(http, {"messages": [{"id": "m", "role": "user", "content": "go"}]})
    lines = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert [e["content"]["summary"] for e in lines] == [
        "used 1 tool",
        "used 2 tools",
        "used 2 tools · ran 1 command",
    ]
    assert {e["messageId"] for e in lines} == {f"{events[0]['runId']}:activity"}
    assert all(e["activityType"] == "tools" and e["replace"] is True for e in lines)
    assert lines[-1]["content"] == {
        "summary": "used 2 tools · ran 1 command",
        "tools": 2,
        "commands": 1,
    }
    # each line follows the call it counts
    results = [i for i, e in enumerate(events) if e["type"] == "TOOL_CALL_RESULT"]
    assert [events[i + 1]["type"] for i in results] == ["ACTIVITY_SNAPSHOT"] * 3


async def test_a_call_replayed_on_resume_is_counted_once(harness: Harness) -> None:
    async def twice(input: Any, agent: Runtime) -> str:
        await agent.tools.call("lookup", topic="a")
        size = await agent.ask("Size?", options=["S", "L"])
        return f"size {size}"

    agent = arm(harness.wrap(twice, id="replays"), [lookup])
    async with served(agent) as http:
        first = await post(http, {"messages": [{"id": "m", "role": "user", "content": "go"}]})
        entry = first[-1]["outcome"]["interrupts"][0]
        resumed = await post(http, {"resume": [{"interruptId": entry["id"], "payload": "S"}]})
    counted = [
        e["content"]["tools"] for e in [*first, *resumed] if e["type"] == "ACTIVITY_SNAPSHOT"
    ]
    assert counted == [1, 1]  # the replayed lookup has the same id: the line says 1 again


# --------------------------------------------------------------------------- offered by itself
SIZE = {"question": "Which size?", "options": ["S", "L"]}


class Seen:
    """A function target that keeps the names of the tools each of its runs was offered, and
    calls the tools it is told to."""

    def __init__(self) -> None:
        self.toolboxes: list[set[str]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, input: Any, agent: Runtime) -> str:
        self.toolboxes.append(set(agent.toolbox))
        said = [str(await agent.tools.call(name, **args)) for name, args in self.calls]
        return said[-1] if said else "ok"


async def test_a_chat_front_ends_run_is_offered_ask_user_by_itself(harness: Harness) -> None:
    """A person reads a chat front end's run (``serve_chat``) and answers its pauses: the model
    is given ``ask_user`` with nothing declared, whatever the app sent besides the message."""
    agent = harness.wrap(ReAct(system_prompt="Help.", model=PlannedChat([("ask_user", SIZE)])),
                         id="auto")  # fmt: skip
    async with served(agent) as http:
        [entry] = (await post(http, {"messages": SAID}))[-1]["outcome"]["interrupts"]
        assert entry["metadata"]["component"] == "ask_user"
        assert entry["metadata"]["props"] == {"kind": "text", "options": ["S", "L"]}
        done = (await post(http, {"resume": [{"interruptId": entry["id"], "payload": "L"}]}))[-1]
    assert done["result"] == "Done. L"


async def test_a_run_nobody_is_known_to_read_is_offered_none(harness: Harness) -> None:
    """Started from code (``run``, ``stream``), queued for a worker (``start``) or started by
    another run, nobody is known to answer: a pause there may never be resumed, so the tool is
    the developer's to add."""
    seen = Seen()
    agent = harness.wrap(seen, id="code")

    async def parent(input: Any, runtime: Runtime) -> str:
        return str((await agent.run("child", user="u1")).answer)

    await agent.run("hi", user="u1")
    assert [e.type async for e in agent.stream("hi", user="u1")]
    await agent.start("hi", user="u1")
    assert await harness.worker([agent]).run_once() is True
    async with served(harness.wrap(parent, id="parent")) as http:  # a chat's run starts it
        await post(http, {"messages": SAID})
    assert len(seen.toolboxes) == 4 and not any("ask_user" in t for t in seen.toolboxes)


async def test_a_schedule_set_in_the_chat_fires_without_it(harness: Harness) -> None:
    """The chat's run has it; the run its schedule fires — nobody is there then — has not."""
    seen = Seen()
    agent = harness.wrap(seen, id="desk")
    seen.calls = [("schedule_task", {"task": "check stock", "when": "0 8 * * *"})]
    async with served(agent) as http:
        await post(http, {"messages": SAID})
    seen.calls = []
    [schedule] = harness.runs._schedules.values()  # type: ignore[union-attr]
    await harness.runs.schedules.fire(schedule.schedule_id)
    assert await harness.worker([agent]).run_once() is True
    assert ["ask_user" in t for t in seen.toolboxes] == [True, False]


async def test_without_ask_turns_it_off(harness: Harness) -> None:
    seen = Seen()
    async with served(harness.wrap(seen, id="quiet", without=["ask"])) as http:
        await post(http, {"messages": SAID})
    assert seen.toolboxes and "ask_user" not in seen.toolboxes[-1]


async def test_the_agents_own_ask_user_wins_and_is_offered_once(harness: Harness) -> None:
    """A tool of the agent's named ``ask_user`` is the one the model calls — its own function,
    or ``ask_user()`` passed as before — and the harness adds no second."""

    @tool(name="ask_user", side_effects="read")
    async def own(question: str) -> str:
        """Ask the person."""
        return f"asked my way: {question}"

    seen = Seen()
    seen.calls = [("ask_user", {"question": "Which size?"})]
    async with served(arm(harness.wrap(seen, id="own"), [own])) as http:
        done = (await post(http, {"messages": SAID}))[-1]
    assert done["result"] == "asked my way: Which size?"
    explicit, offered = ask_user(), []

    class Offered(Hooks):
        async def on_run_start(self, run: Runtime) -> None:
            offered.extend(run.toolbox.values())

    given = harness.wrap(seen, id="given", hooks=[Offered()])
    seen.calls = []
    async with served(arm(given, [explicit])) as http:
        await post(http, {"messages": SAID})
    assert [t for t in offered if t.name == "ask_user"] == [explicit.tool]


async def test_a_framework_with_a_question_tool_of_its_own_keeps_that_one() -> None:
    """Claude Code and Kimi Code ask with their own ``AskUserQuestion``, asked as the run's
    question: such a target says so, and its runs get no second tool."""
    from agents import Agent as SDKAgent
    from claude_agent_sdk import ClaudeAgentOptions

    from tests.support.planned import PlannedModel
    from trellis.harness.adapters import detect
    from trellis.harness.adapters.acp import ACPAgent, kimi

    async def function(input: Any, agent: Runtime) -> str:
        return "ok"

    plain = ACPAgent(command=("some-agent", "acp"))
    sdk = SDKAgent(name="a", model=PlannedModel([]))
    graph = ReAct(system_prompt="x", model=PlannedChat([]))
    targets = [ClaudeAgentOptions(), kimi(), plain, function, graph, sdk]
    assert [detect(t).asks(t) for t in targets] == [True, True, False, False, False, False]


# --------------------------------------------------------------------------- several values
FORM = {
    "question": "Confirm the reorder.",
    "fields": [
        {"name": "quantity", "label": "Units", "kind": "number", "default": 200,
         "why": "your order of 2026-11-02"},
        {"name": "supplier", "label": "Supplier", "options": ["SUP-04", "SUP-07"],
         "default": "SUP-04", "why": "your usual supplier"},
        {"name": "deliver_on", "label": "Delivery date", "kind": "date", "required": False},
    ],
}  # fmt: skip


async def test_several_values_are_one_form_filled_in_with_what_is_remembered(
    harness: Harness,
) -> None:
    """One question, one submit: the app's control gets each field (its kind, its options,
    the remembered value to show filled in and where it is from), any other UI the same as a
    JSON Schema with widget hints, plain chat a short numbered list."""
    agent = asking(harness, FORM, "ask-form")
    async with served(agent) as http:
        [entry] = (await post(http, {"messages": SAID}))[-1]["outcome"]["interrupts"]
        assert entry["reason"] == "question"
        assert entry["metadata"]["component"] == "ask_user"
        assert entry["metadata"]["props"] == {
            "kind": "form",
            "fields": [
                {"name": "quantity", "label": "Units", "kind": "number", "default": 200,
                 "why": "your order of 2026-11-02", "required": True},
                {"name": "supplier", "label": "Supplier", "kind": "text",
                 "options": ["SUP-04", "SUP-07"], "default": "SUP-04",
                 "why": "your usual supplier", "required": True},
                {"name": "deliver_on", "label": "Delivery date", "kind": "date",
                 "required": False},
            ],
        }  # fmt: skip
        assert entry["responseSchema"] == {
            "type": ["object", "string"],
            "properties": {
                "quantity": {"type": ["number", "string"], "title": "Units", "default": 200,
                             "description": "your order of 2026-11-02"},
                "supplier": {"type": "string", "title": "Supplier", "enum": ["SUP-04", "SUP-07"],
                             "default": "SUP-04", "description": "your usual supplier"},
                "deliver_on": {"type": "string", "format": "date", "title": "Delivery date"},
            },
            "required": [],  # each required one is filled in: its default stands
        }  # fmt: skip
        assert entry["metadata"]["ui_schema"] == {
            "ui:order": ["quantity", "supplier", "deliver_on"],
            "quantity": {"ui:widget": "updown"},
            "deliver_on": {"ui:widget": "date"},
        }
        assert entry["message"] == (
            "Confirm the reorder.\n"
            "1. Units (a number): 200 (your order of 2026-11-02)\n"
            "2. Supplier (SUP-04 or SUP-07): SUP-04 (your usual supplier)\n"
            "3. Delivery date (a date, YYYY-MM-DD), if any\n"
            "(Answer each, or say what to keep and what to change.)"
        )
        changed = [{"interruptId": entry["id"], "payload": {"quantity": 40}}]
        done = (await post(http, {"resume": changed}))[-1]
    # one object: what the person changed over what was filled in
    assert done["result"] == 'Done. {"quantity": 40, "supplier": "SUP-04"}'


async def test_a_form_is_answered_in_words_in_plain_chat(harness: Harness) -> None:
    agent = asking(harness, FORM, "ask-form-words")
    async with served(agent) as http:
        [entry] = (await post(http, {"messages": SAID}))[-1]["outcome"]["interrupts"]
        words = [{"interruptId": entry["id"], "payload": "keep all, deliver on the 20th"}]
        done = (await post(http, {"resume": words}))[-1]
    assert done["result"] == "Done. keep all, deliver on the 20th"


async def test_a_remembered_value_is_shown_filled_in_a_single_question_too(
    harness: Harness,
) -> None:
    """``recommended`` is the one field's default: for a number or a date, the value shown
    filled in, with where it is from."""
    args = {"question": "Which threshold?", "kind": "number", "recommended": "20",
            "why": "your usual, said 2025-11-03"}  # fmt: skip
    async with served(asking(harness, args, "ask-filled")) as http:
        [entry] = (await post(http, {"messages": SAID}))[-1]["outcome"]["interrupts"]
    assert entry["message"] == "Which threshold? (a number): 20 (your usual, said 2025-11-03)"
    assert entry["responseSchema"] == {"type": ["number", "string"], "default": "20"}
    assert entry["metadata"]["props"] == {"kind": "number", "options": [], "default": "20",
                                          "why": "your usual, said 2025-11-03"}  # fmt: skip


async def test_one_field_is_the_single_question_it_always_was(harness: Harness) -> None:
    """One mechanism: a question with one value asked as a field is the question its plain
    arguments make — the same interrupt, no form."""
    flat = {"question": "Which region?", "options": ["north", "west"], "recommended": "west",
            "why": "least stock"}  # fmt: skip
    field = {"question": "Which region?", "fields": [
        {"name": "region", "options": ["north", "west"], "default": "west",
         "why": "least stock"}]}  # fmt: skip
    seen = []
    for n, args in enumerate((flat, field)):
        async with served(asking(harness, args, f"ask-one-{n}")) as http:
            [entry] = (await post(http, {"messages": SAID}))[-1]["outcome"]["interrupts"]
        seen.append({k: entry[k] for k in ("reason", "message", "responseSchema", "metadata")})
    assert seen[0] == seen[1] and seen[0]["reason"] == "choice"
    # the recommended one first, as for any choice
    assert seen[0]["metadata"]["props"] == {"kind": "text", "options": ["west", "north"]}


async def test_a_forms_fields_are_named_once_each(harness: Harness) -> None:
    twice = {"question": "Confirm.", "fields": [{"name": "a"}, {"name": "a"}]}
    async with served(asking(harness, twice, "ask-twice")) as http:
        done = (await post(http, {"messages": SAID}))[-1]
    assert done["outcome"] == {"type": "success"} and "named once" in done["result"]


async def test_ask_user_names_the_apps_own_controls_as_the_ones_to_call(
    harness: Harness,
) -> None:
    """Beside ``ask_user`` the app's date picker is the control for a date: the tool says so
    first, naming the app's tools; with an app that declared none it says nothing of them."""
    from trellis.harness.tools.ask import APPS

    picker = {"name": "pick_date", "description": "Show a date picker.",
              "parameters": {"type": "object", "properties": {}}}  # fmt: skip
    model = PlannedChat([])
    agent = harness.wrap(ReAct(system_prompt="Help.", model=model), id="app-tools")
    async with served(agent) as http:
        await post(http, {"messages": SAID, "tools": [picker]})
        await post(http, {"messages": SAID})
    with_tools, without = (
        {t["function"]["name"]: t["function"] for t in request["tools"]}
        for request in (model.requests[0], model.requests[-1])
    )
    said = APPS.format(tools="pick_date")
    plain = without["ask_user"]
    assert with_tools["pick_date"]["description"] == "Show a date picker."
    assert with_tools["ask_user"]["description"] == f"{said} {plain['description']}"
    kinds = [f["parameters"]["properties"]["kind"]["description"] for f in
             (with_tools["ask_user"], plain)]  # fmt: skip
    assert kinds[0] == f"{kinds[1]} {said}"  # and where the kind of answer is chosen
    assert "controls" not in json.dumps(plain)


def test_a_forms_option_field_lists_its_recommended_option_first() -> None:
    from trellis.harness.tools.ask import Field

    region = Field(name="region", options=["north", "west"], default="west", why="least stock")
    assert region.in_form()["enum"] == ["west", "north"]
    assert region.shown()["options"] == ["west", "north"]
    assert region.line("Region") == "Region (west or north): west (least stock)"


def test_a_forms_field_with_several_picks_is_an_array_of_its_options() -> None:
    from trellis.harness.tools.ask import Field

    regions = Field(name="regions", options=["north", "west"], multiple=True, default=["west"])
    assert regions.in_form() == {
        "type": "array",
        "items": {"type": "string", "enum": ["north", "west"]},
        "uniqueItems": True,
        "title": "regions",
        "default": ["west"],
    }
    both = regions.model_copy(update={"default": ["north", "west"]})
    assert both.line("Regions") == "Regions (north or west): north, west"


def test_the_sandbox_deep_agents_and_claude_shells_each_ran_a_command() -> None:
    from trellis.contracts import AgentExecutionContext, RunEvent, RunEventType
    from trellis.harness.agui.activity import Activity

    context = AgentExecutionContext.create(tenant_id="t", agent_id="a", agent_run_id="run_1")
    line = Activity()
    said = []
    for n, tool_name in enumerate(("sandbox_exec", "execute", "Bash", "lookup")):
        ended = RunEvent.of(
            context,
            RunEventType.TOOL_CALL_RESULT,
            0,
            tool_call_id=f"c{n}",
            data={"tool": tool_name},
        )
        event = line.seen(ended)
        assert event is not None
        said.append(event.content["summary"])  # type: ignore[index]
    assert said[-1] == "used 1 tool · ran 3 commands"
