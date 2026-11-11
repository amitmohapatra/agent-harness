"""Whatever the window, a tool the task needs is offered when the model reaches for it, and a
tool held back can be found: ``FitWindow``'s narrowing over generated situations — any number
of tools, the chosen ones at any position, any window, a latest turn of any size."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from tests.support.chat_model import ScriptedChatModel
from trellis.harness import middleware as ours
from trellis.harness import runtime as runtime_module
from trellis.harness.clients.memory import TOOL_SEARCH
from trellis.harness.middleware import FitWindow

COUNTS = (3, 9, 18, 40, 120)
WINDOWS = (2_000, 6_000, 8_000, 16_000, 64_000)
#: The latest turn's own steps, in characters: none, some, and more than any window here.
TURNS = (0, 6_000, 400_000)


class Run:
    """What the narrowing reads of a run: the tools it used and those the hints chose."""

    def __init__(self, used: set[str], offered: set[str] | None) -> None:
        self.used, self.offered = used, offered
        self.events = SimpleNamespace(warning=lambda *said: None)
        self.replay = SimpleNamespace(journal=SimpleNamespace(results={}))
        self.earlier_results: dict[str, str] = {}


def made(name: str, rng: random.Random) -> Any:
    def run(key: str, limit: int = 10) -> str:
        return key

    words = " ".join(rng.choice(("stock", "price", "store", "order", "audit")) for _ in range(40))
    return StructuredTool.from_function(func=run, name=name, description=f"Looks up {words}.")


def situation(count: int, turn: int, seed: int) -> tuple[ModelRequest[Any], Run, set[str]]:
    """A request with ``count`` tools, a few of them chosen for the task (the hints'), a few
    used by the run, at random positions, and a latest turn of ``turn`` characters of the
    model's own steps (what it wrote, and results too short to cut)."""
    rng = random.Random(f"{count}-{turn}-{seed}")
    names = [f"tool_{n}" for n in range(count)]
    chosen = set(rng.sample(names, k=min(3, count)))
    used = set(rng.sample(names, k=min(2, count)))
    messages: list[Any] = [HumanMessage("What is the on-time rate of the top supplier?")]
    for step in range(turn // 1_500):  # the model's own steps: nothing in them is cut shorter
        call = {"id": f"c{step}", "name": sorted(used)[0], "args": {}}
        messages += [
            AIMessage(content="thinking it over " * 60, tool_calls=[call]),
            ToolMessage("row,value\n" * 50, tool_call_id=f"c{step}"),
        ]
    request = ModelRequest(
        model=ScriptedChatModel(turns=[]),
        messages=messages,
        system_message=SystemMessage(content="You help."),
        tools=[made(name, rng) for name in names],
        tool_choice=None,
        response_format=None,
        state={"messages": messages},
        runtime=None,  # type: ignore[arg-type]
        model_settings={},
    )
    return request, Run(used, chosen), chosen | used


async def fitted(window: int, request: ModelRequest[Any], runtime: Run) -> ModelRequest[Any]:
    seen: list[ModelRequest[Any]] = []

    async def handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(r)
        return ModelResponse(result=[AIMessage(content="ok")])

    token = runtime_module._current.set(runtime)  # type: ignore[arg-type]
    try:
        await FitWindow(window).awrap_model_call(request, handler)
    finally:
        runtime_module._current.reset(token)
    return seen[0]


def offered(request: ModelRequest[Any]) -> dict[str, Any]:
    return {ours._named(t): t for t in request.tools}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("turn", TURNS)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("count", COUNTS)
async def test_a_tool_the_task_chose_is_offered_at_any_window(
    count: int, window: int, turn: int, seed: int
) -> None:
    """The tools the run used and those the hints chose are in every request, whatever the
    window, the number of tools, their position among them, or the size of the latest turn."""
    request, runtime, chosen = situation(count, turn, seed)
    sent = offered(await fitted(window, request, runtime))
    assert chosen <= set(sent), (sorted(chosen - set(sent)), len(sent))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("turn", TURNS)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("count", COUNTS)
async def test_a_tool_held_back_is_named_and_can_be_found(
    count: int, window: int, turn: int, seed: int
) -> None:
    """A tool the request no longer offers is never dropped in silence: the instructions name
    every one of them, and say what finds one."""
    request, runtime, _ = situation(count, turn, seed)
    request = request.override(tools=[*request.tools, made(TOOL_SEARCH, random.Random(seed))])
    out = await fitted(window, request, runtime)
    sent = offered(out)
    held = {ours._named(t) for t in request.tools} - set(sent)
    told = out.system_message.text if out.system_message is not None else ""
    if not held:
        assert ours._HELD not in told  # nothing held: nothing said
        return
    assert TOOL_SEARCH in sent  # what finds a held tool is never held itself
    line = next(part for part in told.split("\n\n") if part.startswith(ours._HELD))
    assert held == set(line.split(": ")[1].split(", "))
    assert ours.FIT_FOUND in line


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("count", COUNTS)
async def test_the_tools_ranked_first_stay_callable_when_the_latest_turn_fills_the_window(
    count: int, window: int
) -> None:
    """With nothing chosen for the task (no hints) and a latest turn that alone fills the
    window, the tools ranked first are still offered — by their signature, the name and the
    arguments a call takes — and the run's used ones as they were: the model reaches for a
    tool it has not called yet and it is there (``long-multi-02`` at 8k: three of three runs
    said "none of my tools hold supplier delivery data" of a tool dropped a step before)."""
    request, runtime, _ = situation(count, TURNS[-1], 0)
    runtime.offered = None
    used = runtime.used
    sent = offered(await fitted(window, request, runtime))
    assert used <= set(sent)
    beside = [name for name in sent if name not in used]
    assert len(beside) >= min(ours.FIT_KEEP, count - len(used))
    for name in beside[: ours.FIT_KEEP]:
        function = sent[name]["function"] if isinstance(sent[name], dict) else None
        assert function is None or set(function["parameters"]["properties"]) == {"key", "limit"}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("window", WINDOWS[1:])
@pytest.mark.parametrize("count", COUNTS)
async def test_a_question_with_any_number_of_tools_fits_with_what_is_said_of_the_held_ones(
    count: int, window: int, seed: int
) -> None:
    """The line naming the held tools is part of what must fit: its room is set aside."""
    request, runtime, _ = situation(count, 0, seed)
    out = await fitted(window, request, runtime)
    assert ours._size(out) <= int(window * 0.95)


def test_a_signature_is_a_tools_name_and_arguments() -> None:
    tool = made("supplier_info", random.Random(0))
    bare = ours._signature(tool)
    function = bare["function"]
    assert function["name"] == "supplier_info" and function["description"] == ours.FIT_SIGNATURE
    assert function["parameters"]["properties"] == {
        "key": {"type": "string"},
        "limit": {"type": "integer", "default": 10},
    }
    assert function["parameters"]["required"] == ["key"]
    assert ours._signature(bare) == bare  # a signature already
