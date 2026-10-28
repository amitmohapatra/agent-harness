"""The reasoning class (``benchmark.cost.reasoning``): its items and their answers, checked
again independently; the score of an answer; each technique an arm is measured with
(``benchmark.cost.techniques``) on a scripted model."""

from __future__ import annotations

import collections
import datetime
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool as lc_tool

from benchmark.cost import reasoning, techniques
from benchmark.cost.variants import VARIANTS, Setup
from tests.support.chat_model import ScriptedChatModel
from trellis.harness.adapters.langgraph import graph_tools


# --------------------------------------------------------------------------- the items
def test_forty_items_half_dev_half_test_each_family_split() -> None:
    items = reasoning.items()
    assert len(items) == 40 and len({i["metadata"]["task"] for i in items}) == 40
    splits = collections.Counter((i["metadata"]["family"], i["metadata"]["split"]) for i in items)
    assert splits == {
        (family, split): count
        for family, count in (("maths", 8), ("dates", 4), ("multi-hop", 4), ("logic", 4))
        for split in ("dev", "test")
    }
    maths = [i for i in items if i["metadata"]["family"] == "maths"]
    assert all(i["metadata"]["source"].startswith("gsm8k test #") for i in maths)
    assert (Path(reasoning.GSM8K).parent / "GSM8K_LICENSE").read_text().startswith("MIT License")


def test_the_generated_answers_follow_from_their_questions() -> None:
    for item in reasoning.items():
        meta, question = item["metadata"], item["input"]
        if meta["family"] == "dates":
            day = datetime.date.fromisoformat(re.findall(r"\d{4}-\d{2}-\d{2}", question)[-1])
            later = re.match(r"What date is (\d+) days after", question)
            want = (day + datetime.timedelta(int(later[1]))).isoformat() if later else f"{day:%A}"
            assert meta["answer"] == want
        elif meta["family"] == "multi-hop":
            n = [int(x) for x in re.findall(r"\d+", question)]
            if "years" in question:  # c, times, more, in
                assert int(meta["answer"]) == n[0] * n[1] + n[2] + n[3]
            else:  # c, more, times, away
                assert int(meta["answer"]) == (n[0] + n[1]) * n[2] - n[3]
        elif meta["family"] == "logic":
            taller = re.findall(r"(\w+) is taller than (\w+)", question)
            people = {p for pair in taller for p in pair}
            above = {p: {a for a, b in taller if b == p} for p in people}
            order = sorted(people, key=lambda p: len(_ancestors(p, above)))
            place = 0 if "tallest?" in question and "second" not in question else 1
            place = -1 if "shortest" in question else place
            assert meta["answer"] == order[place]


def _ancestors(person: str, above: dict[str, set[str]]) -> set[str]:
    found: set[str] = set()
    todo = list(above[person])
    while todo:
        p = todo.pop()
        if p not in found:
            found.add(p)
            todo += above[p]
    return found


@pytest.mark.parametrize(
    ("output", "expected", "kind", "right"),
    [
        ("so 2 + 3\nAnswer: $1,250", "1250", "number", True),
        ("Answer: 12.0 marbles", "12", "number", True),
        ("Answer: 13", "12", "number", False),
        ("It is 12", "12", "number", True),  # no answer line: the last line
        (
            "<think>maybe 3</think><response>Answer: 2024-05-01</response>",
            "2024-05-01",
            "date",
            True,
        ),
        ("Answer: Tuesday", "Tuesday", "text", True),
        ("Answer: dora", "Dora", "text", True),
        ("Answer:", "Dora", "text", False),
    ],
)
def test_an_answer_is_scored_on_what_it_concludes(
    output: str, expected: str, kind: str, right: bool
) -> None:
    assert reasoning.correct(output, expected, kind) is right


async def test_exact_answer_scores_only_a_reasoning_item() -> None:
    case = SimpleNamespace(metadata={"answer": "7", "answer_kind": "number"}, output="Answer: 7")
    score = await reasoning.exact_answer(case)
    assert score is not None and score.value
    assert await reasoning.exact_answer(SimpleNamespace(metadata={}, output="x")) is None


# --------------------------------------------------------------------------- the techniques
@lc_tool
def look(item: str) -> str:
    """Look an item up."""
    return f"{item} found"


async def _ran(model: ScriptedChatModel, *middleware: Any, messages: Any = None) -> Any:
    graph = create_agent(model, tools=[look], middleware=list(middleware))
    return await graph.ainvoke({"messages": messages or [HumanMessage("q")]})


async def test_thinking_is_the_templates_switch_on_every_call() -> None:
    model = ScriptedChatModel(turns=[("look", {"item": "a"}), "Answer: 1"])
    await _ran(model, techniques.TECHNIQUES["think"]())
    assert all(r["extra_body"] == techniques.THINKING for r in model.requests)


async def test_a_run_keeps_its_slot_and_settings_apply_to_calls_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedChatModel(turns=["ok"])
    await _ran(model, techniques.SlotReuse(2))
    assert "extra_body" not in model.requests[0]  # outside a run: as it is
    monkeypatch.setattr(techniques, "current", lambda: SimpleNamespace(run_id="run-1"))
    slotted = ScriptedChatModel(turns=["ok", "ok"])
    thinking = techniques.TECHNIQUES["think"]()
    await _ran(slotted, techniques.SlotReuse(2), thinking)
    await _ran(slotted, techniques.SlotReuse(2), thinking)
    bodies = [r["extra_body"] for r in slotted.requests]
    assert bodies[0] == bodies[1] and bodies[0]["cache_prompt"] is True
    assert bodies[0]["id_slot"] in (0, 1) and bodies[0]["chat_template_kwargs"]  # merged
    cold = ScriptedChatModel(turns=["ok"])
    await _ran(cold, techniques.TECHNIQUES["temp0"]())
    assert cold.requests[0]["temperature"] == 0
    serial = ScriptedChatModel(turns=["ok"])
    await _ran(serial, techniques.TECHNIQUES["serial-tools"]())
    assert serial.requests[0]["parallel_tool_calls"] is False
    keyed = ScriptedChatModel(turns=["ok"])
    await _ran(keyed, techniques.TECHNIQUES["openai-cache-key"]())
    assert keyed.requests[0]["prompt_cache_key"] == "trellis-react"


async def test_settings_for_calls_with_tools_leave_a_call_without_them() -> None:
    model = ScriptedChatModel(turns=["ok"])
    graph = create_agent(model, tools=[], middleware=[techniques.TECHNIQUES["serial-tools"]()])
    await graph.ainvoke({"messages": [HumanMessage("q")]})
    assert "parallel_tool_calls" not in model.requests[0]


def test_anthropic_caching_is_langchain_anthropics_middleware() -> None:
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

    assert isinstance(techniques.TECHNIQUES["anthropic-cache"](), AnthropicPromptCachingMiddleware)


async def test_the_examples_are_the_dev_items_only() -> None:
    added: list[Any] = []

    async def add_document(file: Any, **kwargs: Any) -> None:
        added.append(file[1].decode())

    await techniques.seed_examples(SimpleNamespace(memory=object(), add_document=add_document), "u")
    dev = [i for i in reasoning.items() if i["metadata"]["split"] == "dev"]
    assert len(added) == len(dev) == 20
    tests = [i["input"] for i in reasoning.items() if i["metadata"]["split"] == "test"]
    assert not any(question in text for question in tests for text in added)
    with pytest.raises(RuntimeError, match="--memory"):
        await techniques.seed_examples(SimpleNamespace(memory=None), "u")


def _setup() -> Setup:
    return Setup(system="s", model=ScriptedChatModel(turns=[]), window=16_384, max_steps=4)


@pytest.mark.parametrize("name", list(techniques.TECHNIQUES))
def test_a_technique_arm_is_main_with_its_middleware(name: str) -> None:
    graph, options = VARIANTS[name].build(_setup())
    assert options == {} and graph is not None


def test_the_code_and_examples_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    import trellis
    from tests.support.sandbox import FakeSandboxes

    real = trellis.sandbox
    monkeypatch.setattr(trellis, "sandbox", lambda: real(FakeSandboxes()))
    graph, _ = VARIANTS["code"].build(_setup())
    assert "execute" in {t.name for t in graph_tools(graph)}
    _, options = VARIANTS["examples"].build(_setup())
    assert options == {"user": techniques.EXAMPLES_USER, "prepare": techniques.seed_examples}
