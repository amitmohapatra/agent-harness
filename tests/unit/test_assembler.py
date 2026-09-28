"""The assembler renders the prompt under budget and compacts what outgrows it."""

from __future__ import annotations

from typing import Any

import pytest
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.model import ModelResponse

from trellis.harness.reasoning import ContextAssembler
from trellis.harness.reasoning.assembler import estimate_tokens


class _Model:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **kwargs: Any) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(text="The user wants a refund for order 91.", model="m")


class _Memory:
    enabled = True

    def __init__(self) -> None:
        self.observed: list[MemoryObservation] = []

    async def observe(self, observation: MemoryObservation, /) -> Any:
        self.observed.append(observation)
        return {"observation_id": "obs_1"}


class _Bundle:
    rendered = "- [mem_1] Amit's timezone is Europe/Berlin\n- [mem_2] prefers concise answers"


class _Runtime:
    def __init__(self) -> None:
        self.model = _Model()
        self.memory = _Memory()
        self.memory_context = _Bundle()
        self.state: dict[str, Any] = {}


class _Skill:
    skill_id = "billing.refund"
    description = "Refunds within policy"


def test_the_system_prompt_names_skills_and_memory_under_budget() -> None:
    assembler = ContextAssembler(
        "You are the refund agent.", skills=[_Skill(), "search"], memory_tokens=6
    )
    prompt = assembler.system_prompt(_Runtime())
    assert prompt.startswith("You are the refund agent.")
    assert "billing.refund: Refunds within policy" in prompt and "- search" in prompt
    assert (
        "What is remembered" in prompt and "…" in prompt
    )  # the bundle was cut to the memory share
    bare = ContextAssembler("prompt").system_prompt(
        type("R", (), {"memory_context": None, "state": {}})()
    )
    assert bare == "prompt"


async def test_compaction_summarises_older_turns_and_remembers_the_summary() -> None:
    assembler = ContextAssembler("prompt", budget_tokens=40, keep_recent=2)
    runtime = _Runtime()
    turns = [{"role": "system", "content": "prompt"}] + [
        {"role": "user" if i % 2 else "assistant", "content": f"turn {i} " + "x" * 30}
        for i in range(8)
    ]
    assert assembler.over_budget(turns)
    compacted = await assembler.compact(runtime, turns)
    assert [t["role"] for t in compacted] == ["system", "assistant", "assistant", "user"]
    assert compacted[1]["content"].startswith("Summary of the conversation so far: The user wants")
    assert compacted[-2:] == turns[-2:]
    assert assembler.compactions == 1
    assert runtime.memory.observed[0].content.startswith("Conversation summary:")
    assert runtime.memory.observed[0].kind == "AGENT_RESULT"
    assert "turn 0" in runtime.model.requests[0].messages[1]["content"]
    assert runtime.model.requests[0].metadata == {"internal": "compaction"}
    assert runtime.memory.observed[0].hints == {"visibility": "RUN"}
    assert runtime.memory.observed[0].metadata["source"] == "compaction"
    short = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "hi"}]
    assert await assembler.compact(runtime, short) == short and assembler.compactions == 1
    headless = [{"role": "user", "content": "x" * 40}] * 5
    with pytest.raises(ValueError, match="system turn"):
        await assembler.compact(runtime, headless)


async def test_tool_observations_stay_out_of_the_summary_unless_the_policy_allows() -> None:
    """A tool result is customer data until the memory policy says otherwise; the summary
    the model writes must not launder it into a durable memory."""

    class _Policy:
        observe_tool_results = False

    runtime = _Runtime()
    runtime.memory.policy = _Policy()
    assembler = ContextAssembler("prompt", budget_tokens=10, keep_recent=1)
    turns = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "refund order 91 please"},
        {"role": "assistant", "content": "Calling lookup"},
        {"role": "user", "content": "Observation (data returned by the tool): card 4111 1111"},
        {"role": "assistant", "content": "done"},
    ]
    await assembler.compact(runtime, turns)
    transcript = runtime.model.requests[-1].messages[1]["content"]
    assert "refund order 91" in transcript and "4111" not in transcript
    runtime.memory.policy.observe_tool_results = True
    await assembler.compact(runtime, turns)
    assert "4111" in runtime.model.requests[-1].messages[1]["content"]
    assert estimate_tokens("") == 0 and estimate_tokens("abcd" * 3) == 3
