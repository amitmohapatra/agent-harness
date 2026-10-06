"""Prompt and skill sources, live: a ``ReAct`` through the gateway with its instructions from a
``.md`` prompt and a skill from a ``SKILL.md`` folder, and prompts read from a Langfuse prompt
API served on this machine (no internet).

The model is small, slow and shared, so its tool choice is forced (``Forcing``) and each call
may wait long: the tests assert what the harness sent, offered, ran and pinned — not what the
model would have chosen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_gateway
from tests.live.support import Forcing, Sent, gateway_model
from tests.live.test_live_gateway_features import started
from tests.support import langfuse as lf
from trellis import ReAct
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.skills import LOAD_SKILL, READ_SKILL_FILE, SECTION

pytestmark = [pytest.mark.live, needs_gateway]


#: the gateway alone: no memory context or runs service in the way of a small, slow model
ALONE: dict[str, Any] = {"memory_url": None, "runs_url": None}
#: how long one model call may wait for the shared local model
WAIT_SECONDS = 600.0


def customs(events: list[Any], name: str) -> list[dict[str, Any]]:
    return [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and (e.data or {}).get("name") == name
    ]


async def test_a_react_through_the_gateway_with_a_folder_prompt_and_a_folder_skill(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "review.md").write_text(
        "---\nversion: 4\n---\nYou review SQL for the {{team}} team. Be brief.\n"
    )
    skill = tmp_path / "skills" / "sql-review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: sql-review\ndescription: Reviews SQL queries.\nversion: 1.0.0\n---\n"
        "Read rules.md, then name each rule the query breaks.\n"
    )
    (skill / "rules.md").write_text("No SELECT *.")
    async with live_harness(
        prompts_dir=str(tmp_path / "prompts"), skills_dir=str(tmp_path / "skills"), **ALONE
    ) as h:
        assert h.gateway is not None
        sent = Sent()
        target = ReAct(
            system="",
            model=gateway_model(h, sent, wait=WAIT_SECONDS),
            prompt="review",
            prompt_vars={"team": "data"},
            middleware=[Forcing([LOAD_SKILL, None])],
        )
        agent = h.wrap(
            target,
            id="live-sources",
            skills=["sql-review"],
        )
        events = [e async for e in agent.stream("Review: SELECT * FROM t", user="live")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    # what was sent: the folder's prompt as the instructions, then the skills section (the
    # pushed context's system message)
    first = sent.requests[0]["messages"]
    assert first[0]["content"].startswith("You review SQL for the data team. Be brief.")
    context = first[1]["content"]
    assert SECTION.splitlines()[0] in context and "- sql-review: Reviews SQL queries." in context
    offered = [t["function"]["name"] for t in sent.requests[0]["tools"]]
    assert {LOAD_SKILL, READ_SKILL_FILE} <= set(offered)  # what was offered
    calls = started(events)  # the forced call (a small model may add more beside it)
    assert LOAD_SKILL in calls and set(calls) <= {LOAD_SKILL, READ_SKILL_FILE}
    loaded = next(
        e.data["output"]
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT and e.data["tool"] == LOAD_SKILL
    )
    assert "Read rules.md, then name each rule" in str(loaded) and "- rules.md" in str(loaded)
    # what was pinned (journaled), and said
    assert customs(events, "prompt") == [
        {
            "name": "prompt",
            "prompt": "review",
            "version": "4",
            "source": f"prompts_dir({tmp_path / 'prompts'})",
        }
    ]
    assert customs(events, "skills") == [{"name": "skills", "versions": {"sql-review": "1.0.0"}}]


async def test_prompts_from_a_langfuse_prompt_api_on_this_machine() -> None:
    store = lf.LangfusePrompts(
        prompts={"triage": ["Old.", "Answer with one word: {{word}}."]},
        labels={("triage", "staging"): 1},
    )
    with store.serve() as host:
        async with live_harness(
            langfuse_host=host,
            langfuse_public_key=lf.PUBLIC,
            langfuse_secret_key=lf.SECRET,
            prompts_dir=None,
            **ALONE,
        ) as h:
            assert await h.prompt("triage", word="yes") == "Answer with one word: yes."
            assert await h.prompt("triage@staging") == "Old."
            assert await h.prompt("triage@1") == "Old."
            assert h.gateway is not None
            sent = Sent()
            target = ReAct(
                system="",
                model=gateway_model(h, sent, wait=WAIT_SECONDS),
                prompt="triage",
                prompt_vars={"word": "yes"},
                middleware=[Forcing([None])],
            )
            agent = h.wrap(target, id="live-langfuse")
            result = await agent.run("Is the sky blue?", user="live")
    assert result.status is RunStatus.SUCCESS, result.error
    assert ("triage", {"label": "production"}) in store.asked  # Langfuse's own API shape
    assert ("triage", {"version": "1"}) in store.asked
    assert sent.requests[0]["messages"][0]["content"].startswith("Answer with one word: yes.")
