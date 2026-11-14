"""Scenario 7 — a long chat (LR-06, LR-02): 55 turns in one conversation. Every turn answered
with its own token; the request never outgrows the window it is fitted to (``ReAct``'s
``context_window``); an early fact is still recalled at the end (memory: the user's long-term
memories, read with ``memory_search``); and what the turns spent adds up exactly to what the
model reported, turn by turn (no turn charged twice, none missed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.scenarios.brains import SYSTEM, BrainChat, command, parse, token, wrapped
from tests.scenarios.kit import TOOLS, answered, eventually, history, user_scope
from tests.support.arm import arm
from trellis import Harness, ReAct
from trellis.contracts import RunEventType, RunStatus

pytestmark = [pytest.mark.timeout(1800), pytest.mark.plan("LR-06", "LR-02")]
TURNS = 55
#: what every scripted reply reports (``tests/support/models.py``, ``tests/support/kimi.py``)
REPLY_TOKENS = 15


async def _spent(agent: object, run_id: str) -> dict[str, float]:
    events = [e async for e in agent.events(run_id)]  # type: ignore[attr-defined]
    [end] = [e for e in events if e.type is RunEventType.RUN_FINISHED][-1:]
    return end.data["spent"]


@pytest.mark.parametrize(
    "adapter", ["react", "react_fitted", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_fifty_five_turns(h: Harness, adapter: str, uid: str, tmp_path: Path) -> None:
    if adapter == "react_fitted":  # a small window: every request fitted to it
        model = BrainChat()
        agent = arm(
            h.wrap(ReAct(SYSTEM, model, context_window=6_000), id=f"s7-{adapter}-{uid}"), TOOLS
        )
        built_model = model
    else:
        agent, built = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s7-{adapter}-{uid}")
        built_model = built.model
    user, thread = f"ada-{uid}", f"long-{uid}"
    fact = f"my locker code for {uid} is 4242"
    tokens = [token(uid, n) for n in range(1, TURNS + 1)]
    spent_tokens = spent_calls = 0.0
    for n, tok in enumerate(tokens):
        said = f"{tok} {fact}" if n == 0 else f"{tok} turn {n} " + "lorem ipsum " * 20
        if n == TURNS - 1 and adapter != "claude_agent_sdk":
            said = f"{tok} what is my locker code? {command('memory_search', query=f'locker code {uid}', kinds=['message'])}"
        done = await agent.run(said, user=user, thread=thread)
        assert done.status is RunStatus.SUCCESS, (n, done.error)
        assert parse(done.answer).mine == tok, (n, done.answer[:300])
        spent = await _spent(agent, done.run_id)
        spent_tokens += float(spent["tokens"])
        spent_calls += float(spent["model_calls"])
        if n == 0:  # the fact is the user's memory before the chat goes on
            await h.writes.drain()
            scope = await user_scope(h, user=user, agent_id=agent.id)

            async def indexed() -> bool:
                return any(
                    "4242" in i.text
                    for i in await scope.search(f"locker code {uid}", kinds=["message"])
                )

            assert await eventually(indexed, within=90)
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    assert answered(kept) == tokens, [t for t in tokens if t not in answered(kept)][
        :5
    ]  # each once, in order
    if adapter != "claude_agent_sdk":
        assert "4242" in done.answer, ("the early fact was not recalled", done.answer[:500])
    # what the turns spent is exactly what the model reported
    if adapter in ("react", "react_fitted"):
        requests = built_model.requests
        assert spent_calls == len(requests), (spent_calls, len(requests))
        assert spent_tokens == REPLY_TOKENS * len(requests), (spent_tokens, len(requests))
    if adapter == "kimi":
        assert spent_tokens == REPLY_TOKENS * len(built_model.requests), (
            spent_tokens,
            len(built_model.requests),
        )
    if adapter == "react_fitted":  # never past its window (4 characters a token, generously)
        biggest = max(len(json.dumps(r["messages"])) for r in built_model.requests)
        assert biggest / 4 <= 6_000 * 1.25, biggest
