"""Scenario 1 — one chat, several messages: sent one after another, in a burst (2, 5, 20), and to
a busy conversation with ``mode`` queue, interrupt and steer. Every message is answered once, in
order, with the earlier turns in what reached the model, and the memory service keeps every turn
once."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.scenarios.brains import ADAPTERS, STEERS, command, parse, token, wrapped
from tests.scenarios.kit import TOOLS, answered, asked, history
from trellis import Harness
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(900)


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_messages_one_after_another(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s1seq-{adapter}-{uid}")
    user, thread = f"ada-{uid}", f"chat-{uid}"
    tokens = [token(uid, n) for n in range(1, 5)]
    for n, tok in enumerate(tokens):
        extra = f" {command('lookup', key=f'k{n}')}" if adapter != "claude_agent_sdk" else ""
        done = await agent.run(f"{tok} message {n}{extra}", user=user, thread=thread)
        assert done.status is RunStatus.SUCCESS, (adapter, n, done.error)
        said = parse(done.answer)
        assert said.mine == tok, (adapter, n, done.answer)
        assert set(tokens[:n]) <= said.seen, (adapter, n, "earlier turns missing", done.answer)
        assert said.seen <= set(tokens), (adapter, n, "a token from nowhere", done.answer)
        if extra:
            assert said.last == f"value-of-k{n}", done.answer
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    assert asked(kept, tokens) == tokens, kept  # every message once, in order
    assert answered(kept) == tokens, kept  # every answer once, in order


@pytest.mark.parametrize("burst", [2, 5, 20])
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_burst_of_messages(
    h: Harness, adapter: str, burst: int, uid: str, tmp_path: Path
) -> None:
    """``burst`` messages sent at once to one chat: queued in this process, each answered once
    with the turns answered before it in its history, never two at once."""
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s1burst-{adapter}-{uid}")
    user, thread = f"ada-{uid}", f"chat-{uid}"
    tokens = [token(uid, n) for n in range(1, burst + 1)]
    sent = [asyncio.create_task(agent.run(f"{t} burst", user=user, thread=thread)) for t in tokens]
    results = await asyncio.gather(*sent)
    for tok, done in zip(tokens, results, strict=True):
        assert done.status is RunStatus.SUCCESS, (adapter, tok, done.error)
        assert parse(done.answer).mine == tok, (adapter, tok, done.answer)
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    order = answered(kept)
    assert sorted(order) == sorted(tokens) and len(order) == burst, kept  # each answered once
    assert asked(kept, tokens) == order, kept  # each question before its own answer, once
    assert order == tokens, ("not answered in the order sent", order)
    # each turn was answered with the turns before it, and only those, in its history
    by_token = {t: parse(r.answer) for t, r in zip(tokens, results, strict=True)}
    for n, tok in enumerate(order):
        assert set(order[:n]) <= by_token[tok].seen, (adapter, tok, by_token[tok])
        assert not set(order[n + 1 :]) & by_token[tok].seen, ("a later turn seen", tok)


@pytest.mark.parametrize("mode", ["queue", "interrupt", "steer"])
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_message_to_a_busy_chat(
    h: Harness, adapter: str, mode: str, uid: str, tmp_path: Path
) -> None:
    """The first message works (a slow tool call); the second arrives with ``mode``."""
    claude = [{"tool": "wait", "args": {"seconds": 4}}]
    agent, built = await wrapped(
        h, adapter, TOOLS, tmp_path, id=f"s1busy-{adapter}-{uid}", claude=claude
    )
    user, thread = f"ada-{uid}", f"chat-{uid}"
    first_tok, second_tok = token(uid, 1), token(uid, 2)
    first = asyncio.create_task(
        agent.run(f"{first_tok} {command('wait', seconds=4)}", user=user, thread=thread)
    )
    await asyncio.sleep(2.0)  # the first run is in its slow call
    second = await agent.run(f"{second_tok} also this", user=user, thread=thread, mode=mode)
    done = await first
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    takes_steer = adapter in STEERS
    if mode == "queue" or (mode == "steer" and not takes_steer):
        assert done.status is RunStatus.SUCCESS and parse(done.answer).mine == first_tok
        assert second.status is RunStatus.SUCCESS, second.error
        assert parse(second.answer).mine == second_tok and first_tok in parse(second.answer).seen
        assert answered(kept) == [first_tok, second_tok], kept
    elif mode == "interrupt":
        assert done.status is RunStatus.CANCELLED, (done.status, done.answer)
        assert second.status is RunStatus.SUCCESS, second.error
        said = parse(second.answer)
        assert said.mine == second_tok and first_tok in said.seen, second.answer
        assert answered(kept) == [second_tok], kept  # the cancelled turn has no answer
        assert asked(kept, [first_tok, second_tok]) == [first_tok, second_tok], kept
    else:  # steered into the working run: its answer is the message's
        assert done.status is RunStatus.SUCCESS and second.status is RunStatus.SUCCESS
        assert second.answer == done.answer, (second.answer, done.answer)  # one answer covers both
        if adapter == "claude_agent_sdk":  # steered after the tool call: the hook's added context
            added = Path(f"{built.model}.added")
            assert added.exists() and second_tok in added.read_text(), (
                "the steer never reached Claude"
            )
        else:
            said = parse(second.answer)
            assert said.mine == second_tok and first_tok in said.seen, second.answer
        assert asked(kept, [first_tok, second_tok]) == [first_tok, second_tok], kept
        assert answered(kept).count(second_tok) == 1, kept
