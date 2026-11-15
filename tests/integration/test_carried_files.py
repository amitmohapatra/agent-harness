"""A conversation's working files carry from one turn to the next (``sandbox.carry``): the
files a run's commands wrote in its sandbox — here this machine's isolation layer, a real
workspace and real commands — are kept in the file store when the run succeeds or pauses, and
given back to the conversation's next run, on any worker, in its fresh workspace; never to
another conversation, user or tenant. Skips where the layer cannot isolate."""

from __future__ import annotations

import json
import secrets
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.isolation import fresh_root, root_helper
from tests.support.models import ScriptedChat
from trellis import Harness, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness.adapters.acp import KIMI, ACPAgent
from trellis.harness.react import ReAct
from trellis.harness.sandbox import EXEC, sandbox
from trellis.harness.sandbox.carry import FILE_BYTES
from trellis.harness.sandbox.isolation import HostSandbox
from trellis.memory.models import KeyInfo

#: The scripted ACP agent, as Kimi Code.
FAKE_ACP = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")
#: What a run is told of the files its conversation gave back.
RESTORED = "Restored from this conversation's earlier turns"


@pytest.fixture
def layer() -> Iterator[HostSandbox]:
    """This machine's isolation layer, a root and a uid range of the test's own."""
    helper = root_helper()
    for root in fresh_root(helper):
        base = 1_700_000_000 + secrets.randbelow(100_000) * 1_000
        yield HostSandbox(root, helper=helper, uid_base=base, tenants=2)


@pytest.fixture
async def harnesses() -> AsyncIterator[tuple[Harness, Harness]]:
    """Two workers with the same file store: a turn on one, the next on the other."""
    async with Harness(config=Settings()) as first, Harness(config=Settings()) as second:
        yield first, second
    AGENTS.clear()


@tool(side_effects="write")
def ship(name: str) -> str:
    """Ship a build to the customers."""
    return f"shipped {name}"


#: Each worker's agent and its model: wrapped once by its harness, its model given each turn's
#: script.
AGENTS: dict[int, tuple[Any, ScriptedChat]] = {}


async def turn(
    harness: Harness, layer: HostSandbox, turns: list[Any], **kw: Any
) -> tuple[Any, list[dict[str, Any]]]:
    """One run of a ReAct agent with the layer's sandbox tools (and :func:`ship`): its result,
    and the requests its model was sent."""
    if id(harness) not in AGENTS:
        model = ScriptedChat()
        tools = [sandbox(layer).tools(), ship]
        AGENTS[id(harness)] = arm(harness.wrap(ReAct("s", model), id="carrier"), tools), model
    agent, model = AGENTS[id(harness)]
    model.turns, sent = turns, len(model.requests)
    result = await agent.run("go", user=kw.pop("user", "u1"), **kw)
    return result, model.requests[sent:]


def told(requests: list[dict[str, Any]], call: int = 1) -> str:
    """What the model was told of a tool call (in its request after it)."""
    return str(requests[call]["messages"][-1]["content"])


def run(command: str) -> tuple[str, dict[str, Any]]:
    return (EXEC, {"command": command})


async def test_a_conversations_working_files_carry_to_its_next_turn_on_another_worker(
    layer: HostSandbox, harnesses: tuple[Harness, Harness], caplog: Any
) -> None:
    first, second = harnesses
    made = (
        "mkdir -p data outputs __pycache__ .cache && echo kept > data/a.txt"
        " && echo out > outputs/r.txt && echo c > __pycache__/x && echo c > .cache/y"
    )
    done, _ = await turn(first, layer, [run(made), "made"], thread="t1")
    assert done.status is RunStatus.SUCCESS, done
    script = [run("cat data/a.txt; ls -A"), run("true"), "read"]
    again, sent = await turn(second, layer, script, thread="t1")
    assert again.status is RunStatus.SUCCESS, again
    said = json.loads(told(sent))
    assert said["stdout"] == "kept\n.results\n.tmp\ndata\n"  # not the outputs, nor caches
    assert RESTORED in said["restored"] and "data/a.txt" in said["restored"]  # in one line
    assert "restored" not in json.loads(told(sent, 2))  # told once: with its first result
    # kept in a folder of its own: the graph state's files beside it are read as ever
    assert not [r for r in caplog.records if "carries was not read" in r.getMessage()]


@pytest.mark.parametrize(
    "other", [{"thread": "t2"}, {"thread": "t1", "user": "u2"}, {"thread": "t1", "tenant": "acme"}]
)
async def test_another_conversation_user_or_tenant_gets_none_of_them(
    layer: HostSandbox, harnesses: tuple[Harness, Harness], other: dict[str, str]
) -> None:
    first, second = harnesses
    mine: dict[str, str] = {}
    if "tenant" in other:  # a platform key: each call names its tenant
        platform = KeyInfo(key_id="k", tenant_id=None, principal="p", role="service")

        async def key() -> KeyInfo:
            return platform

        for h in harnesses:
            h._key_info = key  # type: ignore[method-assign]
        mine = {"tenant": "default"}
    await turn(first, layer, [run("echo kept > a.txt"), "made"], thread="t1", **mine)
    result, sent = await turn(second, layer, [run("cat a.txt"), "read"], **other)
    assert result.status is RunStatus.SUCCESS, result
    assert RESTORED not in told(sent) and "kept" not in told(sent)


async def test_a_file_past_the_largest_carried_is_left_out_and_the_run_told(
    layer: HostSandbox, harnesses: tuple[Harness, Harness]
) -> None:
    first, second = harnesses
    big = f"head -c {FILE_BYTES + 1} /dev/zero > big.bin && echo kept > small.txt"
    done, _ = await turn(first, layer, [run(big), "made"], thread="t1")
    assert done.status is RunStatus.SUCCESS, done
    _, sent = await turn(second, layer, [run("ls"), "read"], thread="t1")
    said = json.loads(told(sent))
    assert said["stdout"] == "small.txt\n"  # the large one left out
    assert "small.txt" in said["restored"] and "big.bin" in said["restored"]  # said so


async def test_a_paused_run_keeps_them_a_failed_one_keeps_nothing_new(
    layer: HostSandbox, harnesses: tuple[Harness, Harness]
) -> None:
    first, second = harnesses
    script = [run("echo one > a.txt"), ("ship", {"name": "b"})]  # a person is asked
    paused, _ = await turn(first, layer, script, thread="t1")
    assert paused.status is RunStatus.PAUSED, paused  # its files kept as it paused
    script = [run("cat a.txt; echo two > a.txt")]  # then the model's endpoint fails
    failed, sent = await turn(second, layer, script, thread="t1")
    assert failed.status is RunStatus.ERROR, failed
    assert "one" in told(sent)
    _, after = await turn(second, layer, [run("cat a.txt"), "read"], thread="t1")
    assert "one" in told(after) and "two" not in told(after)


async def test_a_sandboxed_kimis_files_carry_to_its_next_turn(
    layer: HostSandbox, harnesses: tuple[Harness, Harness], tmp_path: Path
) -> None:
    """Kimi Code's terminal in the layer's workspace (its working folder there): what its
    ``Bash`` wrote comes back next turn, told before its first command's output."""
    answers: list[str] = []
    for harness, command in zip(harnesses, ("echo kept > a.txt", "cat a.txt"), strict=True):
        script = [{"builtin": "Bash", "args": {"command": command}}, {"text": "{last}"}]
        target = ACPAgent(
            command=(sys.executable, FAKE_ACP),
            env={
                "FAKE_ACP_SCRIPT": json.dumps(script),
                "FAKE_ACP_RECORD": str(tmp_path / "record.json"),
                "KIMI_CODE_HOME": str(tmp_path / "home"),
            },
            cwd=str(tmp_path),
            profile=KIMI,
            sandbox=sandbox(layer),
        )
        done = await harness.wrap(target, id="kimi").run("go", user="u1", thread="t1")
        assert done.status is RunStatus.SUCCESS, done
        answers.append(str(done.answer))
    assert answers[1].startswith(RESTORED) and answers[1].endswith("a.txt.\nkept\n")
