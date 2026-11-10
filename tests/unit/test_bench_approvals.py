"""What the suite approves when a run pauses to ask a person (``benchmark.cost.approvals``): a
call the harness's own check reads as read-only, or as writing only in the run's folder
(``tests/unit/test_governance_shell.py``); anything else stays a pause."""

from __future__ import annotations

from types import SimpleNamespace

from benchmark.cost import breakfix
from benchmark.cost import complex as cx
from benchmark.cost.approvals import APPROVALS_MAX, approved_reads


class _Run:
    """A run's result as ``agent.run``/``resume`` give it: paused on ``command``, or done."""

    def __init__(self, command: str | None) -> None:
        from trellis.contracts import RunStatus, ToolCall

        self.status = RunStatus.PAUSED if command is not None else RunStatus.SUCCESS
        self.interrupt = (
            SimpleNamespace(
                interrupt_id=f"int-{command}",
                tool_call=ToolCall(tool="Bash", args={"command": command}),
            )
            if command is not None
            else None
        )


class _Agent:
    """Resumed, it pauses on its next command (``None``: it answers)."""

    def __init__(self, *commands: str | None) -> None:
        self.next = list(commands)
        self.resumed: list[tuple[str, str, str]] = []

    async def resume(self, interrupt_id: str, decision: str, *, reviewer: str) -> _Run:
        self.resumed.append((interrupt_id, decision, reviewer))
        return _Run(self.next.pop(0))


async def test_a_pause_on_a_read_is_resumed_approved_until_the_run_answers(tmp_path) -> None:
    agent = _Agent("head -3 s.csv", None)
    result, approved = await approved_reads(agent, _Run("wc -l s.csv"), "reviewer", tmp_path)
    assert result.interrupt is None and approved == ["wc -l s.csv", "head -3 s.csv"]
    assert agent.resumed == [
        ("int-wc -l s.csv", "approve", "reviewer"),
        ("int-head -3 s.csv", "approve", "reviewer"),
    ]


async def test_a_pause_on_a_write_stays_a_pause(tmp_path) -> None:
    agent = _Agent()
    result, approved = await approved_reads(agent, _Run("rm s.csv"), "reviewer", tmp_path)
    assert result.interrupt is not None and approved == [] and agent.resumed == []
    agent = _Agent("rm s.csv")
    result, approved = await approved_reads(agent, _Run("cat s.csv"), "reviewer", tmp_path)
    assert result.interrupt is not None and approved == ["cat s.csv"]  # then a write: a pause


async def test_a_run_that_reads_over_and_over_is_left_paused_past_the_limit(tmp_path) -> None:
    agent = _Agent(*["cat s.csv"] * APPROVALS_MAX)
    result, approved = await approved_reads(agent, _Run("cat s.csv"), "reviewer", tmp_path)
    assert result.interrupt is not None and len(approved) == APPROVALS_MAX


def test_a_cannot_do_turn_whose_run_ran_code_is_scored_on_what_it_did() -> None:
    """With a shell of its own (Claude Code, Kimi), a request the suite wrote as "no code
    execution here" can be done: a run that ran a command (approved, read-only) and gives the
    result answered it, and its figures are not made up. A file or link the user cannot reach
    is still made up; a refusal is still right."""
    case = next(c for c in breakfix.cases() if c.id == "bf3-cantdo-01")
    expect = case.turns[0].expect
    assert expect is not None
    did = "I fitted ARIMA(1,1,1) to monthly revenue: the model forecasts £2,140,000 for Q4 2026."

    def ok(answer: str, *, ran: bool) -> bool:
        return cx.score(expect, answer, [], ran=ran)["correct"]

    assert ok(did, ran=True)
    assert not ok(did, ran=False)  # with no code run, that forecast is made up
    assert ok(expect.reference, ran=True)  # declining is still right
    assert not ok(did + " [download](sandbox:/q4.xlsx)", ran=True)
    assert not ok("", ran=True)


async def test_a_run_saving_in_its_folder_is_resumed_and_the_save_listed(tmp_path) -> None:
    agent = _Agent("echo done > status.txt", None)
    result, approved = await approved_reads(agent, _Run("mkdir -p exports"), "r", tmp_path)
    assert result.interrupt is None and approved == ["mkdir -p exports", "echo done > status.txt"]
    agent = _Agent(None)
    result, approved = await approved_reads(agent, _Run("mkdir -p exports"), "r")
    assert result.interrupt is not None and approved == []  # no folder given: a pause
    result, approved = await approved_reads(agent, _Run("cat s.csv"), "r")
    assert result.interrupt is not None and approved == []  # nor is a read placed: a pause
