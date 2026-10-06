"""Extension points: features in flight, each a probe of the API the plan proposes. Every cell
is a strict xfail with the gap or plan item: the day the feature lands its probe passes, the
suite fails on XPASS, and the row is turned into a real one (a scenario in ``features.py``,
its notes in its ``Feature``, its switch in ``dimensions.SWITCHES``).

* ``F71`` hooks (W6, G4): ``h.wrap(..., hooks=[...])`` before/after/on_error around the run, the
  model call and the tool call.
* ``F34`` approval rule in code (W5, G9): ``@tool(approval=fn)``.
* ``F36v2`` HITL v2 (W5, G9): ``ask(options=[Option(...)], multiple=True)``.
* ``F73`` sandbox (W7): ``trellis.harness.sandbox`` — a provider whose ops are harness tools.
* ``F70`` the framework's own run options (G13): ``agent.run(..., framework_options=...)``.
* ``F75`` per-agent selection (G2): ``h.wrap(..., without={...})`` — the pending switches of
  ``dimensions.PENDING`` (one selection cell each).
"""

from __future__ import annotations

from typing import Any, Final

from tests.matrix.kit import Desk
from tests.matrix.model import ADAPTERS, NA, Feature, Gap
from tests.matrix.world import USER, World
from trellis import tool


async def hooks(w: World) -> None:
    seen: list[str] = []

    async def before_tool(call: Any) -> None:
        seen.append(f"before {call.tool}")

    d = Desk()
    o = await w.go([d.lookup()], [("lookup", {"topic": "x"})], hooks=[before_tool])
    o.succeeded()
    assert seen == ["before lookup"], seen


async def approval_fn(w: World) -> None:
    def big(args: dict[str, Any]) -> bool:
        return args.get("amount", 0) > 100

    proposed: dict[str, Any] = {"approval": big}

    @tool(side_effects="write", **proposed)
    async def pay(amount: int) -> str:
        """Pay."""
        return f"paid {amount}"

    o = (await w.go([pay], [("pay", {"amount": 500})])).succeeded()
    assert len(o.pauses) == 1


async def hitl_v2(w: World) -> None:
    from trellis.contracts import Option

    @tool(side_effects="read")
    async def pick(item: str) -> str:
        """Pick sizes."""
        from trellis import current

        runtime = current()
        assert runtime is not None
        proposed: dict[str, Any] = {"options": [Option(value="S", label="Small")], "multiple": True}
        chosen = await runtime.ask("Which sizes?", **proposed)
        return f"{item}: {chosen}"

    def answer(interrupt: Any) -> tuple[str, Any]:
        return "answer", ["S"]

    o = (await w.go([pick], [("pick", {"item": "shirt"})], answer=answer)).succeeded()
    assert "S" in o.text


async def sandbox(w: World) -> None:
    import importlib

    module = importlib.import_module("trellis.harness.sandbox")
    assert hasattr(module, "SandboxProvider")  # landed: write its scenario in features.py


async def framework_options(w: World) -> None:
    agent = await w.agent([], [])
    proposed: dict[str, Any] = {"framework_options": {"recursion_limit": 5}}
    result = await agent.run("x", user=USER, **proposed)
    assert result.status.value == "SUCCESS"


def _pending(feature_id: str, title: str, *, audit: str, how: str, gap: Gap, probe: Any) -> Feature:
    return Feature(
        feature_id,
        title,
        audit,
        how,
        probe,
        adapters=dict.fromkeys(ADAPTERS, gap),
        way2=NA("probed in Way 1; the plan names no Way 2 form yet"),
    )


EXTENSIONS: Final[list[Feature]] = [
    _pending(
        "F71",
        "hooks around run, model and tool",
        audit="F71",
        how="h.wrap(..., hooks=[...]) (W6)",
        gap=Gap("G4", "no hooks yet"),
        probe=hooks,
    ),
    _pending(
        "F34",
        "an approval rule in code",
        audit="F34",
        how="@tool(approval=fn) (W5)",
        gap=Gap("G9", "no approval= on tool() yet"),
        probe=approval_fn,
    ),
    _pending(
        "F36v2",
        "HITL v2: options with labels, several answers",
        audit="F36 (W5)",
        how="ask(options=[Option], multiple=True)",
        gap=Gap("G9", "ask takes plain string options only"),
        probe=hitl_v2,
    ),
    _pending(
        "F73",
        "sandbox: a provider's ops as harness tools",
        audit="F73 (W7)",
        how="trellis.harness.sandbox",
        gap=Gap("W7", "no sandbox yet"),
        probe=sandbox,
    ),
    _pending(
        "F70",
        "the framework's own run options",
        audit="F70",
        how="agent.run(..., framework_options=)",
        gap=Gap("G13", "no way to pass the framework's own per-run options"),
        probe=framework_options,
    ),
]
