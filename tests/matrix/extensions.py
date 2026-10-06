"""Extension points: features in flight, each a probe of the API the plan proposes. Every cell
is a strict xfail with the gap or plan item: the day the feature lands its probe passes, the
suite fails on XPASS, and the row is turned into a real one (a scenario in ``features.py``,
its notes in its ``Feature``, its switch in ``dimensions.SWITCHES``).

* ``F34`` approval rule in code (W5): a ``before_tool`` hook asking by the call's arguments on a
  ``write`` tool — a real row.
* ``F36v2`` HITL v2 (W5): ``ask(options=[Option(...)], multiple=True)`` — landed, a real row now.
* the switches ``without=`` does not name yet (``dimensions.PENDING``: governance, redaction),
  one selection cell each.

Hooks (F71*), ``without=`` (the selection's switches, F75r), ``timeout=`` (F09, F09r), the
framework's own run options (F70, G13) and the sandbox (F73) have landed: they are rows of
``features.py``.
"""

from __future__ import annotations

from typing import Any, Final

from tests.matrix.model import ADAPTERS, NA, Bug, Feature, Gap
from tests.matrix.world import World
from trellis import Ask, Hooks, tool
from trellis.contracts import ToolCall


class _OverHundred(Hooks):
    """An approval rule in code: a payment over 100 asks; a smaller one runs unasked."""

    async def before_tool(self, call: ToolCall) -> Ask | None:
        return Ask("Over 100: pay it?") if call.args.get("amount", 0) > 100 else None


async def approval_rule(w: World) -> None:
    @tool(side_effects="write")
    async def pay(amount: int) -> str:
        """Pay."""
        return f"paid {amount}"

    plan = [("pay", {"amount": 500}), ("pay", {"amount": 50})]
    o = (await w.go([pay], plan, hooks=[_OverHundred()])).succeeded()
    assert len(o.pauses) == 1, o.pauses  # the big payment asks; the small one runs
    assert "Over 100: pay it?" in o.pauses[0].question, o.pauses


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


def _pending(
    feature_id: str, title: str, *, audit: str, how: str, gap: Gap | Bug | None, probe: Any
) -> Feature:
    return Feature(
        feature_id,
        title,
        audit,
        how,
        probe,
        adapters={} if gap is None else dict.fromkeys(ADAPTERS, gap),
        way2=NA("probed in Way 1; the plan names no Way 2 form yet"),
    )


EXTENSIONS: Final[list[Feature]] = [
    _pending(
        "F34",
        "an approval rule in code",
        audit="F34",
        how="a before_tool hook returning Ask by the call's arguments; the tool write (W5)",
        gap=None,
        probe=approval_rule,
    ),
    _pending(
        "F36v2",
        "HITL v2: options with labels, several answers",
        audit="F36 (W5)",
        how="ask(options=[Option], multiple=True)",
        gap=None,
        probe=hitl_v2,
    ),
]
