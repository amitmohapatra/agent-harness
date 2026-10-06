"""The tools the scenarios give their agents, and what those tools did (a :class:`Desk` per
cell: the outside world they touch)."""

from __future__ import annotations

import asyncio
from typing import Any, Final

from trellis import current, tool
from trellis.contracts import ToolError
from trellis.harness.journal import MAX_CHECKPOINT_BYTES

#: What a write past its time limit reads as, to the model.
UNKNOWN: Final = (
    "transfer timed out after 0.05s; it may or may not have taken effect: check before "
    "calling it again"
)
EMAIL: Final = "ada@example.com"
SECRET: Final = "sk-live0123456789abcdefghij"
#: A tool output larger than a checkpoint may be (the journal goes to an artifact).
HUGE: Final = MAX_CHECKPOINT_BYTES + 1
#: How long ``wait`` waits at most (a cancel or a time limit ends it sooner).
WAIT_SECONDS: Final = 1.0


class Desk:
    """The tools of one cell, and everything they did."""

    def __init__(self) -> None:
        self.done: list[tuple[str, dict[str, Any]]] = []
        self.keys: list[str | None] = []
        self.quotes = 0
        self.inflight = 0
        self.overlapped = False
        self.started = asyncio.Event()

    def ran(self, name: str) -> list[dict[str, Any]]:
        return [args for tool_name, args in self.done if tool_name == name]

    # ------------------------------------------------------------------ by risk
    def lookup(self) -> Any:
        @tool(side_effects="read")
        async def lookup(topic: str) -> str:
            """Look a topic up."""
            self.done.append(("lookup", {"topic": topic}))
            return f"facts about {topic}"

        return lookup

    def note(self) -> Any:
        @tool(side_effects="write")
        async def note(text: str) -> str:
            """Write a note down."""
            runtime = current()
            self.keys.append(runtime.idempotency_key if runtime else None)
            self.done.append(("note", {"text": text}))
            return f"noted {text}"

        return note

    def refund(self) -> Any:
        @tool(side_effects="irreversible")
        async def refund(order: str) -> str:
            """Refund an order."""
            self.done.append(("refund", {"order": order}))
            return f"refunded {order}"

        return refund

    # ------------------------------------------------------------------ reliability
    def quote(self) -> Any:
        @tool(side_effects="read")
        async def quote(sku: str) -> str:
            """The price of a SKU (the price service is busy at first)."""
            self.quotes += 1
            if self.quotes < 3:
                raise ToolError("the price service is busy", retryable=True)
            return f"{sku} costs 7"

        return quote

    def transfer(self) -> Any:
        @tool(side_effects="write", timeout=0.05)
        async def transfer(amount: int) -> str:
            """Transfer money (the bank never answers in time)."""
            self.done.append(("transfer", {"amount": amount}))
            await asyncio.sleep(2)
            return "sent"

        return transfer

    def slow(self) -> Any:
        @tool(side_effects="read", timeout=0.05)
        async def slow(sku: str) -> str:
            """A read that takes too long."""
            await asyncio.sleep(2)
            return "late"

        return slow

    def wait(self) -> Any:
        @tool(side_effects="read")
        async def wait(seconds: int) -> str:
            """Wait a while."""
            self.started.set()
            await asyncio.sleep(min(seconds, WAIT_SECONDS))
            self.done.append(("wait", {"seconds": seconds}))
            return "waited"

        return wait

    # ------------------------------------------------------------------ people
    def size(self) -> Any:
        @tool(side_effects="read")
        async def size(item: str) -> str:
            """Ask which size the item should be."""
            runtime = current()
            assert runtime is not None
            chosen = await runtime.ask(f"Which size of {item}?", options=["S", "L"])
            self.done.append(("size", {"item": item, "size": chosen}))
            return f"{item} in size {chosen}"

        return size

    # ------------------------------------------------------------------ data
    def notify(self) -> Any:
        @tool(side_effects="write")
        async def notify(email: str, api_key: str) -> dict[str, str]:
            """Notify a customer."""
            self.done.append(("notify", {"email": email, "api_key": api_key}))
            return {"sent_to": email, "receipt": api_key}

        return notify

    def report(self, chars: int = 300) -> Any:
        text = "".join(f"{n:04d}" for n in range(chars // 4 + 1))[:chars]

        @tool(side_effects="read")
        async def report(name: str) -> str:
            """A long report."""
            self.done.append(("report", {"name": name}))
            return text

        return report

    def export(self) -> Any:
        @tool(side_effects="write")
        async def export(rows: int) -> str:
            """Export a report (larger than a checkpoint may be)."""
            self.done.append(("export", {"rows": rows}))
            return "x" * HUGE

        return export

    def parallel(self, name: str, side_effects: str) -> Any:
        @tool(name=name, side_effects=side_effects)  # type: ignore[arg-type]
        async def call(key: str) -> str:
            """A call that notices when another runs at the same time."""
            self.inflight += 1
            self.overlapped = self.overlapped or self.inflight > 1
            await asyncio.sleep(0.05)
            self.inflight -= 1
            self.done.append((name, {"key": key}))
            return f"{name} {key}"

        return call
