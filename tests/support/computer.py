"""A computer for the OpenAI Agents SDK's ``ComputerTool`` in tests: it does nothing but keep
what it was asked to do."""

from __future__ import annotations

from agents import Computer


class Screen(Computer):
    """Every action kept in :attr:`done`; a screenshot is a blank image's text."""

    def __init__(self) -> None:
        self.done: list[str] = []

    @property
    def environment(self) -> str:  # type: ignore[override]
        return "linux"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (800, 600)

    def screenshot(self) -> str:
        self.done.append("screenshot")
        return "aW1hZ2U="

    def click(self, x: int, y: int, button: str) -> None:  # type: ignore[override]
        self.done.append(f"click {x} {y}")

    def double_click(self, x: int, y: int) -> None:
        self.done.append(f"double_click {x} {y}")

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.done.append(f"scroll {scroll_x} {scroll_y}")

    def type(self, text: str) -> None:
        self.done.append(f"type {text}")

    def wait(self) -> None:
        self.done.append("wait")

    def move(self, x: int, y: int) -> None:
        self.done.append(f"move {x} {y}")

    def keypress(self, keys: list[str]) -> None:
        self.done.append(f"keypress {'+'.join(keys)}")

    def drag(self, path: list[tuple[int, int]]) -> None:
        self.done.append(f"drag {len(path)}")
