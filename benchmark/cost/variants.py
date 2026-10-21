"""The variants the benchmark compares, each a ``(target, h.wrap options)`` built from the
same instructions and model. A new cost feature lands as one more entry of :data:`VARIANTS`
(with its ``without=`` the way to turn it off), so its saving is measured against these two.

* ``baseline`` — the plain tool-calling loop, everything off: LangChain's ``create_agent``
  with only what runs the harness's tools and bounds the loop (``HarnessTools``,
  ``StallGuard``, ``StepLimit``, ``PatchToolCallsMiddleware``, ``ModelHooks``), no result
  paging, no context editing, no summarization; ``without={"code_mode", "hints"}``.
* ``main`` — ``ReAct(...)`` as ``main`` builds it, nothing changed: result paging
  (``FilesystemMiddleware``: a result over 80,000 characters saved, a preview sent), older
  results cleared past half the window, summarization near its end, ``ToolGuidance`` and
  ``AnswerCheck`` off, Code Mode where it applies (read-only Code Mode MCP servers of the
  gateway; this benchmark's catalogues are the harness's own tools, so it does not).
* ``main-<piece>`` — ``main`` with one of its pieces off (:data:`PIECES`: ``paging``,
  ``context_editing``, ``summarization``), so each piece is measured on its own: the default
  middleware of that name replaced by one that does nothing (``ReAct(middleware=)`` replaces a
  default by name).

:data:`DEFAULT` are the variants a run compares unless ``--variants`` says.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from langchain.agents.middleware import AgentMiddleware

from trellis import ReAct


@dataclass(frozen=True)
class Setup:
    """What every variant is built from."""

    system: str
    model: Any  # a gateway model name, or a LangChain chat model
    window: int
    max_steps: int


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    build: Callable[[Setup], tuple[Any, dict[str, Any]]]


def baseline(setup: Setup) -> tuple[Any, dict[str, Any]]:
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from langchain.agents import create_agent

    from trellis.harness import middleware as ours
    from trellis.harness.react import _gateway_model

    graph = create_agent(
        _gateway_model(setup.model, timeout=None, window=setup.window),
        tools=[],
        system_prompt=setup.system,
        middleware=[
            ours.HarnessTools(),
            ours.StallGuard(ours.MAX_REPEATS),
            ours.StepLimit(setup.max_steps),
            PatchToolCallsMiddleware(),
            ours.ModelHooks(),
        ],
        checkpointer=ours.RunCheckpointer(),
        name="baseline",
    )
    return graph, {"without": {"code_mode", "hints"}}


def main(setup: Setup, *, off: tuple[str, ...] = ()) -> tuple[Any, dict[str, Any]]:
    target = ReAct(
        setup.system,
        setup.model,
        max_steps=setup.max_steps,
        context_window=setup.window,
        middleware=[Off(PIECES[piece]) for piece in off],
    )
    return target, {}


class Off(AgentMiddleware):
    """A default middleware turned off: one of its name that does nothing."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name


#: ``main``'s pieces a ``main-<piece>`` variant turns off, by the name of their middleware.
PIECES: Final = {
    "paging": "FilesystemMiddleware",
    "context_editing": "ContextEditingMiddleware",
    "summarization": "SummarizationMiddleware",
}


VARIANTS: Final[dict[str, Variant]] = {
    "baseline": Variant("baseline", "plain tools, everything off", baseline),
    "main": Variant(
        "main",
        "ReAct as main builds it: result paging, context editing, summarization; ToolGuidance off",
        main,
    ),
    **{
        f"main-{piece}": Variant(
            f"main-{piece}",
            f"main with {piece.replace('_', ' ')} off",
            functools.partial(main, off=(piece,)),
        )
        for piece in PIECES
    },
}
#: The variants a run compares by default.
DEFAULT: Final = ("baseline", "main")
