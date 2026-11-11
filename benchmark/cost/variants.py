"""The variants the benchmark compares, each a ``(target, h.wrap options)`` built from the
same instructions and model. A new cost feature lands as one more entry of :data:`VARIANTS`
(its middleware left out the way to turn it off), so its saving is measured against these two.

* ``baseline`` — the plain tool-calling loop, everything off: LangChain's ``create_agent``
  with only what runs the harness's tools and bounds the loop (``HarnessTools``,
  ``StallGuard``, LangChain's ``ModelCallLimitMiddleware``, ``PatchToolCallsMiddleware``,
  ``ModelHooks``), no context editing, no summarization, no automatic feature (a large result
  is still kept and paged: ``HarnessTools`` runs the harness's tools with the convention every
  adapter's model reads, ``tools.results``); ``without={"code_mode", "tool_search"}``.
* ``main`` — ``ReAct(...)`` as this branch builds it, nothing changed: with no Deep Agents
  capability asked for, LangChain's ``create_agent`` with ``trellis.middleware()`` (and Deep
  Agents' ``PatchToolCallsMiddleware``) — planning on a long task, older results cleared past
  half the window, the request fitted to the window (``FitWindow``).
* ``deep`` — ``ReAct`` on Deep Agents (a backend given: the journal's), ``create_deep_agent``
  with the harness's middleware: result paging (``FilesystemMiddleware``: a large result saved,
  a preview sent), summarization near the window's end, the scratch files and the ``task``
  helper on a long task.
* ``pieces`` — LangChain's ``create_agent`` with ``trellis.middleware()``'s pieces and Deep
  Agents' own pieces attached by hand (its ``FilesystemMiddleware`` on the journal, its
  summarization placed after the narrowing, ``PatchToolCallsMiddleware``; the scratch tools on a
  long task): Deep Agents' capabilities gated per call without ``create_deep_agent``'s assembly.
* ``main-planning``, ``deep-files`` — without a feature turned on by itself (:data:`AUTOMATIC`:
  its middleware left out); ``main@8k`` — ``main`` on an 8,192-token window;
* ``deep-<piece>`` — ``deep`` with one of Deep Agents' pieces off (:data:`PIECES`: ``paging``,
  ``context_editing``, ``summarization``), each measured on its own (``react.defaults_off``);
  with Deep Agents' paging off a large result is paged by the harness's own convention.

* ``sandbox``, ``subagents``, ``memory-files``, ``permissions`` — ``ReAct`` given one Deep
  Agents capability (:data:`CAPABILITIES`): the run's sandbox as its backend (Docker, the image
  ``TRELLIS_BENCH_SANDBOX_IMAGE`` names, else the provider's; a fake one under the scripted
  model, which needs no daemon), a sub-agent of its own, a memory file on a disk backend, and
  permissions on one. Every case of the suite runs on each, at any ``--window``: what a
  capability must not change is the answer.

:data:`DEFAULT` are the variants a run compares unless ``--variants`` says.
"""

from __future__ import annotations

import atexit
import functools
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from trellis import ReAct
from trellis.harness.react import PLACED, defaults_off


@dataclass(frozen=True)
class Setup:
    """What every variant is built from."""

    system: str
    model: Any  # a gateway model name, or a LangChain chat model
    window: int | None
    max_steps: int
    #: the harness tools every variant runs (a ``ReAct``'s ``tools=``)
    tools: tuple[Any, ...] = ()
    #: a skill source (``trellis.skills``) for ``ReAct``'s own slot (``skills=``), if any
    skills: Any = None


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    build: Callable[[Setup], tuple[Any, dict[str, Any]]]


def baseline(setup: Setup) -> tuple[Any, dict[str, Any]]:
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from langchain.agents import create_agent

    from trellis.harness import middleware as ours
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    graph = create_agent(
        routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model,
        tools=[],
        system_prompt=setup.system,
        middleware=[
            ours.HarnessTools(),
            ours.StallGuard(ours.MAX_REPEATS),
            ours.step_limit(setup.max_steps, scoped=True),
            PatchToolCallsMiddleware(),
            ours.ModelHooks(),
        ],
        checkpointer=ours.RunCheckpointer(),
        name="baseline",
    )
    setattr(graph, TOOLS_ATTRIBUTE, list(setup.tools))  # run by HarnessTools, as ReAct's are
    return graph, {"without": {"code_mode", "tool_search"}}


def main(
    setup: Setup,
    *,
    off: tuple[str, ...] = (),
    window: int | None = None,
    deep: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """``ReAct`` as it is built — ``deep``: on Deep Agents (a backend given: the journal's)."""
    from trellis.harness.automatic import JournalBackend

    named = PIECES | AUTOMATIC
    with defaults_off(*(named[name] for name in off)):
        target = ReAct(
            setup.system,
            setup.model,
            max_steps=setup.max_steps,
            context_window=window or setup.window,
            tools=setup.tools,
            backend=JournalBackend() if deep else None,
            skills=setup.skills,
        )
    return target, {}


def pieces(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``create_agent`` with the harness's middleware and Deep Agents' pieces, gated per call."""
    from deepagents.middleware import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.summarization import create_summarization_middleware
    from langchain.agents import create_agent

    from trellis.harness import automatic, react
    from trellis.harness import middleware as ours
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    chat = routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model
    model = react._profiled(chat, setup.window)
    backend = automatic.JournalBackend()
    stack = react._middleware(
        max_steps=setup.max_steps,
        max_repeats=ours.MAX_REPEATS,
        call_timeout=None,
        context_window=setup.window,
        system_prompt=None,
        held=automatic.SCRATCH,
        files=False,
        summarization=react._placed(create_summarization_middleware(model, backend), react.PLACED),
    )
    graph = create_agent(
        model,
        tools=[],
        system_prompt=setup.system,
        middleware=cast(
            list[Any], [FilesystemMiddleware(backend=backend), PatchToolCallsMiddleware(), *stack]
        ),
        checkpointer=ours.RunCheckpointer(),
        name="pieces",
    )
    setattr(graph, TOOLS_ATTRIBUTE, list(setup.tools))  # run by HarnessTools, as ReAct's are
    return graph, {}


def capable(setup: Setup, **given: Any) -> tuple[Any, dict[str, Any]]:
    """``ReAct`` as ``main`` builds it, given a Deep Agents capability (``given``)."""
    target = ReAct(
        setup.system,
        setup.model,
        max_steps=setup.max_steps,
        context_window=setup.window,
        tools=setup.tools,
        skills=setup.skills,
        **given,
    )
    return target, {}


def with_sandbox(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``ReAct(backend=sandbox())``: a real one for a model on an endpoint, a fake one (no
    daemon, no interpreter: files and the listing only) for the scripted model."""
    from trellis import sandbox

    if isinstance(setup.model, str):
        return capable(setup, backend=sandbox(image=os.environ.get(SANDBOX_IMAGE) or None))
    from tests.support.sandbox import FakeSandboxes

    return capable(setup, backend=sandbox(FakeSandboxes()))


def _disk(files: dict[str, str]) -> Any:
    """A disk backend of the variant's own (a folder made for it), holding ``files``."""
    from deepagents.backends import FilesystemBackend

    root = Path(tempfile.mkdtemp(prefix="trellis-bench-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)  # removed when the run ends
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return FilesystemBackend(root_dir=root, virtual_mode=True)


def with_memory_files(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``ReAct(memory_files=[...])``: a memory file of the deployment's on a disk backend."""
    return capable(setup, backend=_disk({"AGENTS.md": MEMORY_FILE}), memory_files=["/AGENTS.md"])


def with_permissions(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``ReAct(permissions=[...])``: a disk backend whose ``/reference`` is read-only."""
    from deepagents.middleware.permissions import FilesystemPermission

    rule = FilesystemPermission(operations=["write"], paths=["/reference/**"], mode="deny")
    return capable(setup, backend=_disk({"reference/README.md": MEMORY_FILE}), permissions=[rule])


#: The environment variable naming the ``sandbox`` variant's image (else the provider's own).
SANDBOX_IMAGE: Final = "TRELLIS_BENCH_SANDBOX_IMAGE"
#: What the ``memory-files`` variant's memory file says: house style, no fact of the data.
MEMORY_FILE: Final = "# Notes\n\nAnswer in British English. Amounts are in GBP.\n"
#: The ``subagents`` variant's helper: no ``tools`` named, so it has the run's.
HELPER: Final = {
    "name": "stock-helper",
    "description": "Looks up figures for one SKU, supplier or store with the retail tools and "
    "reports them back.",
    "system_prompt": "You look up the figures asked for with the tools and report them exactly.",
}
#: ``ReAct`` given one Deep Agents capability each: the variants an answer must not change on.
CAPABILITIES: Final[dict[str, Variant]] = {
    "sandbox": Variant("sandbox", "ReAct(backend=sandbox()): the run's sandbox", with_sandbox),
    "subagents": Variant(
        "subagents",
        "ReAct(subagents=[a helper]): a sub-agent of its own",
        functools.partial(capable, subagents=[HELPER]),
    ),
    "memory-files": Variant(
        "memory-files", "ReAct(memory_files=[...]) on a disk backend", with_memory_files
    ),
    "permissions": Variant(
        "permissions", "ReAct(permissions=[...]) on a disk backend", with_permissions
    ),
}
#: The capability variants in which code runs (a sandbox): what a ``cando`` case's file needs.
CODING: Final = frozenset({"sandbox"})


#: ``deep``'s pieces a ``deep-<piece>`` variant turns off, by the name of their middleware.
PIECES: Final = {
    "paging": "FilesystemMiddleware",
    "context_editing": "ContextEditingMiddleware",
    "summarization": PLACED,
}


#: The features ``ReAct`` turns on by itself (``trellis.harness.automatic``), by the name of
#: their middleware: a ``main-<feature>`` variant is ``main`` with it left out.
AUTOMATIC: Final = {"planning": "TodoListMiddleware", "files": "LongTaskTools"}
#: The window ``main@8k`` assumes whatever the model's: a small local model's.
SMALL_WINDOW: Final = 8_192


VARIANTS: Final[dict[str, Variant]] = {
    "baseline": Variant("baseline", "plain tools, everything off", baseline),
    "main": Variant(
        "main",
        "ReAct as it is built: create_agent with middleware() (no Deep Agents capability asked)",
        main,
    ),
    "deep": Variant(
        "deep",
        "ReAct on Deep Agents (a backend given): its file tools, paging, summarization, helper",
        functools.partial(main, deep=True),
    ),
    **{
        f"deep-{piece}": Variant(
            f"deep-{piece}",
            f"deep with {piece.replace('_', ' ')} off",
            functools.partial(main, off=(piece,), deep=True),
        )
        for piece in PIECES
    },
    "main-planning": Variant(
        "main-planning", "main without planning", functools.partial(main, off=("planning",))
    ),
    "deep-files": Variant(
        "deep-files",
        "deep without its long-task files",
        functools.partial(main, off=("files",), deep=True),
    ),
    "pieces": Variant(
        "pieces",
        "create_agent with middleware() and Deep Agents' pieces attached, gated per call",
        pieces,
    ),
    "main@16k": Variant(
        "main@16k",
        "main on a 16,384-token window (the model server's own: -c 16384)",
        functools.partial(main, window=16_384),
    ),
    "main@8k": Variant(
        "main@8k",
        "main on an 8,192-token window, whatever the model's",
        functools.partial(main, window=SMALL_WINDOW),
    ),
    **CAPABILITIES,
}
#: The variants a run compares by default.
DEFAULT: Final = ("baseline", "main")
