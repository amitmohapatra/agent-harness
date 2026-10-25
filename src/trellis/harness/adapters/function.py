"""A plain async function ``(input, agent) -> answer``, where ``agent`` is the run's
:class:`~trellis.harness.runtime.Runtime` (the same object ``trellis.current()`` returns).

The memory context is ``agent.context`` (and a leading system message when the input is a
message list); tools are called with ``await agent.tools.call(name, **args)``; a pause is
``await agent.ask(...)``, and a resume calls the function again against the journal.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, ClassVar

from trellis.contracts import InterruptResolution, ToolSpec
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    Output,
    ToolFormat,
    no_options,
)
from trellis.harness.history import Said, Speaker
from trellis.harness.journal import Pending


class FunctionAdapter:
    name: ClassVar[str] = "function"
    tool_format: ToolFormat = "none"
    fixed_tools: bool = False
    narrows: Narrowing = "none"
    #: the function reads its input as it is: memory's window is in ``agent.context``
    replays_history = False

    def __init__(self) -> None:
        # the function's own LangChain chat models, where LangChain is installed: accounted to
        # the run's budget and watched for a stall (``accounting``), as a graph's are
        with contextlib.suppress(ImportError):
            from trellis.harness import accounting  # noqa: F401
            from trellis.harness.middleware import govern_tool_nodes

            # and a graph it runs itself governed, tool node by tool node
            govern_tool_nodes()

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        return None

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        if context and isinstance(input, list):
            return [{"role": "system", "content": context}, *input]
        return input

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        return await target(native_input, run.runtime)

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        yield Output(await target(native_input, run.runtime))

    def extract(self, target: Any, output: Any) -> Extracted:
        return Extracted(
            answer=output,
            transcript=[Said(Speaker.ASSISTANT, output)]
            if isinstance(output, str) and output
            else [],
        )

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        return native_input

    def check_options(self, options: Mapping[str, Any]) -> None:
        no_options(options, "a function target", "it reads what it needs from its input")

    async def native(self, target: Any) -> list[ToolSpec]:
        return []  # a function calls the run's tools, nothing of its own the harness can read
