"""A plain async function ``(input, agent) -> answer``, where ``agent`` is the run's
:class:`~trellis.harness.runtime.Runtime` (the same object ``trellis.current()`` returns).

The memory context is ``agent.context`` (and a leading system message when the input is a
message list); tools are called with ``await agent.tools.call(name, **args)``; a pause is
``await agent.ask(...)``, and a resume calls the function again against the journal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from trellis.contracts import InterruptResolution
from trellis.harness.adapters.base import Extracted, Invocation, Output, ToolFormat
from trellis.harness.journal import Pending


class FunctionAdapter:
    name: ClassVar[str] = "function"
    tool_format: ClassVar[ToolFormat] = "none"
    fixed_tools: ClassVar[bool] = False

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any:
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
            transcript=[("assistant", output)] if isinstance(output, str) and output else [],
        )

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        return native_input
