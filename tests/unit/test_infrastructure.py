"""The one definition of an infrastructure tool (``governance.infrastructure``, ADR 0008): every
adapter's own list of shells and network tools is inside it, and only a tool the harness or its
framework made is infrastructure."""

from __future__ import annotations

import pytest

from trellis.harness.adapters.acp import KIMI_NETWORK
from trellis.harness.adapters.claude import CLAUDE_NETWORK
from trellis.harness.governance import shell
from trellis.harness.governance.infrastructure import NETWORK, SHELLS, infrastructure
from trellis.harness.host import CLAUDE_SHELLS
from trellis.harness.middleware import EXECUTE
from trellis.harness.sandbox import EXEC, WRITE
from trellis.harness.tools.convert.openai_agents import CAPABILITY_SHELLS


def test_every_adapters_shells_and_network_tools_are_in_the_one_definition() -> None:
    shells = {*CLAUDE_SHELLS, *shell.SHELLS, *CAPABILITY_SHELLS, EXECUTE, EXEC, WRITE}
    assert shells <= SHELLS
    assert CLAUDE_NETWORK | KIMI_NETWORK <= NETWORK


@pytest.mark.parametrize("name", sorted(SHELLS | NETWORK))
def test_a_shell_or_network_tool_is_infrastructure_only_when_the_harness_or_framework_made_it(
    name: str,
) -> None:
    assert infrastructure(name, own=True)
    assert not infrastructure(name, own=False)  # a tool of your own of that name


@pytest.mark.parametrize("name", ["note", "Write", "refund", "schedule_task", "memory_forget"])
def test_a_business_tool_is_not_infrastructure(name: str) -> None:
    assert not infrastructure(name, own=True)
