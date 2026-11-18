"""Which tools are infrastructure: the deployment's shells and its network out, never the run's
user's question (ADR 0008, ``docs/governance.md#who-decides-what``).

A tool is infrastructure when the harness or the framework made it (``own``: a harness built-in
or a framework's own tool, ``Tool.confined``; a framework's built-in decided through its own
permission flow, ``bridge.permitted``) and it is a shell (:data:`SHELLS`) or reaches the network
out (:data:`NETWORK`). A tool of the team's own that happens to have one of these names says
what it does itself, as any other. One definition: the adapters' own lists (Claude Code's
network tools, an OpenAI ``SandboxAgent``'s shell, Deep Agents' ``execute``, the sandbox's
tools) are held inside it by a test.

Infrastructure is the platform operator's, shared by every tenant: never a tenant's or its
user's to approve. Such a call runs unasked and announced (``Decision.unasked``); where a person
would be asked about one all the same (the catalog's risk, an ``approve_when``, a ``before_tool``
hook's ``Ask``) it is denied (:data:`REFUSED`) — the model reads why, the run records it."""

from __future__ import annotations

from typing import Final

from trellis.harness.governance.shell import SHELLS as FRAMEWORK_SHELLS
from trellis.harness.host import CLAUDE_SHELLS

#: The shells: Claude Code's (``host.CLAUDE_SHELLS``) and Kimi Code's (``governance.shell``), an
#: OpenAI ``SandboxAgent``'s ``Shell`` capability, Deep Agents' ``execute``, LangChain's
#: ``ShellToolMiddleware`` (``shell``) and the harness's own sandbox.
SHELLS: Final = frozenset(
    {
        *CLAUDE_SHELLS,
        *FRAMEWORK_SHELLS,
        "exec_command",
        "write_stdin",
        "execute",
        "shell",
        "sandbox_exec",
        "sandbox_write",
    }
)
#: The frameworks' own tools that reach the network outside the deployment: Claude Code's
#: ``WebFetch`` and ``WebSearch``, Kimi Code's ``FetchURL`` and ``WebSearch``.
NETWORK: Final = frozenset({"WebFetch", "WebSearch", "FetchURL"})
#: What the model reads of an infrastructure call a person would have been asked about: denied.
REFUSED: Final = (
    "{tool} was not run: it is the platform's infrastructure, and a call of it that needs "
    "approval is denied; no user can approve it."
)
#: The ``error_class`` of that call.
REFUSED_CLASS: Final = "InfrastructureDenied"


def infrastructure(tool: str, *, own: bool) -> bool:
    """Whether a call of ``tool`` is infrastructure: the harness's or its framework's own
    (``own``) shell or network tool."""
    return own and (tool in SHELLS or tool in NETWORK)
