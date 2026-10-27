"""The framework versions this release of the harness was tested with: kept here, and only here.

The harness pins what it tested and moves only by a release (``docs/adr/0001-version-policy.md``).
``pyproject.toml``'s extras say the same ranges (``tests/unit/test_compat.py`` checks that they
agree), so an install resolves inside them. An environment can still hold another version — a
lock of the application's own, an override — and then :func:`check` says so once, when a target
of that framework is wrapped: the version found, the range tested, and what to do about it. It
never refuses: the run may well work, but nobody has proven it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from trellis.contracts import ConfigurationError

log = logging.getLogger("trellis.harness")

#: Each framework distribution's tested range: ``(lowest, first not tested)``, as
#: ``pyproject.toml`` writes it (``>=lowest,<first not tested``).
TESTED: Final[Mapping[str, tuple[str, str]]] = {
    "langgraph": ("1.2", "1.3"),
    "langchain": ("1.4", "1.5"),
    "langchain-core": ("1.6", "1.7"),
    "langchain-openai": ("1.6", "1.7"),
    "deepagents": ("0.7.19", "0.8"),
    "openai-agents": ("0.22.3", "0.23"),
    "claude-agent-sdk": ("0.2.160", "0.3"),
    "a2a-sdk": ("1.1", "1.2"),
    "agent-client-protocol": ("0.12.1", "0.13"),
}
#: The agent programs an adapter drives over a protocol, not a Python distribution (an ACP
#: agent's server, ``adapters.acp``): each one's tested range, and how to install it. Unlike a
#: distribution's, a version outside it is refused (:func:`require_binary`): the program's
#: protocol surface — its tool names, its permission requests, its usage log — is what the
#: adapter reads, and nothing pins it but this check.
BINARIES: Final[Mapping[str, tuple[str, str]]] = {
    "kimi-code": ("2.1.1", "2.2"),
}
INSTALL: Final[Mapping[str, str]] = {
    "kimi-code": "npm install -g @moonshot-ai/kimi-code@2.1.1",
}
#: The distributions each adapter runs on (by the adapter's name), checked when a target of
#: its type is wrapped; one that is not installed is not checked.
FRAMEWORKS: Final[Mapping[str, tuple[str, ...]]] = {
    "langgraph": ("langgraph", "langchain-core", "langchain", "deepagents", "langchain-openai"),
    "openai_agents": ("openai-agents",),
    "claude_agent_sdk": ("claude-agent-sdk",),
    "acp": ("agent-client-protocol",),
    "function": (),
}
#: What the warning tells the reader to do.
ADVICE: Final = "use a newer trellis-harness (docs/versioning.md)"

#: The distributions already warned about in this process: one warning each.
_warned: set[str] = set()


def release(text: str) -> tuple[int, ...]:
    """A version's release numbers (``"0.2.160"`` → ``(0, 2, 160)``; a pre-release or local
    suffix is ignored: ``"1.4.0rc1"`` → ``(1, 4, 0)``)."""
    numbers = re.match(r"\d+(?:\.\d+)*", text.strip())
    return tuple(int(n) for n in numbers.group(0).split(".")) if numbers else ()


def tested(distribution: str, installed: str) -> bool:
    """Whether ``installed`` is in the range ``distribution`` was tested with (a distribution
    with no range is not the harness's to judge)."""
    if distribution not in TESTED:
        return True
    return _within(installed, TESTED[distribution])


def _within(installed: str, tested_range: tuple[str, str]) -> bool:
    low, high = (release(v) for v in tested_range)
    found = release(installed)
    width = max(len(low), len(high), len(found))

    def padded(v: tuple[int, ...]) -> tuple[int, ...]:
        return v + (0,) * (width - len(v))

    return padded(low) <= padded(found) < padded(high)


def installed(distribution: str) -> str | None:
    """The installed version of ``distribution`` (``None``: not installed)."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def check(adapter: str) -> list[str]:
    """Warn, once per distribution and process, about each distribution ``adapter`` runs on
    whose installed version is outside its tested range; the warnings given now."""
    given = []
    for distribution in FRAMEWORKS.get(adapter, ()):
        found = installed(distribution)
        if found is None or distribution in _warned or tested(distribution, found):
            continue
        _warned.add(distribution)
        low, high = TESTED[distribution]
        message = (
            f"{distribution} {found} is installed, but this trellis-harness was tested with "
            f"{distribution}>={low},<{high}: {ADVICE}"
        )
        log.warning(message, extra={"distribution": distribution, "installed": found})
        given.append(message)
    return given


def require_binary(name: str, found: str, *, command: str) -> None:
    """Refuse an agent program (``BINARIES``) whose version, as it reports it, is outside the
    range this release tested — or that does not report one."""
    low, high = BINARIES[name]
    if found and _within(found, (low, high)):
        return
    said = f"{name} {found}" if found else f"a {name} that reports no version"
    raise ConfigurationError(
        f"{said} answers at {command!r}, but this trellis-harness was tested with "
        f"{name}>={low},<{high}: install it ({INSTALL[name]}), or {ADVICE}"
    )


__all__ = [
    "ADVICE",
    "BINARIES",
    "FRAMEWORKS",
    "INSTALL",
    "TESTED",
    "check",
    "installed",
    "release",
    "require_binary",
    "tested",
]
