"""The framework versions this release of the harness was tested with: kept here, and only here.

The harness pins what it tested and moves only by a release (``docs/adr/0001-version-policy.md``).
``pyproject.toml``'s extras say the same ranges (``tests/unit/test_compat.py`` checks that they
agree), so an install resolves inside them. An environment can still hold another version — a
lock of the application's own, an override — and then :func:`check` says so once, when a target
of that framework is wrapped: the version found, the range tested, and what to do about it. It
never refuses: the run may well work, but nobody has proven it.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from trellis.harness.bounded import SAID_ONCE
from trellis.harness.frameworks import REGISTERED

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
}
#: The distributions each adapter runs on (by the adapter's name: ``frameworks.py``), checked
#: when a target of its type is wrapped; one that is not installed is not checked.
FRAMEWORKS: Final[Mapping[str, tuple[str, ...]]] = {r.name: r.distributions for r in REGISTERED}
#: What the warning tells the reader to do.
ADVICE: Final = "use a newer trellis-harness (docs/versioning.md)"


def tested(distribution: str, installed: str) -> bool:
    """Whether ``installed`` is in the range ``distribution`` was tested with, by its release
    numbers (a pre-release or local suffix is ignored: ``1.4.0rc1`` is ``1.4.0``; a version
    that is not one is outside every range). A distribution with no range is not the
    harness's to judge."""
    if distribution not in TESTED:
        return True
    low, high = TESTED[distribution]
    try:
        release = Version(installed).base_version
    except InvalidVersion:
        return False
    return SpecifierSet(f">={low},<{high}").contains(release)


def installed(distribution: str) -> str | None:
    """The installed version of ``distribution`` (``None``: not installed)."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def check(adapter: str) -> list[str]:
    """What is said about each distribution ``adapter`` runs on whose installed version is
    outside its tested range: a warning, once per distribution and version in a process."""
    untested = []
    for distribution in FRAMEWORKS.get(adapter, ()):
        found = installed(distribution)
        if found is not None and not tested(distribution, found):
            untested.append(_said(distribution, found))
    return untested


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per distribution and version
def _said(distribution: str, found: str) -> str:
    low, high = TESTED[distribution]
    message = (
        f"{distribution} {found} is installed, but this trellis-harness was tested with "
        f"{distribution}>={low},<{high}: {ADVICE}"
    )
    log.warning(message, extra={"distribution": distribution, "installed": found})
    return message


__all__ = ["ADVICE", "FRAMEWORKS", "TESTED", "check", "installed", "tested"]
