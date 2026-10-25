"""What a versioned entry of the gateway's repositories shares (its stored prompts): a
reference — ``"name"`` or ``"name@version"`` (:func:`pinned`) — and the run's pin
(:func:`journaled`): what a run resolved is recorded in its journal, so a resumed run — after a
pause, or on another worker — reads the same version even when the entry changed since."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: What separates a stored prompt's name from the version it is pinned to.
PINNED: Final = "@"
#: How long what a remote source answered is kept before it is read again; while the source
#: cannot be reached the last answer stands (read again after :data:`RETRY_SECONDS`).
TTL_SECONDS: Final = 300.0
RETRY_SECONDS: Final = 30.0
#: The label of the gateway's repositories, as a source (a pin records where it came from).
BIFROST: Final = "Bifrost"


class NotFound(ConfigurationError, LookupError):
    """A name, or a version of it, the gateway does not have: no last copy hides it (a
    ``ConfigurationError``, fatal to :class:`~trellis.harness.fresh.Fresh`)."""


def pinned(ref: str) -> tuple[str, str | None]:
    """``"name@version"`` as its name and version (``None`` when it names none)."""
    name, _, version = ref.partition(PINNED)
    if not name or (PINNED in ref and not version):
        raise ConfigurationError(f"{ref!r} is not a name, or name{PINNED}version")
    return name, version or None


def named(name: str, version: str | None) -> str:
    """The reference back: ``name`` or ``name@version``."""
    return name if version is None else f"{name}{PINNED}{version}"


async def journaled[T](
    runtime: Runtime | None,
    key: str,
    read: Callable[[], Awaitable[T]],
    *,
    dump: Callable[[T], Any],
    load: Callable[[Any], T],
) -> T:
    """What ``read`` gives, pinned for the run: inside a run the journal's, when an earlier
    attempt recorded it, else read now and recorded; outside one, read now."""
    if runtime is None:
        return await read()
    replayed, recorded = runtime.replay.call(key)
    if replayed:
        return load(recorded)
    value = await read()
    runtime.replay.record_call(key, dump(value))
    return value
