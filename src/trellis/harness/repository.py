"""Where prompts and skills come from, and what every source of them shares.

A source answers one question — ``resolve(name, version)``: the prompt or the skill called
``name``, as ``version`` reads (``None``: the version the source serves now) — and raises a
:class:`LookupError` when it has no such name, or not that version. Sources are asked in
order (:class:`Chain`): the first that has it answers, and a name no source has is a
``ConfigurationError`` naming each source tried and what it said.

What the sources share lives here, once:

* a reference — ``"name"`` or ``"name@version"`` (:func:`pinned`);
* the run's pin (:func:`journaled`): what a run resolved is recorded in its journal, so a
  resumed run — after a pause, or on another worker — reads the same text even when the
  source changed since;
* the last good copy of a remote source (:class:`Kept`): kept for a while, read again after
  that, and kept while the source cannot be reached — a name the source said it does not have
  included;
* front matter (``---`` YAML ``---`` at the top of a ``.md`` file: :func:`front_matter`), and a
  content version for a file or a folder that names none (:func:`digest`);
* a path inside a folder, never outside it (:func:`inside`).
"""

from __future__ import annotations

import functools
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from trellis.contracts import ConfigurationError
from trellis.harness.fresh import Fresh

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: What separates a prompt's or a skill's name from the version it is pinned to.
PINNED: Final = "@"
#: How long what a remote source answered is kept before it is read again; while the source
#: cannot be reached the last answer stands (read again after :data:`RETRY_SECONDS`).
TTL_SECONDS: Final = 300.0
RETRY_SECONDS: Final = 30.0
#: The label of the gateway's repositories, as a source (a pin records where it came from).
BIFROST: Final = "Bifrost"
#: The hex digits of a content version (:func:`digest`).
DIGEST_CHARS: Final = 12


class NotFound(ConfigurationError, LookupError):
    """A name, or a version of it, the source does not have: the next source is asked (a
    :class:`LookupError`), and no last copy hides it (a ``ConfigurationError``, fatal to
    :class:`~trellis.harness.fresh.Fresh`)."""


def pinned(ref: str) -> tuple[str, str | None]:
    """``"name@version"`` as its name and version (``None`` when it names none)."""
    name, _, version = ref.partition(PINNED)
    if not name or (PINNED in ref and not version):
        raise ConfigurationError(f"{ref!r} is not a name, or name{PINNED}version")
    return name, version or None


def named(name: str, version: str | None) -> str:
    """The reference back: ``name`` or ``name@version``."""
    return name if version is None else f"{name}{PINNED}{version}"


class Source[T](Protocol):
    """A source of prompts or skills: :attr:`label` says which in messages and in records."""

    @property
    def label(self) -> str: ...

    async def resolve(self, name: str, version: str | None) -> T: ...


class Chain[T]:
    """Sources asked in order: the first that has the name (and the version) answers."""

    #: what the sources hold, in messages (``"prompt"``, ``"skill"``)
    kind: ClassVar[str] = "entry"
    #: how a deployment names a source, in the message for a chain with none
    hint: ClassVar[str] = ""

    def __init__(self, sources: Sequence[Source[T]] = ()) -> None:
        self.sources = list(sources)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.sources]

    async def find(self, name: str, version: str | None = None) -> T:
        """The first source's answer; ``ConfigurationError`` when none has it, naming each
        source tried and what it said. A source that fails otherwise (it cannot be reached
        and was never read) stops the search: a later source is not asked in its place, so
        what a name resolves to never depends on an outage."""
        ref = named(name, version)
        if not self.sources:
            raise ConfigurationError(f"no {self.kind} source for {ref!r}: {self.hint}")
        tried: list[str] = []
        for source in self.sources:
            try:
                return await source.resolve(name, version)
            except LookupError as exc:
                tried.append(f"{source.label}: {_said(exc)}")
        raise ConfigurationError(f"no {self.kind} {ref!r} in any source ({'; '.join(tried)})")

    async def aclose(self) -> None:
        for source in self.sources:
            close: Callable[[], Awaitable[None]] | None = getattr(source, "aclose", None)
            if close is not None:
                await close()


def _said(exc: LookupError) -> str:
    return str(exc.args[0]) if exc.args else type(exc).__name__


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


class Kept[K, T]:
    """What a remote source answered for each key, kept: read again every ``ttl`` seconds
    (``None``: never — an immutable version), the last answer standing while the source
    cannot be reached. An answer of ``None`` (the source does not have it) is kept like any
    other."""

    def __init__(
        self,
        read: Callable[[K], Awaitable[T | None]],
        *,
        what: Callable[[K], str],
        ttl: float = TTL_SECONDS,
        retry: float = RETRY_SECONDS,
    ) -> None:
        self._read = read
        self._what = what
        self._ttl = ttl
        self._retry = retry
        self._kept: dict[K, Fresh[tuple[T | None]]] = {}

    async def get(self, key: K, *, forever: bool = False) -> T | None:
        found = self._kept.get(key)
        if found is None:
            found = self._kept[key] = Fresh(
                functools.partial(self._boxed, key),
                what=self._what(key),
                ttl=float("inf") if forever else self._ttl,
                retry=self._retry,
            )
        return (await found.get())[0]

    async def _boxed(self, key: K) -> tuple[T | None]:
        return (await self._read(key),)


# --------------------------------------------------------------------------- files

_KEY: Final = re.compile(r"^([A-Za-z_][\w.-]*)\s*:\s*(.*)$")
_ITEM: Final = re.compile(r"^-\s+(.*)$")
_BLOCK: Final = frozenset({"|", ">", "|-", ">-", "|+", ">+"})


def front_matter(text: str, *, where: str) -> tuple[dict[str, Any], str]:
    """The front matter of a Markdown file and its body. Front matter is the YAML between a
    first line ``---`` and the next ``---``; the subset read is what prompt and skill files
    use: ``key: value`` (plain or quoted), block scalars (``|``, ``>``), a list of ``- item``
    lines and one level of nested ``key: value`` lines under a key. Anything else is a
    ``ConfigurationError`` naming the file and the line. No front matter: ``{}`` and the text."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((n for n in range(1, len(lines)) if lines[n].strip() == "---"), None)
    if end is None:
        raise ConfigurationError(f"{where}: the front matter has no closing ---")
    return _mapping(lines[1:end], where, first=2), "\n".join(lines[end + 1 :])


def _mapping(lines: list[str], where: str, *, first: int) -> dict[str, Any]:
    found: dict[str, Any] = {}
    n = 0
    while n < len(lines):
        line = lines[n]
        if not line.strip() or line.lstrip().startswith("#"):
            n += 1
            continue
        match = _KEY.match(line)
        if match is None:
            problem = f"{line.strip()!r} is not key: value"
            raise ConfigurationError(f"{where}: line {first + n}: {problem}")
        key, value, at = match.group(1), match.group(2).strip(), first + n
        nested = []
        n += 1
        while n < len(lines) and (not lines[n].strip() or lines[n][:1] in (" ", "\t")):
            nested.append(lines[n])
            n += 1
        found[key] = _value(value, nested, where, at)
    return found


def _value(value: str, nested: list[str], where: str, at: int) -> Any:
    """A key's value: on its line (``at``), or the indented lines under it."""
    indented = [line for line in nested if line.strip()]
    if value in _BLOCK:
        width = min((len(line) - len(line.lstrip()) for line in indented), default=0)
        rows = [line[width:] for line in nested]
        while rows and not rows[-1].strip():
            rows.pop()
        return "\n".join(rows) if value[0] == "|" else " ".join(r.strip() for r in rows if r)
    if not indented:
        return _scalar(value)
    if value:
        raise ConfigurationError(f"{where}: line {at}: a value, then indented lines")
    items = [_ITEM.match(line.strip()) for line in indented]
    if all(items):
        return [_scalar(item.group(1)) for item in items if item is not None]
    width = min(len(line) - len(line.lstrip()) for line in indented)
    return _mapping([line[width:] for line in nested], where, first=at + 1)


def _scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value.split(" #", 1)[0].rstrip()


def digest(*parts: bytes) -> str:
    """A content version: the first :data:`DIGEST_CHARS` hex digits of their SHA-256."""
    hashed = hashlib.sha256()
    for part in parts:
        hashed.update(len(part).to_bytes(8, "big"))
        hashed.update(part)
    return hashed.hexdigest()[:DIGEST_CHARS]


def inside(root: Path, relative: str) -> Path | None:
    """``root / relative`` when it stays inside ``root`` (symbolic links followed), else
    ``None``: a path that climbs out (``..``, an absolute path, a link elsewhere) is refused."""
    base = root.resolve()
    path = (base / relative).resolve()
    return path if path != base and path.is_relative_to(base) else None
