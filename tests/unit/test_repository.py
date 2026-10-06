"""What every prompt and skill source shares: a reference, the order sources are asked in, the
run's pin, the last good copy of a remote source, front matter, content versions and paths
kept inside their folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from trellis.contracts import ConfigurationError
from trellis.harness import fresh
from trellis.harness.repository import (
    TTL_SECONDS,
    Chain,
    Kept,
    NotFound,
    digest,
    front_matter,
    inside,
    journaled,
    named,
    pinned,
)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [("triage", ("triage", None)), ("triage@3", ("triage", "3")), ("a@b@c", ("a", "b@c"))],
)
def test_a_reference_is_a_name_and_maybe_a_version(
    ref: str, expected: tuple[str, str | None]
) -> None:
    assert pinned(ref) == expected
    assert named(*expected) == ref


@pytest.mark.parametrize("ref", ["", "@3", "triage@"])
def test_a_reference_without_a_name_or_after_its_at_is_refused(ref: str) -> None:
    with pytest.raises(ConfigurationError, match="is not a name"):
        pinned(ref)


class Held:
    """A source holding some names (``LookupError`` for the rest), or failing outright."""

    def __init__(self, label: str, held: dict[str, str], *, down: bool = False) -> None:
        self.label = label
        self.held = held
        self.down = down
        self.closed = False

    async def resolve(self, name: str, version: str | None) -> str:
        if self.down:
            raise ConnectionError(f"{self.label} is down")
        if name not in self.held:
            raise NotFound(f"no {name}")
        return f"{self.label}:{self.held[name]}"

    async def aclose(self) -> None:
        self.closed = True


class Bare:
    label = "bare"

    async def resolve(self, name: str, version: str | None) -> str:
        raise LookupError


async def test_the_first_source_that_has_a_name_answers() -> None:
    first, second = Held("one", {"a": "1"}), Held("two", {"a": "2", "b": "2"})
    chain: Chain[str] = Chain([first, second])
    assert chain.labels == ["one", "two"]
    assert await chain.find("a") == "one:1"  # found in two: the first in the order
    assert await chain.find("b") == "two:2"
    with pytest.raises(ConfigurationError) as raised:
        await Chain([first, Bare()]).find("c", "2")
    assert str(raised.value) == "no entry 'c@2' in any source (one: no c; bare: LookupError)"
    await Chain([first, Bare()]).aclose()
    assert first.closed


async def test_a_source_that_fails_stops_the_search() -> None:
    chain: Chain[str] = Chain([Held("one", {}, down=True), Held("two", {"a": "2"})])
    with pytest.raises(ConnectionError, match="one is down"):
        await chain.find("a")  # never the next source in its place


async def test_a_chain_with_no_source_says_how_to_name_one() -> None:
    with pytest.raises(ConfigurationError, match="no entry source for 'a'"):
        await Chain().find("a")


async def test_outside_a_run_nothing_is_journaled() -> None:
    async def read() -> int:
        return 7

    assert await journaled(None, "k", read, dump=str, load=int) == 7


async def test_a_kept_answer_stands_while_the_source_is_down_a_missing_one_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: clock[0])
    answers: dict[str, str | None] = {"a": "1", "b": None}
    reads: list[str] = []
    down = [False]

    async def read(key: str) -> str | None:
        reads.append(key)
        if down[0]:
            raise ConnectionError("down")
        return answers[key]

    kept: Kept[str, str] = Kept(read, what=lambda key: f"entry {key}")
    assert (await kept.get("a"), await kept.get("b")) == ("1", None)
    assert await kept.get("a") == "1" and reads == ["a", "b"]  # kept for the TTL
    clock[0] += TTL_SECONDS + 1
    down[0] = True
    assert (await kept.get("a"), await kept.get("b")) == ("1", None)  # the last answers stand
    with pytest.raises(ConnectionError):
        await kept.get("c")  # never read: nothing to stand in
    down[0] = False
    answers["c"] = "3"
    assert await kept.get("c", forever=True) == "3"
    clock[0] += 10 * TTL_SECONDS
    down[0] = True
    assert await kept.get("c", forever=True) == "3"  # an immutable version is read once


def test_front_matter_reads_what_prompt_and_skill_files_use() -> None:
    text = """---
name: sql-review   # the folder's
description: >
  Reviews SQL
  queries.
version: "1.2.0"
empty:
notes: |
  line one
    indented

license: 'MIT'
metadata:
  author: ops
  version: 9
tags:
  - a
  - "b c"
# a comment
---
Body here.
"""
    meta, body = front_matter(text, where="SKILL.md")
    assert meta == {
        "name": "sql-review",
        "description": "Reviews SQL queries.",
        "version": "1.2.0",
        "empty": "",
        "notes": "line one\n  indented",
        "license": "MIT",
        "metadata": {"author": "ops", "version": "9"},
        "tags": ["a", "b c"],
    }
    assert body == "Body here."
    assert front_matter("No front matter.", where="x") == ({}, "No front matter.")
    assert front_matter("", where="x") == ({}, "")
    assert front_matter("---\nliteral: |\n---\n", where="x") == ({"literal": ""}, "")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("---\nname: a\n", "no closing ---"),
        ("---\njust words\n---\n", "line 2: 'just words' is not key: value"),
        ("---\nname: a\n  more: b\n---\n", "line 2: a value, then indented lines"),
        ("---\nmeta:\n  ok: 1\n  bad\n---\n", "line 4: 'bad' is not key: value"),
    ],
)
def test_front_matter_it_cannot_read_names_the_file_and_line(text: str, problem: str) -> None:
    with pytest.raises(ConfigurationError, match=problem) as raised:
        front_matter(text, where="p/SKILL.md")
    assert str(raised.value).startswith("p/SKILL.md: ")


def test_a_content_version_changes_with_any_part() -> None:
    assert len(digest(b"a")) == 12
    assert digest(b"a", b"b") != digest(b"ab")
    assert digest(b"a") == digest(b"a")


def test_a_path_stays_inside_its_folder(tmp_path: Path) -> None:
    (tmp_path / "skill").mkdir()
    (tmp_path / "skill" / "a.md").write_text("a")
    (tmp_path / "secret.txt").write_text("s")
    (tmp_path / "skill" / "link").symlink_to(tmp_path / "secret.txt")
    root = tmp_path / "skill"
    assert inside(root, "a.md") == (root / "a.md").resolve()
    for refused in ("../secret.txt", "/etc/passwd", "link", ".", ""):
        assert inside(root, refused) is None, refused
