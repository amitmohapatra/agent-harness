"""Skill sources: a skill in code, a folder in the Agent Skills layout (paths kept inside it),
the gateway's Skills Repository, the order they are asked in, and a pin — made, kept and
restored — for code that is not wrapped."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.gateway import FakeGateway, SkillVersions
from trellis import Settings
from trellis.contracts import ConfigurationError, ToolError
from trellis.harness.repository import NotFound
from trellis.harness.skills import (
    SECTION,
    BifrostSkills,
    ResolvedSkill,
    Skill,
    SkillSources,
    refs_of,
    skills_dir,
)

TONE = Skill("tone", "How we write.", "Short sentences.", files={"words.md": "Use: refund."})


def write_skill(root: Path, name: str, front: str, body: str, **files: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(f"---\n{front}\n---\n{body}\n")
    for path, text in files.items():
        (folder / path.replace("__", "/")).parent.mkdir(parents=True, exist_ok=True)
        (folder / path.replace("__", "/")).write_text(text)
    return folder


# --------------------------------------------------------------------------- code
async def test_a_skill_in_code_is_its_own_source() -> None:
    assert TONE.label == "code"
    found = await TONE.resolve("tone", None)
    assert found == ResolvedSkill(
        "tone", "1", "How we write.", "Short sentences.", ("words.md",), "code"
    )
    assert found.origin is TONE and await found.read("words.md") == "Use: refund."
    for name, version in (("other", None), ("tone", "2")):
        with pytest.raises(NotFound, match="holds tone@1 only"):
            await TONE.resolve(name, version)
    with pytest.raises(ToolError, match="this run uses version 0, and the code has 1 now"):
        await TONE.read("tone", "0", "words.md")
    with pytest.raises(ConfigurationError, match="a name without @"):
        Skill("a@1", "d", "b")


async def test_a_file_a_skill_does_not_list_or_has_no_source_for_now_is_refused() -> None:
    found = await TONE.resolve("tone", None)
    with pytest.raises(ToolError, match=r"skill tone 1 has no file 'x\.md'"):
        await found.read("x.md")
    orphan = ResolvedSkill("tone", "1", "d", "b", ("words.md",), "skills_dir(gone)")
    with pytest.raises(ToolError, match="cannot be read: no source is skills_dir\\(gone\\) now"):
        await orphan.read("words.md")


# --------------------------------------------------------------------------- files
async def test_a_folder_of_agent_skills(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "sql-review",
        "name: sql-review\ndescription: Reviews SQL.\nversion: 1.2.0",
        "Read rules.md first.",
        **{"rules.md": "No SELECT *.", "examples__good.sql": "SELECT id"},
    )
    write_skill(
        tmp_path,
        "refunds",
        "name: refunds\ndescription: >\n  Handles\n  refunds.\nmetadata:\n  version: 2.0.0",
        "Refund within 30 days.",
    )
    write_skill(tmp_path, "plain", "name: plain\ndescription: Plain.", "Body.")
    (tmp_path / "secret.txt").write_text("secret")
    (tmp_path / "plain" / "leak.md").symlink_to(tmp_path / "secret.txt")
    source = skills_dir(tmp_path)
    assert source.label == f"skills_dir({tmp_path})"
    sql = await source.resolve("sql-review", None)
    assert sql == ResolvedSkill(
        "sql-review",
        "1.2.0",
        "Reviews SQL.",
        "Read rules.md first.",
        ("examples/good.sql", "rules.md"),
        source.label,
    )
    assert await sql.read("rules.md") == "No SELECT *."
    assert await sql.read("examples/good.sql") == "SELECT id"
    refunds = await source.resolve("refunds", "2.0.0")
    assert (refunds.version, refunds.description) == ("2.0.0", "Handles refunds.")
    plain = await source.resolve("plain", None)
    assert len(plain.version) == 12 and plain.files == ()  # a digest; the link is left out
    with pytest.raises(NotFound, match=r"sql-review is version 1\.2\.0, not 9"):
        await source.resolve("sql-review", "9")


async def test_a_folder_refuses_what_leaves_it_and_what_is_not_a_skill(tmp_path: Path) -> None:
    write_skill(tmp_path, "sql", "name: sql\ndescription: Reviews SQL.", "B", **{"a.md": "A"})
    write_skill(tmp_path, "wrong", "name: other\ndescription: D.", "B")
    write_skill(tmp_path, "mute", "name: mute", "B")
    (tmp_path / "empty").mkdir()
    source = skills_dir(tmp_path)
    for name in ("../sql", "sql/../sql", "/etc"):
        with pytest.raises(NotFound, match="is not a folder name inside it"):
            await source.resolve(name, None)
    with pytest.raises(NotFound, match=r"it has no empty/SKILL\.md"):
        await source.resolve("empty", None)
    with pytest.raises(ConfigurationError, match="its front matter's name must be 'wrong'"):
        await source.resolve("wrong", None)
    with pytest.raises(ConfigurationError, match="its front matter has no description"):
        await source.resolve("mute", None)
    sql = await source.resolve("sql", None)
    for path in ("../wrong/SKILL.md", "SKILL.md"):
        with pytest.raises(ToolError, match="has no file"):
            await source.read("sql", sql.version, path)
    with pytest.raises(ConfigurationError, match="no such folder"):
        skills_dir(tmp_path / "missing")


async def test_a_file_of_a_version_the_folder_no_longer_holds_is_refused(tmp_path: Path) -> None:
    folder = write_skill(tmp_path, "sql", "name: sql\ndescription: D.", "B", **{"a.md": "A"})
    source = skills_dir(tmp_path)
    pinned = await source.resolve("sql", None)
    (folder / "a.md").write_text("changed")
    with pytest.raises(ToolError, match=f"uses version {pinned.version}, and skills_dir"):
        await pinned.read("a.md")
    (folder / "SKILL.md").unlink()
    with pytest.raises(ToolError, match="has none now"):
        await pinned.read("a.md")


# --------------------------------------------------------------------------- Bifrost
async def test_the_gateways_skills_behind_the_protocol() -> None:
    fake = FakeGateway(
        skills={"sql": SkillVersions({"1.0.0": ("Reviews SQL.", "B", {"a.md": "A"})}, "1.0.0")}
    )
    source = BifrostSkills(fake.gateway())
    found = await source.resolve("sql", None)
    assert found == ResolvedSkill("sql", "1.0.0", "Reviews SQL.", "B", ("a.md",), "Bifrost")
    assert await found.read("a.md") == "A"
    with pytest.raises(LookupError, match="no skill named 'ghost'"):
        await source.resolve("ghost", None)
    await source.gateway.aclose()


# --------------------------------------------------------------------------- the order and the pin
def test_each_skill_is_named_once() -> None:
    assert refs_of(["a@1", TONE]) == [("a", "1", None), ("tone", "1", TONE)]
    for refs in ([], ["tone", TONE]):
        with pytest.raises(ConfigurationError, match="each skill once"):
            refs_of(refs)


async def test_skills_of_every_source_pin_together_the_first_source_wins(tmp_path: Path) -> None:
    write_skill(tmp_path, "sql", "name: sql\ndescription: From the folder.", "F")
    write_skill(tmp_path, "tone", "name: tone\ndescription: From the folder.", "F")
    fake = FakeGateway(
        skills={
            "sql": SkillVersions({"1.0.0": ("From the gateway.", "G", {})}, "1.0.0"),
            "refunds": SkillVersions({"2.0.0": ("Handles refunds.", "R", {})}, "2.0.0"),
        }
    )
    sources = SkillSources.of(Settings(skills_dir=str(tmp_path)), gateway=fake.gateway())
    assert sources.labels == [f"skills_dir({tmp_path})", "Bifrost"]
    pinned = await sources.pin(["sql", "refunds", TONE, "ghost", "sql-old@9"])
    assert {n: s.description for n, s in pinned.skills.items()} == {
        "sql": "From the folder.",  # found in both: the first in the order
        "refunds": "Handles refunds.",
        "tone": "How we write.",  # given: its own, before any source
    }
    assert pinned.versions["refunds"] == "2.0.0" and pinned.versions["tone"] == "1"
    assert pinned.section == "\n".join(
        [
            SECTION,
            "- sql: From the folder.",
            "- refunds: Handles refunds.",
            "- tone: How we write.",
        ]
    )
    assert set(pinned.problems) == {"ghost", "sql-old"}
    assert pinned.problems["ghost"].startswith(
        f"no skill 'ghost' in any source (skills_dir({tmp_path})"
    )
    assert "Bifrost: the gateway has no skill named 'ghost'" in pinned.problems["ghost"]
    await sources.aclose()
    await fake.gateway().aclose()


async def test_a_pin_restored_from_its_record_reads_the_same_whatever_the_source_holds(
    tmp_path: Path,
) -> None:
    folder = write_skill(tmp_path, "sql", "name: sql\ndescription: D.", "Body v1", **{"a.md": "A"})
    sources = SkillSources.of(Settings(skills_dir=str(tmp_path)))
    first = await sources.pin(["sql", TONE])
    record = first.record()
    (folder / "SKILL.md").write_text("---\nname: sql\ndescription: D2.\n---\nBody v2\n")
    again = await sources.pin(["sql", TONE], recorded=record)
    assert again.skills == first.skills  # the journal's body, description and files
    assert "Body v1" in await again.load("sql")
    with pytest.raises(ToolError, match="uses version"):
        await again.read("sql", "a.md")  # the folder holds another version now
    assert await again.read("tone", "words.md") == "Use: refund."  # the code's own Skill
    elsewhere = await SkillSources().pin(["sql", TONE], recorded=record)
    with pytest.raises(ToolError, match="no source is skills_dir"):
        await elsewhere.read("sql", "a.md")  # no such source in this process
    assert (await elsewhere.load("tone")).startswith("# tone (version 1)")


async def test_an_earlier_harness_journaled_versions_only() -> None:
    fake = FakeGateway(
        skills={
            "sql": SkillVersions(
                {"1.0.0": ("Old.", "v1", {}), "1.1.0": ("New.", "v2", {})}, "1.1.0"
            )
        }
    )
    sources = SkillSources([BifrostSkills(fake.gateway())])
    pinned = await sources.pin(["sql"], recorded={"sql": "1.0.0"})
    assert pinned.versions == {"sql": "1.0.0"}
    await sources.aclose()


async def test_names_with_no_source_at_all_are_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="no skill source: pass Harness"):
        await SkillSources().pin(["sql"])
    only_given = await SkillSources().pin([TONE])
    assert only_given.versions == {"tone": "1"}
    assert await SkillSources().pin([TONE], recorded={"tone": 3}) is not None


async def test_a_pinned_set_loads_and_reads_by_name() -> None:
    pinned = await SkillSources().pin([TONE])
    assert await pinned.load("tone") == (
        "# tone (version 1)\n\nShort sentences.\n\nFiles (read_skill_file):\n- words.md"
    )
    with pytest.raises(ToolError, match=r"no skill 'x' in this run \(its skills: tone\)"):
        await pinned.load("x")
    bare = await SkillSources().pin([Skill("bare", "No files.", "B")])
    assert await bare.load("bare") == "# bare (version 1)\n\nB"
    empty = type(pinned)()
    assert empty.section == "" and empty.record() == {}
    with pytest.raises(ToolError, match="its skills: none"):
        await empty.read("x", "y")
