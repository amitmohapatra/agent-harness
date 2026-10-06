"""Prompt sources: a resolved prompt rendered and read as messages, a prompt in code, a folder of
``.md`` files, Langfuse's prompt management (its public API, faked in its shapes), the
gateway's Prompt Repository, and the order they are asked in."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from tests.support import langfuse as lf
from tests.support.gateway import FakeGateway
from trellis import Settings
from trellis.contracts import ConfigurationError, HarnessError
from trellis.harness import fresh
from trellis.harness.clients.bifrost import Gateway, PromptPin
from trellis.harness.prompts import (
    BifrostPrompts,
    LangfusePrompts,
    Prompt,
    PromptSources,
    ResolvedPrompt,
    langfuse_prompts,
    prompts_dir,
)
from trellis.harness.repository import TTL_SECONDS, NotFound

SYSTEM = [{"role": "system", "content": "You triage."}]


# --------------------------------------------------------------------------- resolved
def test_a_text_prompt_is_rendered_with_its_variables() -> None:
    found = ResolvedPrompt("triage", "3", "code", text="Triage for {{ team }}, {{team}} only.")
    assert found.ref == "triage@3" and found.variables == ["team"]
    assert found.render(team="EU", unused=1) == "Triage for EU, EU only."
    assert found.messages(team="EU") == [{"role": "system", "content": "Triage for EU, EU only."}]
    with pytest.raises(ConfigurationError, match=r"triage@3 \(code\) needs team: pass them"):
        found.render()


def test_a_chat_prompt_fills_its_messages_and_placeholders() -> None:
    chat = (
        {"role": "system", "content": "You help {{user}}.", "type": "chatmessage"},
        {"type": "placeholder", "name": "history"},
        {"role": "user", "content": [{"type": "text", "text": "Hi {{user}}"}, {"type": "image"}]},
        {"role": "assistant", "content": None},
    )
    found = ResolvedPrompt("help", "1", "Langfuse", chat=chat)
    assert found.variables == ["user", "history"]
    history = [{"role": "user", "content": "earlier"}]
    assert found.messages(user="Ada", history=history) == [
        {"role": "system", "content": "You help Ada."},
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": [{"type": "text", "text": "Hi Ada"}, {"type": "image"}]},
        {"role": "assistant", "content": None},
    ]
    with pytest.raises(ConfigurationError, match="history is a list of messages"):
        found.messages(user="Ada", history="no")
    with pytest.raises(ConfigurationError, match="is 4 chat messages: read it as messages"):
        found.render(user="Ada", history=history)
    one = ResolvedPrompt("one", "1", "Bifrost", chat=({"role": "system", "content": "Be {{x}}."},))
    assert one.render(x="brief") == "Be brief."
    assert ResolvedPrompt("none", "1", "Bifrost", chat=()).variables == []


def test_a_prompt_says_how_a_call_selects_it_and_what_a_span_says() -> None:
    stored = ResolvedPrompt("triage", "2", "Bifrost", chat=tuple(SYSTEM), selection="p-triage")
    assert stored.pin() == PromptPin("triage", "p-triage", 2)
    assert stored.attributes() == {
        "trellis.prompt.name": "triage",
        "trellis.prompt.id": "p-triage",
        "trellis.prompt.version": 2,
        "trellis.prompt.source": "Bifrost",
    }
    text = ResolvedPrompt("greet", "1", "code", text="Hi")
    assert text.pin() is None
    assert text.attributes() == {
        "trellis.prompt.name": "greet",
        "trellis.prompt.version": "1",
        "trellis.prompt.source": "code",
    }


def test_a_prompt_comes_back_from_the_journal_as_it_was() -> None:
    for found in (
        ResolvedPrompt("a", "1", "code", text="x {{y}}", config={"t": 0}),
        ResolvedPrompt("b", "2", "Bifrost", chat=tuple(SYSTEM), selection="p-b"),
    ):
        assert ResolvedPrompt.of_record(found.record()) == found
    # an earlier harness journaled a ReAct's stored prompt as its selection only
    old = ResolvedPrompt.of_record({"name": "triage", "id": "p-triage", "version": 2})
    assert old == ResolvedPrompt("triage", "2", "Bifrost", selection="p-triage")


# --------------------------------------------------------------------------- code
async def test_a_prompt_in_code_is_its_own_source() -> None:
    prompt = Prompt("greet", "Hello {{name}}.", config={"temperature": 0})
    assert prompt.label == "code"
    found = await prompt.resolve("greet", None)
    assert found == ResolvedPrompt(
        "greet", "1", "code", "Hello {{name}}.", None, {"temperature": 0}
    )
    assert (await prompt.resolve("greet", "1")) == found
    for name, version in (("other", None), ("greet", "2")):
        with pytest.raises(NotFound, match="holds greet@1 only"):
            await prompt.resolve(name, version)
    chat = Prompt("chat", [{"role": "system", "content": "Hi"}], version="7").resolved
    assert chat.chat == ({"role": "system", "content": "Hi"},) and chat.text is None
    for bad in (("", "x"), ("a@1", "x")):
        with pytest.raises(ConfigurationError, match="a name without @"):
            Prompt(*bad)
    with pytest.raises(ConfigurationError):
        Prompt("a", "x", version="")


# --------------------------------------------------------------------------- files
async def test_a_folder_of_prompts(tmp_path: Path) -> None:
    (tmp_path / "support").mkdir()
    (tmp_path / "triage.md").write_text(
        "---\nversion: 3\ndescription: Triage tickets.\ntemperature: 0\n---\n\nTriage for {{team}}.\n"
    )
    (tmp_path / "support" / "greet.md").write_text("Hello.\n")
    (tmp_path / "outside.md").write_text("no")
    source = prompts_dir(tmp_path / "support")
    assert source.label == f"prompts_dir({tmp_path / 'support'})"
    found = await prompts_dir(tmp_path).resolve("triage", None)
    assert found == ResolvedPrompt(
        "triage",
        "3",
        f"prompts_dir({tmp_path})",
        text="Triage for {{team}}.",
        config={"description": "Triage tickets.", "temperature": "0"},
    )
    assert (await prompts_dir(tmp_path).resolve("triage", "3")).version == "3"
    greet = await prompts_dir(tmp_path).resolve("support/greet", None)
    assert greet.text == "Hello." and len(greet.version) == 12  # its content digest
    (tmp_path / "support" / "greet.md").write_text("Hello again.\n")
    again = await source.resolve("greet", None)
    assert again.version != greet.version  # what is on disk now
    with pytest.raises(NotFound, match=r"triage\.md is version 3, not 2"):
        await prompts_dir(tmp_path).resolve("triage", "2")
    with pytest.raises(NotFound, match=r"it has no nope\.md"):
        await source.resolve("nope", None)
    with pytest.raises(NotFound, match="is not a file name inside it"):
        await source.resolve("../outside", None)
    with pytest.raises(ConfigurationError, match="no such folder"):
        prompts_dir(tmp_path / "missing")


# --------------------------------------------------------------------------- Langfuse
@pytest.fixture
def langfuse() -> lf.LangfusePrompts:
    return lf.LangfusePrompts(
        prompts={
            "triage": ["Old {{team}}.", "Triage for {{team}}."],
            "team/chat": [[{"role": "system", "content": "Hi", "type": "chatmessage"}]],
        },
        labels={("triage", "staging"): 1},
        configs={"triage": {"temperature": 0}},
    )


def source() -> LangfusePrompts:
    return langfuse_prompts(public_key=lf.PUBLIC, secret_key=lf.SECRET, host=lf.HOST + "/")


@respx.mock
async def test_langfuse_prompts_by_label_version_and_kind(langfuse: lf.LangfusePrompts) -> None:
    route = respx.get(url__startswith=lf.HOST + lf.PATH).mock(side_effect=langfuse.handle)
    prompts = source()
    assert prompts.label == f"Langfuse ({lf.HOST})"
    production = await prompts.resolve("triage", None)
    assert production == ResolvedPrompt(
        "triage", "2", prompts.label, text="Triage for {{team}}.", config={"temperature": 0}
    )
    assert (await prompts.resolve("triage", "1")).text == "Old {{team}}."
    assert (await prompts.resolve("triage", "staging")).version == "1"
    chat = await prompts.resolve("team/chat", None)
    assert chat.chat == ({"role": "system", "content": "Hi", "type": "chatmessage"},)
    assert chat.text is None and chat.messages() == [{"role": "system", "content": "Hi"}]
    assert langfuse.asked == [
        ("triage", {"label": "production"}),
        ("triage", {"version": "1"}),
        ("triage", {"label": "staging"}),
        ("team/chat", {"label": "production"}),
    ]
    assert route.calls[-1].request.url.raw_path.startswith(b"/api/public/v2/prompts/team%2Fchat?")
    with pytest.raises(NotFound, match="no prompt 'triage' version 9"):
        await prompts.resolve("triage", "9")
    with pytest.raises(NotFound, match="no prompt 'ghost' labelled production"):
        await prompts.resolve("ghost", None)
    await prompts.aclose()


@respx.mock
async def test_langfuse_down_the_last_copy_stands_and_a_prompt_never_read_says_so(
    langfuse: lf.LangfusePrompts, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(fresh, "_now", lambda: clock[0])
    route = respx.get(url__startswith=lf.HOST + lf.PATH).mock(side_effect=langfuse.handle)
    prompts = source()
    first = await prompts.resolve("triage", None)
    await prompts.resolve("triage", "1")
    clock[0] += TTL_SECONDS + 1
    route.mock(side_effect=httpx.ConnectError("down"))
    assert await prompts.resolve("triage", None) == first  # the last good copy
    assert (await prompts.resolve("triage", "1")).version == "1"  # a version: read once
    with pytest.raises(HarnessError, match="never read before: ConnectError: down") as raised:
        await prompts.resolve("ghost", None)
    assert raised.value.retryable
    route.mock(return_value=httpx.Response(401, json={"message": "Invalid credentials"}))
    with pytest.raises(HarnessError, match="HTTPStatusError"):
        await prompts.resolve("other", None)
    await prompts.aclose()


def test_langfuse_is_its_cloud_unless_a_host_is_named() -> None:
    assert langfuse_prompts(public_key="p", secret_key="s").host == "https://cloud.langfuse.com"


# --------------------------------------------------------------------------- Bifrost
async def test_the_gateways_prompts_by_number_with_their_messages() -> None:
    older = [{"role": "system", "content": "Old."}]
    fake = FakeGateway(prompts={"triage": [older, SYSTEM]})
    prompts = BifrostPrompts(fake.gateway())
    assert prompts.label == "Bifrost"
    latest = await prompts.resolve("triage", None)
    assert latest == ResolvedPrompt(
        "triage", "2", "Bifrost", chat=tuple(SYSTEM), selection="p-triage"
    )
    assert (await prompts.resolve("triage", "1")).chat == tuple(older)
    assert fake.asked("/api/prompt-repo/prompts/p-triage/versions") == 1
    for version in ("staging", "0"):
        with pytest.raises(NotFound, match="a stored prompt's version is a number from 1"):
            await prompts.resolve("triage", version)
    with pytest.raises(NotFound, match="no committed prompt named 'ghost'"):
        await prompts.resolve("ghost", None)
    await prompts.gateway.aclose()


@respx.mock
async def test_a_version_the_gateway_does_not_list_is_not_found() -> None:
    gw = "http://gw.test"
    latest = {"id": 9, "prompt_id": "p-1", "version_number": 3, "messages": []}
    respx.get(f"{gw}/api/prompt-repo/prompts").mock(
        return_value=httpx.Response(
            200, json={"prompts": [{"id": "p-1", "name": "triage", "latest_version": latest}]}
        )
    )
    respx.get(f"{gw}/api/prompt-repo/prompts/p-1/versions").mock(
        return_value=httpx.Response(200, json={"versions": [latest]})
    )
    gateway = Gateway(f"{gw}/v1", "vk")
    with pytest.raises(NotFound, match="has no version 2"):
        await gateway.prompt("triage@2")
    await gateway.aclose()


# --------------------------------------------------------------------------- the order
@respx.mock
async def test_the_environments_order_is_the_folder_then_langfuse_then_the_gateway(
    tmp_path: Path, langfuse: lf.LangfusePrompts
) -> None:
    respx.get(url__startswith=lf.HOST + lf.PATH).mock(side_effect=langfuse.handle)
    (tmp_path / "triage.md").write_text("From the folder.")
    fake = FakeGateway(prompts={"triage": [SYSTEM], "gw": [SYSTEM]})
    settings = Settings(
        prompts_dir=str(tmp_path),
        langfuse_host=lf.HOST,
        langfuse_public_key=lf.PUBLIC,
        langfuse_secret_key=lf.SECRET,
    )
    sources = PromptSources.of(settings, gateway=fake.gateway())
    assert sources.labels == [f"prompts_dir({tmp_path})", f"Langfuse ({lf.HOST})", "Bifrost"]
    assert await sources.render("triage") == "From the folder."  # in all three: the first
    assert await sources.render("triage@2", team="EU") == "Triage for EU."  # Langfuse: v2
    assert await sources.messages("gw") == SYSTEM
    with pytest.raises(ConfigurationError) as raised:
        await sources.get("ghost")
    assert str(raised.value) == (
        "no prompt 'ghost' in any source ("
        f"prompts_dir({tmp_path}): it has no ghost.md; Langfuse ({lf.HOST}): no prompt "
        "'ghost' labelled production; Bifrost: the gateway has no committed prompt named "
        "'ghost')"
    )
    assert await sources.render(Prompt("inline", "Inline {{x}}."), x=1) == "Inline 1."
    await sources.aclose()


async def test_no_prompt_source_says_how_to_name_one() -> None:
    sources = PromptSources.of(Settings())
    with pytest.raises(ConfigurationError, match=r"no prompt source for 'a@1': pass Harness"):
        await sources.get("a@1")
