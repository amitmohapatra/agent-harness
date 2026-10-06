"""Prompts, from wherever a team keeps them, for any framework: ``h.prompt("triage@3", **vars)``.

    h = Harness(prompts=[prompts_dir("prompts"), Prompt("greet", "Greet {{name}}.")])
    agent = h.wrap(ReAct(system="", model=model, prompt="triage", prompt_vars={"team": "EU"}),
                   id="triage")
    instructions = await h.prompt("triage@3", team="EU")      # LangGraph, OpenAI Agents, ...

A source answers ``resolve(name, version) -> ResolvedPrompt`` (:class:`PromptSource`): the text
— or the chat messages — with ``{{var}}`` placeholders, the version it is, the source it came
from and its config. ``version=None`` is the version the source serves now. Shipped:

* code — :class:`Prompt` (``Prompt("greet", "Greet {{name}}.", version="1")``);
* files — :func:`prompts_dir`: ``<name>.md`` (a ``support/triage.md`` is ``support/triage``),
  its optional front matter a ``version`` (else the file's content digest), a ``description``
  and any config;
* Langfuse — :class:`LangfusePrompts`, Langfuse's prompt management (its public API: a version
  by number, a label by name, ``production`` by default; text and chat prompts), reached with
  ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` at ``LANGFUSE_HOST``;
* Bifrost — :class:`BifrostPrompts`, the gateway's Prompt Repository (``BIFROST_URL``): where
  the harness sets the model call's headers (``ReAct``, the LLM judge) the stored prompt is
  *selected* — the gateway prepends it — and elsewhere its messages are its text.

Where a name is looked up (:class:`PromptSources`): the sources the code passes
(``Harness(prompts=[...])``, ``[]`` for none) in their order, instead of the environment's;
not passed, ``PROMPTS_DIR``, then Langfuse (its keys set), then Bifrost (``BIFROST_URL`` set).
The first that has it answers; a :class:`Prompt` given where a prompt is named is its own. A
name none has is a ``ConfigurationError`` naming each source tried. Inside a run what was
resolved is pinned — journaled, so a resumed run reads the same text even after the source
changed — and said: a ``prompt`` event, and the ``trellis.prompt`` attribute of the span
current then.

A remote source keeps what it read (``repository.Kept``): while Langfuse cannot be reached the
last copy read stands; a prompt never read is an error that says so.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from trellis.contracts import ConfigurationError, HarnessError
from trellis.harness import telemetry
from trellis.harness.clients.bifrost import Gateway, PromptPin
from trellis.harness.journal import content_key
from trellis.harness.repository import (
    BIFROST,
    PINNED,
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
from trellis.harness.runtime import current
from trellis.harness.settings import Settings

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: The event (and the span attribute ``trellis.prompt``) a run's pinned prompt is said with.
EVENT: Final = "prompt"
#: Langfuse's own defaults: its cloud, and the label a prompt is served by.
LANGFUSE_HOST: Final = "https://cloud.langfuse.com"
PRODUCTION: Final = "production"
#: A ``{{variable}}`` placeholder (Langfuse's syntax; spaces inside the braces allowed).
VARIABLE: Final = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """One version of a prompt, as its source gave it: a text, or chat messages."""

    name: str
    version: str
    #: which source it came from (its ``label``)
    source: str
    text: str | None = None
    chat: tuple[dict[str, Any], ...] | None = None
    #: what the source keeps beside it (Langfuse's ``config``, a file's other front matter)
    config: dict[str, Any] = field(default_factory=dict)
    #: a stored prompt of the gateway: its id, which a model call selects it by
    selection: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.name}{PINNED}{self.version}"

    @property
    def variables(self) -> list[str]:
        """The ``{{variables}}`` it needs filled, and the chat placeholders, in order."""
        found: list[str] = []
        for text in self._texts():
            found.extend(VARIABLE.findall(text))
        found.extend(str(m.get("name")) for m in self.chat or () if _placeholder(m))
        return list(dict.fromkeys(found))

    def render(self, **values: Any) -> str:
        """The text with ``values`` filled in; a chat prompt of one message reads as its text.
        ``ConfigurationError`` for a variable left unfilled."""
        if self.text is not None:
            self._check(values)
            return _filled(self.text, values)
        rendered = self.messages(**values)
        if len(rendered) == 1 and isinstance(rendered[0].get("content"), str):
            return rendered[0]["content"]
        raise ConfigurationError(
            f"the prompt {self.ref} ({self.source}) is {len(rendered)} chat messages: read it "
            "as messages (h.prompt_messages)"
        )

    def messages(self, **values: Any) -> list[dict[str, Any]]:
        """Chat messages with ``values`` filled in: a text prompt is one system message; a
        chat placeholder (Langfuse's) is the list of messages its value is."""
        self._check(values)
        if self.text is not None:
            return [{"role": "system", "content": _filled(self.text, values)}]
        rendered: list[dict[str, Any]] = []
        for message in self.chat or ():
            if _placeholder(message):
                filled = values[str(message.get("name"))]
                if not isinstance(filled, list):
                    raise ConfigurationError(
                        f"the prompt {self.ref}: {message.get('name')} is a list of messages"
                    )
                rendered.extend(dict(m) for m in filled)
                continue
            content = message.get("content")
            rendered.append(
                {k: v for k, v in message.items() if k != "type"}
                | {"content": _content(content, values)}
            )
        return rendered

    def pin(self) -> PromptPin | None:
        """How a model call through the gateway selects it (a stored prompt of the gateway's)."""
        if self.selection is None:
            return None
        return PromptPin(self.name, self.selection, int(self.version), self.chat or ())

    def attributes(self) -> dict[str, Any]:
        """What a ``chat`` span says about it."""
        pin = self.pin()
        named_ = (
            pin.attributes()
            if pin is not None
            else {"trellis.prompt.name": self.name, "trellis.prompt.version": self.version}
        )
        return {**named_, "trellis.prompt.source": self.source}

    def record(self) -> dict[str, Any]:
        """What the journal keeps of it."""
        found = asdict(self)
        found["chat"] = None if self.chat is None else list(self.chat)
        return found

    @classmethod
    def of_record(cls, data: Mapping[str, Any]) -> ResolvedPrompt:
        """It again, from the journal (an earlier harness journaled a ``ReAct``'s stored
        prompt as its gateway selection only: ``name``, ``id``, ``version``)."""
        if "source" not in data:
            return cls(
                name=data["name"],
                version=str(data["version"]),
                source=BIFROST,
                selection=data["id"],
            )
        chat = data.get("chat")
        return cls(
            name=data["name"],
            version=data["version"],
            source=data["source"],
            text=data.get("text"),
            chat=None if chat is None else tuple(dict(m) for m in chat),
            config=dict(data.get("config") or {}),
            selection=data.get("selection"),
        )

    def _texts(self) -> list[str]:
        if self.text is not None:
            return [self.text]
        texts: list[str] = []
        for message in self.chat or ():
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(p["text"] for p in content if isinstance(p, dict) and "text" in p)
        return texts

    def _check(self, values: Mapping[str, Any]) -> None:
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise ConfigurationError(
                f"the prompt {self.ref} ({self.source}) needs {', '.join(missing)}: pass "
                "them (h.prompt(ref, name=value), ReAct(prompt_vars=...))"
            )


def _placeholder(message: Mapping[str, Any]) -> bool:
    return message.get("type") == "placeholder"


def _filled(text: str, values: Mapping[str, Any]) -> str:
    return VARIABLE.sub(lambda m: str(values[m.group(1)]), text)


def _content(content: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(content, str):
        return _filled(content, values)
    if isinstance(content, list):
        return [
            {**p, "text": _filled(p["text"], values)}
            if isinstance(p, dict) and isinstance(p.get("text"), str)
            else p
            for p in content
        ]
    return content


class PromptSource(Protocol):
    """Where prompts come from: ``resolve`` raises a ``LookupError`` for a name — or a
    version — it does not have."""

    @property
    def label(self) -> str: ...

    async def resolve(self, name: str, version: str | None) -> ResolvedPrompt: ...


# --------------------------------------------------------------------------- code


@dataclass(frozen=True, slots=True)
class Prompt:
    """A prompt in code: a text (or chat messages) with ``{{var}}`` placeholders. It is its
    own source: ``Harness(prompts=[Prompt(...)])``, or given where a prompt is named
    (``ReAct(prompt=Prompt(...))``, ``h.prompt(Prompt(...))``)."""

    name: str
    text: str | Sequence[Mapping[str, Any]]
    version: str = "1"
    config: Mapping[str, Any] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        if not self.name or PINNED in self.name or not self.version:
            raise ConfigurationError(f"Prompt({self.name!r}): a name without {PINNED}, a version")

    @property
    def label(self) -> str:
        return "code"

    @property
    def resolved(self) -> ResolvedPrompt:
        text, chat = (
            (self.text, None)
            if isinstance(self.text, str)
            else (None, tuple(dict(m) for m in self.text))
        )
        return ResolvedPrompt(
            self.name, self.version, self.label, text, chat, dict(self.config), None
        )

    async def resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        if name != self.name or version not in (None, self.version):
            raise NotFound(f"holds {self.name}{PINNED}{self.version} only")
        return self.resolved


# --------------------------------------------------------------------------- files


class PromptsDir:
    """A folder of prompts (:func:`prompts_dir`), read at each resolve: what is on disk now."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise ConfigurationError(f"prompts_dir({str(self.root)!r}): no such folder")

    @property
    def label(self) -> str:
        return f"prompts_dir({self.root})"

    async def resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        return await asyncio.to_thread(self._resolve, name, version)

    def _resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        file = inside(self.root, f"{name}.md")
        if file is None:
            raise NotFound(f"{name!r} is not a file name inside it (refused)")
        if not file.is_file():
            raise NotFound(f"it has no {name}.md")
        raw = file.read_bytes()
        meta, body = front_matter(raw.decode("utf-8", errors="replace"), where=str(file))
        current = str(meta.pop("version", "") or digest(raw))
        if version not in (None, current):
            raise NotFound(f"{name}.md is version {current}, not {version}")
        return ResolvedPrompt(name, current, self.label, text=body.strip(), config=meta)


def prompts_dir(path: str | os.PathLike[str]) -> PromptsDir:
    """A folder of prompts: ``<name>.md`` each (in subfolders too: ``support/triage``), with an
    optional front matter — ``version`` (else the file's content digest), ``description``,
    anything else kept as its config — and ``{{var}}`` placeholders in the body. A name that
    would leave the folder (``..``, an absolute path, a link elsewhere) is refused."""
    return PromptsDir(path)


# --------------------------------------------------------------------------- Langfuse


class LangfusePrompts:
    """Langfuse's prompt management, through its public API (``GET
    /api/public/v2/prompts/{name}``): ``name@3`` is version 3, ``name@staging`` the version
    labelled ``staging``, ``name`` the one labelled ``production``. A version is read once; a
    label again every ``repository.TTL_SECONDS``, the last copy standing while Langfuse cannot
    be reached."""

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str | None = None,
        client: telemetry.Langfuse | None = None,
    ) -> None:
        self.host = (host or LANGFUSE_HOST).rstrip("/")
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.client = client or telemetry.Langfuse(self.host, f"Basic {token}")
        self._kept: Kept[tuple[str, str], dict[str, Any]] = Kept(
            self._read, what=lambda key: f"the Langfuse prompt {named(*key)!r}"
        )

    @property
    def label(self) -> str:
        return f"Langfuse ({self.host})"

    async def resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        chosen = version or PRODUCTION
        number = chosen.isdigit()
        try:
            found = await self._kept.get((name, chosen), forever=number)
        except Exception as exc:
            raise HarnessError(
                f"the prompt {named(name, version)!r} could not be read from {self.label} and "
                f"was never read before: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc
        if found is None:
            how = "version" if number else "labelled"
            raise NotFound(f"no prompt {name!r} {how} {chosen}")
        body = found.get("prompt")
        chat = found.get("type") == "chat" or isinstance(body, list)
        return ResolvedPrompt(
            name=str(found.get("name") or name),
            version=str(found["version"]),
            source=self.label,
            text=None if chat else str(body),
            chat=tuple(dict(m) for m in body or ()) if chat else None,
            config=dict(found.get("config") or {}),
        )

    async def _read(self, key: tuple[str, str]) -> dict[str, Any] | None:
        name, chosen = key
        if chosen.isdigit():
            return await self.client.prompt(name, version=int(chosen))
        return await self.client.prompt(name, label=chosen)

    async def aclose(self) -> None:
        await self.client.aclose()


def langfuse_prompts(
    *, public_key: str, secret_key: str, host: str | None = None
) -> LangfusePrompts:
    """Langfuse's prompts, with this project's keys (``host`` unset: Langfuse Cloud). A
    deployment that sets ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` has it already."""
    return LangfusePrompts(public_key=public_key, secret_key=secret_key, host=host)


# --------------------------------------------------------------------------- Bifrost


class BifrostPrompts:
    """The gateway's Prompt Repository: ``name@3`` is committed version 3, ``name`` the latest
    (``Gateway.prompt``: resolved once, kept fresh). Its id is what a model call through the
    gateway selects it by; its messages are its text elsewhere."""

    @property
    def label(self) -> str:
        return BIFROST

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway

    async def resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        if version is not None and not (version.isdigit() and int(version) >= 1):
            raise NotFound(f"a stored prompt's version is a number from 1, not {version!r}")
        pin = await self.gateway.prompt(named(name, version))
        return ResolvedPrompt(
            name=pin.name,
            version=str(pin.version),
            source=self.label,
            chat=pin.messages,
            selection=pin.id,
        )


# --------------------------------------------------------------------------- the order


class PromptSources(Chain[ResolvedPrompt]):
    """The prompt sources, in the order they are asked: the ones passed, or the deployment's
    (:meth:`of`: ``PROMPTS_DIR``, Langfuse, Bifrost)."""

    kind = "prompt"
    hint = "pass Harness(prompts=[...]), or set PROMPTS_DIR, the Langfuse keys or BIFROST_URL"

    def __init__(self, sources: Sequence[PromptSource] = ()) -> None:
        super().__init__(sources)

    @classmethod
    def of(cls, settings: Settings, *, gateway: Gateway | None = None) -> PromptSources:
        """The sources ``settings`` name, in this order: ``prompts_dir``, Langfuse (both keys
        set), the gateway (``gateway``)."""
        sources: list[PromptSource] = []
        if settings.prompts_dir:
            sources.append(prompts_dir(settings.prompts_dir))
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            sources.append(
                LangfusePrompts(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            )
        if gateway is not None:
            sources.append(BifrostPrompts(gateway))
        return cls(sources)

    async def get(
        self, ref: str | Prompt, *, runtime: Runtime | None = None, kind: str = EVENT
    ) -> ResolvedPrompt:
        """The prompt ``ref`` names (``"name"``, ``"name@version"``, or a :class:`Prompt`), as
        the first source that has it gives it. Inside a run (``runtime``) it is pinned: an
        earlier attempt's, from the journal, else resolved now and journaled."""
        if isinstance(ref, Prompt):
            key = content_key(kind, [ref.name, ref.version])

            async def read() -> ResolvedPrompt:
                return ref.resolved

        else:
            name, version = pinned(ref)
            key = content_key(kind, ref)

            async def read() -> ResolvedPrompt:
                return await self.find(name, version)

        found = await journaled(
            runtime, key, read, dump=ResolvedPrompt.record, load=ResolvedPrompt.of_record
        )
        if runtime is not None:
            runtime.events.custom(
                EVENT, prompt=found.name, version=found.version, source=found.source
            )
            telemetry.attribute("trellis.prompt", found.ref)
        return found

    async def render(self, ref: str | Prompt, /, **values: Any) -> str:
        """The prompt's text, ``values`` filled in (pinned inside a run)."""
        return (await self.get(ref, runtime=current())).render(**values)

    async def messages(self, ref: str | Prompt, /, **values: Any) -> list[dict[str, Any]]:
        """The prompt as chat messages, ``values`` filled in (pinned inside a run)."""
        return (await self.get(ref, runtime=current())).messages(**values)
