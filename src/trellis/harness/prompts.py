"""Stored prompts: the gateway's Prompt Repository (Bifrost) is where a team keeps its prompts
— versioned, edited without a redeploy — and the one place the harness reads them from.

    agent = h.wrap(ReAct(system_prompt=stored("triage", team="EU"), model="gpt-4o"), id="triage")
    prompt = await h.prompt("triage@3", team="EU")       # any framework: .text, .messages

``"name"`` is the version the gateway serves now, ``"name@3"`` committed version 3
(:func:`~trellis.harness.repository.pinned`). Inside a run what was resolved is pinned —
journaled, so a resumed run reads the same version even after the prompt changed — and said: a
``prompt`` event, and the ``trellis.prompt.*`` attributes of the span current then. A stored
prompt with no ``{{variables}}`` to fill is *selected* by a ``ReAct`` model call through
Bifrost (its headers: the gateway prepends it); one with variables, or off Bifrost, is rendered
into the instructions. A prompt in code is a string.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final

from bifrost_sdk import NO_GATEWAY_TOOLS

from trellis.contracts import ConfigurationError
from trellis.harness import telemetry
from trellis.harness.clients.bifrost import Gateway, PromptPin
from trellis.harness.journal import content_key
from trellis.harness.repository import BIFROST, PINNED, NotFound, journaled, named, pinned
from trellis.harness.runtime import current

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: The event (and the span attribute ``trellis.prompt``) a run's pinned prompt is said with.
EVENT: Final = "prompt"
#: A ``{{variable}}`` placeholder (spaces inside the braces allowed).
VARIABLE: Final = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
#: The Claude Code CLI's own extra request headers (``Name: value`` lines), read as it starts.
CUSTOM_HEADERS: Final = "ANTHROPIC_CUSTOM_HEADERS"


@dataclass(frozen=True, slots=True)
class Stored:
    """A stored prompt of the gateway (:func:`stored`): its reference and the values its
    ``{{variables}}`` are filled with."""

    ref: str
    values: Mapping[str, Any] = field(default_factory=dict)


def stored(ref: str, /, **values: Any) -> Stored:
    """The gateway's stored prompt ``ref`` (``"name"`` or ``"name@version"``), ``values``
    filling its ``{{variables}}``: ``ReAct(system_prompt=stored("triage", team="EU"))``."""
    pinned(ref)  # a malformed reference is refused where it is written
    return Stored(ref, dict(values))


@dataclass(frozen=True, slots=True)
class Rendered:
    """A prompt with its variables filled (``h.prompt``): its text, and its chat messages."""

    text: str
    messages: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """One version of a stored prompt, as the gateway gave it: its chat messages, and the id a
    model call through the gateway selects it by."""

    name: str
    version: str
    #: where it came from (the gateway's repository)
    source: str
    chat: tuple[dict[str, Any], ...] = ()
    #: its id in the gateway, which a model call selects it by
    selection: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.name}{PINNED}{self.version}"

    @property
    def variables(self) -> list[str]:
        """The ``{{variables}}`` it needs filled, in order."""
        found: list[str] = []
        for text in self._texts():
            found.extend(VARIABLE.findall(text))
        return list(dict.fromkeys(found))

    def render(self, **values: Any) -> str:
        """The text with ``values`` filled in: its messages' contents, one after the other."""
        texts = (message_text(m.get("content")) for m in self.messages(**values))
        return "\n\n".join(t for t in texts if t)

    def messages(self, **values: Any) -> list[dict[str, Any]]:
        """Its chat messages with ``values`` filled in. ``ConfigurationError`` for a variable
        left unfilled."""
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise ConfigurationError(
                f"the prompt {self.ref} needs {', '.join(missing)}: pass them "
                f"(stored({self.name!r}, {missing[0]}=...), h.prompt(ref, {missing[0]}=...))"
            )
        return [
            {k: v for k, v in m.items() if k != "type"} | {"content": _content(m, values)}
            for m in self.chat
        ]

    def pin(self) -> PromptPin | None:
        """How a model call through the gateway selects it."""
        if self.selection is None:
            return None
        return PromptPin(self.name, self.selection, int(self.version), self.chat)

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
        found["chat"] = list(self.chat)
        return found

    @classmethod
    def of_record(cls, data: Mapping[str, Any]) -> ResolvedPrompt:
        """It again, from the journal."""
        return cls(
            name=data["name"],
            version=str(data["version"]),
            source=data.get("source") or BIFROST,
            chat=tuple(dict(m) for m in data.get("chat") or ()),
            selection=data.get("selection") or data.get("id"),
        )

    def _texts(self) -> list[str]:
        texts: list[str] = []
        for message in self.chat:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(p["text"] for p in content if isinstance(p, dict) and "text" in p)
        return texts


def _filled(text: str, values: Mapping[str, Any]) -> str:
    return VARIABLE.sub(lambda m: str(values[m.group(1)]), text)


def _content(message: Mapping[str, Any], values: Mapping[str, Any]) -> Any:
    content = message.get("content")
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


class Prompts:
    """The gateway's stored prompts (``Gateway.prompt``: resolved once, kept fresh), pinned
    inside a run."""

    def __init__(self, gateway: Gateway | None) -> None:
        self.gateway = gateway

    async def get(self, ref: str, *, runtime: Runtime | None = None) -> ResolvedPrompt:
        """The stored prompt ``ref`` names. Inside a run (``runtime``) it is pinned: an earlier
        attempt's, from the journal, else resolved now and journaled, and said (its ``prompt``
        event, the ``trellis.prompt`` attribute)."""
        name, version = pinned(ref)

        async def read() -> ResolvedPrompt:
            return await self._resolve(name, version)

        found = await journaled(
            runtime,
            content_key(EVENT, ref),
            read,
            dump=ResolvedPrompt.record,
            load=ResolvedPrompt.of_record,
        )
        if runtime is not None:
            runtime.events.custom(
                EVENT, prompt=found.name, version=found.version, source=found.source
            )
            telemetry.attribute("trellis.prompt", found.ref)
        return found

    async def pin(self, ref: str, runtime: Runtime) -> ResolvedPrompt:
        """The stored prompt ``ref`` as ``runtime``'s run uses it on every model call: pinned
        once per attempt (:meth:`get`), kept on the runtime, and said on the span current then
        — the run's agent span — as ``trellis.prompt.*``."""
        key = content_key(EVENT, ref)
        found = runtime.prompts.get(key)
        if found is None:
            found = runtime.prompts[key] = await self.get(ref, runtime=runtime)
            telemetry.attributes(found.attributes())
        return found

    async def rendered(self, ref: str | Stored, /, **values: Any) -> Rendered:
        """The stored prompt with its variables filled (pinned inside a run): its text and its
        chat messages."""
        given = ref if isinstance(ref, Stored) else Stored(ref)
        filled = {**given.values, **values}
        found = await self.get(given.ref, runtime=current())
        return Rendered(found.render(**filled), found.messages(**filled))

    async def _resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        if self.gateway is None:
            raise ConfigurationError(
                f"the prompt {named(name, version)!r} is a stored prompt of the gateway: set "
                "BIFROST_URL (a prompt in code is a string)"
            )
        if version is not None and not (version.isdigit() and int(version) >= 1):
            raise NotFound(f"a stored prompt's version is a number from 1, not {version!r}")
        pin = await self.gateway.prompt(named(name, version))
        return ResolvedPrompt(
            name=pin.name,
            version=str(pin.version),
            source=BIFROST,
            chat=tuple(pin.messages),
            selection=pin.id,
        )


class GatewayHeaders(Mapping[str, str]):
    """The gateway's deny-all MCP scope and — read each time they are read, so a client that
    keeps the mapping (``AsyncOpenAI(default_headers=...)``) sends them per request — the run's
    Bifrost dimensions and its ``traceparent``."""

    def now(self) -> dict[str, str]:
        from trellis.harness.telemetry import gateway_headers  # noqa: PLC0415

        return {**NO_GATEWAY_TOOLS, **gateway_headers()}

    def __getitem__(self, name: str) -> str:
        return self.now()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.now())

    def __len__(self) -> int:
        return len(self.now())

    def __repr__(self) -> str:
        return f"GatewayHeaders({self.now()!r})"


def message_text(content: Any) -> str:
    """A message's text: a string as it is, the text parts of a multi-part content."""
    if isinstance(content, str):
        return content
    parts = content if isinstance(content, list) else []
    return "\n".join(p["text"] for p in parts if isinstance(p, dict) and p.get("type") == "text")
