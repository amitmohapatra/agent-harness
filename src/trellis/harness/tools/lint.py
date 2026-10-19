"""What in a tool set makes a model choose its tools badly, found before any run: tools whose
description is missing or too short to choose by, parameters nobody described, names a model
API refuses or that differ only in case and separators, tools described the same or almost the
same way (a model cannot tell when to use which), and more tools at once than a model chooses
among well — or than a provider takes.

:func:`lint` reads the tools' definitions (``ToolSpec``: a function's, an MCP server's through
the gateway, an OpenAPI operation's — whatever the source) and says what it found; it changes
nothing. The toolbox runs it each time it lists the tools anew and logs each finding once
(``trellis.tools``, a warning); a team that wants a tool set clean in CI asserts
``lint(...) == []`` in a test of its own. What a description should say — what the tool is
for, when to use it rather than a broader or narrower one — only its author knows: the
findings say where it is missing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Final, Literal

from trellis.contracts import ToolSpec

Code = Literal[
    "no_description",
    "short_description",
    "undescribed_parameter",
    "invalid_name",
    "similar_names",
    "same_description",
    "overlapping_descriptions",
    "many_tools",
    "too_many_tools",
]

#: Fewer words than this do not say what a tool is for, let alone when to use it.
MIN_DESCRIPTION_WORDS: Final = 5
#: The tool names the OpenAI API takes (letters, digits, ``_`` and ``-``, at most 64): a name
#: outside it fails the model call there and on gateways that follow it.
NAME: Final = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
#: Two descriptions sharing this share of their words (Jaccard) or more read as one: say in
#: each when to use it rather than the other.
OVERLAP: Final = 0.75
#: From this many tools offered at once, narrow them (the memory service's tool hints, a tool
#: selector): OpenAI's function-calling guide suggests fewer than 20 for accuracy.
MANY_TOOLS: Final = 20
#: The most tools the OpenAI API takes in one request.
MOST_TOOLS: Final = 128
#: Words that say nothing about when to use a tool.
_STOP: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing to fix: which tool (``None``: the set), what, and how to say it better."""

    tool: str | None
    code: Code
    message: str

    def __str__(self) -> str:
        return f"{self.tool}: {self.message}" if self.tool else self.message


def lint(specs: Iterable[ToolSpec]) -> list[Finding]:
    """What makes these tools hard for a model to choose among, in the tools' order: each
    tool's own findings, then the pairs', then the set's."""
    tools = list(specs)
    found = [f for spec in tools for f in _one(spec)]
    for a, b in combinations(tools, 2):
        found.extend(_pair(a, b))
    if len(tools) > MOST_TOOLS:
        found.append(
            Finding(
                None,
                "too_many_tools",
                f"{len(tools)} tools: the OpenAI API takes at most {MOST_TOOLS} in one request — "
                "narrow them (tool hints, a tool selector) or split the agent",
            )
        )
    elif len(tools) >= MANY_TOOLS:
        found.append(
            Finding(
                None,
                "many_tools",
                f"{len(tools)} tools: models choose less reliably among many — narrow what each "
                "call is offered (tool hints, a tool selector)",
            )
        )
    return found


def _one(spec: ToolSpec) -> list[Finding]:
    found: list[Finding] = []
    name = spec.name
    if not NAME.match(name):
        found.append(
            Finding(
                name,
                "invalid_name",
                "a tool name has only letters, digits, _ and - and at most 64 characters",
            )
        )
    words = _words(spec.description)
    if not words:
        found.append(
            Finding(name, "no_description", "no description: say what it is for, and when")
        )
    elif len(spec.description.split()) < MIN_DESCRIPTION_WORDS:
        found.append(
            Finding(
                name,
                "short_description",
                f"the description is {len(spec.description.split())} words: say what it is "
                "for, and when to use it rather than a similar tool",
            )
        )
    properties = (spec.input_schema or {}).get("properties") or {}
    undescribed = [p for p, schema in properties.items() if not (schema or {}).get("description")]
    if undescribed:
        found.append(
            Finding(
                name,
                "undescribed_parameter",
                f"parameters with no description: {', '.join(undescribed)} — say what each "
                "takes (its format, an example)",
            )
        )
    return found


def _pair(a: ToolSpec, b: ToolSpec) -> list[Finding]:
    if a.name != b.name and _key(a.name) == _key(b.name):
        return [
            Finding(
                b.name,
                "similar_names",
                f"named like {a.name} (they differ in case or separators only)",
            )
        ]
    if a.description.strip() and a.description.strip() == b.description.strip():
        return [
            Finding(
                b.name,
                "same_description",
                f"described as {a.name} is: say what each is for, and when to use which",
            )
        ]
    one, other = _words(a.description), _words(b.description)
    if one and other and len(one & other) / len(one | other) >= OVERLAP:
        return [
            Finding(
                b.name,
                "overlapping_descriptions",
                f"described much as {a.name} is: say in each when to use it rather than the other",
            )
        ]
    return []


def _key(name: str) -> str:
    return re.sub(r"[-_]", "", name).casefold()


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in _STOP}


__all__ = ["Finding", "lint"]
