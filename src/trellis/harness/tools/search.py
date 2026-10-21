"""Tool search without a model call: the tools whose definitions — name, description, and
each parameter's name, description and allowed values — best match a text, ranked by BM25
(Robertson's, ``k1`` 1.2, ``b`` 0.75) over word stems. It narrows a large catalogue to the
tools a task is about (the ``tool_search`` feature) where neither the provider's own tool
search nor the memory service's tool hints are there to do it."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Final

from trellis.contracts import ToolSpec
from trellis.harness.tools.base import Tool

#: A catalogue of this many tools or more is narrowed (the OpenAI guide's "fewer than 20").
TOOL_SEARCH_MIN: Final = 20
#: How many tools a search offers.
TOP_K: Final = 10
K1: Final = 1.2
B: Final = 0.75
_WORD: Final = re.compile(r"[a-z0-9]+")
_STOP: Final = frozenset(
    re.findall(
        r"\w+",
        "a an and are as at be by can do does for from has have how i in is it its me my of on "
        "or our should so such that the their them then there these this to us was we what when "
        "where which who will with you your please tell show give get list",
    )
)


def stem(word: str) -> str:
    """A crude English stem: plurals and ``-ing`` folded (``tickets``/``ticket``,
    ``policies``/``policy``, ``booking``/``book``)."""
    for suffix, keep in (("ies", "y"), ("ing", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + keep
    return word


def terms(text: str) -> list[str]:
    """The text's searchable terms: lower-cased words (``snake_case`` and ``kebab-case`` names
    split), stop words dropped, stemmed."""
    return [stem(w) for w in _WORD.findall(text.lower().replace("_", " ")) if w not in _STOP]


def described(spec: ToolSpec) -> str:
    """Everything a tool's definition says, as one text: its name, its description, and each
    parameter's name, description and allowed values."""
    parts = [spec.name, spec.description]
    for name, schema in ((spec.input_schema or {}).get("properties") or {}).items():
        given = schema if isinstance(schema, dict) else {}  # a JSON schema may be `true`
        parts.extend([name, str(given.get("description") or "")])
        parts.extend(str(v) for v in given.get("enum") or ())
    return " ".join(parts)


def ranked(query: str, specs: Sequence[ToolSpec]) -> list[tuple[float, str]]:
    """The tools that match ``query`` at all, best first: ``(score, name)``."""
    docs = {spec.name: Counter(terms(described(spec))) for spec in specs}
    if not docs:
        return []
    average = sum(sum(d.values()) for d in docs.values()) / len(docs)
    holding = Counter(term for d in docs.values() for term in d)
    wanted = Counter(terms(query))
    scores = []
    for name, doc in docs.items():
        length = sum(doc.values())
        score = 0.0
        for term, asked in wanted.items():
            found = doc.get(term, 0)
            if not found:
                continue
            idf = math.log(1 + (len(docs) - holding[term] + 0.5) / (holding[term] + 0.5))
            norm = found * (K1 + 1) / (found + K1 * (1 - B + B * length / (average or 1)))
            score += idf * norm * asked
        if score > 0:
            scores.append((score, name))
    return sorted(scores, key=lambda s: (-s[0], s[1]))


def search(query: str, specs: Sequence[ToolSpec], k: int = TOP_K) -> list[str]:
    """The ``k`` tools that best match ``query`` (fewer when fewer match at all)."""
    return [name for _, name in ranked(query, specs)[:k]]


def named(text: str, names: Iterable[str]) -> list[str]:
    """The tool names ``text`` mentions as they are (a user naming the tool to use)."""
    found = set(re.findall(r"[A-Za-z0-9_.-]+", text))
    return [n for n in names if n in found]


#: The tools a search always keeps: Code Mode's meta-tools and the skills' tools.
KEPT: Final = frozenset({"code_mode", "skills"})


def offered_for(text: str, tools: Sequence[Tool], k: int = TOP_K) -> list[str]:
    """What a search offers for ``text`` among ``tools``: the ``k`` best matches, every tool
    the text names, and the tools of :data:`KEPT` features."""
    names = [t.name for t in tools]
    kept = [t.name for t in tools if t.feature in KEPT]
    found = search(text, [t.spec for t in tools if t.feature not in KEPT], k)
    return list(dict.fromkeys([*found, *named(text, names), *kept]))


def definition_tokens(specs: Iterable[ToolSpec]) -> int:
    """About how many tokens the definitions take in a request (4 characters a token)."""
    size = sum(
        len(json.dumps({"n": s.name, "d": s.description, "p": s.input_schema or {}})) for s in specs
    )
    return size // 4


__all__ = [
    "TOOL_SEARCH_MIN",
    "TOP_K",
    "definition_tokens",
    "named",
    "offered_for",
    "ranked",
    "search",
    "terms",
]
