"""Tool chains: which tool's output feeds which tool's input, read from the schemas the tools
publish — a function's return annotation (``tool(fn)``), an MCP tool's ``outputSchema``, set
against the other tools' input schemas. Data, not hand-written rules: a link is a field of
one tool's output that a declared argument of another takes —

* by name: ``get_stock``'s ``supplier_id`` is ``supplier_lead_time``'s ``supplier_id`` (any
  depth of the output, records in a list included: ``items[].sku_id``), of a compatible type
  where both declare one; a generic name (``name``, ``status``, ``items``...) links nothing,
  every catalog has those;
* an ``id`` by its noun: the ``id`` of ``find_sku``'s output (or of its ``sku`` record) is a
  ``sku_id``, the way MCP servers answer.

The links are said where every framework's model reads how to use a tool, its description
(:func:`chained`: ``Chains: find_sku → sku_id → get_stock.``), on both ends — so a model that
starts cold, with nothing learned, knows that one call's answer is the next one's argument
and in what order to call them; and the tool hints that narrow what a run is offered keep a
candidate's chain whole (:func:`related`), so the next tool of it is still there. A tool that
publishes no output schema (an untyped function, a gateway's MCP tool) links nothing: nothing
is guessed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from trellis.contracts import ToolSpec

#: Field names too generic to say what feeds what: every catalog has them.
GENERIC: Final = frozenset(
    {
        "id",
        "ids",
        "name",
        "title",
        "description",
        "type",
        "kind",
        "status",
        "state",
        "value",
        "data",
        "result",
        "results",
        "text",
        "message",
        "content",
        "count",
        "total",
        "limit",
        "offset",
        "page",
        "cursor",
        "query",
        "q",
        "error",
        "ok",
        "items",
        "url",
        "created_at",
        "updated_at",
    }
)
#: How deep an output schema is read (records in a list of records...).
DEPTH: Final = 4
#: The most links one tool's description names: what it says stays short.
SAID_MAX: Final = 4
#: How a tool's description starts saying its links.
SAID: Final = "Chains:"
_WORD = re.compile(r"[a-z0-9]+")
#: JSON types one is given where the other is declared.
_FITS: Final = {("integer", "number")}


@dataclass(frozen=True, order=True, slots=True)
class Link:
    """``source``'s output ``field`` (a dotted path, ``[]`` a list's records) is ``target``'s
    ``argument``."""

    source: str
    field: str
    target: str
    argument: str

    def said(self) -> str:
        leaf = self.field.rsplit(".", 1)[-1].removesuffix("[]")
        shown = leaf if leaf == self.argument else f"{leaf} as {self.argument}"
        return f"{self.source} → {shown} → {self.target}"


def _norm(name: str) -> str:
    return "".join(_WORD.findall(name.casefold()))


def _type(schema: Mapping[str, Any]) -> str | None:
    found = schema.get("type")
    if isinstance(found, list):  # ["string", "null"]: the one that is not null
        found = next((t for t in found if t != "null"), None)
    return found if isinstance(found, str) else None


def _resolved(schema: Any, root: Mapping[str, Any]) -> Mapping[str, Any]:
    """A schema with its local ``$ref`` (``#/$defs/Stock``) followed, and a nullable one
    (``anyOf`` of it and null) as itself."""
    seen = 0
    while isinstance(schema, Mapping) and seen < DEPTH:
        seen += 1
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            target: Any = root
            for part in ref[2:].split("/"):
                target = target.get(part) if isinstance(target, Mapping) else None
            schema = target
            continue
        options = [o for o in schema.get("anyOf") or [] if _type(o or {}) != "null"]
        if "type" not in schema and len(options) == 1:
            schema = options[0]
            continue
        break
    return schema if isinstance(schema, Mapping) else {}


def fields(schema: Mapping[str, Any] | None) -> Iterator[tuple[str, str, str | None]]:
    """The fields of an output schema: (path, the record it is in — its parent's name, or ""
    at the top —, its JSON type)."""
    if not schema:
        return

    def walk(
        node: Any, path: str, parent: str, depth: int
    ) -> Iterator[tuple[str, str, str | None]]:
        node = _resolved(node, schema)
        if depth > DEPTH:
            return
        if _type(node) == "array":
            yield from walk(node.get("items"), f"{path}[]", parent, depth + 1)
            return
        for name, declared in (node.get("properties") or {}).items():
            child = _resolved(declared, schema)
            at = f"{path}.{name}" if path else name
            yield at, parent, _type(child)
            if _type(child) in ("object", "array"):
                yield from walk(child, at, name, depth + 1)

    yield from walk(schema, "", "", 0)


def _fits(given: str | None, wanted: str | None) -> bool:
    return given is None or wanted is None or given == wanted or (given, wanted) in _FITS


def _nouns(tool: str) -> set[str]:
    """The words of a tool's name, its MCP server's prefix left out (``crm-find_sku``)."""
    return set(_WORD.findall(tool.rsplit("-", 1)[-1].casefold()))


def links(specs: Sequence[ToolSpec]) -> list[Link]:
    """Every link between ``specs`` (see the module), sorted."""
    arguments: dict[str, list[tuple[str, str, str | None]]] = defaultdict(list)
    for spec in specs:
        for name, declared in ((spec.input_schema or {}).get("properties") or {}).items():
            schema = _resolved(declared, spec.input_schema or {})
            arguments[_norm(name)].append((spec.name, name, _type(schema)))
    found: set[Link] = set()
    for spec in specs:
        for path, parent, kind in fields(spec.output_schema):
            leaf = path.rsplit(".", 1)[-1].removesuffix("[]")
            if leaf.casefold() == "id":
                nouns = {parent.casefold()} if parent else _nouns(spec.name)
                wanted = [_norm(f"{noun}_id") for noun in nouns if noun]
            elif leaf.casefold() in GENERIC:
                continue
            else:
                wanted = [_norm(leaf)]
            for key in wanted:
                for target, argument, declared in arguments.get(key, ()):
                    if target != spec.name and _fits(kind, declared):
                        found.add(Link(spec.name, path, target, argument))
    return sorted(found)


def chained(specs: Sequence[ToolSpec], found: Sequence[Link] | None = None) -> list[ToolSpec]:
    """``specs``, each that a link starts or ends at with the links in its description (at
    most :data:`SAID_MAX`, those it ends at first: what to call before it); the others as
    they are."""
    found = links(specs) if found is None else found
    by_tool: dict[str, list[Link]] = defaultdict(list)
    for link in found:
        by_tool[link.target].append(link)
    for link in found:
        by_tool[link.source].append(link)
    out: list[ToolSpec] = []
    for spec in specs:
        said = list(dict.fromkeys(link.said() for link in by_tool.get(spec.name, ())))[:SAID_MAX]
        line = f"{SAID} {'; '.join(said)}." if said else ""
        if not line or line in spec.description:
            out.append(spec)
            continue
        described = f"{spec.description.rstrip()}\n{line}".strip()
        out.append(spec.model_copy(update={"description": described}))
    return out


def related(
    names: Collection[str], found: Sequence[Link], required: Mapping[str, Collection[str]]
) -> set[str]:
    """The tools a chain through ``names`` needs besides them: those their outputs feed, on and
    on, and those whose output gives a ``required`` argument (by tool) of one of them, back and
    back."""
    onward: dict[str, set[str]] = defaultdict(set)
    back: dict[str, set[str]] = defaultdict(set)
    for link in found:
        onward[link.source].add(link.target)
        if link.argument in required.get(link.target, ()):
            back[link.target].add(link.source)
    out: set[str] = set()
    for edges in (onward, back):
        seen, todo = set(names), list(names)
        while todo:
            for name in edges.get(todo.pop(), ()):
                if name not in seen:
                    seen.add(name)
                    todo.append(name)
        out |= seen
    return out - set(names)
