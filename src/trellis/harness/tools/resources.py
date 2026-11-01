"""A server's resources (MCP resources) as one read tool, ``<server>-read_resource``: offered
only when the server says it has resources (its ``resources`` capability) and lists at least
one, its description the map of what there is (the first :data:`MAX_LISTED`, a template's
placeholders to fill in), its call a resource's text by its URI, or every resource listed when
it names none.

What it reads is a tool result like any other: governed (a read), journaled, and — over what
the model reads as it is — kept in the run, previewed and paged with ``read_file``, its rows
queried with ``query_result`` (``tools.results``): no store of its own. Its data is live (the
default freshness, ``tools.base``): replayed on a later turn it is a dated snapshot, and the
server's ``lastModified`` is said with each resource. A listing is rows (one
per resource), so a long one is kept and queried the same way."""

from __future__ import annotations

import re
from typing import Any, Final

#: The read tool's name after its server's (``<server>-read_resource``).
READ: Final = "read_resource"
#: The most resources (and templates) the tool's description lists; the rest are listed by a
#: call that names no URI.
MAX_LISTED: Final = 20
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "uri": {
            "type": "string",
            "description": (
                "The resource's URI as listed, a template's {placeholders} filled in; leave it "
                "out to list every resource."
            ),
        }
    },
    "additionalProperties": False,
}
#: What the description says when the server has more than it lists.
MORE: Final = "- … and more: call it without a uri to list them all"
_PLACEHOLDER: Final = re.compile(r"\{[^}]*\}")


def described(server: str, resources: list[Any], templates: list[Any], *, more: bool) -> str:
    """The read tool's description: what it does, then one line per resource and template
    (at most :data:`MAX_LISTED`), and :data:`MORE` when the server has others."""
    lines = [
        f"Reads a resource of the {server} server by its URI: its text (a binary one is "
        "described, not read). Without a uri, lists every resource. Its resources:"
    ]
    for resource in resources:
        lines.append(f"- {resource.uri}: {_about(resource, [])}")
    for template in templates:
        fill = _PLACEHOLDER.findall(template.uri_template)
        lines.append(f"- {template.uri_template}: {_about(template, fill)}")
    shown = lines[: MAX_LISTED + 1]
    if more or len(lines) > len(shown):
        shown.append(MORE)
    return "\n".join(shown)


def _about(item: Any, fill: list[str]) -> str:
    """A resource's or template's name, description, type and when it last changed (the
    server's ``lastModified``), as one line says them."""
    said = ", ".join(p for p in (item.title or item.name, item.description) if p)
    changed = modified(item)
    kind = "; ".join(
        p
        for p in (
            item.mime_type,
            f"modified {changed}" if changed else "",
            f"fill in {' '.join(fill)}" if fill else "",
        )
        if p
    )
    return f"{said} ({kind})" if kind else said


def modified(item: Any) -> str | None:
    """When the server says a resource last changed (its annotations' ``lastModified``)."""
    return getattr(getattr(item, "annotations", None), "last_modified", None)


def row(item: Any) -> dict[str, Any]:
    """A resource or template as a listing's row (what it does not say left out)."""
    keys = ("uri", "uri_template", "name", "title", "description", "mime_type", "size")
    found = {key: getattr(item, key, None) for key in keys}
    found["last_modified"] = modified(item)
    return {key: value for key, value in found.items() if value not in (None, "")}


def text_of(result: Any) -> str:
    """A read's contents as the model reads them: each text as it is, a binary one described
    (its URI, type and size)."""
    parts = []
    for content in result.contents:
        text = getattr(content, "text", None)
        if text is not None:
            parts.append(text)
            continue
        blob = content.blob
        size = len(blob) * 3 // 4 - blob[-2:].count("=")
        parts.append(f"[{content.uri}: {content.mime_type or 'binary'}, {size} bytes, not text]")
    return "\n\n".join(parts)
