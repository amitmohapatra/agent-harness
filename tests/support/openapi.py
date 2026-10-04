"""A sibling service's committed OpenAPI document, as a checker of what the harness sends and
what a test double answers: the operation must exist, a JSON request body must match its
schema, and a response's status must be documented and its JSON body match the schema of that
status and media type. OpenAPI 3.1 schemas are JSON Schema 2020-12, so ``jsonschema`` checks
them, the document's own ``#/components/...`` references resolving within it."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
from jsonschema import Draft202012Validator

ROOT: Final = Path(__file__).resolve().parents[2]
#: Where CI checks the siblings out (next to this repository), and where their documents are
#: (``TRELLIS_MEMORY_OPENAPI`` / ``TRELLIS_RUNS_OPENAPI`` point elsewhere).
MEMORY_OPENAPI: Final = Path(
    os.environ.get("TRELLIS_MEMORY_OPENAPI")
    or ROOT.parent / "agent-memory-service" / "docs" / "openapi.json"
)
RUNS_OPENAPI: Final = Path(
    os.environ.get("TRELLIS_RUNS_OPENAPI") or ROOT.parent / "agent-runs" / "docs" / "openapi.json"
)
#: Answers no document lists: a conditional GET's ``304`` carries no body.
UNDOCUMENTED_OK: Final = frozenset({304})


@dataclass
class OpenAPI:
    document: dict[str, Any]
    #: whether a request's query parameters must be documented ones: checked for what the
    #: harness's own client sends (agent-runs), not for the lineage the memory SDK adds to
    #: every body-less call (the service ignores what a route does not declare)
    strict_query: bool = False
    #: (method, compiled path template, template) for every operation
    operations: list[tuple[str, re.Pattern[str], str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, *, strict_query: bool = False) -> OpenAPI:
        api = cls(json.loads(path.read_text()), strict_query)
        for template, item in api.document["paths"].items():
            pattern = re.compile("^" + re.sub(r"\{[^/]+\}", "[^/]+", template) + "$")
            for method in item:
                if method in ("get", "put", "post", "patch", "delete"):
                    api.operations.append((method, pattern, template))
        # a literal path wins over a templated one (/v1/runs/claim over /v1/runs/{id})
        api.operations.sort(key=lambda op: op[2].count("{"))
        return api

    def operation(self, method: str, path: str) -> dict[str, Any] | None:
        for verb, pattern, template in self.operations:
            if verb == method.lower() and pattern.match(path):
                return self.document["paths"][template][verb]
        return None

    def request(self, request: httpx.Request) -> list[str]:
        """What is wrong with a request the harness sent (empty: nothing)."""
        where = f"{request.method} {request.url.path}"
        op = self.operation(request.method, request.url.path)
        if op is None:
            return [f"{where}: no such operation"]
        known = {p["name"] for p in op.get("parameters", []) if p.get("in") == "query"}
        wrong = [
            f"{where}: no query parameter {name!r}"
            for name in request.url.params
            if self.strict_query and name not in known
        ]
        media = request.headers.get("content-type", "").split(";")[0]
        content = (op.get("requestBody") or {}).get("content", {})
        if not request.content or media != "application/json" or "*/*" in content:
            return wrong
        if media not in content:
            return [*wrong, f"{where}: takes no {media} body"]
        body = json.loads(request.content)
        return wrong + self._check(where, content[media].get("schema", {}), body)

    def response(self, request: httpx.Request, response: httpx.Response) -> list[str]:
        """What is wrong with an answer to ``request`` (empty: nothing)."""
        where = f"{request.method} {request.url.path} -> {response.status_code}"
        op = self.operation(request.method, request.url.path)
        if op is None:
            return [f"{where}: no such operation"]
        if response.status_code in UNDOCUMENTED_OK:
            return []
        documented = op.get("responses", {})
        spec = documented.get(str(response.status_code)) or documented.get("default")
        if spec is None:
            return [f"{where}: status not documented ({sorted(documented)})"]
        media = response.headers.get("content-type", "").split(";")[0]
        content = spec.get("content") or {}
        if not response.content or media not in ("application/json", "application/problem+json"):
            return []
        if media not in content:
            return [f"{where}: answers {media}, documented {sorted(content)}"]
        return self._check(where, content[media].get("schema", {}), response.json())

    def _check(self, where: str, schema: dict[str, Any], instance: Any) -> list[str]:
        rooted = {**schema, "components": self.document.get("components", {})}
        validator = Draft202012Validator(rooted)
        return [
            f"{where}: {'/'.join(map(str, e.absolute_path)) or '(body)'}: {e.message[:300]}"
            for e in validator.iter_errors(instance)
        ]
