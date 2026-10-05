"""A fake Bifrost gateway behind the real bifrost-sdk (an ``httpx.MockTransport``): the routes
the harness calls, answered in the shapes the running gateway answers with.

* ``/v1/chat/completions`` — each request recorded (:attr:`FakeGateway.completions`) and
  answered by :attr:`FakeGateway.chat`, a ``ScriptedChat`` (``tests/support/models.py``);
* ``/api/prompt-repo/prompts`` and ``/api/prompt-repo/prompts/{id}/versions`` —
  :attr:`FakeGateway.prompts`, by name: each a list of committed versions (their messages);
* ``/api/skills``, ``/api/skills/{id}``, ``/api/skills/serve/{name}/files/{path}`` —
  :attr:`FakeGateway.skills`, by name: the versions (description, body, files) and the one
  served;
* ``/api/mcp/clients`` — :attr:`FakeGateway.auto`, each client's ``tools_to_auto_execute``;
* ``/mcp`` and ``/mcp/{slug}`` — :attr:`FakeGateway.bundles`, the tools listed (``""``: the
  key's whole reach), and a ``tools/call`` answered with the tool's name and arguments.

A request to a path ``down`` starts with fails as a gateway that cannot be reached (``"/"``:
every request).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

import httpx
from bifrost_sdk import Bifrost
from bifrost_sdk.admin import Admin

from tests.support.models import ScriptedChat, Turn
from trellis import Harness
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.prompts import PromptSource, PromptSources
from trellis.harness.skills import SkillSource, SkillSources

URL = "http://gw.test/v1"


@dataclass
class SkillVersions:
    """One skill: each version's description, body and files (path -> text), and the one
    served."""

    versions: dict[str, tuple[str, str, dict[str, str]]]
    served: str


@dataclass
class FakeGateway:
    turns: list[Turn] = field(default_factory=list)
    prompts: dict[str, list[list[dict[str, Any]]]] = field(default_factory=dict)
    skills: dict[str, SkillVersions] = field(default_factory=dict)
    auto: dict[str, list[str]] = field(default_factory=dict)
    bundles: dict[str, list[str]] = field(default_factory=dict)
    down: tuple[str, ...] = ()
    completions: list[httpx.Request] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = ScriptedChat(self.turns)

    def gateway(self) -> Gateway:
        transport = httpx.MockTransport(self.handle)
        http = httpx.AsyncClient(transport=transport, base_url=URL)
        api = httpx.AsyncClient(transport=transport, base_url=URL.removesuffix("/v1"))
        bifrost = Bifrost(URL, api_key="vk", client=http, admin_client=api, max_retries=0)
        return Gateway(URL, "vk", client=bifrost, admin=Admin(URL, client=api))

    async def attach(
        self,
        h: Harness,
        *,
        prompts: Sequence[PromptSource] = (),
        skills: Sequence[SkillSource] = (),
    ) -> Harness:
        """``h`` with this fake as its gateway: the client, and the gateway's prompt and skill
        sources (after ``prompts`` and ``skills``, as ``Harness(prompts=, skills=)`` orders
        them)."""
        if h.gateway is not None:
            await h.gateway.aclose()
        h.gateway = gateway = self.gateway()
        h.prompts = PromptSources.of(h.settings, gateway=gateway, given=prompts)
        h.skills = SkillSources.of(h.settings, gateway=gateway, given=skills)
        h.evals.prompts = h.prompts
        return h

    def asked(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == path)

    # ------------------------------------------------------------------ the routes
    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.startswith(self.down or "\0"):
            raise httpx.ConnectError("the gateway is down", request=request)
        path = unquote(request.url.path)
        if path == "/v1/chat/completions":
            self.completions.append(request)
            body = json.loads(request.content)
            return httpx.Response(200, json=await self.chat.complete(**body))
        if path.startswith("/mcp"):
            return self._rpc(
                path.removeprefix("/mcp").removeprefix("/"), json.loads(request.content)
            )
        if path == "/api/prompt-repo/prompts":
            return httpx.Response(200, json={"prompts": [self._prompt(n) for n in self.prompts]})
        if path.startswith("/api/prompt-repo/prompts/p-"):
            name = path.removeprefix("/api/prompt-repo/prompts/p-").removesuffix("/versions")
            rows = [self._version(name, n) for n in range(1, len(self.prompts[name]) + 1)]
            return httpx.Response(200, json={"versions": rows})
        if path == "/api/mcp/clients":
            return httpx.Response(200, json={"clients": self._clients(), "count": len(self.auto)})
        return self._skills(path, request)

    def _skills(self, path: str, request: httpx.Request) -> httpx.Response:
        if path.startswith("/api/skills/serve/"):
            name, _, file = path.removeprefix("/api/skills/serve/").partition("/files/")
            skill = self.skills[name]
            return httpx.Response(200, content=skill.versions[skill.served][2][file].encode())
        if path == "/api/skills":
            search = request.url.params.get("search", "")
            found = [self._skill(n) | {"skill_md_body": ""} for n in self.skills if search in n]
            return httpx.Response(200, json={"skills": found})
        name = path.removeprefix("/api/skills/").removeprefix("s-")
        version = request.url.params.get("version")
        if name not in self.skills or version not in (None, *self.skills[name].versions):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json={"skill": self._skill(name, version)})

    def _rpc(self, slug: str, message: dict[str, Any]) -> httpx.Response:
        if message["method"] == "tools/list":
            names = self.bundles.get(slug, [])
            result: dict[str, Any] = {"tools": [{"name": n, "inputSchema": {}} for n in names]}
        else:
            text = json.dumps(message["params"])
            result = {"content": [{"type": "text", "text": text}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})

    def _prompt(self, name: str) -> dict[str, Any]:
        versions = self.prompts[name]
        latest = self._version(name, len(versions)) if versions else None
        return {"id": f"p-{name}", "name": name, "latest_version": latest}

    def _version(self, name: str, number: int) -> dict[str, Any]:
        rows = [
            {"order_index": i, "message": m} for i, m in enumerate(self.prompts[name][number - 1])
        ]
        return {
            "id": number,
            "prompt_id": f"p-{name}",
            "version_number": number,
            "messages": rows,
            "provider": "local",
            "model": "small",
        }

    def _skill(self, name: str, version: str | None = None) -> dict[str, Any]:
        skill = self.skills[name]
        chosen = version or skill.served
        description, body, files = skill.versions[chosen]
        return {
            "id": f"s-{name}",
            "name": name,
            "description": description,
            "skill_md_body": body,
            "latest_version": chosen,
            "files": [
                {"path": p, "source_type": "text", "file_size_bytes": len(t)}
                for p, t in files.items()
            ],
        }

    def _clients(self) -> list[dict[str, Any]]:
        return [
            {
                "config": {
                    "client_id": name,
                    "name": name,
                    "connection_type": "http",
                    "connection_string": f"http://{name}.test/mcp",
                    "tools_to_execute": ["*"],
                    "tools_to_auto_execute": names,
                },
                "tools": [],
                "state": "healthy",
            }
            for name, names in self.auto.items()
        ]


def completion(request: httpx.Request) -> dict[str, Any]:
    """A recorded completion's body."""
    return json.loads(request.content)
