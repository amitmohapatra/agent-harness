"""``trellis doctor`` (``python -m trellis.harness.doctor``): the effective settings and where
each came from — the environment variable, derived, detected, or the default — secrets
redacted; which route serves model names — its name, auth and profile, the endpoint checked
against the route ``LLM_GATEWAY`` declares — and what it cannot do
(``trellis.harness.gateways``); whether each service is reachable and takes its key — for the
memory service also the key's role, the endpoint each agent told it, its readiness
(``/health/ready``) and its version (``/version``); the approval rules accepted from the
memory service's learned suggestions — learned and accepted per agent: those of each agent the
key may act as, and of each ``--agent`` named (governance accepts them automatically, but for a
tool at the highest risk); whether the spool can be written, and the memory writes that failed
and wait in it ("records failing"); whether the Docker daemon answers; and what to set. It
changes nothing: it suggests.

Exit code ``0`` when every check passes, ``1`` otherwise."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import httpx

from trellis.contracts import ConfigurationError
from trellis.harness.gateways import (
    CAPABILITIES,
    Capability,
    ModelGateway,
    OpenAICompatible,
    model_gateway,
    provider_keys,
)
from trellis.harness.settings import Settings, service_version, sourced

#: The settings whose values are secrets: shown as set, never as their value.
SECRETS: Final = frozenset(
    {
        "llm_api_key",
        "api_key",
        "judge_virtual_key",
        "langfuse_secret_key",
        "langfuse_public_key",
        "otlp_headers",
    }
)
#: How long each check waits for its service, in seconds.
CHECK_SECONDS: Final = 3.0
#: Where the memory service lists the approval suggestions it learned.
SUGGESTIONS_PATH: Final = "/v1/tools/approval-suggestions"
#: Where it says what the key is, whether it is ready, which version it is, and the model
#: endpoint an agent told it.
KEY_PATH: Final = "/v1/keys/self"
READY_PATH: Final = "/health/ready"
VERSION_PATH: Final = "/version"
MODEL_KEY_PATH: Final = "/v1/agents/model-key"
#: Where a gateway with MCP (``Capability.MCP``: Bifrost) answers MCP's JSON-RPC, under its
#: root: the request ``doctor`` checks a declared route with.
MCP_PATH: Final = "/mcp"
MCP_LIST: Final = {"jsonrpc": "2.0", "id": "trellis-doctor", "method": "tools/list"}
#: The memory service's readiness answers that still serve (``degraded``: an optional
#: dependency is down).
READY: Final = frozenset({"ready", "degraded"})
#: The check of the memory writes that failed and wait in the spool.
RECORDS: Final = "records failing"
#: How ``may_act_as`` names an agent a key may act as.
AGENT_PRINCIPAL: Final = "agent:"


class KeyRole(StrEnum):
    """A memory key's role: ``service`` is what the harness acts with (for users and agents);
    ``platform`` names the tenant per call; ``admin`` administers one tenant."""

    SERVICE = "service"
    PLATFORM = "platform"
    ADMIN = "admin"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    said: str


def shown(field: str, value: Any) -> str:
    """A setting as ``doctor`` prints it: a secret as ``set (…last 4)``, never in full."""
    if value is None:
        return "unset"
    if field in SECRETS:
        text = str(value)
        return f"set (...{text[-4:]})" if isinstance(value, str) and len(text) > 8 else "set"
    return str(value)


def effective(environ: Mapping[str, str] | None = None) -> list[tuple[str, str, str]]:
    """Each setting: its name, its value as shown, and where it came from."""
    found = sourced(environ)
    settings = Settings.from_env(environ)
    return [
        (
            field,
            shown(field, getattr(settings, field)),
            found[field][1] if field in found else "default",
        )
        for field in Settings.model_fields
    ]


def gateway_said(settings: Settings, environ: Mapping[str, str]) -> list[str]:
    """How model names are routed — the route, its auth, where each model's profile comes
    from —, the provider keys it leaves unused (an endpoint wins over them) and what is off on
    the route."""
    try:
        gateway = model_gateway(settings)
    except ConfigurationError as exc:
        return [f"misconfigured: {exc}"]
    keys = provider_keys(environ)
    lines = [f"route: {gateway.name} ({gateway.title}) at {gateway.where}"]
    if gateway.url is not None:
        key = ", with LLM_API_KEY" if gateway.key else ""
        lines.append(f"auth: {gateway.auth}{key}")
        if keys:
            lines.append(
                f"note: {', '.join(keys)} set, not used for model calls: the endpoint wins "
                "over a provider key"
            )
    else:
        said = ", ".join(keys) if keys else "none set"
        lines.append(f"provider keys: each provider's own SDK reads its own ({said})")
    lines += gateway.notes()
    lines.append(
        "profile: each model's LangChain profile (reasoning, temperature, output tokens, "
        "tools); the memory service is told it per agent with the endpoint"
        if gateway.endpoint(None) is not None
        else "profile: each model's LangChain profile; the memory service uses its own settings"
    )
    off = sorted(set(CAPABILITIES) - gateway.capabilities)
    lines += [f"off: {CAPABILITIES[c]}" for c in off]
    return lines


async def checks(
    settings: Settings, *, agents: Sequence[str] = (), docker: bool = False
) -> list[Check]:
    """Each service the settings name, reached with its key; the approval rules learned for
    ``agents`` (and the agents the key may act as); the spool; the Docker daemon (``docker``:
    ``DOCKER_HOST`` is set, where ``sandbox()`` makes its sandboxes)."""
    async with httpx.AsyncClient(timeout=CHECK_SECONDS) as client:
        gateway, (memory, key), runs = await asyncio.gather(
            _gateway(client, settings),
            _memory(client, settings),
            _runs(client, settings),
        )
        rules = await _approval_rules(client, settings, [*agents, *_acting_as(key)])
    done = [c for c in (gateway, *memory, rules, runs) if c is not None]
    if settings.spool_dir:
        done.append(_spool(settings.spool_dir))
        done.append(_records(settings.spool_dir))
    if docker:
        done.append(await _docker())
    return done


def suggestions(settings: Settings, environ: Mapping[str, str]) -> list[str]:
    """What to set, from what is and is not set (never applied)."""
    said: list[str] = []
    route = _route(settings)
    if route is not None and route.has(Capability.FALLBACKS):
        said.append(
            "model retries are the gateway's: set the provider's network_config.max_retries in "
            "Bifrost (its default is 0), and the virtual key's fallbacks"
        )
    if not settings.spool_dir:
        said.append(
            "set TRELLIS_SPOOL_DIR to a lasting folder, or memory writes the service "
            "could not take are lost at a restart"
        )
    if not service_version(environ):
        said.append(
            "bake the version in at build time: OTEL_RESOURCE_ATTRIBUTES=service.version=..."
        )
    if settings.memory_url and not settings.runs_url:
        said.append("runs are kept in this process: set RUNS_URL for durable runs and workers")
    return said


def _route(settings: Settings) -> ModelGateway | None:
    """The route ``settings`` name; ``None`` when they fit none (the gateway check says why)."""
    try:
        return model_gateway(settings)
    except ConfigurationError:
        return None


async def _gateway(client: httpx.AsyncClient, settings: Settings) -> Check | None:
    """The endpoint the route sends model names to, asked for its models with the route's
    auth, then checked against the route ``LLM_GATEWAY`` declares: a route with MCP
    (Bifrost) must answer MCP's ``tools/list`` at the endpoint's root, and an endpoint that
    does under another route is said to have more to give. A provider's own SDK is not asked:
    its key is its own."""
    try:
        gateway: ModelGateway = model_gateway(settings)
    except ConfigurationError as exc:
        return Check("gateway", False, str(exc))
    if not isinstance(gateway, OpenAICompatible):
        return None
    url = f"{gateway.url.rstrip('/')}/models"
    reached, answer = await _asked(client, "gateway", url, {}, auth=gateway.http_auth)
    if answer is None:
        return reached
    mcp = await _serves_mcp(client, gateway)
    if gateway.has(Capability.MCP) and not mcp:
        return Check(
            "gateway",
            False,
            f"LLM_GATEWAY={gateway.name}, but {gateway.origin}{MCP_PATH} answers no MCP: the "
            "endpoint is not Bifrost (set LLM_GATEWAY=openai-compatible, or point LLM_BASE_URL "
            "at Bifrost's /v1)",
        )
    if mcp and not gateway.has(Capability.MCP):
        return Check(
            "gateway",
            True,
            f"{url} answered; {gateway.origin} answers MCP as Bifrost does: LLM_GATEWAY=bifrost "
            "turns on its MCP tools, stored prompts, skills and cost",
        )
    return Check("gateway", True, f"{url} answered as {gateway.name} ({gateway.title})")


async def _serves_mcp(client: httpx.AsyncClient, gateway: OpenAICompatible) -> bool:
    """Whether the endpoint's root answers MCP's ``tools/list`` (:data:`MCP_PATH`)."""
    try:
        response = await client.post(
            f"{gateway.origin}{MCP_PATH}", json=MCP_LIST, auth=gateway.http_auth
        )
    except httpx.HTTPError:
        return False
    return response.is_success


async def _memory(
    client: httpx.AsyncClient, s: Settings
) -> tuple[list[Check], Mapping[str, Any] | None]:
    """The memory service: reached with the key (``/v1/keys/self``) and what the key is (its
    role, the agents' model key), whether it is ready and which version it is; and what it said
    about the key (``None``: nothing)."""
    if not s.memory_url:
        return [], None
    base = s.memory_url.rstrip("/")
    headers = {"Authorization": f"Bearer {s.api_key or ''}"}
    reached, answer = await _asked(client, "memory", f"{base}{KEY_PATH}", headers)
    found = [reached, await _ready(client, base), await _version(client, base)]
    if answer is None:  # the key is not taken: nothing more can be asked with it
        return found, None
    key = answer.json()
    return [*found, _role(key), await _model_key(client, s, base, headers, key)], key


def _acting_as(key: Mapping[str, Any] | None) -> list[str]:
    """The agents a key is restricted to (its ``may_act_as``), by id."""
    return [
        principal.removeprefix(AGENT_PRINCIPAL)
        for principal in (key or {}).get("may_act_as") or ()
        if principal.startswith(AGENT_PRINCIPAL)
    ]


def _role(key: Mapping[str, Any]) -> Check:
    """The key's role: a service key is what the harness acts with."""
    role, tenant = key.get("role"), key.get("tenant_id")
    if role == KeyRole.SERVICE:
        return Check("memory key", True, f"a service key of tenant {tenant}")
    if role == KeyRole.PLATFORM:
        return Check("memory key", True, "the platform key: every call names its tenant (tenant=)")
    if role == KeyRole.ADMIN:
        return Check(
            "memory key",
            False,
            f"the admin key of tenant {tenant}: the verdicts it sends for people (h.feedback) "
            "skip the review they wait for, and it manages the tenant's keys; give the "
            "harness a service key (POST /v1/keys with role service)",
        )
    return Check("memory key", False, f"a key of role {role!r}: the harness needs a service key")


async def _model_key(
    client: httpx.AsyncClient,
    s: Settings,
    base: str,
    headers: Mapping[str, str],
    key: Mapping[str, Any],
) -> Check:
    """The model endpoint the memory service's model work for the agents is sent to: each
    agent tells it the route's (``ModelGateway.endpoint``) at its first run; the agents the
    key is restricted to (``may_act_as``) are asked what it holds for them."""
    route = _route(s)
    if route is None or route.endpoint(None) is None:
        return Check(
            "model key",
            True,
            "no model endpoint to tell (no LLM_BASE_URL, or a route the gateway check says is "
            "misconfigured): the memory service's model work for these agents runs on its own "
            "settings",
        )
    named = _acting_as(key)
    if not named:
        return Check(
            "model key",
            True,
            "each agent tells the memory service its model endpoint (LLM_BASE_URL, route, "
            "auth, model, profile) at its first run",
        )
    said: list[str] = []
    for agent in named:
        url = str(httpx.URL(f"{base}{MODEL_KEY_PATH}", params={"agent_id": agent}))
        reached, answer = await _asked(client, "model key", url, headers)
        if answer is None:
            return reached
        said.append(f"{agent}: {_key_state(answer.json())}")
    return Check("model key", True, "; ".join(said))


def _key_state(status: Mapping[str, Any]) -> str:
    """An agent's model-endpoint status, as the report says it: whether it is registered, the
    route, auth, model and profile it holds, and which endpoint the agent's memory work uses
    now, and why."""
    if status.get("revoked"):
        said = "revoked (told again at its next run)"
    elif status.get("registered"):
        said = f"registered (revision {status.get('revision')})"
        endpoint = status.get("endpoint")
        if endpoint:
            accepts = [k for k, v in (endpoint.get("profile") or {}).items() if v is True]
            said += (
                f": {endpoint.get('gateway')} at {endpoint.get('base_url')}, auth "
                f"{endpoint.get('auth')}, model {endpoint.get('model') or 'the service picks'}, "
                f"profile {', '.join(sorted(accepts)) or 'a plain chat model'}"
            )
    else:
        said = "told at its first run"
    if status.get("source"):
        said += f"; its memory work uses the {status['source']} endpoint"
        said += f" ({status['reason']})" if status.get("reason") else ""
    return said


async def _ready(client: httpx.AsyncClient, base: str) -> Check:
    """``/health/ready``: ready, or degraded (an optional dependency down: named)."""
    url = f"{base}{READY_PATH}"
    try:
        body = (await client.get(url)).json()
    except (httpx.HTTPError, ValueError) as exc:
        return Check("memory ready", False, f"{url} cannot be read ({type(exc).__name__})")
    status = body.get("status")
    down = sorted(n for n, d in (body.get("dependencies") or {}).items() if not d.get("ok"))
    where = f" (down: {', '.join(down)})" if down else ""
    return Check("memory ready", status in READY, f"{url}: {status}{where}")


async def _version(client: httpx.AsyncClient, base: str) -> Check:
    """``/version``: the memory service's version and API version."""
    url = f"{base}{VERSION_PATH}"
    try:
        response = await client.get(url)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Check("memory version", False, f"{url} cannot be read ({type(exc).__name__})")
    return Check(
        "memory version",
        True,
        f"{body.get('service')} {body.get('version')} (API {body.get('api_version')})",
    )


async def _approval_rules(
    client: httpx.AsyncClient, s: Settings, agents: Sequence[str]
) -> Check | None:
    """The learned approval suggestions accepted so far, for each of ``agents`` (each by its
    tool and kind): the service learns them from one agent's decisions and lists them in that
    agent's scope only, so a tenant-wide read lists none."""
    if not s.memory_url:
        return None
    if not agents:
        return Check(
            "approval_rules",
            True,
            "learned per agent: trellis doctor --agent <id> lists the rules accepted for one",
        )
    url = f"{s.memory_url.rstrip('/')}{SUGGESTIONS_PATH}"
    headers = {"Authorization": f"Bearer {s.api_key or ''}"}
    said: list[str] = []
    for agent in dict.fromkeys(agents):
        asked = str(httpx.URL(url, params={"agent_id": agent}))
        try:
            response = await client.get(asked, headers=headers)
        except httpx.HTTPError as exc:
            return Check(
                "approval_rules", False, f"{asked} cannot be reached ({type(exc).__name__})"
            )
        if not response.is_success:
            return Check("approval_rules", False, f"{asked} answered {response.status_code}")
        found = response.json().get("suggestions") or []
        accepted = [f"{x['tool']} ({x['suggestion']})" for x in found if x.get("accepted")]
        said.append(f"{agent}: {', '.join(accepted) or 'none'}")
    return Check("approval_rules", True, f"accepted: {'; '.join(said)}")


async def _runs(client: httpx.AsyncClient, s: Settings) -> Check | None:
    if not s.runs_url:
        return None
    headers = {"Authorization": f"Bearer {s.api_key or ''}"}
    url = f"{s.runs_url.rstrip('/')}/v1/runs/doctor-check"
    return await _reached(client, "runs", url, headers, found_ok=True)


async def _reached(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    headers: Mapping[str, str],
    *,
    found_ok: bool = False,
) -> Check:
    return (await _asked(client, name, url, headers, found_ok=found_ok))[0]


async def _asked(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    headers: Mapping[str, str],
    *,
    found_ok: bool = False,
    auth: httpx.Auth | None = None,
) -> tuple[Check, httpx.Response | None]:
    """``url`` asked with the key (in ``headers``, or ``auth``'s): the check, and the answer
    when it was a success."""
    try:
        response = await client.get(
            url, headers=dict(headers), auth=auth or httpx.USE_CLIENT_DEFAULT
        )
    except httpx.HTTPError as exc:
        return Check(name, False, f"{url} cannot be reached ({type(exc).__name__})"), None
    if response.status_code in (401, 403):
        return Check(name, False, f"{url} refused the key ({response.status_code})"), None
    if response.is_success:
        return Check(name, True, f"{url} answered, the key taken"), response
    if found_ok and response.status_code == 404:
        return Check(name, True, f"{url} answered, the key taken"), None
    return Check(name, False, f"{url} answered {response.status_code}"), None


def _spool(folder: str) -> Check:
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder):
            pass
    except OSError as exc:
        return Check("spool", False, f"{folder} cannot be written ({exc.strerror or exc})")
    return Check("spool", True, f"{folder} can be written")


def _records(folder: str) -> Check:
    """Memory writes that failed — tool records, transcripts, outcomes, decisions — kept in
    the spool for the next start to deliver (``writes.py``): the "records failing" check, which
    fails while any wait there, counting them by kind."""
    from trellis.harness.writes import SPOOL_FILE  # noqa: PLC0415 - writes imports telemetry

    kinds: Counter[str] = Counter()
    for path in sorted(Path(folder).glob(f"{SPOOL_FILE}*")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return Check(RECORDS, False, f"{path} cannot be read ({exc.strerror or exc})")
        for line in lines:
            try:
                kinds[str(json.loads(line).get("op") or "unknown")] += 1
            except (ValueError, AttributeError):
                kinds["unreadable"] += 1
    if not kinds:
        return Check(RECORDS, True, f"no memory write waits in {folder}")
    waiting = ", ".join(f"{op}: {n}" for op, n in sorted(kinds.items()))
    return Check(
        RECORDS,
        False,
        f"{sum(kinds.values())} memory write(s) failed and wait in {folder} ({waiting}): they "
        "are delivered at the next start; check the memory service (memory ready above)",
    )


async def _docker() -> Check:
    from trellis.harness.sandbox.docker import DockerSandbox  # noqa: PLC0415

    provider = DockerSandbox()
    try:
        async with provider._client() as client:
            response = await client.get("/_ping")
    except httpx.HTTPError as exc:
        return Check("docker", False, f"{provider.host} cannot be reached ({type(exc).__name__})")
    return Check("docker", response.is_success, f"{provider.host} answered {response.status_code}")


def report(
    rows: Sequence[tuple[str, str, str]],
    found: Sequence[Check],
    said: Sequence[str],
    gateway: Sequence[str] = (),
) -> str:
    width = max(len(name) for name, _, _ in rows)
    lines = ["Settings (value, and where it came from):"]
    lines += [f"  {name.ljust(width)}  {value}  [{source}]" for name, value, source in rows]
    if gateway:
        lines.append("Gateway (where a model name goes):")
        lines += [f"  {line}" for line in gateway]
    lines.append("Checks:")
    lines += [f"  {'ok  ' if c.ok else 'FAIL'}  {c.name}: {c.said}" for c in found] or ["  none"]
    lines.append("Suggestions:")
    lines += [f"  - {line}" for line in said] or ["  none"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """``trellis doctor [--agent ID ...]``: ``--agent`` names an agent whose learned approval
    rules are listed (beside the agents the key may act as)."""
    agents = _agents(list(argv or ()))
    env = os.environ if environ is None else environ
    settings = Settings.from_env(env)
    found = asyncio.run(checks(settings, agents=agents, docker=bool(env.get("DOCKER_HOST"))))
    said = report(effective(env), found, suggestions(settings, env), gateway_said(settings, env))
    sys.stdout.write(said + "\n")
    return 0 if all(c.ok for c in found) else 1


def _agents(argv: list[str]) -> list[str]:
    """The ``--agent ID`` (or ``--agent=ID``) options; anything else is refused."""
    agents: list[str] = []
    while argv:
        given = argv.pop(0)
        if given.startswith("--agent="):
            agents.append(given.removeprefix("--agent="))
        elif given == "--agent" and argv:
            agents.append(argv.pop(0))
        else:
            raise SystemExit(f"usage: trellis doctor [--agent ID ...] (not {given!r})")
    return agents


def cli() -> int:
    """``trellis <command>``: ``trellis doctor``."""
    args = sys.argv[1:]
    if args[:1] != ["doctor"]:
        sys.stderr.write("usage: trellis doctor [--agent ID ...]\n")
        return 2
    return main(args[1:])


if __name__ == "__main__":  # pragma: no cover - the module's own entry
    raise SystemExit(main())
