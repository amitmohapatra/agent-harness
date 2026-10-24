"""``trellis doctor`` (``python -m trellis.harness.doctor``): the effective settings and where
each came from — the environment variable, derived, detected, or the default — secrets
redacted; which gateway serves model names and what it cannot do (``trellis.harness.gateways``);
whether each service is reachable and takes its key — for the memory service also the key's
role, the agents' model key, its readiness (``/health/ready``) and its version (``/version``);
the approval rules accepted from the
memory service's learned suggestions (governance accepts them automatically, but for a tool at
the highest risk); whether the spool can be written and the Docker daemon answers; and what to
set. It changes nothing: it suggests.

Exit code ``0`` when every check passes, ``1`` otherwise."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import httpx

from trellis.harness.gateways import CAPABILITIES, ModelGateway, model_gateway
from trellis.harness.settings import DEPRECATED, Settings, sourced

#: The settings whose values are secrets: shown as set, never as their value.
SECRETS: Final = frozenset(
    {
        "bifrost_virtual_key",
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
#: Where it says what the key is, whether it is ready, which version it is, and an agent's
#: model-key status.
KEY_PATH: Final = "/v1/keys/self"
READY_PATH: Final = "/health/ready"
VERSION_PATH: Final = "/version"
MODEL_KEY_PATH: Final = "/v1/agents/model-key"
#: The memory service's readiness answers that still serve (``degraded``: an optional
#: dependency is down).
READY: Final = frozenset({"ready", "degraded"})
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
    """Which gateway serves model names, what named it, and what is off without it."""
    gateway = model_gateway(settings, environ)
    if gateway is None:
        off = sorted(CAPABILITIES)
        lines = [
            "none: models are objects passed in code (no BIFROST_URL, no OPENAI_BASE_URL or "
            "OPENAI_API_KEY)"
        ]
    else:
        off = sorted(set(CAPABILITIES) - gateway.capabilities)
        lines = [f"{gateway.title} at {gateway.url} ({gateway.source})"]
    lines += [f"off: {CAPABILITIES[c]}" for c in off]
    return lines


async def checks(settings: Settings, environ: Mapping[str, str] | None = None) -> list[Check]:
    """Each service the settings name, reached with its key; the spool; the Docker daemon."""
    env = os.environ if environ is None else environ
    async with httpx.AsyncClient(timeout=CHECK_SECONDS) as client:
        gateway, memory, rules, runs = await asyncio.gather(
            _gateway(client, model_gateway(settings, env)),
            _memory(client, settings),
            _approval_rules(client, settings),
            _runs(client, settings),
        )
    done = [c for c in (gateway, *memory, rules, runs) if c is not None]
    if settings.spool_dir:
        done.append(_spool(settings.spool_dir))
    if settings.sandbox == "docker":
        done.append(await _docker())
    return done


def suggestions(settings: Settings, environ: Mapping[str, str]) -> list[str]:
    """What to set, from what is and is not set (never applied)."""
    said = [
        f"{name} is deprecated: use {instead}"
        for name, instead in DEPRECATED.items()
        if environ.get(name)
    ]
    if settings.bifrost_url:
        said.append(
            "model retries are the gateway's: set the provider's network_config.max_retries in "
            "Bifrost (its default is 0), and the virtual key's fallbacks"
        )
    if not settings.spool_dir:
        said.append(
            "set TRELLIS_SPOOL_DIR to a lasting folder, or memory writes the service "
            "could not take are lost at a restart"
        )
    if not settings.agent_version:
        said.append(
            "bake the version in at build time: OTEL_RESOURCE_ATTRIBUTES=service.version=..."
        )
    if settings.memory_url and not settings.runs_url:
        said.append("runs are kept in this process: set RUNS_URL for durable runs and workers")
    return said


async def _gateway(client: httpx.AsyncClient, gateway: ModelGateway | None) -> Check | None:
    """The gateway that serves model names, asked for its models with its key."""
    if gateway is None:
        return None
    url = f"{gateway.url.rstrip('/')}/models"
    return await _reached(client, "gateway", url, {"Authorization": f"Bearer {gateway.probe_key}"})


async def _memory(client: httpx.AsyncClient, s: Settings) -> list[Check]:
    """The memory service: reached with the key (``/v1/keys/self``) and what the key is (its
    role, the agents' model key), whether it is ready and which version it is."""
    if not s.memory_url:
        return []
    base = s.memory_url.rstrip("/")
    headers = {"Authorization": f"Bearer {s.api_key or ''}"}
    reached, answer = await _asked(client, "memory", f"{base}{KEY_PATH}", headers)
    found = [reached, await _ready(client, base), await _version(client, base)]
    if answer is None:  # the key is not taken: nothing more can be asked with it
        return found
    key = answer.json()
    return [*found, _role(key), await _model_key(client, s, base, headers, key)]


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
    """The model key the memory service's model work for the agents is paid with: each agent
    registers ``BIFROST_VIRTUAL_KEY`` at its first run; the agents the key is restricted to
    (``may_act_as``) are asked for their status."""
    if not s.bifrost_virtual_key:
        return Check(
            "model key",
            True,
            "BIFROST_VIRTUAL_KEY unset: the memory service's model work for these agents runs "
            "on the tenant's or the operator's key",
        )
    named = [
        principal.removeprefix(AGENT_PRINCIPAL)
        for principal in key.get("may_act_as") or ()
        if principal.startswith(AGENT_PRINCIPAL)
    ]
    if not named:
        return Check(
            "model key",
            True,
            "each agent registers BIFROST_VIRTUAL_KEY as its memory model key at its first run",
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
    """An agent's model-key status, as the report says it."""
    if status.get("revoked"):
        return "revoked (registered again at its next run)"
    if status.get("registered"):
        return f"registered (revision {status.get('revision')})"
    return "registered at its first run"


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


async def _approval_rules(client: httpx.AsyncClient, s: Settings) -> Check | None:
    """The learned approval suggestions accepted so far, each by its tool and kind."""
    if not s.memory_url:
        return None
    url = f"{s.memory_url.rstrip('/')}{SUGGESTIONS_PATH}"
    headers = {"Authorization": f"Bearer {s.api_key or ''}"}
    try:
        response = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return Check("approval_rules", False, f"{url} cannot be reached ({type(exc).__name__})")
    if not response.is_success:
        return Check("approval_rules", False, f"{url} answered {response.status_code}")
    found = response.json().get("suggestions") or []
    accepted = [f"{x['tool']} ({x['suggestion']})" for x in found if x.get("accepted")]
    return Check("approval_rules", True, f"accepted: {', '.join(accepted) or 'none'}")


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
) -> tuple[Check, httpx.Response | None]:
    """``url`` asked with the key: the check, and the answer when it was a success."""
    try:
        response = await client.get(url, headers=dict(headers))
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
    del argv  # no options
    env = os.environ if environ is None else environ
    settings = Settings.from_env(env)
    found = asyncio.run(checks(settings, env))
    said = report(effective(env), found, suggestions(settings, env), gateway_said(settings, env))
    sys.stdout.write(said + "\n")
    return 0 if all(c.ok for c in found) else 1


def cli() -> int:
    """``trellis <command>``: ``trellis doctor``."""
    args = sys.argv[1:]
    if args[:1] != ["doctor"]:
        sys.stderr.write("usage: trellis doctor\n")
        return 2
    return main(args[1:])


if __name__ == "__main__":  # pragma: no cover - the module's own entry
    raise SystemExit(main())
