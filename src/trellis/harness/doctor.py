"""``trellis doctor`` (``python -m trellis.harness.doctor``): the effective settings and where
each came from — the environment variable, derived, detected, or the default — secrets
redacted; which gateway serves model names and what it cannot do (``trellis.harness.gateways``);
whether each service is reachable and takes its key; whether the spool can be
written and the Docker daemon answers; and what to set. It changes nothing: it suggests.

Exit code ``0`` when every check passes, ``1`` otherwise."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Final

import httpx

from trellis.harness.gateways import CAPABILITIES, ModelGateway, model_gateway
from trellis.harness.settings import DEPRECATED, Settings, sourced

#: How many trailing characters of a secret are shown (of one at least twice as long).
SECRET_SHOWN: Final = 4
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
        return (
            f"set (...{text[-SECRET_SHOWN:]})"
            if isinstance(value, str) and len(text) > SECRET_SHOWN * 2
            else "set"
        )
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
        found = await asyncio.gather(
            _gateway(client, model_gateway(settings, env)),
            _memory(client, settings),
            _runs(client, settings),
        )
    done = [c for c in found if c is not None]
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


async def _memory(client: httpx.AsyncClient, s: Settings) -> Check | None:
    if not s.memory_url:
        return None
    headers = {"Authorization": f"Bearer {s.api_key or ''}"}
    return await _reached(client, "memory", f"{s.memory_url.rstrip('/')}/v1/keys/self", headers)


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
    try:
        response = await client.get(url, headers=dict(headers))
    except httpx.HTTPError as exc:
        return Check(name, False, f"{url} cannot be reached ({type(exc).__name__})")
    if response.status_code in (401, 403):
        return Check(name, False, f"{url} refused the key ({response.status_code})")
    if response.is_success or (found_ok and response.status_code == HTTPStatus.NOT_FOUND):
        return Check(name, True, f"{url} answered, the key taken")
    return Check(name, False, f"{url} answered {response.status_code}")


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
        async with provider.client() as client:
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
