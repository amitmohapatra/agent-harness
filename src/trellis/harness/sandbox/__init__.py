"""Sandboxes: a place where commands run and files change away from the host — the
provider-neutral interface (``base``) and the Docker provider (``docker``)."""

from __future__ import annotations

from trellis.harness.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxLost,
    SandboxProvider,
    SandboxRef,
    SandboxSpec,
    SupportsPause,
    SupportsSnapshot,
)
from trellis.harness.sandbox.docker import DockerSandbox

__all__ = [
    "DockerSandbox",
    "ExecResult",
    "Sandbox",
    "SandboxLost",
    "SandboxProvider",
    "SandboxRef",
    "SandboxSpec",
    "SupportsPause",
    "SupportsSnapshot",
]
