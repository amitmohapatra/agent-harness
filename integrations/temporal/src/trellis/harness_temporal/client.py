"""One connection, shared by both adapters.

``Client.connect`` is a coroutine and a harness is built synchronously, so a deployment that
configures ``runs.engine = temporal`` cannot hand over a connected client at construction
time. This holds either: a client somebody already connected, or a target to connect to on
first use. Both adapters take one of these, so a process that stores runs *and* schedules in
Temporal opens one connection, not two.

The data converter is not optional: every payload crossing the workflow boundary is a
``trellis-contracts`` pydantic model, and Temporal's default JSON converter cannot round-trip
one. Connecting here is how that mistake stops being possible.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

#: What a deployment passes: a connected client, or the target to connect to.
ClientSource = Client | str


class TemporalConnection:
    """A lazily connected Temporal client. Safe to share between adapters and tasks."""

    __slots__ = ("_client", "_lock", "_options", "_owns_client", "_target", "namespace")

    def __init__(
        self,
        source: ClientSource,
        *,
        namespace: str = "default",
        api_key: str | None = None,
        tls: bool = False,
        rpc_metadata: Mapping[str, str] | None = None,
        **connect_options: Any,
    ) -> None:
        if isinstance(source, Client):
            self._client: Client | None = source
            self._target: str | None = None
            self.namespace = source.namespace
            self._owns_client = False
        else:
            if not source:
                raise ValueError("TemporalConnection needs a client or a target address")
            self._client = None
            self._target = source
            self.namespace = namespace
            self._owns_client = True
        self._options: dict[str, Any] = {
            "namespace": namespace,
            "data_converter": pydantic_data_converter,
            **({"api_key": api_key} if api_key else {}),
            **({"tls": tls} if tls else {}),
            **({"rpc_metadata": dict(rpc_metadata)} if rpc_metadata else {}),
            **connect_options,
        }
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def client(self) -> Client:
        """The client, connecting once. Concurrent first calls share one connection."""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                self._client = await Client.connect(str(self._target), **self._options)
        return self._client

    async def aclose(self) -> None:
        """Release the connection this object opened.

        temporalio's client has no ``close``: its gRPC connection belongs to the Rust runtime
        and goes when the client does. So this drops the reference (and only ours — a client
        somebody else passed in is theirs to manage), which is what makes a second use
        reconnect instead of reusing a client the caller believes is finished with.
        """
        if self._owns_client:
            self._client = None


__all__ = ["ClientSource", "TemporalConnection"]
