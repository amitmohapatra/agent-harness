"""A checkpointer standing for a durable one (LangGraph's ``PostgresSaver``): what it holds lives
outside the worker — here in a module-level table by name, which a "restarted" worker (a new
harness, the graph built again) reads as a new process would read the database."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

#: The "database": each named saver's checkpoints, outliving any one worker.
DATABASE: dict[str, InMemorySaver] = {}


class SharedSaver(BaseCheckpointSaver[str]):
    """A durable checkpointer named ``name``: every instance of that name reads the same data."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._kept = DATABASE.setdefault(name, InMemorySaver())

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._kept.get_tuple(config)

    def list(self, config: RunnableConfig | None, **kwargs: Any) -> Iterator[CheckpointTuple]:
        return self._kept.list(config, **kwargs)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._kept.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._kept.put_writes(config, writes, task_id, task_path)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._kept.aget_tuple(config)

    async def alist(
        self, config: RunnableConfig | None, **kwargs: Any
    ) -> AsyncIterator[CheckpointTuple]:
        async for found in self._kept.alist(config, **kwargs):
            yield found

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await self._kept.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self._kept.aput_writes(config, writes, task_id, task_path)

    def get_next_version(self, current: str | None, channel: None) -> str:
        return self._kept.get_next_version(current, channel)
