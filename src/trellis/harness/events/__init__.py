"""The run event stream (ADR 0001 in trellis-contracts, design §4): every execution emits
ordered ``RunEvent``s to the sinks the harness was given, and never blocks on them."""

from trellis.harness.events.sinks import (
    CollectingEventSink,
    CompositeEventSink,
    FilteringEventSink,
)
from trellis.harness.events.stream import RunEventStream
from trellis.harness.events.webhook import WebhookEventSink

__all__ = [
    "CollectingEventSink",
    "CompositeEventSink",
    "FilteringEventSink",
    "RunEventStream",
    "WebhookEventSink",
]
