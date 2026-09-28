"""The AG-UI surface of trellis-harness (design §10): ``agui_router`` streams a run's
events as AG-UI events over SSE and takes the protocol's ``resume`` entries; ``translate``
is the mapping on its own for deployments with their own transport."""

from trellis.harness_agui.events import (
    AGUIEvent,
    AGUIEventType,
    InterruptEntry,
    Outcome,
    OutcomeType,
    Resume,
    ResumeStatus,
    RunAgentInput,
)
from trellis.harness_agui.router import agui_router
from trellis.harness_agui.sse import MEDIA_TYPE, decode, encode
from trellis.harness_agui.translate import interrupt_entry, translate

__version__ = "0.1.0"

__all__ = [
    "MEDIA_TYPE",
    "AGUIEvent",
    "AGUIEventType",
    "InterruptEntry",
    "Outcome",
    "OutcomeType",
    "Resume",
    "ResumeStatus",
    "RunAgentInput",
    "__version__",
    "agui_router",
    "decode",
    "encode",
    "interrupt_entry",
    "translate",
]
