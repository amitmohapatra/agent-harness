"""One pause mechanism for every framework (design §7): the signal an agent raises becomes a
contracts ``Interrupt``; the answer comes back as an ``InterruptResolution``."""

from trellis.harness.interrupts.resolutions import ANSWER, ResolutionRegistry, Resolved
from trellis.harness.interrupts.signals import ApprovalRequired, interrupt_from_signal

__all__ = ["ANSWER", "ApprovalRequired", "ResolutionRegistry", "Resolved", "interrupt_from_signal"]
