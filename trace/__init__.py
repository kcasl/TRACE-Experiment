"""TRACE: Task-Regret-Aware instance association.

The package is host-agnostic. A host calls ``choose_interaction_mask`` at the
moment it would otherwise pick an interaction mask. ``None`` means control
stays with the host.
"""

from .selector import TraceSelector, choose_interaction_mask

__all__ = ["TraceSelector", "choose_interaction_mask"]
