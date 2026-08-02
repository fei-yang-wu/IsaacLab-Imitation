"""Command terms for the manager-based imitation environments.

Two channels (see ``tasks/manager_based/imitation/command_interface.py``): the
always-present :class:`~.reference.ReferenceCommandTerm`, which owns reference
selection, reset-start sampling, and the tracking metrics; and exactly one
actor emitter -- latent, explicit, or chunk -- under the manager name
``actor``. :class:`~.published_command.PublishedCommandTerm` is the shared base
for the two externally-written emitters (env-side mirror of
``contracts/command_publisher.py``).
"""

from .actor import (
    ACTOR_TERM_NAME,
    REFERENCE_TERM_NAME,
    ActorCommandCfg,
    ChunkActorCommand,
    ChunkCommandCfg,
    ExplicitActorCommand,
    ExplicitCommandCfg,
    LatentActorCommand,
    LatentCommandCfg,
)
from .published_command import PublishedCommandTerm, PublishedCommandTermCfg
from .reference import (
    ReferenceChannelCfg,
    ReferenceCommandTerm,
    ReferenceSelectionCfg,
)

__all__ = [
    "ACTOR_TERM_NAME",
    "REFERENCE_TERM_NAME",
    "ActorCommandCfg",
    "ChunkActorCommand",
    "ChunkCommandCfg",
    "ExplicitActorCommand",
    "ExplicitCommandCfg",
    "LatentActorCommand",
    "LatentCommandCfg",
    "PublishedCommandTerm",
    "PublishedCommandTermCfg",
    "ReferenceChannelCfg",
    "ReferenceCommandTerm",
    "ReferenceSelectionCfg",
]
