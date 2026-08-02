"""Command terms for the manager-based imitation environments.

First increments of the v2 CommandManager redesign: adapter-phase terms that
expose the existing ImitationRLEnv reference machinery through the native
Isaac Lab :class:`~isaaclab.managers.CommandManager` surface, plus the
externally-published command layer (env-side mirror of
``contracts/command_publisher.py``).
"""

from .held_chunk_command import HeldChunkCommand, HeldChunkCommandCfg
from .motion_command import MotionCommand, MotionCommandCfg
from .published_command import PublishedCommandTerm, PublishedCommandTermCfg
from .reference_command import ReferenceCommand, ReferenceCommandCfg
from .skill_command import SkillCommand, SkillCommandCfg

__all__ = [
    "HeldChunkCommand",
    "HeldChunkCommandCfg",
    "MotionCommand",
    "MotionCommandCfg",
    "PublishedCommandTerm",
    "PublishedCommandTermCfg",
    "ReferenceCommand",
    "ReferenceCommandCfg",
    "SkillCommand",
    "SkillCommandCfg",
]
