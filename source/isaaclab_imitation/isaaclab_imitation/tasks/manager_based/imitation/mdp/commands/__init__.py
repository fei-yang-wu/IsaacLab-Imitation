"""Command terms for the manager-based imitation environments.

First increment of the v2 CommandManager redesign: adapter-phase terms that
expose the existing ImitationRLEnv reference machinery through the native
Isaac Lab :class:`~isaaclab.managers.CommandManager` surface.
"""

from .motion_command import MotionCommand, MotionCommandCfg

__all__ = ["MotionCommand", "MotionCommandCfg"]
