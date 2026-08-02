"""Lean v2 ``command`` term: latent command + reset ownership + tracking metrics.

The thin v2 default carries ONE command term, named ``command``, instead of
the three-term (motion / skill / chunk) surface the full layout needs:

- **Command surface**: serves the agent-published latent skill command
  (z + phase) from the env's ``LatentCommandBuffer``, exactly what
  ``SkillCommand`` served as ``get_command("skill")``.
- **Reset-start sampling ownership** (v2 step 3c, absorbed): constructs the
  ``StartFrameSampler`` / ``SonicAdaptiveResetSampler`` pair and owns the
  adaptive-failure bookkeeping (``record_visits`` / ``record_failures`` /
  ``set_weight_fn`` / ``resample_reference``), exactly like ``MotionCommand``.
- **Tracking metrics**: ``Metrics/command/mpjpe_mm`` / ``anchor_pos_err_m`` /
  ``anchor_ori_err_rad``, computed from the env's per-step-cached fast paths.

The explicit 67-D command publishing (``MotionCommand``) and the held-chunk
packet consumption (``HeldChunkCommand``) are not part of the lean surface;
explicit / chunk / reconstruction variants declare those terms via their own
configs. ``ReferenceCommand`` implements this by inheriting the metrics and
reset machinery from ``MotionCommand`` and replacing the published command
with the agent-latent buffer.
"""

from __future__ import annotations

import torch

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .motion_command import MotionCommand, MotionCommandCfg


class ReferenceCommand(MotionCommand):
    """Lean command term: agent-latent command + reset ownership + metrics.

    The ``command`` property serves the env's agent-latent buffer (published
    by the RLOpt agent via ``set_agent_latent_command``), so the manager
    surface, the observation funcs (``mdp.reference_latent_command``), and
    the metrics all read one producer. Everything else (reset-start sampler
    construction, adaptive-failure record hooks, resample_reference, and the
    tracking metrics) is inherited unchanged from :class:`MotionCommand`.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: ReferenceCommandCfg

    def __str__(self) -> str:
        msg = "ReferenceCommand (lean v2 command term):\n"
        msg += f"\tCommand dimension: {int(self.cfg.latent_command_dim)}\n"
        msg += f"\tAnchor body: {self.cfg.anchor_body_name}\n"
        msg += f"\tMPJPE bodies: {len(self.cfg.mpjpe_body_names)}"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Agent-published latent skill command. Shape is (num_envs, latent_command_dim)."""
        return self._imitation_env().get_agent_latent_command()

    """
    Implementation specific functions.
    """

    def _update_command(self):
        """No-op: the latent command lives in the env buffer between publications."""

    def _refresh_command(self) -> None:
        """Unused by the latent surface; retained for MotionCommand parity."""

    def _command_dim(self) -> int:
        return int(self.cfg.latent_command_dim)


@configclass
class ReferenceCommandCfg(MotionCommandCfg):
    """Configuration for the lean v2 ``command`` term."""

    class_type: type = ReferenceCommand

    # pyrefly: ignore[bad-assignment]  # Isaac Lab required-field idiom
    latent_command_dim: int = MISSING
    """Width of the agent-published latent command (z + phase).

    Required (no default): wired from the env cfg's ``latent_command_dim``
    (258 for the default recipe), mirroring ``SkillCommandCfg``.
    """

    # The explicit joint half is not part of the lean surface; the base
    # MotionCommandCfg joint_names field stays unused (None = ignored here).
