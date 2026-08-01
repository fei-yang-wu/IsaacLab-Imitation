"""Latent skill command term for the manager-based CommandManager surface.

Strangler-pattern adapter (v2 redesign, step 4a): :class:`SkillCommand` wraps
the env's EXISTING agent-latent command state
(``ImitationRLEnv._agent_latent_command`` behind
``get_agent_latent_command`` / ``set_agent_latent_command`` /
``reset_agent_latent_command``) rather than owning new buffers. Its
``command`` property returns exactly what the v1 observation func
``mdp.agent_latent_command(env)`` returns -- the ``latent_command_dim``-wide
tensor (z + phase, already composed by the agent today) -- so rebinding the
``latent_command`` observation terms onto this term is value-identical.

Ownership of the latent buffer moves into this term in a later step of the
redesign; in this phase the term is the CommandManager-facing view plus the
published/hold bookkeeping from :class:`PublishedCommandTerm`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .published_command import PublishedCommandTerm, PublishedCommandTermCfg

if TYPE_CHECKING:
    from isaaclab_imitation.envs import ImitationRLEnv


class SkillCommand(PublishedCommandTerm):
    """Agent-published latent skill command, adapted from the env's buffer.

    Constructor-ordering note: ``ImitationRLEnv`` allocates
    ``_agent_latent_command`` (width ``cfg.latent_command_dim``) before
    ``super().__init__`` runs ``load_managers()``, so the buffer exists by the
    time this term is constructed and the width check below can fail loudly on
    a cfg/env mismatch.

    Unpublished-consumption policy (adapter phase): serve the env buffer
    as-is. The env zero-fills it on reset and the observation managers read it
    before the agent's first in-step publication, so failing loudly here would
    break the existing training loop; the fail-loud policy arrives when this
    term owns the storage and the publication schedule.
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: SkillCommandCfg

    def __init__(self, cfg: SkillCommandCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        env_latent_dim = getattr(env, "_agent_latent_dim", None)
        if env_latent_dim is None:
            raise RuntimeError(
                "SkillCommand requires an ImitationRLEnv with the agent-latent "
                "command buffer (`_agent_latent_dim`); it was not found on "
                f"{type(env).__name__}."
            )
        if int(env_latent_dim) != int(cfg.latent_command_dim):
            raise ValueError(
                "SkillCommandCfg.latent_command_dim does not match the env's "
                f"latent command width: {int(cfg.latent_command_dim)} vs "
                f"{int(env_latent_dim)}. The adapter-phase term serves the "
                "env's buffer, so the two must be identical."
            )

    def __str__(self) -> str:
        msg = "SkillCommand (adapter over ImitationRLEnv agent-latent buffer):\n"
        msg += f"\tCommand dimension: {int(self.cfg.latent_command_dim)}\n"
        msg += f"\tHold steps: {self.hold_steps}"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Latent skill command. Shape is (num_envs, latent_command_dim).

        Exactly ``mdp.agent_latent_command(env)``: the env-owned buffer the
        agent publishes into (z + phase composed by the agent today).
        """
        return self._imitation_env().get_agent_latent_command()

    """
    Implementation specific functions.
    """

    def _apply_published_payload(
        self, env_ids: torch.Tensor, payload: torch.Tensor
    ) -> None:
        self._imitation_env().set_agent_latent_command(payload, env_ids=env_ids)

    def _update_command(self):
        """No-op: the command lives in the env buffer between publications."""

    def _update_metrics(self):
        """No-op: no skill-command metrics in the adapter phase."""

    def _resample_command(self, env_ids: Sequence[int]):
        """Documented no-op for the buffer; only clears the published mask.

        ``ImitationRLEnv._reset_idx`` already calls
        ``reset_agent_latent_command(env_ids)`` for the resetting envs
        *before* ``super()._reset_idx`` triggers the CommandManager reset that
        lands here, so delegating another buffer reset would be a double
        reset. The base class clears the published mask (fail-loud policy is
        deferred; see the class docstring).
        """
        super()._resample_command(env_ids)

    """
    Helper functions.
    """

    def _imitation_env(self) -> ImitationRLEnv:
        return self._env  # type: ignore[return-value]


@configclass
class SkillCommandCfg(PublishedCommandTermCfg):
    """Configuration for the latent skill command term."""

    class_type: type = SkillCommand

    # pyrefly: ignore[bad-assignment]  # Isaac Lab required-field idiom
    latent_command_dim: int = MISSING
    """Width of the latent skill command (z + phase).

    Required (no default): the mdp layer must not hardcode a recipe-specific
    width, so the env cfg wires it from its own ``latent_command_dim`` (258
    for the v1/v2 default recipe). Must match the env's buffer width while the
    term is an adapter over ``ImitationRLEnv._agent_latent_command``.
    """
