"""Externally published command terms for the manager-based CommandManager surface.

Second increment of the v2 redesign (step 4a): :class:`PublishedCommandTerm` is
the env-side mirror of the client-side
:class:`~isaaclab_imitation.contracts.command_publisher.CommandPublisher`
protocol. Where a regular :class:`~isaaclab.managers.CommandTerm` *samples* its
own command, a published command term receives it from an external writer (an
agent or a high-level planner) through :meth:`publish` and only *serves* it.

Vocabulary is kept aligned with ``contracts/command_publisher.py``:

* **publish** -- an external writer stores a new command for selected envs;
* **consume** -- a reader (observation term) fetches the served command;
* **due** -- env ids whose hold window restarts on this control step;
* **reset** -- the term returns to the unpublished state for resetting envs.

The renewal schedule is the one schedule authority shared by every interface:
:func:`~isaaclab_imitation.contracts.planner_publish_schedule.planner_renew_env_ids`
(``episode_length_buf % hold_steps == 0``). It is imported, not reimplemented,
so publisher and consumer cannot disagree about when a window begins even when
environments reset asynchronously.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_imitation.contracts.planner_publish_schedule import (
    planner_renew_env_ids,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PublishedCommandTerm(CommandTerm):
    """Base class for command terms written by an external publisher.

    Owns the per-env published mask and the hold-phase machinery; subclasses
    own the storage (:meth:`_apply_published_payload`) and the policy for
    consuming an unpublished command (fail loudly vs serve a neutral value --
    see the reset invariant in ``contracts/command_publisher.py``).
    """

    # pyrefly: ignore[bad-override-mutable-attribute]  # Isaac Lab term idiom
    cfg: PublishedCommandTermCfg

    def __init__(self, cfg: PublishedCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if int(cfg.hold_steps) <= 0:
            raise ValueError(
                f"PublishedCommandTermCfg.hold_steps must be positive, got "
                f"{cfg.hold_steps}."
            )
        # True once an external writer has published since the env's last
        # reset. Cleared (not zero-filled: a zeroed command is a valid-looking
        # but meaningless payload) by ``_resample_command`` on reset.
        self._published = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    """
    Properties.
    """

    @property
    def published(self) -> torch.Tensor:
        """Per-env bool mask: True once published since the last reset."""
        return self._published

    @property
    def hold_steps(self) -> int:
        """Control steps a published command is held before renewal is due."""
        return int(self.cfg.hold_steps)

    @property
    def hold_phase(self) -> torch.Tensor:
        """Episode-local step within the current hold window (0..hold_steps-1).

        Derived from ``episode_length_buf`` -- the same counter the schedule
        authority uses -- so there is no separate phase buffer that could
        drift from the env's reset behavior.
        """
        return self._env.episode_length_buf.remainder(self.hold_steps)

    """
    Operations (publisher-facing API, mirroring CommandPublisher).
    """

    def due(self, *, initial: bool = False) -> torch.Tensor:
        """Env ids whose hold window restarts on this control step.

        Delegates to :func:`planner_renew_env_ids` on the env's live
        ``episode_length_buf`` (renewal at ``episode_length_buf % hold_steps
        == 0``). Set ``initial`` on the first call of a rollout to request
        publication for every environment regardless of episode phase.
        """
        return planner_renew_env_ids(
            self._env.episode_length_buf,
            self.hold_steps,
            initial_publication=initial,
        )

    def publish(self, env_ids: torch.Tensor, payload: Any) -> None:
        """Store an externally produced command for the selected envs.

        Delegates storage to :meth:`_apply_published_payload`, then marks the
        rows published. The payload contract (shape/keys) is subclass-owned.
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._apply_published_payload(env_ids, payload)
        self._published.index_fill_(0, env_ids, True)

    """
    Implementation specific functions.
    """

    @abstractmethod
    def _apply_published_payload(self, env_ids: torch.Tensor, payload: Any) -> None:
        """Write the published payload into the term's command storage."""
        raise NotImplementedError

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset the hold phase and mark the envs unpublished.

        The hold phase is derived from ``episode_length_buf``, which the env's
        reset path returns to zero, so clearing the published mask is the only
        state transition needed here. The mask is cleared rather than the
        command zero-filled: a zeroed command is well formed and would be
        consumed silently (reset invariant in
        ``contracts/command_publisher.py``); whether consumption of an
        unpublished command fails loudly is subclass policy.
        """
        self._published[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        """No-op: published command terms carry no debug visualization."""


@configclass
class PublishedCommandTermCfg(CommandTermCfg):
    """Configuration base for externally published command terms."""

    # The manager's timer never resamples a published command; renewal is the
    # publisher's job via ``due()`` + ``publish()`` (same idiom as
    # MotionCommandCfg).
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    hold_steps: int = 1
    """Control steps each published command is held before renewal is due.

    Renewal follows ``episode_length_buf % hold_steps == 0`` (the shared
    schedule authority). The default of 1 means a fresh command is due every
    control step, which matches an agent publishing inside ``env.step``.
    """
