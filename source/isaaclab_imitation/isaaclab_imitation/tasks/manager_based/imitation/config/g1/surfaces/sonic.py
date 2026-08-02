# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC release-recipe surfaces on the single v2 env (rebase 2026-08-02).

The SONIC delta over the v2 default: SONIC's 10-step proprioceptive
histories applied to the v2 policy/critic groups (no-history-by-default
convention: the declarations stay history-free and the histories are applied
post-init), the termination-threshold anneal curriculum, and SONIC's
full-trajectory adaptive-failure reset sampling
(``random_reset_full_trajectory=True`` with
``failure_rate_max_over_mean=200``); the SONIC rewards/events/actions are
already the v2 components.
"""

from isaaclab.utils.configclass import configclass

from ..common.terminations import G1SonicTerminationCurriculumCfg
from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg

_SONIC_HISTORY_TERMS_POLICY = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)
_SONIC_HISTORY_TERMS_CRITIC = (
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


@configclass
class ImitationG1SonicSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Single v2 env matched to the public SONIC release recipe.

    Termination thresholds are annealed from the release's base/eval values
    to its strict training values over the curriculum window; every frame
    after the window uses the strict release protocol. Disable with
    ``env.curriculum=null`` for strict-from-scratch release fidelity.
    """

    curriculum = G1SonicTerminationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # SONIC's 10-step proprio histories, applied post-init so the v2
        # declarations stay history-free by default.
        policy = self.observations.policy
        critic = self.observations.critic
        for term_name in _SONIC_HISTORY_TERMS_POLICY:
            term = getattr(policy, term_name)
            if term is not None:
                term.history_length = 10
        for term_name in _SONIC_HISTORY_TERMS_CRITIC:
            term = getattr(critic, term_name)
            if term is not None:
                term.history_length = 10

        # SONIC's motion library samples over the complete trajectory, with
        # adaptive failure weighting and a uniform component. The v2 default
        # intentionally limits starts to [0, 200], so undo that only here.
        selection = self.command_interface.reference.selection
        selection.random_step_max = 0
        selection.full_trajectory = True
        selection.adaptive_failure_rate_max_over_mean = 200.0


@configclass
class ImitationG1SonicNoHistorySurfaceEnvCfg(ImitationG1SonicSurfaceEnvCfg):
    """SONIC release environment with this repo's single-frame observations.

    Everything on the environment side stays the SONIC release recipe; the
    one deliberate departure is the observation set: the 2026-07-21 isolated
    history ablation showed SONIC's 10-step proprioceptive histories buy
    little at our scale, so this surface keeps the single-frame v2
    observations (the surface is otherwise identical).
    """


@configclass
class ImitationG1SonicOfficialFSQSurfaceEnvCfg(ImitationG1SonicSurfaceEnvCfg):
    """SONIC release environment with a renewed 10-frame FSQ window command."""

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor.dim = 64
        self.command_interface.encoder.past_steps = 0
        self.command_interface.encoder.future_steps = 9
        # Keep the sample-efficient reset sampler established by the Stable
        # reset screen; full-trajectory adaptive-failure starts need far more
        # data at our single-GPU scale.
        selection = self.command_interface.reference.selection
        selection.random_step_max = 200
        selection.full_trajectory = False
        selection.adaptive_failure_rate_max_over_mean = 50.0
        # The command window advances with the live reference; the agent-side
        # code_period=1 independently renews the quantized code.


__all__ = [
    "ImitationG1SonicNoHistorySurfaceEnvCfg",
    "ImitationG1SonicOfficialFSQSurfaceEnvCfg",
    "ImitationG1SonicSurfaceEnvCfg",
]
