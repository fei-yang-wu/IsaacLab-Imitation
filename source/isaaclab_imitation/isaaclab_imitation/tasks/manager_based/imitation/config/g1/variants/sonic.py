# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC release-recipe variants (deprecated as defaults; kept reproducible).

The Stable default (``imitation_g1_env_v1.py``) shares this recipe's
components from ``common`` but is defined standalone; nothing outside this
file inherits from these classes.
"""

from isaaclab.utils.configclass import configclass

from ..common.actions import G1SonicActionsCfg
from ..common.events import G1SonicEventCfg
from ..common.latent_env import ImitationG1LatentEnvCfg
from ..common.observations_latent import (
    G1LatentObservationCfg,
    G1SonicLatentObservationCfg,
)
from ..common.presets import G1SonicRobotCfg
from ..common.rewards import G1SonicRewardsCfg
from ..common.terminations import (
    G1SonicTerminationCurriculumCfg,
    G1SonicTerminationsCfg,
)
from ..common.tracking_env import (
    _apply_pelvis_protocol,
    _bind_lafan_track_from_dict,
)


@configclass
class ImitationG1LatentSonicEnvCfg(ImitationG1LatentEnvCfg):
    """Latent G1 task matched to the public SONIC release recipe.

    Termination thresholds are annealed from the release's base/eval values
    to its strict training values over the curriculum window; every frame
    after the window uses the strict release protocol. Disable with
    ``env.curriculum=null`` for strict-from-scratch release fidelity.
    """

    actions = G1SonicActionsCfg()
    observations = G1SonicLatentObservationCfg()
    rewards = G1SonicRewardsCfg()  # type: ignore
    terminations = G1SonicTerminationsCfg()  # type: ignore
    events = G1SonicEventCfg()
    curriculum = G1SonicTerminationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        robot_preset = G1SonicRobotCfg()
        for variant in (
            robot_preset.default,
            robot_preset.physx,
            robot_preset.newton_mjwarp,
        ):
            variant.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_preset  # type: ignore

        # SONIC's motion library samples over the complete trajectory, with
        # adaptive failure weighting and a uniform component. The parent latent
        # task intentionally limits starts to [0, 200], so undo that only here.
        _apply_pelvis_protocol(
            self,
            reset_step_max=0,
            random_reset_full_trajectory=True,
            failure_rate_max_over_mean=200.0,
        )


@configclass
class ImitationG1LatentSonicNoHistoryEnvCfg(ImitationG1LatentSonicEnvCfg):
    """SONIC release environment with this repo's single-frame observations.

    Everything on the environment side stays the SONIC release recipe --
    ``G1SonicRewardsCfg`` (pelvis anchor, 3-point local reward points,
    anti-shake, feet joint acceleration, elbow-exempt contact penalty),
    ``G1SonicTerminationsCfg`` (adaptive ``anchor_pos``/``ee_body_pos``,
    full ``anchor_ori``, ``foot_pos_xyz``, no ``base_too_low``),
    ``G1SonicTerminationCurriculumCfg``, ``G1SonicEventCfg`` (level0_4
    randomization), ``G1SonicActionsCfg``, ``G1SonicRobotCfg``, and SONIC's
    full-trajectory adaptive-failure reset sampling.

    The one deliberate departure is the observation set: the 2026-07-21
    isolated history ablation (``ImitationG1LatentStrictHistoryEnvCfg`` vs.
    the single-frame strict surface) showed SONIC's 10-step proprioceptive
    histories buy little at our scale, so this surface keeps the repo's
    single-frame ``G1LatentObservationCfg``. Term *names* are unchanged, so
    ``G1ImitationLatentSonicRLOptIPMDConfig``'s SONIC input-key selection
    (which adds ``projected_gravity`` and drops the robot body-pose terms
    from the actor) still resolves; only the per-term history length differs.
    """

    observations = G1LatentObservationCfg()


@configclass
class ImitationG1LatentSonicOfficialFSQEnvCfg(ImitationG1LatentSonicEnvCfg):
    """SONIC release environment with a renewed 10-frame FSQ window command."""

    latent_command_dim: int = 64

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        # Keep the sample-efficient reset sampler established by the Stable
        # reset screen; full-trajectory adaptive-failure starts need far more
        # data at our single-GPU scale. (Anchoring came from the SONIC parent.)
        _apply_pelvis_protocol(self, failure_rate_max_over_mean=50.0, set_anchor=False)
        # Zero means the observation window advances with the live reference.
        # The agent-side code_period=1 independently renews the quantized code.
        self.command_hold_steps = 0
        self._sync_expert_window_observation_params()


_bind_lafan_track_from_dict(
    ImitationG1LatentSonicEnvCfg,
    ImitationG1LatentSonicNoHistoryEnvCfg,
    ImitationG1LatentSonicOfficialFSQEnvCfg,
)
