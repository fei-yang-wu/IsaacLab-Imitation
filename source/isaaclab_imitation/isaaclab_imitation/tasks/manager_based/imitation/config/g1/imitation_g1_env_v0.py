# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v0``: LafanTrack recipe x explicit full-body command.

The original torso-anchored, loose-termination tracking recipe on the
vanilla observation surface -- the frozen v0 release. Everything the release
pins is visible below; the LAFAN1 dataset/manifest machinery comes from
``ImitationG1BaseTrackingEnvCfg`` in ``common``.

The flagship ``ImitationG1EnvCfg`` name moved to the newest release
(``imitation_g1_env_v1.py``) in the 2026-07-31 overhaul; this class keeps
its recipe name only.
"""

from isaaclab.utils.configclass import configclass

from .common.actions import G1ActionsCfg
from .common.events import G1EventCfg
from .common.observations import G1ObservationCfg
from .common.rewards import G1RewardsCfg
from .common.terminations import G1TerminationsCfg
from .common.tracking_env import (
    ImitationG1BaseTrackingEnvCfg,
    _bind_lafan_track_from_dict,
)


@configclass
class ImitationG1LafanTrackEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """General 29-DoF motion-tracking env driven by a LAFAN1 manifest.

    The v0 release protocol, spelled out:

    - mimic actuators/action scale (``G1ActionsCfg``) on the bundled G1 USD;
    - the vanilla explicit-command observation surface
      (``G1ObservationCfg``, torso_link anchor);
    - the original tracking rewards and loose z-only terminations
      (``G1RewardsCfg`` / ``G1TerminationsCfg``, torso_link anchor);
    - baseline domain randomization and pushes (``G1EventCfg``);
    - full-trajectory random reset starts (below); no curriculum.
    """

    actions = G1ActionsCfg()
    observations = G1ObservationCfg()
    rewards = G1RewardsCfg()  # type: ignore
    terminations = G1TerminationsCfg()  # type: ignore
    events = G1EventCfg()

    random_reset_full_trajectory: bool = True


_bind_lafan_track_from_dict(ImitationG1LafanTrackEnvCfg)

__all__ = ["ImitationG1LafanTrackEnvCfg"]
