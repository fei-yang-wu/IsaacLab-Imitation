"""Strict-protocol environment for controlled latent-representation ablations."""

from isaaclab.utils.configclass import configclass

from .imitation_g1_env_cfg import _g1_lafan_track_env_cfg_from_dict
from .imitation_g1_latent_env_cfg import ImitationG1LatentStrictEnvCfg


@configclass
class ImitationG1LatentAblationEnvCfg(ImitationG1LatentStrictEnvCfg):
    """Expose current + nine future frames on the strict LAFAN1 surface.

    The reconstruction learners publish a 64-value code plus a two-value
    within-chunk phase clock. Individual arms may override the command width
    (for example the phase-free CVAE row) without changing the environment
    protocol.
    """

    latent_command_dim: int = 66

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        self.command_hold_steps = 0
        self._sync_expert_window_observation_params()


ImitationG1LatentAblationEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
