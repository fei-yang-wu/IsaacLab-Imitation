"""Matched latent-learning configs for the corrected-LAFAN1 ablation."""

from isaaclab.utils.configclass import configclass

from .rlopt_ipmd_cfg import (
    FUTURE_CVAE_POSTERIOR_INPUT_KEYS,
    FUTURE_CVAE_PRIOR_INPUT_KEYS,
    G1ImitationLatentRLOptIPMDConfig,
)


@configclass
class G1ImitationLatentAblationRLOptIPMDConfig(G1ImitationLatentRLOptIPMDConfig):
    """Online reconstruction baseline with a held, phase-aware latent.

    Quantizer and joint-policy-gradient behavior are intentionally selected by
    launcher overrides. The default is a continuous autoencoder
    (``quantizer=identity``); VQ-VAE, FSQ, and SONIC-style FSQ+PG arms differ
    only in those explicit overrides.
    """

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.ipmd.latent_learning.posterior_input_keys = list(
            FUTURE_CVAE_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = list(FUTURE_CVAE_PRIOR_INPUT_KEYS)
        self.ipmd.latent_learning.reconstruction_target_keys = list(
            FUTURE_CVAE_POSTERIOR_INPUT_KEYS
        )

    def __post_init__(self):
        super().__post_init__()
        self.ipmd.latent_dim = 66
        self.ipmd.latent_steps_min = 10
        self.ipmd.latent_steps_max = 10
        self.ipmd.command_source = "posterior"
        self.ipmd.latent_learning.method = "patch_vqvae"
        self.ipmd.latent_learning.quantizer = "identity"
        self.ipmd.latent_learning.code_latent_dim = 64
        self.ipmd.latent_learning.command_phase_mode = "sin_cos"
        self.ipmd.latent_learning.code_period = 10
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 9
        self.ipmd.latent_learning.posterior_command_period = 10
        self.ipmd.latent_learning.encoder_hidden_dims = [512, 256]
        self.ipmd.latent_learning.decoder_hidden_dims = [256, 512]
        self.ipmd.latent_learning.lr = 3.0e-4
        self.ipmd.latent_learning.grad_clip_norm = 1.0
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.action_recon_coeff = 0.0
        self.ipmd.latent_learning.kl_coeff = 0.0
        self.ipmd.latent_learning.freeze_encoder = False
        self.ipmd.latent_learning.train_posterior_through_policy = False
        self.ipmd.latent_learning.fsq_levels = [4, 4, 4, 4, 4]
        self.ipmd.latent_learning.codebook_size = 512
        self.ipmd.latent_learning.codebook_embed_dim = 64
        self.ipmd.latent_learning.commitment_coeff = 0.25
        self.ipmd.latent_learning.ema_decay = 0.99
        self.ipmd.latent_learning.dead_code_reset_iters = 1000
        self.sync_input_keys()
