"""Flat v2 surface configs (v2.1 consolidation, 2026-08-01).

Every non-default G1 surface as a standalone flat config on the v2 full
surface base (:class:`~...imitation_g1_env_v2.ImitationG1FullSurfaceEnvCfg`).
Each file states its complete delta (command width, latent window, protocol
choices) with no shared machinery beyond the flat v2 base; duplicated
declarations are fine -- these are deliberately thin.

Families:

- ``vqvae``: causal 9-step expert window for the in-loop VQ-VAE encoder.
- ``future_cvae``: current + nine future frames (CVAE / per-step-VQ).
- ``goal``: held 25-step future-goal state for hierarchical skills.
- ``ablation``: reconstruction-ablation protocol (66-D phase clock).
- ``sonic``: SONIC release recipe (h10 histories, full-trajectory adaptive
  resets) plus the official renewed-FSQ window arm.

The legacy ``variants/`` classes these replaced are deleted; their task ids
were re-registered here on the v2 env (see ``config/g1/__init__.py``).
"""

from .ablation import ImitationG1AblationSurfaceEnvCfg
from .future_cvae import (
    ImitationG1FutureCVAESurfaceEnvCfg,
    ImitationG1PerStepVQSurfaceEnvCfg,
)
from .goal import ImitationG1GoalSurfaceEnvCfg
from .sonic import (
    ImitationG1SonicNoHistorySurfaceEnvCfg,
    ImitationG1SonicOfficialFSQSurfaceEnvCfg,
    ImitationG1SonicSurfaceEnvCfg,
)
from .vqvae import ImitationG1VQVAESurfaceEnvCfg

__all__ = [
    "ImitationG1AblationSurfaceEnvCfg",
    "ImitationG1FutureCVAESurfaceEnvCfg",
    "ImitationG1GoalSurfaceEnvCfg",
    "ImitationG1PerStepVQSurfaceEnvCfg",
    "ImitationG1SonicNoHistorySurfaceEnvCfg",
    "ImitationG1SonicOfficialFSQSurfaceEnvCfg",
    "ImitationG1SonicSurfaceEnvCfg",
    "ImitationG1VQVAESurfaceEnvCfg",
]
