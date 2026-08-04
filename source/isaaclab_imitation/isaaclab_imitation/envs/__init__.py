from .imitation_rl_env_legacy import ImitationRLEnvLegacy
from .imitation_rl_env_v2 import ImitationRLEnv, ImitationRLEnvV2

# `ImitationRLEnv` is the flagship env class (the v2 fork, `-G1-v2` task);
# `ImitationRLEnvLegacy` is the byte-frozen v0/v1 env. `ImitationRLEnvV2`
# remains exported as a back-compat alias for configs recorded against the
# pre-flip entry point `isaaclab_imitation.envs:ImitationRLEnvV2`.
__all__ = ["ImitationRLEnv", "ImitationRLEnvLegacy", "ImitationRLEnvV2"]
