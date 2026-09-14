"""The shared Sharpa PPO recipe with a useful local-check cadence."""

from isaaclab.utils.configclass import configclass

from ..sharpa.sharpa_agents import SHARPA_REFERENCE_BATCH, SharpaRLOptPPOConfig


@configclass
class VegaSharpaRLOptPPOConfig(SharpaRLOptPPOConfig):
    save_interval_iterations: int = 50
    log_interval_iterations: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = self.save_interval_iterations * SHARPA_REFERENCE_BATCH
        self.trainer.log_interval = (
            self.log_interval_iterations * SHARPA_REFERENCE_BATCH
        )
        self.trainer.progress_bar = False
