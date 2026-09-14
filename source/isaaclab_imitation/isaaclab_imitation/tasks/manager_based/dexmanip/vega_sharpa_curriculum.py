"""Optional endpoint-success assistance schedule for the mounted local task."""

from __future__ import annotations

from isaaclab_imitation.contracts.object_assistance_gate import ObjectAssistanceGate

from .curriculum import FixedTimestepCurriculum


class ObjectSuccessAssistanceCurriculum(FixedTimestepCurriculum):
    """Keep reward timing, but reduce assistance only after endpoint success."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.gate = ObjectAssistanceGate(
            env.num_envs, env.device, len(self._scale_factors)
        )

    def training_state_dict(self):
        return {"gate": self.gate.state_dict(), "scale_factors": self._scale_factors}

    def load_training_state_dict(self, state):
        if tuple(state["scale_factors"]) != tuple(self._scale_factors):
            raise ValueError("Assistance levels differ from the checkpoint.")
        self.gate.load_state_dict(state["gate"])

    def __call__(
        self,
        env,
        env_ids,
        command_name,
        num_steps_per_env,
        timestep_schedule,
        virtual_object_control_scale_factor,
        reward_weight_schedules,
    ):
        # CurriculumManager runs before command reset. Refresh the terminal
        # physical state now; metrics from the preceding step are insufficient.
        self._command._update_object_task_metrics()
        stage = self.gate.reset(
            env_ids,
            self._command.metrics["object_task_success"],
            env.episode_length_buf,
            int(env.common_step_counter),
        )
        super().__call__(
            env,
            env_ids,
            command_name,
            num_steps_per_env,
            timestep_schedule,
            virtual_object_control_scale_factor,
            reward_weight_schedules,
        )
        scale = self._command.virtual_object_controller_curriculum_scale
        scale.fill_(self._scale_factors[stage])
        return scale.detach().clone()
