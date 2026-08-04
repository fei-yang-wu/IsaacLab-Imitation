"""Privileged-teacher IPMD with an online-distilled G1 student actor."""

from __future__ import annotations

import copy
from dataclasses import field

from isaaclab.utils.configclass import configclass

from isaaclab_imitation.envs.rlopt import IPMDL2TConfig

from .rlopt_ipmd_cfg import G1ImitationLatentSonicRLOptIPMDConfig


@configclass
class G1ImitationRLOptIPMDL2TConfig(G1ImitationLatentSonicRLOptIPMDConfig):
    """Current-v2 IPMD teacher/student configuration.

    The teacher keeps the ordinary actor architecture but reads the exact
    ordered critic input contract. The student keeps the original deployable
    actor observations and architecture. Command-interface binding reruns
    :meth:`sync_input_keys`, so the same class supports latent, explicit, and
    chunk v2 surfaces without stale key lists.
    """

    ipmd_l2t: IPMDL2TConfig = field(default_factory=IPMDL2TConfig)

    def _sync_student_architecture(self) -> None:
        """Keep the deployable actor architecture tied to the teacher actor."""
        teacher = self.policy
        student = self.ipmd_l2t.student_policy
        student.num_cells = list(teacher.num_cells)
        student.output_dim = teacher.output_dim
        student.activation_fn = teacher.activation_fn
        student.normalize_input = teacher.normalize_input
        student.normalization_epsilon = teacher.normalization_epsilon
        student.normalization_clip = teacher.normalization_clip
        student.kwargs = copy.deepcopy(teacher.kwargs)

    def sync_input_keys(self) -> None:
        """Resolve ordinary actor/critic keys, then assign L2T roles."""
        super().sync_input_keys()
        if self.value_function is None:
            raise ValueError("IPMDL2T requires a value-function configuration.")

        self._sync_student_architecture()
        student_keys = list(self.policy.get_input_keys())
        teacher_keys = list(self.value_function.get_input_keys())
        self.ipmd_l2t.student_policy.input_keys = student_keys
        self.policy.input_keys = teacher_keys

        if bool(self.ipmd.use_latent_command):
            teacher_latent_key = ("critic", "latent_command")
            student_latent_key = ("policy", "latent_command")
            if teacher_latent_key not in teacher_keys:
                raise ValueError(
                    "Latent IPMDL2T teacher inputs must contain "
                    f"{teacher_latent_key!r}."
                )
            if student_latent_key not in student_keys:
                raise ValueError(
                    "Latent IPMDL2T student inputs must contain "
                    f"{student_latent_key!r}."
                )
            self.ipmd.latent_key = teacher_latent_key
            self.ipmd_l2t.student_latent_key = student_latent_key
            self.policy.normalize_input_exclude_keys = [teacher_latent_key]
            self.ipmd_l2t.student_policy.normalize_input_exclude_keys = [
                student_latent_key
            ]
        else:
            self.policy.normalize_input_exclude_keys = []
            self.ipmd_l2t.student_policy.normalize_input_exclude_keys = []

    def __post_init__(self) -> None:
        super().__post_init__()
        # Start from the exact actor architecture selected by the parent recipe.
        # Input keys are reassigned below, so this copy never leaks critic keys
        # into the deployable student contract.
        self.ipmd_l2t.student_policy = copy.deepcopy(self.policy)
        self.sync_input_keys()
        self.ipmd_l2t.validate()


__all__ = ["G1ImitationRLOptIPMDL2TConfig"]
