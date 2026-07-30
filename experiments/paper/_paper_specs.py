"""Frozen tables describing the command interfaces and latent-learning arms.

These are the numbers the paper's claims rest on, so they live in one validated
place rather than being re-derived in each launcher. Two things they buy:

* **Packet widths are checked, not assumed.** A planner trained on samples whose
  ``target_dim`` disagrees with the interface's declared width is training on a
  different interface than the config says. That mismatch is silent at training
  time and only shows up as a bad closed-loop number days later.
* **Unimplemented arms fail at config time.** Several latent strategies exist in
  the online latent-learning lineage but not in the offline DiffSR trainer (and
  vice versa). Selecting one that cannot run should be a config error, not a
  confusing argparse rejection after Isaac Sim has booted.

All per-frame widths below are for the G1 with 29 actuated joints, 4 end-effector
bodies, and 5 keypoint bodies. They are asserted against the environment's own
term widths at collection time; see ``check_packet_width``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# G1 morphology constants that determine the packet widths.
G1_NUM_JOINTS = 29
G1_NUM_EE_BODIES = 4
G1_NUM_KEYPOINT_BODIES = 5

#: Anchor pose is a 3-vector position plus a 6D rotation representation.
_ANCHOR_WIDTH = 3 + 6


@dataclass(frozen=True)
class InterfaceSpec:
    """One command interface: how it is published, and by whom it is consumed."""

    name: str
    kind: str  # "latent" | "explicit"
    per_frame_values: int | None
    command_terms: tuple[str, ...]
    default_task: str
    #: Which collection entrypoint understands this interface.
    collector: str  # "skill_commander" | "interface_rollout"
    #: ``agent.command_space`` for the natively-trained low-level controller.
    command_space: str | None = None
    #: ``env.policy_command_mode`` used when a packet is consumed slot-by-slot.
    policy_command_mode: str | None = None
    #: Command observation terms to keep when training the low-level controller.
    command_observation_terms: tuple[str, ...] = ()
    notes: str = ""

    def packet_values(self, *, horizon_steps: int, latent_dim: int = 0) -> int:
        """Total values published per planner decision."""
        if self.kind == "latent":
            return latent_dim
        assert self.per_frame_values is not None
        return self.per_frame_values * horizon_steps


EXPLICIT_INTERFACES: dict[str, InterfaceSpec] = {
    "full_body_trajectory": InterfaceSpec(
        name="full_body_trajectory",
        kind="explicit",
        # 29 qpos + 29 qvel + anchor pose.
        per_frame_values=2 * G1_NUM_JOINTS + _ANCHOR_WIDTH,
        command_terms=("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b"),
        default_task="Isaac-Imitation-G1-Strict-v0",
        collector="interface_rollout",
        command_space="full_body_trajectory",
        policy_command_mode="full_body_chunk_current_slot",
        command_observation_terms=(
            "expert_motion",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        ),
        notes="The paper's explicit row: 670 values at 5 Hz, consumed one slot per step.",
    ),
    "root_qpos": InterfaceSpec(
        name="root_qpos",
        kind="explicit",
        per_frame_values=G1_NUM_JOINTS + _ANCHOR_WIDTH,
        command_terms=(
            "expert_motion_qpos",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        ),
        default_task="Isaac-Imitation-G1-Strict-v0",
        collector="interface_rollout",
        command_space="root_qpos",
        policy_command_mode="full_body_chunk_current_slot",
        command_observation_terms=(
            "expert_motion_qpos",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        ),
        notes="Joint positions plus root, no velocities.",
    ),
    "root_points5": InterfaceSpec(
        name="root_points5",
        kind="explicit",
        per_frame_values=3 * G1_NUM_KEYPOINT_BODIES + _ANCHOR_WIDTH,
        command_terms=(
            "expert_keypoint_pos_b",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        ),
        default_task="Isaac-Imitation-G1-Strict-v0",
        collector="interface_rollout",
        command_space="root_points5",
        policy_command_mode="full_body_chunk_current_slot",
        command_observation_terms=(
            "expert_keypoint_pos_b",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        ),
        notes="Sparse keypoint positions plus root.",
    ),
    "ee_trajectory": InterfaceSpec(
        name="ee_trajectory",
        kind="explicit",
        per_frame_values=(3 + 6) * G1_NUM_EE_BODIES,
        command_terms=("expert_ee_pos_b", "expert_ee_ori_b"),
        default_task="Isaac-Imitation-G1-Strict-v0",
        collector="interface_rollout",
        command_space="ee_trajectory",
        policy_command_mode="ee_chunk_current_slot",
        command_observation_terms=("expert_ee_pos_b", "expert_ee_ori_b"),
        notes=(
            "Rootless control. End-effector poses in the anchor frame never say "
            "where the anchor should go, so this interface is under-determined "
            "and its measured oracle floor is an order of magnitude worse than "
            "the others. Kept as a deliberate control, not as a candidate row."
        ),
    ),
}

LATENT_INTERFACE = InterfaceSpec(
    name="latent_skill",
    kind="latent",
    per_frame_values=None,
    command_terms=("z",),
    default_task="Isaac-Imitation-G1-Latent-v0",
    collector="skill_commander",
    command_space=None,
    policy_command_mode=None,
    notes="DiffSR latent command published at 5 Hz and held by the tracker.",
)

INTERFACES: dict[str, InterfaceSpec] = {
    LATENT_INTERFACE.name: LATENT_INTERFACE,
    **EXPLICIT_INTERFACES,
}


def get_interface(name: str) -> InterfaceSpec:
    try:
        return INTERFACES[name]
    except KeyError:
        raise KeyError(
            f"Unknown interface {name!r}. Known interfaces: {sorted(INTERFACES)}"
        ) from None


def check_packet_width(
    interface: str,
    *,
    observed_target_dim: int,
    horizon_steps: int,
    latent_dim: int = 0,
) -> None:
    """Fail when collected samples do not match the declared interface width.

    The planner trainer derives ``target_dim`` from the sample metadata, so a
    horizon or interface mismatch produces a trainable-but-wrong planner. This
    turns that into a config-time error.
    """
    spec = get_interface(interface)
    expected = spec.packet_values(horizon_steps=horizon_steps, latent_dim=latent_dim)
    if observed_target_dim != expected:
        detail = (
            f"latent_dim={latent_dim}"
            if spec.kind == "latent"
            else f"{spec.per_frame_values} values/frame x {horizon_steps} frames"
        )
        raise ValueError(
            f"Interface {interface!r}: collected samples declare target_dim="
            f"{observed_target_dim}, but the configured packet is {expected} "
            f"({detail}).\n"
            "The planner would train on a different interface than the config "
            "names. Re-collect with a matching horizon, or fix the interface."
        )


# --------------------------------------------------------------------------
# Latent-learning arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentModeSpec:
    """One latent-learning strategy for the encoder.

    ``lineage`` distinguishes two genuinely different code paths:

    ``offline``
        Trained by ``scripts/rlopt/train_hl_skill_diffsr.py`` against sampled
        expert macro transitions, then frozen and consumed by the low-level
        policy. This is the DiffSR path the paper's latent row uses.
    ``online``
        Trained jointly with the low-level policy through
        ``agent.ipmd.latent_learning.*``. There is no separate pretrain stage,
        so the pipeline skips pretraining for these arms.
    """

    name: str
    lineage: str  # "offline" | "online"
    #: ``--latent_mode`` value for the offline trainer.
    latent_mode: str | None = None
    #: ``agent.ipmd.latent_learning.method`` for the online lineage.
    online_method: str | None = None
    #: Extra config keys this arm reads from ``latent.<key>``.
    hyperparameter_group: str | None = None
    #: Task override required by the online arms that have their own registration.
    task: str | None = None
    notes: str = ""
    extra_overrides: dict[str, str] = field(default_factory=dict)


LATENT_MODES: dict[str, LatentModeSpec] = {
    "deterministic": LatentModeSpec(
        name="deterministic",
        lineage="offline",
        latent_mode="deterministic",
        notes="Continuous bottleneck with an L2 penalty on z. The paper default.",
    ),
    "gaussian": LatentModeSpec(
        name="gaussian",
        lineage="offline",
        latent_mode="gaussian",
        notes=(
            "Stochastic bottleneck with a KL penalty -- the offline VAE arm. This "
            "is the reconstruction-side counterpart to the online future_cvae "
            "arm; they are different code paths and are not interchangeable."
        ),
    ),
    "vq": LatentModeSpec(
        name="vq",
        lineage="offline",
        latent_mode="vq",
        hyperparameter_group="vq",
        notes="EMA vector quantization over a single codebook.",
    ),
    "fsq": LatentModeSpec(
        name="fsq",
        lineage="offline",
        latent_mode="fsq",
        hyperparameter_group="fsq",
        notes="Finite scalar quantization with a small per-dimension level set.",
    ),
    "sonic_fsq": LatentModeSpec(
        name="sonic_fsq",
        lineage="offline",
        latent_mode="sonic_fsq",
        hyperparameter_group="sonic_fsq",
        notes="SONIC-matched FSQ: 64 dimensions x 32 levels. Requires z_dim == len(levels).",
    ),
    "categorical": LatentModeSpec(
        name="categorical",
        lineage="offline",
        latent_mode="categorical",
        hyperparameter_group="categorical",
        notes="Multi-categorical bottleneck trained with a KL penalty.",
    ),
    "gumbel": LatentModeSpec(
        name="gumbel",
        lineage="offline",
        latent_mode="gumbel",
        hyperparameter_group="gumbel",
        notes="Single Gumbel-softmax codebook with temperature annealing.",
    ),
    "gumbel_multicat": LatentModeSpec(
        name="gumbel_multicat",
        lineage="offline",
        latent_mode="gumbel_multicat",
        hyperparameter_group="gumbel",
        notes="Grouped Gumbel-softmax codebooks.",
    ),
    "future_cvae": LatentModeSpec(
        name="future_cvae",
        lineage="online",
        online_method="future_cvae",
        task="Isaac-Imitation-G1-Latent-FutureCVAE-v0",
        notes=(
            "Conditional VAE over the current-and-future window, trained jointly "
            "with the policy. There is no offline DiffSR pretrain for this arm: "
            "the encoder is not frozen, so the pipeline runs low-level training "
            "directly. Use the 'gaussian' arm for the offline VAE comparison."
        ),
    ),
    "patch_vqvae": LatentModeSpec(
        name="patch_vqvae",
        lineage="online",
        online_method="patch_vqvae",
        task="Isaac-Imitation-G1-Latent-VQVAE-v0",
        notes="Online VQ-VAE over a causal past window.",
    ),
    "per_step_vq_sequence": LatentModeSpec(
        name="per_step_vq_sequence",
        lineage="online",
        online_method="per_step_vq_sequence",
        task="Isaac-Imitation-G1-Latent-PerStepVQ-v0",
        notes="Per-step VQ token sequence consumed as a token plan.",
    ),
}


def get_latent_mode(name: str) -> LatentModeSpec:
    try:
        return LATENT_MODES[name]
    except KeyError:
        offline = sorted(k for k, v in LATENT_MODES.items() if v.lineage == "offline")
        online = sorted(k for k, v in LATENT_MODES.items() if v.lineage == "online")
        raise KeyError(
            f"Unknown latent mode {name!r}.\n"
            f"  offline (DiffSR pretrain): {offline}\n"
            f"  online  (joint training):  {online}"
        ) from None


# --------------------------------------------------------------------------
# Planner families
# --------------------------------------------------------------------------

#: Planner objective families the shared trainer implements.
PLANNER_FAMILIES = ("flow", "diffusion", "deterministic")

#: Named architecture presets in the shared trainer.
PLANNER_SIZES = ("tiny", "small", "medium", "large")

#: Planning schedules compared by the hold-out versus receding-horizon study.
#:
#: ``holdout``
#:     One planner decision per publication interval; the tracker holds that
#:     command for the whole interval. For the explicit interfaces the packet is
#:     consumed one slot per control step.
#: ``receding_horizon``
#:     The planner replans every control step and only slot 0 is ever used.
PLANNING_SCHEDULES = ("holdout", "receding_horizon")
