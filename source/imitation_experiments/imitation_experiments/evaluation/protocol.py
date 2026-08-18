"""Versioned evaluation contracts, protocols, boards, and CLI capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TRACKING_TERMINATION_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos")
FOOT_TRACKING_TERMINATION_NAME = "foot_pos_xyz"
FALL_TERMINATION_NAME = "base_too_low"

_SONIC_OVERRIDES = (
    "env.events.push_robot=null",
    "env.terminations.anchor_pos.params.threshold=0.25",
    "env.terminations.anchor_pos.params.down_threshold=0.25",
    "env.terminations.anchor_ori.params.threshold=1.0",
    "env.terminations.ee_body_pos.params.threshold=0.25",
    "env.terminations.ee_body_pos.params.down_threshold=0.25",
    "env.terminations.foot_pos_xyz=null",
    "env.terminations.base_too_low=null",
)
_FALL_ONLY_OVERRIDES = (
    "env.terminations.anchor_pos=null",
    "env.terminations.anchor_ori=null",
    "env.terminations.ee_body_pos=null",
    "env.terminations.foot_pos_xyz=null",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize one hash payload with stable key and separator rules."""

    return json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def content_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ProtocolUnsupported(ValueError):
    """Requested evaluator cannot realize one or more protocol fields."""

    def __init__(self, evaluator: str, fields: Sequence[str]):
        self.evaluator = evaluator
        self.fields = tuple(fields)
        super().__init__(
            f"{evaluator} cannot realize protocol fields: {', '.join(self.fields)}"
        )


class ProtocolMismatch(ValueError):
    """Realized evaluator state differs from requested protocol state."""

    def __init__(self, differences: Mapping[str, tuple[Any, Any]]):
        self.differences = dict(differences)
        details = ", ".join(
            f"{name}: requested={requested!r}, realized={realized!r}"
            for name, (requested, realized) in sorted(self.differences.items())
        )
        super().__init__(f"Realized evaluation protocol differs: {details}")


@dataclass(frozen=True)
class TrackerEvalContractV1:
    """What tracker, data, and policy contract one evaluation used."""

    checkpoint_sha256: str
    cumulative_env_frames: int | None
    task_id: str
    algorithm: str
    agent_entry_point: str | None
    actor_interface: tuple[tuple[str, Any], ...]
    encoder_binding: tuple[tuple[str, Any], ...]
    dataset: tuple[tuple[str, Any], ...]
    tracked_body_names: tuple[str, ...]
    policy: tuple[tuple[str, Any], ...]
    physics: tuple[tuple[str, Any], ...]
    resolved_config_sha256: str
    schema_version: str = field(default="tracker_eval_contract_v1", init=False)

    def hash_payload(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return content_hash(self.hash_payload())

    def stamp(self) -> dict[str, Any]:
        return {**_json_value(asdict(self)), "content_hash": self.content_hash()}


@dataclass(frozen=True)
class EvalProtocolV1:
    """How one evaluation measures behavior, independent of episode population."""

    protocol_id: str
    backend: str
    episode_horizon: str
    outer_safety_cap_steps: int
    metric_interval: int
    action_sampling: str
    randomization_profile: str
    randomization_kept: tuple[tuple[str, bool], ...]
    start_mode: str
    termination_profile: str
    active_terminations: tuple[str, ...]
    disabled_terminations: tuple[str, ...]
    success_definition: str
    fall_height_m: float | None
    observation_corruption: bool
    # Uniform half-widths applied to the controller's view of the robot, as
    # ordered pairs so they are part of the protocol hash: two runs at
    # different noise levels must never share an identity. Empty when
    # ``observation_corruption`` is False.
    observation_noise: tuple[tuple[str, float], ...] = ()
    tracked_body_names: tuple[str, ...] = ()
    mpjpe_definition: str = "root_position_subtracted_and_world_frame"
    headline_metric: str = "tracking_mpjpe_mm"
    reduction: str = "equal_episode_mean"
    description: str = ""
    schema_version: str = field(default="eval_protocol_v1", init=False)

    def hash_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("description", None)
        return payload

    def content_hash(self) -> str:
        return content_hash(self.hash_payload())

    def stamp(self) -> dict[str, Any]:
        return {**_json_value(asdict(self)), "content_hash": self.content_hash()}


@dataclass(frozen=True)
class EvalEpisodeCaseV1:
    """One ordered evaluation episode.

    ``motion_name`` is human-readable provenance. Board identity uses rank,
    start, seed, and repeat because dataset identity lives in the contract.

    ``population_weight`` is how much of the population this case stands for on
    a board that deliberately over-samples rare motions. It is 1.0 on a board
    that samples uniformly, and it is part of the board hash: two boards over
    the same motions with different weights report different numbers and must
    not share an identity.
    """

    trajectory_rank: int
    start_frame: int
    env_seed: int
    repeat_index: int = 0
    motion_name: str | None = None
    population_weight: float = 1.0

    def identity(self) -> tuple[int, int, int, int]:
        return (
            int(self.trajectory_rank),
            int(self.start_frame),
            int(self.env_seed),
            int(self.repeat_index),
        )


@dataclass(frozen=True)
class EvalBoardV1:
    """Which ordered episodes one evaluation scores."""

    board_id: str
    cases: tuple[EvalEpisodeCaseV1, ...]
    description: str = ""
    schema_version: str = field(default="eval_board_v1", init=False)

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("Evaluation board must contain at least one episode case.")
        identities = [case.identity() for case in self.cases]
        if len(set(identities)) != len(identities):
            raise ValueError("Evaluation board contains duplicate episode identities.")

    def hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "board_id": self.board_id,
            "cases": [
                {
                    "trajectory_rank": case.trajectory_rank,
                    "start_frame": case.start_frame,
                    "env_seed": case.env_seed,
                    "repeat_index": case.repeat_index,
                    "population_weight": round(float(case.population_weight), 9),
                }
                for case in self.cases
            ],
        }

    def content_hash(self) -> str:
        return content_hash(self.hash_payload())

    def stamp(self) -> dict[str, Any]:
        return {**_json_value(asdict(self)), "content_hash": self.content_hash()}

    @property
    def trajectory_ranks(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(case.trajectory_rank for case in self.cases))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(case.env_seed for case in self.cases))

    @property
    def start_frames(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(case.start_frame for case in self.cases))


@dataclass(frozen=True)
class EvalProfileV1:
    """Bind one evaluator adapter to immutable protocol and board identities."""

    profile_id: str
    evaluator: str
    protocol_id: str
    protocol_hash: str
    board_id: str
    board_hash: str
    adapter_version: str = "v1"
    description: str = ""
    schema_version: str = field(default="eval_profile_v1", init=False)

    def hash_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("description", None)
        return payload

    def content_hash(self) -> str:
        return content_hash(self.hash_payload())

    def stamp(self) -> dict[str, Any]:
        return {**_json_value(asdict(self)), "content_hash": self.content_hash()}


def frozen_pairs(values: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Convert a JSON-like mapping into deterministic immutable pairs."""

    return tuple(
        (str(key), _json_value(value)) for key, value in sorted(values.items())
    )


def validate_realized_protocol(
    requested: EvalProtocolV1, realized: EvalProtocolV1
) -> None:
    """Fail with only fields whose realized values differ."""

    requested_payload = requested.hash_payload()
    realized_payload = realized.hash_payload()
    ignored = {"protocol_id"}
    differences = {
        key: (requested_payload.get(key), realized_payload.get(key))
        for key in sorted((set(requested_payload) | set(realized_payload)) - ignored)
        if requested_payload.get(key) != realized_payload.get(key)
    }
    if differences:
        raise ProtocolMismatch(differences)


def unpinned_protocol_stamp(*, backend: str = "unknown") -> dict[str, Any]:
    """Explicit marker for a run outside the versioned protocol registry."""

    return {
        "schema_version": "eval_protocol_v1",
        "protocol_id": "unpinned",
        "backend": backend,
        "content_hash": None,
    }


def make_rank_board(
    board_id: str,
    ranks: Iterable[int],
    *,
    seed: int = 0,
    repeats: int = 1,
    start_frame: int = 0,
    description: str = "",
) -> EvalBoardV1:
    return EvalBoardV1(
        board_id=board_id,
        cases=tuple(
            EvalEpisodeCaseV1(
                trajectory_rank=int(rank),
                start_frame=int(start_frame),
                env_seed=int(seed),
                repeat_index=repeat,
            )
            for rank in ranks
            for repeat in range(int(repeats))
        ),
        description=description,
    )


# SONIC's own released policy-group noise, read verbatim from the release
# `config.yaml` (`enable_corruption: true`, gravity +/-0.05, base_ang_vel
# +/-0.2, joint_pos +/-0.01, joint_vel +/-0.5; `last_action` carries none).
# Our G1 observation config replicates these, so a rehearsal that injects them
# is reproducing the distribution the policy was actually optimized against —
# and real hardware never delivers a cleaner reading than this.
SONIC_OBSERVATION_NOISE: tuple[tuple[str, float], ...] = (
    ("base_ang_vel", 0.2),
    ("joint_pos", 0.01),
    ("joint_vel", 0.5),
    ("projected_gravity", 0.05),
)


G1_TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def _protocol(
    protocol_id: str,
    *,
    backend: str = "newton_mjwarp",
    termination_profile: str,
    active: tuple[str, ...],
    disabled: tuple[str, ...],
    success_definition: str,
    randomization: str = "no_push",
    metric_interval: int = 1,
    cap: int = 10_000,
    observation_corruption: bool = False,
    observation_noise: tuple[tuple[str, float], ...] = (),
    reduction: str = "equal_episode_mean",
) -> EvalProtocolV1:
    kept = {
        "startup": randomization in {"all", "no_push"},
        "reset": randomization in {"all", "no_push"},
        "push": randomization == "all",
    }
    return EvalProtocolV1(
        protocol_id=protocol_id,
        backend=backend,
        episode_horizon="reference_length",
        outer_safety_cap_steps=cap,
        metric_interval=metric_interval,
        action_sampling="mode",
        randomization_profile=randomization,
        randomization_kept=tuple(sorted(kept.items())),
        start_mode="board",
        termination_profile=termination_profile,
        active_terminations=active,
        disabled_terminations=disabled,
        success_definition=success_definition,
        fall_height_m=0.4 if FALL_TERMINATION_NAME in active else None,
        observation_corruption=observation_corruption,
        observation_noise=observation_noise if observation_corruption else (),
        tracked_body_names=G1_TRACKED_BODY_NAMES,
        reduction=reduction,
    )


PROTOCOLS: dict[str, EvalProtocolV1] = {
    "tracker_fall_only_v1": _protocol(
        "tracker_fall_only_v1",
        termination_profile="fall_only",
        active=(FALL_TERMINATION_NAME, "reference_finished"),
        disabled=TRACKING_TERMINATION_NAMES + (FOOT_TRACKING_TERMINATION_NAME,),
        success_definition="no_base_too_low_termination",
    ),
    "tracker_full_horizon_v1": _protocol(
        "tracker_full_horizon_v1",
        termination_profile="full_horizon",
        active=("reference_finished",),
        disabled=TRACKING_TERMINATION_NAMES
        + (FOOT_TRACKING_TERMINATION_NAME, FALL_TERMINATION_NAME),
        success_definition="diagnostic_only",
    ),
    "tracker_task_strict_v1": _protocol(
        "tracker_task_strict_v1",
        termination_profile="task_strict",
        active=TRACKING_TERMINATION_NAMES
        + (FOOT_TRACKING_TERMINATION_NAME, FALL_TERMINATION_NAME, "reference_finished"),
        disabled=(),
        success_definition="reference_finished_without_task_failure",
    ),
    "sonic_sr_v1": _protocol(
        "sonic_sr_v1",
        termination_profile="sonic",
        active=TRACKING_TERMINATION_NAMES + ("reference_finished",),
        disabled=(FOOT_TRACKING_TERMINATION_NAME, FALL_TERMINATION_NAME),
        success_definition="reference_finished_without_sonic_tracking_failure",
    ),
    # The paper-facing quality protocol. Same success contract as `sonic_sr_v1`
    # -- which is SONIC's published definition verbatim: unsuccessful if root
    # height or end-effector height deviates by more than 0.25 m from the
    # reference, or root orientation by more than 1 radian, with `foot_pos_xyz`
    # and `base_too_low` off -- but with randomization fully off, because the
    # paper's simulation tables are quality numbers and startup plus reset
    # randomization costs the released checkpoint 2.75 mm of MPJPE-L on the
    # canonical block (28.65 -> 25.90) while moving success rate by 0.001.
    # Report a headline MPJPE-L here and its robustness partner under
    # `sonic_sr_v1`; never mix the two in one table.
    #
    # `reduction` is frame-weighted over successful episodes only, which is
    # what the 4,096 board has always published as "micro". A success-only
    # figure is meaningless beside a different success rate, so the success
    # rate travels with it in the same sentence.
    "sonic_sr_clean_v1": _protocol(
        "sonic_sr_clean_v1",
        termination_profile="sonic",
        active=TRACKING_TERMINATION_NAMES + ("reference_finished",),
        disabled=(FOOT_TRACKING_TERMINATION_NAME, FALL_TERMINATION_NAME),
        success_definition="reference_finished_without_sonic_tracking_failure",
        randomization="none",
        reduction="frame_weighted_success_only",
    ),
    "gr00t_planner_v1": _protocol(
        "gr00t_planner_v1",
        termination_profile="fall_only",
        active=(FALL_TERMINATION_NAME, "reference_finished"),
        disabled=TRACKING_TERMINATION_NAMES + (FOOT_TRACKING_TERMINATION_NAME,),
        success_definition="no_base_too_low_termination",
        randomization="all",
        metric_interval=10,
        cap=2000,
    ),
    # Cross-backend calibration. Physics randomization stays off so the
    # comparison isolates the backends, but sensor noise stays ON to match the
    # EC rehearsal protocol: comparing a noisy backend against a clean one
    # would measure the noise, not the physics.
    "cross_backend_isaac_v1": _protocol(
        "cross_backend_isaac_v1",
        termination_profile="fall_only",
        active=(FALL_TERMINATION_NAME, "reference_finished"),
        disabled=TRACKING_TERMINATION_NAMES + (FOOT_TRACKING_TERMINATION_NAME,),
        success_definition="no_base_too_low_termination",
        randomization="none",
        observation_corruption=True,
        observation_noise=SONIC_OBSERVATION_NOISE,
    ),
    # The hardware rehearsal protocol, and the EC default. A perfect sensor
    # reading is not an achievable operating point, so a noise-free rehearsal
    # measures an idealization the robot will never see. Noise is drawn from a
    # seeded generator, so the run stays bit-identical under host load.
    "ec_latent_rehearsal_v1": _protocol(
        "ec_latent_rehearsal_v1",
        backend="ec_mujoco",
        termination_profile="fall_only",
        active=(FALL_TERMINATION_NAME, "reference_finished"),
        disabled=TRACKING_TERMINATION_NAMES + (FOOT_TRACKING_TERMINATION_NAME,),
        success_definition="no_base_too_low_termination",
        randomization="none",
        observation_corruption=True,
        observation_noise=SONIC_OBSERVATION_NOISE,
    ),
    # Same rollout as `ec_latent_rehearsal_v1` — the episode still ends only on
    # a fall or on the reference running out — but success is judged by the
    # released SONIC thresholds the 4,096 scoreboard uses (anchor height and
    # end-effector height 0.25 m, squared anchor orientation error 1.0 rad^2),
    # which Embodied-Control already evaluates per rollout. Fall-free rate on
    # the ten-motion board was measured NOT to track scoreboard success rate
    # (Spearman -0.24 over eight arms); this success definition is what makes
    # the CPU screen's survival axis mean the same thing as the scoreboard's.
    "ec_sonic_rehearsal_v1": _protocol(
        "ec_sonic_rehearsal_v1",
        backend="ec_mujoco",
        termination_profile="fall_only",
        active=(FALL_TERMINATION_NAME, "reference_finished"),
        disabled=TRACKING_TERMINATION_NAMES + (FOOT_TRACKING_TERMINATION_NAME,),
        success_definition="sonic_release_thresholds_over_complete_motion",
        randomization="none",
        observation_corruption=True,
        observation_noise=SONIC_OBSERVATION_NOISE,
    ),
}

# --------------------------------------------------------------------------- #
# The stratified 64-motion screen board (2026-08-17)
# --------------------------------------------------------------------------- #
#
# The ten-motion language board tracks the 4,096-motion scoreboard on quality
# (Spearman +0.69 on success-only MPJPE-L over eight arms) and NOT AT ALL on
# survival (-0.24 on success rate). The cause is structural: 3,545 of the 4,096
# scoreboard motions are passed by every arm, so a small uniform sample is
# nearly all easy motions and carries no survival signal.
#
# This board fixes that by over-sampling failures. Each of the 4,096 held-out
# motions was labelled with how many of the eight 2026-08-15 latent-bottleneck
# arms fail it (0-8), and seven or eight motions were drawn from each label.
# Reported numbers must be re-weighted by `population_weight`, which carries
# each bucket's share of the 4,096: over the eight arms the weighted estimate
# reproduces the full board's success rate to 0.7 points and ranks the arms at
# Spearman +0.95 on success rate and +0.91 on success-only MPJPE-L. The RAW
# board mean is not comparable with a scoreboard number and ranks arms at only
# +0.55, so publish the weighted estimate.
#
# The strata come from those same eight arms, so the board is mildly fitted to
# them; refresh the labels when a materially different arm family lands.
#
# Motions live in `data/bones_seed_strat64_v1` (manifest sha 224f561c),
# built from the same NPZ tree as the scoreboard's reference arrays. Board rank
# is the position in that tree; the scoreboard rank is kept as provenance.

# The paper-facing hardware-plausible board: 123 clips drawn from the canonical
# 4,096 block by `evaluation.clip_features.select_deployable_ranks(count=123,
# seed=20260817)` under `DEPLOYABLE_CLIP_RULE_V1`. It exists so our headline is
# comparable with SONIC's own headline, which is its 123-clip HARDWARE
# deployment set scored in simulation (22.3 mm MPJPE-L, 100% success), not a
# large held-out benchmark. The rule reads reference kinematics only, so no
# checkpoint influenced the selection; see that module for the held-out
# validation. The list is frozen: regenerate it only by bumping the rule.
DEPLOYABLE123_MOTIONS: tuple[tuple[int, str], ...] = (
    (12321, "big_heavy_one_hand_right_side_high_to_front_high_R_001_A525"),
    (12390, "medium_big_heavy_one_hand_front_medium_to_front_high_R_001_A520"),
    (12420, "high_lever_switch_over_back_001_A481_M"),
    (12451, "turn_start_walk_135_001_A038_M"),
    (12477, "injured_torso_jog_ff_stop_360_R_004_A233"),
    (
        12514,
        "medium_big_light_two_hands_right_side_low_to_right_side_medium_R_001_A531",
    ),
    (12531, "itching_neck_R_005_A101_M"),
    (12546, "injured_R_leg_idle_turn_360_003_A173_M"),
    (12564, "turn_lift_crate_walk_360_001_A144"),
    (12567, "omg_R_002_A059"),
    (12599, "looking_around_on_ground_003_A098_M"),
    (12616, "walk_arc_cw_start_005_A058_M"),
    (12617, "walk_ff_start_360_002_A054"),
    (12638, "cutting_masterchiefstyle_R_002_A298"),
    (12656, "brush_off_dust_003_A234_M"),
    (12715, "idle_right_to_idle_002_A096_M"),
    (12722, "itching_left_forearm_R_003_A055"),
    (12751, "walk_the_dog_ff_045_pull_back_leash_R_001_A494_M"),
    (12824, "lift_crate_walk_ff_stop_315_002_A144"),
    (12849, "pounding_meat_R_004_A297_M"),
    (12867, "yawn_R_002_A408"),
    (12870, "injured_torso_walk_ff_start_270_R_001_A229"),
    (12909, "listening_R_002_A168_M"),
    (12913, "injured_torso_walk_ff_stop_180_R_002_A233"),
    (12924, "on_the_edge_001_A274"),
    (12932, "medium_big_light_two_hands_walk_ff_stop_270_R_001_A511_M"),
    (12942, "take_a_sip_270_start_R_001_A550"),
    (12989, "walk_ff_loop_270_002_A068_M"),
    (13012, "medium_big_heavy_one_hand_front_medium_to_right_side_medium_R_001_A522"),
    (13063, "walk_sideway_090_stop_003_A040_M"),
    (13131, "change_idle_right_to_idle_002_A035"),
    (13156, "lift_crate_walk_ff_stop_225_002_A160_M"),
    (13178, "walk_backward_start_001_A035_M"),
    (13193, "walk_ff_stop_315_002_A093_M"),
    (13223, "rubbing_eyes_fist_006_A259"),
    (13274, "change_idle_to_idle_right_002_A025_M"),
    (13403, "legs_relax_002_A099"),
    (13429, "walk_ff_start_180_R_003_A093_M"),
    (13553, "freezing_cold_001_A141"),
    (13558, "turn_start_walk_000_001_A041_M"),
    (13576, "itching_right_thigh_R_003_A053"),
    (13585, "injured_R_leg_walk_ff_loop_225_003_A171"),
    (13741, "screaming_002_A167"),
    (13758, "medium_big_light_two_hands_behind_medium_to_behind_high_R_002_A530"),
    (13789, "walk_hands_on_back_start_002_A181_M"),
    (13837, "medium_big_light_one_hand_right_side_high_to_behind_high_R_001_A526_M"),
    (13919, "walk_ff_loop_180_R_001_A054_M"),
    (13937, "injured_R_leg_walk_ff_start_180_003_A171"),
    (14076, "crouch_ff_start_315_R_003_A244"),
    (14080, "street_avoid_obstacle_180_walk_R_003_A430"),
    (14138, "walk_ff_start_315_002_A069_M"),
    (14151, "reach_jump_R_001_A188_M"),
    (14173, "medium_big_light_one_hand_pick_up_front_medium_R_003_A511_M"),
    (14230, "jump_right_004_A031"),
    (14253, "looking_around_on_ground_003_A056"),
    (14294, "high_small_crank_over_cw_001_A484_M"),
    (14305, "reaching_far_R_003_A143"),
    (14373, "walk_arc_cw_start_R_very_slow_001_A444_M"),
    (14381, "walk_backward_stop_001_A029_M"),
    (14426, "low_vertical_handle_vertical_lever_down_001_A485"),
    (14447, "take_a_sip_360_stop_R_001_A552_M"),
    (14472, "medium_heavy_two_hands_idle_turn_270_R_002_A506_M"),
    (14482, "change_idle_to_idle_left_004_A036_M"),
    (14533, "looking_R_003_A119"),
    (14553, "itching_head_003_A038"),
    (14582, "walk_forward_stop_001_A038"),
    (14596, "medium_big_heavy_two_hands_pick_up_right_side_low_R_001_A504_M"),
    (14597, "legs_relax_004_A238"),
    (14619, "dancing_routine_2_003_A073_M"),
    (14626, "itching_left_arm_R_002_A046_M"),
    (14642, "door_knob_right_side_peep_R_001_A513_M"),
    (14676, "fixing_something_003_A405_M"),
    (14700, "turn_walk_270_R_001_A236"),
    (14761, "injured_R_leg_jump_ff_270_R_003_A335_M"),
    (14810, "walk_ff_stop_225_002_A093_M"),
    (14839, "object_looking_at_R_003_A351_M"),
    (14907, "injured_R_leg_walk_ff_loop_315_R_002_A230_M"),
    (14915, "jump_ff_180_R_002_A065_M"),
    (14935, "jump_sideway_045_003_A030"),
    (14968, "jump_ff_360_003_A049"),
    (14991, "street_watching_carcrash_standing_180_R_001_A429"),
    (15068, "triumph_one_handed_R_001_A097_M"),
    (15128, "alone_009_A099_M"),
    (15193, "pocket_searching_001_A259"),
    (15211, "injured_torso_jog_ff_stop_360_R_002_A217_M"),
    (15216, "sweep_floor_side_walk_R_003_A296_M"),
    (15219, "clearing_ear_R_003_A166_M"),
    (15257, "chefs_kiss_R_002_A269"),
    (15342, "walk_sideway_045_stop_002_A033_M"),
    (15348, "no_see_001_A168_M"),
    (15394, "don_t_know_2_003_A124"),
    (15395, "big_heavy_one_hand_walk_ff_stop_270_R_002_A510_M"),
    (15412, "prepare_knuckles_R_001_A457"),
    (15432, "look_225_R_002_A454"),
    (15453, "small_heavy_one_hand_walk_ff_start_360_R_001_A507_M"),
    (15465, "injured_R_leg_jog_ff_loop_270_R_003_A229"),
    (15475, "confusion_001_A055"),
    (15484, "turn_lift_crate_walk_360_002_A140"),
    (15485, "reaching_up_R_004_A047_M"),
    (15491, "injured_R_leg_idle_turn_270_004_A169"),
    (15588, "walk_sideway_135_stop_001_A022"),
    (15665, "walk_forward_loop_002_A036_M"),
    (15681, "walk_sideway_090_stop_001_A033_M"),
    (15750, "itching_left_body_side_R_003_A122"),
    (15795, "show_bicep_001_A043_M"),
    (15797, "itching_right_thigh_R_001_A100_M"),
    (15808, "walk_hands_on_back_loop_002_A038"),
    (15810, "itching_head_R_002_A051"),
    (15811, "clap_forced_001_A068_M"),
    (15824, "wall_leaning_idle_to_idle_270_R_003_A289_M"),
    (15879, "show_bicep_R_002_A272_M"),
    (15923, "small_light_two_hands_behind_high_to_right_side_high_R_001_A526"),
    (16046, "rubbing_hands_003_A261"),
    (16157, "idle_to_idle_right_R_001_A256_M"),
    (16168, "walk_ff_stop_225_004_A147_M"),
    (16170, "jump_ff_270_003_A185_M"),
    (16252, "walk_ff_start_315_002_A053_M"),
    (16266, "no_hear_001_A184_M"),
    (16273, "idle_hands_on_back_start_002_A236_M"),
    (16313, "walk_ff_stop_360_R_002_A267"),
    (16322, "inside_door_handle_right_side_close_R_002_A515"),
    (16349, "small_light_two_hands_walk_ff_stop_180_R_001_A504"),
    (16356, "clap_enthusiastic_003_A042"),
)


# (board rank, motion, source scoreboard rank, arms failing of eight)
STRAT64_MOTIONS: tuple[tuple[int, str, int, int], ...] = (
    (0, "inj_torso_idle_turn_360_002_A104", 12446, 0),
    (1, "kneeling_start_002_A023", 12453, 4),
    (2, "dance_hiphop_indiana_step_R_fast_002_A320_M", 12474, 6),
    (3, "horse_riding_R_003_A438", 12512, 0),
    (4, "painful_stand_on_jog_ff_270_R_001_A461", 12548, 4),
    (5, "injured_torso_jog_ff_stop_180_R_003_A218_M", 12584, 0),
    (6, "body_search_002_A058_M", 12640, 5),
    (7, "crawl_ff_loop_225_001_A133_M", 12695, 1),
    (8, "dance_basic_chaines_180_R_001_A308_M", 12752, 1),
    (9, "dance_basic_cross_turn_360_R_002_A307", 12846, 3),
    (10, "dance_hiphop_wutang_R_fast_002_A318_M", 12852, 6),
    (11, "on_the_edge_002_A238_M", 12866, 4),
    (12, "mohak_forward_stop_003_A034_M", 12981, 6),
    (13, "high_jump_R_001_A395", 13045, 8),
    (14, "victory_dance_wednesday_dance_R_003_A307_M", 13073, 2),
    (15, "mohak_turn_270_003_A129", 13076, 1),
    (16, "kneeling_loop_003_A034_M", 13090, 6),
    (17, "exercise_2_002_A123_M", 13091, 5),
    (18, "high_jump_R_002_A405", 13188, 8),
    (19, "mohak_backward_loop_003_A121", 13226, 4),
    (20, "dance_retro_up_down_step_R_005_A312_M", 13291, 2),
    (21, "dance_hiphop_bart_simpson_R_loop_fast_002_A323", 13294, 5),
    (22, "exercise_1_002_A121_M", 13415, 2),
    (23, "kneeling_start_001_A187_M", 13511, 2),
    (24, "jog_arc_cw_start_R_turn_jump_270_R_crawl_idle_R_002_A476_M", 13619, 7),
    (25, "mohak_loop_002_A029_M", 13675, 7),
    (26, "jump_ff_271_A252", 13891, 0),
    (27, "reach_jump_R_103_A414_M", 14024, 8),
    (28, "mohak_backward_loop_003_A126_M", 14030, 3),
    (29, "high_jump_R_002_A312", 14056, 8),
    (
        30,
        "dance_hiphop_point_right_left_side_side_strong_men_R_loop_fast_002_A324",
        14413,
        3,
    ),
    (31, "reach_jump_R_002_A292", 14527, 1),
    (32, "dance_hiphop_bart_simpson_R_fast_001_A317_M", 14579, 3),
    (33, "jog_sideway_right_loop_001_A030_M", 14585, 5),
    (34, "idle_crawl_start_001_A218", 14710, 7),
    (35, "change_idle_to_idle_left_003_A028", 14721, 0),
    (36, "mohak_turn_045_001_A030", 14754, 6),
    (37, "crouch_ff_start_225_003_A149", 14848, 4),
    (38, "shadow_boxing_R_001_A360_M", 14857, 1),
    (39, "dance_hiphop_running_man_4_directions_R_fast_001_A319", 14879, 6),
    (40, "crouch_ff_start_180_R_003_A148", 15012, 3),
    (41, "high_jump_R_001_A354_M", 15016, 8),
    (42, "crouch_ff_loop_180_R_003_A149_M", 15035, 7),
    (43, "dance_basic_turn_v2_270_R_003_A310", 15053, 1),
    (44, "jog_avoid_bump_spin_270_R_003_A165_M", 15087, 7),
    (45, "kneeling_stop_001_A030_M", 15135, 8),
    (46, "reach_jump_R_003_A123_M", 15299, 7),
    (47, "walk_sideway_090_start_002_A035_M", 15343, 0),
    (48, "rock_out_002_A487_M", 15372, 5),
    (49, "kneeling_loop_003_A407_M", 15413, 8),
    (50, "jog_ff_start_180_R_001_A304_M", 15541, 0),
    (51, "reach_jump_R_002_A351", 15691, 3),
    (52, "dance_latino_mambo_180_mambo_360_R_001_A315", 15853, 4),
    (53, "high_jump_ff_180_R_opt_2_001_A477_M", 15895, 5),
    (54, "body_check_005_A139_M", 15903, 2),
    (55, "mohak_backward_stop_003_A128", 15904, 6),
    (56, "krakowiak_R_003_A407_M", 15921, 3),
    (57, "dance_hiphop_bounce_360_R_fast_002_A320", 15947, 1),
    (58, "toosie_slide_007_A466", 15966, 2),
    (59, "walk_sideway_045_stop_005_A023_M", 15972, 0),
    (60, "dance_hiphop_hip_hop_ii_R_004_A313_M", 15990, 7),
    (61, "inj_right_leg_stoop_down_R_001_A076_M", 16183, 4),
    (62, "crouch_ff_start_270_R_002_A245", 16205, 2),
    (63, "exercise_2_003_A051_M", 16336, 5),
)

# How many of the 4,096 scoreboard motions carry each failure count.
STRAT64_BUCKET_POPULATION: dict[int, int] = {
    0: 3545,
    1: 124,
    2: 72,
    3: 57,
    4: 29,
    5: 28,
    6: 27,
    7: 38,
    8: 176,
}


def make_strat64_board(board_id: str, *, repeats: int = 3) -> EvalBoardV1:
    """Build the stratified screen board with population weights attached."""
    total = sum(STRAT64_BUCKET_POPULATION.values())
    sampled: dict[int, int] = {}
    for _, _, _, failing in STRAT64_MOTIONS:
        sampled[failing] = sampled.get(failing, 0) + 1
    cases = []
    for rank, motion, _source_rank, failing in STRAT64_MOTIONS:
        weight = (STRAT64_BUCKET_POPULATION[failing] / total) / sampled[failing]
        for repeat in range(int(repeats)):
            cases.append(
                EvalEpisodeCaseV1(
                    trajectory_rank=rank,
                    start_frame=0,
                    env_seed=0,
                    repeat_index=repeat,
                    motion_name=motion,
                    population_weight=weight,
                )
            )
    return EvalBoardV1(board_id=board_id, cases=tuple(cases))


BOARDS: dict[str, EvalBoardV1] = {
    "bones_milestone256_v1": make_rank_board(
        "bones_milestone256_v1", (12288 + 16 * index for index in range(256))
    ),
    "bones_scoreboard4096_v1": make_rank_board(
        "bones_scoreboard4096_v1", range(12288, 16384)
    ),
    "selected10_v1": make_rank_board("selected10_v1", range(10)),
    "selected10_repeats3_v1": make_rank_board(
        "selected10_repeats3_v1", range(10), repeats=3
    ),
    # Under sensor noise each repeat is an independent draw, so repeats are
    # samples rather than a reproducibility check. Five keeps the per-motion
    # spread measurable while the whole board still runs in about a minute.
    "selected10_repeats5_v1": make_rank_board(
        "selected10_repeats5_v1", range(10), repeats=5
    ),
    "gr00t28x20_v1": make_rank_board("gr00t28x20_v1", range(28), repeats=20),
    # 64 motions drawn from the scoreboard ranks, stratified by difficulty, so
    # a one-minute CPU board can say something about falls. Read it through
    # `population_weight`; see STRAT64_MOTIONS above.
    "ec_strat64_v1": make_strat64_board("ec_strat64_v1", repeats=3),
    "cross_backend64_v1": make_rank_board(
        "cross_backend64_v1", (12288 + 64 * index for index in range(64))
    ),
    # The falsification partner of `bones_scoreboard4096_v1`: a disjoint block
    # of the same size. The released SONIC checkpoint scores 25.86 mm on it
    # against 25.90 mm on the canonical block, so block-to-block population
    # noise on the full-block figure is under 0.1 mm.
    "bones_heldout4096_v1": make_rank_board(
        "bones_heldout4096_v1", range(20480, 24576)
    ),
    # See DEPLOYABLE123_MOTIONS.
    "bones_deployable123_v1": make_rank_board(
        "bones_deployable123_v1", (rank for rank, _ in DEPLOYABLE123_MOTIONS)
    ),
}


def make_profile(
    profile_id: str,
    *,
    evaluator: str,
    protocol: EvalProtocolV1,
    board: EvalBoardV1,
    description: str = "",
) -> EvalProfileV1:
    return EvalProfileV1(
        profile_id=profile_id,
        evaluator=evaluator,
        protocol_id=protocol.protocol_id,
        protocol_hash=protocol.content_hash(),
        board_id=board.board_id,
        board_hash=board.content_hash(),
        description=description,
    )


PROFILES: dict[str, EvalProfileV1] = {
    "milestone_fall_only_v1": make_profile(
        "milestone_fall_only_v1",
        evaluator="evaluate_checkpoint",
        protocol=PROTOCOLS["tracker_fall_only_v1"],
        board=BOARDS["bones_milestone256_v1"],
    ),
    "milestone_full_horizon_v1": make_profile(
        "milestone_full_horizon_v1",
        evaluator="evaluate_checkpoint",
        protocol=PROTOCOLS["tracker_full_horizon_v1"],
        board=BOARDS["bones_milestone256_v1"],
    ),
    "deployment_ec_v1": make_profile(
        "deployment_ec_v1",
        evaluator="ec_latent_playground",
        protocol=PROTOCOLS["ec_latent_rehearsal_v1"],
        board=BOARDS["selected10_repeats5_v1"],
    ),
    # The frequent per-checkpoint CPU sidecar. Every repeat draws a fresh noise
    # stream, so the five repeats are samples of the rehearsal distribution,
    # not a reproducibility check — the per-motion spread they produce is the
    # quantity that says whether a checkpoint difference is real.
    "sidecar_ec_v1": make_profile(
        "sidecar_ec_v1",
        evaluator="ec_latent_playground",
        protocol=PROTOCOLS["ec_latent_rehearsal_v1"],
        board=BOARDS["selected10_repeats5_v1"],
    ),
    # The screen that replaces `sidecar_ec_v1` for promotion decisions: 64
    # difficulty-stratified motions x 3 noise draws = 192 episodes, judged on
    # the SONIC thresholds and re-weighted to the population. Its reference
    # tree is `data/bones_seed_strat64_v1`, not the ten-motion language set.
    # ---- Paper-facing rows. Cite these three, in this order. ----
    # Headline quality + success on the hardware-plausible population. This is
    # the row comparable with SONIC's own headline of 22.3 mm at 100% success.
    "paper_deployable123_v1": make_profile(
        "paper_deployable123_v1",
        evaluator="evaluate_checkpoint",
        protocol=PROTOCOLS["sonic_sr_clean_v1"],
        board=BOARDS["bones_deployable123_v1"],
    ),
    # Breadth. The 4,096-clip block is the analogue of SONIC's large held-out
    # sets (test-content 98.7% / 23.2 mm), NOT of its 22.3 mm headline: it
    # includes deep-crouch and ground clips that no hardware set contains, and
    # those clips alone move the figure by about 3.7 mm.
    "paper_scoreboard4096_v1": make_profile(
        "paper_scoreboard4096_v1",
        evaluator="evaluate_checkpoint",
        protocol=PROTOCOLS["sonic_sr_clean_v1"],
        board=BOARDS["bones_scoreboard4096_v1"],
    ),
    # Robustness. Identical board, startup and reset randomization on. Every
    # pre-2026-08-17 scoreboard row in this repo was measured here.
    "paper_scoreboard4096_robust_v1": make_profile(
        "paper_scoreboard4096_robust_v1",
        evaluator="evaluate_checkpoint",
        protocol=PROTOCOLS["sonic_sr_v1"],
        board=BOARDS["bones_scoreboard4096_v1"],
    ),
    "sidecar_ec_strat64_v1": make_profile(
        "sidecar_ec_strat64_v1",
        evaluator="ec_latent_playground",
        protocol=PROTOCOLS["ec_sonic_rehearsal_v1"],
        board=BOARDS["ec_strat64_v1"],
    ),
}


def _single_seed_and_start(evaluator: str, board: EvalBoardV1) -> tuple[int, int]:
    unsupported = []
    if len(board.seeds) != 1:
        unsupported.append("board.env_seed")
    if len(board.start_frames) != 1:
        unsupported.append("board.start_frame")
    if unsupported:
        raise ProtocolUnsupported(evaluator, unsupported)
    return board.seeds[0], board.start_frames[0]


def evaluate_checkpoint_argv(
    protocol: EvalProtocolV1,
    board: EvalBoardV1,
    *,
    checkpoint: str | Path,
    output_json: str | Path,
) -> list[str]:
    """Translate one supported profile to ``evaluate_checkpoint`` arguments."""

    evaluator = "evaluate_checkpoint"
    unsupported = []
    if protocol.metric_interval != 1:
        unsupported.append("metric_interval")
    if protocol.start_mode != "board":
        unsupported.append("start_mode")
    if protocol.backend not in {"newton_mjwarp", "physx"}:
        unsupported.append("backend")
    if unsupported:
        raise ProtocolUnsupported(evaluator, unsupported)
    seed, start_frame = _single_seed_and_start(evaluator, board)
    args = [
        "--checkpoint",
        str(checkpoint),
        "--output_json",
        str(output_json),
        "--num_envs",
        str(len(board.cases)),
        "--steps",
        str(protocol.outer_safety_cap_steps),
        "--seed",
        str(seed),
        "--reference_start_frame",
        str(start_frame),
        "--randomization",
        protocol.randomization_profile,
        "--action_sampling",
        protocol.action_sampling,
        "--trajectory_ranks",
        *(str(rank) for rank in board.trajectory_ranks),
        f"physics={protocol.backend}",
    ]
    if protocol.termination_profile == "full_horizon":
        args.append("--disable_early_terminations")
    elif protocol.termination_profile == "fall_only":
        args.extend(_FALL_ONLY_OVERRIDES)
    elif protocol.termination_profile == "sonic":
        args.extend(_SONIC_OVERRIDES)
    elif protocol.termination_profile != "task_strict":
        raise ProtocolUnsupported(evaluator, ["termination_profile"])
    return args


def eval_skill_commander_argv(
    protocol: EvalProtocolV1,
    board: EvalBoardV1,
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Translate one supported profile to SkillCommander evaluator arguments."""

    evaluator = "eval_skill_commander_closed_loop"
    unsupported = []
    if protocol.action_sampling != "mode":
        unsupported.append("action_sampling")
    if protocol.start_mode != "board":
        unsupported.append("start_mode")
    if protocol.backend not in {"newton_mjwarp", "physx"}:
        unsupported.append("backend")
    if unsupported:
        raise ProtocolUnsupported(evaluator, unsupported)
    seed, start_frame = _single_seed_and_start(evaluator, board)
    if start_frame != 0:
        raise ProtocolUnsupported(evaluator, ["board.start_frame"])
    args = [
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
        "--num_envs",
        str(len(board.cases)),
        "--max_steps",
        str(protocol.outer_safety_cap_steps),
        "--metric_interval",
        str(protocol.metric_interval),
        "--seed",
        str(seed),
        "--trajectory_ranks",
        *(str(rank) for rank in board.trajectory_ranks),
        f"physics={protocol.backend}",
    ]
    if protocol.randomization_profile == "none":
        args.append("--deterministic_tracking")
    elif protocol.randomization_profile == "no_push":
        args.append("--disable_push_event")
    elif protocol.randomization_profile != "all":
        raise ProtocolUnsupported(evaluator, ["randomization_profile"])
    if protocol.termination_profile == "fall_only":
        args.extend(("--fall_only_success", "--disable_tracking_terminations"))
    elif protocol.termination_profile == "full_horizon":
        pass
    elif protocol.termination_profile == "sonic":
        args.append("--sonic_success_terminations")
    elif protocol.termination_profile == "task_strict":
        args.append("--keep_early_terminations")
    else:
        raise ProtocolUnsupported(evaluator, ["termination_profile"])
    return args


__all__ = [
    "BOARDS",
    "FALL_TERMINATION_NAME",
    "FOOT_TRACKING_TERMINATION_NAME",
    "G1_TRACKED_BODY_NAMES",
    "PROFILES",
    "PROTOCOLS",
    "TRACKING_TERMINATION_NAMES",
    "EvalBoardV1",
    "EvalEpisodeCaseV1",
    "EvalProfileV1",
    "EvalProtocolV1",
    "ProtocolMismatch",
    "ProtocolUnsupported",
    "TrackerEvalContractV1",
    "canonical_json",
    "content_hash",
    "eval_skill_commander_argv",
    "evaluate_checkpoint_argv",
    "frozen_pairs",
    "make_profile",
    "make_rank_board",
    "unpinned_protocol_stamp",
    "validate_realized_protocol",
]
