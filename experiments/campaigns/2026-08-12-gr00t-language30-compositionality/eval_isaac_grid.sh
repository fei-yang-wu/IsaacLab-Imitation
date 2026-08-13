#!/usr/bin/env bash
# Isaac closed-loop evaluation of one (arm, route) over ALL 30 goals in a
# single process — the number of record.
#
# Usage: eval_isaac_grid.sh <arm> <latent|chunk_encoded|chunk_native> [extra args...]
#
# One environment per goal, each pinned to its own reference motion, with the
# language goal assigned explicitly at start-up in manifest/rank order. That
# ordering is load-bearing: `--trajectory_ranks` indexes the FULL dataset, not
# the `--motion_names` list, so a hand-picked goal order silently pairs each
# environment with someone else's language. The sampler re-checks the binding
# at every publication and fails loudly on divergence.
#
# Protocol (M3): tracking-error terminations disabled, `base_too_low` active,
# so survival means "finished without falling"; tracking error stays a
# continuous metric. DIAGNOSTIC=1 runs the mandated full-horizon pass instead
# (no early termination, video retained).
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: eval_isaac_grid.sh <arm> <route>}"
ROUTE="${2:?usage: eval_isaac_grid.sh <arm> <route>}"
shift 2

DIAGNOSTIC="${DIAGNOSTIC:-0}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-0}"
# Tracking metrics are INSTANTANEOUS snapshots taken every metric_interval
# steps. Setting the interval to the episode length samples only step 0, where
# the robot still sits on its reset placement and every error is exactly 0 --
# a measurement of the reset, not of tracking. Sample at the planner's
# publication cadence instead.
METRIC_INTERVAL="${METRIC_INTERVAL:-10}"
DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
MANIFEST_LANG="${DATA_ROOT}/manifests/g1_bones_seed_language30_compositionality_v1_manifest_language.json"

# Motions in manifest order == trajectory ranks 0..29 == per-environment goals.
readarray -t MOTIONS < <(
    "${REPO_ROOT}/.pixi/envs/default/bin/python" - "${MANIFEST_LANG}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
motions = manifest["motions"] if "motions" in manifest else manifest
print("\n".join(m["name"] for m in motions))
PY
)
NUM_GOALS="${#MOTIONS[@]}"
RANKS=($(seq 0 $((NUM_GOALS - 1))))
# Single-motion ablation: score one motion at the same total episode count as
# the 30-goal grid, so the two are comparable on sample size. SINGLE_MOTION is
# the manifest RANK, which is what `--trajectory_ranks` indexes.
# EXCLUDE_RANKS drops motions by manifest rank, so an arm trained on a subset
# is scored on exactly that subset.
if [ -n "${EXCLUDE_RANKS:-}" ]; then
    KEEP_M=(); KEEP_R=()
    for i in "${RANKS[@]}"; do
        skip=0
        for x in ${EXCLUDE_RANKS}; do [ "${i}" = "${x}" ] && skip=1; done
        [ "${skip}" = "0" ] && { KEEP_M+=("${MOTIONS[$i]}"); KEEP_R+=("${i}"); }
    done
    MOTIONS=("${KEEP_M[@]}"); RANKS=("${KEEP_R[@]}"); NUM_GOALS="${#MOTIONS[@]}"
fi
if [ -n "${SINGLE_MOTION:-}" ]; then
    MOTIONS=("${MOTIONS[${SINGLE_MOTION}]}")
    RANKS=("${SINGLE_MOTION}")
    NUM_GOALS=1
fi
# Episodes per goal. The head's flow sampler is stochastic, and measured
# run-to-run MPJPE spread is 0.2-6.4% (worst on the replay-trained arms), so a
# single episode per goal is not a stable number. Extra episodes are extra
# ENVIRONMENTS in the same process — goals and reference ranks both cycle — so
# averaging costs GPU memory rather than another simulator start-up.
EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-5}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))
# The environment assigns references in BLOCKS -- env i plays motion
# i / EPISODES_PER_GOAL, not i % NUM_GOALS -- so the goal list must be blocked
# the same way. Verified the hard way: a cycling goal list trips the
# goal/reference guard on env 1. Expand each motion EPISODES_PER_GOAL times.
GOALS=()
for motion in "${MOTIONS[@]}"; do
    for _ in $(seq 1 "${EPISODES_PER_GOAL}"); do
        GOALS+=("${motion}")
    done
done

# The tracker is a property of the ROUTE, not of the head: a latent head and a
# chunk-encoded head must share one tracker for their comparison to isolate the
# interface, while chunk_native drives the explicit tracker.
case "${ROUTE}" in
    latent|chunk_encoded)
        case "${ARM}" in
            fsq64*)
                TRACKER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt"
                ENCODER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt"
                ACTOR_DIM=66; CODE_DIM=64; NJMAX=320
                EXTRA_CFG=(
                    'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
                    'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
                    agent.policy.activation_fn=silu agent.value_function.activation_fn=silu
                )
                ;;
            *)
                TRACKER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt"
                ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
                ACTOR_DIM=258; CODE_DIM=256; NJMAX=289
                EXTRA_CFG=()
                ;;
        esac
        ACTOR_CFG=(env.command_interface.actor.dim="${ACTOR_DIM}")
        ;;
    chunk_native)
        TRACKER="${REPO_ROOT}/logs/bones129k_4096_scoreboard/root_qpos_explicit/model_step_7600078848.pt"
        ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
        CODE_DIM=256; NJMAX=289; ACTOR_DIM=258
        EXTRA_CFG=()
        ACTOR_CFG=(
            env.command_interface.actor=chunk
            env.command_interface.actor.source=external
            env.command_interface.actor.horizon=30
            env.command_interface.actor.hold_steps=10
            'env.command_interface.actor.components=[joint_qpos,root_pos,root_ori]'
        )
        ;;
    *)
        echo "route must be latent|chunk_encoded|chunk_native, got ${ROUTE}" >&2
        exit 1
        ;;
esac

HEAD="${REPO_ROOT}/outputs/gr00t_language30/arms/${ARM}/checkpoints/update_${UPDATE:-0012000}.pt"
GOAL_FEATURES="${REPO_ROOT}/outputs/gr00t_language30/goal_features/goal_features.pt"
for required in "${HEAD}" "${GOAL_FEATURES}" "${TRACKER}" "${ENCODER}"; do
    [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
done

# LABEL_SUFFIX separates protocol variants of the same (arm, route) — a
# different consumption mode, step cap, or ODE-step count — into their own
# retained directories, because the launcher refuses to overwrite one.
LABEL="${ARM}__${ROUTE}${LABEL_SUFFIX:+__${LABEL_SUFFIX}}"
MODE_ARGS=(--disable_tracking_terminations --fall_only_success)
if [ "${VIDEO:-0}" = "1" ]; then
    # Same protocol as the reported table, plus rendering, so the clip shows
    # exactly the run the numbers came from (falls included).
    MODE_ARGS+=(--video)
    LABEL="${LABEL}__video"
fi
if [ "${DIAGNOSTIC}" = "1" ]; then
    # Full-horizon diagnostic: fall detection dropped to floor level so nothing
    # terminates early. Deliberately NOT the reported protocol.
    MODE_ARGS=(--disable_tracking_terminations --fall_only_success --fall_height_m 0.01 --video)
    LABEL="${ARM}__${ROUTE}__diagnostic"
fi
OUTPUT_DIR="${REPO_ROOT}/logs/gr00t_language30_isaac_eval/${LABEL}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    --gr00t_checkpoint "${HEAD}" --gr00t_goal_features "${GOAL_FEATURES}" \
    --gr00t_goals_per_env "${GOALS[@]}" --gr00t_route "${ROUTE}" \
    --gr00t_consumption "${GR00T_CONSUMPTION:-open_loop}" \
    --gr00t_inference_steps "${ODE_STEPS:-4}" \
    --gr00t_samples_per_publication "${SAMPLES:-1}" \
    --gr00t_temporal_ensemble "${ENSEMBLE:-none}" \
    --gr00t_temporal_ensemble_decay "${ENSEMBLE_DECAY:-0.5}" \
    ${CONSUME_SLOTS:+--gr00t_consume_slots ${CONSUME_SLOTS}} \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 --output_dir "${OUTPUT_DIR}" --label "${LABEL}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS}" --seed "${SEED}" \
    --metric_interval "${METRIC_INTERVAL}" \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    "${MODE_ARGS[@]}" \
    physics=newton_mjwarp env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    "${ACTOR_CFG[@]}" \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim="${ACTOR_DIM}" \
    agent.ipmd.latent_steps_min="${HOLD:-10}" agent.ipmd.latent_steps_max="${HOLD:-10}" \
    agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim="${CODE_DIM}" \
    agent.ipmd.latent_learning.code_period=10 \
    env.sim.physics.solver_cfg.njmax="${NJMAX}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    ${EXTRA_CFG[@]+"${EXTRA_CFG[@]}"} \
    "$@"

echo "retained: ${OUTPUT_DIR}"
