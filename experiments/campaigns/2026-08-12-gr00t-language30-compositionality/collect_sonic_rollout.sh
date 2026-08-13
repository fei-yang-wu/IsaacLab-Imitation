#!/usr/bin/env bash
# SONIC-style planner data: the RELEASED NVIDIA SONIC v1.1 tracker drives the
# robot from its own encoder, and every control step is recorded as one planner
# training row.
#
# Why one row per control step rather than per publication: the released
# tracker re-encodes its 64-D FSQ token every 50 Hz step (hold 1). A planner
# that replaces that encoder must therefore supply one latent per step, so it
# predicts 30 consecutive latents per publication instead of 3 latents each
# held for 10 steps. `prepare_gr00t_dataset.py` builds that target by joining
# rows at `control_step + k` for k in 0..29, which only exists if every step
# was written.
#
# The stored `z_target` is the encoder's CONTINUOUS pre-FSQ latent. The
# quantizer is a fixed, parameter-free lattice (tanh, round, divide by 16), so
# snapping at publication reproduces the exact value the tracker would have
# seen while keeping the regression target continuous. Pass
# `--sample_target post_quantization` for the snapped-target ablation.
#
# Usage: collect_sonic_rollout.sh [extra args...]
#        SMOKE=1 collect_sonic_rollout.sh   # 30 envs, 30 steps, throwaway root
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

SONIC_CHECKPOINT="${SONIC_CHECKPOINT:-${HOME}/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/blobs/af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43}"
[ -f "${SONIC_CHECKPOINT}" ] || {
    echo "missing SONIC v1.1 checkpoint: ${SONIC_CHECKPOINT}" >&2
    exit 1
}

DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
NUM_GOALS=30
# Trajectories per motion. `build_env_rank_assignment` splits the environments
# into contiguous equal blocks, so num_envs must divide evenly by the goal
# count or it refuses.
PER_GOAL="${PER_GOAL:-15}"
NUM_ENVS=$(( NUM_GOALS * PER_GOAL ))
# The longest of the 30 motions needs about 2000 control steps to reach
# `reference_finished`; a shorter cap silently truncates it.
STEPS="${STEPS:-2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/gr00t_language30_sonic_hold1/collection}"

if [ "${SMOKE:-0}" = "1" ]; then
    NUM_ENVS=30
    STEPS=30
    OUTPUT_ROOT="${REPO_ROOT}/logs/gr00t_language30_sonic_hold1/smoke"
    rm -rf "${OUTPUT_ROOT}"
fi
if [ -e "${OUTPUT_ROOT}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_ROOT}" >&2
    exit 1
fi
RANKS=($(seq 0 $(( NUM_GOALS - 1 ))))

pixi run -e isaaclab python -m imitation_experiments.lowlevel.evaluate_sonic_release \
    --headless --task Isaac-Imitation-G1-v2 \
    --sonic_checkpoint "${SONIC_CHECKPOINT}" --sonic_version v1_1 \
    --label sonic_hold1_collection \
    --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed "${SEED:-0}" \
    --trajectory_ranks "${RANKS[@]}" --reference_start_frame 0 \
    --randomization no_push \
    --save_rollout_training_samples \
    --sample_output_dir "${OUTPUT_ROOT}" \
    --sample_target "${SAMPLE_TARGET:-pre_quantization}" \
    --sample_rows_per_file 8192 --state_history_steps 9 \
    --sample_future_window_frames "${FUTURE_FRAMES:-0}" \
    --output_json "${OUTPUT_ROOT}/summary.json" \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    "$@"

echo "retained: ${OUTPUT_ROOT}"
