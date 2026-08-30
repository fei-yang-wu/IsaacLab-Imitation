#!/usr/bin/env bash
# Local qualification for the affine spectral arm: the round-4 diffntp_chunk
# pretrain recipe with one variable moved, --diffsr_phi_parameterization.
#
# Usage: pretrain_affine_arm.sh [affine|concat] [extra args]
#   affine (default) -- phi(s,z) = F(s)^T (A z + b) / sqrt(E), affine in z.
#   concat           -- the production parameterization, for a matched control
#                       run when the reference numbers must come from this host.
#
# NUM_UPDATES=... overrides the update budget. Use the same budget for every
# arm you intend to compare. NUM_UPDATES=500 is the smoke run.
set -euo pipefail
PARAMETERIZATION="${1:-affine}"
if [[ $# -gt 0 ]]; then shift; fi
case "${PARAMETERIZATION}" in
    affine | concat) ;;
    *)
        echo "usage: $0 [affine|concat] [extra args]" >&2
        exit 1
        ;;
esac
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

REF_ARRAYS="${REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/linear_closure_affine/${PARAMETERIZATION}/encoder}"
NUM_UPDATES="${NUM_UPDATES:-50000}"

exec pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py \
    --task Isaac-Imitation-G1-v2 \
    --num_envs 16 \
    --seed 0 \
    --device cuda:0 \
    --headless \
    --assert-kitless \
    --logger_backend none \
    --output_dir "${OUTPUT_DIR}" \
    --horizon_steps 10 \
    --encoder_window_mode intermediate \
    --transition_objective jepa_ntp \
    --jepa_loss sigreg_ebm \
    --jepa_ntp_head diff_chunk \
    --diffsr_phi_parameterization "${PARAMETERIZATION}" \
    --latent_mode deterministic \
    --z_dim 256 \
    --encoder_layer_norm \
    --encoder_hidden_dims 2048 1024 512 512 \
    --encoder_activation silu \
    --diffsr_feature_dim 256 \
    --diffsr_embed_dim 1024 \
    --diffsr_g_hidden_dims 1024 1024 512 \
    --diffsr_mu_hidden_dims 1024 1024 512 \
    --batch_size 8192 \
    --num_updates "${NUM_UPDATES}" \
    --log_interval 1000 \
    --eval_batches 4 \
    "$@" \
    physics=newton_mjwarp \
    env.data.manifest=null \
    "env.data.reference_arrays_dir=${REF_ARRAYS}" \
    env.data.persist_id=bones_seed_sonic_full_129785@e714bbff \
    env.data.reference_arrays_resident=false \
    env.data.runtime_cache_device=cpu \
    env.data.macro_cache_device=cuda:0 \
    "env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]" \
    "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]" \
    env.expert_macro_frame_stride=1 \
    env.expert_macro_anchor_mode=robot_heading
