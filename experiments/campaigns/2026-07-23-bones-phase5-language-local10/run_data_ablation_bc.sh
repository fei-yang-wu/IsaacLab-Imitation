#!/usr/bin/env bash
set -euo pipefail

# Data-composition ablation, arms B and C, reusing arm A's shared pools.
#
# Arm A (run separately via run.sh at MODEL_SIZE=large): demo(oracle) +
#   planner-rollout(DAgger). Produces the shared demo pool, the pretrained-large
#   planner, and the planner-rollout pool.
# Arm B (here): demo + oracle-rollout                  (no planner-driven data)
# Arm C (here): demo + oracle-rollout + planner-rollout (all three)
#
# The oracle-rollout pool is a second oracle-driven (command_source=hl_skill)
# collection at a different seed, so it is additional expert-distribution data
# rather than the exact demo samples. All arms share the same large pretrained
# planner init and are scored by the trusted eval_skill_commander_closed_loop.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
IMPL="experiments/paper/interface_baselines"

SEED="${SEED:-0}"
ORACLE_ROLLOUT_SEED="${ORACLE_ROLLOUT_SEED:-1}"
ROWS_PER_MOTION="${ROWS_PER_MOTION:-1000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-4000}"
MODEL_SIZE="${MODEL_SIZE:-large}"
EVAL_STEPS="${EVAL_STEPS:-500}"

ARM_A_ROOT="${ARM_A_ROOT:-logs/interface_baselines/bones_seed_phase5_local10_large_A_dagger_demo_seed0}"
BC_ROOT="${BC_ROOT:-logs/interface_baselines/bones_seed_phase5_local10_large_ablation_seed0}"

LOW="${LOW:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt}"
SKILL="${SKILL:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
LANG="${LANG:-data/bones_seed_phase5_corrected/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
MANIFEST="${MANIFEST:-data/bones_seed_phase5_local10/manifests/g1_bones_seed_phase5_local10_manifest.json}"
LATENT_DS="${LATENT_DS:-data/bones_seed_phase5_local10/zarr/latent_seed0}"

DEMO_POOL="${ARM_A_ROOT}/latent_skill/demonstration_samples"
PRETRAINED="${ARM_A_ROOT}/latent_skill/planner_pretrain_demonstration/checkpoints/latest.pt"
PLANNER_ROLLOUT_POOL="${ARM_A_ROOT}/latent_skill/planner_rollout_samples"

GOALS=(Neutral_stoop_down_001_A057 avoid_bump_let_go_R_003_A460 axe_cutting_tree_horizontal_R_004_A355 big_heavy_two_hands_front_high_to_front_high_R_001_A524 big_light_two_hands_pick_up_front_medium_R_001_A509 body_check_001_A180 burning_loop_R_001_A528 casual_greeting_R_001_A428 cellphone_typing_sequence_one_hand_idle_R_001_A423 cough_tuberculosis_R_001_A500)

for p in "${DEMO_POOL}" "${PLANNER_ROLLOUT_POOL}"; do
    [[ -d "${p}" ]] || { echo "[ERROR] Arm A pool missing (is arm A finished?): ${p}" >&2; exit 2; }
done
[[ -f "${PRETRAINED}" ]] || { echo "[ERROR] Arm A pretrained planner missing: ${PRETRAINED}" >&2; exit 2; }

PYX=(pixi run python)
IPX=(pixi run -e isaaclab python)

mkdir -p "${BC_ROOT}"

# --- 1. Oracle-rollout pool (oracle-driven, seed ORACLE_ROLLOUT_SEED) ---------
ORACLE_DIR="${BC_ROOT}/oracle_rollout_batched/latent_skill"
ORACLE_POOL="${ORACLE_DIR}/rollout_training_samples"
if [[ ! -f "${ORACLE_DIR}/summary.json" ]]; then
    echo "[INFO] Collecting oracle-rollout pool (oracle-driven, seed ${ORACLE_ROLLOUT_SEED})..."
    OMNI_KIT_ACCEPT_EULA=YES "${IPX[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --task Isaac-Imitation-G1-Latent-v0 --algorithm IPMD \
        --checkpoint "${LOW}" --skill_checkpoint "${SKILL}" \
        --language_embeddings "${LANG}" --state_history_steps 9 \
        --output_dir "${ORACLE_DIR}" --label bones_seed_oracle_rollout_batched \
        --num_envs 10 --max_steps $((ROWS_PER_MOTION * 10 * 8)) --seed "${ORACLE_ROLLOUT_SEED}" \
        --metric_interval $((ROWS_PER_MOTION * 10 * 8 + 1)) \
        --keep_time_out --allow_random_reset --keep_early_terminations \
        --disable_tracking_terminations --disable_reward_clipping \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 --assert-kitless \
        --motion_names "${GOALS[@]}" \
        --balanced_motion_names "${GOALS[@]}" --balanced_rows_per_motion "${ROWS_PER_MOTION}" \
        --save_rollout_training_samples --continue_after_reset --sample_rows_per_file "${ROWS_PER_MOTION}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL}" \
        agent.logger.backend= agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" "env.dataset_path=${LATENT_DS}" \
        env.reset_schedule=sequential env.wrap_steps=false \
        env.observations.policy.enable_corruption=false env.refresh_zarr_dataset=false \
        env.latent_command_dim=258 agent.ipmd.latent_dim=258 \
        agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
        agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=256 agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 \
        agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0 \
        physics=newton_mjwarp env.sim.physics.solver_cfg.njmax=320 env.sim.physics.solver_cfg.nconmax=40
else
    echo "[SKIP] Oracle-rollout pool already collected: ${ORACLE_DIR}/summary.json"
fi

# --- 2/3/4. Compose, finetune, eval each arm ---------------------------------
eval_one_goal() {  # $1=planner_ckpt  $2=out_dir  $3=goal  $4=label
    local planner="$1" out="$2" goal="$3" label="$4"
    [[ -f "${out}/summary.json" ]] && { echo "[SKIP] eval exists: ${out}"; return 0; }
    OMNI_KIT_ACCEPT_EULA=YES "${IPX[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --task Isaac-Imitation-G1-Latent-v0 --algorithm IPMD \
        --checkpoint "${LOW}" --skill_checkpoint "${SKILL}" \
        --language_embeddings "${LANG}" --state_history_steps 9 \
        --output_dir "${out}" --label "${label}" --num_envs 1 --max_steps "${EVAL_STEPS}" --seed "${SEED}" \
        --metric_interval $((EVAL_STEPS + 1)) \
        --keep_time_out --allow_random_reset --keep_early_terminations \
        --disable_tracking_terminations --disable_reward_clipping \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 --assert-kitless \
        --motion_name "${goal}" --require_goal_motion_match \
        --planner_checkpoint "${planner}" \
        agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${planner}" \
        "agent.ipmd.skill_commander_embeddings_path=${LANG}" \
        "agent.ipmd.skill_commander_goal_name=${goal}" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        agent.ipmd.skill_commander_flow_num_inference_steps=16 \
        agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL}" \
        agent.logger.backend= agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" "env.dataset_path=${LATENT_DS}" \
        env.reset_schedule=sequential env.wrap_steps=false \
        env.observations.policy.enable_corruption=false env.refresh_zarr_dataset=false \
        env.latent_command_dim=258 agent.ipmd.latent_dim=258 \
        agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
        agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=256 agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 \
        agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0 \
        physics=newton_mjwarp env.sim.physics.solver_cfg.njmax=320 env.sim.physics.solver_cfg.nconmax=40
}

run_arm() {  # $1=arm_name  $2=merged_dir  (sources passed after)
    local arm="$1"; shift
    local merged="$1"; shift
    local arm_root="${BC_ROOT}/${arm}"
    local ckpt="${arm_root}/finetune/checkpoints/latest.pt"
    # compose
    if [[ ! -f "${merged}/merge_manifest.json" ]]; then
        local margs=()
        for s in "$@"; do margs+=(--source "${s}" --source_limit 0); done
        "${PYX[@]}" -m imitation_experiments.data.merge_planner_samples --replace_incomplete "${margs[@]}" \
            --seed "${SEED}" --output_dir "${merged}"
    fi
    # finetune (init from shared pretrained-large)
    if [[ ! -f "${ckpt}" ]]; then
        "${PYX[@]}" -m imitation_experiments.planner.train_chunked_transformer_planner \
            --samples_dir "${merged}" --output_dir "${arm_root}/finetune" \
            --interface latent_skill --state_key planner_state --model_size "${MODEL_SIZE}" \
            --seed "${SEED}" --max_samples 0 --num_updates "${FINETUNE_UPDATES}" \
            --batch_size 256 --micro_batch_size 32 --lr 0.0001 --weight_decay 0.0001 \
            --flow_num_inference_steps 16 --endpoint_num_inference_steps 4 --flow_inference_noise_std 0.0 \
            --checkpoint "${PRETRAINED}"
    fi
    # eval per goal
    local i=0
    for g in "${GOALS[@]}"; do
        eval_one_goal "${ckpt}" "${arm_root}/eval/$(printf '%04d' "${i}")_${g}" "${g}" "abl_${arm}_${g}"
        i=$((i + 1))
    done
    echo "[INFO] Arm ${arm} done: ${arm_root}"
}

# Arm B: demo + oracle-rollout
run_arm "B_oracle_demo" "${BC_ROOT}/B_oracle_demo/samples" "${DEMO_POOL}" "${ORACLE_POOL}"
# Arm C: demo + oracle-rollout + planner-rollout
run_arm "C_dagger_oracle_demo" "${BC_ROOT}/C_dagger_oracle_demo/samples" "${DEMO_POOL}" "${ORACLE_POOL}" "${PLANNER_ROLLOUT_POOL}"

# --- 5. Summary table across arms (A from its own run, B/C here) --------------
"${PYX[@]}" - "$ARM_A_ROOT" "$BC_ROOT" <<'PY'
import json, sys, glob, os
import statistics as st
arm_a_root, bc_root = sys.argv[1], sys.argv[2]
def load_eval_dir(d):
    rows=[]
    for f in sorted(glob.glob(os.path.join(d, "*", "summary.json"))):
        s=json.load(open(f))
        rows.append(s)
    return rows
def agg(rows):
    def num(vals):
        vals=[v for v in vals if isinstance(v,(int,float)) and v==v]
        return st.mean(vals) if vals else float("nan")
    succ=num([r.get("aggregate",{}).get("tracking_success_rate") for r in rows])
    surv=num([r.get("aggregate",{}).get("survival_steps_mean") for r in rows])
    mpjpe=num([r.get("metrics",{}).get("tracking_mpjpe_mm",{}).get("mean") for r in rows])
    return (succ, surv, mpjpe, len(rows))
arms={
 "A_dagger_demo": os.path.join(arm_a_root, "latent_skill", "eval_finetuned_per_goal"),
 "B_oracle_demo": os.path.join(bc_root, "B_oracle_demo", "eval"),
 "C_dagger_oracle_demo": os.path.join(bc_root, "C_dagger_oracle_demo", "eval"),
}
print(f"\n{'arm':<24} {'succ':>6} {'survival':>9} {'mpjpe_mm':>9} {'goals':>6}")
for name, d in arms.items():
    if not os.path.isdir(d):
        print(f"{name:<24} {'(missing)':>6}"); continue
    s,sv,m,n=agg(load_eval_dir(d))
    print(f"{name:<24} {s:>6.2f} {sv:>9.1f} {m:>9.1f} {n:>6}")
out=os.path.join(bc_root, "ablation_summary.txt")
print(f"\n[INFO] arms: A={arms['A_dagger_demo']}\n            B={arms['B_oracle_demo']}\n            C={arms['C_dagger_oracle_demo']}")
PY
echo "[INFO] Data-composition ablation (B/C) complete: ${BC_ROOT}"
