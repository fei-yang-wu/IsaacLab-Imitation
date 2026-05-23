#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Isaac-Imitation-G1-v0}"
NUM_ENVS="${NUM_ENVS:-128}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SEEDS_STR="${SEEDS:-2024}"
COMBOS_STR="${COMBOS:-A B C D E F G H}"

if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] 'timeout' command is required but not found."
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 'conda' command is required but not found."
    exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a COMBO_LIST <<< "$COMBOS_STR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/local_debug_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    scripts/rlopt/train.py
    --task "$TASK"
    --num_envs "$NUM_ENVS"
    --headless
    --algo ipmd
)

A_OVERRIDES=(
    "agent.collector.init_random_frames=0"
    "agent.ipmd.reward_lr=1e-3"
    "agent.ipmd.reward_update_interval=1"
    "agent.ipmd.reward_margin=0.0"
    "agent.ipmd.reward_consistency_coeff=0.0"
    "agent.ipmd.use_reward_target_network=false"
    "agent.ipmd.use_reward_target_for_ppo=false"
    "agent.ipmd.normalize_reward_input=false"
    "agent.ipmd.reward_grad_penalty_coeff=0.0"
    "agent.ipmd.reward_logit_reg_coeff=0.0"
    "agent.ipmd.reward_param_weight_decay_coeff=0.0"
    "agent.ipmd.reward_replay_size=0"
    "agent.ipmd.reward_replay_ratio=0.0"
    "agent.ipmd.reward_replay_keep_prob=1.0"
    "agent.ipmd.reward_mix_alpha_start=0.0"
    "agent.ipmd.reward_mix_alpha_end=1.0"
    "agent.ipmd.reward_mix_anneal_updates=20000"
    "agent.ipmd.reward_mix_gate_estimated_std_min=0.05"
    "agent.ipmd.reward_mix_alpha_when_unstable=0.15"
    "agent.ipmd.reward_mix_gate_after_updates=500"
    "agent.ipmd.entropy_coeff_start=0.005"
    "agent.ipmd.entropy_coeff_end=0.005"
    "agent.ipmd.entropy_schedule_updates=0"
    "agent.ipmd.bc_loss_coeff=0.0"
    "agent.ipmd.bc_warmup_updates=0"
    "agent.ipmd.bc_final_coeff=0.0"
)

B_EXTRA=(
    "agent.ipmd.reward_lr=2e-4"
    "agent.ipmd.reward_update_interval=2"
)

C_EXTRA=(
    "agent.ipmd.normalize_reward_input=true"
    "agent.ipmd.reward_grad_penalty_coeff=0.2"
    "agent.ipmd.reward_logit_reg_coeff=0.02"
    "agent.ipmd.reward_param_weight_decay_coeff=1e-5"
    "agent.ipmd.reward_replay_size=200000"
    "agent.ipmd.reward_replay_ratio=0.5"
    "agent.ipmd.reward_replay_keep_prob=0.25"
)

D_EXTRA=(
    "agent.ipmd.use_reward_target_network=true"
    "agent.ipmd.use_reward_target_for_ppo=true"
    "agent.ipmd.reward_target_polyak=0.995"
    "agent.ipmd.reward_target_update_interval=1"
    "agent.ipmd.reward_margin=0.05"
    "agent.ipmd.reward_consistency_coeff=0.2"
)

E_EXTRA=(
    "agent.collector.init_random_frames=49152"
    "agent.ipmd.entropy_coeff_start=0.02"
    "agent.ipmd.entropy_coeff_end=0.005"
    "agent.ipmd.entropy_schedule_updates=15000"
    "agent.ipmd.bc_loss_coeff=0.02"
    "agent.ipmd.bc_warmup_updates=20000"
    "agent.ipmd.bc_final_coeff=0.0"
)

F_EXTRA=(
    "agent.ipmd.reward_updates_per_policy_update=2"
    "agent.ipmd.reward_update_warmup_updates=500"
    "agent.ipmd.reward_balance_policy_and_expert=true"
    "agent.ipmd.reward_train_on_logits=true"
)

G_EXTRA=(
    "agent.ipmd.reward_input_noise_std=0.01"
    "agent.ipmd.reward_input_dropout_prob=0.05"
    "agent.ipmd.reward_replay_reset_interval_updates=5000"
)

H_EXTRA=(
    "agent.ipmd.policy_random_action_prob_start=0.05"
    "agent.ipmd.policy_random_action_prob_end=0.0"
    "agent.ipmd.policy_random_action_schedule_updates=10000"
    "agent.ipmd.reward_mix_gate_abs_gap_max=0.5"
    "agent.ipmd.reward_mix_alpha_when_gap_large=0.1"
    "agent.ipmd.reward_scheduler=cosineannealinglr"
    "agent.ipmd.reward_scheduler_kwargs.T_max=50000"
    "agent.ipmd.reward_scheduler_kwargs.eta_min=5e-05"
    "agent.ipmd.reward_scheduler_step=update"
)

get_combo_overrides() {
    local combo="$1"
    OVERRIDES=("${A_OVERRIDES[@]}")
    case "$combo" in
        A)
            ;;
        B)
            OVERRIDES+=("${B_EXTRA[@]}")
            ;;
        C)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}")
            ;;
        D)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}")
            ;;
        E)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}")
            ;;
        F)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}" "${F_EXTRA[@]}")
            ;;
        G)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}" "${F_EXTRA[@]}" "${G_EXTRA[@]}")
            ;;
        H)
            OVERRIDES+=("${B_EXTRA[@]}" "${C_EXTRA[@]}" "${D_EXTRA[@]}" "${E_EXTRA[@]}" "${F_EXTRA[@]}" "${G_EXTRA[@]}" "${H_EXTRA[@]}")
            ;;
        *)
            echo "[ERROR] Unknown combo '$combo'. Supported combos: A B C D E F G H"
            exit 1
            ;;
    esac
}

force_kill_python_processes() {
    # Force cleanup after each run: timeout may leave child processes alive.
    pkill -9 python >/dev/null 2>&1 || true
}

run_one() {
    local combo="$1"
    local seed="$2"
    get_combo_overrides "$combo"

    local run_name="${combo}_seed${seed}"
    local log_file="$LOG_DIR/${run_name}.log"

    local cmd=(
        conda run -n SL python
        "${COMMON_ARGS[@]}"
        "agent.seed=${seed}"
        "agent.logger.exp_name=${run_name}"
        "${OVERRIDES[@]}"
    )

    printf "\n[%s] Running %s (timeout=%ss)\n" "$(date '+%F %T')" "$run_name" "$TIMEOUT_SECONDS"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    set +e
    timeout --signal=TERM --kill-after=20s --preserve-status "${TIMEOUT_SECONDS}" "${cmd[@]}" >"$log_file" 2>&1
    local rc=$?
    set -e

    case "$rc" in
        0)
            echo "[DONE] ${run_name} completed (log: $log_file)"
            ;;
        124|137|143)
            echo "[TIMEOUT] ${run_name} hit timeout as expected (log: $log_file)"
            ;;
        *)
            echo "[FAIL] ${run_name} exited with code ${rc} (log: $log_file)"
            ;;
    esac

    force_kill_python_processes
    sleep 2
}

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Logs dir:  $LOG_DIR"
echo "[INFO] Task=${TASK}, num_envs=${NUM_ENVS}, seeds='${SEEDS_STR}', combos='${COMBOS_STR}'"

for combo in "${COMBO_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        run_one "$combo" "$seed"
    done
done

echo
echo "[INFO] Finished local debug sweep. Check logs under: $LOG_DIR"
