#!/usr/bin/env bash

# Each arm changes only the DiffSR transition factorization used to pretrain z.
# Encoder input, bottleneck, low-level controller, data, and optimizer stay fixed.

SKILL_ENCODING_ARMS=(
    state_occupancy
    semimarkov_chain
    endpoint_delta
)

configure_skill_encoding_arm() {
    local arm="$1"
    ARM_PRETRAIN_ARGS=()
    ARM_DESCRIPTION=""
    ARM_WANDB_TAGS=""

    case "${arm}" in
        state_occupancy)
            ARM_DESCRIPTION="uniform option-state occupancy: p(s[t+h_k] | s[t], z), h_k={2,4,6,8,10}"
            ARM_WANDB_TAGS="state-occupancy,multicheckpoint,h2-4-6-8-10"
            ARM_PRETRAIN_ARGS+=(
                --transition_objective state_occupancy
                --transition_offsets 2 4 6 8 10
            )
            ;;
        semimarkov_chain)
            ARM_DESCRIPTION="held-z checkpoint chain: product_k p(s[t+h_k] | s[t+h_{k-1}], z)"
            ARM_WANDB_TAGS="semimarkov-chain,multicheckpoint,h2-4-6-8-10"
            ARM_PRETRAIN_ARGS+=(
                --transition_objective semimarkov_chain
                --transition_offsets 2 4 6 8 10
            )
            ;;
        endpoint_delta)
            ARM_DESCRIPTION="relative option outcome: p(s[t+10]-s[t] | s[t], z)"
            ARM_WANDB_TAGS="endpoint-delta,relative-outcome,h10"
            ARM_PRETRAIN_ARGS+=(
                --transition_objective endpoint_delta
                --transition_offsets 10
            )
            ;;
        *)
            echo "[FATAL] Unknown skill-encoding arm: ${arm}" >&2
            return 2
            ;;
    esac
}
