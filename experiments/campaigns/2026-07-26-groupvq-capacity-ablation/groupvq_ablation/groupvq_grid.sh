#!/usr/bin/env bash
# Shared arm table for the DiffSR (spectral) grouped-VQ capacity ablation.
#
# Bottleneck is fixed to `gumbel_multicat`: G independent per-group codebooks of
# C entries each, per-group Gumbel-softmax with hard straight-through, and a
# per-group code dim of z_dim // G. Only G (groups) and C (codebook size) move.
#
# Row format: name:groups:categories
# z_dim is 256 for every row, so groups must divide 256.
#
# g64_c128 is the anchor: it is the configuration that previously tracked the
# continuous deterministic latent, and it is protocol-identical to the
# `gumbel_multicat` arm of the 2026-07-22 DiffSR bottleneck study.

GROUPVQ_Z_DIM=256

GROUPVQ_ARMS=(
    # groups sweep at the anchor codebook size
    "g16_c128:16:128"
    "g32_c128:32:128"
    "g64_c128:64:128"
    "g128_c128:128:128"
    # codebook-size sweep at the anchor group count
    "g64_c16:64:16"
    "g64_c64:64:64"
    "g64_c512:64:512"
)

groupvq_arm_names() {
    local row
    for row in "${GROUPVQ_ARMS[@]}"; do
        printf '%s\n' "${row%%:*}"
    done
}

# Populates GROUPVQ_GROUPS / GROUPVQ_CATEGORIES / GROUPVQ_CODE_DIM /
# GROUPVQ_BITS for the named arm; exits non-zero on an unknown name.
groupvq_lookup_arm() {
    local want="$1" row name groups categories
    for row in "${GROUPVQ_ARMS[@]}"; do
        IFS=: read -r name groups categories <<<"${row}"
        if [[ "${name}" == "${want}" ]]; then
            if (( GROUPVQ_Z_DIM % groups != 0 )); then
                echo "[ERROR] arm ${name}: groups=${groups} does not divide z_dim=${GROUPVQ_Z_DIM}." >&2
                return 2
            fi
            GROUPVQ_GROUPS="${groups}"
            GROUPVQ_CATEGORIES="${categories}"
            GROUPVQ_CODE_DIM=$((GROUPVQ_Z_DIM / groups))
            # Nominal bits per 5 Hz command = G * log2(C); C is a power of two
            # in every row of this grid, so integer log2 is exact.
            local bits_per_group=0 value="${categories}"
            while (( value > 1 )); do
                if (( value % 2 != 0 )); then
                    echo "[ERROR] arm ${name}: categories=${categories} must be a power of two." >&2
                    return 2
                fi
                value=$((value / 2))
                bits_per_group=$((bits_per_group + 1))
            done
            GROUPVQ_BITS=$((groups * bits_per_group))
            return 0
        fi
    done
    echo "[ERROR] Unknown grouped-VQ arm: ${want}." >&2
    return 2
}
