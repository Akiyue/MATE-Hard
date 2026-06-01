#!/usr/bin/env bash
# Transfer-matrix driver: evals every (method, seed) checkpoint trained on
# MATE-Hard-4v8-9 on MATE-{2v4,4v4,8v8}-9 (plain MATE, no Het/Energy/Fog).
#
# Each call writes results/<method>_seed<S>_transfer.csv.
#
# Usage: run_transfer_matrix.sh <gpu_id> <pipeline_a|pipeline_b>

set -euo pipefail

GPU="${1:?usage: run_transfer_matrix.sh <gpu> <pipeline_a|pipeline_b>}"
PIPE="${2:?usage: run_transfer_matrix.sh <gpu> <pipeline_a|pipeline_b>}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /home/aiteam1/sonthh/MATE

SCENARIOS=(MATE-2v4-9 MATE-4v4-9 MATE-8v8-9)
SEEDS="0 1 2"  # eval seeds per ckpt
NEP=30         # 30 ep × 3 seeds = 90 ep per cell

run_eval () {
    local method="$1" seed="$2" ckpt="$3" extra="$4"
    local out="results/${method}_seed${seed}_transfer.csv"
    local log="logs/transfer_${method}_seed${seed}.log"
    if [[ -f "$out" ]]; then
        echo ">>> SKIP $method seed=$seed (already exists at $out)"
        return 0
    fi
    echo "==========================================================="
    echo ">>> $method seed=$seed on GPU $GPU"
    echo ">>> ckpt: $ckpt"
    echo ">>> out:  $out"
    echo "==========================================================="

    case "$method" in
        DQN)
            uv run python -m mate_marl.scripts.eval_dqn \
                --checkpoint "$ckpt" \
                --scenarios "${SCENARIOS[@]}" \
                --seeds $SEEDS --num-episodes $NEP \
                --out "$out" 2>&1 | tee "$log"
            ;;
        TCQMIX)
            uv run python -m mate_marl.scripts.eval_tcqmix \
                --checkpoint "$ckpt" \
                --scenarios "${SCENARIOS[@]}" \
                --seeds $SEEDS --num-episodes $NEP \
                --mixer-hyper-input state+types \
                --out "$out" 2>&1 | tee "$log"
            ;;
        TCQMIX_types)
            uv run python -m mate_marl.scripts.eval_tcqmix \
                --checkpoint "$ckpt" \
                --scenarios "${SCENARIOS[@]}" \
                --seeds $SEEDS --num-episodes $NEP \
                --mixer-hyper-input types \
                --out "$out" 2>&1 | tee "$log"
            ;;
        QMIX_vanilla)
            uv run python -m mate_marl.scripts.eval_tcqmix \
                --checkpoint "$ckpt" \
                --scenarios "${SCENARIOS[@]}" \
                --seeds $SEEDS --num-episodes $NEP \
                --mixer-hyper-input state \
                --out "$out" 2>&1 | tee "$log"
            ;;
        VDN)
            uv run python -m mate_marl.scripts.eval_vdn \
                --checkpoint "$ckpt" \
                --scenarios "${SCENARIOS[@]}" \
                --seeds $SEEDS --num-episodes $NEP \
                --out "$out" 2>&1 | tee "$log"
            ;;
        MAPPO)
            uv run python -m mate_marl.scripts.eval_mappo \
                --checkpoint "$ckpt" \
                --scenarios "${SCENARIOS[@]}" \
                --seeds $SEEDS --num-episodes $NEP \
                --out "$out" 2>&1 | tee "$log"
            ;;
        *)
            echo "unknown method: $method" >&2; exit 1
            ;;
    esac
}

# Pipeline assignment: 16 cells across 2 GPUs.
# (DQN_seed0 and TCQMIX_seed0 transfers already exist — skipped automatically.)

case "$PIPE" in
    pipeline_a)
        # GPU 0: 8 cells
        run_eval DQN  1 "checkpoints/dqn/dqn_4v8_seed1/dqn_20260513-200641.pt" ""
        run_eval DQN  2 "checkpoints/dqn/dqn_4v8_seed2/dqn_20260514-105838.pt" ""
        run_eval TCQMIX 1 "checkpoints/tcqmix/tcqmix_4v8_seed1/tcqmix_20260514-215855.pt" ""
        run_eval TCQMIX 2 "checkpoints/tcqmix/tcqmix_4v8_seed2/tcqmix_20260515-090759.pt" ""
        run_eval TCQMIX_types 0 "checkpoints/tcqmix_types/tcqmix_types_4v8_seed0/tcqmix_20260522-031533.pt" ""
        run_eval TCQMIX_types 1 "checkpoints/tcqmix_types/tcqmix_types_4v8_seed1/tcqmix_20260523-013155.pt" ""
        run_eval TCQMIX_types 2 "checkpoints/tcqmix_types/tcqmix_types_4v8_seed2/tcqmix_20260524-060145.pt" ""
        run_eval QMIX_vanilla 0 "$(ls -t checkpoints/qmix_vanilla/qmix_vanilla_4v8_seed0/*.pt | head -1)" ""
        ;;
    pipeline_b)
        # GPU 1: 8 cells
        run_eval QMIX_vanilla 1 "$(ls -t checkpoints/qmix_vanilla/qmix_vanilla_4v8_seed1/*.pt | head -1)" ""
        run_eval QMIX_vanilla 2 "$(ls -t checkpoints/qmix_vanilla/qmix_vanilla_4v8_seed2/*.pt | head -1)" ""
        run_eval VDN 0 "$(ls -t checkpoints/vdn/vdn_4v8_seed0/*.pt | head -1)" ""
        run_eval VDN 1 "$(ls -t checkpoints/vdn/vdn_4v8_seed1/*.pt | head -1)" ""
        run_eval VDN 2 "$(ls -t checkpoints/vdn/vdn_4v8_seed2/*.pt | head -1)" ""
        run_eval MAPPO 0 "$(ls -t checkpoints/mappo/mappo_4v8_seed0/*.pt | head -1)" ""
        run_eval MAPPO 1 "$(ls -t checkpoints/mappo/mappo_4v8_seed1/*.pt | head -1)" ""
        run_eval MAPPO 2 "$(ls -t checkpoints/mappo/mappo_4v8_seed2/*.pt | head -1)" ""
        ;;
    *)
        echo "unknown pipeline: $PIPE" >&2; exit 1
        ;;
esac

echo "transfer pipeline $PIPE done."
