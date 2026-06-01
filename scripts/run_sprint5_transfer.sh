#!/usr/bin/env bash
# Sprint 5 transfer matrix: eval each of the 12 newly-trained checkpoints
# (DQN+TCQMIX × 3 seeds × {2v4, 8v8}) on all 4 scenarios.
#
# Usage: run_sprint5_transfer.sh <gpu_id> <pipeline_a|pipeline_b>
#   pipeline_a (GPU 0): all DQN evals (6 ckpts × 4 scenarios)
#   pipeline_b (GPU 1): all TCQMIX evals (6 ckpts × 4 scenarios)

set -euo pipefail

GPU="${1:?usage: run_sprint5_transfer.sh <gpu> <pipeline_a|pipeline_b>}"
PIPE="${2:?usage: run_sprint5_transfer.sh <gpu> <pipeline_a|pipeline_b>}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /home/aiteam1/sonthh/MATE

SCENARIOS=(MATE-2v4-9 MATE-4v4-9 MATE-4v8-9 MATE-8v8-9)
SEEDS="0 1 2"
NEP=30  # 30 ep × 3 seeds = 90 ep per cell

latest_ckpt () {
    local dir="$1"
    ls -t "$dir"/*.pt 2>/dev/null | head -1
}

run_eval () {
    local method="$1" trained_on="$2" seed="$3"
    local run_dir="checkpoints/${method}_${trained_on}/${method}_${trained_on}_seed${seed}"
    local ckpt
    ckpt=$(latest_ckpt "$run_dir")
    if [[ -z "$ckpt" ]]; then
        echo ">>> ERROR: no checkpoint in $run_dir" >&2
        return 1
    fi
    local out="results/sprint5_${method}_trainedOn_${trained_on}_seed${seed}_transfer.csv"
    local log="logs/sprint5_transfer_${method}_${trained_on}_seed${seed}.log"
    if [[ -f "$out" ]]; then
        echo ">>> SKIP $method/${trained_on}/seed${seed} (exists: $out)"
        return 0
    fi
    echo "==========================================================="
    echo ">>> $method trained-on=$trained_on seed=$seed   GPU $GPU"
    echo ">>> ckpt: $ckpt"
    echo ">>> out:  $out"
    echo "==========================================================="

    if [[ "$method" == "dqn" ]]; then
        uv run python -m mate_marl.scripts.eval_dqn \
            --checkpoint "$ckpt" \
            --scenarios "${SCENARIOS[@]}" \
            --seeds $SEEDS --num-episodes $NEP \
            --out "$out" 2>&1 | tee "$log"
    elif [[ "$method" == "tcqmix" ]]; then
        uv run python -m mate_marl.scripts.eval_tcqmix \
            --checkpoint "$ckpt" \
            --scenarios "${SCENARIOS[@]}" \
            --seeds $SEEDS --num-episodes $NEP \
            --mixer-hyper-input state+types \
            --out "$out" 2>&1 | tee "$log"
    else
        echo "unknown method: $method"; return 1
    fi
}

case "$PIPE" in
    pipeline_a)
        for s in 0 1 2; do run_eval dqn MATE-2v4-9 "$s"; done
        for s in 0 1 2; do run_eval dqn MATE-8v8-9 "$s"; done
        ;;
    pipeline_b)
        for s in 0 1 2; do run_eval tcqmix MATE-2v4-9 "$s"; done
        for s in 0 1 2; do run_eval tcqmix MATE-8v8-9 "$s"; done
        ;;
    *)
        echo "unknown pipeline: $PIPE"; exit 1
        ;;
esac

echo "sprint5 transfer pipeline $PIPE DONE"
