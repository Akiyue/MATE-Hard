#!/usr/bin/env bash
# Sprint 5: train Double-DQN and TC-QMIX on MATE-Hard-{2v4,8v8}-9 with 3 seeds
# each. Two parallel pipelines (one per GPU), 6 runs per pipeline, ~6h per run.
#
# Total: 12 runs, ~72 GPU-hours, ~36h wall-clock per pipeline.
#
# Usage: run_sprint5_train.sh <gpu_id> <pipeline_a|pipeline_b>

set -euo pipefail

GPU="${1:?usage: run_sprint5_train.sh <gpu> <pipeline_a|pipeline_b>}"
PIPE="${2:?usage: run_sprint5_train.sh <gpu> <pipeline_a|pipeline_b>}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /home/aiteam1/sonthh/MATE
mkdir -p logs checkpoints

# Common training args (same as Sprint 2 headline runs)
COMMON_ARGS=(
    --reward-shaping --shaping-weight 0.02
    --num-envs 8 --buffer-size 50000
    --batch-size 128 --learning-starts 5000
    --train-freq 4 --target-tau 5e-3
    --total-timesteps 3000000
    --lr 3e-4 --gamma 0.99
    --eps-start 1.0 --eps-end 0.05
    --eps-decay-steps 400000
    --embed-dim 96 --num-blocks 2 --num-heads 4
    --num-experts 4 --top-k 2
)

run_train () {
    local method="$1" scenario="$2" seed="$3"
    local run_name="${method}_${scenario}_seed${seed}"
    local save_dir="checkpoints/${method}_${scenario}"
    local log="logs/sprint5_${method}_${scenario}_seed${seed}.log"
    if [[ -d "${save_dir}/${run_name}" ]] && [[ -n "$(ls ${save_dir}/${run_name}/*.pt 2>/dev/null | tail -1)" ]]; then
        local latest=$(ls -t "${save_dir}/${run_name}"/*.pt 2>/dev/null | head -1)
        local mtime=$(stat -c %Y "$latest")
        local now=$(date +%s)
        local age=$(( now - mtime ))
        # If a checkpoint exists and is recent (<30min), assume still training; skip
        if [[ "$age" -lt 1800 ]]; then
            echo ">>> SKIP $run_name (active checkpoint at $latest, ${age}s old)"
            return 0
        fi
        # Otherwise assume it's a completed prior run — skip
        echo ">>> SKIP $run_name (completed checkpoint already at $latest)"
        return 0
    fi
    echo "==========================================================="
    echo ">>> TRAIN $method on $scenario seed $seed   GPU $GPU"
    echo ">>> log: $log"
    echo "==========================================================="

    if [[ "$method" == "dqn" ]]; then
        uv run python -m mate_marl.scripts.train_dqn \
            --mate-config "${scenario}-fast.yaml" \
            --seed "$seed" --run-name "$run_name" \
            --save-dir "$save_dir" --tb-log-dir tb \
            "${COMMON_ARGS[@]}" 2>&1 | tee "$log"
    elif [[ "$method" == "tcqmix" ]]; then
        uv run python -m mate_marl.scripts.train_tcqmix \
            --mate-config "${scenario}-fast.yaml" \
            --seed "$seed" --run-name "$run_name" \
            --save-dir "$save_dir" --tb-log-dir tb \
            --mixer-hyper-input state+types \
            "${COMMON_ARGS[@]}" 2>&1 | tee "$log"
    else
        echo "unknown method: $method"; return 1
    fi
}

case "$PIPE" in
    pipeline_a)
        # GPU 0: all DQN runs
        run_train dqn MATE-2v4-9 0
        run_train dqn MATE-2v4-9 1
        run_train dqn MATE-2v4-9 2
        run_train dqn MATE-8v8-9 0
        run_train dqn MATE-8v8-9 1
        run_train dqn MATE-8v8-9 2
        ;;
    pipeline_b)
        # GPU 1: all TC-QMIX runs
        run_train tcqmix MATE-2v4-9 0
        run_train tcqmix MATE-2v4-9 1
        run_train tcqmix MATE-2v4-9 2
        run_train tcqmix MATE-8v8-9 0
        run_train tcqmix MATE-8v8-9 1
        run_train tcqmix MATE-8v8-9 2
        ;;
    *)
        echo "unknown pipeline: $PIPE"; exit 1
        ;;
esac

echo "sprint5 pipeline $PIPE DONE"
