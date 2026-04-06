#!/bin/bash
set -euo pipefail

SCRIPT="train_fsdp_tp.py"
LOG_FSDP_TP="log.txt"
LOG_FSDP_ONLY="ref.txt"
DEBUG_FSDP_TP="./debug_fsdp_tp"
DEBUG_FSDP_ONLY="./debug_fsdp_only"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
COMMON_ARGS="--model_name $MODEL_NAME --num_steps 20 --lr 3e-4 --seed 42"

echo "=== Generating fixed batches for $MODEL_NAME ==="
python generate_fixed_batches.py "$MODEL_NAME"

echo "=== Launching both runs in parallel (GPUs 0-3 for FSDP+TP, GPUs 4-5 for FSDP-only) ==="

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 $SCRIPT $COMMON_ARGS --fsdp_size 2 --tp_size 2 --enable_sp --fixed_batches --debug_dump $DEBUG_FSDP_TP > "$LOG_FSDP_TP" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29501 $SCRIPT $COMMON_ARGS --fsdp_size 2 --fixed_batches --debug_dump $DEBUG_FSDP_ONLY > "$LOG_FSDP_ONLY" 2>&1 &
PID2=$!

echo "FSDP+TP PID=$PID1 | FSDP-only PID=$PID2"
wait $PID1 && echo "FSDP+TP done" || echo "FSDP+TP failed (exit $?)"
wait $PID2 && echo "FSDP-only done" || echo "FSDP-only failed (exit $?)"

echo ""
echo "=== Loss & Grad Diff ==="
git diff --no-index --color --word-diff=color "$LOG_FSDP_TP" "$LOG_FSDP_ONLY" || true

echo ""
echo "=== Debug Dump Comparison ==="
python compare_debug_dumps.py "$DEBUG_FSDP_TP" "$DEBUG_FSDP_ONLY"
