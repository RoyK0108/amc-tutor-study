#!/bin/bash
# Autonomous 4B pipeline: base ("before") eval on AMC+AIME, then QLoRA training.
# The fine-tuned ("after") eval is run separately afterwards so we can pick the
# best checkpoint by validation loss rather than blindly using the final one.
set -x
cd /Users/YNA/amc-tutor || exit 1
V=/Users/YNA/amc-tutor/.venv/bin

echo "=== STAGE 1: base ('before') eval, Qwen3-4B ==="
$V/python eval/evaluate.py --model mlx-community/Qwen3-4B-4bit --data eval/amc.jsonl  --name amc_base4b_zeroshot  --out-dir results
$V/python eval/evaluate.py --model mlx-community/Qwen3-4B-4bit --data eval/aime.jsonl --name aime_base4b_zeroshot --out-dir results

echo "=== STAGE 2: QLoRA training, Qwen3-4B ==="
$V/mlx_lm.lora -c configs/lora_qwen3_4b.yaml

echo "=== 4B PIPELINE (base + train) DONE ==="
