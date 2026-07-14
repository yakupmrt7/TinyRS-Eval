#!/usr/bin/env bash
#SBATCH --job-name=eval-cls-smoke
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-smoke-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-smoke-%j.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_cls_vllm.py
MODEL=/arf/scratch/aalatan/Re-CoT/models/Qwen3.5-0.8B
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

# model + data are local; keep everything offline (system HTTPS is broken anyway)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json cls_AID.json \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-smoke --limit 16 \
    --tensor_parallel_size 1 \
    --max_new_tokens 128 \
    --temperature 0.0

echo "### DONE date=$(date)"
