#!/usr/bin/env bash
#SBATCH --job-name=eval-cls-aid-merged
#SBATCH --partition=kolyoz-cuda                # recot-eval (cu130 + vLLM) is kolyoz-only
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz19,kolyoz24   # corrupt GPUs: CUDA init fails ("CUDA unknown error") or no device handle
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-aid-merged-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-aid-merged-%j.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_cls_vllm.py
# LoRA-merged CLS model (base Qwen3.5-0.8B + qwen35-cls-lora, merged in bf16)
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cls-merged
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

# model + data are local; keep everything offline (system HTTPS is broken anyway)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # deterministic; avoids CUDA fork crash

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Several kolyoz nodes have GPUs that pass nvidia-smi but fail torch's CUDA init.
# Fail fast with a clear message instead of a deep vLLM traceback.
$PY -c "import torch,sys; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))" \
    || { echo "### CUDA BROKEN on $(hostname) — add it to --exclude and resubmit"; exit 1; }

$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json cls_AID.json \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-cls-merged \
    --tensor_parallel_size 1 \
    --max_new_tokens 128 \
    --temperature 0.0

echo "### DONE date=$(date)"
