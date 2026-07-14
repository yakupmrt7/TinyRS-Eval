#!/usr/bin/env bash
#SBATCH --job-name=eval-vg-merged
#SBATCH --partition=kolyoz-cuda                                            # recot-eval (cu130 + vLLM) is kolyoz-only
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24    # corrupt GPUs: CUDA init fails ("CUDA unknown error") or no device handle
#SBATCH --requeue                                                          # allow self-requeue when we land on a corrupt GPU
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vg-merged-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vg-merged-%j.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_vg_vllm.py
# LoRA-merged VG model (base Qwen3.5-0.8B + qwen35-vg-lora, merged in bf16)
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-vg-merged
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

# model + data are local; keep everything offline (system HTTPS is broken anyway)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # deterministic; avoids CUDA fork crash

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Several kolyoz nodes have GPUs that pass nvidia-smi but fail torch's CUDA init
# ("CUDA unknown error"). Detect that before vLLM spends minutes loading, and bounce
# the job back to the queue so SLURM retries it on a different node.
if ! $PY -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing job onto another node"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60   # hold the allocation briefly so SLURM is unlikely to hand back the same node
    exit 1
fi

# DIOR-RSVG images are 800x800 and GT boxes are absolute pixels, so a model trained
# on 0-800 normalized boxes maps 1:1 to pixels (bbox_normalize_bound=800).
$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json VG_DOIR_RSVG_test.json \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-vg-merged \
    --bbox_normalize_bound 800 \
    --iou_threshold 0.5 \
    --tensor_parallel_size 1 \
    --max_new_tokens 128 \
    --temperature 0.0

echo "### DONE date=$(date)"
