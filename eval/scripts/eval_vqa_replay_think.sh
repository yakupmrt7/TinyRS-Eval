#!/usr/bin/env bash
#SBATCH --job-name=eval-vqa-replay-think
#SBATCH --partition=kolyoz-cuda                                            # recot-eval (cu130 + vLLM) is kolyoz-only
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24    # corrupt GPUs
#SBATCH --requeue
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vqa-replay-think-%A_%a.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vqa-replay-think-%A_%a.err

# THINKING mode eval of the stage-2 REPLAY VQA model (SFT-merged + replay LoRA). Prompt opens the native <think> block, so
# the model reasons then answers -- compare against the CoT model (mean 72.9%).


set -euo pipefail
PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_cls_vllm.py
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-replay-vqa-merged
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

DATASETS=(
    RSVQA_LR-rural_urban_RSVQA.json
    RSVQA_LR-presence_RSVQA.json
    RSVQA_LR-comp_RSVQA.json
    RSVQA_HR-comp_RSVQA.json
    RSVQA_HR-presence_RSVQA.json
)
ANNS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "### task=$SLURM_ARRAY_TASK_ID dataset=$ANNS node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if ! $PY -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing"; scontrol requeue "$SLURM_JOB_ID"; sleep 60; exit 1
fi

# --enable_thinking opens the <think> block; max_new_tokens 1024 for the reasoning trace
$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json "$ANNS" \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-replay-vqa-think \
    --tensor_parallel_size 1 \
    --enable_thinking \
    --max_new_tokens 1024 \
    --temperature 0.0

echo "### DONE $ANNS date=$(date)"
