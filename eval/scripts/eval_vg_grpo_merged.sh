#!/usr/bin/env bash
#SBATCH --job-name=eval-vg-grpo
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
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vg-grpo-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vg-grpo-%j.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_vg_vllm.py
# Stage-3 GRPO model: Qwen3.5-0.8B -> +SFT LoRA -> vg-merged -> +CoT LoRA -> cot-vg-merged -> +GRPO (full FT, small-object-oversampled dataset) -> grpo run 2026-07-24-10-25-28
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/grpo/Qwen3.5-VL-VG-Small2x-GRPO-2026-07-24-10-25-28
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

# No system prompt is sent (eval_vg_vllm.py sends a bare user turn), which matches how the
# CoT model was trained. See diagnosis_vqa.md: injecting the thinking template's
# meta_instruction made the VQA CoT model emit unparseable output on 6/6 samples.
# extract_bboxes() already strips <reasoning>/<answer> tags, so CoT output parses as-is.
# The GRPO model was trained with the same <reasoning>/<answer> format and 0-800 pixel
# bounding boxes as the CoT stage, so nothing else needs to change here.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Several kolyoz nodes have GPUs that pass nvidia-smi but fail torch's CUDA init.
if ! $PY -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing onto another node"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 1
fi

# bbox_normalize_bound 800: the GRPO dataset (VHM_dataset_grpo_vg_only_2k) boxes are on the
# same 0-800 pixel scale as the CoT stage. DIOR images are 800x800 with absolute-pixel GT,
# so 0-800 preds map 1:1 to pixels.
# max_new_tokens 1024: VG CoT/GRPO traces are ~250-300 tokens of reasoning before <answer>.
$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json VG_DOIR_RSVG_test.json \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-grpo-vg-small2x-merged \
    --bbox_normalize_bound 800 \
    --iou_threshold 0.5 \
    --tensor_parallel_size 1 \
    --max_new_tokens 1024 \
    --temperature 0.0

echo "### DONE date=$(date)"
