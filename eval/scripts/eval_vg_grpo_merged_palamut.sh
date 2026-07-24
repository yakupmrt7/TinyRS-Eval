#!/usr/bin/env bash
#SBATCH --job-name=eval-vg-grpo-palamut
#SBATCH --partition=palamut-cuda                                          # EXPERIMENTAL: recot-eval (cu130 + vLLM) is documented kolyoz-only (H100);
                                                                            # trying palamut (A100) anyway since kolyoz-cuda is fully saturated.
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vg-grpo-palamut-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-vg-grpo-palamut-%j.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_vg_vllm.py
# Stage-3 GRPO model: Qwen3.5-0.8B -> +SFT LoRA -> vg-merged -> +CoT LoRA -> cot-vg-merged -> +GRPO (full FT) -> grpo run 2026-07-23-13-59-29
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/grpo/Qwen3.5-VL-VG-GRPO-2026-07-23-13-59-29
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# No --requeue here on purpose: unlike the kolyoz bad-GPU case, a failure on palamut is more
# likely a systemic CUDA/driver incompatibility (A100 vs the cu130 build), which would just
# fail identically on every node and loop forever if requeued. Fail once and report instead.
if ! $PY -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) -- palamut (A100) likely incompatible with the cu130 recot-eval build. Not requeueing."
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
    --model_name qwen3.5-0.8b-grpo-vg-merged \
    --bbox_normalize_bound 800 \
    --iou_threshold 0.5 \
    --tensor_parallel_size 1 \
    --max_new_tokens 1024 \
    --temperature 0.0

echo "### DONE date=$(date)"
