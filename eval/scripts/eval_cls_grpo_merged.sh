#!/usr/bin/env bash
#SBATCH --job-name=eval-cls-grpo
#SBATCH --partition=kolyoz-cuda                                            # recot-eval (cu130 + vLLM) is kolyoz-only
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24    # corrupt GPUs: CUDA init fails ("CUDA unknown error") or no device handle
#SBATCH --requeue                                                          # allow self-requeue when we land on a corrupt GPU
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --array=0-4                                                        # one task per CLS benchmark, run in parallel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-grpo-%A_%a.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-grpo-%A_%a.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_cls_vllm.py
# Stage-3 GRPO model: Qwen3.5-0.8B -> +SFT LoRA -> cls-merged -> +CoT LoRA -> cot-cls-merged -> +GRPO (full FT) -> grpo run 2026-07-23-15-36-01
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/grpo/Qwen3.5-VL-CLS-GRPO-2026-07-23-15-36-01
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

# NOTE: do NOT pass --template_config. The CoT/GRPO models were trained with no system turn
# at all (Qwen3.5 gets no default system message), so the thinking template's
# meta_instruction is an unseen ~90-token prefix. Measured A/B on the VQA CoT model: with it,
# 0/6 outputs were well-formed (accuracy 0.0); without it, 6/6. See diagnosis_vqa.md.

DATASETS=(
    cls_AID.json
    cls_NWPU_RESISC45.json
    cls_WHU_RS19.json
    cls_METER_ML.json
    cls_SIRI_WHU.json
)
ANNS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "### task=$SLURM_ARRAY_TASK_ID dataset=$ANNS node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Several kolyoz nodes have GPUs that pass nvidia-smi but fail torch's CUDA init.
if ! $PY -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing task $SLURM_ARRAY_TASK_ID onto another node"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 1
fi

# max_new_tokens 1024 (not the 128 used for the SFT models): CLS CoT/GRPO traces are ~400
# tokens of reasoning BEFORE the <answer> tag, so at 128 the answer never appears and every
# sample scores 0.
$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json "$ANNS" \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-grpo-cls-merged \
    --tensor_parallel_size 1 \
    --max_new_tokens 1024 \
    --temperature 0.0

echo "### DONE $ANNS date=$(date)"
