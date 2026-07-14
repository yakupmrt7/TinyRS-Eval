#!/usr/bin/env bash
#SBATCH --job-name=eval-cls-merged
#SBATCH --partition=kolyoz-cuda                          # recot-eval (cu130 + vLLM) is kolyoz-only
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24    # corrupt GPUs: CUDA init fails ("CUDA unknown error") or no device handle
#SBATCH --requeue                                        # allow self-requeue when we land on a corrupt GPU
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --array=0-4                                      # one task per CLS benchmark, run in parallel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-merged-%A_%a.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output/eval-cls-merged-%A_%a.err

set -euo pipefail

PY=/arf/home/aalatan/mert/envs/recot-eval/bin/python
SCRIPT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/python_script/evaluation/eval_cls_vllm.py
# LoRA-merged CLS model (base Qwen3.5-0.8B + qwen35-cls-lora, merged in bf16)
MODEL=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cls-merged
DATA_ROOT=/arf/scratch/aalatan/Re-CoT/datasets_eval
OUTPUT=/arf/scratch/aalatan/Re-CoT/TinyRS/eval/output

# Each array task evaluates one benchmark on its own GPU.
DATASETS=(
    cls_AID.json
    cls_NWPU_RESISC45.json
    cls_WHU_RS19.json
    cls_METER_ML.json
    cls_SIRI_WHU.json
)
ANNS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# model + data are local; keep everything offline (system HTTPS is broken anyway)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # deterministic; avoids CUDA fork crash

echo "### task=$SLURM_ARRAY_TASK_ID dataset=$ANNS node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Several kolyoz nodes have GPUs that pass nvidia-smi but fail torch's CUDA init
# ("CUDA unknown error"). Detect that before vLLM spends minutes loading, and bounce
# the task back to the queue so SLURM retries it on a different node.
if ! $PY -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing task $SLURM_ARRAY_TASK_ID onto another node"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60   # hold the allocation briefly so SLURM is unlikely to hand back the same node
    exit 1
fi

$PY "$SCRIPT" \
    --model_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --anns_json "$ANNS" \
    --output_dir "$OUTPUT" \
    --model_name qwen3.5-0.8b-cls-merged \
    --tensor_parallel_size 1 \
    --max_new_tokens 128 \
    --temperature 0.0

echo "### DONE $ANNS date=$(date)"
