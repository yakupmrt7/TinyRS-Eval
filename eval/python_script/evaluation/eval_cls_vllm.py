#!/usr/bin/env python
"""Standalone classification benchmark for Qwen3.5-VL models via vLLM.

The stock rs_evaluation.py cannot run in the recot-eval env (it imports the
vendored `lmdeploy` at module load and none of its backends support qwen3_5),
so this script talks to vLLM directly while reproducing TinyRS's exact cls
scoring: optional <reasoning>/<answer> extraction followed by a cleaned
substring match (see eval_results_vqa in rs_evaluation.py).
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

Image.MAX_IMAGE_PIXELS = None


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_root", required=True,
                   help="folder containing the *.json annotation file")
    p.add_argument("--anns_json", nargs="+", default=["cls_AID.json"],
                   help="one or more annotation files; all are scored in a single model load")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--model_name", default=None,
                   help="label used in output filenames (default: model dir name)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                   help="cap vision tokens; Qwen smart-resize upper bound")
    p.add_argument("--template_config", default=None,
                   help="json holding a `meta_instruction` to send as the system message "
                        "(e.g. config/qwen2_thinking_template.json for CoT models)")
    return p.parse_args()


def load_system_prompt(template_config):
    if not template_config:
        return None
    with open(template_config) as f:
        return json.load(f).get("meta_instruction") or None


def resolve_image_path(data_root: Path, anns_json_path: Path, anns: dict) -> Path:
    """Mirror infer_single() image resolution from rs_evaluation.py."""
    fn = anns["image"]
    if "image_path" not in anns:
        return anns_json_path.parent / anns_json_path.stem / fn
    if Path(anns["image_path"]).is_absolute():
        return Path(anns["image_path"]) / fn
    return anns_json_path.parent / anns["image_path"] / fn


def clean_prediction(raw):
    """Identical cleaning to eval_results_vqa (cls branch).

    Returns (prediction, malformed). `malformed` is True when the output opened a
    reasoning/answer block but never closed it, which means no answer can be
    extracted. The original code swallowed that with a bare `except: pass` and left
    `pred` as the entire reasoning paragraph, which then silently scored False on
    every sample -- an unparseable model reads as 0.0 accuracy rather than an error.
    """
    pred = str(raw)
    malformed = False

    if "<reasoning>" in pred:
        parts = pred.split("<reasoning>")[1].split("</reasoning>")
        if len(parts) > 1:
            pred = parts[1]
        else:
            malformed = True  # opened <reasoning> but never closed it

    if "<answer>" in pred:
        parts = pred.split("<answer>")[1].split("</answer>")
        pred = parts[0]
    elif malformed:
        # no answer block at all after an unterminated <reasoning>: nothing to score
        return "", True

    pred = pred.replace(" ", "")
    if "." in pred:
        pred = pred.split(".")[0]
    if "," in pred:
        pred = pred.split(",")[0]
    return pred.strip().lower(), malformed


def evaluate_dataset(anns_json, args, data_root, model_name, out_dir, processor, llm,
                     sampling, system_prompt=None):
    """Score one annotation file against an already-loaded model."""
    anns_json_path = data_root / anns_json
    test_name = anns_json_path.stem
    save_jsonl = out_dir / f"{test_name}_{model_name}_eval.jsonl"
    save_metric = out_dir / f"{test_name}_{model_name}_eval.json"

    with open(anns_json_path) as f:
        anns_dict = json.load(f)
    if args.limit is not None:
        anns_dict = anns_dict[: args.limit]
    print(f"[eval] {test_name}: {len(anns_dict)} samples | model={model_name}", flush=True)

    records, llm_inputs = [], []
    for anns in anns_dict:
        convs = anns["conversations"]
        question = convs[0]["value"].replace("<image>\n", "").strip()
        answer = convs[1]["value"]
        img_path = resolve_image_path(data_root, anns_json_path, anns)
        image = Image.open(img_path).convert("RGB")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [
                {"type": "text", "text": system_prompt},
            ]})
        messages.append({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]})
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        llm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        records.append({"filename": str(img_path), "query": question, "answer": answer})

    outputs = llm.generate(llm_inputs, sampling_params=sampling)

    correct = 0
    n_malformed = 0
    lines = []
    for rec, out in zip(records, outputs):
        raw = out.outputs[0].text
        pred, malformed = clean_prediction(raw)
        n_malformed += int(malformed)
        ans = rec["answer"].replace(" ", "").strip().lower()
        score = (not malformed) and pred in ans
        correct += int(score)
        rec.update({"pred": raw, "prediction": pred,
                    "malformed": bool(malformed), "score": bool(score)})
        lines.append(json.dumps(rec))

    accuracy = correct / len(records) if records else 0.0
    with open(save_jsonl, "w") as f:
        f.write("\n".join(lines))
    with open(save_metric, "w") as f:
        json.dump({"dataset": test_name, "model": model_name,
                   "num_samples": len(records), "accuracy": accuracy,
                   "malformed_outputs": n_malformed}, f, indent=4)

    print(f"[eval] {test_name} accuracy: {accuracy:.4f} "
          f"({correct}/{len(records)})", flush=True)
    if n_malformed:
        pct = 100 * n_malformed / len(records)
        print(f"[eval] *** WARNING: {n_malformed}/{len(records)} ({pct:.1f}%) outputs were "
              f"UNPARSEABLE (no closing </reasoning> or <answer> block). The accuracy above "
              f"is NOT a model-quality signal -- these samples had no extractable answer. "
              f"Check the prompt matches training (e.g. do not send a system prompt the "
              f"model never saw).", flush=True)
    print(f"[eval] wrote {save_jsonl}\n[eval] wrote {save_metric}", flush=True)

    return {"dataset": test_name, "num_samples": len(records),
            "accuracy": accuracy, "malformed_outputs": n_malformed}


def main():
    args = arg_parser()
    data_root = Path(args.data_root)
    model_name = args.model_name or Path(args.model_path).name
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, max_pixels=args.max_pixels
    )

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    system_prompt = load_system_prompt(args.template_config)
    if system_prompt:
        print(f"[eval] system prompt from {args.template_config}", flush=True)

    results = [
        evaluate_dataset(anns_json, args, data_root, model_name, out_dir,
                         processor, llm, sampling, system_prompt)
        for anns_json in args.anns_json
    ]

    print(f"\n[eval] ===== summary | model={model_name} =====", flush=True)
    for r in results:
        print(f"[eval] {r['dataset']:<24s} {r['accuracy']:.4f}  (n={r['num_samples']})", flush=True)
    if len(results) > 1:
        mean_acc = sum(r["accuracy"] for r in results) / len(results)
        print(f"[eval] {'mean':<24s} {mean_acc:.4f}", flush=True)
        summary = out_dir / f"summary_{model_name}_eval.json"
        with open(summary, "w") as f:
            json.dump({"model": model_name, "results": results,
                       "mean_accuracy": mean_acc}, f, indent=4)
        print(f"[eval] wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
