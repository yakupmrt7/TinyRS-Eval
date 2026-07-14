#!/usr/bin/env python
"""Standalone visual-grounding benchmark for Qwen3.5-VL models via vLLM.

Companion to eval_cls_vllm.py (the stock rs_evaluation.py cannot run in the
recot-eval env). Reproduces TinyRS's exact bbox scoring: IoU@0.5 precision with
an S/M/L area breakdown (see eval_results_bbox in rs_evaluation.py).

Predicted boxes are normalized to [0, --bbox_normalize_bound] and are rescaled to
pixels by (coord * w / bound). The Re-CoT VG model is trained on boxes normalized
to 0-800, hence the 800 default. Ground-truth boxes in the annotation file are
already absolute pixels.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

Image.MAX_IMAGE_PIXELS = None

BBOX_RE = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")
AREA_LEVEL = (32**2, 96**2, float("inf"))
LEVEL_NAME = ("S", "M", "L")


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_root", required=True,
                   help="folder containing the *.json annotation file")
    p.add_argument("--anns_json", default="VG_DOIR_RSVG_test.json")
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
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--bbox_normalize_bound", type=float, default=800,
                   help="range the model emits boxes in; Re-CoT VG is trained on 0-800")
    return p.parse_args()


def resolve_image_path(anns_json_path: Path, anns: dict) -> Path:
    """Mirror infer_single() image resolution from rs_evaluation.py."""
    fn = anns["image"]
    if "image_path" not in anns:
        return anns_json_path.parent / anns_json_path.stem / fn
    if Path(anns["image_path"]).is_absolute():
        return Path(anns["image_path"]) / fn
    return anns_json_path.parent / anns["image_path"] / fn


def extract_bboxes(text: str) -> list:
    """All [x1,y1,x2,y2] groups in a string, after stripping reasoning/answer tags."""
    pred = str(text).strip()
    if "<reasoning>" in pred:
        try:
            pred = pred.split("<reasoning>")[1].split("</reasoning>")[1]
        except Exception:
            pass
    if "<answer>" in pred:
        try:
            pred = pred.split("<answer>")[1].split("</answer>")[0]
        except Exception:
            pass
    return [[float(x) for x in m] for m in BBOX_RE.findall(pred)]


def area(box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou(box1, box2) -> float:
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    x_min_int, y_min_int = max(x_min1, x_min2), max(y_min1, y_min2)
    x_max_int, y_max_int = min(x_max1, x_max2), min(y_max1, y_max2)
    inter = max(0.0, x_max_int - x_min_int) * max(0.0, y_max_int - y_min_int)
    union = area(box1) + area(box2) - inter
    return inter / union if union > 0 else 0.0


def area_level(box) -> int:
    a = area(box)
    for i, bound in enumerate(AREA_LEVEL):
        if a <= bound:
            return i
    return len(AREA_LEVEL) - 1


def main():
    args = arg_parser()
    data_root = Path(args.data_root)
    anns_json_path = data_root / args.anns_json
    test_name = anns_json_path.stem
    model_name = args.model_name or Path(args.model_path).name
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl = out_dir / f"{test_name}_{model_name}_eval.jsonl"
    save_metric = out_dir / f"{test_name}_{model_name}_eval.json"

    with open(anns_json_path) as f:
        anns_dict = json.load(f)
    if args.limit is not None:
        anns_dict = anns_dict[: args.limit]
    print(f"[eval] {test_name}: {len(anns_dict)} samples | model={model_name} "
          f"| bbox bound={args.bbox_normalize_bound}", flush=True)

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

    records, llm_inputs = [], []
    for anns in anns_dict:
        convs = anns["conversations"]
        question = convs[0]["value"].replace("<image>\n", "").strip()
        answer = convs[1]["value"]
        img_path = resolve_image_path(anns_json_path, anns)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        llm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        records.append({"filename": str(img_path), "size": [h, w],
                        "query": question, "answer": answer})

    outputs = llm.generate(llm_inputs, sampling_params=sampling)

    level_count = np.zeros(len(AREA_LEVEL))
    level_hit = np.zeros(len(AREA_LEVEL))
    bound = args.bbox_normalize_bound
    lines = []

    for rec, out in zip(records, outputs):
        raw = out.outputs[0].text
        h, w = rec["size"]
        gt_boxes = extract_bboxes(rec["answer"])       # already absolute pixels
        pred_boxes = extract_bboxes(raw)               # normalized to [0, bound]

        ious = []
        for i, gt in enumerate(gt_boxes):
            lvl = area_level(gt)
            level_count[lvl] += 1

            if i < len(pred_boxes):
                p = pred_boxes[i]
                pred_px = [p[0] * w / bound, p[1] * h / bound,
                           p[2] * w / bound, p[3] * h / bound]
                iou = calculate_iou(gt, pred_px)
                if iou >= args.iou_threshold:
                    level_hit[lvl] += 1
            else:
                iou = 0.0                              # model emitted no box for this gt
            ious.append(iou)

        rec.update({"pred": raw, "pred_bbox": pred_boxes,
                    "iou": ious, "score": bool(ious and max(ious) >= args.iou_threshold)})
        lines.append(json.dumps(rec))

    total = level_count.sum()
    precision = level_hit.sum() / total if total else 0.0
    level_precision = np.divide(level_hit, level_count,
                                out=np.zeros_like(level_hit), where=level_count > 0)

    metrics = {"dataset": test_name, "model": model_name,
               "num_samples": len(records), "num_boxes": int(total),
               "iou_threshold": args.iou_threshold,
               "bbox_normalize_bound": bound,
               "precision": float(precision)}
    for i, name in enumerate(LEVEL_NAME):
        metrics[f"precision_{name}"] = float(level_precision[i])
        metrics[f"count_{name}"] = int(level_count[i])

    with open(save_jsonl, "w") as f:
        f.write("\n".join(lines))
    with open(save_metric, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[eval] {test_name} precision@IoU{args.iou_threshold}: {precision:.4f} "
          f"({int(level_hit.sum())}/{int(total)})", flush=True)
    for i, name in enumerate(LEVEL_NAME):
        print(f"[eval]   precision_{name}: {level_precision[i]:.4f} "
              f"(n={int(level_count[i])})", flush=True)
    print(f"[eval] wrote {save_jsonl}\n[eval] wrote {save_metric}", flush=True)


if __name__ == "__main__":
    main()
