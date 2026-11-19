#!/usr/bin/env python
"""Run a D-FINE checkpoint on COCO val2017 and capture predictions and metrics."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "D-FINE"

PAPER_TARGETS = {
    "s": [48.5, 65.6, 52.6, 29.1, 52.2, 65.4],
    "l": [54.0, 71.6, 58.4, 36.5, 58.0, 71.9],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--skip-predictions", action="store_true")
    return parser.parse_args()


def resolve_from_launch_dir(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def to_coco_detections(result: dict, image_id: int) -> list[dict]:
    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result["labels"].detach().cpu()
    detections = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1, y1, x2, y2 = box
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return detections


def main() -> None:
    args = parse_args()
    config_path = resolve_from_launch_dir(args.config)
    checkpoint_path = resolve_from_launch_dir(args.checkpoint)
    output_dir = resolve_from_launch_dir(args.output_dir)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    from src.core import YAMLConfig
    from src.solver import TASKS
    from src.solver.det_engine import evaluate

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"predictions_{args.tag}.jsonl"
    flat_predictions_path = output_dir / f"predictions_{args.tag}.json"
    metrics_path = output_dir / f"metrics_{args.tag}.json"
    eval_path = output_dir / f"eval_{args.tag}.pth"

    cfg = YAMLConfig(
        str(config_path),
        **{
            "device": "cpu",
            "resume": str(checkpoint_path),
            "test_only": True,
        },
    )
    solver = TASKS[cfg.task](cfg)
    solver.eval()

    module = solver.ema.module if solver.ema else solver.model
    forward_times: list[tuple[int, float]] = []
    original_forward = module.forward

    def timed_forward(samples, *forward_args, **forward_kwargs):
        start = time.perf_counter()
        output = original_forward(samples, *forward_args, **forward_kwargs)
        forward_times.append((int(samples.shape[0]), time.perf_counter() - start))
        return output

    module.forward = timed_forward

    original_update = solver.evaluator.update
    jsonl = None if args.skip_predictions else predictions_path.open("w", encoding="utf-8")

    def capture_update(res):
        if jsonl is not None:
            for image_id, result in res.items():
                detections = to_coco_detections(result, int(image_id))
                jsonl.write(
                    json.dumps(
                        {"image_id": int(image_id), "detections": detections},
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        original_update(res)

    solver.evaluator.update = capture_update

    n_images = len(solver.val_dataloader.dataset)
    process = psutil.Process()
    start = time.perf_counter()
    stats, coco_evaluator = evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        epoch=-1,
        use_wandb=False,
    )
    elapsed = time.perf_counter() - start

    if jsonl is not None:
        jsonl.close()
        with flat_predictions_path.open("w", encoding="utf-8") as output:
            output.write("[")
            first = True
            with predictions_path.open("r", encoding="utf-8") as source:
                for line in source:
                    for detection in json.loads(line)["detections"]:
                        if not first:
                            output.write(",")
                        output.write(json.dumps(detection, ensure_ascii=True))
                        first = False
            output.write("]")

    torch.save(coco_evaluator.coco_eval["bbox"].eval, eval_path)

    forward_seconds = sum(seconds for _, seconds in forward_times)
    forward_images = sum(batch_size for batch_size, _ in forward_times)
    target_key = "s" if args.tag.lower().startswith("dfine_s") else "l" if args.tag.lower().startswith("dfine_l") else None

    metrics = {
        "tag": args.tag,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "image_count": n_images,
        "coco_eval_bbox": stats["coco_eval_bbox"],
        "metric_names": [
            "AP",
            "AP50",
            "AP75",
            "APs",
            "APm",
            "APl",
            "AR1",
            "AR10",
            "AR100",
            "ARs",
            "ARm",
            "ARl",
        ],
        "paper_target": PAPER_TARGETS.get(target_key),
        "wall_time_seconds": elapsed,
        "seconds_per_image": elapsed / max(n_images, 1),
        "forward_seconds": forward_seconds,
        "forward_images": forward_images,
        "forward_seconds_per_image": forward_seconds / max(forward_images, 1),
        "peak_rss_bytes": process.memory_info().rss,
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "predictions_jsonl": None if args.skip_predictions else str(predictions_path),
        "predictions_json": None if args.skip_predictions else str(flat_predictions_path),
    }
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)

    print(f"Wrote metrics to {metrics_path}")
    if not args.skip_predictions:
        print(f"Wrote predictions to {predictions_path} and {flat_predictions_path}")


if __name__ == "__main__":
    main()
