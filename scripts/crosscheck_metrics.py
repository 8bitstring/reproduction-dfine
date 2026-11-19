#!/usr/bin/env python
"""Cross-check saved D-FINE predictions with pycocotools."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--official-metrics")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    image_ids = sorted({int(item["image_id"]) for item in predictions})
    coco_gt = COCO(args.gt)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = [float(value) for value in coco_eval.stats]
    result = {
        "image_count": len(image_ids),
        "prediction_count": len(predictions),
        "pycocotools": stats,
        "metric_names": METRIC_NAMES,
    }

    if args.official_metrics:
        with Path(args.official_metrics).open("r", encoding="utf-8") as handle:
            official = json.load(handle)
        result["official"] = official["coco_eval_bbox"]
        result["differences"] = [
            float(py - official_value)
            for py, official_value in zip(stats, official["coco_eval_bbox"])
        ]
        result["max_abs_difference"] = float(
            np.max(np.abs(result["differences"]))
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
