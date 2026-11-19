#!/usr/bin/env python
"""Create a deterministic COCO val2017 subset annotation file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-ann", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wanted = []
    for line in args.val_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        wanted.append(int(Path(line).stem))

    with args.full_ann.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    wanted_set = set(wanted)
    images = [image for image in data["images"] if int(image["id"]) in wanted_set]
    annotations = [
        annotation
        for annotation in data["annotations"]
        if int(annotation["image_id"]) in wanted_set
    ]

    if len(images) != len(wanted_set):
        missing = wanted_set - {int(image["id"]) for image in images}
        raise SystemExit(f"Missing {len(missing)} requested images: {sorted(missing)[:10]}")

    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": images,
        "annotations": annotations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle)

    print(f"Wrote {len(images)} images and {len(annotations)} annotations to {args.output}")


if __name__ == "__main__":
    main()
