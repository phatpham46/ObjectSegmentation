"""Import a Roboflow COCO Segmentation ZIP as Ultralytics YOLO segmentation data."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "coco-segmentation.zip")
    args = parser.parse_args()
    if not args.zip.is_file():
        raise SystemExit(f"COCO ZIP not found: {args.zip}")

    output = ROOT / "data" / "processed"
    existing = [p for p in output.rglob("*") if p.is_file()] if output.exists() else []
    if existing:
        raise SystemExit(f"Output already contains files ({existing[0]}). Move it aside before importing.")

    totals: dict[str, int] = {}
    with zipfile.ZipFile(args.zip) as archive:
        members = {name: name for name in archive.namelist()}
        for source_split, target_split in SPLITS.items():
            json_name = f"{source_split}/_annotations.coco.json"
            if json_name not in members:
                raise SystemExit(f"Missing {json_name} in {args.zip}")
            coco = json.loads(archive.read(json_name))
            pet_ids = {c["id"] for c in coco.get("categories", []) if c["name"].strip().lower() == "pet"}
            if not pet_ids:
                raise SystemExit(f"No 'pet' category in {json_name}")

            images = {item["id"]: item for item in coco.get("images", [])}
            annotations: dict[int, list[dict]] = defaultdict(list)
            for annotation in coco.get("annotations", []):
                if annotation.get("category_id") in pet_ids:
                    annotations[annotation["image_id"]].append(annotation)

            image_dir = output / "images" / target_split
            label_dir = output / "labels" / target_split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for image_id, image in images.items():
                filename = PurePosixPath(image["file_name"]).name
                archive_name = f"{source_split}/{filename}"
                if archive_name not in members:
                    raise SystemExit(f"Missing image {archive_name} in ZIP")
                width, height = image["width"], image["height"]
                rows: list[str] = []
                for annotation in annotations.get(image_id, []):
                    segmentation = annotation.get("segmentation")
                    if not isinstance(segmentation, list):
                        raise SystemExit(f"Expected polygon segmentation for image {filename}")
                    for polygon in segmentation:
                        if len(polygon) < 6 or len(polygon) % 2:
                            continue
                        coords = []
                        for index, value in enumerate(polygon):
                            limit = width if index % 2 == 0 else height
                            coords.append(f"{min(1.0, max(0.0, value / limit)):.6f}")
                        rows.append("0 " + " ".join(coords))
                if not rows:
                    raise SystemExit(f"No usable pet polygon for image {filename}")
                (image_dir / filename).write_bytes(archive.read(archive_name))
                (label_dir / f"{Path(filename).stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            totals[target_split] = len(images)

    config = ROOT / "configs" / "data.yaml"
    config.write_text(
        "path: data/processed\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pet\n",
        encoding="utf-8",
    )
    print("Imported COCO polygons as YOLO segmentation labels:")
    for split, count in totals.items():
        print(f"  {split}: {count} images")
    print(f"Dataset: {output}")


if __name__ == "__main__":
    main()
