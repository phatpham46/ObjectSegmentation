"""Prepare annotated LabelMe/YOLO polygon labels into YOLO segmentation splits."""
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "labeling" / "images"
LABEL_DIR = ROOT / "labeling" / "labels"
OUT = ROOT / "data" / "processed"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.val < 1:
        ap.error("--val must be between 0 and 1")
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTS)
    if len(images) < 2:
        raise SystemExit(f"Put labeled images in {IMAGE_DIR}; at least 2 are required.")
    missing = [p.name for p in images if not (LABEL_DIR / f"{p.stem}.txt").is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} YOLO polygon label files in {LABEL_DIR}; first: {missing[0]}")
    rng = random.Random(args.seed)
    rng.shuffle(images)
    nval = max(1, min(len(images)-1, round(len(images)*args.val)))
    split = {"val": images[:nval], "train": images[nval:]}
    for name, group in split.items():
        for kind in ("images", "labels"):
            dest = OUT / kind / name
            dest.mkdir(parents=True, exist_ok=True)
            for p in dest.iterdir():
                if p.is_file(): p.unlink()
        for image in group:
            shutil.copy2(image, OUT / "images" / name / image.name)
            shutil.copy2(LABEL_DIR / f"{image.stem}.txt", OUT / "labels" / name / f"{image.stem}.txt")
    (ROOT / "configs" / "data.yaml").write_text(
        "path: data/processed\ntrain: images/train\nval: images/val\nnames:\n  0: pet\n", encoding="utf-8")
    print(f"Prepared {len(split['train'])} train + {len(split['val'])} validation images at {OUT}")

if __name__ == "__main__": main()
