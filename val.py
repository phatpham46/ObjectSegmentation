"""Validate a trained YOLO segmentation checkpoint and print key metrics."""
import argparse
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="Path to best.pt")
    ap.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to evaluate (default: test)")
    args=ap.parse_args()
    metrics=YOLO(args.checkpoint).val(
        data=str(ROOT/"configs"/"data.yaml"),
        split=args.split,
        plots=True,
        project=str(ROOT/"runs"/"validation"),
        name=Path(args.checkpoint).parent.parent.name,
        exist_ok=True,
    )
    print(f"Box mAP50-95: {metrics.box.map:.4f}; Box mAP50: {metrics.box.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}; Mask mAP50: {metrics.seg.map50:.4f}")
if __name__ == "__main__": main()
