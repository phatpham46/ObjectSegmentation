"""Train a pretrained YOLO11 instance segmentation model."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("n", "s", "m"), required=True, help="YOLO11 model size")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="For example 0, cpu; omit for auto")
    args = ap.parse_args()
    data = ROOT / "configs" / "data.yaml"
    if not (ROOT / "data" / "processed" / "images" / "train").exists():
        raise SystemExit("Dataset missing. Annotate images, then run: python scripts/prepare_dataset.py")
    model = YOLO(f"yolo11{args.size}-seg.pt")
    train_args = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      project=str(ROOT / "runs" / "segment"), name=f"yolo11{args.size}",
                      pretrained=True, plots=True)
    if args.device is not None: train_args["device"] = args.device
    model.train(**train_args)
if __name__ == "__main__": main()
