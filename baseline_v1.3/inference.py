"""
Runs inference on the test set using a trained checkpoint.

Usage:
    python inference.py
    python inference.py --checkpoint outputs/checkpoints/best.pth
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import PillTestDataset
from data.transforms import get_transforms
from models.detector import build_model, load_checkpoint
from utils.coco_utils import export_predictions_coco, export_predictions_csv, load_class_mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pth",
        help="Path to trained model checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    class_to_idx, idx_to_class, idx_to_orig_id = load_class_mapping(ckpt_dir / "class_mapping.json")
    num_classes = len(class_to_idx)

    model = build_model(num_classes, cfg)
    model, _, _ = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    transforms = get_transforms(cfg, split="val")
    data_root = Path(cfg["data"]["data_root"])
    test_dir = data_root / cfg["data"]["test_images"]

    dataset = PillTestDataset(test_dir, transforms=transforms)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=PillTestDataset.collate_fn,
    )

    all_predictions = []

    with torch.no_grad():
        for images, file_names in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, fname in zip(outputs, file_names):
                boxes = out["boxes"].cpu().tolist()
                labels = out["labels"].cpu().tolist()
                scores = out["scores"].cpu().tolist()

                # keep top-4 detections per image
                max_det = cfg["inference"]["max_detections"]
                all_predictions.append({
                    "file_name": fname,
                    "boxes": boxes[:max_det],
                    "labels": labels[:max_det],
                    "scores": scores[:max_det],
                })

            print(f"Processed {len(all_predictions)}/{len(dataset)} images", end="\r")

    pred_dir = Path(cfg["output"]["prediction_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    export_predictions_coco(all_predictions, idx_to_class, pred_dir / "predictions.json")
    export_predictions_csv(all_predictions, idx_to_orig_id, pred_dir / "submission.csv")
    print(f"\nDone.")
    print(f"  JSON : {pred_dir / 'predictions.json'}")
    print(f"  CSV  : {pred_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
