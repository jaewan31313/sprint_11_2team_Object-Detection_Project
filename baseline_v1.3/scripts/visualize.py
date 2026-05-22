"""
Visualizes ground-truth bounding boxes on training images.

Usage:
    python scripts/visualize.py --n 5 --output outputs/vis
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def draw_boxes(image, boxes, labels, title=""):
    fig, ax = plt.subplots(1, figsize=(8, 10))
    ax.imshow(image)
    colors = ["red", "blue", "green", "orange"]
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colors[i % len(colors)], facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, str(label), color=colors[i % len(colors)], fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--output", type=str, default="outputs/vis")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    image_dir = data_root / cfg["data"]["train_images"]
    ann_path = Path(cfg["data"]["processed_dir"]) / "annotations.json"

    if not ann_path.exists():
        print("Run scripts/preprocess.py first to generate annotations.json")
        return

    with open(ann_path, encoding="utf-8") as f:
        samples = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples[:args.n]:
        img = Image.open(image_dir / sample["file_name"]).convert("RGB")
        fig = draw_boxes(img, sample["boxes"], sample["labels"], title=sample["file_name"])
        out_path = output_dir / (Path(sample["file_name"]).stem + ".jpg")
        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
