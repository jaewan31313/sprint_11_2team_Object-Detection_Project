"""
검증셋에서 예측이 틀린 이미지를 시각화합니다.

초록 박스 : GT (정답)
빨강 박스 : 예측 (모델 출력)

Usage:
    python scripts/visualize_errors.py
    python scripts/visualize_errors.py --n 20 --iou_thr 0.5 --output outputs/errors
"""
import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# 코랩 한글 폰트 설정
_nanum = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if Path(_nanum).exists():
    fm.fontManager.addfont(_nanum)
    plt.rcParams['font.family'] = 'NanumGothic'
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import build_annotation_list, parse_annotations, PillDataset
from data.transforms import get_transforms
from models.detector import build_model, load_checkpoint
from utils.coco_utils import load_class_mapping
from utils.metrics import box_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--n", type=int, default=10, help="시각화할 이미지 수")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU 임계값")
    parser.add_argument("--output", default="outputs/errors")
    return parser.parse_args()


def is_wrong(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr):
    """예측이 하나라도 틀렸으면 True 반환."""
    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        return True
    if len(pred_boxes) > 0 and len(gt_boxes) == 0:
        return True

    matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    for i in range(len(pred_boxes)):
        if len(gt_boxes) == 0:
            return True
        ious = box_iou(pred_boxes[i:i+1], gt_boxes)[0]
        best_iou, best_j = ious.max(0)
        if best_iou >= iou_thr and gt_labels[best_j] == pred_labels[i] and not matched[best_j]:
            matched[best_j] = True

    return not matched.all()


def draw(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, idx_to_class, title):
    fig, ax = plt.subplots(1, figsize=(8, 10))
    ax.imshow(image)

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"GT:{idx_to_class.get(int(label), label)}",
                color="lime", fontsize=6, backgroundcolor="black")

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y2 + 12, f"Pred:{idx_to_class.get(int(label), label)} {score:.2f}",
                color="red", fontsize=6, backgroundcolor="black")

    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return fig


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    class_to_idx, idx_to_class, idx_to_orig_id = load_class_mapping(ckpt_dir / "class_mapping.json")
    num_classes = len(class_to_idx)

    model = build_model(num_classes, cfg)
    model, _, _ = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    from pathlib import Path as P
    data_root = P(cfg["data"]["data_root"])
    annotation_root = data_root / cfg["data"]["train_annotations"]
    image_dir = data_root / cfg["data"]["train_images"]

    merged, categories = parse_annotations(annotation_root)
    samples, _, _, _ = build_annotation_list(merged, categories)

    # val split (train.py와 동일한 seed)
    import torch as th
    generator = th.Generator().manual_seed(42)
    val_size = int(len(samples) * cfg["data"]["val_ratio"])
    train_size = len(samples) - val_size
    from torch.utils.data import random_split
    _, val_samples = random_split(samples, [train_size, val_size], generator=generator)

    transforms = get_transforms(cfg, split="val")
    val_dataset = PillDataset(image_dir, list(val_samples), class_to_idx, transforms=transforms)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        collate_fn=PillDataset.collate_fn)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for images, targets in loader:
            if saved >= args.n:
                break

            images_gpu = [img.to(device) for img in images]
            outputs = model(images_gpu)

            for img_tensor, out, tgt in zip(images, outputs, targets):
                pred_boxes  = out["boxes"].cpu()
                pred_labels = out["labels"].cpu()
                pred_scores = out["scores"].cpu()
                gt_boxes    = tgt["boxes"]
                gt_labels   = tgt["labels"]
                file_name   = tgt["file_name"]

                if not is_wrong(pred_boxes, pred_labels, gt_boxes, gt_labels, args.iou_thr):
                    continue

                # tensor → PIL
                img_pil = Image.fromarray(
                    (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
                )

                fig = draw(img_pil, gt_boxes.tolist(), gt_labels.tolist(),
                           pred_boxes.tolist(), pred_labels.tolist(), pred_scores.tolist(),
                           idx_to_class, title=file_name)

                out_path = output_dir / (Path(file_name).stem + ".jpg")
                fig.savefig(out_path, bbox_inches="tight", dpi=100)
                plt.close(fig)
                print(f"[{saved+1}/{args.n}] 저장: {out_path}")
                saved += 1

    print(f"\n총 {saved}개 저장 완료 → {output_dir}")


if __name__ == "__main__":
    main()
