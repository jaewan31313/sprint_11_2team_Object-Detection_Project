"""
YOLO11 앙상블 추론 스크립트 — WBF(Weighted Box Fusion) 사용

실행:
    python inference_yolo_ensemble.py \
        --checkpoints outputs/yolo/train_m/weights/best.pt \
                      outputs/yolo/train_l/weights/best.pt
"""
import argparse
import csv
import json
from pathlib import Path

import yaml
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="모델 체크포인트 경로 (여러 개 입력)")
    parser.add_argument("--conf",        type=float, default=0.2,  help="confidence threshold")
    parser.add_argument("--iou",         type=float, default=0.5,  help="NMS IoU threshold")
    parser.add_argument("--wbf_iou",     type=float, default=0.5,  help="WBF IoU threshold")
    parser.add_argument("--max_det",     type=int,   default=4,    help="이미지당 최대 검출 수")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    from ensemble_boxes import weighted_boxes_fusion

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    test_dir  = data_root / cfg["data"]["test_images"]
    pred_dir  = Path(cfg["output"]["prediction_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)

    orig_id_map_path = Path("data/yolo/orig_id_map.json")
    with open(orig_id_map_path) as f:
        orig_id_map = {int(k): v for k, v in json.load(f).items()}

    models = [YOLO(ckpt) for ckpt in args.checkpoints]
    print(f"모델 {len(models)}개 로드 완료")
    for ckpt in args.checkpoints:
        print(f"  - {ckpt}")

    test_images = sorted(test_dir.glob("*.png"))
    print(f"\n테스트 이미지 {len(test_images)}개 앙상블 추론 중...")

    rows = []
    ann_id = 1

    for img_path in test_images:
        img = Image.open(img_path)
        W, H = img.size

        stem = img_path.stem
        image_id = int(stem) if stem.isdigit() else stem

        all_boxes, all_scores, all_labels = [], [], []

        for model in models:
            result = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                augment=True,
                save=False,
                verbose=False,
            )[0]

            boxes  = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            labels = result.boxes.cls.cpu().tolist()

            # WBF는 [0,1] 정규화 좌표 필요
            norm_boxes = [
                [max(0.0, x1/W), max(0.0, y1/H), min(1.0, x2/W), min(1.0, y2/H)]
                for x1, y1, x2, y2 in boxes
            ]
            all_boxes.append(norm_boxes)
            all_scores.append(scores)
            all_labels.append([int(l) for l in labels])

        # 모든 모델 예측이 비어있으면 스킵
        if not any(b for b in all_boxes):
            continue

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=args.wbf_iou,
            skip_box_thr=args.conf,
        )

        # max_det 제한 (score 높은 순)
        if len(merged_scores) > args.max_det:
            top_idx = sorted(range(len(merged_scores)),
                             key=lambda i: merged_scores[i], reverse=True)[:args.max_det]
            merged_boxes  = [merged_boxes[i]  for i in top_idx]
            merged_scores = [merged_scores[i] for i in top_idx]
            merged_labels = [merged_labels[i] for i in top_idx]

        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            x1, y1, x2, y2 = box[0]*W, box[1]*H, box[2]*W, box[3]*H
            orig_cat_id = orig_id_map.get(int(label), int(label))
            rows.append({
                "annotation_id": ann_id,
                "image_id":      image_id,
                "category_id":   orig_cat_id,
                "bbox_x":        round(x1, 2),
                "bbox_y":        round(y1, 2),
                "bbox_w":        round(x2 - x1, 2),
                "bbox_h":        round(y2 - y1, 2),
                "score":         round(float(score), 4),
            })
            ann_id += 1

    fieldnames = ["annotation_id", "image_id", "category_id",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    csv_path = pred_dir / "submission_ensemble.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n완료 — {len(rows)}개 예측 저장 → {csv_path}")


if __name__ == "__main__":
    main()
