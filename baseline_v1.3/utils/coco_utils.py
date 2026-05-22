import csv
import json
from pathlib import Path


def export_predictions_coco(predictions, idx_to_class, output_path):
    """
    Saves model predictions to a JSON file in COCO results format.

    predictions: list of dicts with keys:
        'file_name', 'boxes' (x1,y1,x2,y2), 'labels', 'scores'
    """
    results = []
    for pred in predictions:
        file_name = pred["file_name"]
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x1, y1, x2, y2 = box
            results.append({
                "file_name": file_name,
                "category_name": idx_to_class.get(int(label), "unknown"),
                "label": int(label),
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")
    return results


def export_predictions_csv(predictions, idx_to_orig_id, output_path):
    """
    Saves predictions to CSV matching the submission format:
        annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score

    image_id    : 파일명 숫자 추출 (예: '42.png' → 42)
    category_id : 원본 dl_idx 정수값 (예: 1900, 16548) — Kaggle 채점 기준
    bbox        : x, y, w, h 형식
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    ann_id = 1
    for pred in predictions:
        stem = Path(pred["file_name"]).stem
        image_id = int(stem) if stem.isdigit() else stem

        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x1, y1, x2, y2 = box
            # 내부 label → 원본 category_id(dl_idx) 변환
            orig_cat_id = idx_to_orig_id.get(int(label), int(label))
            rows.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": orig_cat_id,
                "bbox_x": round(float(x1), 2),
                "bbox_y": round(float(y1), 2),
                "bbox_w": round(float(x2 - x1), 2),
                "bbox_h": round(float(y2 - y1), 2),
                "score": round(float(score), 4),
            })
            ann_id += 1

    fieldnames = ["annotation_id", "image_id", "category_id",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    return rows


def load_class_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_to_idx = data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
    idx_to_orig_id = {int(k): v for k, v in data["idx_to_orig_id"].items()}
    return class_to_idx, idx_to_class, idx_to_orig_id


def save_class_mapping(class_to_idx, idx_to_class, idx_to_orig_id, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "idx_to_orig_id": idx_to_orig_id,
            },
            f, ensure_ascii=False, indent=2,
        )
