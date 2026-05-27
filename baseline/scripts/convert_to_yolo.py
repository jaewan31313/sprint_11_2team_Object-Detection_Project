"""
어노테이션을 YOLO 형식으로 변환합니다.

YOLO 형식: class_idx cx cy w h (0~1 정규화)

실행:
    python scripts/convert_to_yolo.py
"""
import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import build_annotation_list, parse_annotations


def convert(cfg_path="configs/default.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    annotation_root = data_root / cfg["data"]["train_annotations"]
    train_image_dir  = data_root / cfg["data"]["train_images"]
    test_image_dir   = data_root / cfg["data"]["test_images"]

    yolo_dir = Path("data/yolo")
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("어노테이션 파싱 중...")
    merged, categories = parse_annotations(annotation_root)
    samples, class_to_idx, idx_to_class, idx_to_orig_id = build_annotation_list(merged, categories)

    # YOLO는 0-indexed
    yolo_class_to_idx = {name: i for i, name in enumerate(sorted(class_to_idx.keys()))}
    idx_to_name = {v: k for k, v in class_to_idx.items()}

    # train/val split (train.py와 동일한 seed)
    import torch
    generator = torch.Generator().manual_seed(42)
    val_size   = int(len(samples) * cfg["data"]["val_ratio"])
    train_size = len(samples) - val_size
    from torch.utils.data import random_split
    train_samples, val_samples = random_split(samples, [train_size, val_size], generator=generator)

    def write_split(split_samples, split_name):
        for sample in split_samples:
            img_path = train_image_dir / sample["file_name"]
            img = Image.open(img_path)
            W, H = img.size

            # 이미지 복사
            dst_img = yolo_dir / "images" / split_name / sample["file_name"]
            shutil.copy2(img_path, dst_img)

            # 라벨 txt 생성
            dst_lbl = yolo_dir / "labels" / split_name / (Path(sample["file_name"]).stem + ".txt")
            lines = []
            for box, label in zip(sample["boxes"], sample["labels"]):
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w  = (x2 - x1) / W
                h  = (y2 - y1) / H
                # label은 1-indexed → 0-indexed로 변환
                cls = label - 1
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            dst_lbl.write_text("\n".join(lines))

    print(f"train {train_size}개, val {val_size}개 변환 중...")
    write_split(list(train_samples), "train")
    write_split(list(val_samples), "val")

    # 테스트 이미지 복사 (라벨 없음)
    print("test 이미지 복사 중...")
    for img_path in sorted(test_image_dir.glob("*.png")):
        shutil.copy2(img_path, yolo_dir / "images" / "test" / img_path.name)

    # dataset.yaml 생성
    class_names = [k for k, v in sorted(yolo_class_to_idx.items(), key=lambda x: x[1])]
    dataset_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, default_flow_style=False)

    # 원본 ID 매핑 저장 (추론 시 category_id 복원용)
    orig_id_map = {i: idx_to_orig_id[i + 1] for i in range(len(class_names))}
    with open(yolo_dir / "orig_id_map.json", "w") as f:
        json.dump(orig_id_map, f, ensure_ascii=False, indent=2)

    print(f"\n완료!")
    print(f"  클래스 수    : {len(class_names)}")
    print(f"  dataset.yaml : {yaml_path}")


if __name__ == "__main__":
    convert()
