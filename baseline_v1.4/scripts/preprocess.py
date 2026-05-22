"""
Parses the raw annotation structure and saves a merged COCO-format JSON
and a class mapping file to data/processed/.

Run once before training:
    python scripts/preprocess.py
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import build_annotation_list, parse_annotations
from utils.coco_utils import save_class_mapping


def main():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    annotation_root = data_root / cfg["data"]["train_annotations"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing annotations...")
    merged, categories = parse_annotations(annotation_root)
    samples, class_to_idx, idx_to_class, idx_to_orig_id = build_annotation_list(merged, categories)

    print(f"  Images with annotations : {len(samples)}")
    print(f"  Unique pill classes     : {len(class_to_idx)}")

    # save merged annotation list
    ann_path = processed_dir / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  Saved annotations  -> {ann_path}")

    # save class mapping
    mapping_path = processed_dir / "class_mapping.json"
    save_class_mapping(class_to_idx, idx_to_class, idx_to_orig_id, mapping_path)
    print(f"  Saved class mapping -> {mapping_path}")

    # print a few sample entries for sanity check
    print("\nSample annotations (first 3):")
    for s in samples[:3]:
        print(f"  {s['file_name']} | boxes={len(s['boxes'])} | labels={s['labels']}")


if __name__ == "__main__":
    main()
