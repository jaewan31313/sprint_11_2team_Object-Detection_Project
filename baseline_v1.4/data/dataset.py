import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class PillDataset(Dataset):
    """
    Dataset for pill object detection.
    Loads images and their merged COCO-format annotations.
    """

    def __init__(self, image_dir, annotations, class_to_idx, transforms=None):
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.class_to_idx = class_to_idx

        # annotations: list of dicts with keys 'file_name', 'boxes', 'labels'
        self.samples = annotations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.image_dir / sample["file_name"]
        image = Image.open(img_path).convert("RGB")

        boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(sample["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "file_name": sample["file_name"],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)


def parse_annotations(annotation_root):
    """
    Parses the nested annotation folder structure into a flat list.

    Each *_json folder contains one sub-folder per drug class,
    each with one JSON per image. We merge all annotations for
    the same image filename into a single record.

    Returns:
        merged: dict mapping file_name -> {'boxes': [...], 'labels': [...]}
        categories: dict mapping category_name -> category_id (original int)
    """
    annotation_root = Path(annotation_root)
    merged = {}   # file_name -> {'boxes': [], 'label_ids': []}
    categories = {}  # original category_id -> name

    for group_dir in sorted(annotation_root.iterdir()):
        if not group_dir.is_dir() or not group_dir.name.endswith("_json"):
            continue
        for drug_dir in sorted(group_dir.iterdir()):
            if not drug_dir.is_dir():
                continue
            for json_file in sorted(drug_dir.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # register categories
                for cat in data.get("categories", []):
                    categories[cat["id"]] = cat["name"]

                # get the single image entry
                if not data.get("images"):
                    continue
                file_name = data["images"][0]["file_name"]

                if file_name not in merged:
                    merged[file_name] = {"boxes": [], "label_ids": []}

                for ann in data.get("annotations", []):
                    x, y, w, h = ann["bbox"]
                    # convert to [x1, y1, x2, y2]
                    merged[file_name]["boxes"].append([x, y, x + w, y + h])
                    merged[file_name]["label_ids"].append(ann["category_id"])

    return merged, categories


def build_annotation_list(merged, categories):
    """
    Converts merged dict and categories into a flat annotation list
    and a class_to_idx mapping (1-indexed; 0 is background for Faster R-CNN).

    Returns:
        samples        : list of {'file_name', 'boxes', 'labels'}
        class_to_idx   : drug_name -> internal label (1-indexed)
        idx_to_class   : internal label -> drug_name
        idx_to_orig_id : internal label -> original category_id (dl_idx 정수값)
    """
    sorted_names = sorted(set(categories.values()))
    class_to_idx = {name: i + 1 for i, name in enumerate(sorted_names)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # original category_id (dl_idx) -> internal label
    orig_id_to_label = {
        cat_id: class_to_idx[name] for cat_id, name in categories.items()
    }
    # internal label -> original category_id (제출 시 사용)
    idx_to_orig_id = {label: cat_id for cat_id, label in orig_id_to_label.items()}

    samples = []
    for file_name, ann in merged.items():
        if not ann["boxes"]:
            continue
        labels = [orig_id_to_label[lid] for lid in ann["label_ids"]]
        samples.append({
            "file_name": file_name,
            "boxes": ann["boxes"],
            "labels": labels,
        })

    return samples, class_to_idx, idx_to_class, idx_to_orig_id


def build_dataloaders(cfg, transforms_train, transforms_val):
    """
    Builds train and validation DataLoaders from config.
    """
    data_root = Path(cfg["data"]["data_root"])
    annotation_root = data_root / cfg["data"]["train_annotations"]
    image_dir = data_root / cfg["data"]["train_images"]

    merged, categories = parse_annotations(annotation_root)
    samples, class_to_idx, idx_to_class, idx_to_orig_id = build_annotation_list(merged, categories)

    val_size = int(len(samples) * cfg["data"]["val_ratio"])
    train_size = len(samples) - val_size

    # reproducible split
    generator = torch.Generator().manual_seed(42)
    train_samples, val_samples = random_split(
        samples, [train_size, val_size], generator=generator
    )

    train_dataset = PillDataset(
        image_dir, list(train_samples), class_to_idx, transforms=transforms_train
    )
    val_dataset = PillDataset(
        image_dir, list(val_samples), class_to_idx, transforms=transforms_val
    )

    num_workers = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=PillDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=PillDataset.collate_fn,
    )

    return train_loader, val_loader, class_to_idx, idx_to_class, idx_to_orig_id


class PillTestDataset(Dataset):
    """Dataset for test images (no annotations)."""

    def __init__(self, image_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob("*.png"))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image, _ = self.transforms(image, None)
        return image, img_path.name

    @staticmethod
    def collate_fn(batch):
        images, names = zip(*batch)
        return list(images), list(names)
