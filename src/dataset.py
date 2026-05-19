"""
dataset.py - 알약 감지용 Dataset & DataLoader
Pascal VOC XML 형식의 어노테이션을 지원합니다.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 800):
    return v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.RandomRotation(degrees=15),
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_val_transform(image_size: int = 800):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])


# ─────────────────────────────────────────────────────────────
# XML 파싱 유틸
# ─────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: str, classes: List[str]) -> dict:
    """
    Pascal VOC XML에서 바운딩 박스와 레이블 추출
    Returns:
        {"boxes": [[x1,y1,x2,y2], ...], "labels": [int, ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes, labels = [], []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue

        bndbox = obj.find("bndbox")
        x_min = int(float(bndbox.find("xmin").text))
        y_min = int(float(bndbox.find("ymin").text))
        x_max = int(float(bndbox.find("xmax").text))
        y_max = int(float(bndbox.find("ymax").text))

        # 유효성 검사
        if x_max > x_min and y_max > y_min:
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(classes.index(class_name))

    return {"boxes": boxes, "labels": labels}


# ─────────────────────────────────────────────────────────────
# Train / Validation Dataset
# ─────────────────────────────────────────────────────────────

class PillDataset(Dataset):
    """
    알약 감지용 Dataset (학습/검증용)
    디렉토리 구조:
        data/raw/images/         ← *.jpg
        data/raw/annotations/    ← *.xml (Pascal VOC 형식)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        classes: List[str],
        image_list: List[str],           # 확장자 없는 파일명 리스트
        transforms=None,
        max_detections: int = 4,
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.image_files = image_list
        self.transforms = transforms
        self.max_detections = max_detections

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, f"{name}.jpg")
        xml_path = os.path.join(self.annotation_dir, f"{name}.xml")

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 어노테이션 로드
        annotation = parse_voc_xml(xml_path, self.classes)
        boxes = annotation["boxes"][: self.max_detections]
        labels = annotation["labels"][: self.max_detections]

        # 빈 박스 처리 (객체 없는 이미지 대비)
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


# ─────────────────────────────────────────────────────────────
# Test Dataset
# ─────────────────────────────────────────────────────────────

class PillTestDataset(Dataset):
    """테스트용 Dataset (레이블 없음)"""

    def __init__(
        self,
        image_dir: str,
        image_list: List[str],
        transforms=None,
    ):
        self.image_dir = image_dir
        self.image_files = image_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, f"{name}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, name


# ─────────────────────────────────────────────────────────────
# 데이터 분할 유틸
# ─────────────────────────────────────────────────────────────

def get_valid_image_list(image_dir: str, annotation_dir: str) -> List[str]:
    """XML 어노테이션이 존재하는 이미지만 필터링"""
    xml_names = {
        Path(f).stem
        for f in os.listdir(annotation_dir)
        if f.endswith(".xml")
    }
    img_names = {
        Path(f).stem
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    valid = sorted(xml_names & img_names)
    print(f"유효한 이미지 수: {len(valid)}  "
          f"(이미지: {len(img_names)}, XML: {len(xml_names)})")
    return valid


def split_dataset(
    image_list: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Train / Val / Test 분할"""
    test_ratio = 1.0 - train_ratio - val_ratio
    train, temp = train_test_split(image_list, test_size=(1 - train_ratio), random_state=seed)
    val, test = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed)
    print(f"데이터 분할 → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# DataLoader 생성
# ─────────────────────────────────────────────────────────────

def build_dataloaders(config: dict):
    """
    config.yaml을 받아 Train/Val/Test DataLoader 반환
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    from src.utils import collate_fn

    classes = config["data"]["classes"]
    image_dir = config["data"]["image_dir"]
    annotation_dir = config["data"]["annotation_dir"]
    image_size = config["train"]["image_size"]
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    max_det = config["model"]["max_detections"]

    # 유효 이미지 목록
    all_images = get_valid_image_list(image_dir, annotation_dir)

    # 분할
    train_list, val_list, test_list = split_dataset(
        all_images,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["project"]["seed"],
    )

    # Dataset
    train_dataset = PillDataset(
        image_dir, annotation_dir, classes, train_list,
        transforms=get_train_transform(image_size),
        max_detections=max_det,
    )
    val_dataset = PillDataset(
        image_dir, annotation_dir, classes, val_list,
        transforms=get_val_transform(image_size),
        max_detections=max_det,
    )
    test_dataset = PillTestDataset(
        image_dir, test_list,
        transforms=get_val_transform(image_size),
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, classes
