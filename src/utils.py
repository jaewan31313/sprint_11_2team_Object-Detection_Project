"""
utils.py - 공통 유틸리티 함수
"""

import os
import random
import yaml
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """사용 가능한 디바이스 반환"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device


def collate_fn(batch):
    """DataLoader용 collate 함수 (가변 크기 박스 처리)"""
    return tuple(zip(*batch))


def save_checkpoint(model, optimizer, epoch, mAP, config, filename="best_model.pth"):
    """모델 체크포인트 저장"""
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    path = os.path.join(config["output"]["checkpoint_dir"], filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mAP": mAP,
    }, path)
    print(f"  체크포인트 저장: {path} (mAP: {mAP:.4f})")


def load_checkpoint(model, optimizer, path, device):
    """체크포인트 로드"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"체크포인트 로드: {path}")
    print(f"  Epoch: {checkpoint['epoch']}, mAP: {checkpoint['mAP']:.4f}")
    return checkpoint["epoch"], checkpoint["mAP"]


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    단일 박스와 여러 박스 간의 IoU 계산
    Args:
        box: [x_min, y_min, x_max, y_max]
        boxes: [[x_min, y_min, x_max, y_max], ...]
    Returns:
        IoU 배열
    """
    if len(boxes) == 0:
        return np.array([])

    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return intersection / (union + 1e-6)


def visualize_prediction(
    image: torch.Tensor,
    prediction: dict,
    classes: list,
    score_threshold: float = 0.5,
    max_detections: int = 4,
    save_path: str = None,
    title: str = ""
):
    """
    예측 결과 시각화 (바운딩 박스 + 클래스명 + 스코어)
    Args:
        image: (C, H, W) Tensor
        prediction: {"boxes", "labels", "scores"}
        classes: 클래스 이름 리스트
        score_threshold: 시각화 임계값
        max_detections: 최대 표시 박스 수
        save_path: 저장 경로 (None이면 화면 출력)
        title: 제목
    """
    COLORS = ["#FF4444", "#4CAF50", "#2196F3", "#FF9800"]

    image_np = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    # Score 임계값 + 최대 감지 수 필터링
    mask = scores > score_threshold
    boxes = boxes[mask][:max_detections]
    labels = labels[mask][:max_detections]
    scores = scores[mask][:max_detections]

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x_min, y_min, x_max, y_max = box.tolist()
        color = COLORS[i % len(COLORS)]
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 8,
            f"{classes[label]}: {score:.2f}",
            color="white", fontsize=11, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
        )

    detected = len(boxes)
    ax.set_title(f"{title}  |  감지된 알약: {detected}개", fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_sample(image_path: str, annotation: dict, classes: list):
    """Ground Truth 어노테이션 시각화 (데이터 확인용)"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box, label in zip(annotation["boxes"], annotation["labels"]):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="#FF4444", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 8, classes[label],
            color="white", fontsize=10,
            bbox=dict(facecolor="#FF4444", alpha=0.8, pad=2, edgecolor="none"),
        )

    ax.set_title(f"Ground Truth: {len(annotation['boxes'])}개 알약")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
