"""
evaluate.py - mAP@0.5 평가 지표
"""

import numpy as np
import torch
from typing import List


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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


def calculate_ap_per_class(
    predictions: List[dict],
    ground_truths: List[dict],
    class_idx: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    특정 클래스에 대한 Average Precision 계산
    Args:
        predictions: [{"boxes": Tensor, "labels": Tensor, "scores": Tensor}, ...]
        ground_truths: [{"boxes": Tensor, "labels": Tensor}, ...]
        class_idx: 평가 클래스 인덱스
        iou_threshold: IoU 임계값
    Returns:
        AP (float)
    """
    true_positives = []
    scores_list = []
    num_ground_truths = 0

    for pred, gt in zip(predictions, ground_truths):
        # 해당 클래스만 필터
        pred_mask = pred["labels"] == class_idx
        gt_mask = gt["labels"] == class_idx

        pred_boxes = pred["boxes"][pred_mask].cpu().numpy() if pred_mask.any() else np.zeros((0, 4))
        pred_scores = pred["scores"][pred_mask].cpu().numpy() if pred_mask.any() else np.array([])
        gt_boxes = gt["boxes"][gt_mask].cpu().numpy() if gt_mask.any() else np.zeros((0, 4))

        num_ground_truths += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Score 내림차순 정렬
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        matched = np.zeros(len(gt_boxes), dtype=bool)

        for box, score in zip(pred_boxes, pred_scores):
            if len(gt_boxes) == 0:
                true_positives.append(0)
                scores_list.append(score)
                continue

            ious = calculate_iou(box, gt_boxes)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold and not matched[max_iou_idx]:
                true_positives.append(1)
                matched[max_iou_idx] = True
            else:
                true_positives.append(0)

            scores_list.append(score)

    if num_ground_truths == 0 or len(scores_list) == 0:
        return 0.0

    # Score 기준 정렬
    sort_idx = np.argsort(-np.array(scores_list))
    true_positives = np.array(true_positives)[sort_idx]

    # Precision-Recall 계산
    cum_tp = np.cumsum(true_positives)
    precision = cum_tp / (np.arange(len(true_positives)) + 1)
    recall = cum_tp / (num_ground_truths + 1e-6)

    # AP = area under PR curve (trapezoid)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    # precision을 단조 감소로 보정
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)


def evaluate_model(
    predictions: List[dict],
    ground_truths: List[dict],
    classes: List[str],
    iou_threshold: float = 0.5,
) -> dict:
    """
    전체 클래스에 대한 mAP 계산
    Returns:
        {"mAP": float, "class_ap": {class_name: ap, ...}}
    """
    class_aps = {}
    for class_idx, class_name in enumerate(classes[1:], start=1):  # background 제외
        ap = calculate_ap_per_class(predictions, ground_truths, class_idx, iou_threshold)
        class_aps[class_name] = ap

    mAP = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
    return {"mAP": mAP, "class_ap": class_aps}


def print_eval_results(results: dict):
    """평가 결과 출력"""
    print("\n" + "=" * 40)
    print(f"  mAP@0.5: {results['mAP']:.4f}")
    print("-" * 40)
    for cls, ap in results["class_ap"].items():
        print(f"  {cls:<20s}: {ap:.4f}")
    print("=" * 40 + "\n")
