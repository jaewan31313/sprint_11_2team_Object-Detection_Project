"""
model.py - Object Detection 모델 정의
Faster R-CNN (ResNet-50-FPN v2) 기반 베이스라인
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ─────────────────────────────────────────────────────────────
# 모델 빌더
# ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    config["model"]["name"]에 따라 모델 반환
    지원 모델:
        - fasterrcnn_resnet50_fpn_v2  (기본, 추천)
        - fasterrcnn_resnet50_fpn
        - ssd300_vgg16
    """
    num_classes = len(config["data"]["classes"])  # background 포함
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    max_det = config["model"]["max_detections"]
    score_thresh = config["model"]["score_threshold"]
    nms_thresh = config["model"]["nms_threshold"]

    if model_name == "fasterrcnn_resnet50_fpn_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            box_detections_per_img=max_det,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
        )
        # 분류 헤드를 알약 클래스 수에 맞게 교체
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            box_detections_per_img=max_det,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "ssd300_vgg16":
        weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model = ssd300_vgg16(weights=weights)
        # SSD 클래스 헤드 수정
        model.head.classification_head.num_classes = num_classes

    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}\n"
                         f"선택 가능: fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, ssd300_vgg16")

    print(f"모델 로드 완료: {model_name}  (클래스 수: {num_classes})")
    return model


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수 반환"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터: {total:,}")
    return total
