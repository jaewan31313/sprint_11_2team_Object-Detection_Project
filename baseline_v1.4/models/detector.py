import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def build_model(num_classes, cfg):
    """
    Builds a Faster R-CNN model with a pretrained ResNet-50 FPN backbone.

    num_classes: number of foreground classes (background is added internally).
    The torchvision API expects num_classes to include background,
    so we pass num_classes + 1.
    """
    pretrained = cfg["model"].get("pretrained", True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        box_detections_per_img=cfg["inference"]["max_detections"],
        box_score_thresh=cfg["inference"]["score_threshold"],
        box_nms_thresh=cfg["inference"]["nms_threshold"],
    )

    # replace the classification head to match the number of pill classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_map = checkpoint.get("best_map", 0.0)
    print(f"Loaded checkpoint from epoch {epoch} (best mAP={best_map:.4f})")
    return model, epoch, best_map


def save_checkpoint(model, optimizer, epoch, best_map, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_map": best_map,
    }, path)
