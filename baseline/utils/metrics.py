from collections import defaultdict

import torch


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU between two sets of boxes (x1,y1,x2,y2)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union_area.clamp(min=1e-6)


def compute_ap(recalls, precisions):
    """Compute AP using the 11-point interpolation."""
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if len(p) > 0 else torch.tensor(0.0))
    return (ap / 11).item()


def compute_map(predictions, targets, iou_threshold=0.5):
    """
    Computes mAP@iou_threshold over a list of prediction/target pairs.

    predictions: list of dicts with 'boxes', 'labels', 'scores'
    targets:     list of dicts with 'boxes', 'labels'

    Returns: dict with 'mAP' and per-class AP values.
    """
    # gather per-class TP/FP/scores and number of GT
    class_records = defaultdict(lambda: {"scores": [], "tp": [], "n_gt": 0})

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]

        # track which GT boxes have been matched
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        # sort predictions by score descending
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

        for cls in gt_labels.unique().tolist():
            class_records[cls]["n_gt"] += (gt_labels == cls).sum().item()

        for i in range(len(pred_boxes)):
            cls = pred_labels[i].item()
            score = pred_scores[i].item()
            class_records[cls]["scores"].append(score)

            gt_mask = gt_labels == cls
            gt_cls_boxes = gt_boxes[gt_mask]
            gt_cls_indices = gt_mask.nonzero(as_tuple=True)[0]

            if len(gt_cls_boxes) == 0:
                class_records[cls]["tp"].append(0)
                continue

            ious = box_iou(pred_boxes[i:i+1], gt_cls_boxes)[0]
            best_iou, best_j = ious.max(0)

            global_idx = gt_cls_indices[best_j].item()
            if best_iou >= iou_threshold and not matched[global_idx]:
                class_records[cls]["tp"].append(1)
                matched[global_idx] = True
            else:
                class_records[cls]["tp"].append(0)

    # compute AP per class
    aps = {}
    for cls, rec in class_records.items():
        if rec["n_gt"] == 0:
            continue
        scores = torch.tensor(rec["scores"])
        tp = torch.tensor(rec["tp"], dtype=torch.float32)
        order = scores.argsort(descending=True)
        tp = tp[order]

        cumtp = tp.cumsum(0)
        cumfp = (1 - tp).cumsum(0)
        recalls = cumtp / rec["n_gt"]
        precisions = cumtp / (cumtp + cumfp).clamp(min=1e-6)

        aps[cls] = compute_ap(recalls, precisions)

    mean_ap = sum(aps.values()) / max(len(aps), 1)
    return {"mAP": mean_ap, "per_class_AP": aps}
