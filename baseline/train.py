"""
Training entry point.

Usage:
    python train.py
    python train.py --config configs/default.yaml --resume outputs/checkpoints/last.pth
"""
import argparse
import os
from pathlib import Path

import torch
import yaml

from data import build_dataloaders, get_transforms
from models.detector import build_model, load_checkpoint, save_checkpoint
from utils.coco_utils import save_class_mapping
from utils.metrics import compute_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    predictions, targets_all = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                predictions.append({
                    "boxes": out["boxes"].cpu(),
                    "labels": out["labels"].cpu(),
                    "scores": out["scores"].cpu(),
                })
                targets_all.append({
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                })

    return compute_map(predictions, targets_all)


def train_one_epoch(model, optimizer, loader, device, epoch, log_interval,
                    use_clip=False, max_norm=1.0):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                   for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        if use_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += losses.item()
        if (i + 1) % log_interval == 0:
            print(f"  [Epoch {epoch}] step {i+1}/{len(loader)} | loss={losses.item():.4f}")

    return total_loss / len(loader)


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transforms_train = get_transforms(cfg, split="train")
    transforms_val = get_transforms(cfg, split="val")

    train_loader, val_loader, class_to_idx, idx_to_class, idx_to_orig_id = build_dataloaders(
        cfg, transforms_train, transforms_val
    )

    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

    # save class mapping for inference
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_class_mapping(class_to_idx, idx_to_class, idx_to_orig_id, ckpt_dir / "class_mapping.json")

    model = build_model(num_classes, cfg)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["train"]["learning_rate"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    sched_cfg = cfg["train"]["lr_scheduler"]
    scheduler = None
    if sched_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg["step_size"],
            gamma=sched_cfg["gamma"],
        )
        print("lr_scheduler: 활성화")
    else:
        print("lr_scheduler: 비활성화")

    clip_cfg = cfg["train"]["clip_grad_norm"]
    use_clip = clip_cfg.get("enabled", False)
    print(f"clip_grad_norm: {'활성화' if use_clip else '비활성화'}")

    start_epoch = 0
    best_map = 0.0

    if args.resume and Path(args.resume).exists():
        model, start_epoch, best_map = load_checkpoint(model, args.resume, device)

    log_interval = cfg["output"]["log_interval"]
    epochs = cfg["train"]["epochs"]

    for epoch in range(start_epoch + 1, epochs + 1):
        avg_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, log_interval,
            use_clip=use_clip, max_norm=clip_cfg.get("max_norm", 1.0),
        )
        if scheduler:
            scheduler.step()

        metrics = evaluate(model, val_loader, device)
        map_val = metrics["mAP"]
        print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | mAP={map_val:.4f}")

        # always save last checkpoint
        save_checkpoint(model, optimizer, epoch, map_val, ckpt_dir / "last.pth")

        if map_val > best_map:
            best_map = map_val
            save_checkpoint(model, optimizer, epoch, best_map, ckpt_dir / "best.pth")
            print(f"  -> New best mAP: {best_map:.4f}")

    print(f"\nTraining complete. Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
