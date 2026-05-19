"""
train.py - 학습 루프
Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.model import build_model, count_parameters
from src.evaluate import evaluate_model, print_eval_results
from src.utils import (
    load_config,
    set_seed,
    get_device,
    save_checkpoint,
)


# ─────────────────────────────────────────────────────────────
# Optimizer & Scheduler
# ─────────────────────────────────────────────────────────────

def build_optimizer(model, config):
    opt_cfg = config["train"]["optimizer"]
    name = opt_cfg["name"].lower()
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0005),
        )
    elif name == "adam":
        return torch.optim.Adam(params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0))
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.01))
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {name}")


def build_scheduler(optimizer, config):
    sch_cfg = config["train"]["scheduler"]
    name = sch_cfg["name"].lower()

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 5),
            gamma=sch_cfg.get("gamma", 0.1),
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["train"]["epochs"],
        )
    elif name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.1,
        )
    else:
        return None


# ─────────────────────────────────────────────────────────────
# 단일 에폭 학습
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    loss_breakdown = {}

    pbar = tqdm(data_loader, desc=f"[Train] Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()

        # 손실 항목별 누적
        for k, v in loss_dict.items():
            loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v.item()

        pbar.set_postfix(loss=f"{losses.item():.4f}")

    n = len(data_loader)
    avg_loss = total_loss / n
    avg_breakdown = {k: v / n for k, v in loss_breakdown.items()}
    return avg_loss, avg_breakdown


# ─────────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, data_loader, device, classes, iou_threshold=0.5):
    model.eval()
    all_predictions, all_ground_truths = [], []

    for images, targets in tqdm(data_loader, desc="[Val]  Evaluating"):
        images = [img.to(device) for img in images]
        predictions = model(images)
        all_predictions.extend(predictions)
        all_ground_truths.extend(targets)

    results = evaluate_model(all_predictions, all_ground_truths, classes, iou_threshold)
    return results


# ─────────────────────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config["project"]["seed"])
    device = get_device()

    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    # 데이터
    train_loader, val_loader, _, classes = build_dataloaders(config)

    # 모델
    model = build_model(config).to(device)
    count_parameters(model)

    # 옵티마이저 & 스케줄러
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    num_epochs = config["train"]["epochs"]
    iou_threshold = config["evaluate"]["iou_threshold"]
    best_mAP = 0.0

    log_path = os.path.join(config["output"]["log_dir"], "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,mAP,lr\n")

    print(f"\n{'='*50}")
    print(f"  학습 시작: {num_epochs} 에폭")
    print(f"{'='*50}\n")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        # 학습
        avg_loss, breakdown = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # 검증
        results = validate(model, val_loader, device, classes, iou_threshold)
        mAP = results["mAP"]

        # 스케줄러 업데이트
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mAP)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        # 출력
        print(f"\nEpoch [{epoch}/{num_epochs}]  "
              f"Loss: {avg_loss:.4f}  |  mAP@0.5: {mAP:.4f}  |  "
              f"LR: {current_lr:.6f}  |  Time: {elapsed:.1f}s")
        print_eval_results(results)

        # 로그 저장
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{mAP:.4f},{current_lr:.6f}\n")

        # 체크포인트 저장
        if mAP > best_mAP:
            best_mAP = mAP
            save_checkpoint(model, optimizer, epoch, mAP, config, "best_model.pth")

        # 마지막 에폭 저장
        save_checkpoint(model, optimizer, epoch, mAP, config, "last_model.pth")

    print(f"\n학습 완료! 최고 mAP@0.5: {best_mAP:.4f}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="알약 감지 모델 학습")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
