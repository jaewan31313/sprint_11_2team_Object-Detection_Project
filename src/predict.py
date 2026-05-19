"""
predict.py - 추론 및 시각화
Usage:
    # 단일 이미지
    python src/predict.py --config configs/config.yaml --image data/raw/images/sample.jpg

    # 전체 테스트셋 배치 추론
    python src/predict.py --config configs/config.yaml --batch
"""

import argparse
import os

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2

from src.dataset import build_dataloaders, get_val_transform
from src.model import build_model
from src.utils import load_config, get_device, load_checkpoint, visualize_prediction


@torch.no_grad()
def predict_single(model, image_path: str, config: dict, device, classes: list, save: bool = False):
    """단일 이미지 추론 및 시각화"""
    transform = get_val_transform(config["train"]["image_size"])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)

    model.eval()
    predictions = model([image_tensor])
    pred = predictions[0]

    save_path = None
    if save:
        fname = os.path.basename(image_path).replace(".jpg", "_pred.jpg")
        save_path = os.path.join(config["output"]["prediction_dir"], fname)

    visualize_prediction(
        image_tensor.cpu(),
        pred,
        classes,
        score_threshold=config["model"]["score_threshold"],
        max_detections=config["model"]["max_detections"],
        save_path=save_path,
        title=os.path.basename(image_path),
    )

    if save_path:
        print(f"저장 완료: {save_path}")


@torch.no_grad()
def predict_batch(model, test_loader, config: dict, device, classes: list):
    """테스트셋 배치 추론"""
    model.eval()
    pred_dir = config["output"]["prediction_dir"]
    os.makedirs(pred_dir, exist_ok=True)

    for images, image_names in tqdm(test_loader, desc="Batch Inference"):
        images = [img.to(device) for img in images]
        predictions = model(images)

        for img, pred, name in zip(images, predictions, image_names):
            save_path = os.path.join(pred_dir, f"{name}_pred.jpg")
            visualize_prediction(
                img.cpu(), pred, classes,
                score_threshold=config["model"]["score_threshold"],
                max_detections=config["model"]["max_detections"],
                save_path=save_path,
                title=name,
            )

    print(f"\n추론 완료. 결과 저장: {pred_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="알약 감지 추론")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="체크포인트 경로 (기본: outputs/checkpoints/best_model.pth)")
    parser.add_argument("--image", type=str, default=None, help="단일 이미지 경로")
    parser.add_argument("--batch", action="store_true", help="테스트셋 배치 추론")
    parser.add_argument("--save", action="store_true", help="결과 이미지 저장")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    classes = config["data"]["classes"]

    # 모델 로드
    model = build_model(config).to(device)
    ckpt_path = args.checkpoint or os.path.join(
        config["output"]["checkpoint_dir"], "best_model.pth"
    )
    load_checkpoint(model, None, ckpt_path, device)

    if args.image:
        predict_single(model, args.image, config, device, classes, save=args.save)
    elif args.batch:
        _, _, test_loader, _ = build_dataloaders(config)
        predict_batch(model, test_loader, config, device, classes)
    else:
        print("--image 또는 --batch 옵션을 지정하세요.")
