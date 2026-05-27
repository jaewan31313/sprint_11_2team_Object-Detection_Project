"""
YOLO11 학습 스크립트

실행:
    python train_yolo.py
    python train_yolo.py --model yolo11m.pt --epochs 30 --batch 8
    python train_yolo.py --model yolo11l.pt --epochs 80 --batch 4 --imgsz 1280 --mixup 0.1 --copy_paste 0.1 --degrees 10 --flipud 0.5
"""
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="yolo11m.pt", help="모델 크기: yolo11n/s/m/l/x.pt")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--data",       default="data/yolo/dataset.yaml")
    parser.add_argument("--output",     default="outputs/yolo")
    parser.add_argument("--name",       default="train")
    # 증강 옵션 (기본값 = YOLO 기본값과 동일)
    parser.add_argument("--mosaic",     type=float, default=1.0)
    parser.add_argument("--mixup",      type=float, default=0.0)
    parser.add_argument("--copy_paste", type=float, default=0.0)
    parser.add_argument("--degrees",    type=float, default=0.0)
    parser.add_argument("--flipud",     type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    if not Path(args.data).exists():
        print("dataset.yaml 없음 → 먼저 실행: python scripts/convert_to_yolo.py")
        return

    model = YOLO(args.model)

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  GPU 없음 → CPU로 학습 (매우 느림). Colab에서 GPU 런타임을 선택하세요.")

    project = str(Path(args.output).resolve())

    result = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=project,
        name=args.name,
        exist_ok=True,
        device=device,
        patience=10,
        save=True,
        plots=True,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        flipud=args.flipud,
    )

    best_pt = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\n학습 완료 → {best_pt}")


if __name__ == "__main__":
    main()
