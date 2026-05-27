"""더미 모델 동작 확인용 테스트 스크립트.

train_images 폴더에서 이미지 하나를 불러와 모델에 통과시킨 뒤,
입력 이미지와 출력 이미지를 나란히 보여줍니다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from apptest_model import model

# 이미지 폴더 경로
IMAGE_DIR = Path(
    r"C:\Users\trium\Downloads\ai11-level1-project"
    r"\sprint_ai_project1_data\train_images"
)


def main():
    # 폴더에서 첫 번째 이미지 파일 하나 선택
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_DIR}")

    image_path = image_paths[0]
    print(f"불러온 이미지: {image_path.name}")

    # PIL 이미지로 불러오기
    input_image = Image.open(image_path).convert("RGB")
    print(f"입력 이미지 크기: {input_image.size}")

    # 모델에 통과시키기 (PIL -> PIL)
    output_image = model(input_image)
    print(f"출력 이미지 크기: {output_image.size}")

    # 입력 / 출력 이미지 나란히 표시
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(output_image)
    axes[1].set_title("Output (model)")
    axes[1].axis("off")
    plt.tight_layout()


    # GUI 창으로도 표시 시도 (환경에 따라 안 뜰 수 있음)
    plt.show()


if __name__ == "__main__":
    main()
