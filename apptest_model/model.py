"""애플리케이션 동작 확인용 더미 AI 모델.

실제 학습 없이, PIL 이미지를 입력받아 PIL 이미지를 출력하는
fully-convolutional 신경망입니다. fc 층이 없으므로 임의의 크기의
이미지를 그대로 처리할 수 있습니다.
"""

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


class TestModel(nn.Module):
    """conv 레이어만으로 구성된 간단한 fully-convolutional 네트워크.

    입력 채널 수(3, RGB)와 출력 채널 수(3, RGB)가 동일하며,
    공간 해상도를 유지하므로 입력과 출력 이미지 크기가 같습니다.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 출력값을 [0, 1] 범위로 제한
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CallableModel:
    """PIL 이미지를 받아 PIL 이미지를 반환하는 호출 가능한 래퍼.

    사용 예::

        from apptest_model import model
        result_image = model(image)
    """

    def __init__(self):
        self.net = TestModel()
        self.net.eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        # PIL -> Tensor (RGB로 변환하여 채널 수 고정)
        tensor = to_tensor(image.convert("RGB")).unsqueeze(0)  # (1, 3, H, W)
        output = self.net(tensor)  # (1, 3, H, W)
        # Tensor -> PIL
        return to_pil_image(output.squeeze(0).clamp(0, 1))


# 패키지에서 바로 import 하여 호출할 수 있는 인스턴스
model = _CallableModel()
