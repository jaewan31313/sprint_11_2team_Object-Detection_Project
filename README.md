# 💊 Pill Detector — 알약 인식 & 의약품 정보 제공 서비스

> 사진 한 장으로 알약을 탐지하고, 해당 의약품의 **금기사항·주의사항**을 즉시 알려주는 웹 서비스
>
> 코드잇 스프린트 AI 11기 · 2팀 · Object Detection 프로젝트

<!-- 시연 GIF: assets/demo.gif 준비되면 아래 주석 해제 -->
<!-- ![시연 영상](assets/demo.gif) -->
🎬 **시연 영상 준비 중**

---

## 프로젝트 소개

약국이 아닌 곳에서 낱알 상태의 알약을 발견하면 무슨 약인지, 함께 먹어도 되는지 알기 어렵습니다.
이 프로젝트는 **AIHub 한국 의약품 이미지 데이터셋**으로 객체 탐지 모델을 학습해,
카메라 촬영 또는 이미지 업로드만으로 알약을 식별하고 복약 주의 정보를 제공합니다.

- **3-모델 앙상블 탐지**: YOLO11 + YOLO26 + RT-DETR을 WBF(Weighted Boxes Fusion)로 결합
- **웹 서비스**: FastAPI 서버 + 반응형 웹 UI, ngrok으로 모바일 카메라 촬영 지원
- **의약품 정보**: 한국 의약품 56종의 금기사항·주의사항 DB 내장
- **HITL 피드백**: 사용자의 정답/오답 피드백을 로깅해 개선에 활용

## 결과

| 모델 | mAP@50 (val) |
|---|---|
| YOLO11-m 단독 | *(수치 기입)* |
| YOLO26-x 단독 | *(수치 기입)* |
| RT-DETR-l 단독 | *(수치 기입)* |
| **3-모델 WBF 앙상블** | ***(수치 기입)*** |

## 시스템 구성

```
[학습 — Google Colab GPU]
AIHub 데이터셋(COCO) → 전처리·포맷 변환 → YOLO11 / YOLO26 / RT-DETR 개별 학습
                                              └─→ WBF 앙상블 → 최종 예측

[서비스 — FastAPI]
카메라·이미지 업로드 → YOLO 추론 → 알약 클래스 판별 → 의약품 DB 조회 → 금기·주의사항 표시
```

### 앙상블 구성

| 모델 | 가중치 파일 | imgsz | TTA | WBF 가중치 |
|---|---|---|---|---|
| YOLO11 | yolo11m.pt | 1280 | O | 2.0 |
| YOLO26 | yolo11x.pt | 1280 | O | 2.0 |
| RT-DETR | rtdetr-l.pt | 1024 | X | 1.0 |

> 같은 YOLO 계열이라도 크기(m/x)를 달리해 다양성을 확보하고, Transformer 기반 RT-DETR로 구조적 다양성을 더했습니다.

## 기술 스택

| 분류 | 기술 |
|---|---|
| 객체 탐지 | PyTorch · ultralytics (YOLO11/26) · RT-DETR · Faster R-CNN(베이스라인) |
| 앙상블 | ensemble-boxes (WBF) |
| 서비스 | FastAPI · Streamlit(대체 UI) · ngrok |
| 이미지 처리 | OpenCV · Pillow |
| 학습 환경 | Google Colab (T4/A100) |

## 실행 방법

<details>
<summary><b>모델 학습 (Colab)</b></summary>

1. Google Drive에 프로젝트 폴더와 데이터셋 zip 업로드
2. `baseline_vomega/colab_run.ipynb` 열고 `DRIVE_PROJECT_DIR` 경로 수정 후 순서대로 실행

```bash
# 개별 학습
python train_yolo.py   --model yolo11m.pt  --epochs 60  --batch 8  --imgsz 1280
python train_yolo26.py --model yolo11x.pt  --epochs 60  --batch 4  --imgsz 1280
python train_rtdetr.py --model rtdetr-l.pt --epochs 150 --batch 16 --imgsz 1024

# 최종 앙상블
python inference_ensemble_vomega.py \
    --yolo11 outputs/yolo/train/weights/best.pt \
    --yolo26 outputs/yolo/yolo26/weights/best.pt \
    --rtdetr outputs/yolo/rtdetr/weights/best.pt \
    --weights 2.0 2.0 1.0 --wbf_iou 0.55
```

하이퍼파라미터 상세 → [`baseline_vomega/하이퍼파라미터_설명서.md`](baseline_vomega/하이퍼파라미터_설명서.md)
</details>

<details>
<summary><b>웹 애플리케이션</b></summary>

```bash
pip install -r baseline_vomega/requirements.txt

cd ver4_0529_FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# 브라우저 → http://localhost:8000/
```

모바일 카메라로 쓰려면 터미널을 하나 더 열어:

```bash
ngrok http 8000
# 출력된 https://xxxx.ngrok.io 주소를 모바일 브라우저로 접속
```
</details>

<details>
<summary><b>폴더 구조</b></summary>

```
.
├── baseline_vomega/          # 학습 파이프라인 (전처리 → 학습 → 추론 → 앙상블)
│   ├── scripts/              # 전처리·포맷 변환·시각화
│   ├── train_yolo.py         # YOLO11 학습
│   ├── train_yolo26.py       # YOLO26 학습
│   ├── train_rtdetr.py       # RT-DETR 학습
│   ├── inference_*.py        # 단독/앙상블 추론
│   └── colab_run.ipynb       # Colab 전체 실행 노트북
└── ver4_0529_FastAPI/        # 웹 애플리케이션
    ├── main.py               # FastAPI 서버 (의약품 DB 56종 내장)
    ├── app_wrapper.py        # YOLO 추론 래퍼
    └── static/index.html     # 웹 UI
```
</details>

## 팀

| 이름 | 협업 일지 |
|---|---|
| 전재완 | [Notion](https://traveling-hisser-ce1.notion.site/36422576234380e6b81cd2130ee8fd28?source=copy_link) |
| 이태훈 | [PDF](https://github.com/user-attachments/files/28632600/AI.2.1.pdf) |
| 황인홍 | [Notion](https://concise-snowboard-3e4.notion.site/364082810eef80b49dabc5c1b76f25d2?v=364082810eef8031b2c3000caea27c76&source=copy_link) |
| 김효진 | [Notion](https://insidious-flower-de8.notion.site/36912d20bf0a8063ac96d79c9c2c6e4a?source=copy_link) |

**문서** — [최종 보고서 (PDF)](https://github.com/user-attachments/files/28588713/2team.pdf) · [발표 자료 (Google Drive)](https://drive.google.com/drive/folders/1HCIzn5HzvsI8LEeaaDdMsgf5cf72I48Q?usp=sharing)

## 데이터셋

- **출처**: [AIHub — 한국 의약품 이미지](https://aihub.or.kr) (COCO 포맷)
- **분할**: Train 80% / Val 20% (seed 고정)
