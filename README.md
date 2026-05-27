# Pill Detection - Healthcare Object Detection

사진 속 최대 4개의 알약을 감지하는 Object Detection 프로젝트

---

## 디렉토리 구조

```
baseline_v1.4/
├── configs/default.yaml      ← 모든 설정값 (하이퍼파라미터 등)
├── data/
│   ├── dataset.py            ← 데이터 로딩/파싱
│   └── transforms.py         ← 데이터 증강
├── models/
│   └── detector.py           ← 모델 정의/저장/로드
├── utils/
│   ├── metrics.py            ← mAP 계산
│   └── coco_utils.py         ← 결과 저장 (JSON/CSV)
├── scripts/
│   ├── preprocess.py         ← 전처리
│   └── visualize.py          ← 시각화
├── train.py                  ← 학습 진입점
├── inference.py              ← 추론(예측) 진입점
└── colab_run.ipynb           ← Colab 실행용 노트북
```

---

## 전체 파이프라인

```
[데이터 준비] → [모델 생성] → [학습 루프] → [평가] → [저장]
                                                         ↓
                                              [inference.py로 예측]
                                                         ↓
                                              [submission.csv 제출]
```

---

## 실행 방법

```bash
# 1. 학습
python train.py

# 2. 특정 체크포인트에서 이어서 학습
python train.py --resume outputs/checkpoints/last.pth

# 3. 예측 (best 모델 사용)
python inference.py

# 4. 제출
# outputs/predictions/submission.csv 파일을 Kaggle에 업로드
```

---

## 모델

**Faster R-CNN** (2-stage detector)

```
이미지 입력
    ↓
ResNet-50 + FPN (특징 추출, ImageNet pretrained)
    ↓
RPN (Region Proposal Network) — 물체가 있을 것 같은 영역 제안
    ↓
RoI Head — 각 영역이 어떤 클래스인지 분류 + 박스 정밀 조정
    ↓
(클래스, 박스, 점수) 출력
```

- Backbone: ResNet-50 + FPN (pretrained)
- 마지막 분류 헤드만 알약 클래스 수에 맞게 교체
- 최대 감지 수: 이미지당 4개
- 평가 지표: mAP@0.5

---

## 설정 (configs/default.yaml)

| 섹션 | 주요 설정 | 현재값 |
|------|-----------|--------|
| `data` | 데이터 경로, val 비율 | val 20% |
| `model` | 모델명, backbone | Faster R-CNN + ResNet50 |
| `train` | epoch, batch, lr | 10 epoch, batch=4, lr=0.005 |
| `augmentation` | flip, brightness | 현재 모두 비활성화 |
| `inference` | score 임계값, NMS, 최대 탐지 수 | 0.5, 0.5, 4개 |

`enabled: false` → `enabled: true`로 바꾸면 해당 기능이 켜집니다.

---

## 성능 개선 포인트

`default.yaml`에서 아래를 `true`로 바꾸면 성능이 올라갈 수 있습니다.

| 설정 | 효과 |
|------|------|
| `augmentation.train.horizontal_flip.enabled` | 데이터 다양성 증가 |
| `augmentation.train.random_brightness.enabled` | 조명 변화에 강해짐 |
| `train.lr_scheduler.enabled` | 학습 후반 lr 감소로 정밀도 향상 |
| `train.clip_grad_norm.enabled` | 학습 안정성 향상 |

---

## 데이터 형식

- 어노테이션: COCO JSON 형식 (`train_annotations/XXX_json/약품명/image.json`)
- 박스 형식: COCO `[x, y, w, h]` → 내부적으로 `[x1, y1, x2, y2]`로 변환
- 클래스 라벨: 0번은 background 예약, 실제 알약 클래스는 1부터 시작

### 출력 파일

```
outputs/
├── checkpoints/
│   ├── best.pth           ← 가장 높은 mAP 체크포인트
│   ├── last.pth           ← 마지막 epoch 체크포인트
│   └── class_mapping.json ← 클래스 이름-숫자 매핑
└── predictions/
    ├── predictions.json   ← 상세 예측 결과
    └── submission.csv     ← Kaggle 제출용
```

---

## 의존성

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
PyYAML>=6.0
matplotlib>=3.5.0
```
