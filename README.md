# SOLIDER 멀티 카메라 Person Re-ID 테스트

오프라인 CCTV 영상 2개 이상에 대해 **YOLOv8 사람 검출 → 카메라별 IoU 추적 → SOLIDER-REID 임베딩 추출 → 카메라 간 Global ID 매칭 → MP4/CSV 저장**을 수행하는 최소 테스트 프로젝트입니다.

> 본 코드는 운영용 MTMCT 시스템이 아닌 **검증용 baseline**입니다. SOLIDER 공식 Re-ID 모델의 fine-tuned checkpoint가 있어야 합니다. SOLIDER 자가지도 사전학습 체크포인트는 Re-ID fine-tuned 가중치와 다릅니다. 또한 공식 발표 논문의 mAP/Rank-1 수치와 이 영상 매칭 결과는 직접 비교할 수 없습니다.

## 구성

```text
solider_multicam_test/
├── main.py                  # 영상 입력, 검출, 시각화, 저장
├── reid.py                  # 공식 SOLIDER-REID 모델 어댑터
├── tracking.py              # 카메라별 local tracker / global matcher
├── config_swin_small.yaml   # Swin Small 설정 예시
├── requirements.txt
└── tests/test_tracking.py
```

## 1. 환경 및 설치

권장: Linux, NVIDIA GPU, Python 3.10 (공식 SOLIDER-REID의 원래 환경은 Python 3.7 / PyTorch 1.7.1 기반이므로 최신 PyTorch에서는 공식 코드 호환성 패치가 필요할 수 있음).

```bash
python3 -m venv .venv
source .venv/bin/activate
# 먼저 자신의 GPU/CUDA에 맞는 PyTorch, torchvision 설치: https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements.txt
# 현재 프로젝트 폴더의 상위에 공식 저장소 설치
git clone https://github.com/tinyvision/SOLIDER-REID.git ../SOLIDER-REID
```

SOLIDER 저장소의 내부 의존성에 따라 추가 패키지 또는 구버전 호환 수정이 필요할 수 있습니다. `timm==0.4.12`는 공식 코드가 작성된 시기의 API를 기준으로 한 선택입니다.

## 2. 체크포인트 준비

공식 [SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID) README의 **Performance** 섹션에서 Market-1501 또는 MSMT17용 **Re-ID fine-tuned** checkpoint를 내려받아 `weights/solider_reid.pth`에 저장합니다. 정확한 architecture(Swin Tiny / Small / Base)와 fine-tuning 설정에 맞는 YAML을 지정해야 합니다. 본 프로젝트에 가중치는 포함되지 않습니다.

`config_swin_small.yaml`은 **Swin Small / 384×128** 예시일 뿐입니다. checkpoint가 Tiny/Base 또는 다른 입력 크기로 학습되었다면 `MODEL.TRANSFORMER_TYPE`, `INPUT.SIZE_TRAIN`, `INPUT.SIZE_TEST`, `MODEL.SEMANTIC_WEIGHT`를 학습 조건과 일치시키세요. 특히 **실제 학습 config를 사용하는 것이 가장 안전**합니다. 설정이 불일치하면 로더가 예외를 발생시킵니다.

Checkpoint의 `base.*` 및 `bottleneck.*` 텐서가 모델과 95% 이상 일치해야 사용합니다. 분류 헤드(classifier)의 불일치는 허용합니다. checkpoint 구조가 `model`, `net` 등의 래퍼를 사용한다면 코드에서 변환이 필요합니다.

## 3. 영상 입력

예를 들어 `data/cam1.mp4`와 `data/cam2.mp4`를 준비합니다. **두 영상의 시작 시각(t=0)이 동기화되어 있다고 가정**합니다. 영상마다 시간차가 있는 경우 정확한 재식별 판정을 할 수 없으며 시간 보정 코드를 추가해야 합니다.

```bash
python main.py \
  --videos data/cam1.mp4 data/cam2.mp4 \
  --repo ../SOLIDER-REID \
  --config config_swin_small.yaml \
  --weights weights/solider_reid.pth \
  --device cuda \
  --fps-sample 2 \
  --max-seconds 60 \
  --similarity 0.75 \
  --max-gap 120 \
  --out outputs
```

CPU 테스트: `--device cpu`. 카메라 추가: `--videos cam1.mp4 cam2.mp4 cam3.mp4`. `--detector yolov8n.pt`로 기본 YOLOv8n을 사용하며 최초 실행 때 Ultralytics가 가중치를 내려받을 수 있습니다.

## 4. 출력

- `outputs/cam1.mp4`, `outputs/cam2.mp4`, ...: 사람 bbox, 로컬 T ID, 전역 G ID 표시
- `outputs/matches.csv`: 카메라, 입력 소스, 영상 시각(초), 로컬 ID, Global ID, 신규 카메라 간 연결 여부, 코사인 유사도, bbox 좌표

CSV 예시(임의):

```csv
camera,source,timestamp_s,local_track_id,global_id,matched_other_camera,similarity,x1,y1,x2,y2
cam1,data/cam1.mp4,0.000,1,1,0,0.0000,100,50,200,340
cam2,data/cam2.mp4,2.000,3,1,1,0.8900,45,70,170,355
```

한 카메라의 T ID와 다른 카메라의 T ID가 같아도 동일 인물이라는 의미가 없습니다. Global ID가 같으면 테스트 매처가 동일 인물로 **추정**한 것입니다.

## 5. 파이프라인 / 매칭 알고리즘

1. 각 카메라에서 초당 2프레임 표본을 가져옵니다 (`--fps-sample`).
2. YOLOv8에서 class=0(person)만 검출합니다.
3. 카메라별 IoU 기반 greedy tracker가 로컬 트랙 번호를 유지합니다.
4. SOLIDER가 인물 crop을 384×128로 리사이즈·ImageNet 정규화한 후 특징 벡터를 생성합니다.
5. L2 정규화한 feature 사이의 코사인 유사도 `s = f_a · f_b`를 계산합니다.
6. 신규 로컬 트랙만 기존 **다른 카메라**의 gallery identity와 비교합니다. `s >= --similarity`이고 마지막 관측 시간이 `--max-gap`초 이내이면 기존 전역 ID를 할당합니다. 아니면 새 ID를 발급합니다.
7. 이미 연결된 로컬 트랙은 기존 전역 ID를 사용하며 EMA로 gallery feature를 갱신합니다.

## 6. 테스트 및 한계

```bash
python -m unittest discover -s tests -v
python -m py_compile main.py reid.py tracking.py
```

이 코드는 **동작 구조 검증용**입니다. 모델 가중치 및 실제 영상이 포함되지 않으므로 end-to-end 정확도 검증은 별도로 수행해야 합니다.

- IoU tracker는 장시간 가림, 동일 카메라 재등장, 빠른 이동에 약합니다. 실제 적용에는 ByteTrack / BoT-SORT 등의 교체를 권장합니다.
- 코사인 임계값 **0.75는 예시값**일 뿐입니다. 카메라/현장별 GT에 기반한 검증으로 보정해야 합니다.
- 단순 gallery matching은 시간·공간 토폴로지, 진입/퇴장 zone, 동일 시간대 인물 존재, 전역 one-to-one assignment를 고려하지 않습니다. 따라서 잘못된 Global ID 병합이 발생할 수 있습니다.
- 이미 매칭된 로컬 트랙은 중간에 다른 전역 ID로 재분류하지 않습니다. 한 번의 오매칭이 전파될 수 있습니다.
- 각 영상의 fps와 t=0에 근거하여 순차적으로 처리합니다. 원래 영상의 프레임을 개별 탐색하므로 고효율 실시간 구현은 아닙니다. RTSP 입력이나 재연결, 이벤트 타임스탬프는 지원하지 않습니다.
- gallery는 오래된 ID를 메모리에서 제거하지 않습니다. 긴 영상이나 많은 인물에는 TTL 정리/DB가 필요합니다.
- 두 영상이 시각적으로 동일 환경이어도 카메라별 인물 크기/광학 특성에 따라 성능이 달라집니다. GT를 만들고 IDF1, IDP, IDR, HOTA 등을 별도로 측정하세요.
- 실제 CCTV 이용 시 개인정보 및 영상 처리 관련 규정·권한을 준수해야 합니다.

## 출처

- [SOLIDER 공식 프로젝트](https://github.com/tinyvision/SOLIDER)
- [SOLIDER-REID 공식 PyTorch 저장소](https://github.com/tinyvision/SOLIDER-REID)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
