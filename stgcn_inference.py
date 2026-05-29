# ST-GCN 추론 모듈.
# MMAction2 기반 ST-GCN 모델을 로드하고 2D keypoints로 행동을 분류한다.

import numpy as np
import torch
from mmaction.apis import init_recognizer, inference_recognizer

# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────
CONFIG_PATH     = "/home/ubuntu/mmaction2/checkpoints/stgcn_config.py"
CHECKPOINT_PATH = "/home/ubuntu/mmaction2/checkpoints/stgcn_ntu60_xsub.pth"
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"

# NTU RGB+D 60 행동 레이블 중 낙상 관련 항목
# NTU60 기준: 43번 = fall down
FALL_LABEL_IDX  = 43
FALL_THRESHOLD  = 0.5  # 낙상으로 판단하는 confidence 임계값

# ───────────────────────────────────────────
# 모델 로드 (서버 시작 시 1회만 실행)
# ───────────────────────────────────────────
print("[ST-GCN] 모델 로드 중...")
model = init_recognizer(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
print(f"[ST-GCN] 모델 로드 완료 (디바이스: {DEVICE})")

# ───────────────────────────────────────────
# keypoints 전처리
# ───────────────────────────────────────────
def preprocess_keypoints(frames: list) -> np.ndarray:
    """
    FastAPI에서 받은 frames 데이터를 ST-GCN 입력 형식으로 변환합니다.

    입력: frames = [{"frame_id": 1, "keypoints": [[x,y,conf], ...]}, ...]
    출력: np.ndarray shape [1, 2, T, 17, 1]
          (배치, 채널(x/y), 프레임, 관절, 사람 수)
    """
    T = len(frames)       # 프레임 수 (243)
    J = len(frames[0].keypoints)  # 관절 수 (17)

    # [T, J, 3] 배열 생성
    kp_array = np.zeros((T, J, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        for j, kp in enumerate(frame.keypoints):
            kp_array[i, j, 0] = kp[0]  # x
            kp_array[i, j, 1] = kp[1]  # y
            kp_array[i, j, 2] = kp[2] if len(kp) == 3 else 1.0  # confidence

    # x, y만 추출 → [T, J, 2]
    xy = kp_array[:, :, :2]

    # ST-GCN 입력 형식: [1, 2, T, J, 1] (배치, 채널, 프레임, 관절, 사람)
    # [T, J, 2] → [2, T, J] → [1, 2, T, J, 1]
    xy_transposed = xy.transpose(2, 0, 1)           # [2, T, J]
    xy_expanded = xy_transposed[np.newaxis, :, :, :, np.newaxis]  # [1, 2, T, J, 1]

    return xy_expanded

# ───────────────────────────────────────────
# 추론
# ───────────────────────────────────────────
def run_inference(frames: list) -> dict:
    """
    2D keypoints를 받아 ST-GCN으로 행동을 분류합니다.

    반환값:
    {
        "is_fall": bool,         낙상 여부
        "fall_confidence": float, 낙상 confidence
        "top_label": str,        가장 높은 확률의 행동
        "top_confidence": float  가장 높은 confidence
    }
    """
    try:
        kp_input = preprocess_keypoints(frames)

        # ST-GCN 추론
        fake_anno = dict(
            keypoint=kp_input,
            keypoint_score=np.ones((1, 243, 17, 1), dtype=np.float32),
            total_frames=243,
            frame_inds=np.arange(243),
            img_shape=(480, 640),
            original_shape=(480, 640),
            label=-1
        )

        result = inference_recognizer(model, fake_anno)
        scores = result.pred_score.cpu().numpy()

        top_idx        = int(np.argmax(scores))
        top_confidence = float(scores[top_idx])
        fall_confidence = float(scores[FALL_LABEL_IDX]) if FALL_LABEL_IDX < len(scores) else 0.0
        is_fall        = fall_confidence >= FALL_THRESHOLD

        print(f"[ST-GCN] 추론 완료 - 낙상: {is_fall} (confidence: {fall_confidence:.3f})")

        return {
            "is_fall"         : is_fall,
            "fall_confidence" : fall_confidence,
            "top_label"       : str(top_idx),
            "top_confidence"  : top_confidence,
        }

    except Exception as e:
        print(f"[ST-GCN] 추론 오류: {e}")
        return {
            "is_fall"         : False,
            "fall_confidence" : 0.0,
            "top_label"       : "error",
            "top_confidence"  : 0.0,
        }
