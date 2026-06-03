# ST-GCN 추론 모듈.
# MMAction2 기반 ST-GCN 모델을 로드하고 2D keypoints로 행동을 분류한다.

import numpy as np
import torch
from mmaction.apis import init_recognizer, inference_recognizer

# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────
CONFIG_PATH     = "/home/ubuntu/mmaction2/configs/skeleton/stgcn/stgcn_config.py"
CHECKPOINT_PATH = "/home/ubuntu/mmaction2/checkpoints/stgcn_ntu60_xsub.pth"
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"

FALL_LABEL_IDX  = 43
FALL_THRESHOLD  = 0.5

# ───────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────
print("[ST-GCN] 모델 로드 중...")
model = init_recognizer(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
print(f"[ST-GCN] 모델 로드 완료 (디바이스: {DEVICE})")

# ───────────────────────────────────────────
# 추론
# ───────────────────────────────────────────
def run_inference(frames: list) -> dict:
    try:
        T = len(frames)
        J = len(frames[0].keypoints)

        # [T, J, 2] 배열 생성 (x, y만 사용)
        kp_array = np.zeros((T, J, 2), dtype=np.float32)
        score_array = np.ones((T, J), dtype=np.float32)

        for i, frame in enumerate(frames):
            for j, kp in enumerate(frame.keypoints):
                kp_array[i, j, 0] = kp[0]  # x
                kp_array[i, j, 1] = kp[1]  # y
                if len(kp) == 3:
                    score_array[i, j] = kp[2]  # confidence

        # ST-GCN 입력: keypoint [1, T, J, 2] → [1, 1, T, J, 2]
        # MMAction2 형식: (M, T, V, C) → (1, T, V, C) M=사람수
        keypoint = kp_array[np.newaxis]        # [1, T, J, 2]
        keypoint_score = score_array[np.newaxis]  # [1, T, J]

        fake_anno = dict(
            keypoint=keypoint,
            keypoint_score=keypoint_score,
            total_frames=T,
            frame_inds=np.arange(T),
            img_shape=(480, 640),
            original_shape=(480, 640),
            label=-1
        )

        result = inference_recognizer(model, fake_anno)
        scores = result.pred_score.cpu().numpy()

        top_idx         = int(np.argmax(scores))
        top_confidence  = float(scores[top_idx])
        fall_confidence = float(scores[FALL_LABEL_IDX]) if FALL_LABEL_IDX < len(scores) else 0.0
        is_fall         = fall_confidence >= FALL_THRESHOLD

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
