# MMAction2 추론 모듈.
# MMAction2 기반 스켈레톤 행동 인식 모델을 로드하고 2D keypoints로 행동을 분류한다.

import numpy as np
import torch
from mmaction.apis import init_recognizer, inference_recognizer

# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────
CONFIG_PATH = "/home/ubuntu/mmaction2/configs/skeleton/stgcn/stgcn_config.py"
CHECKPOINT_PATH = "/home/ubuntu/mmaction2/checkpoints/stgcn_ntu60_xsub.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

FALL_LABEL_IDX = 43
FALL_THRESHOLD = 0.5

# ───────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────
print("[MMAction2] 모델 로드 중...")
model = init_recognizer(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
print(f"[MMAction2] 모델 로드 완료 (디바이스: {DEVICE})")


# ───────────────────────────────────────────
# 추론
# ───────────────────────────────────────────
def run_inference(frames: list) -> dict:
    try:
        t = len(frames)
        j = len(frames[0].keypoints)

        # [T, J, 2] 배열 생성 (x, y만 사용)
        kp_array = np.zeros((t, j, 2), dtype=np.float32)
        score_array = np.ones((t, j), dtype=np.float32)

        for i, frame in enumerate(frames):
            for k, kp in enumerate(frame.keypoints):
                kp_array[i, k, 0] = kp[0]
                kp_array[i, k, 1] = kp[1]
                if len(kp) == 3:
                    score_array[i, k] = kp[2]

        # MMAction2 스켈레톤 입력 형식
        keypoint = kp_array[np.newaxis]
        keypoint_score = score_array[np.newaxis]

        fake_anno = dict(
            keypoint=keypoint,
            keypoint_score=keypoint_score,
            total_frames=t,
            frame_inds=np.arange(t),
            img_shape=(480, 640),
            original_shape=(480, 640),
            label=-1,
        )

        result = inference_recognizer(model, fake_anno)
        scores = result.pred_score.cpu().numpy()

        top_idx = int(np.argmax(scores))
        top_confidence = float(scores[top_idx])
        fall_confidence = float(scores[FALL_LABEL_IDX]) if FALL_LABEL_IDX < len(scores) else 0.0
        is_fall = fall_confidence >= FALL_THRESHOLD

        print(f"[MMAction2] 추론 완료 - 낙상 여부: {is_fall} (confidence: {fall_confidence:.3f})")

        return {
            "is_fall": is_fall,
            "fall_confidence": fall_confidence,
            "top_label": str(top_idx),
            "top_confidence": top_confidence,
        }

    except Exception as exc:
        print(f"[MMAction2] 추론 오류: {exc}")
        return {
            "is_fall": False,
            "fall_confidence": 0.0,
            "top_label": "error",
            "top_confidence": 0.0,
        }
