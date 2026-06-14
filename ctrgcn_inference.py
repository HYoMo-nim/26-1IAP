# CTR-GCN 추론 모듈 (최적화 통합)
# 3D skeleton keypoints로 낙상 판별 및 오탐 감소

import os
import numpy as np
from ctrgcn_model.ctrgcn import Model
from typing import Optional, Dict

# 최적화 모듈 import
try:
    from fall_detection_optimizer import optimize_fall_detection, FALL_LABEL_IDX
    OPTIMIZER_AVAILABLE = True
except ImportError:
    print("[경고] fall_detection_optimizer 미설치")
    OPTIMIZER_AVAILABLE = False
    FALL_LABEL_IDX = 42

# PyTorch 조건부 로드
TORCH_AVAILABLE = False
MODEL = None
DEVICE = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[CTR-GCN] PyTorch 사용 가능 (디바이스: {DEVICE})")
except ImportError:
    print("[경고] PyTorch 미설치: pip install -r requirements.txt")
    TORCH_AVAILABLE = False

# 모델 경로
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models", "weight", "ctr-gcn",
    "runs-60-37560.pt"
)

def _init_model(model_path: str = MODEL_PATH):
    """모델 초기화"""
    global MODEL, TORCH_AVAILABLE

    if MODEL is not None:
        return MODEL

    if not TORCH_AVAILABLE:
        return None

    try:
        import torch

        if not os.path.exists(model_path):
            print(f"[오류] 모델 파일 없음: {model_path}")
            return None

        print(f"[CTR-GCN] 모델 로드: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)

        MODEL = Model(
            num_class=60,
            num_point=25,
            num_person=2,
            graph='graph.ntu_rgb_d.Graph',
            graph_args={'labeling_mode': 'spatial'},
            in_channels=3
        )
        MODEL.load_state_dict(checkpoint)

        # BN(Batch Norm) 에러 방지용 추론 모드 전환
        MODEL.eval()
        MODEL.to(DEVICE)

        print(f"[CTR-GCN] 모델 로드 완료")
        return MODEL

    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return None


def run_inference(frames_3d: np.ndarray) -> Dict:
    """
    3D skeleton keypoints로 낙상 판별 (최적화 통합)
    """
    if not TORCH_AVAILABLE:
        print("[오류] PyTorch 필수: pip install -r requirements.txt")
        return _get_error_response()

    try:
        # 1. 입력 검증 및 변환
        if isinstance(frames_3d, list):
            frames_3d = np.array(frames_3d, dtype=np.float32)
        else:
            frames_3d = np.asarray(frames_3d, dtype=np.float32)

        if len(frames_3d.shape) != 3 or frames_3d.shape[-1] != 3:
            print(f"[오류] 입력 형태 오류: {frames_3d.shape}")
            return _get_error_response()

        T, J = frames_3d.shape[0], frames_3d.shape[1]

        # 2. 모델 초기화
        model = _init_model()
        if model is None:
            print("[오류] 모델 초기화 실패")
            return _get_error_response()

        import torch

        # 3. 텐서 차원 변경 (T, V, C -> C, T, V) 및 축 추가
        tensor_3d = torch.tensor(frames_3d, dtype=torch.float32)
        tensor_3d = tensor_3d.permute(2, 0, 1)
        input_tensor = tensor_3d.unsqueeze(0).unsqueeze(-1).to(DEVICE)

        N, C, T_size, V, M = input_tensor.size()

        # 4. 좌표 정규화 (Root Centering & Scaling)
        # [핵심 수정 1] 매 프레임이 아닌, '첫 번째 프레임(T=0)'의 골반 좌표만 추출합니다.
        # 이렇게 해야 사람이 바닥으로 떨어지는 전체 궤적(중력)이 보존됩니다!
        root_joint_first_frame = input_tensor[:, :, 0:1, 0:1, :]
        input_tensor = input_tensor - root_joint_first_frame
        
        # [핵심 수정 2] 카메라 렌즈(Y축 하강)와 인공지능(Y축 상승)의 방향을 맞춰줍니다. (상하 반전 방지)
        # C차원(X, Y, Z)에서 인덱스 1이 Y축입니다.
        input_tensor[:, 1, :, :, :] = -input_tensor[:, 1, :, :, :]

        # 뼈대 전체 스케일 압축
        max_val = torch.max(torch.abs(input_tensor))
        if max_val > 0:
            input_tensor = input_tensor / max_val

        # 5. 17관절 -> 25관절 맵핑
        if V == 17:
            ntu_tensor = torch.zeros(N, C, T_size, 25, M).to(DEVICE)

            # 척추, 머리
            ntu_tensor[:, :, :, 0, :] = input_tensor[:, :, :, 0, :]
            ntu_tensor[:, :, :, 1, :] = input_tensor[:, :, :, 7, :]
            ntu_tensor[:, :, :, 20, :] = input_tensor[:, :, :, 8, :]
            ntu_tensor[:, :, :, 2, :] = input_tensor[:, :, :, 8, :]
            ntu_tensor[:, :, :, 3, :] = input_tensor[:, :, :, 9, :]

            # 왼팔
            ntu_tensor[:, :, :, 4, :] = input_tensor[:, :, :, 11, :]
            ntu_tensor[:, :, :, 5, :] = input_tensor[:, :, :, 12, :]
            ntu_tensor[:, :, :, 6, :] = input_tensor[:, :, :, 13, :]
            ntu_tensor[:, :, :, 7, :] = input_tensor[:, :, :, 13, :]
            ntu_tensor[:, :, :, 21, :] = input_tensor[:, :, :, 13, :]
            ntu_tensor[:, :, :, 22, :] = input_tensor[:, :, :, 13, :]

            # 오른팔
            ntu_tensor[:, :, :, 8, :] = input_tensor[:, :, :, 14, :]
            ntu_tensor[:, :, :, 9, :] = input_tensor[:, :, :, 15, :]
            ntu_tensor[:, :, :, 10, :] = input_tensor[:, :, :, 16, :]
            ntu_tensor[:, :, :, 11, :] = input_tensor[:, :, :, 16, :]
            ntu_tensor[:, :, :, 23, :] = input_tensor[:, :, :, 16, :]
            ntu_tensor[:, :, :, 24, :] = input_tensor[:, :, :, 16, :]

            # 왼다리
            ntu_tensor[:, :, :, 12, :] = input_tensor[:, :, :, 4, :]
            ntu_tensor[:, :, :, 13, :] = input_tensor[:, :, :, 5, :]
            ntu_tensor[:, :, :, 14, :] = input_tensor[:, :, :, 6, :]
            ntu_tensor[:, :, :, 15, :] = input_tensor[:, :, :, 6, :]

            # 오른다리
            ntu_tensor[:, :, :, 16, :] = input_tensor[:, :, :, 1, :]
            ntu_tensor[:, :, :, 17, :] = input_tensor[:, :, :, 2, :]
            ntu_tensor[:, :, :, 18, :] = input_tensor[:, :, :, 3, :]
            ntu_tensor[:, :, :, 19, :] = input_tensor[:, :, :, 3, :]

            input_tensor = ntu_tensor

        # 6. 사람 수 차원 패딩 (CTR-GCN은 기본적으로 M=2를 요구함)
        N, C, T_size, V, M = input_tensor.shape
        if M == 1:
            zeros_m = torch.zeros(N, C, T_size, V, 1).to(DEVICE)
            input_tensor = torch.cat([input_tensor, zeros_m], dim=4)

        # 7. 대망의 모델 추론
        with torch.no_grad():
            output = model(input_tensor)

        # 8. 가장 높은 확률 추출 로직
        if output.dim() == 2 and output.shape[0] > 1:
            probs = F.softmax(output, dim=1)
            max_probs, _ = torch.max(probs, dim=0)
            scores = max_probs.cpu().detach().numpy()
        else:
            logits = output.squeeze()
            probs = F.softmax(logits, dim=0)
            scores = probs.cpu().detach().numpy()

        # 호환성을 위한 배열 복제
        scores_sequence = np.tile(scores[np.newaxis, :], (T, 1))

        # 9. 결과 도출
        top_idx = int(np.argmax(scores))
        top_confidence = float(scores[top_idx])
        fall_confidence = float(scores[FALL_LABEL_IDX]) if FALL_LABEL_IDX < len(scores) else 0.0

        response = {
            "is_fall": fall_confidence >= 0.75,
            "confidence": float(min(fall_confidence, 1.0)),
            "fall_confidence": fall_confidence,
            "top_label": top_idx,
            "top_confidence": top_confidence,
            "model_status": "ok"
        }

        # 10. 최적화 모듈 적용
        if OPTIMIZER_AVAILABLE:
            try:
                optimization = optimize_fall_detection(
                    scores_sequence,
                    frames_3d,
                    FALL_LABEL_IDX
                )
                
                # 최적화 모듈의 결과를 안전하게 가져옴
                response["is_fall"] = optimization.get("is_fall", response["is_fall"])
                response["confidence"] = optimization.get("final_score", response["confidence"])
                
                if "confidence_level" in optimization:
                    response["confidence_level"] = optimization["confidence_level"]
                if "phase1_validation" in optimization:
                    response["optimization_details"] = {
                        "final_score": optimization.get("final_score"),
                        "phase1": optimization.get("phase1_validation"),
                        "phase2": optimization.get("phase2_temporal"),
                        "phase3": optimization.get("phase3_stability")
                    }
            except Exception as e:
                print(f"[경고] 최적화 실패: {e}, 기본 판정 사용")

        return response

    except Exception as e:
        print(f"[오류] 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        return _get_error_response()


def _get_error_response() -> Dict:
    """에러 응답"""
    return {
        "is_fall": False,
        "confidence": 0.0,
        "fall_confidence": 0.0,
        "top_label": -1,
        "top_confidence": 0.0,
        "model_status": "error"
    }
