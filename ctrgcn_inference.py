# CTR-GCN 추론 모듈 (최적화 통합)
# 3D skeleton keypoints로 낙상 판별 및 오탐 감소

import os
import numpy as np
from typing import Optional, Dict

# 최적화 모듈 import
try:
    from fall_detection_optimizer import optimize_fall_detection, FALL_LABEL_IDX
    OPTIMIZER_AVAILABLE = True
except ImportError:
    print("[경고] fall_detection_optimizer 미설치")
    OPTIMIZER_AVAILABLE = False
    FALL_LABEL_IDX = 43

# PyTorch 조건부 로드
TORCH_AVAILABLE = False
MODEL = None
DEVICE = None

try:
    import torch
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
        
        print(f"[CTR-GCN] 모델 로드 완료")
        MODEL = load_model(checkpoint)
        return MODEL
        
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return None


def run_inference(frames_3d: np.ndarray) -> Dict:
    """
    3D skeleton keypoints로 낙상 판별 (최적화 통합)
    
    Args:
        frames_3d: [T, J, 3] 형태의 3D skeleton 배열
    
    Returns:
        {
            "is_fall": bool,
            "confidence": float,
            "fall_confidence": float,
            "top_label": int,
            "top_confidence": float,
            "model_status": str,
            "optimization_details": dict (선택)
        }
    """
    
    if not TORCH_AVAILABLE:
        print("[오류] PyTorch 필수: pip install -r requirements.txt")
        return _get_error_response()
    
    try:
        # 입력 검증
        if isinstance(frames_3d, list):
            frames_3d = np.array(frames_3d, dtype=np.float32)
        else:
            frames_3d = np.asarray(frames_3d, dtype=np.float32)
        
        if len(frames_3d.shape) != 3 or frames_3d.shape[-1] != 3:
            print(f"[오류] 입력 형태 오류: {frames_3d.shape}")
            return _get_error_response()
        
        T, J = frames_3d.shape[0], frames_3d.shape[1]
        print(f"[CTR-GCN] 입력: {T} 프레임, {J} 관절")
        
        # 모델 초기화
        model = _init_model()
        if model is None:
            print("[오류] 모델 초기화 실패")
            return _get_error_response()
        
        # 배치 추가 [T, J, 3] → [1, T, J, 3]
        import torch
        input_array = frames_3d[np.newaxis, ...]
        input_tensor = torch.from_numpy(input_array).to(DEVICE)
        
        # 추론
        with torch.no_grad():
            if callable(model):
                output = model(input_tensor)
            else:
                print("[오류] 모델이 callable 형태가 아닙니다")
                return _get_error_response()
        
        # 스코어 추출
        scores = output[0].cpu().numpy()  # [num_classes]
        
        # 스코어 시퀀스 생성 (매 프레임 추론하지 않으므로 근사)
        # 실제로는 매 프레임별 스코어가 있어야 하지만, 현재는 최종 스코어만 사용
        scores_sequence = np.tile(scores[np.newaxis, :], (T, 1))  # [T, num_classes]
        
        print(f"[CTR-GCN] 추론 완료: shape={scores_sequence.shape}")
        
        # 기본 결과
        top_idx = int(np.argmax(scores))
        top_confidence = float(scores[top_idx])
        fall_confidence = float(scores[FALL_LABEL_IDX]) if FALL_LABEL_IDX < len(scores) else 0.0
        
        # ─────────────────────────────────────────
        # 최적화 적용
        # ─────────────────────────────────────────
        response = {
            "is_fall": fall_confidence >= 0.75,  # 기본값
            "confidence": float(min(fall_confidence, 1.0)),
            "fall_confidence": fall_confidence,
            "top_label": top_idx,
            "top_confidence": top_confidence,
            "model_status": "ok"
        }
        
        if OPTIMIZER_AVAILABLE:
            try:
                # 최적화된 판정
                optimization = optimize_fall_detection(
                    scores_sequence, 
                    frames_3d,
                    FALL_LABEL_IDX
                )
                
                response["is_fall"] = optimization["is_fall"]
                response["confidence"] = optimization["final_score"]
                response["confidence_level"] = optimization["confidence_level"]
                response["optimization_details"] = {
                    "final_score": optimization["final_score"],
                    "phase1": optimization["phase1_validation"],
                    "phase2": optimization["phase2_temporal"],
                    "phase3": optimization["phase3_stability"]
                }
                
                print(f"[CTR-GCN] 최적화 판정: {response['is_fall']} (score: {optimization['final_score']:.3f})")
                
            except Exception as e:
                print(f"[경고] 최적화 실패: {e}, 기본 판정 사용")
        else:
            print("[경고] 최적화 모듈 미사용, 기본 판정만 사용")
        
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

