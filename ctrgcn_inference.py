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
        MODEL = Model(
            num_class=60,          
            num_point=25,         
            num_person=2,         
            graph='graph.ntu_rgb_d.Graph', 
            graph_args={'labeling_mode': 'spatial'},
            in_channels=3         
        )
        MODEL.load_state_dict(checkpoint)
        
        # BN(Batch Norm) 에러 방지용 추론 모드 전환 (매우 중요)
        MODEL.eval()
        
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

        import torch
        
        # 1. 넘파이 배열(T, V, C)을 텐서로 변환
        input_tensor = torch.from_numpy(frames_3d).to(DEVICE)
        
        # 추론
        with torch.no_grad():
            if callable(model):
                tensor_3d = torch.tensor(frames_3d, dtype=torch.float32)
        
        # 1. 차원 순서 변경: (T=243, V=17, C=3) -> (C=3, T=243, V=17)
                tensor_3d = tensor_3d.permute(2, 0, 1)
        
        # 2. 앞뒤에 빈 축(차원) 추가: (1, 3, 243, 17, 1)
                input_tensor = tensor_3d.unsqueeze(0).unsqueeze(-1).to('cpu')  # gpu 사용 시 'cuda'로 변경

        # 차원 크기 확인 (N=1, C=3, T=243, V=17, M=1)
                N, C, T, V, M = input_tensor.size()

                # 4. 17관절 -> 25관절 해부학적 매핑 (H36M -> NTU-RGB+D 포맷 변환)
                if V == 17:
                    ntu_tensor = torch.zeros(N, C, T, 25, M).to(input_tensor.device)
                    
                    ntu_tensor[:, :, :, 0, :] = input_tensor[:, :, :, 0, :] # 골반
                    ntu_tensor[:, :, :, 1, :] = input_tensor[:, :, :, 7, :] # 척추 중앙
                    ntu_tensor[:, :, :, 2, :] = input_tensor[:, :, :, 8, :] # 가슴
                    ntu_tensor[:, :, :, 3, :] = input_tensor[:, :, :, 10, :] # 머리
                    ntu_tensor[:, :, :, 4, :] = input_tensor[:, :, :, 11, :] # 왼어깨
                    ntu_tensor[:, :, :, 5, :] = input_tensor[:, :, :, 12, :] # 왼팔꿈치
                    ntu_tensor[:, :, :, 6, :] = input_tensor[:, :, :, 13, :] # 왼손목
                    ntu_tensor[:, :, :, 7, :] = input_tensor[:, :, :, 13, :] # 왼손(대체)
                    ntu_tensor[:, :, :, 8, :] = input_tensor[:, :, :, 14, :] # 오른어깨
                    ntu_tensor[:, :, :, 9, :] = input_tensor[:, :, :, 15, :] # 오른팔꿈치
                    ntu_tensor[:, :, :, 10, :] = input_tensor[:, :, :, 16, :] # 오른손목
                    ntu_tensor[:, :, :, 11, :] = input_tensor[:, :, :, 16, :] # 오른손(대체)
                    ntu_tensor[:, :, :, 12, :] = input_tensor[:, :, :, 4, :] # 왼골반
                    ntu_tensor[:, :, :, 13, :] = input_tensor[:, :, :, 5, :] # 왼무릎
                    ntu_tensor[:, :, :, 14, :] = input_tensor[:, :, :, 6, :] # 왼발목
                    ntu_tensor[:, :, :, 15, :] = input_tensor[:, :, :, 6, :] # 왼발(대체)
                    ntu_tensor[:, :, :, 16, :] = input_tensor[:, :, :, 1, :] # 오른골반
                    ntu_tensor[:, :, :, 17, :] = input_tensor[:, :, :, 2, :] # 오른무릎
                    ntu_tensor[:, :, :, 18, :] = input_tensor[:, :, :, 3, :] # 오른발목
                    ntu_tensor[:, :, :, 19, :] = input_tensor[:, :, :, 3, :] # 오른발(대체)
                    ntu_tensor[:, :, :, 20, :] = input_tensor[:, :, :, 9, :] # 목
                    ntu_tensor[:, :, :, 21, :] = input_tensor[:, :, :, 13, :] # 왼손끝(대체)
                    ntu_tensor[:, :, :, 22, :] = input_tensor[:, :, :, 13, :] # 왼엄지(대체)
                    ntu_tensor[:, :, :, 23, :] = input_tensor[:, :, :, 16, :] # 오른손끝(대체)
                    ntu_tensor[:, :, :, 24, :] = input_tensor[:, :, :, 16, :] # 오른엄지(대체)
                    
                    input_tensor = ntu_tensor
# 0번 관절(골반)의 좌표를 추출하여 모든 관절의 위치에서 빼줍니다.
                    # 이를 통해 사람이 화면 어디에 있든 모델이 정확한 '자세'만 볼 수 있게 됩니다.
                    root_joint = input_tensor[:, :, :, 0:1, :]
                    input_tensor = input_tensor - root_joint
                    
                    # --- [신규 추가 부분: 3D 스케일 정규화 (Scale Normalization)] ---
                    # 추출된 3D 뼈대의 전체 크기를 모델이 학습한 규격(-1.0 ~ 1.0)으로 압축합니다.
                    max_val = torch.max(torch.abs(input_tensor))
                    if max_val > 0:
                        input_tensor = input_tensor / max_val
                    # ---------------------------------------------------
                # 5. 1명 -> 2명 패딩
                N, C, T, V, M = input_tensor.shape
                if M == 1:
                    zeros_m = torch.zeros(N, C, T, V, 1).to(input_tensor.device)
                    input_tensor = torch.cat([input_tensor, zeros_m], dim=4)
                
                # 6. 대망의 모델 추론!
                output = model(input_tensor)
            else:
                print("[오류] 모델이 callable 형태가 아닙니다")
                return _get_error_response()

        # 스코어 추출
        # --- [수정된 부분: 전체 프레임 평균 및 확률 변환] ---
        import torch.nn.functional as F
        
        # 출력이 (243, 60) 형태인 경우, 243개 프레임의 결과를 평균내어 1개의 종합 점수(60,)로 만듭니다.
        if output.dim() == 2 and output.shape[0] > 1:
            logits = output.mean(dim=0)
        else:
            # (1, 60) 형태인 경우 불필요한 차원을 제거하여 (60,)로 만듭니다.
            logits = output.squeeze()
            
        # 종합된 로짓 점수를 0.0~1.0 사이의 100% 확률로 변환합니다.
        probs = F.softmax(logits, dim=0)
        scores = probs.cpu().numpy()

        scores_sequence = np.tile(scores[np.newaxis, :], (T, 1))

        print(f"[CTR-GCN] 추론 완료: shape={scores_sequence.shape}")

        # 기본 결과
        top_idx = int(np.argmax(scores))
        top_confidence = float(scores[top_idx])
        fall_confidence = float(scores[FALL_LABEL_IDX]) if FALL_LABEL_IDX < len(scores) else 0.0

        # 최적화 적용
        response = {
            "is_fall": fall_confidence >= 0.75,
            "confidence": float(min(fall_confidence, 1.0)),
            "fall_confidence": fall_confidence,
            "top_label": top_idx,
            "top_confidence": top_confidence,
            "model_status": "ok"
        }

        if OPTIMIZER_AVAILABLE:
            try:
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
