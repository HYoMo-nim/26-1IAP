# CTR-GCN 전용 낙상 탐지 최적화 모듈 (최종 호환성 패치 버전)
import numpy as np

# NTU-RGB+D 데이터셋 기준 낙상(Falling down) 클래스 인덱스
FALL_LABEL_IDX = 42

# ───────────────────────────────────────────
# 설정값
# ───────────────────────────────────────────
FALL_CONFIDENCE_THRESHOLD = 0.70      # 기준치를 살짝 낮춰서 감지율을 높임 (70%)
CONFIDENCE_MARGIN = 0.10              # 1위와 2위 행동 간의 최소 격차

def optimize_fall_detection(scores: np.ndarray, *args, **kwargs):
    """
    CTR-GCN이 예측한 행동 점수를 받아 낙상 여부를 최종 판별합니다.
    """
    
    # 1. 차원 축소: (243, 60) 형태로 들어올 경우, 전체 프레임의 평균을 내어 (60,) 형태로 압축
    if len(scores.shape) == 2:
        scores = np.mean(scores, axis=0)

    # 2. 42번 인덱스(낙상)의 신뢰도를 추출
    fall_confidence = float(scores[FALL_LABEL_IDX])
    
    # 3. 최고 신뢰도 및 2위 신뢰도 추출을 위한 정렬
    sorted_scores = np.sort(scores)
    top_confidence = float(sorted_scores[-1])
    second_highest_confidence = float(sorted_scores[-2])
    
    # 4. 모델이 예측한 1위 행동이 '낙상'인지 확인
    top_label_idx = int(np.argmax(scores))
    is_top_class_fall = (top_label_idx == FALL_LABEL_IDX)
    
    # 5. 낙상 신뢰도가 기준치를 넘는지 확인
    meets_threshold = fall_confidence >= FALL_CONFIDENCE_THRESHOLD
    
    # 6. 마진 검사
    margin = top_confidence - second_highest_confidence
    has_clear_margin = margin >= CONFIDENCE_MARGIN
    
    # 최종 판정
    is_fall = bool(is_top_class_fall and meets_threshold and has_clear_margin)
    
    # 메인 코드(ctrgcn_inference.py)가 에러를 내지 않도록 요구하는 모든 키값을 채워 반환
    return {
        "is_fall": is_fall,
        "final_score": fall_confidence,
        "confidence_level": "high" if is_fall else "low",
        # 아래부터는 메인 코드의 로그 출력을 위한 호환성 유지용 데이터입니다.
        "phase1_validation": {"confidence_score": fall_confidence, "reasons": []},
        "phase2_temporal": {"consistency_score": 1.0 if is_fall else 0.0},
        "phase3_stability": {"stability_score": 1.0 if is_fall else 0.0},
        "final_decision": {"is_fall": is_fall, "final_score": fall_confidence}
    }
