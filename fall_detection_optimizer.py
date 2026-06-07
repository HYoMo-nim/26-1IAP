# 낙상 탐지 최적화 모듈
# 오탐 감소 및 정확도 향상을 위한 다중 검증 시스템

import numpy as np
from typing import Dict, Tuple

# ───────────────────────────────────────────
# 설정값
# ───────────────────────────────────────────

# 신뢰도 임계값 (Phase 1)
FALL_CONFIDENCE_THRESHOLD = 0.75      # 낙상 신뢰도 최소값
TOP_CLASS_CONFIDENCE_MIN = 0.40       # 최고 클래스 신뢰도 최소값
CONFIDENCE_MARGIN = 0.15              # 낙상과 2위 신뢰도 차이

# 시간 연속성 (Phase 2)
MIN_CONSECUTIVE_FRAMES = 60           # 약 2초 (30fps 기준)
CONSISTENCY_PERCENTILE = 70           # 상위 30% 프레임만 분석

# 포즈 안정성 (Phase 3)
MOTION_THRESHOLD_RATIO = 2.5          # peak_motion / avg_motion 비율
STABILITY_RATIO = 0.8                 # 시작/종료 움직임 비율

# 앙상블 가중치 (Phase 4)
ENSEMBLE_WEIGHTS = {
    "fall_confidence": 0.4,
    "temporal_consistency": 0.3,
    "pose_stability": 0.2,
    "confidence_margin": 0.1
}
ENSEMBLE_THRESHOLD = 0.65             # 최종 판정 임계값


# ───────────────────────────────────────────
# Phase 1: 신뢰도 검증
# ───────────────────────────────────────────

def validate_confidence(fall_confidence: float, 
                       top_confidence: float,
                       second_highest_confidence: float) -> Dict:
    """
    신뢰도 기반 1차 필터링
    
    Returns:
        {
            "is_valid": bool,
            "reasons": list of str,
            "confidence_score": float  # 0~1
        }
    """
    reasons = []
    is_valid = True
    
    # 1. 낙상 신뢰도 확인
    if fall_confidence < FALL_CONFIDENCE_THRESHOLD:
        reasons.append(f"Fall confidence too low: {fall_confidence:.3f} < {FALL_CONFIDENCE_THRESHOLD}")
        is_valid = False
    
    # 2. 최고 신뢰도 확인
    if top_confidence < TOP_CLASS_CONFIDENCE_MIN:
        reasons.append(f"Top confidence too low: {top_confidence:.3f} < {TOP_CLASS_CONFIDENCE_MIN}")
        is_valid = False
    
    # 3. 신뢰도 마진 확인 (낙상과 2위의 차이)
    margin = fall_confidence - second_highest_confidence
    if margin < CONFIDENCE_MARGIN:
        reasons.append(f"Confidence margin too small: {margin:.3f} < {CONFIDENCE_MARGIN}")
        is_valid = False
    
    # 신뢰도 점수 계산
    confidence_score = 0.0
    if fall_confidence >= FALL_CONFIDENCE_THRESHOLD:
        confidence_score += 0.4 * (min(fall_confidence, 1.0) / FALL_CONFIDENCE_THRESHOLD)
    if margin >= CONFIDENCE_MARGIN:
        confidence_score += 0.3 * (min(margin, 0.5) / CONFIDENCE_MARGIN)
    if top_confidence >= TOP_CLASS_CONFIDENCE_MIN:
        confidence_score += 0.3 * (min(top_confidence, 1.0) / TOP_CLASS_CONFIDENCE_MIN)
    
    return {
        "is_valid": is_valid,
        "reasons": reasons,
        "confidence_score": float(min(confidence_score, 1.0))
    }


# ───────────────────────────────────────────
# Phase 2: 시간 연속성 검사
# ───────────────────────────────────────────

def analyze_temporal_consistency(scores_sequence: np.ndarray, 
                                fall_label_idx: int = 43) -> Dict:
    """
    여러 프레임의 신뢰도 시퀀스 분석
    
    Args:
        scores_sequence: [T, num_classes] 신뢰도 배열
        fall_label_idx: 낙상 클래스 인덱스
    
    Returns:
        {
            "is_consistent_fall": bool,
            "consistency_score": float,
            "max_consecutive_frames": int,
            "peak_frame": int
        }
    """
    
    T = scores_sequence.shape[0]
    fall_scores = scores_sequence[:, fall_label_idx]  # [T]
    
    # 1. 상위 CONSISTENCY_PERCENTILE% 구간만 분석
    threshold = np.percentile(fall_scores, CONSISTENCY_PERCENTILE)
    high_confidence_frames = fall_scores > threshold
    
    # 2. 연속된 구간 찾기
    consecutive_groups = []
    current_group = []
    
    for i, is_high in enumerate(high_confidence_frames):
        if is_high:
            current_group.append(i)
        else:
            if len(current_group) > 0:
                consecutive_groups.append(current_group)
                current_group = []
    
    if len(current_group) > 0:
        consecutive_groups.append(current_group)
    
    # 3. 최대 연속 구간
    if not consecutive_groups:
        return {
            "is_consistent_fall": False,
            "consistency_score": 0.0,
            "max_consecutive_frames": 0,
            "peak_frame": int(np.argmax(fall_scores))
        }
    
    max_group = max(consecutive_groups, key=len)
    max_group_length = len(max_group)
    
    # 4. 연속성 점수
    if max_group_length < MIN_CONSECUTIVE_FRAMES:
        consistency_score = 0.0
        is_consistent = False
    else:
        consistency_score = min(1.0, max_group_length / (MIN_CONSECUTIVE_FRAMES * 2))
        is_consistent = True
    
    # 5. 피크 프레임
    peak_frame = int(np.argmax(fall_scores))
    
    return {
        "is_consistent_fall": is_consistent,
        "consistency_score": float(consistency_score),
        "max_consecutive_frames": max_group_length,
        "peak_frame": peak_frame
    }


# ───────────────────────────────────────────
# Phase 3: 포즈 안정성 분석
# ───────────────────────────────────────────

def analyze_pose_stability(frames_3d: np.ndarray) -> Dict:
    """
    포즈의 변화율을 분석하여 낙상 특징 확인
    
    낙상의 특징:
    - 처음: 안정적 (motion 낮음)
    - 중간: 급격한 변화 (peak motion 높음)
    - 후반: 움직임 거의 없음
    
    Args:
        frames_3d: [T, J, 3] 3D skeleton 배열
    
    Returns:
        {
            "is_stable": bool,
            "stability_score": float,
            "motion_profile": list
        }
    """
    
    # 프레임 간 포즈 변화 계산 (Euclidean distance)
    diff = np.diff(frames_3d, axis=0)  # [T-1, J, 3]
    motion = np.sqrt(np.sum(diff**2, axis=(1, 2)))  # [T-1]
    
    # 통계
    avg_motion = np.mean(motion)
    peak_motion = np.max(motion)
    first_half = motion[:len(motion)//2]
    second_half = motion[len(motion)//2:]
    
    avg_first_half = np.mean(first_half) if len(first_half) > 0 else 0
    avg_second_half = np.mean(second_half) if len(second_half) > 0 else 0
    
    # 낙상 특징 확인
    is_stable_start = avg_first_half < avg_motion * STABILITY_RATIO
    has_peak = peak_motion > avg_motion * MOTION_THRESHOLD_RATIO
    is_stable_end = avg_second_half < avg_motion * STABILITY_RATIO
    
    # 안정성 점수
    if is_stable_start and has_peak and is_stable_end:
        stability_score = 1.0
    elif is_stable_start or is_stable_end:
        stability_score = 0.5
    else:
        stability_score = 0.0
    
    return {
        "is_stable": float(stability_score) >= 0.5,
        "stability_score": float(stability_score),
        "motion_profile": motion.tolist(),
        "avg_motion": float(avg_motion),
        "peak_motion": float(peak_motion)
    }


# ───────────────────────────────────────────
# Phase 4: 앙상블 최종 판정
# ───────────────────────────────────────────

def ensemble_final_decision(confidence_validation: Dict,
                           temporal_consistency: Dict,
                           pose_stability: Dict) -> Dict:
    """
    4가지 지표를 통합하여 최종 판정
    
    Returns:
        {
            "is_fall": bool,
            "final_score": float,
            "confidence_level": str,  # "high", "medium", "low"
            "component_scores": dict
        }
    """
    
    # 각 컴포넌트 정규화
    component_scores = {
        "confidence": min(confidence_validation["confidence_score"] / FALL_CONFIDENCE_THRESHOLD, 1.0),
        "temporal_consistency": temporal_consistency["consistency_score"],
        "pose_stability": pose_stability["stability_score"]
    }
    
    # 가중 합산
    final_score = (
        component_scores["confidence"] * ENSEMBLE_WEIGHTS["fall_confidence"] +
        component_scores["temporal_consistency"] * ENSEMBLE_WEIGHTS["temporal_consistency"] +
        component_scores["pose_stability"] * ENSEMBLE_WEIGHTS["pose_stability"]
    )
    
    # 최종 판정
    is_fall = final_score >= ENSEMBLE_THRESHOLD
    
    # 신뢰도 레벨
    if final_score >= 0.85:
        confidence_level = "high"
    elif final_score >= 0.65:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    return {
        "is_fall": is_fall,
        "final_score": float(final_score),
        "confidence_level": confidence_level,
        "component_scores": component_scores,
        "ensemble_weights": ENSEMBLE_WEIGHTS
    }


# ───────────────────────────────────────────
# 통합 함수
# ───────────────────────────────────────────

def optimize_fall_detection(scores_sequence: np.ndarray,
                           frames_3d: np.ndarray,
                           fall_label_idx: int = 43) -> Dict:
    """
    전체 최적화 파이프라인 실행
    
    Args:
        scores_sequence: [T, num_classes] 신뢰도 배열
        frames_3d: [T, J, 3] 3D skeleton 배열
        fall_label_idx: 낙상 클래스 인덱스
    
    Returns:
        {
            "is_fall": bool,
            "final_score": float,
            "confidence_level": str,
            "phase1": {...},
            "phase2": {...},
            "phase3": {...},
            "decision": {...}
        }
    """
    
    # Phase 1: 신뢰도 검증
    fall_conf = float(scores_sequence[-1, fall_label_idx])  # 마지막 프레임
    top_idx = int(np.argmax(scores_sequence[-1]))
    top_conf = float(scores_sequence[-1, top_idx])
    
    # 2위 신뢰도
    sorted_scores = np.sort(scores_sequence[-1])
    second_highest = float(sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
    
    phase1 = validate_confidence(fall_conf, top_conf, second_highest)
    
    # Phase 2: 시간 연속성
    phase2 = analyze_temporal_consistency(scores_sequence, fall_label_idx)
    
    # Phase 3: 포즈 안정성
    phase3 = analyze_pose_stability(frames_3d)
    
    # Phase 4: 앙상블 판정
    decision = ensemble_final_decision(phase1, phase2, phase3)
    
    return {
        "is_fall": decision["is_fall"],
        "final_score": decision["final_score"],
        "confidence_level": decision["confidence_level"],
        "phase1_validation": phase1,
        "phase2_temporal": phase2,
        "phase3_stability": phase3,
        "final_decision": decision
    }
