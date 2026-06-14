# AWS EC2 서버에서 실행되는 FastAPI 백엔드.
# Jetson Nano로부터 이상 탐지 로그 및 2D keypoints 데이터를 HTTPS로 수신하고 DB에 저장한다.

import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from auth import verify_api_key
from ctrgcn_inference import run_inference as run_ctrgcn_inference
from database import init_db, get_db, DetectionLogDB, KeypointLogDB
from videopose_inference import run_videopose3d

app = FastAPI()

init_db()

# ───────────────────────────────────────────
# 문자 알림 설정
# ───────────────────────────────────────────
ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
ORIGINATION_IDENTITY = os.getenv("ORIGINATION_IDENTITY", "TESTSMS")

ALERT_ACTIONS = {"fall", "collapse", "unconscious"}
RISK_THRESHOLD = 7
CONFIDENCE_THRESHOLD = 0.015
COOLDOWN_SECONDS = 300
SMS_ENABLED = os.getenv("SMS_ENABLED", "true").lower() == "true"

# AWS End User Messaging SMS 클라이언트
sms_client = boto3.client("pinpoint-sms-voice-v2", region_name=AWS_REGION)

last_alert_sent_at: dict[str, datetime] = {}


def should_send_sms(
    device_id: str,
    action_label: str,
    risk_level: int,
    confidence: float,
) -> tuple[bool, str]:
    action = action_label.lower()

    if action not in ALERT_ACTIONS:
        return False, "action_not_target"

    if risk_level < RISK_THRESHOLD:
        return False, "risk_too_low"

    if confidence < CONFIDENCE_THRESHOLD:
        return False, "confidence_too_low"

    cooldown_key = f"{device_id}:{action}"
    now = datetime.now(timezone.utc)
    last_sent = last_alert_sent_at.get(cooldown_key)

    if last_sent and (now - last_sent) < timedelta(seconds=COOLDOWN_SECONDS):
        return False, "cooldown_active"

    return True, "ready"


def build_sms_message(log: "DetectionLog") -> str:
    event_time = log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "[이상행동 감지 알림]\n"
        f"장치: {log.device_id}\n"
        f"위치: {log.location}\n"
        f"행동: {log.action_label}\n"
        f"위험도: {log.risk_level}/10\n"
        f"신뢰도: {log.confidence:.2f}\n"
        f"발생시각: {event_time}"
    )


def send_sms(message: str):
    if not SMS_ENABLED:
        return "sms-disabled"

    if not ALERT_PHONE_NUMBER:
        raise HTTPException(
            status_code=500,
            detail="ALERT_PHONE_NUMBER is not configured.",
        )

    if not ORIGINATION_IDENTITY:
        raise HTTPException(
            status_code=500,
            detail="ORIGINATION_IDENTITY is not configured.",
        )

    try:
        response = sms_client.send_text_message(
            DestinationPhoneNumber=ALERT_PHONE_NUMBER,
            OriginationIdentity=ORIGINATION_IDENTITY,
            MessageBody=message,
            MessageType="TRANSACTIONAL",
        )
        return response.get("MessageId")

    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"SMS send failed: {exc}",
        ) from exc


# ───────────────────────────────────────────
# 이상 탐지 로그 수신 + 문자 알림
# ───────────────────────────────────────────
class DetectionLog(BaseModel):
    device_id: str
    timestamp: datetime
    loss_value: float = Field(..., ge=0)
    risk_level: int = Field(..., ge=0, le=10)
    action_label: str = Field(..., min_length=1, max_length=50)
    confidence: float = Field(..., ge=0.0, le=1.0)
    location: str = Field(..., min_length=1, max_length=100)


@app.get("/")
def root():
    return {"message": "확인중"}


@app.post("/log/detection")
def receive_log(
    log: DetectionLog,
    device_id: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    if log.device_id != device_id:
        raise HTTPException(
            status_code=403,
            detail="device_id와 API 키가 일치하지 않습니다.",
        )

    db_log = DetectionLogDB(
        device_id=log.device_id,
        timestamp=log.timestamp,
        loss_value=log.loss_value,
        risk_level=log.risk_level,
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    send_now, reason = should_send_sms(
        log.device_id,
        log.action_label,
        log.risk_level,
        log.confidence,
    )

    sms_message_id = None
    alert_sent = False

    if send_now:
        sms_message = build_sms_message(log)
        sms_message_id = send_sms(sms_message)

        if sms_message_id and sms_message_id != "sms-disabled":
            alert_sent = True
            cooldown_key = f"{log.device_id}:{log.action_label.lower()}"
            last_alert_sent_at[cooldown_key] = datetime.now(timezone.utc)

    print(
        f"[저장완료] device={log.device_id}, "
        f"action={log.action_label}, risk_level={log.risk_level}, "
        f"confidence={log.confidence}, loss={log.loss_value}, "
        f"alert_sent={alert_sent}, db_id={db_log.id}"
    )

    return {
        "status": "log received",
        "db_id": db_log.id,
        "alert_sent": alert_sent,
        "decision_reason": reason,
        "sms_message_id": sms_message_id,
        "data": log.model_dump(),
    }

def visualize_3d_skeleton(frames_3d, frame_idx=0):
    """
    VideoPose3D가 변환한 3D 뼈대(1개 프레임)를 이미지 파일로 저장하여 눈으로 확인합니다.
    """
    try:
        # 특정 프레임의 3D 관절 좌표 추출 (17, 3)
        skeleton = frames_3d[frame_idx] 
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 각 관절을 점으로 찍기
        xs, ys, zs = skeleton[:, 0], skeleton[:, 1], skeleton[:, 2]
        ax.scatter(xs, ys, zs, c='r', marker='o')
        
        # 관절 번호 텍스트 달기
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, str(i), fontsize=8, color='blue')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Skeleton (Frame {frame_idx})')
        
        # 파일로 저장 (서버 폴더에 skeleton_test.png로 저장됨)
        plt.savefig('skeleton_test.png')
        plt.close()
        print(f"[디버깅] 3D 스켈레톤 시각화 이미지가 skeleton_test.png로 저장되었습니다.")
        
    except Exception as e:
        print(f"[시각화 오류] {e}")

# ───────────────────────────────────────────
# 2D keypoints 수신 + VideoPose3D + CTR-GCN 판별
# ───────────────────────────────────────────
NUM_KEYPOINTS = 17
NUM_FRAMES = 243
KEYPOINT_DIM = 3


class Frame(BaseModel):
    frame_id: int
    keypoints: List[List[float]]

    @field_validator("keypoints")
    @classmethod
    def check_keypoints(cls, v):
        if len(v) != NUM_KEYPOINTS:
            raise ValueError(f"keypoints는 {NUM_KEYPOINTS}개여야 합니다. 받은 개수: {len(v)}")
        for pt in v:
            if len(pt) not in [2, 3]:
                raise ValueError("각 keypoint는 [x, y] 또는 [x, y, confidence] 형식이어야 합니다.")
        return v


class KeypointPayload(BaseModel):
    device_id: str
    timestamp: datetime
    frames: List[Frame]

    @field_validator("frames")
    @classmethod
    def check_frames(cls, v):
        if len(v) != NUM_FRAMES:
            raise ValueError(f"frames는 {NUM_FRAMES}개여야 합니다. 받은 개수: {len(v)}")
        return v


@app.post("/skeleton/keypoints")
def receive_keypoints(
    payload: KeypointPayload,
    device_id: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    if payload.device_id != device_id:
        raise HTTPException(
            status_code=403,
            detail="device_id와 API 키가 일치하지 않습니다.",
        )

    db_log = KeypointLogDB(
        device_id=payload.device_id,
        timestamp=payload.timestamp,
        frame_count=len(payload.frames),
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    print(
        f"[저장완료] device={payload.device_id}, "
        f"frames={len(payload.frames)}, "
        f"db_id={db_log.id}"
    )

    # ───────────────────────────────────────────
    # 파이프라인: 2D keypoints → 3D 변환 → 낙상 판별
    # ───────────────────────────────────────────
    
    try:
        # Step 1: 2D keypoints 추출
        frames_2d = []
        for frame in payload.frames:
            keypoints_2d = [[kp[0], kp[1]] for kp in frame.keypoints]  # [x, y]만 추출
            frames_2d.append(keypoints_2d)
        
        frames_2d_array = np.array(frames_2d, dtype=np.float32)
        print(f"[Step 1] 2D keypoints 추출: {frames_2d_array.shape}")
        
        # Step 2: VideoPose3D: 2D → 3D 변환
        print(f"[Step 2] VideoPose3D 변환 시작...")
        frames_3d = run_videopose3d(frames_2d_array)
        
        if frames_3d is None:
            print(f"[오류] VideoPose3D 변환 실패")
            return {
                "status": "error",
                "message": "VideoPose3D 변환 실패",
                "db_id": db_log.id
            }
        
        # numpy 배열로 변환
        if isinstance(frames_3d, list):
            frames_3d = np.array(frames_3d, dtype=np.float32)
        
        print(f"[Step 2] VideoPose3D 변환 완료: {frames_3d.shape}")
        visualize_3d_skeleton(frames_3d, frame_idx=120)

        # Step 3: CTR-GCN: 3D keypoints로 낙상 판별
        print(f"[Step 3] CTR-GCN 판별 시작...")
        inference_result = run_ctrgcn_inference(frames_3d)
        
        print(f"[Step 3] CTR-GCN 판별 완료")
        print(f"         is_fall: {inference_result['is_fall']}")
        print(f"         confidence: {inference_result.get('confidence', inference_result.get('fall_confidence', 0)):.3f}")
        print(f"         [디버그] 1위 예측 행동(Label): {inference_result.get('top_label')} (확률: {inference_result.get('top_confidence'):.3f})")
        
        # Step 4: 결과 분석 및 알림
        is_fall = inference_result.get("is_fall", False)
        confidence = inference_result.get("confidence", inference_result.get("fall_confidence", 0))
        top_label = inference_result.get("top_label", "unknown")
        top_confidence = inference_result.get("top_confidence", 0)
        
        # 낙상 감지 시 알림 전송
        alert_sent = False
        alert_reason = ""
        
        if is_fall:
            # 1. 현재 시간 및 쿨다운 키 생성
            now = datetime.now(timezone.utc)
            cooldown_key = f"{payload.device_id}:fall"
            last_sent = last_alert_sent_at.get(cooldown_key)
            
            # 2. 쿨다운 적용 중인지 확인 (COOLDOWN_SECONDS는 상단에 300으로 정의되어 있음)
            if last_sent and (now - last_sent) < timedelta(seconds=COOLDOWN_SECONDS):
                alert_reason = "cooldown_active"
                remaining_time = COOLDOWN_SECONDS - (now - last_sent).seconds
                print(f"[알림 스킵] 쿨다운 적용 중 (남은 시간: {remaining_time}초)")
            
            # 3. 쿨다운이 지났거나 처음 발생한 경우 문자 전송
            else:
                print(f"[경고] 낙상 감지!")
                print(f"       device={payload.device_id}")
                print(f"       confidence={confidence:.3f}")
                print(f"       top_label={top_label}")
                
                # SMS 알림 발송
                sms_message = (
                    f"[낙상 감지 알림]\n"
                    f"장치: {payload.device_id}\n"
                    f"신뢰도: {confidence:.1%}\n"
                    f"시각: {payload.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                try:
                    sms_message_id = send_sms(sms_message)
                    if sms_message_id and sms_message_id != "sms-disabled":
                        alert_sent = True
                        alert_reason = "SMS sent"
                        # 4. 문자 전송 성공 시 타이머 갱신
                        last_alert_sent_at[cooldown_key] = now
                        print(f"[성공] SMS 전송됨: {sms_message_id}")
                except Exception as e:
                    alert_reason = f"SMS 전송 실패: {str(e)}"
                    print(f"[경고] {alert_reason}")

        # Step 5: 응답 생성
        response = {
            "status": "success",
            "db_id": db_log.id,
            "device_id": payload.device_id,
            "frames": len(payload.frames),
            "timestamp": payload.timestamp.isoformat(),
            "pipeline_result": {
                "is_fall": is_fall,
                "confidence": float(confidence),
                "confidence_level": inference_result.get("confidence_level", "unknown"),
                "top_label": str(top_label),
                "top_confidence": float(top_confidence),
                "model_status": inference_result.get("model_status", "unknown")
            },
            "alert": {
                "sent": alert_sent,
                "reason": alert_reason
            }
        }
        
        # 최적화 세부사항 추가 (있으면)
        if "optimization_details" in inference_result:
            response["optimization"] = inference_result["optimization_details"]
        
        return response
        
    except Exception as e:
        print(f"[치명적 오류] 파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()
        
# DB 조회 엔드포인트
# ───────────────────────────────────────────
@app.get("/logs/detection")
def get_detection_logs(
    db: Session = Depends(get_db),
    device_id: str = Depends(verify_api_key),
):
    logs = db.query(DetectionLogDB).order_by(DetectionLogDB.received_at.desc()).all()
    return {"count": len(logs), "logs": logs}


@app.get("/logs/keypoints")
def get_keypoint_logs(
    db: Session = Depends(get_db),
    device_id: str = Depends(verify_api_key),
):
    logs = db.query(KeypointLogDB).order_by(KeypointLogDB.received_at.desc()).all()
    return {"count": len(logs), "logs": logs}
