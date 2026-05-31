# AWS EC2 서버에서 실행되는 FastAPI 백엔드.
# Jetson Nano로부터 이상 탐지 로그 및 2D keypoints 데이터를 HTTPS로 수신하고 DB에 저장한다.
import os
from datetime import datetime, timedelta, timezone
from typing import List

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from auth import verify_api_key
from database import init_db, get_db, DetectionLogDB, KeypointLogDB
app = FastAPI()


init_db()

# ───────────────────────────────────────────
# 문자 알림 설정(확정x)
# ───────────────────────────────────────────
ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
ALERT_ACTIONS = {"fall", "collapse", "unconscious"}
RISK_THRESHOLD = 7   # 임시 기준값을 7로 설정(0~10)
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_SECONDS = 300
SMS_ENABLED = os.getenv("SMS_ENABLED", "true").lower() == "true"

sms_client = boto3.client("sns", region_name=AWS_REGION)
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

    try:
        response = sms_client.publish(
            PhoneNumber=ALERT_PHONE_NUMBER,
            Message=message,
            MessageAttributes={
                "AWS.SNS.SMS.SMSType": {
                    "DataType": "String",
                    "StringValue": "Transactional",
                }
            },
        )
        return response.get("MessageId")
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"SMS send failed: {exc}",
        ) from exc


def run_keypoint_inference(frames: List["Frame"]):
    try:
        from mmaction_inference import run_inference
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="서버 환경에 아직 준비되지 않았습니다.",
        ) from exc

    return run_inference(frames)


# ───────────────────────────────────────────
# 기존: 이상 탐지 로그 수신 + 문자 알림
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
    db: Session = Depends(get_db)
):
    if log.device_id != device_id:
        raise HTTPException(
            status_code=403,
            detail="device_id와 API 키가 일치하지 않습니다."
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
    alert_send = False

    if send_now:
        sms_message = build_sms_message(log)
        sms_message_id = send_sms(sms_message)

        if sms_message_id and sms_message_id != "sms-disabled":
            alert_send = True
            cooldown_key = f"{log.device_id}:{log.action_label.lower()}"
            last_alert_sent_at[cooldown_key] = datetime.now(timezone.utc)

    print(
        f"[저장완료] device={log.device_id}, "
        f"action={log.action_label}, risk_level={log.risk_level}, "
        f"confidence={log.confidence}, loss={log.loss_value}, "
        f"alert_send={alert_send}, db_id={db_log.id}"
    )

    return {
        "status": "log received",
        "db_id": db_log.id,
        "alert_sent": alert_send,
        "decision_reason": reason,
        "sms_message_id": sms_message_id,
        "data": log.model_dump()
    }


# ───────────────────────────────────────────
# 신규: 2D keypoints 수신 + MMAction2 추론
# ───────────────────────────────────────────
NUM_KEYPOINTS = 17  
NUM_FRAMES = 243     
KEYPOINT_DIM = 3    


class Frame(BaseModel):
    frame_id: int
    keypoints: List[List[float]]  # shape: [17, 3] (x, y, confidence)

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
    db: Session = Depends(get_db)
):
    if payload.device_id != device_id:
        raise HTTPException(
            status_code=403,
            detail="device_id와 API 키가 일치하지 않습니다."
        )

    # DB 저장
    db_log = KeypointLogDB(
        device_id=payload.device_id,
        timestamp=payload.timestamp,
        frame_count=len(payload.frames),
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)


    # DB 저장
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

    # 판별 모델(MMAction2) 추론
    result = run_keypoint_inference(payload.frames)

    # 낙상 감지 시 알림 전송
    if result["is_fall"]:
        print(
            f"[경고] 낙상 감지! "
            f"device={payload.device_id}, "
            f"confidence={result['fall_confidence']:.3f}"
        )
        # TODO: 알림 모듈 연결 (팀원 담당)

    return {
        "status": "keypoints received",
        "db_id": db_log.id,
        "device_id": payload.device_id,
        "frames": len(payload.frames),
        "timestamp": payload.timestamp,
        "is_fall": result["is_fall"],
        "fall_confidence": result["fall_confidence"],
        "top_label": result["top_label"],
        "top_confidence": result["top_confidence"],
    }


# ───────────────────────────────────────────
# DB 조회 엔드포인트 (저장된 로그 확인용)
# ───────────────────────────────────────────
@app.get("/logs/detection")
def get_detection_logs(
    db: Session = Depends(get_db),
    device_id: str = Depends(verify_api_key)
):
    """저장된 이상 탐지 로그 전체 조회"""
    logs = db.query(DetectionLogDB).order_by(DetectionLogDB.received_at.desc()).all()
    return {"count": len(logs), "logs": logs}


@app.get("/logs/keypoints")
def get_keypoint_logs(
    db: Session = Depends(get_db),
    device_id: str = Depends(verify_api_key)
):
    """저장된 keypoints 수신 로그 전체 조회"""
    logs = db.query(KeypointLogDB).order_by(KeypointLogDB.received_at.desc()).all()
    return {"count": len(logs), "logs": logs}
