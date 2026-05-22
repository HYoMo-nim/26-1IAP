# AWS EC2 서버에서 실행되는 FastAPI 백엔드.
# Jetson Nano로부터 이상 탐지 로그 및 2D keypoints 데이터를 HTTPS로 수신하고 DB에 저장한다.

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List
from sqlalchemy.orm import Session

from auth import verify_api_key
from database import init_db, get_db, DetectionLogDB, KeypointLogDB

app = FastAPI()

# 서버 시작 시 DB 테이블 자동 생성
init_db()

# ───────────────────────────────────────────
# 기존: 이상 탐지 로그 수신
# ───────────────────────────────────────────
class DetectionLog(BaseModel):
    device_id  : str
    timestamp  : datetime
    loss_value : float = Field(..., ge=0)
    risk_level : int = Field(..., ge=0, le=10)

@app.get("/")
def root():
    return {"message" : "확인중"}

@app.post("/log/detection")
def receive_log(
    log       : DetectionLog,
    device_id : str = Depends(verify_api_key),
    db        : Session = Depends(get_db)
):
    db_log = DetectionLogDB(
        device_id  = log.device_id,
        timestamp  = log.timestamp,
        loss_value = log.loss_value,
        risk_level = log.risk_level,
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    print(f"[저장완료] device={log.device_id}, 위험도={log.risk_level}, loss={log.loss_value}, db_id={db_log.id}")
    return {"status": "log received", "db_id": db_log.id, "data": log.model_dump()}


# ───────────────────────────────────────────
# 신규: 2D keypoints 수신
# ───────────────────────────────────────────
NUM_KEYPOINTS  = 17   # YOLO26n-pose 관절 수
NUM_FRAMES     = 243  # VideoPose3D receptive field
KEYPOINT_DIM   = 3    # [x, y, confidence]

class Frame(BaseModel):
    frame_id  : int
    keypoints : List[List[float]]  # shape: [17, 3] (x, y, confidence)

    @field_validator("keypoints")
    @classmethod
    def check_keypoints(cls, v):
        if len(v) != NUM_KEYPOINTS:
            raise ValueError(f"keypoints는 {NUM_KEYPOINTS}개여야 합니다. 받은 개수: {len(v)}")
        for pt in v:
            if len(pt) not in [2, 3]:  # [x,y] 또는 [x,y,conf] 둘 다 허용
                raise ValueError("각 keypoint는 [x, y] 또는 [x, y, confidence] 형식이어야 합니다.")
        return v

class KeypointPayload(BaseModel):
    device_id : str
    timestamp : datetime
    frames    : List[Frame]  # 243프레임 묶음

    @field_validator("frames")
    @classmethod
    def check_frames(cls, v):
        if len(v) != NUM_FRAMES:
            raise ValueError(f"frames는 {NUM_FRAMES}개여야 합니다. 받은 개수: {len(v)}")
        return v

@app.post("/skeleton/keypoints")
def receive_keypoints(
    payload   : KeypointPayload,
    device_id : str = Depends(verify_api_key),
    db        : Session = Depends(get_db)
):
    db_log = KeypointLogDB(
        device_id   = payload.device_id,
        timestamp   = payload.timestamp,
        frame_count = len(payload.frames),
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    print(f"[저장완료] device={payload.device_id}, frames={len(payload.frames)}, db_id={db_log.id}")

    # TODO: 3D 변환 모델 (VideoPose3D) 연결
    # TODO: 판별 모델 (LSTM) 연결
    # TODO: 이상 징후 감지 시 알림 전송

    return {
        "status"    : "keypoints received",
        "db_id"     : db_log.id,
        "device_id" : payload.device_id,
        "frames"    : len(payload.frames),
        "timestamp" : payload.timestamp,
    }


# ───────────────────────────────────────────
# DB 조회 엔드포인트 (저장된 로그 확인용)
# ───────────────────────────────────────────
@app.get("/logs/detection")
def get_detection_logs(
    db        : Session = Depends(get_db),
    device_id : str = Depends(verify_api_key)
):
    """저장된 이상 탐지 로그 전체 조회"""
    logs = db.query(DetectionLogDB).order_by(DetectionLogDB.received_at.desc()).all()
    return {"count": len(logs), "logs": logs}

@app.get("/logs/keypoints")
def get_keypoint_logs(
    db        : Session = Depends(get_db),
    device_id : str = Depends(verify_api_key)
):
    """저장된 keypoints 수신 로그 전체 조회"""
    logs = db.query(KeypointLogDB).order_by(KeypointLogDB.received_at.desc()).all()
    return {"count": len(logs), "logs": logs}
