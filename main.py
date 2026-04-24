from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List
from auth import verify_api_key

app = FastAPI()

# ───────────────────────────────────────────
# 기존: 이상 탐지 로그 수신(데이터 형식 정의)
# ───────────────────────────────────────────
class DetectionLog(BaseModel):
    device_id : str
    timestamp : datetime
    loss_value : float = Field(..., ge=0)
    risk_level : int = Field(..., ge=0, le=10)

@app.get("/")
def root():
    return {"message" : "확인중"}

@app.post("/log/detection")
def receive_log(log: DetectionLog, device_id: str = Depends(verify_api_key)):
    print(f"Received Log: {log.device_id}, 위험도: {log.risk_level}, loss={log.loss_value}")
    return {"status": "log received", "data": log.model_dump()}


# ───────────────────────────────────────────
# 신규: 2D keypoints 수신
# ───────────────────────────────────────────
NUM_KEYPOINTS = 17  # 관절 개수 (17 or 18)
NUM_FRAMES    = 30  # LSTM 입력 시퀀스 길이

class Frame(BaseModel):
    frame_id  : int
    keypoints : List[List[float]]  # shape: [NUM_KEYPOINTS, 2] (x, y)

    @field_validator("keypoints")
    @classmethod
    def check_keypoints(cls, v):
        if len(v) != NUM_KEYPOINTS:
            raise ValueError(f"keypoints는 {NUM_KEYPOINTS}개여야 합니다. 받은 개수: {len(v)}")
        for pt in v:
            if len(pt) != 2:
                raise ValueError("각 keypoint는 [x, y] 형식이어야 합니다.")
        return v

class KeypointPayload(BaseModel):
    device_id : str
    timestamp : datetime
    frames    : List[Frame]  # 30프레임 묶음

    @field_validator("frames")
    @classmethod
    def check_frames(cls, v):
        if len(v) != NUM_FRAMES:
            raise ValueError(f"frames는 {NUM_FRAMES}개여야 합니다. 받은 개수: {len(v)}")
        return v

@app.post("/skeleton/keypoints")
def receive_keypoints(
    payload   : KeypointPayload,
    device_id : str = Depends(verify_api_key)
):
    print(f"[{payload.device_id}] keypoints 수신 - {len(payload.frames)}프레임 / {payload.timestamp}")

    # TODO: 3D 변환 모델 연결
    # TODO: 판별 모델 (LSTM) 연결
    # TODO: 이상 징후 감지 시 알림 전송

    return {
        "status"    : "keypoints received",
        "device_id" : payload.device_id,
        "frames"    : len(payload.frames),
        "timestamp" : payload.timestamp,
    }