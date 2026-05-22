# SQLite DB 연결 및 테이블 정의 모듈.
# SQLAlchemy ORM을 사용하여 detection_logs, keypoints_logs 테이블을 관리한다.

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# ───────────────────────────────────────────
# DB 연결 설정
# ───────────────────────────────────────────
DATABASE_URL = "sqlite:///./iot_server.db"  # 서버 폴더에 iot_server.db 파일 생성

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ───────────────────────────────────────────
# 테이블 정의
# ───────────────────────────────────────────

class DetectionLogDB(Base):
    """이상 탐지 로그 테이블 (/log/detection 수신 데이터)"""
    __tablename__ = "detection_logs"

    id         = Column(Integer, primary_key=True, index=True)
    device_id  = Column(String, index=True)           # 디바이스 식별자
    timestamp  = Column(DateTime)                      # 탐지 시각
    loss_value = Column(Float)                         # 손실값
    risk_level = Column(Integer)                       # 위험도 (0~10)
    received_at = Column(DateTime, default=datetime.utcnow)  # 서버 수신 시각


class KeypointLogDB(Base):
    """keypoints 수신 로그 테이블 (/skeleton/keypoints 수신 메타데이터)"""
    __tablename__ = "keypoints_logs"

    id          = Column(Integer, primary_key=True, index=True)
    device_id   = Column(String, index=True)          # 디바이스 식별자
    timestamp   = Column(DateTime)                     # 촬영 시각
    frame_count = Column(Integer)                      # 수신 프레임 수
    received_at = Column(DateTime, default=datetime.utcnow)  # 서버 수신 시각

# ───────────────────────────────────────────
# DB 초기화 (테이블 생성)
# ───────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)

# ───────────────────────────────────────────
# DB 세션 의존성
# ───────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
