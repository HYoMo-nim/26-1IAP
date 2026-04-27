# 26-1IAP 서버

낙상 감지 시스템의 백엔드 서버입니다.  
촬영된 영상을 스켈레톤 데이터로 변환하여 이상 징후(낙상 등) 발생 시 지정된 사용자에게 알림을 전송합니다.

---

## 시스템 구조

```
Jetson Nano (현장)                AWS 서버
──────────────────                ──────────────────────
카메라
  ↓
2D 탐지 모델
  ↓
2D keypoints 추출
  ↓
HTTPS 전송 ──────────────────→   /skeleton/keypoints 수신
                                  ↓
                                  3D 변환 모델 (TODO)
                                  ↓
                                  판별 모델 LSTM (TODO)
                                  ↓
                                  이상 징후 → 알림 전송 (TODO)
```

---

## 서버 정보

| 항목 | 내용 |
|---|---|
| 클라우드 | AWS EC2 |
| 리전 | 아시아 태평양 (서울) |
| 서버 IP | 3.34.122.34 |
| 포트 | 8443 (HTTPS) |
| OS | Ubuntu 24.04 LTS |

---

## API 엔드포인트

### 인증 방식
모든 POST 요청에 아래 헤더 필요:
```
X-API-Key: {발급된 API Key}
```

---

### GET /
서버 상태 확인

**응답 예시**
```json
{"message": "확인중"}
```

---

### POST /log/detection
이상 탐지 로그 수신

**요청 예시**
```json
{
  "device_id": "jetson-01",
  "timestamp": "2024-06-01T12:00:00",
  "loss_value": 0.82,
  "risk_level": 3
}
```

**응답 예시**
```json
{
  "status": "log received",
  "data": {
    "device_id": "jetson-01",
    "timestamp": "2024-06-01T12:00:00",
    "loss_value": 0.82,
    "risk_level": 3
  }
}
```

---

### POST /skeleton/keypoints
Jetson Nano에서 추출한 2D keypoints 수신

**요청 형식**
- 관절 수: 17개
- 프레임 수: 30프레임 (1초 분량)

---

## 파일 구조

```
26-1IAP/
├── main.py          # API 엔드포인트
├── auth.py          # API Key 인증
├── requirements.txt # 패키지 목록
└── test_main.py     # 테스트 코드
```

---

## 서버 실행 방법

```bash
cd ~/app
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile ./certs/key.pem \
  --ssl-certfile ./certs/cert.pem
```

---

## 보안

- **API Key 인증**: 등록된 디바이스만 데이터 전송 가능
- **TLS/HTTPS**: 전송 구간 암호화 (자체 서명 인증서)

---

## TODO

- [ ] 3D 변환 모델 연결
- [ ] LSTM 판별 모델 연결
- [ ] 이상 징후 감지 시 알림 전송
- [ ] Jetson 2D keypoints 추출 코드 연동