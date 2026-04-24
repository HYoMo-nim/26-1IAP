# API 동작 확인
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_KEY = "jetson-01-secret-key"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "확인중"}

def test_receive_log_success():
    """유효한 API Key로 정상 수신되는지 확인"""
    payload = {
        "device_id": "jetson-01",
        "timestamp": "2024-06-01T12:00:00",
        "loss_value": 0.82,
        "risk_level": 3
    }
    response = client.post(
        "/log/detection",
        json=payload,
        headers={"X-API-Key": VALID_KEY}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "log received"
    assert body["data"]["device_id"] == "jetson-01"
    assert body["data"]["risk_level"] == 3

def test_receive_log_no_key():
    """API Key 없이 요청 시 401 반환 확인"""
    payload = {
        "device_id": "jetson-01",
        "timestamp": "2024-06-01T12:00:00",
        "loss_value": 0.82,
        "risk_level": 3
    }
    response = client.post("/log/detection", json=payload)
    assert response.status_code == 401

def test_receive_log_invalid_key():
    """잘못된 API Key로 요청 시 403 반환 확인"""
    payload = {
        "device_id": "jetson-01",
        "timestamp": "2024-06-01T12:00:00",
        "loss_value": 0.82,
        "risk_level": 3
    }
    response = client.post(
        "/log/detection",
        json=payload,
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 403