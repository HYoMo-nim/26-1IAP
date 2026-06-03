from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

# 요청 헤더에서 X-API-Key 값을 추출
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# 등록된 디바이스별 API Key 목록
VALID_API_KEYS: dict[str, str] = {
    os.getenv("DEVICE_KEY_JETSON01", "jetson-01-secret-key"): "jetson-01",
    os.getenv("DEVICE_KEY_JETSON02", "jetson-02-secret-key"): "jetson-02",
}

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key가 없습니다. 헤더에 X-API-Key를 포함해주세요.",
        )
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="유효하지 않은 API Key입니다.",
        )
    return VALID_API_KEYS[api_key]