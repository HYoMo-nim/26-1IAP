# Jetson Nano에서 실행되는 클라이언트 코드.
# CSI 카메라로 영상을 읽고 YOLO26n-pose로 2D keypoints(confidence 포함)를 추출하여
# 243프레임씩 묶어 AWS 서버로 HTTPS 전송한다.

import cv2
import requests
import urllib3
from datetime import datetime
from ultralytics import YOLO

# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────
SERVER_URL = "https://3.34.13.29:8443/skeleton/keypoints"
API_KEY    = "jetson-01-secret-key"
DEVICE_ID  = "jetson-01"

NUM_FRAMES    = 243  # VideoPose3D receptive field
NUM_KEYPOINTS = 17   # YOLO26n-pose 관절 수

# 인증서 검증 비활성화 (테스트용)
# 실제 운영 시 아래 두 줄을 주석 처리하고
# requests.post의 verify=False를 verify="/path/to/cert.pem" 으로 변경
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY = False

# ───────────────────────────────────────────
# CSI 카메라 설정 (Jetson Nano)
# ───────────────────────────────────────────
def get_camera():
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        # CSI 카메라 연결 실패 시 USB 카메라로 폴백
        print("[경고] CSI 카메라 연결 실패, USB 카메라로 전환합니다.")
        cap = cv2.VideoCapture(0)
    return cap

# ───────────────────────────────────────────
# keypoints 추출 (confidence 포함)
# ───────────────────────────────────────────
def extract_keypoints(results):
    """
    YOLO26n-pose 결과에서 첫 번째 사람의 keypoints를 추출합니다.
    confidence 포함 [x, y, conf] 형식으로 반환합니다.
    사람이 감지되지 않으면 None을 반환합니다.
    """
    if results[0].keypoints is None:
        return None

    kp_xy   = results[0].keypoints.xy    # shape: [사람 수, 17, 2]
    kp_conf = results[0].keypoints.conf  # shape: [사람 수, 17]

    if len(kp_xy) == 0:
        return None

    # 첫 번째 사람의 keypoints만 사용
    xy   = kp_xy[0].tolist()    # [[x1,y1], [x2,y2], ...]
    conf = kp_conf[0].tolist()  # [c1, c2, ...]

    # [x, y, confidence] 형식으로 합치기
    keypoints = [[x, y, c] for (x, y), c in zip(xy, conf)]

    if len(keypoints) != NUM_KEYPOINTS:
        return None

    return keypoints

# ───────────────────────────────────────────
# 서버 전송
# ───────────────────────────────────────────
def send_keypoints(frames: list):
    payload = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now().isoformat(),
        "frames": [
            {"frame_id": i + 1, "keypoints": kp}
            for i, kp in enumerate(frames)
        ]
    }

    try:
        response = requests.post(
            SERVER_URL,
            json=payload,
            headers={"X-API-Key": API_KEY},
            verify=VERIFY,  # 운영 시: verify="/path/to/cert.pem"
            timeout=10      # 243프레임이라 데이터가 크므로 timeout 여유있게 설정
        )
        print(f"[전송 성공] 상태코드: {response.status_code} | 응답: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[전송 실패] {e}")

# ───────────────────────────────────────────
# 메인 루프
# ───────────────────────────────────────────
def main():
    print("[시작] YOLO26n-pose 모델 로드 중...")
    model = YOLO("yolo26n-pose.pt")

    print("[시작] 카메라 연결 중...")
    cap = get_camera()

    frames = []
    print(f"[시작] 영상 수집 시작 (목표: {NUM_FRAMES}프레임)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[오류] 프레임 읽기 실패")
            break

        # YOLO26n-pose로 keypoints 추출
        results = model(frame, verbose=False)
        keypoints = extract_keypoints(results)

        if keypoints is None:
            print("[스킵] 사람이 감지되지 않았습니다.")
            frames = []  # 사람 미감지 시 프레임 초기화
            continue

        frames.append(keypoints)
        print(f"[수집] {len(frames)}/{NUM_FRAMES} 프레임")

        # 243프레임 모이면 서버로 전송
        if len(frames) == NUM_FRAMES:
            print("[전송] 서버로 keypoints 전송 중...")
            send_keypoints(frames)
            frames = []  # 전송 후 초기화

    cap.release()

if __name__ == "__main__":
    main()
