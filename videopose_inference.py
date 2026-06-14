import torch
import numpy as np
import os

from model import TemporalModel

def run_videopose3d(frames_2d, model_weights_path="./models/pretrained_h36m_cpn.bin"):
    try:
        input_2d = np.array(frames_2d, dtype=np.float32)
        h36m_2d = np.zeros_like(input_2d)
        
        # 0: 골반 (좌우 엉덩이 11, 12의 중간)
        h36m_2d[:, 0] = (input_2d[:, 11] + input_2d[:, 12]) / 2.0
        # 1~3: 우측 다리 (고관절, 무릎, 발목)
        h36m_2d[:, 1] = input_2d[:, 12]
        h36m_2d[:, 2] = input_2d[:, 14]
        h36m_2d[:, 3] = input_2d[:, 16]
        # 4~6: 좌측 다리
        h36m_2d[:, 4] = input_2d[:, 11]
        h36m_2d[:, 5] = input_2d[:, 13]
        h36m_2d[:, 6] = input_2d[:, 15]
        # 8: 목 (좌우 어깨 5, 6의 중간)
        h36m_2d[:, 8] = (input_2d[:, 5] + input_2d[:, 6]) / 2.0
        # 7: 척추 (골반 0과 목 8의 중간)
        h36m_2d[:, 7] = (h36m_2d[:, 0] + h36m_2d[:, 8]) / 2.0
        # 9, 10: 머리 (코 위치로 통일)
        h36m_2d[:, 9] = input_2d[:, 0]
        h36m_2d[:, 10] = input_2d[:, 0]
        # 11~13: 좌측 팔 (어깨, 팔꿈치, 손목)
        h36m_2d[:, 11] = input_2d[:, 5]
        h36m_2d[:, 12] = input_2d[:, 7]
        h36m_2d[:, 13] = input_2d[:, 9]
        # 14~16: 우측 팔
        h36m_2d[:, 14] = input_2d[:, 6]
        h36m_2d[:, 15] = input_2d[:, 8]
        h36m_2d[:, 16] = input_2d[:, 10]

        # 번역된 뼈대로 교체
        input_2d = h36m_2d

        valid_x = input_2d[..., 0][input_2d[..., 0] > 0.1]
        valid_y = input_2d[..., 1][input_2d[..., 1] > 0.1]
        
        if len(valid_x) > 0 and len(valid_y) > 0:
            # 2. 사람이 움직인 가장 왼쪽/오른쪽/위/아래 경계선을 찾습니다.
            min_x, max_x = np.min(valid_x), np.max(valid_x)
            min_y, max_y = np.min(valid_y), np.max(valid_y)
            
            # 3. 사람의 중심점과, 화면을 꽉 채울 스케일(가장 긴 축 기준 + 20% 여백)을 계산합니다.
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            scale = max(max_x - min_x, max_y - min_y) * 1.2
            
            # 4. 모든 좌표를 화면 정중앙으로 옮기고, -1.0 ~ 1.0 사이의 완벽한 비율로 압축합니다.
            if scale > 0:
                input_2d[..., 0] = (input_2d[..., 0] - center_x) / (scale / 2.0)
                input_2d[..., 1] = (input_2d[..., 1] - center_y) / (scale / 2.0)
     
        input_tensor = torch.from_numpy(input_2d).unsqueeze(0)

        # --- [추가된 부분] VideoPose3D 프레임 손실 방지를 위한 패딩 ---
        # 모델의 수용 영역(Receptive Field)이 243이므로,
        # 양 끝에 121프레임씩 복사해서 붙여넣어 출력 3D 프레임 수를 243개로 유지합니다.
        pad = 121
        padded_tensor = torch.cat((
            input_tensor[:, :1, :, :].repeat(1, pad, 1, 1),
            input_tensor,
            input_tensor[:, -1:, :, :].repeat(1, pad, 1, 1)
        ), dim=1)
        # ---------------------------------------------------------------

        # 로그에서 확인한 정확한 하이퍼파라미터 적용
        model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=[3, 3, 3, 3, 3],
            causal=False,
            dropout=0.25,
            channels=1024
        )

        # 가중치 파일 입히기
        if os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(checkpoint['model_pos'])
        else:
            print(f"[오류] {model_weights_path} 파일이 없습니다!")
            return None

        model.eval()

        with torch.no_grad():
            # 원본 input_tensor 대신 패딩된 padded_tensor를 입력합니다.
            output_tensor = model(padded_tensor)

        # 패딩을 거쳤으므로 output_tensor의 형태는 이제 (1, 243, 17, 3)이 됩니다.
        output_3d = output_tensor.squeeze(0).cpu().numpy()
        return output_3d.tolist()

    except Exception as e:
        print(f"[VideoPose3D 오류] {e}")
        return None
