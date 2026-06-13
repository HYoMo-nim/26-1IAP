import torch
import numpy as np
import os

from model import TemporalModel

def run_videopose3d(frames_2d, model_weights_path="./models/pretrained_h36m_cpn.bin"):
    try:
        input_2d = np.array(frames_2d, dtype=np.float32)
        input_2d[..., 0] = (input_2d[..., 0] / w) * 2 - 1
        input_2d[..., 1] = (input_2d[..., 1] / h) * 2 - 1
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
