import torch
import numpy as np
import os

# 방금 다운받은 model.py에서 TemporalModel 클래스를 불러옵니다.
from model import TemporalModel 

def run_videopose3d(frames_2d, model_weights_path="./models/pretrained_h36m_cpn.bin"):
    try:
        input_2d = np.array(frames_2d, dtype=np.float32)
        input_tensor = torch.from_numpy(input_2d).unsqueeze(0)
        
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
            model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            print(f"[오류] {model_weights_path} 파일이 없습니다!")
            return None
            
        model.eval()
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        output_3d = output_tensor.squeeze(0).cpu().numpy()
        return output_3d.tolist()

    except Exception as e:
        print(f"[VideoPose3D 오류] {e}")
        return None
