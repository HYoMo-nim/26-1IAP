# 0. 공통 사항
    - 데이터셋: human36m
    - 사전학습된 모델을 git clone하여 evaluate 만 진행

# 1. VideoPose3D

Protocol #1   (MPJPE) action-wise average: 46.8 mm
Protocol #2 (P-MPJPE) action-wise average: 36.5 mm
Protocol #3 (N-MPJPE) action-wise average: 45.0 mm
Velocity      (MPJVE) action-wise average: 2.79 mm

[result_videopose3d.txt]

# 2. PoseFormerV2

- 모델 종류
    
    ```markdown
    | Model        | Sequence Leng. |  f   |  n   | #Depth | Hidden Dim. | #MFLOPs | MPJPE (mm) |                           Download                           |
    | :----------- | :------------: | :--: | :--: | :----: | :---------: | :-----: | :--------: | :----------------------------------------------------------: |
    | PoseFormerV2 |       27       |  1   |  3   |   4    |     32      |  77.2   |    48.7    | [model](https://drive.google.com/file/d/14J0GYIzk_rGKSMxAPI2ydzX76QB70-g3/view?usp=share_link) |
    | /            |       27       |  3   |  3   |   4    |     32      |  117.3  |    47.9    | [model](https://drive.google.com/file/d/13oJz5-aBVvvPVFvTU_PrLG_m6kdbQkYs/view?usp=share_link) |
    | /            |       81       |  1   |  3   |   4    |     32      |  77.2   |    47.6    | [model](https://drive.google.com/file/d/14WgFFBsP0DtTq61XZWI9X2TzvFLCWEnd/view?usp=share_link) |
    | /            |       81       |  3   |  3   |   4    |     32      |  117.3  |    47.1    | [model](https://drive.google.com/file/d/13rXCkYnVnkbT-cz4XCo0QkUnUEYiSeoi/view?usp=share_link) |
    | /            |       81       |  9   |  9   |   4    |     32      |  351.7  |    46.0    | [model](https://drive.google.com/file/d/13wla4b5RgJGKX5zVehv4qKhCrQEFhfzG/view?usp=share_link) |
    | /            |      243       |  27  |  27  |   4    |     32      | 1054.8  |    45.2    | [model](https://drive.google.com/file/d/14SpqPyq9yiblCzTH5CorymKCUsXapmkg/view?usp=share_link) |
    ```
    
- 첨부된 모델 중 두번째로 선택

Protocol #1   (MPJPE) action-wise average: 47.9 mm
Protocol #2 (P-MPJPE) action-wise average: 37.4 mm
Protocol #3 (N-MPJPE) action-wise average: 46.4 mm
Velocity      (MPJVE) action-wise average: 2.72 mm

[poseformerv2_result.txt]

# 3. 정리
| Model        | Input Setting                                  | MPJPE ↓     | P-MPJPE ↓ | N-MPJPE ↓ | MPJVE ↓ | 특징                              |
| ------------ | ---------------------------------------------- | ----------- | --------- | --------- | ------- | ------------------------------- |
| VideoPose3D  | Human3.6M + CPN 2D + 243-frame receptive field | **46.8 mm** | 36.5 mm   | 45.0 mm   | 2.79 mm | Temporal CNN 기반, 초고속 baseline   |
| PoseFormerV2 | Human3.6M + CPN 2D + 27-frame setting          | **47.9 mm** | 37.4 mm   | 46.4 mm   | 2.72 mm | Frequency-domain Transformer 기반 |

# 4. 결론
| 항목                  | 우세 모델              |
| ------------------- | ------------------ |
| 정확도(MPJPE)          | VideoPose3D        |
| Temporal 안정성(MPJVE) | PoseFormerV2       |
| 실시간성(FPS)           | VideoPose3D        |
| 구조 복잡도              | PoseFormerV2가 더 복잡 |

=> VideoPose3D 모델로 선택