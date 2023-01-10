# MMSegmentation을 이용한 UPerNet + SwinL 모델 사용

## 사용 방법
1. MMSegmentation Official Documentation에 나온 Instruction에 따라 라이브러리 설치
2. MMSegmentation을 사용할 수 있도록 Dataset Structure 변경 ([참고자료](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-04/tree/main/mmsegmentation#change-dataset-format))
3. 기학습 가중치 다운로드
   ```shell
    python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth pretrained/swin_large_patch4_window12_384_22k.pth
   ```
4. 학습 진행
    ```shell
    python tools/train.py configs/_trash_/_base_/models/UperNet_SwinL.py --seed 42
    ```
5. 추론 진행
    ```shell
    python inference_mmseg.py
    ```
