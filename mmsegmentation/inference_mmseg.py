import os
import pandas as pd
import numpy as np
import json

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel


mdl = "UperNet_SwinL"
cfg = Config.fromfile(f'./mmsegmentation/configs/_trash_/_base_/models/{mdl}.py') # load config file
root = '../mmseg/test/'

# dataset config 수정
cfg.work_dir = f'./mmsegmentation/work_dirs/{mdl}' # set work_dir
cfg.data.test.img_dir = root
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.test.test_mode = True
cfg.data.samples_per_gpu = 1
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'latest.pth')

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, samples_per_gpu=cfg.data.samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader)

# sample_submisson.csv 열기
submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
json_dir = os.path.join("/opt/ml/input/data/test.json")

with open(json_dir, "r", encoding="utf-8") as outfile:
    data = json.load(outfile)

input_size = 512
output_size = 256
bin_size = input_size // output_size

# PredictionString 대입
for image_id, predict in enumerate(output):
    image_id = data["images"][image_id]
    file_name = image_id["file_name"]

    temp_mask = []
    predict = predict.reshape(1, 512, 512)
    # resize predict to 256, 256
    # reference : https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    mask = predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2)
    temp_mask.append(mask)
    oms = np.array(temp_mask)
    oms = oms.reshape([oms.shape[0], output_size * output_size]).astype(int)

    string = oms.flatten()

    submission = submission.append(
        {"image_id": file_name, "PredictionString": ' '.join(str(c) for c in string.tolist())}, ignore_index=True)

# submission.csv로 저장
# submission.to_csv(os.path.join('/opt/ml/input/code/submission', f'{mdl}_submission.csv'), index=False)
submission.to_csv(os.path.join('/opt/ml/input/code/submission', f'UperNet_SwinL_submission.csv'), index=False)