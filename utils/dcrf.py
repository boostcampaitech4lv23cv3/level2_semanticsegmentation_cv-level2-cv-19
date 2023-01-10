import numpy as np
import pandas as pd
import os
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from datetime import datetime, timedelta


DIR_PATH = "/opt/ml/input/code/submission"
FILE_NAME = "UperNet_SwinL_submission"
FILE_PATH = os.path.join(DIR_PATH, FILE_NAME)
DATASET_PATH = "/opt/ml/input/data"


def decode(prediction_string):
    return np.array(list(map(int, prediction_string.split()))).reshape(256, 256)


def encode(img):
    return ' '.join(str(x) for x in img.flatten())


def crf(original_img, mask_img):
    # Original_image = Image which has to labelled
    # Mask image = Which has been labelled by some technique
    labels = mask_img.flatten()
    n_labels = 11

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(256, 256, n_labels)

    # get unary potentials (neg log probability)
    u = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(u)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=30, srgb=5, rgbim=original_img, compat=5)

    # Run Inference for 10 steps
    q = d.inference(45)

    # Find out the most probable class for each pixel.
    m = np.argmax(q, axis=0)

    return m.reshape((256, 256))


print(f":::PROCESS COMMENCING::: {datetime.strftime(datetime.now() + timedelta(hours=9), '%Y-%m-%d %H:%M:%S')}")
df = pd.read_csv(f'{FILE_PATH}.csv')

for i in tqdm(range(df.shape[0])):
    if str(df.loc[i, 'PredictionString']) != str(np.nan):
        decoded_mask = decode(df.loc[i, 'PredictionString'])

        orig_img = imread(os.path.join(DATASET_PATH, df.loc[i, 'image_id']))
        orig_img = resize(orig_img, (256, 256, 3))
        orig_img = np.uint8(255 * orig_img)
        crf_output = crf(orig_img, decoded_mask)

        df.loc[i, 'PredictionString'] = encode(crf_output)

df.to_csv(f"{FILE_PATH}_crf.csv", index=False)
print(f"Result saved at [{FILE_PATH}_crf.csv]")
print(f":::PROCESS COMPLETED::: {datetime.strftime(datetime.now() + timedelta(hours=9), '%Y-%m-%d %H:%M:%S')}")
