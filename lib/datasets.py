import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

wsi_path = "/mnt/HDDData/pdata/train_images/"
wsi_masks_path = "/mnt/HDDData/pdata/train_label_masks/"
wsi_csv_path = "/mnt/HDDData/pdata/train.csv"

patches_path = "/mnt/SSDData/pdata/processed/patches_512/"
patches_csv_path = "/mnt/SSDData/pdata/processed/patches_512.csv"
patches_clean90_csv_path = "/mnt/SSDData/pdata/processed/patches_clean90_512.csv"

patches_pkl_path = "/mnt/SSDData/pdata/processed/patches_512.pkl"
patches_clean90_pkl_path = "/mnt/SSDData/pdata/processed/patches_clean90_512.pkl"

# Here is the list of tiff files which are probably corrupt
wrong_img_ids_path = '/mnt/SSDData/pdata/processed/wrong_img_ids.pkl'

patch_sz = 512

max_lbl_nums = 6
actual_lbl_nums = 6 + 3
provider_nums = 2
isup_grade_nums = 6

gleason_scores = {
    '0+0': 0,
    'negative': 0,
    '3+3': 1,
    '3+4': 2,
    '4+3': 3,
    '4+4': 4,
    '5+3': 5,
    '3+5': 6,
    '4+5': 7,
    '5+4': 8,
    '5+5': 9
}

data_providers = {
    'radboud': 0,
    'karolinska': 1
}

gleason_score_nums = len(gleason_scores) - 1

patches_rgb_mean_av1 = np.array([0.88, 0.76, 0.84])[None, :, None, None]
patches_rgb_std_av1 = np.array([0.15, 0.26, 0.18])[None, :, None, None]


def get_train_test_img_ids_split(test_size=0.2, random_state=42):
    with open(wrong_img_ids_path, 'rb') as f:
        wrong_img_ids = pickle.load(f)

    df = pd.read_csv(wsi_csv_path)
    gr_by_id_df = df.groupby('image_id').count()['data_provider']
    img_ids = gr_by_id_df.index.values.tolist()

    img_ids.sort()
    train_img_ids, test_img_ids = train_test_split(img_ids,
                                                   test_size=test_size,
                                                   random_state=random_state)

    train_img_ids = [a for a in train_img_ids if a not in wrong_img_ids]
    test_img_ids = [a for a in test_img_ids if a not in wrong_img_ids]

    return train_img_ids, test_img_ids


def get_kfolds(train_img_ids, n_splits=4, random_state=42):
    train_img_ids = sorted(train_img_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return kf.split(train_img_ids)
