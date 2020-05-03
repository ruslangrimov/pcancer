import os

import torch
import cv2
import numpy as np
import pandas as pd

from .datasets import (patches_path, patches_csv_path, max_lbl_nums,
                       gleason_scores, actual_lbl_nums, data_providers)


def get_actual_lbl_num(data_provider, lbl):
    return lbl if data_provider == 'radboud' else lbl + max_lbl_nums


def get_provider_num(data_provider):
    return data_providers[data_provider]


def get_g_score_num(score):
    return gleason_scores[score]


def imread(fname, channels_first=False):
    img = cv2.imread(fname)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        if os.path.isfile(fname):
            raise e
        raise Exception('file not found {}'.format(fname))
    if channels_first:
        img = img.transpose([2, 0, 1])

    return img


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, csv_path=patches_csv_path,
                 path=patches_path, scale=1, transform=None,
                 load_masks=True):
        df = pd.read_csv(csv_path)
        self.df = df[df['image_id'].isin(image_ids)]
        self.path = path
        self.scale = scale
        self.transform = transform
        self.load_masks = load_masks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id, y, x = row.image_id, row.y, row.x
        fname = f"{image_id}_{y}_{x}"
        img = imread(os.path.join(self.path, f"imgs/{fname}.jpeg"))

        if self.load_masks:
            r_mask = imread(os.path.join(self.path,
                                         f"masks/{fname}.png"))[..., 0]
            mask = np.zeros(r_mask.shape, dtype=np.uint8)
            for i in range(max_lbl_nums):
                mask[r_mask == i] = get_actual_lbl_num(row.data_provider, i)
        else:
            mask = None

        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
            if mask is not None:
                mask = cv2.resize(mask, None, fx=self.scale, fy=self.scale)

        if self.transform:
            img, mask = self.transform(img, mask)

        labes = np.zeros(actual_lbl_nums, dtype=np.float32)
        for i in range(max_lbl_nums if row.data_provider == 'radboud' else 3):
            labes[get_actual_lbl_num(row.data_provider, i)] =\
                getattr(row, f'label{i}')  # row[f'label{i}']

        # mask.T.astype(np.int64)
        return (img.transpose([2, 0, 1]).astype(np.float32),
                0 if mask is None else mask.astype(np.int64),
                labes.astype(np.float32),
                get_provider_num(row.data_provider),
                int(row.isup_grade), get_g_score_num(row.gleason_score))
