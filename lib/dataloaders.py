import os

import math
import random
import torch
import cv2
import numpy as np
import pandas as pd
import pickle
from turbojpeg import TurboJPEG

import multiprocessing.dummy as mp

from .datasets import (patches_path, patches_csv_path, patches_pkl_path,
                       max_lbl_nums, gleason_scores, actual_lbl_nums,
                       data_providers)


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

        labels = np.zeros(actual_lbl_nums, dtype=np.float32)
        for i in range(max_lbl_nums if row.data_provider == 'radboud' else 3):
            labels[get_actual_lbl_num(row.data_provider, i)] =\
                getattr(row, f'label{i}')  # row[f'label{i}']

        # mask.T.astype(np.int64)
        return (img.transpose([2, 0, 1]).astype(np.float32),
                0 if mask is None else mask.astype(np.int64),
                labels.astype(np.float32),
                get_provider_num(row.data_provider),
                int(row.isup_grade), get_g_score_num(row.gleason_score))


class WSIPatchesDatasetRaw(torch.utils.data.Dataset):
    def __init__(self, image_ids, pkl_path=patches_pkl_path,
                 path=patches_path, scale=1, transform=None):

        if scale != 0.5:
            raise NotImplementedError("Scale different than 0.5 not "
                                      "implemented yet")

        self.image_ids = image_ids

        with open(pkl_path, 'rb') as f:
            self.rows_by_img_id = pickle.load(f)

        self.path = path
        self.scale = scale
        self.transform = transform
        self.jpeg = TurboJPEG()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        imgs = []
        ys = []
        xs = []
        rows = self.rows_by_img_id[image_id]
        for row in rows:
            (_, image_id, data_provider, isup_grade, gleason_score, y, x,
             *_) = row
            fname = f"{image_id}_{y}_{x}"
            # img = imread(os.path.join(self.path, f"imgs/{fname}.jpeg"))
            # if self.scale != 1:
            #     img = cv2.resize(img, None, fx=self.scale, fy=self.scale)

            with open(os.path.join(self.path,
                                   f"imgs/{fname}.jpeg"), 'rb') as f:
                img = self.jpeg.decode(f.read(),
                                       scaling_factor=(1, 2))[..., ::-1]

            imgs.append(img)
            ys.append(y)
            xs.append(x)

        if self.transform:
            imgs = self.transform(imgs)

        ys = np.stack(ys)
        xs = np.stack(xs)
        imgs = np.stack(imgs)

        return (imgs.transpose([0, 3, 1, 2]).astype(np.float32),
                ys.astype(np.int64), xs.astype(np.int64),
                get_provider_num(data_provider),
                int(isup_grade), get_g_score_num(gleason_score))


class WSIPatchesDataloader():
    def __init__(self, dataset, get_features_fn, features_shape,
                 batch_size=1, shuffle=False, num_workers=0, max_len=300):
        self.dataset = dataset
        self.get_features_fn = get_features_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_len = max_len

        self.b_features = torch.zeros((self.batch_size, max_len,) +
                                      features_shape,
                                      dtype=torch.float32)
        self.b_ys = torch.zeros((self.batch_size, max_len), dtype=torch.int64)
        self.b_xs = torch.zeros((self.batch_size, max_len), dtype=torch.int64)
        self.b_provider = torch.zeros((self.batch_size), dtype=torch.int64)
        self.b_isup_grade = torch.zeros((self.batch_size), dtype=torch.int64)
        self.b_gleason_score = torch.zeros((self.batch_size),
                                           dtype=torch.int64)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        return self.produce_batches()

    def produce_batches(self):
        def process_item(idx):
            item_data = self.dataset[idx]
            return item_data

        def clean_batch():
            for a in batch:
                a.fill_(-1)

        idxs = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(idxs)

        max_len = self.max_len

        batch = [self.b_features, self.b_ys, self.b_xs, self.b_provider,
                 self.b_isup_grade, self.b_gleason_score]

        clean_batch()

        c_iter = 0
        with mp.Pool(processes=self.num_workers) as pool:
            for item_data in pool.imap_unordered(process_item, idxs):
                imgs, ys, xs, provider, isup_grade, gleason_score = item_data
                features = self.get_features_fn(imgs)

                b_iter = c_iter % self.batch_size
                p = ys.shape[0]

                self.b_features[b_iter, :p] = features[:max_len]
                self.b_ys[b_iter, :p] = torch.from_numpy(ys)[:max_len]
                self.b_xs[b_iter, :p] = torch.from_numpy(xs)[:max_len]
                self.b_provider[b_iter] = provider
                self.b_isup_grade[b_iter] = isup_grade
                self.b_gleason_score[b_iter] = gleason_score

                if (c_iter + 1) % self.batch_size == 0:
                    # process batch
                    yield batch

                    # clean batch data
                    clean_batch()

                c_iter += 1

        if c_iter % self.batch_size != 0:
            yield [a[:c_iter % self.batch_size] for a in batch]
