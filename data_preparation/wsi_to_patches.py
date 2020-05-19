import sys
import os

import logging

from skimage.io import imread, imsave
import numpy as np
import pandas as pd

import shutil
import warnings
import tempfile
import subprocess

from multiprocessing import Pool

from tqdm.auto import tqdm

import tifffile

sys.path.append('..')
from lib.utils import init_script, array2patches, get_exception
from lib.datasets import (wsi_csv_path, wsi_masks_path, wsi_path,
                          patches_path, patches_csv_path, patch_sz)


init_script(f"{__file__}.log")

TIFFTILES2JPG_BIN = '../../tiff/tifftiles2jpg/tifftiles2jpg'


have_masks = True

threads = 6


def process_wsi(df_row):
    rows = []
    img_id = df_row.image_id

    try:
        with tempfile.TemporaryDirectory() as tmp_path:
            result = subprocess.run([TIFFTILES2JPG_BIN,
                                     os.path.join(wsi_path, f"{img_id}.tiff"),
                                     '0', tmp_path],
                                    stdout=subprocess.PIPE)

            if result.returncode != 0:
                raise Exception(f"TIFFTILES2JPG error." +
                                f" Return code: {result.returncode}. \n" +
                                result.stdout.decode("utf-8"))

            if have_masks:
                img = tifffile.imread(os.path.join(wsi_masks_path,
                                                   f"{img_id}_mask.tiff"))
                mask_patches = array2patches(img, patch_sz)

            for y in range(mask_patches.shape[0]):
                for x in range(mask_patches.shape[1]):
                    t_patch_path = os.path.join(tmp_path, f"{y}_{x}.jpeg")
                    patch = imread(t_patch_path)
                    mask_patch = mask_patches[y, x, ..., 0]

                    !!!
                    # ToDo: add checking background percentage from notebook here

                    if patch.mean() != 255:
                        row = [img_id, df_row.data_provider, df_row.isup_grade,
                               df_row.gleason_score, y, x]

                        mask_sz = np.prod(mask_patch.shape)

                        row += [(mask_patch == i).sum() / mask_sz
                                for i in range(6)]

                        i_fname = os.path.join(patches_path,
                                               f"imgs/{img_id}_{y}_{x}.jpeg")
                        m_fname = os.path.join(patches_path,
                                               f"masks/{img_id}_{y}_{x}.png")

                        shutil.copyfile(t_patch_path, i_fname)

                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore",
                                                    message=".*low contrast image.*")
                            imsave(m_fname, mask_patch)

                        rows.append(row)

        err = None

    except:
        err = get_exception()

    return img_id, rows, err


logging.info("Started")

for d in ["imgs", "masks"]:
    os.makedirs(os.path.join(patches_path, d), exist_ok=True)

wsi_df = pd.read_csv(wsi_csv_path)

patches_rows = []
error_wsis = []

with Pool(processes=threads) as pool:
    for img_id, rows, err in tqdm(pool.imap_unordered(process_wsi,
                                                      (r[1] for r in
                                                       wsi_df.iterrows())),
                                  total=len(wsi_df)):
        if err:
            error_wsis.append({img_id: err})
        else:
            patches_rows.extend(rows)

if len(error_wsis) > 0:
    logging.error("Slides processed with errors %s", str(error_wsis))

patches_df = pd.DataFrame(patches_rows,
                          columns=['image_id', 'data_provider',
                                   'isup_grade', 'gleason_score',
                                   'y', 'x'] + [f"label{i}"
                                                for i in range(6)])

patches_df.to_csv(patches_csv_path, index=False)

logging.info("Finished")
