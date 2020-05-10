import pickle
import pandas as pd
from tqdm.auto import tqdm

sys.path.append('..')
from lib.datasets import patches_csv_path, patches_pkl_path

patches_csv_path = patches_csv_path
patches_pkl_path = patches_pkl_path

df = pd.read_csv(patches_csv_path)

image_ids = df['image_id'].unique().tolist()

rows_by_img_id = {}
for image_id in tqdm(image_ids):
    rows = []
    id_df = df[df['image_id'] == image_id]
    for row in id_df.itertuples():
        rows.append(tuple(row))
    rows_by_img_id[image_id] = rows

with open(patches_pkl_path, 'wb') as f:
    pickle.dump(rows_by_img_id, f)
