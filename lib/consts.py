wsi_path = "/mnt/HDDData/pdata/train_images/"
wsi_masks_path = "/mnt/HDDData/pdata/train_label_masks/"
wsi_csv_path = "/mnt/HDDData/pdata/train.csv"
patches_path = "/mnt/SSDData/pdata/processed/patches_512/"
patches_csv_path = "/mnt/SSDData/pdata/processed/patches_512.csv"

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

gleason_score_nums = len(gleason_scores) - 1
