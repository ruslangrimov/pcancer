import os, sys
sys.path.append(os.path.dirname(__file__))
import _default as d

from lib.datasets import (patches_rgb_mean_av1, patches_rgb_std_av1,
                          max_lbl_nums)

d.epochs = 90
d.warmup_steps = 0

hparams = {
    'epochs': d.epochs,
    'learning_rate': 0.001 * d.batch_size / 256,
    'optimizer': {
        'name': 'torch.optim.Adam',
        'params': {
            'weight_decay': 1e-4
        }
    },
    'dataset': {
        'dataloader': 'dummy',
        'rgb_mean': patches_rgb_mean_av1,
        'rgb_std': patches_rgb_std_av1,
        'classes': max_lbl_nums,
        'precalc_epochs': 1,
        'train_test_split': {},
        'train_batch_path': '/mnt/HDDData/pdata/processed/pretrained_64x1x1/train/{}/',
        'test_batch_path': '/mnt/HDDData/pdata/processed/pretrained_64x1x1/val/',
    },
    'scheduler': {
        'name': 'torch.optim.lr_scheduler.ExponentialLR',
        'params': {
            'gamma': 0.96,
        },
        'interval': 'epoch'
    },
    'source_code': open(__file__, 'rt').read()
}

d.hparams.update(hparams)


def get_hrapams():
    return d.get_hrapams()


def update_hrapams(hparams, steps_in_epoh):
    return d.update_hrapams(hparams, steps_in_epoh)
