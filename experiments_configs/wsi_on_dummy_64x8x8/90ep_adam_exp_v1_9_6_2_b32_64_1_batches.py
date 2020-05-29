import os, sys
sys.path.append(os.path.dirname(__file__))
import _default as d

from lib.utils import update_r

d.epochs = 90
d.warmup_steps = 0
d.batch_size = 32

hparams = {
   'batch_size': d.batch_size,
    'epochs': d.epochs,
    'learning_rate': 0.001 * 64 / 256,
    'optimizer': {
        'name': 'torch.optim.Adam',
        'params': {
            'weight_decay': 1e-4
        }
    },
    'dataset': {
        'precalc_epochs': 11,
        'train_batch_path': '/mnt/SSDData/pdata/processed/pretrained_64x8x8/train/{}/',
        'test_batch_path': '/mnt/SSDData/pdata/processed/pretrained_64x8x8/val/',
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

d.hparams = update_r(d.hparams, hparams)


def get_hrapams():
    return d.get_hrapams()


def update_hrapams(hparams, steps_in_epoh):
    return d.update_hrapams(hparams, steps_in_epoh)
