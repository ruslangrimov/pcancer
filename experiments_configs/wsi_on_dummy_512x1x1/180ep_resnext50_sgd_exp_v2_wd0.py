import os, sys
sys.path.append(os.path.dirname(__file__))
import _default as d

from lib.datasets import (max_lbl_nums)

d.epochs = 180
d.warmup_steps = 0

hparams = {
    'module': {
        'name': 'lib.trainers.WSIModuleV1',
        'params': {
            'model': {
                'name': 'lib.models.wsi_resnets.Resnet_512x1x1',
                'params': {
                    'backbone': 'resnext50_32x4d',
                    'backbone_features': 2048,
                    'classes': max_lbl_nums,
                    'features_do': 0,
                }
            },
        },
    },
    'epochs': d.epochs,
    'learning_rate': 0.04 * d.batch_size / 256,
    'optimizer': {
        'name': 'torch.optim.SGD',
        'params': {
            'momentum': 0.9,
            'weight_decay': 0
        }
    },
    'scheduler': {
        'name': 'lib.schedulers.ExponentialLRWithMin',
        'params': {
            'gamma': 0.97,
            'eta_min': 3e-5
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
