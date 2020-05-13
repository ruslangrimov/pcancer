from lib.datasets import (patches_rgb_mean_av1, patches_rgb_std_av1,
                          max_lbl_nums)

steps_in_epoh = 1

epochs = 90

warmup_epochs = 0
warmup_steps = 0
batch_size = 64

hparams = {
    'module': {
        'name': 'lib.trainers.WSIModuleV1',
        'params': {
            'model': {
                'name': 'lib.models.wsi_resnets.Resnet_512x1x1',
                'params': {
                    'backbone': 'resnet18',
                    'backbone_features': 512,
                    'classes': max_lbl_nums,
                    'features_do': 0,
                }
            },
        },
    },
    'batch_size': batch_size,
    'learning_rate': 0.001 * batch_size / 256,
    'dataset': {
        'dataloader': 'dummy',
        'rgb_mean': patches_rgb_mean_av1,
        'rgb_std': patches_rgb_std_av1,
        'classes': max_lbl_nums,
        'precalc_epochs': 50,
        'train_test_split': {},
    },
    'optimizer': {
        'name': 'torch.optim.Adam',
        'params': {
            'weight_decay': 0
        }
    },
    'scheduler': {
        'name': 'lib.schedulers.ExponentialLRWithMin',
        'params': {
            'gamma': 0.92,
            'eta_min': 1.25e-5
        },
        'interval': 'epoch'
    },
    'loss': {
        'weights': {
            'reg': 1 / 2,
            'class': 9 / 2
        },
        'label_smoothing': 0.1
    },
    'warmup_steps': warmup_steps,
    'steps_in_epoh': steps_in_epoh,
    'epochs': epochs,
    'source_code': open(__file__, 'rt').read()
}


def get_hrapams():
    return hparams


def update_hrapams(hparams, steps_in_epoh):
    hparams['steps_in_batch'] = steps_in_epoh
    if 'T_max' in hparams['scheduler']['params']:
        hparams['scheduler']['params']['T_max'] = (epochs * steps_in_epoh -
                                                   warmup_steps)
    return hparams
