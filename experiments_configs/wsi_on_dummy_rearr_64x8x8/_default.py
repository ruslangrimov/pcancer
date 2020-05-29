from lib.datasets import (patches_rgb_mean_av1, patches_rgb_std_av1,
                          max_lbl_nums)

steps_in_epoh = 1

epochs = -1

warmup_epochs = 0
warmup_steps = -1
batch_size = 64

hparams = {
    'module': {
        'name': 'lib.trainers.WSIModuleV1',
        'params': {
            'model': {
                'name': 'lib.models.wsi_resnets.RearrangedResnet_64x8x8',
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
    'dataset': {
        'dataloader': 'dummy',
        'rgb_mean': patches_rgb_mean_av1,
        'rgb_std': patches_rgb_std_av1,
        'classes': max_lbl_nums,
        'precalc_epochs': 50,
        'train_test_split': {},
        'train_batch_path': '/mnt/HDDData/pdata/processed/pretrained_64x1x1/train/{}/',
        'test_batch_path': '/mnt/HDDData/pdata/processed/pretrained_64x1x1/val/',
    },
    'optimizer': {
        'name': 'torch.optim.Adam',
        'params': {
            'weight_decay': 1e-4
        }
    },
    'loss': {
        'weights': {
            'reg': 1 / 2,
            'class': 9 / 2
        },
        'label_smoothing': 0.1
    }
}


def get_hrapams():
    return hparams


def update_hrapams(hparams, steps_in_epoh):
    hparams['steps_in_epoh'] = steps_in_epoh
    if 'T_max' in hparams['scheduler']['params']:
        hparams['scheduler']['params']['T_max'] = (epochs * steps_in_epoh -
                                                   warmup_steps)
    return hparams
