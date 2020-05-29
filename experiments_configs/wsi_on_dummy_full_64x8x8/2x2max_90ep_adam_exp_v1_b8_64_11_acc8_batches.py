from lib.datasets import (patches_rgb_mean_av1, patches_rgb_std_av1,
                          max_lbl_nums)

steps_in_epoh = 1

epochs = 90
warmup_steps = 0
batch_size = 8

hparams = {
    'module': {
        'name': 'lib.trainers.WSIModuleV1',
        'params': {
            'model': {
                'name': 'lib.models.wsi_resnets.Resnet_64x8x8',
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
    'epochs': epochs,
    'learning_rate': 0.001 * 64 / 256,
    'accumulate_grad_batches': 8,
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
        'precalc_epochs': 11,
        'train_test_split': {},
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
    'loss': {
        'weights': {
            'reg': 1 / 2,
            'class': 9 / 2
        },
        'label_smoothing': 0.1
    },
    'source_code': open(__file__, 'rt').read()
}


def get_hrapams():
    return hparams


def update_hrapams(hparams, steps_in_epoh):
    hparams['steps_in_epoh'] = steps_in_epoh
    if 'T_max' in hparams['scheduler']['params']:
        hparams['scheduler']['params']['T_max'] = (epochs * steps_in_epoh -
                                                   warmup_steps)
    return hparams
