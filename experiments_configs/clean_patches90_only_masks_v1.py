from lib.datasets import (actual_lbl_nums, patches_clean90_csv_path,
                          patches_rgb_mean_av1, patches_rgb_std_av1)

steps_in_epoh = 1

epochs = 3
# warmup_epochs = 1
warmup_epochs = 0
warmup_steps = 3000
batch_size = 60
hparams = {
    'module': {
        'name': 'PatchesModuleV1',
        'params': {
            'model': {
                'name': 'lib.models.unetv1.get_model',
                'params': {
                    'classes': actual_lbl_nums,
                    'decoder': False,
                    'labels': False
                }
            },
        },
    },
    'batch_size': batch_size,
    'learning_rate': 0.1 * batch_size / 256,
    'dataset': {
        'scale': 0.5,
        'csv_path': patches_clean90_csv_path,
        'rgb_mean': patches_rgb_mean_av1,
        'rgb_std': patches_rgb_std_av1,
        'train_test_split': {},  # default
        'train_augmetations': 'lib.augmentations.augment_v1',
        'test_augmetations': 'lib.augmentations.augment_empty',
    },
    'optimizer': {
        'name': 'torch.optim.SGD',
        'params': {
            'momentum': 0.9,
            'weight_decay': 1e-4
        }
    },
    'scheduler': {
        'name': 'torch.optim.import.lr_scheduler.CosineAnnealingLR',
        'params': {
            # 'T_max': (epochs-warmup_epochs) * steps_in_epoh
            'T_max': epochs * steps_in_epoh - warmup_steps
        },
        'interval': 'step'
    },
    'loss': {
        'weights': {
            'dec': 0,
            'mask': 10,
            'label': 0
        },
        'mask_smoothing': 0.1,
        'label_smoothing': 0.1
    },
    # 'warmup_epochs': warmup_epochs,
    'warmup_steps': warmup_steps,
    'steps_in_epoh': steps_in_epoh,
    'epochs': epochs,
    'source_code': open(__file__, 'rt').read()
}


def get_hrapams():
    return hparams


def update_hrapams(hparams, steps_in_epoh):
    hparams['steps_in_batch'] = steps_in_epoh
    hparams['scheduler']['params']['T_max'] = (epochs * steps_in_epoh -
                                               warmup_steps)
    return hparams
