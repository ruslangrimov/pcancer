import sys
import os
import logging

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

sys.path.append('..')
from lib.dataloaders import PatchesDataset
from lib.datasets import (actual_lbl_nums, get_train_test_img_ids_split,
                          patches_rgb_mean_av1, patches_rgb_std_av1,
                          patches_csv_path, patches_clean90_csv_path)
from lib.trainers import PatchesModuleV1
from lib.augmentations import augment_empty, augment_v1
from lib.utils import init_script

from lib.models.unetv1 import get_model


init_script(f"{__file__}.log")
logging.info("Started")

steps_in_epoh = 1

epochs = 3
# warmup_epochs = 1
warmup_epochs = 0
warmup_steps = 3000
batch_size = 60
hparams = {
    'batch_size': batch_size,
    'learning_rate': 0.1 * batch_size / 256,
    'dataset': {
        'scale': 0.5,
        'csv_path': patches_clean90_csv_path
    },
    'optimizer': {
        'name': 'SGD',
        'params': {
            'momentum': 0.9,
            'weight_decay': 1e-4
        }
    },
    'scheduler': {
        'name': 'CosineAnnealingLR',
        'params': {
            # 'T_max': (epochs-warmup_epochs) * steps_in_epoh
            'T_max': epochs * steps_in_epoh - warmup_steps
        },
        'interval': 'step'
    },
    'loss': {
        'weights': {
            'dec': 100,
            'mask': 5,
            'label': 0.3
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

num_workers = 12

train_img_ids, test_img_ids = get_train_test_img_ids_split()

# import random
# train_img_ids = random.sample(train_img_ids, 50)
# test_img_ids = random.sample(test_img_ids, 10)

train_loader = torch.utils.data.DataLoader(
    PatchesDataset(train_img_ids,
                   csv_path=hparams['dataset']['csv_path'],
                   transform=augment_v1,
                   scale=hparams['dataset']['scale'],
                   load_masks=True),
    batch_size=hparams['batch_size'], shuffle=True,
    num_workers=num_workers, pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    PatchesDataset(test_img_ids,
                   csv_path=hparams['dataset']['csv_path'],
                   transform=augment_empty,
                   scale=hparams['dataset']['scale'],
                   load_masks=True),
    batch_size=hparams['batch_size'], shuffle=False,
    num_workers=num_workers, pin_memory=True,
)

steps_in_epoh = len(train_loader)
print("steps_in_epoh", steps_in_epoh)
# Update with actual value
hparams['steps_in_batch'] = steps_in_epoh
hparams['scheduler']['params']['T_max'] = epochs * steps_in_epoh - warmup_steps

model = get_model(actual_lbl_nums)
module = PatchesModuleV1(model, hparams, log_train_every_batch=True)
module.rgb_mean, module.rgb_std = (torch.tensor(patches_rgb_mean_av1,
                                                dtype=torch.float32),
                                   torch.tensor(patches_rgb_std_av1,
                                                dtype=torch.float32))

logger = TensorBoardLogger(
    os.getcwd(), 'Patches256TestRun',
)
# accumulate_grad_batches=1
trainer = Trainer(logger, max_epochs=epochs, gpus=[0, ], fast_dev_run=False)

trainer.fit(module, train_loader, val_loader)
trainer.save_checkpoint(os.path.join(trainer.checkpoint_callback.dirpath,
                                     "last.ckpt"))

logging.info("Finished")
