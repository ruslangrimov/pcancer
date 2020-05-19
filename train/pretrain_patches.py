import os
import argparse
import importlib
import copy

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from lib.dataloaders import PatchesDataset
from lib.datasets import get_train_test_img_ids_split
from lib.utils import init_script, call_function, get_module_attr

import logging
from importlib import reload

reload(logging)  # To override all that pytorch_lightning has done with logging

parser = argparse.ArgumentParser(description='Run experiment.')

parser.add_argument('-e', dest='exp_name', type=str,
                    help='experiment name', required=True)

parser.add_argument('-g', dest='gpu', type=int,
                    help='gpu number', default=0)

parser.add_argument('-w', dest='num_workers', type=int,
                    help='workers number', default=12)

args = parser.parse_args()

name_split = args.exp_name.split('.')
exp_subfolder = name_split[0]
exp_name = '.'.join(name_split[1:])

num_workers = args.num_workers
gpus = [args.gpu, ]

exp_path = os.path.join(os.getcwd(), 'experiments_results', exp_subfolder,
                        exp_name)
os.makedirs(exp_path, exist_ok=True)
init_script(os.path.join(exp_path, 'log.txt'))

logging.info("Started")

e_mod = importlib.\
    import_module(f'experiments_configs.{exp_subfolder}.{exp_name}')

hparams = e_mod.get_hrapams()

train_img_ids, test_img_ids = get_train_test_img_ids_split(
    **hparams['dataset']['train_test_split'])

# import random
# train_img_ids = random.sample(train_img_ids, 50)
# test_img_ids = random.sample(test_img_ids, 10)

train_loader = torch.utils.data.DataLoader(
    PatchesDataset(train_img_ids,
                   csv_path=hparams['dataset']['csv_path'],
                   transform=get_module_attr(hparams['dataset']
                                             ['train_augmetations']),
                   scale=hparams['dataset']['scale'],
                   load_masks=True),
    batch_size=hparams['batch_size'], shuffle=True,
    num_workers=num_workers, pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    PatchesDataset(test_img_ids,
                   csv_path=hparams['dataset']['csv_path'],
                   transform=get_module_attr(hparams['dataset']
                                             ['test_augmetations']),
                   scale=hparams['dataset']['scale'],
                   load_masks=True),
    batch_size=hparams['batch_size'], shuffle=False,
    num_workers=num_workers, pin_memory=True,
)

steps_in_epoh = len(train_loader)
print("steps_in_epoh", steps_in_epoh)
# Update with actual steps_in_epoh value
hparams = e_mod.update_hrapams(hparams, steps_in_epoh)

model = call_function(hparams['module']['params']['model'])
module_name = hparams['module']['name']
module_params = copy.deepcopy(hparams['module']['params'])
module_params['model'] = model
module_params['hparams'] = hparams
module_params['log_train_every_batch'] = True
module = call_function([module_name, module_params])

logger = TensorBoardLogger(
    os.path.join('experiments_results', exp_subfolder), exp_name,
)
# accumulate_grad_batches=1
trainer = Trainer(logger, max_epochs=hparams['epochs'], gpus=gpus,
                  fast_dev_run=False)

trainer.fit(module, train_loader, val_loader)
trainer.save_checkpoint(os.path.join(trainer.checkpoint_callback.dirpath,
                                     "last.ckpt"))

logging.info("Finished")
