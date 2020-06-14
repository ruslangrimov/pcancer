import os
import argparse
import importlib
import copy

from functools import partial

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from lib.dataloaders import (WSIPatchesDatasetRaw, WSIPatchesDummyDataloader,
                             WSIPatchesDataloader,
                             WSIPatchesDatasetTiledV1)
from lib.datasets import get_train_test_img_ids_split
from lib.utils import (init_script, call_function, get_module_attr,
                       get_pretrained_model, get_features)

import logging
from importlib import reload

reload(logging)  # To override all that pytorch_lightning has done with logging

parser = argparse.ArgumentParser(description='Run experiment.')

parser.add_argument('-e', dest='exp_name', type=str,
                    help='experiment name', required=True)

parser.add_argument('-g', dest='gpu', type=int,
                    help='gpu number', default=0)

parser.add_argument('-p_g', dest='patches_gpu', type=int,
                    help='patches gpu number', default=1)

parser.add_argument('-w', dest='num_workers', type=int,
                    help='workers number', default=5)

args = parser.parse_args()

name_split = args.exp_name.split('.')
exp_subfolder = name_split[0]
exp_name = '.'.join(name_split[1:])

num_workers = args.num_workers
gpus = [args.gpu, ]

# exp_path = os.path.join(os.getcwd(), 'experiments_results', exp_subfolder,
#                         exp_name)
exp_path = os.path.join('/mnt/HDDData/notebooks/pcancer/',
                        'experiments_results', exp_subfolder,
                        exp_name)
os.makedirs(exp_path, exist_ok=True)
init_script(os.path.join(exp_path, 'log.txt'))

logging.info("Started")

e_mod = importlib.\
    import_module(f'experiments_configs.{exp_subfolder}.{exp_name}')

hparams = e_mod.get_hrapams()

train_img_ids, test_img_ids = get_train_test_img_ids_split(
    **hparams['dataset']['train_test_split'])

# ToDo: add split into folds
fold_train_img_ids = train_img_ids
fold_test_img_ids = test_img_ids

exp_path = os.path.join(os.getcwd(), 'experiments_results', exp_subfolder,
                        exp_name)

if hparams['dataset']['dataloader'] == 'dummy':
    train_batch_path = hparams['dataset']['train_batch_path']
    test_batch_path = hparams['dataset']['test_batch_path']

    train_loader = WSIPatchesDummyDataloader(train_batch_path,
                                             precalc_epochs=hparams['dataset']
                                             ['precalc_epochs'],
                                             batch_size=hparams['batch_size'],
                                             shuffle=True)
    val_loader = WSIPatchesDummyDataloader(test_batch_path,
                                           precalc_epochs=hparams['dataset']
                                           ['precalc_epochs'],
                                           batch_size=hparams['batch_size'],
                                           shuffle=False)

elif hparams['dataset']['dataloader'] == 'tiled_old':
    train_transforms = get_module_attr(hparams['dataset']
                                       ['train_augmetations'])
    test_transforms = get_module_attr(hparams['dataset']['test_augmetations'])

    h_dp = hparams['dataset']

    train_params = {
        'image_ids': fold_train_img_ids,
        'n_tiles': h_dp['n_tiles'],
        'tile_sz': h_dp['tile_sz'],
        'pkl_path': h_dp['patches_pkl_path'],
        'path': h_dp['patches_path'],
        'scale': h_dp['scale'],
        'transform': train_transforms,
        'train': True,
        'max_size': h_dp['max_size']
    }

    test_params = {
        'image_ids': fold_test_img_ids,
        'n_tiles': h_dp['n_tiles'],
        'tile_sz': h_dp['tile_sz'],
        'pkl_path': h_dp['patches_pkl_path'],
        'path': h_dp['patches_path'],
        'scale': h_dp['scale'],
        'transform': test_transforms,
        'train': False,
        'max_size': h_dp['max_size']
    }

    train_dataset = WSIPatchesDatasetTiled(**train_params)
    test_dataset = WSIPatchesDatasetTiled(**test_params)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=hparams['batch_size'],
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=hparams['batch_size'],
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True)

else:
    patches_device = torch.device(f'cuda:{args.patches_gpu}')
    model_fn = get_module_attr(hparams['dataset']['patches_model']['name'])
    model = get_pretrained_model(model_fn,
                                 hparams['dataset']['patches_model']['params'],
                                 hparams['dataset']['patches_model']
                                 ['checkpoint'],
                                 patches_device)

    get_features_fn = partial(
        get_features, model=model,
        rgb_mean=torch.tensor(hparams['dataset']['rgb_mean'],
                              dtype=torch.float32, device=patches_device),
        rgb_std=torch.tensor(hparams['dataset']['rgb_std'],
                             dtype=torch.float32, device=patches_device),
        features_batch_size=512
    )

    train_transforms = get_module_attr(hparams['dataset']
                                       ['train_augmetations'])

    if hparams['module']['params']['model']['params']['f_size'] == 1:
        f_shape = (512,)
    else:
        f_shape = (512, 8, 8)

    train_loader =\
        WSIPatchesDataloader(
            WSIPatchesDatasetRaw(fold_train_img_ids,
                                 hparams['dataset']['patches_pkl_path'],
                                 scale=0.5, transform=train_transforms),
            get_features_fn, f_shape, patches_device, hparams['batch_size'],
            shuffle=True, num_workers=num_workers,
            reduce_mean=hparams['dataset']['reduce_mean'],
            reduce_max=hparams['dataset']['reduce_max'],
            max_len=hparams['dataset']['max_len'])

    test_transforms = get_module_attr(hparams['dataset']['test_augmetations'])
    val_loader =\
        WSIPatchesDataloader(
            WSIPatchesDatasetRaw(fold_test_img_ids,
                                 hparams['dataset']['patches_pkl_path'],
                                 scale=0.5, transform=test_transforms),
            get_features_fn, f_shape, patches_device,
            hparams['batch_size'], shuffle=False, num_workers=num_workers,
            reduce_mean=hparams['dataset']['reduce_mean'],
            reduce_max=hparams['dataset']['reduce_max'],
            max_len=hparams['dataset']['max_len'])

steps_in_epoh = len(train_loader)
print("steps_in_epoh", steps_in_epoh)
# Update with actual steps_in_epoh value
hparams = e_mod.update_hrapams(hparams, steps_in_epoh)

model = call_function(hparams['module']['params']['model'])
module_name = hparams['module']['name']
module_params = copy.deepcopy(hparams['module']['params'])
module_params['model'] = model
module_params['hparams'] = hparams
module_params['log_train_every_batch'] = False
module = call_function([module_name, module_params])

logger = TensorBoardLogger(
    os.path.join('/mnt/HDDData/notebooks/pcancer/',
                 'experiments_results', exp_subfolder), exp_name,
)

acc_grad = hparams.get('accumulate_grad_batches', 1)
trainer = Trainer(logger, max_epochs=hparams['epochs'], gpus=gpus,
                  accumulate_grad_batches=acc_grad,
                  fast_dev_run=False, num_sanity_val_steps=0)

trainer.fit(module, train_loader, val_loader)
trainer.save_checkpoint(os.path.join(trainer.checkpoint_callback.dirpath,
                                     "last.ckpt"))

logging.info("Finished")
