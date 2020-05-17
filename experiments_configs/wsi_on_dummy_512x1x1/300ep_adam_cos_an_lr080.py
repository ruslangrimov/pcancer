import os, sys
sys.path.append(os.path.dirname(__file__))
import _default as d

d.epochs = 300
d.warmup_steps = 0

hparams = {
    'epochs': d.epochs,
    'learning_rate': 0.0008 * d.batch_size / 256,
    'optimizer': {
        'name': 'torch.optim.Adam',
        'params': {
            'weight_decay': 1e-4
        }
    },
    'scheduler': {
        'name': 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 3960,
            'T_mult': 1
        },
        'interval': 'step'
    },
    'source_code': open(__file__, 'rt').read()
}

d.hparams.update(hparams)


def get_hrapams():
    return d.get_hrapams()


def update_hrapams(hparams, steps_in_epoh):
    return d.update_hrapams(hparams, steps_in_epoh)
