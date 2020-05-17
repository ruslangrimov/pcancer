import os, sys
sys.path.append(os.path.dirname(__file__))
import _default as d

d.epochs = 90
d.warmup_steps = 0

hparams = {
    'epochs': d.epochs,
    'learning_rate': 0.001 * d.batch_size / 256,
    'optimizer': {
        'name': 'torch.optim.Adam',
        'params': {
            'weight_decay': 1e-4
        }
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
            'reg': 1 / 2 * 0.8,
            'class': 9 / 2 * 0.2
        },
        'label_smoothing': 0.1
    },
    'source_code': open(__file__, 'rt').read()
}

d.hparams.update(hparams)


def get_hrapams():
    return d.get_hrapams()


def update_hrapams(hparams, steps_in_epoh):
    return d.update_hrapams(hparams, steps_in_epoh)
