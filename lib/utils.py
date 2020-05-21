import torch
from torch import nn

import sys
import logging
from io import StringIO
import collections.abc
import traceback
import linecache
import importlib

import numpy as np
import pandas as pd
from subprocess import Popen, PIPE


SYMBOLS = {
    'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
    'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa',
                      'zetta', 'iotta'),
    'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
    'iec_60027_2': ('BiB', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB',
                    'YiB'),
    'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi',
                'zebi', 'yobi'),
}


def update_r(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_r(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# OS utils


def get_exception():
    _, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno,
                                                        line.strip(),
                                                        exc_obj)


def my_except_hook(etype, value, tb):
    result = StringIO()
    traceback.print_exception(etype, value, tb, file=result)
    logging.error(result.getvalue())
    sys.__excepthook__(etype, value, tb)


def init_script(filename):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(format=log_format, filename=filename,
                        level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Set up global exception hook
    sys.excepthook = my_except_hook


# Work with data


def array2patches(x, p_sz):
    x = x.copy()
    p_height = x.shape[0] // p_sz
    p_width = x.shape[1] // p_sz
    x = x[:p_height*p_sz, :p_width*p_sz]
    x = np.moveaxis(x, -1, 0)
    patches = x.reshape(-1, p_height, p_sz, p_width, p_sz).swapaxes(2, 3)
    patches = np.moveaxis(patches, 0, -1)

    return patches


# Training utils


def get_pretrained_model(get_model_fn, fn_args, checkpoint, device,
                         encoder_only=True):
    tmp = torch.load(checkpoint, map_location=device)

    module = nn.Sequential()
    if isinstance(get_model_fn, str):
        get_model_fn = get_module_attr(get_model_fn)
    model = get_model_fn(**fn_args)
    module.add_module('model', model)
    module.to(device)
    module.load_state_dict(tmp['state_dict'])

    if encoder_only:
        model.segmentation = False
        model.classification_head = None
        model.autodecoder = None

    module.eval()

    return model


def get_features(imgs, model, device, rgb_mean, rgb_std,
                 features_batch_size=512):
    model.eval()

    imgs = imgs if isinstance(imgs, torch.Tensor) else torch.from_numpy(imgs)
    imgs = imgs.to(device)
    if rgb_mean is not None or rgb_std is not None:
        n_imgs = (imgs - rgb_mean) / rgb_std
    else:
        n_imgs = imgs

    b_features = []
    for b in range(0, n_imgs.shape[0], features_batch_size):
        with torch.no_grad():
            features, *_ = model(n_imgs[b:b+features_batch_size],
                                 return_features=True)
            b_features.append(features)

    features = torch.cat(b_features, dim=0)

    return features.cpu()


def get_module_attr(full_name):
    if isinstance(full_name, str):
        mn = full_name.split('.')
        mod = importlib.import_module('.'.join(mn[:-1]))
        attr = getattr(mod, mn[-1])
    else:
        attr = full_name  # it already is an object

    return attr


def call_function(fn):
    if isinstance(fn, str):
        full_name, params = fn, {}
    elif isinstance(fn, list) or isinstance(fn, tuple):
        full_name = fn[0]
        params = fn[1] if len(fn) > 1 else {}
    elif isinstance(fn, dict):
        full_name, params = fn['name'], fn['params']

    function = get_module_attr(full_name)

    return function(**params)


def human2bytes(s):
    '''Convert string with symbolic designation to a number of bytes'''
    s = s.strip()
    init = s
    num = ""
    while s and s[0:1].isdigit() or s[0:1] == '.':
        num += s[0]
        s = s[1:]
    num = float(num)
    letter = s.strip()

    for _, sset in SYMBOLS.items():
        if letter in sset:
            break
    else:
        if letter == 'k':
            # treat 'k' as an alias for 'K' as per: http://goo.gl/kTQMs
            sset = SYMBOLS['customary']
            letter = letter.upper()
        else:
            raise ValueError("can't interpret {}".format(init))
    prefix = {sset[0]: 1}
    for i, s in enumerate(sset[1:]):
        prefix[s] = 1 << (i + 1) * 10
    return int(num * prefix[letter])


def get_GPUs_info(params=['uuid', 'name', 'memory.total', 'memory.used',
                          'memory.free']):
    '''
    Return list of dictionaries with the requested information about GPUs.
    Use nvidia-smi --help-query-gpu to see more info about available
    parameters.
    '''
    proc = Popen(['nvidia-smi', '--format=csv', '--query-gpu=' +
                  ','.join(params)], stdout=PIPE)
    df = pd.read_csv(proc.stdout)
    exit_code = proc.wait()

    if exit_code != 0:
        raise OSError('nvidia-smi exited with code {}'.format(exit_code))

    def parse_s(s):
        try:
            return human2bytes(s)
        except:
            return s.strip()

    return [{pn: parse_s(r.iloc[n]) for n, pn in enumerate(params)}
            for i, r in df.iterrows()]
