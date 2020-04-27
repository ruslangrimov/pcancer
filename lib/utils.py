import sys
import logging
from io import StringIO
import traceback
import linecache

import numpy as np

# Utils


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
