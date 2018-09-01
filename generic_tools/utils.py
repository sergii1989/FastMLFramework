import numpy as np

from time import time
from typing import Any
from functools import wraps


def timing(func):
    """
    Wrapper of a function to show time of its execution
    :param func: func to decorate
    :return: print time of execution of a given function
    """
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        print('\n***** \'{0}\' took {1:2.3f} sec\n'.format(func.__name__, te - ts))
        return result
    return wrap


def merge_two_dicts(x, y):  # type: (dict, dict) -> dict
    """
    Given two dicts, merge them into a new dict as a shallow copy.
    :param x: first dict
    :param y: second dict
    :return: merged dict
    """
    z = x.copy()
    z.update(y)
    return z


def get_binning_list(val_min, val_max, bin_size=None, n_bins=10):  # type: (Any, Any, Any) -> list
    """
    Function to construct list of bins using start/end points of the range and bin size.
    If size of bin (step) is not provided -> compute it based on n_bins (default=10)
    :param val_min: start point of the range (int or float)
    :param val_max: end point of the range (int or float)
    :param bin_size: size of the bin (int or float). If not provided, use n_bins to compute bin_size
    :param n_bins: number of bins (int). Default 10. It is used to estimate the bin_size if bin_size=None.
    :return: list of bins (e.g. [0, 10, 20] if val_min=0 val_max=20, bin_size=10)
    """
    if all(map(lambda x: isinstance(x, int), [val_min, val_max])):
        bin_size = bin_size if bin_size is not None else int((val_max - val_min) / n_bins)
        shrink = range(val_min, val_max, bin_size)
    else:
        bin_size = bin_size if bin_size is not None else round((val_max - val_min) / n_bins, 1)
        shrink = np.arange(val_min, val_max, bin_size)
    return shrink