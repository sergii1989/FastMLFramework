import os
import numpy as np

from time import time
from typing import Any
from functools import wraps
from datetime import datetime
from contextlib import contextmanager


@contextmanager
def timer(text):
    """
    This method wraps the given function and prints its time of execution accompanied by the provided text
    :param text: text to be included in the printouts
    :return: print execution time of the function along with the provided text
    """
    t0 = time()
    yield
    print("{} - done in {:.0f}s".format(text, time() - t0))


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


def get_current_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M')


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


def auto_selector_of_categorical_features(df, cols_exclude=[], int_threshold=10):
    """
    This method is used for selecting categorical features in a pandas DF. It uses features labeled as 'category'
    and 'object', but also 'int8' data types with additional filter applied on max and min values (int_threshold).
    This filter is needed to exclude high cardinality numerical features from being used as categorical feature.
    :param df: pandas DF with the dataset
    :param cols_exclude: columns to be excluded from a DF (e.g. target column)
    :param int_threshold: this threshold is used to limit number of int8-type numerical features to be interpreted
                          as categorical
    :return: sorted list of categorical features
    """
    df_temp = df.loc[:, ~df.columns.isin(cols_exclude)]
    cat_object_cols = df_temp.select_dtypes(include=['category', 'object']).columns
    int8_cols = df_temp.select_dtypes(include=['int8']).columns
    int8_cat_cols = []
    for int8_col in int8_cols:
        if abs(df_temp[int8_col].min()) <= int_threshold and abs(df_temp[int8_col].max()) <= int_threshold:
            int8_cat_cols.append(int8_col)
    return sorted(set(cat_object_cols).union(set(int8_cat_cols)))


def check_file_exists(filename, silent=True):
    if not os.path.exists(filename):
        if not silent: print('{} does not exist'.format(filename))
        return False
    return True


def create_output_dir(path_output_dir):
    if not os.path.exists(path_output_dir):
        print('Output directory {} is created'.format(path_output_dir))
        os.makedirs(path_output_dir)
