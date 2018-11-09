import os
import sys
import six
import logging
import numpy as np
import pandas as pd

from time import time
from functools import wraps
from sklearn import metrics
from datetime import datetime
from builtins import range, map
from contextlib import contextmanager
from generic_tools.loggers import configure_logging

configure_logging()
_logger = logging.getLogger("utils")


@contextmanager
def timer(text):
    """
    This method wraps the given function and prints its time of execution accompanied by the provided text
    :param text: text to be included in the printouts
    :return: print execution time of the function along with the provided text
    """
    t0 = time()
    yield
    _logger.info("{} - done in {:.0f}s".format(text, time() - t0))


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
        _logger.info('***** \'{0}\' took {1:2.3f} sec'.format(func.__name__, te - ts))
        return result
    return wrap


def get_current_timestamp():
    """
    This method returns current timestamp
    :return:
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M')


def _convert_metrics_scorer_to_str(metrics_scorer):
    """
    This method is work around to guarantee compatibility between python2/3. The issue is that in Python 3
    isinstance(variable, unicode) throws an error since unicode is not treated as str type
    :param metrics_scorer: name of metrics scorer in sklearn.metrics (basestring-type)
    :return: str
    """

    if isinstance(metrics_scorer, six.string_types):
        if sys.version_info.major == 2:
            return metrics_scorer.encode()  # since parsing of pyhocon config returns unicode
        else:  # sys.version_info.major == 3:
            return metrics_scorer
    else:
        raise TypeError('Type of metrics_scorer should be either str or unicode. '
                        'Instead received {0}.'.format(type(metrics_scorer)))


def get_metrics_scorer(metrics_scorer):  # type: (str) -> metrics
    """
    This method returns sklearn's metrics scorer function to the given string name
    :param metrics_scorer: name of the metrics score function to be used for results evaluation.
                           The metrics_scorer should be a string representation of any function from:
                           http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metricsname
    :return: sklearn's metrics scorer function
    """
    metrics_scorer = _convert_metrics_scorer_to_str(metrics_scorer)
    scorer = __import__('sklearn.metrics', globals(), locals(), [metrics_scorer], 0)
    try:
        return getattr(scorer, metrics_scorer)
    except AttributeError as e:
        _logger.error("Module {0} has no '{1}' score function. Please use in the config file one of the following "
                      "functions:\n\n{2}".format(scorer.__name__, metrics_scorer, scorer.__all__))


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


def get_binning_list(val_min, val_max, bin_size=None, n_bins=10):
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


def auto_selector_of_categorical_features(df, cols_exclude=[],
                                          int_threshold=10):  # type: (pd.DataFrame, list, int) -> list
    """
    This method is used for selecting categorical features in a pandas DF. It uses features labeled as 'category'
    and 'object', but also 'int8' data types with additional filter applied on max and min values (int_threshold).
    This filter is needed to exclude high cardinality numerical features from being used as categorical feature.
    :param df: pandas DF with the dataset
    :param cols_exclude: columns to be excluded from a DF (e.g. target column)
    :param int_threshold: this threshold is used to limit number of int8-type numerical features to be interpreted
                          as categorical. For instance, if there is a numeric feature with unique integer values ranged
                          from 0 to 8 and int_threshold=9 -> this column will be considered as categorical
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


def create_output_dir(path_output_dir, silent=False):  # type: (str, bool) -> None
    """
    This method creates directory if it not existed at the moment of request
    :param path_output_dir: absolute path to a file / directory
    :param silent: if True -> do not print the message that directory is created
    :return: None
    """
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
        if not silent:
            _logger.info('Output directory {} is created'.format(path_output_dir))


def generate_single_model_solution_id_key(model_name):  # type: (str) -> str
    """
    This method is used to generate a unique id key for single model solution. It is needed in the case if one wants to
    use the out-of-fold results (OOF) from a single model predictions in the sacking or blending process. It is hard
    to keep long names of the files, thus it deems reasonable to generate a unique id key that can point to the
    original file name. For instance:
        File train_OOF.csv in
            single_model_solution/lightgbm/features_dataset_001/target_permutation_fs_001/bayes_hpo_001/bagging_on
                will be assigned to unique id key: lgbm_7649
                    and stored in the oof_data_info.txt file located in the same directory.
    :param model_name: name of the model (e.g. 'lgb', 'xgb', etc.)
    :return: unique id key
    """
    np.random.seed()
    return '_'.join([model_name, ("%04d" % np.random.randint(9999))])
