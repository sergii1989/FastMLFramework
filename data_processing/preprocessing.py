import gc
import numpy as np
import pandas as pd

from builtins import map
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import VarianceThreshold
from generic_tools.utils import timing, auto_selector_of_categorical_features


# ======================================================================
# Functions to optimize data types and gather stats on missing values
# ======================================================================

@timing
def downcast_datatypes(df):  # type: (pd.DataFrame) -> pd.DataFrame
    """
    Optimizes data-types in a pandas DF to reduce memory allocation
    :param df: input pandas DF
    :return: pandas DF with optimized data-types
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def missing_data(df):  # type: (pd.DataFrame) -> pd.DataFrame
    """
    Construct DF with missing values (absolute number / pct)
    :param df: input pandas DF to be analysed for missing values
    :return: pandas DF with feature as index and number/pct of missing values (only cols with NAN values)
    """
    na_absolute_values = df.isnull().sum().sort_values(ascending=False)
    na_relative_values = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    na_relative_values = na_relative_values.map(lambda x: round(x, 2))

    missing_data_df = pd.concat([na_absolute_values, na_relative_values], axis=1, keys=['Total', 'Percent'])
    missing_data_df = missing_data_df[missing_data_df['Total'] != 0].reset_index().rename(columns={"index": "Feature"})

    print('Number of features with missing values: {0}'.format(
        missing_data_df.loc[missing_data_df['Total'] > 0].shape[0]))

    return missing_data_df


def pct_missing_values_train_test(train_df, test_df, feature):  # type: (pd.DataFrame, pd.DataFrame, str) -> None
    pct_missing_train = train_df[feature].isnull().sum() / round(train_df.shape[0], 2) * 100.0
    pct_missing_test = test_df[feature].isnull().sum() / round(test_df.shape[0], 2) * 100.0
    print('Feature {0}. Missing values: {1:0.2f}% - Train, {2:0.2f}% - Test'.format(
        feature, pct_missing_train, pct_missing_test))


def convert_object_to_category_type(df, categorical_cols):  # type: (pd.DataFrame, list) -> pd.DataFrame
    """
    Converts given object columns to category data-type
    :param df: input pandas DF
    :param categorical_cols: list of features to be transformed to categorical data-type
    :return: output pandas DF
    """
    for col in categorical_cols:
        df[col].fillna(-999999, inplace=True)
        df[col] = df[col].astype('category')
    return df


def impute_nan_values_with_group_by(train_df,
                                    test_df,
                                    cols_groupby,
                                    nan_cols_handler):  # type: (pd.DataFrame, pd.DataFrame, list, list(tuple)) -> list
    """
    Imputing NAN values using information from other columns through groupby
    :param train_df: pandas DF containing train data set
    :param test_df: pandas DF containing test data set
    :param cols_groupby: columns to groupby concatenated train and test data set (e.g. SEX: F, M) ->
    :param nan_cols_handler: e.g. [(['feat_1', 'feat_2'], np.mean)]
    :return: list with train and test DF inside
    """
    df_temp = pd.concat([train_df, test_df], axis=0)
    data_sets = [train_df, test_df]
    for cols_nan, agg_type in nan_cols_handler:
        for col in cols_nan:
            grouped = df_temp.loc[df_temp[col].notnull()].groupby(cols_groupby).agg({col: agg_type})
            for data_set in data_sets:
                for item, row in data_set.loc[data_set[col].isnull(), [col] + cols_groupby].iterrows():
                    new_value = grouped.loc[tuple(row[x] for x in cols_groupby)].values[0]
                    data_set.loc[item, col] = new_value
    del df_temp; gc.collect()
    return data_sets


def impute_nan_values(train_df,
                      test_df,
                      nan_cols_handler):  # type: (pd.DataFrame, pd.DataFrame, list(tuple)) -> list(pd.DataFrame)
    """
    Imputing NAN values with aggregation function
    :param train_df: pandas DF containing train data set
    :param test_df: pandas DF containing test data set
    :param nan_cols_handler: e.g. [(['feat_1', 'feat_2'], np.mean)]
    :return: list with train and test DF
    """
    df_temp = pd.concat([train_df, test_df], axis=0)
    data_sets = [train_df, test_df]
    for data_set in data_sets:
        for cols_nan, agg_type in nan_cols_handler:
            for col in cols_nan:
                data_set[col].fillna(agg_type(list(
                    df_temp.loc[df_temp[col].notnull(), col].values)), inplace=True)
    del df_temp; gc.collect()
    return data_sets


def one_hot_encoder(df, nan_as_category=True, uppercase=True):  # type: (pd.DataFrame, bool) -> (pd.DataFrame, list)
    """
    One-hot encoding for categorical columns with pandas get_dummies()
    :param df: input pandas DF
    :param nan_as_category: encode nan as category flag (default True)
    :param uppercase: if True -> make new column name uppercase
    :return: updated pandas DF with added OHE columns + list of added columns
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    if uppercase:
        df.columns = df.columns.map(lambda x: x.upper())
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# ======================================================================
# Functions to aggregate differences between train and test datasets
# ======================================================================

def get_feats_that_can_be_converted_to_int(df, seed=27, n_samples=100):  # type: (pd.DataFrame, int, int) -> list
    """
    This method returns features that can be potentially down-casted to int. It uses random subsample of data of size
    n_samples and checks np.allclose(x, int(x), rtol=0, atol=0)) condition. If all samples are True -> feature can be
    potentially down-casted to int data type.
    :param df: pandas DF containing both train and test data sets
    :param seed: used for random sampler of the dataframe
    :param n_samples: number of samples to take from a DF when checking the possibility of downcast to int
    :return: list of features (or empty list)
    """
    cols_to_int = []
    for col in df.columns:
        if df[col].dtype.name not in ['object', 'category', 'int8', 'int16', 'int32', 'int64']:
            convert_to_int_possible = df.loc[df[col].notnull(), col].sample(n=n_samples, random_state=seed).map(
                lambda x: np.allclose(x, int(x), rtol=0, atol=0)).all()
            if convert_to_int_possible:
                cols_to_int.append(col)
    if len(cols_to_int):
        print('There are {0} numerical features that can be potentially converted to int'.format(len(cols_to_int)))
    return cols_to_int


def get_cat_feats_diff_between_train_and_test(df, target_column, cat_features=None,
                                              rtol_thresh=0.08, atol_thresh=0.1,
                                              int_threshold=10):
    """
    This method constructs DF with the categorical features having considerable diff in train VS test data sets
    :param df: pandas DF containing both train and test data sets
    :param target_column: target column (to be predicted)
    :param cat_features: list of categorical features
    :param rtol_thresh: threshold for max acceptable relative diff between two values
    :param atol_thresh: threshold for max acceptable abs diff between two values
    :param int_threshold: this threshold is used to limit number of int8-type numerical features to be interpreted
                          as categorical
    :return: pandas DF with the features having considerable diff in train VS test data sets
    """
    train_df = df[df[target_column].notnull()]
    test_df = df[df[target_column].isnull()]

    cat_features = auto_selector_of_categorical_features(
        df, cols_exclude=[target_column], int_threshold=int_threshold) if cat_features is None else cat_features

    df_cat_feats_diff = {}
    for feature in cat_features:
        temp_1 = (train_df[feature].value_counts(normalize=True)*100.0).rename(columns={feature: 'VALUE'})
        temp_2 = (test_df[feature].value_counts(normalize=True)*100.0).rename(columns={feature: 'VALUE'})
        temp_3 = pd.concat([temp_1, temp_2], axis=1, keys=['TRAIN', 'TEST']).reset_index().rename(
            columns={'index': 'CATEGORY'})
        temp_3['FEATURE'] = feature
        temp_3.set_index(['FEATURE', 'CATEGORY'], drop=True, inplace=True)
        for key, value in temp_3.iterrows():
            if any(list(map(lambda x: np.isnan(x), [value['TRAIN'], value['TEST']]))):
                df_cat_feats_diff[key] = value
            if not np.isclose(value['TRAIN'], value['TEST'], rtol=rtol_thresh, atol=atol_thresh):
                df_cat_feats_diff[key] = value
    df_cat_feats_diff = pd.DataFrame(df_cat_feats_diff).T
    print('There are %d categorical features having significant differences between '
          'train and test' % len(df_cat_feats_diff.index.get_level_values(0).unique()))
    return df_cat_feats_diff


# ======================================================================
# Functions to find binary features with the near zero variance
# ======================================================================

def get_near_zero_variance_binary_cat_feats(df, target_column,
                                            p_value=0.95):  # type: (pd.DataFrame, str, float) -> (list, list)
    """
    This method searches for features that are either one or zero (on or off) in more than p_value*100.% of the samples.
    Boolean features are Bernoulli random variables, and the variance of such variables is given by
    Var[X] = p_value*(1-p_value). E.g. if p_value=0.95 -> Var[X]=0.0475
    :param df: pandas DF representing concatenated train and test data sets
    :param target_column: target column (to be predicted)
    :param p_value: p-value
    :return: list of binary categorical features having variance below the threshold p_value*(1-p_value) and the
    ones having variance above the threshold
    """

    assert target_column in df.columns, \
        'Please add {target} column to the dataframe'.format(target=target_column)

    # Selecting binary columns only
    bin_cols = [col for col in df if df[col].dropna().value_counts().index.isin([0, 1, 0.0, 1.0, True, False]).all()
                and col != target_column]

    var_threshold = p_value * (1 - p_value)
    sel = VarianceThreshold(threshold=var_threshold)
    sel.fit_transform(df[bin_cols])

    zero_var_bin_cols = list(df[bin_cols].loc[:, ~sel.get_support()].columns)
    bin_cols_remain = list(set(bin_cols).difference(set(zero_var_bin_cols)))

    print('There are {0} features that have variance less than {1}%:\n{2}\n'.format(
        len(zero_var_bin_cols), round(var_threshold*100., 3), zero_var_bin_cols))
    print('There are {0} features that have variance above {1}%:\n{2}'.format(
        len(bin_cols_remain), round(var_threshold * 100., 3), bin_cols_remain))

    return zero_var_bin_cols, bin_cols_remain


# ======================================================================
# Functions transform skewed data
# ======================================================================

def get_skewness_each_num_feature(df):  # type: (pd.DataFrame) -> pd.DataFrame
    """
    Get skewness of the distribution of each numerical feature in a pandas DF
    :param df: input pandas DF
    :return: output DF with skewness per column
    """
    skew_of_features = {}
    dtypes_to_include = ['float64', 'float32', 'float16']
    columns = df.select_dtypes(include=dtypes_to_include).columns
    for col in columns:
        skew_of_features[col] = skew(df[col], nan_policy='omit')
    skewness_df = pd.DataFrame(index=skew_of_features.keys(), data=skew_of_features.values(),
                               columns=['SKEWNESS']).sort_values(by='SKEWNESS')
    return skewness_df.round(decimals=2)


def _transform_right_skewed_distribution(df, feature):  # type: (pd.DataFrame, str) -> pd.DataFrame
    """
    This method applies transformations to right skewed (skew > 0) features distribution.
    :param df: pandas DF with original feature data
    :param feature: name of the feature to which apply transformation
    :return: pandas DF with original feature data + additional columns for corresponding transformations
    """
    df[feature + '^1/2'] = df[feature].map(lambda x: x ** 1 / 2)
    df[feature + '^1/3'] = df[feature].map(lambda x: x ** 1 / 3)
    df[feature + '^1/4'] = df[feature].map(lambda x: x ** 1 / 4)
    return df


def _transform_left_skewed_distribution(df, feature):  # type: (pd.DataFrame, str) -> pd.DataFrame
    """
    This method applies transformations to left skewed (skew < 0) features distribution.
    :param df: pandas DF with original feature data
    :param feature: name of the feature to which apply transformation
    :return: pandas DF with original feature data + additional columns for corresponding transformations
    """
    df[feature + '^2'] = df[feature].map(lambda x: x ** 2)
    df[feature + '^3'] = df[feature].map(lambda x: x ** 3)
    df[feature + '^4'] = df[feature].map(lambda x: x ** 4)
    return df


def _skew_improved(new_skew, original_skew, threshold):  # type: (float, float, float) -> bool
    """
    This method checks whether skewness improvement due to transformation is above the threshold
    :param new_skew: skewness of distribution after applied transformation
    :param original_skew: skewness of distribution prior applied transformation
    :param threshold: min improvement of skewness (to filter cases with insignificant improvement)
    :return: True if skewness improvement is above the threshold, else False
    """
    return abs(abs(new_skew) - abs(original_skew)) > threshold


def compute_skew_kurtosis(df, feature):  # type: (pd.DataFrame, str) -> (float, float)
    """

    :param df: pandas DF with original feature data
    :param feature: name of the feature to which compute skew and kurtosis
    :return: skew and kurtosis
    """
    skew_val = float(skew(df[feature].values, nan_policy='omit'))
    kurtosis_val = float(kurtosis(df[feature].values, nan_policy='omit'))
    return skew_val, kurtosis_val


@timing
def transform_skewed_features(df, min_skew_improvement=0.15,
                              verbose=False):  # type: (pd.DataFrame, float, bool) -> (pd.DataFrame, dict, list)
    """
    This method apply transformations to numerical features so to reduce skewness of distributions. It employs logic
    described in http://seismo.berkeley.edu/~kirchner/eps_120/Toolkits/Toolkit_03.pdf. This method
    :param df: pandas DF to be processed
    :param min_skew_improvement: threshold for min skew improvement. If not sufficient -> leave the feature as it is
    :param verbose: set True if one need to debug (will print all intermediate steps information)
    :return: pandas DF with added columns for modified features, list of transformed features, and list
            of features to which abs() was applied
    """
    skew_df = get_skewness_each_num_feature(df)
    feats_to_transform = list(skew_df.index)

    print('\nNumber of features to transform: %d' % len(feats_to_transform))
    print('Number of columns with NAN skew: %d\n' % skew_df.isnull().sum())

    features_applied_abs = []  # list of features where abs() was applied
    features_transformed = {}  # dict containing original<->best transformed features

    for feature in feats_to_transform:
        df_temp = df[feature].copy()
        df_temp = df_temp.dropna().to_frame(name=feature)  # removing nan values

        min_val, max_val = df_temp[feature].min(), df_temp[feature].max()
        skew_val, kurtosis_val = compute_skew_kurtosis(df_temp, feature)
        if verbose:
            print('\nFeature {0}; \nmin: {1}; max: {2}; skew: {3}; kurtosis: {4}'.format(
                feature, min_val, max_val, skew_val, kurtosis_val))

        if min_val < 0.0 and max_val <= 0.0:
            if verbose:
                print('All values <=0. Applying abs() to make distribution positive.')

            df_temp[feature] = df_temp[feature].apply(abs)
            df[feature] = df_temp[feature]  # update original dataframe
            skew_val, kurtosis_val = compute_skew_kurtosis(df_temp, feature)

            if verbose:
                print('New values: skew: {0}; kurtosis: {1}'.format(skew_val, kurtosis_val))
            features_applied_abs.append(feature)

        if skew_val == 0.0:
            if verbose:
                print('No transformation is needed for feature {0}.'.format(feature))
            continue

        if 0.0 in df_temp[feature]:
            # If zero in values -> use np.log1p to avoid -inf after transformation
            df_temp[feature + '_LOG1P'] = df_temp[feature].map(lambda x: np.log1p(x))
        else:
            df_temp[feature + '_LOG'] = df_temp[feature].map(lambda x: np.log(x))

        if skew_val < 0.0:
            # The data are right-skewed (clustered at lower values) move down the ladder of powers
            # (e.g. square root, cube root, logarithmic, etc. transformations)
            df_temp = _transform_right_skewed_distribution(df_temp, feature)
        elif skew_val > 0.0:
            # The data are left-skewed (clustered at higher values) move up the ladder of powers
            # (cube, square, etc)
            df_temp = _transform_left_skewed_distribution(df_temp, feature)

        results = {col: abs(float(skew(df_temp[col].values, nan_policy='omit'))) for col in df_temp.columns}
        if verbose:
            print(results)

        column_min_skew = min(results, key=results.get)
        min_skew = results[column_min_skew]

        if column_min_skew != feature and _skew_improved(min_skew, skew_val, min_skew_improvement):
            if verbose:
                print('{0} transformation: skew {1} vs original {2}'.format(column_min_skew, min_skew, skew_val))
            features_transformed[feature] = column_min_skew
            df[column_min_skew] = df_temp[column_min_skew]
        else:
            if verbose:
                print('Transformations did not improve skewness of distribution as compared to the threshold')
            pass
    del skew_df; gc.collect()

    print
    print('=' * 70)
    print('Number of features transformed: %d' % len(features_transformed))
    print('Number of features where transformation did not improve skewness: %d' %
          (len(feats_to_transform) - len(features_transformed)))
    print('Number of features applied abs(): %d' % len(features_applied_abs))
    print('=' * 70)
    print
    return df, features_transformed, features_applied_abs
