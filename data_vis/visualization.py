import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any
from data_processing.preprocessing import missing_data, compute_skew_kurtosis
from generic_tools.utils import get_binning_list

# ======================================================================
# Function to plot correlation heatmap
# ======================================================================

def plot_features_corr_heatmap(df, vmin=-1, vmax=1, center=0, square=True,
                               figsize_x=14, figsize_y=10):  # type: (pd.DataFrame, Any, Any, Any, bool, Any, Any) -> None
    """
    This method plots features correlation heatmap
    :param df: pandas DF containing data set
    :param vmin: lower limit for the numerical range (feature distribution)
    :param vmax: upper limit for the numerical range (feature distribution)
    :param center: center of range (used for proper coloring of heatmap)
    :param square: if True -> heatmap will be squared (reduced num of feats / figure size to keep the square shape)
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :return: None
    """
    dtypes_to_include = ['float64', 'float32', 'int64', 'int32', 'int16', 'int8']
    corr = df.select_dtypes(include = dtypes_to_include).iloc[:, 1:].corr()
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    sns.heatmap(corr, vmin=vmin, vmax=vmax, center=center, square=square)
    fig.tight_layout()
    del corr; gc.collect()


# ======================================================================
# Functions to plot statistics on missing values
# ======================================================================

def plot_missing_values_stats(train_df, test_df=None, grid=(1, 1),
                              figsize_x=7, figsize_y=9): # type: (pd.DataFrame, pd.DataFrame, tuple, Any, Any) -> None
    """
    This method plots barplot with relative number of missing values in feature columns
    :param train_df: pandas DF containing train data set
    :param test_df: pandas DF containing test data set
    :param grid: figure's grid (by default (2, 1))
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :return: None
    """
    if test_df is not None:
        assert any(map(lambda x: x > 1, grid)), 'Please set the grid to either (1, 2) or (2, 1)..'

    fig, ax = plt.subplots(*grid, figsize=(figsize_x, figsize_y))

    # DF with missing values only
    train_df_na = missing_data(train_df)
    test_df_na = missing_data(test_df) if test_df is not None else None
    list_df = [train_df_na, test_df_na] if test_df_na is not None else [train_df_na]

    for i, df in enumerate(list_df):
        axis = ax if grid == (1, 1) else ax[i]
        sns.barplot(x="Percent", y="Feature", data=df, ax=axis)
        plt.setp(axis.get_yticklabels(), rotation=0, size=8)
        axis.set_ylabel('')

    plt.tight_layout()
    del list_df; gc.collect()


# ======================================================================
# Functions to aggregate differences between train and test datasets
# ======================================================================

def get_categorical_features_diff_between_train_and_test(train_df, test_df, rtol_thresh=0.08, atol_thresh=0.1):
    """
    This method constructs DF with the categorical features having considerable diff in train VS test data sets
    :param train_df: pandas DF containing train data set
    :param test_df: pandas DF containing test data set
    :param rtol_thresh: threshold for max acceptable relative diff between two values
    :param atol_thresh: threshold for max acceptable abs diff between two values
    :return: pandas DF with the features having considerable diff in train VS test data sets
    """
    cat_features = train_df.select_dtypes(include=['category']).columns
    df_cat_feats_diff = {}
    for feature in cat_features:
        temp_1 = (train_df[feature].value_counts(normalize=True)*100.0).rename(columns={feature: 'VALUE'})
        temp_2 = (test_df[feature].value_counts(normalize=True)*100.0).rename(columns={feature: 'VALUE'})
        temp_3 = pd.concat([temp_1, temp_2], axis=1, keys=['TRAIN', 'TEST']).reset_index().rename(
            columns={'index': 'CATEGORY'})
        temp_3['FEATURE'] = feature
        temp_3.set_index(['FEATURE', 'CATEGORY'], drop=True, inplace=True)
        for key, value in temp_3.iterrows():
            if any(map(lambda x: np.isnan(x), [value['TRAIN'], value['TEST']])):
                df_cat_feats_diff[key] = value
            if not np.isclose(value['TRAIN'], value['TEST'], rtol=rtol_thresh, atol=atol_thresh):
                df_cat_feats_diff[key] = value
    df_cat_feats_diff = pd.DataFrame(df_cat_feats_diff).T
    print 'There are %d categorcial features having significant differences between ' \
          'train and test' % len(df_cat_feats_diff.index.get_level_values(0).unique())
    return df_cat_feats_diff

# ======================================================================
# Functions to plot numerical features
# ======================================================================

def plot_numerical_feature_vs_target(train_df, feature, target,
                                     val_min=None, val_max=None, bin_size=None,
                                     figsize_x=14, figsize_y=4,
                                     label_rotation=90):  # type: (pd.DataFrame, str, str, Any, Any, Any, Any, Any, Any) -> None
    """
    This method plots binned numerical feature VS target
    :param train_df: pandas DF containing train data set
    :param feature: name of feature to plot
    :param target: name of target column
    :param val_min: lower limit for the numerical range (feature distribution)
    :param val_max: upper limit for the numerical range (feature distribution)
    :param bin_size: size of 1 bin (int / float)
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :param label_rotation: rotation angle of labels (0-horizontal, 90-vertical)
    :return: None
    """
    val_min = val_min if val_min is not None else train_df[feature].min()
    val_max = val_max if val_max is not None else train_df[feature].max()
    assert val_min < val_max, "val_max should be larger than val_min."

    shrink = get_binning_list(val_min, val_max, bin_size)

    list_df = []
    for target_val, temp_df in train_df.groupby(target)[feature]:
        list_df.append((temp_df.groupby(pd.cut(temp_df, shrink, right=False)).count()).rename(str(target_val)))

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    pd.concat(list_df, axis=1, names=map(lambda x: x.name, list_df)).plot.bar(stacked=True, ax=ax)
    ax.set_ylabel('Number of {0} in the range'.format(feature), size=12)
    ax.set_title("Train: {0} VS {1}".format(feature, target), size=13)
    plt.setp(ax.get_xticklabels(), rotation=label_rotation, size=11)
    plt.tight_layout()
    del list_df; gc.collect()


def plot_numerical_feature_train_vs_test(train_df, test_df, feature, val_min=None, val_max=None, bin_size=None,
                                         grid=(2, 1), figsize_x=14, figsize_y=7,
                                         label_rotation=90):  # type: (pd.DataFrame, pd.DataFrame, str, Any, Any, Any, tuple, Any, Any) -> None
    """
    This method plots train vs test comparison of a binned numerical feature
    :param train_df: pandas DF containing train data set
    :param test_df: pandas DF containing test data set
    :param feature: name of feature to plot
    :param val_min: lower limit for the numerical range (feature distribution)
    :param val_max: upper limit for the numerical range (feature distribution)
    :param bin_size: size of 1 bin (int / float)
    :param grid: figure's grid (by default (2, 1))
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :param label_rotation: rotation angle of labels (0-horizontal, 90-vertical)
    :return: None
    """
    temp_df = train_df.append(test_df).reset_index(drop=True)
    val_min = val_min if val_min is not None else temp_df[feature].min()
    val_max = val_max if val_max is not None else temp_df[feature].max()
    assert val_min < val_max, "val_max should be larger than val_min."

    shrink = get_binning_list(val_min, val_max, bin_size)

    train = (train_df[feature].groupby(pd.cut(train_df[feature], shrink, right=False)).count())
    test = (test_df[feature].groupby(pd.cut(test_df[feature], shrink, right=False)).count())
    fig, ax = plt.subplots(*grid, figsize=(figsize_x, figsize_y))

    for i, df in enumerate([train, test]):
        df.plot(kind='bar', ax=ax[i])
        ax[i].set_ylabel('Number {0}'.format(feature), size=12)
        ax[i].set_xlabel('{0} range'.format(feature), size=12)
        ax[i].set_title("{0}: number of unique {1} in selected range".format(
            'Train' if i == 0 else 'Test', feature), size=13)
        plt.setp(ax[i].get_xticklabels(), rotation=label_rotation, size=11)
        plt.setp(ax[i].get_yticklabels(), size=12)
        plt.tight_layout()

    del temp_df, train, test; gc.collect()


# ======================================================================
# Functions to plot categorical features
# ======================================================================

def countplot_cat_feature_vs_target(df, feature, target, figsize_x=7, figsize_y=4, label_rotation=0,
                                    print_cross_tab=True):  # type: (pd.DataFrame, str, str, Any, Any, int, bool) -> None
    """
    This method shows how a given categorical feature is related to the target (using countplot)
    :param df: pandas DF containing train data set
    :param feature: name of feature to plot
    :param target: name of target column
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :param label_rotation: rotation angle of labels (0-horizontal, 90-vertical)
    :param print_cross_tab: if True -> print pd.crosstab with feature vs target values
    :return: if print_cross_tab=True -> returns pd.crosstab with feature vs target values else None
    """
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    sns.countplot(x=feature, hue=target, data=df, ax=ax)
    ax.set_title("{0} vs {1}".format(feature, target), size=13)
    plt.setp(ax.get_xticklabels(), rotation=label_rotation, size=12)
    plt.setp(ax.get_yticklabels(), size=10)
    plt.tight_layout()

    if print_cross_tab:
        return pd.crosstab(df[feature], df[target], normalize='index').sort_values(
            df[target].unique()[0])


def countplot_cat_feature_train_vs_test(train_df, test_df, feature, normalize=False, grid=(1, 2),
                                        label_rotation=0, figsize_x=14, figsize_y=4, annot_size=8,
                                        annot_rotation=0):  # type: (pd.DataFrame, pd.DataFrame, str, bool, tuple, int, Any, Any, Any, int) -> None
    """
    This method plots a train vs test comparison for a given categorical feature (countplot-based)
    :param train_df: pandas DF containing train data set
    :param test_df: pandas DF containing test data set
    :param feature: name of feature to plot
    :param normalize: if True -> plot relative number of records (barplot), if False -> plot abs number of occurrences
    :param grid: figure's grid (by default (2, 1))
    :param label_rotation: rotation angle of labels (0-horizontal, 90-vertical)
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :param annot_size: size of text annotations (on the figure)
    :param annot_rotation: rotation angle of annotation text (0-horizontal, 90-vertical)
    :return: None
    """
    df_temp_1 = train_df[feature].value_counts(normalize=True)*100.0
    df_temp_1.index = df_temp_1.index.reorder_categories(list(df_temp_1.index.values))

    df_temp_2 = test_df[feature].value_counts(normalize=True)*100.0
    df_temp_2.index = df_temp_2.index.reorder_categories(list(df_temp_2.index.values))

    fig, ax = plt.subplots(*grid, figsize=(figsize_x, figsize_y))

    if normalize:
        sns.barplot(x=df_temp_1.index, y=df_temp_1.values, ax=ax[0]).set(ylim=(0, 108.0))
        ax[0].set_ylabel('Percentage, %', size=12)
        i = 0
        for index, value in df_temp_1.iteritems():
            if annot_rotation == 0:
                ax[0].text(i, value + 2, round(value, 2), color='black', ha="center",
                           fontsize=annot_size, rotation=annot_rotation)
            else:
                ax[0].text(i, value + 15, round(value, 2), color='black', ha="center",
                           fontsize=annot_size, rotation=annot_rotation)
            i += 1
    else:
        sns.countplot(x=feature, data=train_df, ax=ax[0])
    ax[0].set_title("Train: countplot of {0}".format(feature), size=13)
    plt.setp(ax[0].get_xticklabels(), rotation=label_rotation, size=12)
    plt.setp(ax[0].get_yticklabels(), size=12)
    plt.tight_layout()

    if normalize:
        sns.barplot(x=df_temp_2.index, y=df_temp_2.values, ax=ax[1]).set(ylim=(0, 108.0))
        ax[1].set_ylabel('Percentage, %', size=12)
        i = 0
        for index, value in df_temp_2.iteritems():
            if annot_rotation == 0:
                ax[1].text(i, value + 2, round(value, 2), color='black', ha="center",
                           fontsize=annot_size, rotation=annot_rotation)
            else:
                ax[1].text(i, value + 15, round(value, 2), color='black', ha="center",
                           fontsize=annot_size, rotation=annot_rotation)
            i += 1
    else:
        sns.countplot(x=feature, data=test_df, ax=ax[1])
    ax[1].set_title("Test: countplot of {0}".format(feature), size=13)
    plt.setp(ax[1].get_xticklabels(), rotation=label_rotation, size=12)
    plt.setp(ax[1].get_yticklabels(), size=12)
    plt.tight_layout()
    del df_temp_1, df_temp_2; gc.collect()


# ======================================================================
# Functions to plot distribution (count, density) of numerical features
# ======================================================================

def distplot_numerical_feature_vs_target(df, feature, target, density=True, dropna=True, bins=50,
                                         figsize_x=14, figsize_y=4): # type: (pd.DataFrame, str, str, bool, bool, int, Any, Any) -> None
    """
    This method plots density distribution of a numerical feature grouped by target
    :param df: pandas DF containing train data set
    :param feature: name of feature to plot
    :param target: name of target column
    :param density: if True -> plot kde density estimator
    :param dropna: if True -> drop all nan
    :param bins: number of bins
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :return: None
    """
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    for target_val, temp_df in df.groupby(target)[feature]:
        sns.distplot(temp_df.dropna() if dropna else temp_df, kde=density, bins=bins,
                     label=str(target_val), norm_hist=density, ax=ax)
    ax.set_xlabel('{0}'.format(feature), size=12)
    ax.set_ylabel('Density' if density else 'Number of entries', size=12)
    ax.set_title("{0} distribution by target".format(feature), size=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':12})
    plt.tight_layout()


def distplot_numerical_feature_before_and_after_transformation(df, feature_before, feature_after, target,
                                                               density=True, dropna=True, bins=50, figsize_x=14,
                                                               figsize_y=4):  # type: (pd.DataFrame, str, str, str, bool, bool, int, Any, Any) -> None
    """
    This method plots 2 density distributions of a numerical feature grouped by target: before and after
    applied transformation (such as log, sqrt, boxcox, etc.). This way one can see the effect of transformation.
    :param df: pandas DF containing train data set
    :param feature_before: name of feature to plot (before transformation)
    :param feature_after: name of feature to plot (after transformation)
    :param target: name of target column
    :param density: if True -> plot kde density estimator
    :param dropna: if True -> drop all nan
    :param bins: number of bins
    :param figsize_x: size of figure in x-direction
    :param figsize_y: size of figure in y-direction
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(figsize_x, figsize_y))
    for i, feat in enumerate([feature_before, feature_after]):
        axis=ax[i]
        for target_val, temp_df in df.groupby(target)[feat]:
            temp_df = temp_df.dropna().to_frame(name=feat) if dropna else temp_df.to_frame(name=feat)

            skew_val, _ = compute_skew_kurtosis(temp_df, feat)
            title_header = "{0} distribution by target".format(feat)
            title = title_header + '. Skew: {0:0.2f}'.format(float(skew_val)) if bool(skew_val) else title_header

            sns.distplot(temp_df, kde=density, bins=bins,
                         label=str(target_val), norm_hist=density, ax=axis)
            axis.set_xlabel('{0}'.format(feat), size=12)
            axis.set_ylabel('Density' if density else 'Number of entries', size=12)
            axis.set_title(title, size=13)
            axis.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':12})
    plt.tight_layout()