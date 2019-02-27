import os
import gc
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import metrics
from builtins import zip, range
from collections import namedtuple
from generic_tools.loggers import configure_logging
from generic_tools.utils import timer, timing, auto_selector_of_categorical_features, create_output_dir

warnings.simplefilter('ignore', UserWarning)

configure_logging()
_logger = logging.getLogger("feature_selection")


class FeatureSelector(object):

    def __init__(self, train_df, target_column, index_column, cat_features, eval_metric, metrics_scorer,
                 metrics_decimals, num_folds, stratified, kfolds_shuffle, int_threshold,
                 seed_val, project_location, output_dirname):

        # Input data
        self.train_df = train_df  # type: pd.DataFrame
        self.target_column = target_column  # type: str
        self.index_column = index_column  # type: dict
        self.cat_features = cat_features  # type: list
        self.int_threshold = int_threshold  # type: int

        # Settings for CV
        self.num_folds = num_folds  # type: int
        self.eval_metric = eval_metric  # type: str
        self.metrics_scorer = metrics_scorer  # type: metrics
        self.stratified = stratified  # type: bool
        self.kfolds_shuffle = kfolds_shuffle  # type: bool
        self.metrics_decimals = metrics_decimals  # type: int
        self.seed_val = seed_val  # type: int
        self.path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))
        create_output_dir(self.path_output_dir)

        self._verify_input_data_is_correct()
        np.random.seed(seed_val)  # seed the numpy random generator

    def _verify_input_data_is_correct(self):
        assert callable(self.metrics_scorer), 'metrics_scorer should be callable function'
        if 'sklearn.metrics' not in self.metrics_scorer.__module__:
            raise TypeError("metrics_scorer should be function from sklearn.metrics module. "
                            "Instead received {0}.".format(self.metrics_scorer.__module__))

        if self.cat_features is None:
            self.cat_features = auto_selector_of_categorical_features(
                self.train_df, cols_exclude=[self.target_column], int_threshold=self.int_threshold)
            _logger.info('{0} class was initialized with no cat_features. Auto-selector of categorical features '
                         'is applied: {1} features found'.format(self.__class__.__name__, len(self.cat_features)))


class FeatureSelectorByTargetPermutation(FeatureSelector):
    FEATURE_SELECTION_METHOD = 'target_permutation'
    FIGNAME_FEATS_SCORE_VS_IMPORTANCE = 'feats_score_vs_importance.png'
    FIGNAME_CV_VERSUS_FEATURES_SCORE_THRESHOLD = 'cv_vs_feats_score_thresh.png'
    FILENAME_CV_RESULTS_VS_FEATS_SCORE_THRESH = 'cv_results_vs_feats_score_thresh.csv'
    FILENAME_FEATS_SCORE = 'feats_score.csv'

    def __init__(self, train_df, target_column, index_column, cat_features, lgbm_params_feats_exploration,
                 lgbm_params_feats_selection, eval_metric, metrics_scorer, metrics_decimals=6, num_folds=5,
                 stratified=False, kfolds_shuffle=True, int_threshold=9, seed_val=27, project_location='',
                 output_dirname=''):
        """
        This class adopts logic for selection of features based on target permutation. This selection process tests
        the actual importance significance against the distribution of features importance when fitted to noise
        (shuffled target). The original approach is described in the following paper:

                        https://academic.oup.com/bioinformatics/article/26/10/1340/193348

        :param train_df: pandas DF with train data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param cat_features: list of categorical features (is used in in lgbm.train)
        :param lgbm_params_feats_exploration: parameters for LightGBM model to be used for feature exploration
        :param lgbm_params_feats_selection: parameters for LightGBM model to be used for feature selection
        :param eval_metric: 'rmse': root mean square error
                            'mae': mean absolute error
                            'logloss': negative log-likelihood
                            'error': Binary classification error rate
                            'error@t': a different than 0.5 binary classification threshold value could be specified
                            'merror': Multiclass classification error rate
                            'mlogloss': Multiclass logloss
                            'auc': Area under the curve
                            'map': Mean average precision
                            ... others
        :param metrics_scorer: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        :param metrics_decimals: rounding precision for evaluation metrics (e.g. for CV printouts)
        :param num_folds: number of folds to be used in CV
        :param stratified: if set True -> preserves the percentage of samples for each class in a fold
        :param kfolds_shuffle: if set True -> shuffle each stratification of the data before splitting into batches
        :param int_threshold: this threshold is used to limit number of int8-type numerical features to be interpreted
                          as categorical (see auto_selector_of_categorical_features() method in utils.py)
        :param seed_val: seed numpy random generator
        :param output_dirname: name of directory to save results of feature selection process
        """

        super(FeatureSelectorByTargetPermutation, self).__init__(
            train_df, target_column, index_column, cat_features, eval_metric, metrics_scorer, metrics_decimals,
            num_folds, stratified, kfolds_shuffle, int_threshold, seed_val, project_location, output_dirname
        )

        # Dicts with LGB model parameters to be used in feature exploration and selection stages, correspondingly
        self.lgbm_params_feats_exploration = lgbm_params_feats_exploration  # type: dict
        self.lgbm_params_feats_selection = lgbm_params_feats_selection  # type: dict

        self.actual_imp_df = None  # type: pd.DataFrame # actual features importance
        self.null_imp_df = None  # type: pd.DataFrame # null-hypothesis features importance
        self.features_scores_df = None  # type: pd.DataFrame # features importance gain/split scores
        self.cv_results_vs_thresh_df = None  # type: pd.DataFrame # impact of feature selection on CV score
        self.best_thresh_df = None  # type: pd.DataFrame # DF with ranks of CV scores and stds to select best threshold

    def get_feature_importances(self, shuffle=False, num_boost_rounds=None, verbose=False):
        """
        This method trains LGBM model and constructs features importance DF. This method is used by both
        get_actual_importances_distribution() and get_null_importances_distribution().
        :param shuffle: whether to permute the target (true -> for Null Importance features distributions)
        :param num_boost_rounds: the number of iterations should be calculated from colsample_bytree and depth
                                 so that features are tested enough times for splits. In boruta.py, for example,
                                 authors use (RF algo): 100.*(n_features/(np.sqrt(n_features)*depth)).
                                 For XGB/LGBM algorithms: 100. * colsample_bytree * n_features / depth
        :param verbose: if True -> make printouts
        :return: pandas DF with features importance
        """

        train_features = [f for f in self.train_df if f not in [self.target_column, self.index_column]]

        if num_boost_rounds is None:
            # Check https://www.kaggle.com/ogrellier/feature-selection-with-null-importances/notebook
            # For RF algorithm (or LGBM in RF mode)
            num_boost_rounds = int(100. * (np.sqrt(len(train_features)) /
                                           self.lgbm_params_feats_exploration['max_depth']))
        # Shuffle target if required
        y = self.train_df[self.target_column].copy()
        if shuffle:
            # Here one could use e.g. a binomial distribution
            y = self.train_df[self.target_column].copy().sample(frac=1.0)

        # Fit LightGBM in RF mode [it's way quicker than sklearn RandomForest]
        dtrain = lgbm.Dataset(data=self.train_df[train_features], label=y, free_raw_data=False,
                              categorical_feature=self.cat_features)
        if verbose:
            _logger.info(
                'Train LGBM on {0} data set with {1} categorical features. LGBM parameters: {2}. Number of boosting '
                'rounds: {3}'.format(self.train_df[train_features].shape, self.cat_features,
                                     self.lgbm_params_feats_exploration, num_boost_rounds))

        # Fit the model
        clf = lgbm.train(params=self.lgbm_params_feats_exploration, train_set=dtrain,
                         num_boost_round=num_boost_rounds)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['train_score'] = self.metrics_scorer(y, clf.predict(self.train_df[train_features]))
        return imp_df

    @timing
    def get_actual_importances_distribution(self, num_boost_rounds=None):
        """
        This method fits the model on the original target and gathers the features importance. This gives us a
        benchmark whose significance can be tested against the Null Importance Distribution.
        :param num_boost_rounds: number of boosting iterations for lgbm train. If None -> will be eval using formula
        :return: None
        """
        actual_imp_df = self.get_feature_importances(shuffle=False, num_boost_rounds=num_boost_rounds)
        self.actual_imp_df = actual_imp_df

    @timing
    def get_null_importances_distribution(self, nb_runs=80, num_boost_rounds=None):
        """
        This method creates distributions of features Null Importance. It is achieved by fitting the model over
        several runs on a shuffled version of the target. This shows how the model can make sense of a feature
        irrespective of the target.
        :param nb_runs: number of runs to be performed on shuffled target
        :param num_boost_rounds: number of boosting iterations for lgbm train. If None -> will be eval using formula
        :return: None
        """
        null_imp_df = []
        for i in range(1, nb_runs + 1):
            with timer('%s: run %4d / %4d' % (self.get_null_importances_distribution.__name__, i, nb_runs)):
                imp_df = self.get_feature_importances(shuffle=True, num_boost_rounds=num_boost_rounds)
                imp_df['run'] = i
                null_imp_df.append(imp_df)
        self.null_imp_df = pd.concat(null_imp_df, axis=0)

    def score_features(self, scoring_function=None):
        """
        This method makes scoring of the features by using provided scoring_function. It worth to mention that there
        are several ways to score features, as for instance, the following:
            - Compute the number of samples in the actual importances that are away from the null importances
              recorded distribution.
            - Compute ratios like Actual / Null Max, Actual / Null Mean, Actual Mean / Null Max
        :return: None
        """

        if scoring_function is not None:
            # Verify is callable and has 2 arguments
            assert callable(scoring_function), \
                'Provided scoring_function is not callable. It has %s type' % type(scoring_function)
            assert scoring_function.__code__.co_argcount == 2, \
                'Provided scoring_function should have 2 arguments. Instead received %d: %s' % (
                    scoring_function.__code__.co_argcount, str(scoring_function.__code__.co_varnames))
        else:
            # If scoring_function is not defined, make score of the features by using log of mean actual feature
            # importance divided by the 75 percentile of null distribution. 1e-10 is used to avoid division by zero.
            scoring_function = \
                lambda f_act_imps, f_null_imps: np.log(1e-10 + f_act_imps / (1 + np.percentile(f_null_imps, 75)))

        feature_scores = {}
        for feat in self.actual_imp_df['feature'].unique():
            feature_scores[feat] = {}
            for importance in ['importance_split', 'importance_gain']:
                f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == feat, importance].values
                f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == feat, importance].mean()
                score = scoring_function(f_act_imps, f_null_imps)
                feature_scores[feat][importance.split('_')[1] + '_score'] = score
        feature_scores = pd.DataFrame.from_dict(feature_scores, orient='index').reset_index()\
            .rename(columns={'index': 'feature'}).sort_values(by=['gain_score', 'split_score', 'feature'])
        self.features_scores_df = feature_scores.reset_index(drop=True)

    def run_lgbm_cv(self, train_features):  # type: (list) -> tuple
        """
        This method runs LGBM's in-built CV to investigate the effect of used feature_score threshold (i.e. number of
        features) on CV score.
        :param train_features: list of features to be used for training
        :return: tuple containing cv_bst_round, cv_bst_score, cv_std_bst_score, n_features
        """
        train_features = [f for f in train_features if f not in [self.target_column, self.index_column]]
        cat_features = list(set(train_features).intersection(set(self.cat_features)))
        dtrain = lgbm.Dataset(data=self.train_df[train_features], label=self.train_df[self.target_column],
                              free_raw_data=False, categorical_feature=cat_features)

        # Run in-built LightGBM cross-validation
        cv_results = lgbm.cv(
            params=self.lgbm_params_feats_selection,
            train_set=dtrain,
            nfold=self.num_folds,
            stratified=self.stratified,
            shuffle=self.kfolds_shuffle,
            seed=self.seed_val
        )

        # Best CV round: mean and std over folds of CV score
        cv_bst_round = np.argmax(cv_results['%s-mean' % self.eval_metric])
        cv_bst_score = round(cv_results['%s-mean' % self.eval_metric][cv_bst_round], self.metrics_decimals)
        cv_std_bst_score = round(cv_results['%s-stdv' % self.eval_metric][cv_bst_round], self.metrics_decimals)

        return cv_bst_round, cv_bst_score, cv_std_bst_score

    @timing
    def eval_feats_removal_impact_on_cv_score(self, thresholds=[], n_thresholds=5):
        """
        This method runs LGBM CV for each feature_score threshold supplied as argument. It allows to see the effect of
        number of the selected features on the CV. The idea is to select optimum feature_score threshold / CV value.
        Feature score can be interpreted as feature importance.
        :param thresholds: list of thresholds with the values in between min and max of the feature_scores
        :param n_thresholds: if thresholds=[], the number of n_thresholds is used to split the range between min and
                             max feature_scores in equal parts (e.g. min=0, max=10, n_thresholds=2 -> thresholds=[0, 5])
        :return: pandas DF with 'threshold', 'importance' CV scores as a function of feature_score threshold
        """

        if not len(thresholds):
            min_score = int(round(self.features_scores_df.describe().loc['min'].min(), 0))
            max_score = int(round(self.features_scores_df.describe().loc['max'].max(), 0))
            assert min_score < max_score, \
                'min score in self.features_scores_df DF [{0}] should be smaller than max score [{1}]. Impossible to ' \
                'construct threshold list.'.format(min_score, max_score)
            step = int((max_score - min_score) / n_thresholds)
            thresholds = range(min_score, max_score, step)

        eval_score = namedtuple('eval_score', ['cv_bst_round', 'cv_bst_score', 'cv_std_bst_score', 'n_features'])
        cv_results = []
        for importance in ['split_score', 'gain_score']:
            importance_type = importance.split('_')[0].upper()
            temp = []
            for i, threshold in enumerate(thresholds):
                with timer(
                        '%s: %s %4d / %4d. Threshold: %3d' % (importance_type, self.run_lgbm_cv.__name__, i + 1,
                                                                len(thresholds), threshold)):
                    train_features = self.get_list_of_features(importance=importance, thresh=threshold)
                    result = self.run_lgbm_cv(train_features)
                    result = eval_score(*(result + (int(len(train_features)),)))
                    temp.append(result)

                _logger.info('  Number of features with score >= %d: %d' % (threshold, len(train_features)))
                _logger.info('  Optimum boost rounds: {}'.format(result.cv_bst_round))
                _logger.info('  Best iteration CV: {0} +/- {1}'.format(result.cv_bst_score, result.cv_std_bst_score))

            temp = pd.DataFrame(temp, columns=eval_score._fields)
            temp.insert(loc=0, column='importance', value=importance_type)  # SPLIT - GAIN
            temp.insert(loc=1, column='threshold', value=thresholds)  # threshold value
            cv_results.append(temp)

        cv_results_vs_thresh_df = pd.concat(cv_results, axis=0, ignore_index=True)
        cv_results_vs_thresh_df.sort_values(by='threshold', inplace=True)
        cv_results_vs_thresh_df.set_index(['threshold', 'importance'], drop=True, inplace=True)
        self.cv_results_vs_thresh_df = cv_results_vs_thresh_df
        del cv_results, temp; gc.collect()

    def get_best_threshold(self, importance='gain_score', cv_asc_rank=True, cv_std_asc_rank=False):
        """
        This method finds threshold for features score that gives best CV score and std error. It is based on ranking.
        :param importance: LGBM feature importance type ('gain_score' or 'split_score')
        :param cv_asc_rank: if True -> higher CV gives higher rank
        :param cv_std_asc_rank: if False > lower std error gives higher rank
        :return: best threshold value
        """
        if importance not in ('gain_score', 'split_score'):
            raise (ValueError, "Importance type should be either 'gain_score' or 'split_score'. "
                               "Instead received {0}".format(importance))

        best_thresh_df = self.cv_results_vs_thresh_df.loc[pd.IndexSlice[:, [importance.split('_')[0].upper()]], :].copy()
        best_thresh_df['cv_bst_score_rank'] = best_thresh_df['cv_bst_score'].rank(method='min', ascending=cv_asc_rank)
        best_thresh_df['cv_std_bst_score_rank'] = best_thresh_df['cv_std_bst_score'].rank(method='min',
                                                                                          ascending=cv_std_asc_rank)
        best_thresh_df['total_rank'] = best_thresh_df['cv_bst_score_rank'] + best_thresh_df['cv_std_bst_score_rank']
        best_thresh_idx = best_thresh_df['total_rank'].argmax()
        best_thresh_results = best_thresh_df.loc[best_thresh_idx, ['cv_bst_score', 'cv_std_bst_score',
                                                                   'n_features']].to_dict()
        _logger.info('Best threshold by {0} is {1}. {2} score: {3} +/- {4}. Total number of features: {5}'.format(
            best_thresh_idx[1], best_thresh_idx[0], self.eval_metric.upper(), best_thresh_results['cv_bst_score'],
            best_thresh_results['cv_std_bst_score'], best_thresh_results['n_features']))

        self.best_thresh_df = best_thresh_df[['cv_bst_score', 'cv_std_bst_score', 'cv_bst_score_rank',
                                              'cv_std_bst_score_rank', 'total_rank']]
        return best_thresh_idx[0]

    def get_list_of_features(self, importance='gain_score', thresh=0):  # type: (str, (float, int)) -> list
        """
        This method returns list of features for given importance type ('gain_score' or 'split_score') with the
        feature_score >= threshold
        :param importance: LGBM feature importance type ('gain_score' or 'split_score')
        :param thresh: limit value for min feature_score to be included into list of features
        :return: list of features for given importance type with the feature_score >= threshold
        """
        if importance not in ('gain_score', 'split_score'):
            raise (ValueError, "Importance type should be either 'gain_score' or 'split_score'. "
                               "Instead received {0}".format(importance))

        list_of_features = list(self.features_scores_df.loc[self.features_scores_df[importance] >= thresh,
                                                            'feature']) + [self.target_column, self.index_column]
        return list_of_features

    def display_distributions(self, feature, figsize_x=14, figsize_y=6, save=False):
        """
        This method plots the actual vs null-importance distributions for the given feature. Comparison of actual
        importance to the mean and max of the null importance can be considered as indicator of the features importance
        that, eventually, allows to see a major features in the data set. It also worth to mention that:
            - Any feature's sufficient variance can be used and made sense of by tree models. One can always
              find splits that help scoring better.
            - Correlated features have decaying importance once one of them is used by the model. The chosen feature
              will have strong importance and its correlated suite will have decaying importance
        :param feature: name of the feature to plot the actual vs null-importance distributions
        :param figsize_x: size of figure in x-direction
        :param figsize_y: size of figure in y-direction
        :param save: if True -> results will be saved to disk
        :return: None
        """
        plt.figure(figsize=(figsize_x, figsize_y))
        gs = gridspec.GridSpec(1, 2)
        for i, importance in enumerate(['importance_split', 'importance_gain']):
            ax = plt.subplot(gs[0, i])
            a = ax.hist(self.null_imp_df.loc[self.null_imp_df['feature'] == feature, importance].values,
                        label='Null importances')
            ax.vlines(x=self.actual_imp_df.loc[self.actual_imp_df['feature'] == feature, importance].mean(),
                      ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
            ax.legend()
            ax.set_title('{0} importance of {1}'.format(importance.split('_')[1].upper(), feature.upper()))
            plt.xlabel('{0}: null importance distribution for {1}'.format(
                importance.split('_')[1].upper(), feature.upper()))
        plt.tight_layout()

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, "actual_vs_null_import_distrib_"
                                                                   "{0}".format(feature.lower()))
            _logger.info('Saving {0} null importance distribution vs actual distribution figure '
                         'into {1}'.format(feature, full_path_to_file))
            plt.savefig(full_path_to_file)

    def feature_score_comparing_to_importance(self, ntop_feats=100, figsize_x=16, figsize_y=16, save=False):
        """
        This method plots feature_score with respect to split/gain importance
        :param ntop_feats: number of features with highest feature_score (to be used in plot)
        :param figsize_x: size of figure in x-direction
        :param figsize_y: size of figure in y-direction
        :param save: if True -> results will be saved to disk
        :return: None
        """
        plt.figure(figsize=(figsize_x, figsize_y))
        gs = gridspec.GridSpec(1, 2)
        for i, importance in enumerate(['split_score', 'gain_score']):
            ax = plt.subplot(gs[0, i])
            sns.barplot(x=importance, y='feature',
                        data=self.features_scores_df.sort_values(importance, ascending=False)
                        .iloc[0:ntop_feats], ax=ax)
            if i == 0:
                ax.set_title('Feature scores wrt {0} importances'.format(importance.split('_')[0]), fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(right=0.95)

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, self.FIGNAME_FEATS_SCORE_VS_IMPORTANCE)
            _logger.info('Saving {0} top features score vs importance figure '
                         'into {1}'.format(ntop_feats, full_path_to_file))
            plt.savefig(full_path_to_file)

    def plot_cv_results_vs_feature_threshold(self, figsize_x=14, figsize_y=4, annot_offset_x=3,
                                             annot_offset_y=5, annot_rotation=90, save=False):
        """
        This method plots CV results against feature score threshold. It allows to find optimum threshold for feature
        score (i.e. number of features) that leads to best CV score.
        :param figsize_x: size of figure in x-direction
        :param figsize_y: size of figure in y-direction
        :param annot_offset_x: offset (points) in x-direction for annotation text
        :param annot_offset_y: offset (points) in y-direction for annotation text
        :param annot_rotation: rotation angle of annotation text (0-horizontal, 90-vertical)
        :param save: if True -> results will be saved to disk
        :return: None
        """
        df = self.cv_results_vs_thresh_df.copy()
        df.reset_index(inplace=True)
        fig, ax = plt.subplots(1, 2, figsize=(figsize_x, figsize_y))
        i = 0
        for importance, df_slice in df.groupby('importance'):
            x = df_slice['threshold']
            y = df_slice['cv_bst_score']
            yerr = df_slice['cv_std_bst_score']
            annotation = list(df_slice['n_features'].astype(str).values)

            # Plot CV score with std error bars
            ax[i].errorbar(x=x, y=y, yerr=yerr, fmt='-o', label=importance)
            ax[i].set_title('{0}: CV {1} score vs feature score threshold'.format(importance,
                                                                                  self.eval_metric), size=13)
            ax[i].set_xlabel('Feature score threshold', size=12)
            ax[i].set_ylabel('CV {0} score'.format(self.eval_metric), size=12)

            # Add annotations with the number of features used per each threshold
            for xpos, ypos, name in list(zip(x, y, annotation)):
                ax[i].annotate(name, (xpos, ypos), xytext=(annot_offset_x, annot_offset_y), va='bottom',
                               textcoords='offset points', rotation=annot_rotation)
            i += 1

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, self.FIGNAME_CV_VERSUS_FEATURES_SCORE_THRESHOLD)
            _logger.info('Saving CV results vs feature score threshold figure into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

        del df; gc.collect()

    def save_features_scores(self):
        """
        This method persists features scores DF to the disk
        :return: None
        """
        full_path_to_file = os.path.join(self.path_output_dir, self.FILENAME_FEATS_SCORE)
        _logger.info('Saving feature scores DF into %s' % full_path_to_file)
        self.features_scores_df.to_csv(full_path_to_file, index=False)

    def save_cv_results_vs_feats_score_thresh(self):
        """
        This method persists DF with CV results at various features scores thresholds to the disk
        :return: None
        """
        full_path_to_file = os.path.join(self.path_output_dir, self.FILENAME_CV_RESULTS_VS_FEATS_SCORE_THRESH)
        _logger.info('Saving DF with CV results at various features scores thresholds into %s' % full_path_to_file)
        self.cv_results_vs_thresh_df.reset_index().to_csv(full_path_to_file, index=False)


class BorutaFeatureSelector(FeatureSelector):
    FEATURE_SELECTION_METHOD = 'boruta'

    def __init__(self):
        super(BorutaFeatureSelector, self).__init__()


class SequentialFeatureSelector(FeatureSelector):
    FEATURE_SELECTION_METHOD = 'sequential'

    def __init__(self):
        super(SequentialFeatureSelector, self).__init__()


def load_feature_selector_class(feature_selector_method):
    if feature_selector_method == 'target_permutation':
        return FeatureSelectorByTargetPermutation
    else:
        raise NotImplemented()


def main_feat_selector_by_target_permutation():
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes

    project_location = 'C:\Kaggle\example'
    output_dirname = 'fts_001'  # where the results of feature selection will be stored

    # Input data
    path_to_data = r'C:\Kaggle\kaggle_home_credit_default_risk\feature_selection'
    full_path_to_file = os.path.join(path_to_data, 'train_dataset_lgbm_10.csv')

    # Read input data
    data = downcast_datatypes(pd.read_csv(full_path_to_file)).reset_index(drop=True)
    print('df_train shape: {0}'.format(data.shape))

    # Parameters
    target_column = 'TARGET'
    index_column = 'SK_ID_CURR'
    eval_metric = 'auc'
    metrics_scorer = roc_auc_score
    metrics_decimals = 4
    num_folds = 5
    stratified = True
    kfolds_shuffle = True
    int_threshold = 9
    seed_val = 27

    feat_cols = [f for f in data.columns if f not in [target_column, index_column]]
    categorical_feats = [f for f in feat_cols if data[f].dtype == 'object']
    print('Number of categorical features: {0}'.format(len(categorical_feats)))

    if len(categorical_feats):
        for f_ in categorical_feats:
            data[f_], _ = pd.factorize(data[f_])
            data[f_] = data[f_].astype('category')
        cat_features = categorical_feats
    else:
        cat_features = 'auto'  # (if None -> will be detected automatically)

    lgbm_params_feats_exploration = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'learning_rate': 0.1,
        'num_leaves': 127,
        'max_depth': 8,
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'bagging_freq': 1,
        'metric': eval_metric,
        'n_jobs': -1,
        'silent': True
    }

    lgbm_params_feats_selection = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'n_estimators': 2000,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_split_gain': 0.00001,
        'reg_alpha': 0.00001,
        'reg_lambda': 0.00001,
        'metric': eval_metric,
        'n_jobs': -1,
        'silent': True
    }

    # Initialization of FeatureSelectorByTargetPermutation
    feat_select_target_perm = \
        FeatureSelectorByTargetPermutation(train_df=data,
                                           target_column=target_column, index_column=index_column,
                                           cat_features=cat_features, int_threshold=int_threshold,
                                           lgbm_params_feats_exploration=lgbm_params_feats_exploration,
                                           lgbm_params_feats_selection=lgbm_params_feats_selection,
                                           eval_metric=eval_metric, metrics_scorer=metrics_scorer,
                                           metrics_decimals=metrics_decimals, num_folds=num_folds,
                                           stratified=stratified, kfolds_shuffle=kfolds_shuffle,
                                           seed_val=seed_val, project_location=project_location,
                                           output_dirname=output_dirname)

    feat_select_target_perm.get_actual_importances_distribution(num_boost_rounds=350)
    print(feat_select_target_perm.actual_imp_df.head())

    feat_select_target_perm.get_null_importances_distribution(nb_runs=5, num_boost_rounds=350)
    print(feat_select_target_perm.null_imp_df.head())

    # This one is used for selection of features in CV manner (using threshold)
    func = lambda f_act_imps, f_null_imps: 100. * (
            f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size

    feat_select_target_perm.score_features(func)
    print(feat_select_target_perm.features_scores_df.head())

    thresholds = [0, 30, 60, 80]
    feat_select_target_perm.eval_feats_removal_impact_on_cv_score(thresholds=thresholds, n_thresholds=5)
    print(feat_select_target_perm.cv_results_vs_thresh_df.head())


if __name__ == '__main__':
    main_feat_selector_by_target_permutation()
