import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import metrics
from generic_tools.utils import timer, timing, auto_selector_of_categorical_features
warnings.simplefilter('ignore', UserWarning)


class FeatureSelector(object):

    def __init__(self, train_df, target_column, index_column, cat_features, metrics_scorer, int_threshold, seed_value):
        self.train_df = train_df  # type: pd.DataFrame
        self.target_column = target_column  # type: str
        self.index_column = index_column  # type: dict
        self.cat_features = cat_features  # type: list
        self.metrics_scorer = metrics_scorer  # type: metrics
        self.int_threshold = int_threshold  # type: int
        self.seed_value = seed_value  # type: int
        self._verify_input_data_is_correct()
        np.random.seed(seed_value) # seed the numpy random generator

    def _verify_input_data_is_correct(self):
        assert callable(self.metrics_scorer), 'metrics_scorer should be callable function'
        if not 'sklearn.metrics' in self.metrics_scorer.__module__:
            raise TypeError("metrics_scorer should be function from sklearn.metrics module. " \
                            "Instead received {0}.".format(self.metrics_scorer.__module__))

        if self.cat_features is None:
            self.cat_features = auto_selector_of_categorical_features(
                self.train_df, cols_exclude=[self.target_column], int_threshold=self.int_threshold)
            print '{0} class was initialized with no cat_features. Auto-selector of categorical features is applied: ' \
                  '{1} features found'.format(self.__class__.__name__, len(self.cat_features))


class FeatureSelectorByTargetPermutation(FeatureSelector):

    def __init__(self, train_df, target_column, index_column, cat_features, lgbm_params, metrics_scorer,
                 int_threshold=9, seed_value=27):
        """
        This class adopts logic for selection of features based on target permutation. This selection process tests
        the actual importance significance against the distribution of features importance when fitted to noise
        (shuffled target). The original approach is described in the following paper:

                        https://academic.oup.com/bioinformatics/article/26/10/1340/193348

        :param train_df: pandas DF with train data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param cat_features: list of categorical features (is used in in lgbm.train)
        :param lgbm_params: parameters for LightGBM model
        :param metrics_scorer: from sklearn.metrics http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        :param int_threshold: this threshold is used to limit number of int8-type numerical features to be interpreted
                          as categorical (see auto_selector_of_categorical_features() method in utils.py)
        :param seed_value: seed numpy random generator
        """

        super(FeatureSelectorByTargetPermutation, self).__init__(train_df, target_column, index_column, cat_features,
                                                                 metrics_scorer, int_threshold, seed_value)
        self.lgbm_params = lgbm_params  # type: dict
        self.actual_imp_df = None  # type: pd.DataFrame # actual features importance
        self.null_imp_df = None  # type: pd.DataFrame # null-hypothesis features importance
        self.feature_scores = None  # type: pd.DataFrame # features importance gain/split scores

    def get_feature_importances(self, shuffle=False, num_boost_rounds=None):
        """
        This method trains LGBM model and constructs features importance DF. This method is used by both
        get_actual_importances_distribution() and get_null_importances_distribution().
        :param shuffle: whether to permute the target (true -> for Null Importance features distributions)
        :param num_boost_rounds: the number of iterations should be calculated from colsample_bytree and depth
                                 so that features are tested enough times for splits. In boruta.py, for example,
                                 they use : 100*(n_features/(np.sqrt(n_features)*depth))
        :return: pandas DF with features importance
        """

        train_features = [f for f in self.train_df if f not in [self.target_column, self.index_column]]

        if num_boost_rounds is None:
            # Check https://www.kaggle.com/ogrellier/feature-selection-with-null-importances/notebook

            # For XGB / LGBM algorithms
            # num_boost_rounds = \
            #     int(100. * self.lgbm_params['colsample_bytree'] * len(train_features) / self.lgbm_params['max_depth'])

            # For RF algorithm
            num_boost_rounds = \
                int(100. * (np.sqrt(len(train_features)) * self.lgbm_params['max_depth']))

        # Shuffle target if required
        y = self.train_df[self.target_column].copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            y = self.train_df[self.target_column].copy().sample(frac=1.0, random_state=self.seed_value)

        # Fit LightGBM in RF mode [it's way quicker than sklearn RandomForest]
        dtrain = lgbm.Dataset(self.train_df[train_features], y, free_raw_data=False, silent=True)

        # Fit the model
        clf = lgbm.train(params=self.lgbm_params, train_set=dtrain, num_boost_round=num_boost_rounds,
                         categorical_feature=self.cat_features)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['train_score'] = self.metrics_scorer(y, clf.predict(self.train_df[train_features]))
        return imp_df

    @timing
    def get_actual_importances_distribution(self):
        """
        This method fits the model on the original target and gathers the features importance. This gives us a
        benchmark whose significance can be tested against the Null Importance Distribution.
        :return: None
        """
        actual_imp_df = self.get_feature_importances(shuffle=False)
        self.actual_imp_df = actual_imp_df

    @timing
    def get_null_importances_distribution(self, nb_runs=80):
        """
        This method creates distributions of features Null Importance. It is achieved by fitting the model over
        several runs on a shuffled version of the target. This shows how the model can make sense of a feature
        irrespective of the target.
        :param nb_runs: number of runs to be performed on shuffled target
        :return: None
        """
        null_imp_df = []
        for i in range(1, nb_runs+1):
            with timer('%s: run %4d / %4d' % (self.get_null_importances_distribution.__name__, i, nb_runs+1)):
                imp_df = self.get_feature_importances(shuffle=True)
                imp_df['run'] = i
                null_imp_df.append(imp_df)
        self.null_imp_df = pd.concat(null_imp_df, axis=0)

    def score_features(self, percentile_null_dist=75):
        """
        This method makes scoring of the features by using log of mean actual feature importance divided by the 75
        percentile of null distribution. It worth to mention that there are several ways to score features, as for
        instance, the following:
            - Compute the number of samples in the actual importances that are away from the null importances
              recorded distribution.
            - Compute ratios like Actual / Null Max, Actual / Null Mean, Actual Mean / Null Max
        :param percentile_null_dist: percentile of null distribution features [must be between 0 and 100 inclusive]
        :return: None
        """
        feature_scores = {}
        for feat in self.actual_imp_df['feature'].unique():
            feature_scores[feat] = {}
            for importance in ['importance_split', 'importance_gain']:
                f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == feat, importance].values
                f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == feat, importance].mean()
                score = np.log(1e-10 + f_act_imps / (1 + np.percentile(f_null_imps, percentile_null_dist))) # avoids division by zero
                feature_scores[feat][importance.split('_')[1] + '_score'] = score
        feature_scores = pd.DataFrame(index=feature_scores.keys(), data=feature_scores.values()).reset_index()\
            .rename(columns={'index': 'feature'}).sort_values(by=['gain_score', 'split_score', 'feature'])
        self.feature_scores = feature_scores.reset_index(drop=True)

    def display_distributions(self, feature, figsize_x=14, figsize_y=6):
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

    def feature_score_comparing_to_importance(self, ntop_feats=100, figsize_x=16, figsize_y=16):
        """
        This method plots feature_score with respect to split/gain importance
        :param ntop_feats: number of features with highest feature_score (to be used in plot)
        :param figsize_x: size of figure in x-direction
        :param figsize_y: size of figure in y-direction
        :return: None
        """
        plt.figure(figsize=(figsize_x, figsize_y))
        gs = gridspec.GridSpec(1, 2)
        for i, importance in enumerate(['split_score', 'gain_score']):
            ax = plt.subplot(gs[0, i])
            sns.barplot(x=importance, y='feature', data=self.feature_scores.sort_values(importance, ascending=False)
                        .iloc[0:ntop_feats], ax=ax)
            ax.set_title('Feature scores wrt {0} importances'.format(importance.split('_')[0]), fontsize=14)
        plt.tight_layout()


class BorutaFeatureSelector(FeatureSelector):
    def __init__(self):
        super(BorutaFeatureSelector, self).__init__()


class SequentialFeatureSelector(FeatureSelector):
    def __init__(self):
        super(SequentialFeatureSelector, self).__init__()