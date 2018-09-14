import os
import gc
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from generic_tools.utils import timing, create_output_dir
from sklearn.metrics import confusion_matrix, classification_report


class Predictor(object):
    SINGLE_MODEL_SOLUTION_DIR = 'single_model_solution'
    FIGNAME_CONFUSION_MATRIX = 'confusion_matrix.png'
    FIGNAME_CV_RESULTS_VERSUS_SEEDS = 'cv_results_vs_seeds.png'

    def __init__(self, train_df, test_df, target_column, index_column, classifier, eval_metric, metrics_scorer,
                 metrics_decimals=6, target_decimals=6, cols_to_exclude=[], num_folds=5, stratified=False,
                 kfolds_shuffle=True, cv_verbosity=1000, bagging=False, seeds_list=[27], predict_test=True,
                 output_dir=''):
        """
        This class run CV and makes OOF and submission predictions. It also allows to run CV in bagging mode using
        different seeds for random generator.
        :param train_df: pandas DF with train data set
        :param test_df: pandas DF with test data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
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
        :param target_decimals: rounding precision for target column
        :param cols_to_exclude: list of columns to exclude from modelling
        :param num_folds: number of folds to be used in CV
        :param stratified: if set True -> preserves the percentage of samples for each class in a fold
        :param kfolds_shuffle: if set True -> shuffle each stratification of the data before splitting into batches
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :param bagging: if set True -> run CV with different seeds and then average the results
        :param seeds_list: list of seeds to be used in CV (1 seed in list -> no bagging is possible)
        :param predict_test: IMPORTANT!! If False -> train model and predict OOF (i.e. validation only). Set True
                             if make a prediction for test data set
        :param output_dir: name of directory to save results of CV and prediction
        :return: out_of_fold predictions, submission predictions, oof_eval_results and feature_importance data frame
        """

        # Input data
        self.train_df = train_df  # type: pd.DataFrame
        self.test_df = test_df  # type: pd.DataFrame
        self.target_column = target_column  # type: str
        self.index_column = index_column  # type: str

        # Model data
        self.classifier = classifier
        self.model_name = classifier.get_model_name()  # type: str

        # Settings for CV and test prediction
        self.num_folds = num_folds  # type: int
        self.eval_metric = eval_metric  # type: str
        self.metrics_scorer = metrics_scorer  # type: metrics
        self.metrics_decimals = metrics_decimals  # type: int
        self.target_decimals = target_decimals  # type: int
        self.cols_to_exclude = cols_to_exclude  # type: list
        self.stratified = stratified  # type: bool
        self.kfolds_shuffle = kfolds_shuffle  # type: bool
        self.cv_verbosity = cv_verbosity  # type: int
        self.bagging = bagging  # type: bool
        self.seeds_list = seeds_list  # type: list
        self.predict_test = predict_test  # type: bool
        self.path_output_dir = os.path.join(os.getcwd(), self.SINGLE_MODEL_SOLUTION_DIR, output_dir)  # type: str
        create_output_dir(self.path_output_dir)

        # Results of CV and test prediction
        self.oof_preds = None  # type: pd.DataFrame
        self.sub_preds = None  # type: pd.DataFrame
        self.oof_eval_results = None  # type: pd.DataFrame
        self.bagged_oof_preds = None  # type: pd.DataFrame
        self.bagged_sub_preds = None  # type: pd.DataFrame
        self.feature_importance = None  # type: pd.DataFrame

    def _get_feature_importances_in_fold(self, feats, n_fold):  # type: (list, int) -> pd.DataFrame
        """
        This method prepares DF with the features importance per fold
        :param feats: list of features names
        :param n_fold: fold index
        :return: pandas DF with feature names and importances (in each considered fold)
        """
        features_names, features_importances = self.classifier.get_features_importance()

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features_names if features_names is not None else feats
        fold_importance_df["importance"] = features_importances
        fold_importance_df["fold"] = n_fold + 1
        return fold_importance_df.sort_values('importance', ascending=False)

    def _run_cv_one_seed(self, seed_val=27, predict_test=True, cv_verbosity=None):
        """
        This method run CV with the single seed. It is called from more global method: run_cv_and_prediction().
        :param seed_val: seeds to be used in CV
        :param predict_test: IMPORTANT!! If False -> train model and predict OOF (i.e. validation only). Set True
                             if make a prediction for test data set
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: out_of_fold predictions, submission predictions, oof_eval_results and feature_importance data frame
        """

        # This expression is needed to be able pass explicitly cv_verbosity=0 when running hyperparameters optimization
        cv_verbosity = cv_verbosity if cv_verbosity is not None else self.cv_verbosity

        target = self.target_column
        feats = [f for f in self.train_df.columns if f not in self.cols_to_exclude]
        feature_importance_df = pd.DataFrame()

        print('\nStarting CV with seed {}. Train shape: {}, test shape: {}\n'.format(
            seed_val, self.train_df[feats].shape, self.test_df[feats].shape))

        np.random.seed(seed_val)  # for reproducibility
        self.classifier.reset_models_seed(seed_val)

        if self.stratified:
            folds = StratifiedKFold(n_splits=self.num_folds, shuffle=self.kfolds_shuffle, random_state=seed_val)
        else:
            folds = KFold(n_splits=self.num_folds, shuffle=self.kfolds_shuffle, random_state=seed_val)

        # Create arrays and data frames to store results
        # Note: if predict_test is False -> sub_preds = None
        oof_preds = np.zeros(self.train_df.shape[0])
        sub_preds = np.zeros(self.test_df.shape[0]) if predict_test else None

        oof_eval_results = []
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(self.train_df[feats], self.train_df[target])):
            train_x, train_y = self.train_df[feats].iloc[train_idx], self.train_df[target].iloc[train_idx]
            valid_x, valid_y = self.train_df[feats].iloc[valid_idx], self.train_df[target].iloc[valid_idx]

            self.classifier.fit_model(train_x, train_y, valid_x, valid_y, eval_metric=self.eval_metric,
                                      cv_verbosity=cv_verbosity)
            best_iter_in_fold = self.classifier.get_best_iteration() if hasattr(
                self.classifier, 'get_best_iteration') else 1

            # Out-of-fold prediction
            oof_preds[valid_idx] = self.classifier.predict_probability(valid_x, best_iter_in_fold)

            # Make a prediction for test data (if flag is True)
            if predict_test:
                sub_preds += self.classifier.predict_probability(
                    self.test_df[feats], int(round(best_iter_in_fold * 1.1, 0))) / folds.n_splits

            if hasattr(self.classifier.estimator, 'feature_importances_'):
                # Get feature importances per fold and store corresponding dataframe to list
                fold_importance_df = self._get_feature_importances_in_fold(feats, n_fold)
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            else:
                # E.g. Logistic regression does not have feature importance attribute
                feature_importance_df = None

            oof_eval_result = round(self.metrics_scorer(valid_y, oof_preds[valid_idx]), self.metrics_decimals)
            oof_eval_results.append(oof_eval_result)
            print('CV: Fold {0} {1} : {2}\n'.format(n_fold + 1, self.eval_metric.upper(), oof_eval_result))

        # CV score and STD of CV score over folds for a given seed
        cv_score = round(self.metrics_scorer(self.train_df[target], oof_preds), self.metrics_decimals)
        cv_std = round(float(np.std(oof_eval_results)), self.metrics_decimals)
        print('CV: list of OOF {0} scores: {1}'.format(self.eval_metric.upper(), oof_eval_results))
        print('CV {0} score: {1} +/- {2}'.format(self.eval_metric.upper(), cv_score, cv_std))
        return oof_preds, sub_preds, oof_eval_results, feature_importance_df, cv_score, cv_std

    @timing
    def run_cv_and_prediction(self):
        """
        This method run CV and makes OOF and submission predictions. It also allows to run CV in bagging mode using
        different seeds for random generator.
        :return: out_of_fold predictions, submission predictions, oof_eval_results and feature_importance data frame
        """
        assert callable(self.metrics_scorer), 'metrics_scorer should be callable function'
        if 'sklearn.metrics' not in self.metrics_scorer.__module__:
            raise TypeError("metrics_scorer should be function from sklearn.metrics module. "
                            "Instead received {0}.".format(self.metrics_scorer.__module__))

        index = self.index_column  # index column
        target = self.target_column  # target column (column to be predicted)

        if self.bagging and len(self.seeds_list) == 1:
            raise ValueError('Number of seeds for bagging should be more than 1. Provided: {0}'
                             .format(len(self.seeds_list)))

        if self.bagging and len(self.seeds_list) > 1:
            oof_pred_bagged = []  # out-of-fold predictions for all seeds [dimension: n_rows_train x n_seeds]
            sub_preds_bagged = []  # test predictions for all seeds [dimension: n_rows_test x n_seeds]
            oof_eval_results_bagged = []  # CV scores in each fold for all seeds [dimension: n_seeds x n_folds]
            feature_importance_bagged = []  # features imp. (averaged over folds) for all seeds
            cv_score_bagged = []  # CV scores (averaged over folds) for all seeds [dimension: n_seeds]
            cv_std_bagged = []  # CV std's (computed over folds) for all seeds [dimension: n_seeds]
            for i, seed_val in enumerate(self.seeds_list):
                oof_preds, sub_preds, oof_eval_results, feature_importance_df, cv_score, cv_std = \
                    self._run_cv_one_seed(seed_val, self.predict_test)

                oof_pred_bagged.append(pd.Series(oof_preds, name='seed_%s' % str(i + 1)).round(self.target_decimals))
                sub_preds_bagged.append(pd.Series(sub_preds, name='seed_%s' % str(i + 1)).round(self.target_decimals))
                oof_eval_results_bagged.append(oof_eval_results)
                feature_importance_bagged.append(feature_importance_df)
                cv_score_bagged.append(cv_score)
                cv_std_bagged.append(cv_std)

            del oof_preds, sub_preds, oof_eval_results, feature_importance_df; gc.collect()

            # Preparing DF with OOF predictions for all seeds
            bagged_oof_preds = pd.concat(oof_pred_bagged, axis=1)
            bagged_oof_preds.insert(loc=0, column=index, value=self.train_df[index].values)
            self.bagged_oof_preds = bagged_oof_preds

            # Preparing DF with submission predictions for all seeds
            bagged_sub_preds = pd.concat(sub_preds_bagged, axis=1)
            bagged_sub_preds.insert(loc=0, column=index, value=self.test_df[index].values)
            self.bagged_sub_preds = bagged_sub_preds

            # Averaging results over seeds to compute single set of OOF predictions
            oof_preds = pd.DataFrame()
            oof_preds[index] = self.train_df[index].values
            oof_preds[target + '_OOF'] = bagged_oof_preds.loc[:, bagged_oof_preds.columns != index].mean(axis=1)\
                .round(self.target_decimals)
            self.oof_preds = oof_preds

            # Store predictions for test data (if flag is True). Use simple averaging over seeds (same as oof_preds)
            if self.predict_test:
                sub_preds = pd.DataFrame()
                sub_preds[index] = self.test_df[index].values
                sub_preds[target] = bagged_sub_preds.loc[:, bagged_sub_preds.columns != index].mean(axis=1)\
                    .round(self.target_decimals)
                self.sub_preds = sub_preds

            # Final stats: CV score and STD of CV score computed over all seeds
            cv_score = round(self.metrics_scorer(self.train_df[target], oof_preds[target + '_OOF']),
                             self.metrics_decimals)
            cv_std = round(float(np.std(cv_score_bagged)), self.metrics_decimals)
            print('\nCV: bagged {0} score {1} +/- {2}\n'.format(self.eval_metric.upper(), cv_score, cv_std))

            # The DF below contains seed number used in the CV run, cv_score averaged over all folds (see above),
            # std of CV score as well as list of CV values (in all folds).
            self.oof_eval_results = pd.DataFrame(
                zip(self.seeds_list, cv_score_bagged, cv_std_bagged, oof_eval_results_bagged),
                columns=['seed', 'cv_mean_score', 'cv_std', 'cv_score_per_each_fold']
            )
            self.feature_importance = pd.concat(feature_importance_bagged).reset_index(drop=True)
            del oof_pred_bagged, sub_preds_bagged; gc.collect()

        else:
            oof_preds, sub_preds, oof_eval_results, feature_importance_df, cv_score, cv_std = \
                self._run_cv_one_seed(self.seeds_list[0], self.predict_test)

            oof_preds_df = pd.DataFrame()
            oof_preds_df[index] = self.train_df[index].values
            oof_preds_df[target + '_OOF'] = np.round(oof_preds, self.target_decimals)
            self.oof_preds = oof_preds_df

            # Store predictions for test data (if flag is True)
            if self.predict_test:
                sub_preds_df = pd.DataFrame()
                sub_preds_df[index] = self.test_df[index].values
                sub_preds_df[target] = np.round(sub_preds, self.target_decimals)
                self.sub_preds = sub_preds_df

            # The DF below contains seed number used in the CV run, cv_score averaged over all folds (see above),
            # std of CV score as well as list of CV values (in all folds).
            self.oof_eval_results = pd.DataFrame([self.seeds_list[0], cv_score, cv_std, oof_eval_results],
                                                 index=['seed', 'cv_mean_score', 'cv_std', 'cv_score_per_each_fold']).T
            self.feature_importance = feature_importance_df
            del oof_preds, sub_preds; gc.collect()

    def plot_confusion_matrix(self, class_names, labels_mapper=None, normalize=False, cmap=plt.cm.Blues, save=False):
        """
        This function prints and plots the confusion matrix. Normalization can be applied by setting normalize=True.
        :param class_names: list of strings defining unique classes names
        :param labels_mapper:
        :param normalize: if set True -> normalizes results in confusion matrix (shows units instead of counting values)
        :param cmap: color map
        :param save: if True -> results will be saved to disk
        :return: plots confusion matrix and print classification report
        """
        true_labels = self.train_df[self.target_column].values.tolist()
        predicted_labels = map(labels_mapper, self.oof_preds[self.target_column + '_OOF'])
        cm = confusion_matrix(true_labels, predicted_labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print('\nNormalized confusion matrix')
        else:
            print('Confusion matrix, without normalization')

        print('{0}\n'.format(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Normalized confusion matrix' if normalize else 'Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names)

        fmt = '.4f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        print ('Classification report:\n{0}'.format(
            classification_report(true_labels, predicted_labels, target_names=class_names)))

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, self.FIGNAME_CONFUSION_MATRIX)
            print('\nSaving confusion matrix graph into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

    def show_features_importance(self, n_features=20, figsize_x=8, figsize_y=10, save=False):
        """
         This method plots features importance and saves the figure and the csv file to the disk.
        :param n_features: number of top most important features to show
        :param figsize_x: size of figure along X-axis
        :param figsize_y: size of figure along Y-axis
        :param save: if True -> results will be saved to disk
        :return: plot of features importance
        """
        features_importance_df = self.feature_importance.copy()
        cols = features_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_features].index
        best_features = features_importance_df.loc[features_importance_df.feature.isin(cols)].sort_values(
            by="importance", ascending=False)

        plt.figure(figsize=(figsize_x, figsize_y))
        sns.barplot(x="importance", y="feature", data=best_features)
        plt.title('{0} features (avg over folds)'.format(self.model_name.upper()))
        plt.tight_layout()

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_import']) + '.png')
            print('\nSaving features importance graph into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_import']) + '.csv')
            print('\nSaving {0} features into {1}'.format(self.model_name.upper(), full_path_to_file))
            features_importance_df = features_importance_df[["feature", "importance"]].groupby(
                "feature").mean().sort_values(by="importance", ascending=False).reset_index()
            features_importance_df.to_csv(full_path_to_file, index=False)
        del features_importance_df; gc.collect()

    def plot_cv_results_vs_used_seeds(self, figsize_x=14, figsize_y=4, annot_offset_x=3,
                                      annot_offset_y=5, annot_rotation=90, save=False):
        """
        This method plots CV results and corresponding std's for all seeds considered in the bagging process.
        The figure allows to see the effect of the seed number used in the model / k-fold split on the CV score
        :param figsize_x: size of figure in x-direction
        :param figsize_y: size of figure in y-direction
        :param annot_offset_x: offset (points) in x-direction for annotation text
        :param annot_offset_y: offset (points) in y-direction for annotation text
        :param annot_rotation: rotation angle of annotation text (0-horizontal, 90-vertical)
        :param save: if True -> results will be saved to disk
        :return: None
        """
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

        x = range(self.oof_eval_results.shape[0])
        y = self.oof_eval_results['cv_mean_score']
        yerr = self.oof_eval_results['cv_std']
        annotation = map(lambda n_seed: 'seed_%s' % n_seed, self.oof_eval_results['seed'].astype(str).values)

        # Plot CV score with std error bars
        ax.errorbar(x=x, y=y, yerr=yerr, fmt='-o', label=self.model_name)
        ax.set_title('CV {0} scores and corresponding stds for considered seeds'.format(self.eval_metric), size=13)
        ax.set_xlabel('Index of CV run', size=12)
        ax.set_ylabel('CV {0} score'.format(self.eval_metric), size=12)
        ax.set_xticks(x)

        # Add annotations with the seed number
        for xpos, ypos, name in zip(x, y, annotation):
            ax.annotate(name, (xpos, ypos), xytext=(annot_offset_x, annot_offset_y), va='bottom',
                        textcoords='offset points', rotation=annot_rotation)
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size': 12})

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, self.FIGNAME_CV_RESULTS_VERSUS_SEEDS)
            print('\nSaving CV results vs seeds graph into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

    def save_oof_results(self):
        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'OOF']) + '.csv')
        print('\nSaving elaborated OOF predictions into %s' % full_path_to_file)
        self.oof_preds.to_csv(full_path_to_file, index=False)

        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'CV']) + '.csv')
        print('\nSaving CV results into %s' % full_path_to_file)
        self.oof_eval_results.to_csv(full_path_to_file, index=False)

        if self.bagged_oof_preds is not None:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'bagged_OOF']) + '.csv')
            print('\nSaving OOF predictions for each seed into %s' % full_path_to_file)
            self.bagged_oof_preds.to_csv(full_path_to_file, index=False)

    def save_submission_results(self):
        if self.sub_preds is None:
            raise ValueError('Submission file is empty. Please set flag predict_test = True in run_cv_and_prediction() '
                             'to generate submission file.')
        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name]) + '.csv')
        print('\nSaving submission predictions into %s' % full_path_to_file)
        self.sub_preds.to_csv(full_path_to_file, index=False)

        if self.bagged_sub_preds is not None:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'bagged_SUBM']) + '.csv')
            print('\nSaving submission predictions for each seed into %s' % full_path_to_file)
            self.bagged_sub_preds.to_csv(full_path_to_file, index=False)


def prepare_lgbm():
    from lightgbm import LGBMClassifier
    from modeling.model_wrappers import LightGBMWrapper

    # LightGBM parameters
    lgbm_params = {}
    lgbm_params['boosting_type'] = 'gbdt'  # gbdt, gbrt, rf, random_forest, dart, goss
    lgbm_params['objective'] = 'binary'
    lgbm_params['num_leaves'] = 32  # 32
    lgbm_params['max_depth'] = 8  # 8
    lgbm_params['learning_rate'] = 0.02  # 0.01
    lgbm_params['n_estimators'] = 10000
    lgbm_params['early_stopping_rounds'] = 200
    lgbm_params['min_split_gain'] = 0.02  # 0.0222415
    lgbm_params['min_child_weight'] = 40  # 40
    lgbm_params['subsample'] = 0.87  # 0.8715623
    lgbm_params['colsample_bytree'] = 0.94  # 0.9497036
    lgbm_params['reg_alpha'] = 0.04  # 0.04
    lgbm_params['reg_lambda'] = 0.07  # 0.073
    lgbm_params['nthread'] = -1
    lgbm_params['verbose'] = -1
    lgbm_params['silent'] = True

    lgbm_wrapped = LightGBMWrapper(model=LGBMClassifier, params=lgbm_params, seed=27)
    return lgbm_wrapped


def prepare_xgb():
    from xgboost import XGBClassifier
    from modeling.model_wrappers import XGBWrapper

    # XGBoost parameters
    xgb_params = {}
    xgb_params['booster'] = 'gbtree'
    xgb_params['objective'] = 'binary:logistic'
    xgb_params['tree_method'] = 'hist'  # 'exact'
    xgb_params['max_depth'] = 6
    xgb_params['learning_rate'] = 0.02
    xgb_params['n_estimators'] = 10000
    xgb_params['early_stopping_rounds'] = 200
    xgb_params['min_child_weight'] = 30
    xgb_params['subsample'] = 0.8
    xgb_params['colsample_bytree'] = 0.7
    xgb_params['reg_alpha'] = 0.0
    xgb_params['reg_lambda'] = 1.2
    xgb_params['nthread'] = -1
    xgb_params['verbose'] = -1
    xgb_params['silent'] = True
    # xgb_params['colsample_bylevel'] = 0.632
    # xgb_params['scale_pos_weight']  = 2.5

    xgb_wrapped = XGBWrapper(model=XGBClassifier, params=xgb_params, seed=27)
    return xgb_wrapped


def main_run_cv_and_prediction(classifier='lgbm', debug=False):
    import warnings
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes
    warnings.filterwarnings("ignore")

    # Settings for debug
    num_rows = 2000

    # Input data
    path_to_data = r'C:\Kaggle\kaggle_home_credit_default_risk\feature_selection'
    full_path_to_file = os.path.join(path_to_data, 'train_dataset_lgbm_10.csv')
    data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
        .reset_index(drop=True)
    full_path_to_file = os.path.join(path_to_data, 'test_dataset_lgbm_10.csv')
    test_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
        .reset_index(drop=True)
    print('df_train shape: {0}'.format(data.shape))
    print('df_test shape: {0}'.format(test_data.shape))

    # Parameters
    solution_output_dir = ''
    target_column = 'TARGET'
    index_column = 'SK_ID_CURR'
    eval_metric = 'auc'
    metrics_scorer = roc_auc_score
    metrics_decimals = 4
    target_decimals = 2
    num_folds = 5
    stratified = True
    kfolds_shuffle = True
    cv_verbosity = 1000
    bagging = True
    predict_test = True
    seeds_list = [27, 999999, 2018, 516, 986]

    # Columns to exclude from input data
    cols_to_exclude = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']

    if classifier is 'lgbm':
        classifier_model = prepare_lgbm()
    elif classifier is 'xgb':
        classifier_model = prepare_xgb()
    else:
        classifier_model = prepare_lgbm()

    predictor = Predictor(
        train_df=data, test_df=test_data, target_column=target_column, index_column=index_column,
        classifier=classifier_model, eval_metric=eval_metric, metrics_scorer=metrics_scorer,
        metrics_decimals=metrics_decimals, target_decimals=target_decimals,
        cols_to_exclude=cols_to_exclude, num_folds=num_folds, stratified=stratified,
        kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity, bagging=bagging,
        predict_test=predict_test, seeds_list=seeds_list, output_dir=solution_output_dir
    )
    predictor.run_cv_and_prediction()


if __name__ == '__main__':
    # main_run_cv_and_prediction(classifier='lgbm', debug=True)
    main_run_cv_and_prediction(classifier='xgb', debug=True)
