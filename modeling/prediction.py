import os
import gc
import shap
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from model_wrappers import ModelWrapper
from sklearn.model_selection import KFold, StratifiedKFold
from generic_tools.utils import timing, create_output_dir
from sklearn.metrics import confusion_matrix, classification_report


class Predictor(object):
    SINGLE_MODEL_SOLUTION_DIR = 'single_model_solution'
    FIGNAME_CONFUSION_MATRIX = 'confusion_matrix.png'
    FIGNAME_CV_RESULTS_VERSUS_SEEDS = 'cv_results_vs_seeds.png'

    def __init__(self, train_df, test_df, target_column, index_column, classifier, predict_probability,
                 eval_metric, metrics_scorer, metrics_decimals=6, target_decimals=6, cols_to_exclude=[],
                 num_folds=5, stratified=False, kfolds_shuffle=True, cv_verbosity=1000, bagging=False,
                 seeds_list=[27], predict_test=True, output_dir=''):
        """
        This class run CV and makes OOF and submission predictions. It also allows to run CV in bagging mode using
        different seeds for random generator.
        :param train_df: pandas DF with train data set
        :param test_df: pandas DF with test data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param classifier: wrapped estimator (object of ModelWrapper class)
        :param predict_probability: if True -> use classifier.predict_proba(), else -> classifier.predict() method
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
        :param stratified: if True -> preserves the percentage of samples for each class in a fold
        :param kfolds_shuffle: if True -> shuffle each stratification of the data before splitting into batches
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :param bagging: if True -> run CV with different seeds and then average the results
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
        self.classifier = classifier  # type: ModelWrapper
        self.model_name = classifier.get_model_name()  # type: str
        self.predict_probability = predict_probability  # type: bool

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
        self.path_output_dir = os.path.normpath(os.path.join(os.getcwd(), self.SINGLE_MODEL_SOLUTION_DIR, output_dir))
        create_output_dir(self.path_output_dir)

        # Results of CV and test prediction
        self.cv_score = None  # type: float
        self.cv_std = None  # type: float
        self.oof_preds = None  # type: pd.DataFrame
        self.sub_preds = None  # type: pd.DataFrame
        self.oof_eval_results = None  # type: pd.DataFrame
        self.bagged_oof_preds = None  # type: pd.DataFrame
        self.bagged_sub_preds = None  # type: pd.DataFrame
        self.feature_importance = None  # type: pd.DataFrame
        self.shap_values = None  # type: pd.DataFrame

    def _index_column_is_defined(self):  # type: (None) -> bool
        """
        This method returns True if index column is defined in the Predictor class and is not equal to ''.
        Note that index column is frequently used when preparing out-of-fold and test predictions.
        :return: True or False
        """
        return self.index_column is not None and self.index_column != ''

    def _concat_bagged_results(self, df_bagged):  # type: (list) -> pd.DataFrame
        """
        This method concatenates pandas DFs which contain either out-of-fold or test prediction results.
        :param df_bagged: list of pandas DFs
        :return: single pandas DF
        """
        df = pd.concat(df_bagged, axis=1)
        if self._index_column_is_defined():
            index_values = self.train_df[self.index_column].values if self.train_df.shape[0] == df.shape[0] \
                else self.test_df[self.index_column].values
            df.insert(loc=0, column=self.index_column, value=index_values)
        return df

    def _average_bagged_results(self, bagged_df, oof_prediction):  # type: (pd.DataFrame, bool) -> pd.DataFrame
        """
        This method creates single pandas DF containing average of either out-of-fold or test prediction results that
        were obtained using different seeds (via bagging process).
        :param bagged_df: pandas DF with bagged predictions (either OOF or test), see self._concat_bagged_results()
        :param oof_prediction: if True -> the provided bagged_df contains OOF results, otherwise -> test predictions
        :return: pandas DF with averaged predictions over different seeds
        """
        df = pd.DataFrame()
        target_col = self.target_column + '_OOF' if oof_prediction else self.target_column

        if self._index_column_is_defined():
            index_values = self.train_df[self.index_column].values if oof_prediction \
                else self.test_df[self.index_column].values
            df[self.index_column] = index_values
            df[target_col] = bagged_df.loc[:, bagged_df.columns != self.index_column]\
                .mean(axis=1).round(self.target_decimals)
        else:
            df[target_col] = bagged_df.mean(axis=1).round(self.target_decimals)

        # Convert to int if target rounding precision is 0 decimals
        if self.target_decimals == 0:
            df[target_col] = df[target_col].astype(int)
        return df

    def _prepare_single_seed_results(self, preds, oof_prediction):  # type: (np.ndarray, bool) -> pd.DataFrame
        """
        This method creates pandas DF containing either OOF or test predictions (obtained with the single seed).
        :param preds: array with a predictions (either oof or sub)
        :param oof_prediction: if True -> the provided bagged_df contains OOF results, otherwise -> test predictions
        :return: pandas DF with either OOF or test predictions (single seed)
        """
        df = pd.DataFrame()
        target_col = self.target_column + '_OOF' if oof_prediction else self.target_column

        if self._index_column_is_defined():
            index_values = self.train_df[self.index_column].values if oof_prediction \
                else self.test_df[self.index_column].values
            df[self.index_column] = index_values
        df[target_col] = np.round(preds, self.target_decimals)

        # Convert to int if target rounding precision is 0 decimals
        if self.target_decimals == 0:
            df[target_col] = df[target_col].astype(int)
        return df

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

    def _get_shap_values_in_fold(self, x, feats, n_fold):  # type: ((np.array, pd.DataFrame), list, int) -> pd.DataFrame
        """
        This method prepares DF with the mean absolute shap value for each feature per fold
        :param x: numpy.array or pandas.DataFrame on which to explain the model's output.
        :param feats: list of features names
        :param n_fold: fold index
        :return: pandas DF with feature names and mean absolute shap values (in each considered fold)
        """
        explainer = shap.TreeExplainer(self.classifier.estimator)
        shap_values = explainer.shap_values(x)

        shap_values_df = pd.DataFrame()
        shap_values_df["feature"] = feats

        # In case of multi-class classification, explainer.expected_value -> list
        if isinstance(shap_values, list):
            for i, shap_value in enumerate(shap_values):
                # Computing average impact of each feature in on model output (mean(abs(shap_values)) / per fold
                abs_mean_shap_values = np.mean(np.abs(shap_value), axis=0)
                expected_value = explainer.expected_value[i] if explainer.expected_value[i] is not None else None
                shap_values_df["shap_value_target_{}".format(str(i))] = abs_mean_shap_values
                shap_values_df["expected_value_target_{}".format(str(i))] = expected_value
        else:
            # Binary classification -> explainer.expected_value -> float
            abs_mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            expected_value = explainer.expected_value if explainer.expected_value is not None else None
            shap_values_df["shap_value"] = abs_mean_shap_values
            shap_values_df["expected_value"] = expected_value
            shap_values_df.sort_values('shap_value', ascending=False)
        shap_values_df["fold"] = n_fold + 1
        return shap_values_df

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
        feats = [f for f in self.train_df.columns if f not in set(self.cols_to_exclude).union(
            {self.target_column, self.index_column})]

        shap_values_df = pd.DataFrame()
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
            oof_preds[valid_idx] = self.classifier.run_prediction(
                x=valid_x,
                predict_probability=self.predict_probability,
                num_iteration=best_iter_in_fold
            )

            # Make a prediction for test data (if flag is True)
            if predict_test:
                sub_preds += self.classifier.run_prediction(
                    x=self.test_df[feats],
                    predict_probability=self.predict_probability,
                    num_iteration=int(round(best_iter_in_fold * 1.1, 0))
                ) / folds.n_splits

            if hasattr(self.classifier.estimator, 'feature_importances_'):
                # Get feature importances per fold and store corresponding dataframe to list
                fold_importance_df = self._get_feature_importances_in_fold(feats, n_fold)
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

                # Get SHAP values per fold and store corresponding dataframe to list
                fold_shap_df = self._get_shap_values_in_fold(valid_x, feats, n_fold)
                shap_values_df = pd.concat([shap_values_df, fold_shap_df], axis=0)
            else:
                # E.g. Logistic regression does not have feature importance attribute
                feature_importance_df = None
                shap_values_df = None

            oof_eval_result = round(self.metrics_scorer(valid_y, oof_preds[valid_idx]), self.metrics_decimals)
            oof_eval_results.append(oof_eval_result)
            print('CV: Fold {0} {1} : {2}\n'.format(n_fold + 1, self.metrics_scorer.func_name.upper(), oof_eval_result))

        # CV score and STD of CV score over folds for a given seed
        cv_score = round(self.metrics_scorer(self.train_df[target], oof_preds), self.metrics_decimals)
        cv_std = round(float(np.std(oof_eval_results)), self.metrics_decimals)
        print('CV: list of OOF {0}: {1}'.format(self.metrics_scorer.func_name.upper(), oof_eval_results))
        print('CV {0}: {1} +/- {2}'.format(self.metrics_scorer.func_name.upper(), cv_score, cv_std))
        return oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df, cv_score, cv_std

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

        if self.bagging and len(self.seeds_list) == 1:
            raise ValueError('Number of seeds for bagging should be more than 1. Provided: {0}'
                             .format(len(self.seeds_list)))

        if self.bagging and len(self.seeds_list) > 1:
            oof_pred_bagged = []  # out-of-fold predictions for all seeds [dimension: n_rows_train x n_seeds]
            sub_preds_bagged = []  # test predictions for all seeds [dimension: n_rows_test x n_seeds]
            oof_eval_results_bagged = []  # CV scores in each fold for all seeds [dimension: n_seeds x n_folds]
            feature_importance_bagged = []  # features imp. (averaged over folds) for all seeds
            shap_values_bagged = []  # features shap values (averaged over folds) for all seeds
            cv_score_bagged = []  # CV scores (averaged over folds) for all seeds [dimension: n_seeds]
            cv_std_bagged = []  # CV std's (computed over folds) for all seeds [dimension: n_seeds]

            for i, seed_val in enumerate(self.seeds_list):
                oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df, cv_score, cv_std = \
                    self._run_cv_one_seed(seed_val, self.predict_test)

                oof_pred_bagged.append(pd.Series(oof_preds, name='seed_%s' % str(i + 1)).round(self.target_decimals))
                sub_preds_bagged.append(pd.Series(sub_preds, name='seed_%s' % str(i + 1)).round(self.target_decimals))
                oof_eval_results_bagged.append(oof_eval_results)
                feature_importance_bagged.append(feature_importance_df)
                shap_values_bagged.append(shap_values_df)
                cv_score_bagged.append(cv_score)
                cv_std_bagged.append(cv_std)

            del oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df; gc.collect()

            # Preparing DF with OOF predictions for all seeds
            bagged_oof_preds = self._concat_bagged_results(oof_pred_bagged)
            self.bagged_oof_preds = bagged_oof_preds

            # Preparing DF with submission predictions for all seeds
            bagged_sub_preds = self._concat_bagged_results(sub_preds_bagged)
            self.bagged_sub_preds = bagged_sub_preds

            # Averaging results over seeds to compute single set of OOF predictions
            oof_preds = self._average_bagged_results(bagged_oof_preds, oof_prediction=True)
            self.oof_preds = oof_preds

            # Store predictions for test data (if flag is True). Use simple averaging over seeds (same as oof_preds)
            if self.predict_test:
                sub_preds = self._average_bagged_results(bagged_sub_preds, oof_prediction=False)
                self.sub_preds = sub_preds

            # Final stats: CV score and STD of CV score computed over all seeds
            cv_score = round(self.metrics_scorer(
                self.train_df[self.target_column], oof_preds[self.target_column + '_OOF']), self.metrics_decimals
            )
            cv_std = round(float(np.std(cv_score_bagged)), self.metrics_decimals)
            print('\nCV: bagged {0} score {1} +/- {2}\n'.format(self.metrics_scorer.func_name.upper(), cv_score, cv_std))

            # The DF below contains seed number used in the CV run, cv_score averaged over all folds (see above),
            # std of CV score as well as list of CV values (in all folds).
            self.oof_eval_results = pd.DataFrame(
                zip(self.seeds_list, cv_score_bagged, cv_std_bagged, oof_eval_results_bagged),
                columns=['seed', 'cv_mean_score', 'cv_std', 'cv_score_per_each_fold']
            )
            self.feature_importance = pd.concat(feature_importance_bagged).reset_index(drop=True)
            self.shap_values = pd.concat(shap_values_bagged).reset_index(drop=True)
            del oof_pred_bagged, sub_preds_bagged; gc.collect()

        else:
            oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df, cv_score, cv_std = \
                self._run_cv_one_seed(self.seeds_list[0], self.predict_test)

            oof_preds_df = self._prepare_single_seed_results(oof_preds, oof_prediction=True)
            self.oof_preds = oof_preds_df

            # Store predictions for test data (if flag is True)
            if self.predict_test:
                sub_preds_df = self._prepare_single_seed_results(sub_preds, oof_prediction=False)
                self.sub_preds = sub_preds_df

            # The DF below contains seed number used in the CV run, cv_score averaged over all folds (see above),
            # std of CV score as well as list of CV values (in all folds).
            self.oof_eval_results = pd.DataFrame([self.seeds_list[0], cv_score, cv_std, oof_eval_results],
                                                 index=['seed', 'cv_mean_score', 'cv_std', 'cv_score_per_each_fold']).T
            self.feature_importance = feature_importance_df
            self.shap_values = shap_values_df
            del oof_preds, sub_preds; gc.collect()

        # Saving final cv score and std
        self.cv_score = cv_score
        self.cv_std = cv_std

    def plot_confusion_matrix(self, class_names, labels_mapper=None, normalize=False, cmap=plt.cm.Blues, save=False):
        """
        This function prints and plots the confusion matrix. Normalization can be applied by setting normalize=True.
        :param class_names: list of strings defining unique classes names
        :param labels_mapper:
        :param normalize: if True -> normalizes results in confusion matrix (shows units instead of counting values)
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

    @staticmethod
    def verify_number_of_features_is_ok(df, n_features):  # type: (pd.DataFrame, int) -> None
        """
        This method verifies that number of features requested to plot is below of the max number of
        unique features in the data set
        :param df: pandas DF containing feature importances / shap values
        :param n_features: number of features requested to plot
        :return: None
        """
        n_max_feats = len(df['feature'].unique())
        if n_features > n_max_feats:
            raise ValueError("Number of features that are requested to plot {0} should be less than or equal to the "
                             "total number of features in the data set: {1}".format(n_features, n_max_feats))

    def plot_features_importance(self, n_features=20, figsize_x=8, figsize_y=10, save=False):
        """
        This method plots features importance and saves the figure and the csv file to the disk.
        :param n_features: number of top most important features to show
        :param figsize_x: size of figure along X-axis
        :param figsize_y: size of figure along Y-axis
        :param save: if True -> results will be saved to disk
        :return: plot of features importance
        """
        features_importance = self.feature_importance.copy()
        self.verify_number_of_features_is_ok(features_importance, n_features)

        cols = features_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_features].index
        best_features = features_importance.loc[features_importance.feature.isin(cols)].sort_values(
            by="importance", ascending=False)

        plt.figure(figsize=(figsize_x, figsize_y))
        sns.barplot(x="importance", y="feature", data=best_features)
        plt.title('{0}: features importance (avg over folds/seeds)'.format(self.model_name.upper()))
        plt.tight_layout()

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_import']) + '.png')
            print('\nSaving features importance graph into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_import']) + '.csv')
            print('\nSaving {0} features into {1}'.format(self.model_name.upper(), full_path_to_file))
            features_importance = features_importance[["feature", "importance"]].groupby(
                "feature").mean().sort_values(by="importance", ascending=False).reset_index()
            features_importance.to_csv(full_path_to_file, index=False)
        del features_importance; gc.collect()

    def plot_shap_values(self, n_features=20, figsize_x=8, figsize_y=10, save=False):
        """
        This method plots features shap values and saves the figure and the csv file to the disk.
        :param n_features: number of top most important features to show
        :param figsize_x: size of figure along X-axis
        :param figsize_y: size of figure along Y-axis
        :param save: if True -> results will be saved to disk
        :return: plot of shap values (mean absolute computed over all folds/seeds) for each feature
        """
        shap_values = self.shap_values.copy()
        self.verify_number_of_features_is_ok(shap_values, n_features)

        cols = shap_values[["feature", "shap_value"]].groupby("feature").mean().sort_values(
            by="shap_value", ascending=False)[:n_features].index
        best_features = shap_values.loc[shap_values.feature.isin(cols)].sort_values(
            by="shap_value", ascending=False)

        plt.figure(figsize=(figsize_x, figsize_y))
        sns.barplot(x="shap_value", y="feature", data=best_features)
        plt.title('{0}: shap values of features (avg over folds/seeds)'.format(self.model_name.upper()))
        plt.tight_layout()

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_shap']) + '.png')
            print('\nSaving features shap graph into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_shap']) + '.csv')
            print('\nSaving {0} shap values into {1}'.format(self.model_name.upper(), full_path_to_file))
            shap_values = shap_values[["feature", "shap_value"]].groupby(
                "feature").mean().sort_values(by="shap_value", ascending=False).reset_index()
            shap_values.to_csv(full_path_to_file, index=False)
        del shap_values; gc.collect()

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
        ax.errorbar(x=x, y=y, yerr=yerr, fmt='-o')
        ax.set_title('{0}: CV {1} scores and corresponding stds for considered seeds. '
                     'Final score: {2} +/- {3}'.format(self.model_name.upper(), self.eval_metric, self.cv_score,
                                                       self.cv_std), size=13)
        ax.set_xlabel('Index of CV run', size=12)
        ax.set_ylabel('CV {0} score'.format(self.eval_metric), size=12)
        ax.set_xticks(x)

        # Add annotations with the seed number
        for xpos, ypos, name in zip(x, y, annotation):
            ax.annotate(name, (xpos, ypos), xytext=(annot_offset_x, annot_offset_y), va='bottom',
                        textcoords='offset points', rotation=annot_rotation)
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
        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'SUBM']) + '.csv')
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
    lgbm_params['objective'] = 'multiclass'
    lgbm_params['num_leaves'] = 16
    lgbm_params['max_depth'] = 4
    lgbm_params['learning_rate'] = 0.02
    lgbm_params['n_estimators'] = 1000
    lgbm_params['early_stopping_rounds'] = 50
    lgbm_params['min_split_gain'] = 0.01
    lgbm_params['min_child_weight'] = 1
    lgbm_params['subsample'] = 1.0
    lgbm_params['colsample_bytree'] = 1.0
    lgbm_params['reg_alpha'] = 0.0
    lgbm_params['reg_lambda'] = 0.0
    lgbm_params['n_jobs'] = -1
    lgbm_params['verbose'] = -1

    lgbm_wrapped = LightGBMWrapper(model=LGBMClassifier, params=lgbm_params, seed=27)
    return lgbm_wrapped


def prepare_xgb():
    from xgboost import XGBClassifier
    from modeling.model_wrappers import XGBWrapper

    # XGBoost parameters
    xgb_params = {}
    xgb_params['booster'] = 'gbtree'
    xgb_params['objective'] = 'multi:softprob'  # 'binary:logistic'
    xgb_params['tree_method'] = 'exact'  # 'exact'
    xgb_params['max_depth'] = 3
    xgb_params['learning_rate'] = 0.05
    xgb_params['n_estimators'] = 500
    xgb_params['early_stopping_rounds'] = 100
    xgb_params['min_child_weight'] = 1
    xgb_params['subsample'] = 1.0
    xgb_params['colsample_bytree'] = 1.0
    xgb_params['reg_alpha'] = 0.0
    xgb_params['reg_lambda'] = 1.0
    xgb_params['n_jobs'] = -1
    xgb_params['verbose'] = -1

    xgb_wrapped = XGBWrapper(model=XGBClassifier, params=xgb_params, seed=27)
    return xgb_wrapped


def main_run_cv_and_prediction(classifier='lgbm'):
    import warnings
    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    warnings.filterwarnings("ignore")
    seed = 2018

    # Input data
    iris = datasets.load_iris()
    df_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_data['TARGET'] = iris.target

    # Train/test split
    train_data, test_data = train_test_split(df_data, stratify=df_data['TARGET'], test_size=0.25,
                                             random_state=seed)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    print('df_train shape: {0}'.format(train_data.shape))
    print('df_test shape: {0}'.format(test_data.shape))

    # Parameters
    predict_probability = False  # if True -> use estimator.predict_proba(), otherwise -> estimator.predict()
    solution_output_dir = ''
    target_column = 'TARGET'
    index_column = ''
    metrics_scorer = accuracy_score
    metrics_decimals = 3
    target_decimals = 0
    num_folds = 2
    stratified = True
    kfolds_shuffle = True
    cv_verbosity = 50
    bagging = False
    predict_test = True
    seeds_list = [27, 55, 999999]

    # Columns to exclude from input data
    cols_to_exclude = ['TARGET']

    if classifier is 'lgbm':
        classifier_model = prepare_lgbm()
        eval_metric = 'multi_error'  # see https://lightgbm.readthedocs.io/en/latest/Parameters.html
    elif classifier is 'xgb':
        classifier_model = prepare_xgb()
        eval_metric = 'merror'  # see https://xgboost.readthedocs.io/en/latest/parameter.html
    else:
        classifier_model = prepare_lgbm()
        eval_metric = 'multi_error'

    predictor = Predictor(
        train_df=train_data, test_df=test_data, target_column=target_column, index_column=index_column,
        classifier=classifier_model, predict_probability=predict_probability, eval_metric=eval_metric,
        metrics_scorer=metrics_scorer, metrics_decimals=metrics_decimals, target_decimals=target_decimals,
        cols_to_exclude=cols_to_exclude, num_folds=num_folds, stratified=stratified, kfolds_shuffle=kfolds_shuffle,
        cv_verbosity=cv_verbosity, bagging=bagging, predict_test=predict_test, seeds_list=seeds_list,
        output_dir=solution_output_dir
    )
    predictor.run_cv_and_prediction()
    test_accuracy = round(metrics_scorer(predictor.test_df[target_column], predictor.sub_preds[target_column]),
                          metrics_decimals)
    print ('\nTest: {0}={1}\n'.format(metrics_scorer.func_name.upper(), test_accuracy))


if __name__ == '__main__':
    main_run_cv_and_prediction(classifier='lgbm')
    # main_run_cv_and_prediction(classifier='xgb')
