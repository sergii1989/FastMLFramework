import os
import gc
import shap
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from modeling.model_wrappers import ModelWrapper
from sklearn.model_selection import KFold, StratifiedKFold
from generic_tools.utils import timing, create_output_dir
from sklearn.metrics import confusion_matrix, classification_report


class BaseEstimator(object):
    FIGNAME_CONFUSION_MATRIX = 'confusion_matrix.png'
    FIGNAME_CV_RESULTS_VERSUS_SEEDS = 'cv_results_vs_seeds.png'

    def __init__(self, train_df, test_df, target_column, index_column, classifier, predict_probability, class_index,
                 eval_metric, metrics_scorer, metrics_decimals=6, target_decimals=6, cols_to_exclude=[],
                 num_folds=5, stratified=False, kfolds_shuffle=True, cv_verbosity=1000, bagging=False,
                 data_split_seed=789987, model_seeds_list=[27], predict_test=True, path_output_dir=''):
        """
        This class is a base class for both single model predictions and models stacking. It run CV and makes OOF and
        submission predictions. It also allows to run CV in bagging mode using different seeds for random generator.
        :param train_df: pandas DF with train data set
        :param test_df: pandas DF with test data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param classifier: wrapped estimator (object of ModelWrapper class)
        :param predict_probability: if True -> use classifier.predict_proba(), else -> classifier.predict() method
        :param class_index: index of the class label for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. It also important to notice that
                            the target should always be label encoded.
                            - if class_index is None -> return probability of all classes in the target
                            - if class_index is int -> return probability of selected class
                            - if class_index is list of int -> return probability of selected classes
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
        :param data_split_seed: seed used in splitting train/test data set
        :param model_seeds_list: list of seeds to be used for CV and results prediction (including bagging)
        :param predict_test: IMPORTANT!! If False -> train model and predict OOF (i.e. validation only). Set True
                             if make a prediction for test data set
        :param path_output_dir: full path to directory where the results of CV and prediction to be saved
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
        self.class_index = class_index

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
        self.predict_test = predict_test  # type: bool
        self.data_split_seed = data_split_seed  # type: int
        self.model_seeds_list = model_seeds_list  # type: list
        self.path_output_dir = path_output_dir
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

    def _concat_bagged_results(self, list_bagged_df, is_oof_prediction):  # type: (list) -> pd.DataFrame
        """
        This method concatenates pandas DFs which contain either out-of-fold or test prediction results.
        :param list_bagged_df: list of pandas DFs containing either OOF or test results for various seeds
        :param is_oof_prediction: if True -> provided list_bagged_df contains OOF results, otherwise -> test predictions
        :return: single pandas DF
        """
        df = pd.concat(list_bagged_df, axis=1)
        if self._index_column_is_defined():
            index_values = self.train_df[self.index_column].values if is_oof_prediction \
                else self.test_df[self.index_column].values
            df.insert(loc=0, column=self.index_column, value=index_values)
        return df

    def _average_bagged_results(self, bagged_df, is_oof_prediction):  # type: (pd.DataFrame, bool) -> pd.DataFrame
        """
        This method creates single pandas DF containing average of either out-of-fold or test prediction results that
        were obtained using different seeds (via bagging process).
        :param bagged_df: pandas DF with bagged predictions (either OOF or test), see self._concat_bagged_results()
        :param is_oof_prediction: if True -> provided bagged_df contains OOF results, otherwise -> test predictions
        :return: pandas DF with averaged predictions over different seeds
        """
        df = pd.DataFrame()
        target_col = self.target_column + '_OOF' if is_oof_prediction else self.target_column

        if self._index_column_is_defined():
            index_values = self.train_df[self.index_column].values if is_oof_prediction \
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

    def _prepare_single_seed_results(self, preds, is_oof_prediction):  # type: (np.ndarray, bool) -> pd.DataFrame
        """
        This method creates pandas DF containing either OOF or test predictions (obtained with the single seed).
        :param preds: array with a predictions (either oof or sub)
        :param is_oof_prediction: if True -> the provided bagged_df contains OOF results, otherwise -> test predictions
        :return: pandas DF with either OOF or test predictions (single seed)
        """
        df = pd.DataFrame()
        target_col = self.target_column + '_OOF' if is_oof_prediction else self.target_column

        if self._index_column_is_defined():
            index_values = self.train_df[self.index_column].values if is_oof_prediction \
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

    def run_cv_one_seed(self, seed_val=27, predict_test=True, cv_verbosity=None):
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
        if self.classifier.model_seed_flag:
            self.classifier.reset_models_seed(seed_val)

        if self.stratified:
            folds = StratifiedKFold(n_splits=self.num_folds,
                                    shuffle=self.kfolds_shuffle,
                                    random_state=self.data_split_seed)
        else:
            folds = KFold(n_splits=self.num_folds,
                          shuffle=self.kfolds_shuffle,
                          random_state=self.data_split_seed)

        # Create arrays and data frames to store results
        # Note: if predict_test is False -> sub_preds = None
        oof_preds = np.zeros(self.train_df.shape[0])
        sub_preds = [] if predict_test else None

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
                num_iteration=best_iter_in_fold,
                predict_probability=self.predict_probability,
                class_index=self.class_index
            )

            # Make a prediction for test data (if predict_test is True)
            if predict_test:
                sub_preds.append(self.classifier.run_prediction(
                    x=self.test_df[feats],
                    num_iteration=int(round(best_iter_in_fold * 1.1, 0)),
                    predict_probability=self.predict_probability,
                    class_index=self.class_index
                ))

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

        if predict_test:
            # If the task is to predict a probability of the classes, use np.mean() on top of the results predicted with
            # the best num_iteration in each fold. Contrarily, if the task is to predict class label -> use mode value
            # of the results (kind of 'hard' majority vote - based on frequency).
            sub_preds = np.mean(sub_preds, axis=0) if self.predict_probability else stats.mode(sub_preds).mode.ravel()

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

        if self.bagging and len(self.model_seeds_list) == 1:
            raise ValueError('Number of seeds for bagging should be more than 1. Provided: {0}'
                             .format(len(self.model_seeds_list)))

        if self.bagging and len(self.model_seeds_list) > 1:
            oof_pred_bagged = []  # out-of-fold predictions for all seeds [dimension: n_rows_train x n_seeds]
            sub_preds_bagged = []  # test predictions for all seeds [dimension: n_rows_test x n_seeds]
            oof_eval_results_bagged = []  # CV scores in each fold for all seeds [dimension: n_seeds x n_folds]
            feature_importance_bagged = []  # features imp. (averaged over folds) for all seeds
            shap_values_bagged = []  # features shap values (averaged over folds) for all seeds
            cv_score_bagged = []  # CV scores (averaged over folds) for all seeds [dimension: n_seeds]
            cv_std_bagged = []  # CV std's (computed over folds) for all seeds [dimension: n_seeds]

            for i, seed_val in enumerate(self.model_seeds_list):
                oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df, cv_score, cv_std = \
                    self.run_cv_one_seed(seed_val, self.predict_test)

                # Convert to int if target rounding precision is 0 decimals

                oof_preds = pd.Series(oof_preds, name='seed_%s' % str(i + 1)).round(self.target_decimals)
                sub_preds = pd.Series(sub_preds, name='seed_%s' % str(i + 1)).round(self.target_decimals)

                if self.target_decimals == 0:
                    oof_preds = oof_preds.astype(int)
                    sub_preds = sub_preds.astype(int)

                oof_pred_bagged.append(oof_preds)
                sub_preds_bagged.append(sub_preds)
                oof_eval_results_bagged.append(oof_eval_results)
                feature_importance_bagged.append(feature_importance_df)
                shap_values_bagged.append(shap_values_df)
                cv_score_bagged.append(cv_score)
                cv_std_bagged.append(cv_std)

            del oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df; gc.collect()

            # Preparing DF with OOF predictions for all seeds
            bagged_oof_preds = self._concat_bagged_results(oof_pred_bagged, is_oof_prediction=True)
            self.bagged_oof_preds = bagged_oof_preds

            # Preparing DF with submission predictions for all seeds
            bagged_sub_preds = self._concat_bagged_results(sub_preds_bagged, is_oof_prediction=False)
            self.bagged_sub_preds = bagged_sub_preds

            # Averaging results over seeds to compute single set of OOF predictions
            oof_preds = self._average_bagged_results(bagged_oof_preds, is_oof_prediction=True)
            self.oof_preds = oof_preds

            # Store predictions for test data (if flag is True). Use simple averaging over seeds (same as oof_preds)
            if self.predict_test:
                sub_preds = self._average_bagged_results(bagged_sub_preds, is_oof_prediction=False)
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
                zip(self.model_seeds_list, cv_score_bagged, cv_std_bagged, oof_eval_results_bagged),
                columns=['seed', 'cv_mean_score', 'cv_std', 'cv_score_per_each_fold']
            )
            self.feature_importance = pd.concat(feature_importance_bagged).reset_index(drop=True)
            self.shap_values = pd.concat(shap_values_bagged).reset_index(drop=True)
            del oof_pred_bagged, sub_preds_bagged; gc.collect()

        else:
            oof_preds, sub_preds, oof_eval_results, feature_importance_df, shap_values_df, cv_score, cv_std = \
                self.run_cv_one_seed(self.model_seeds_list[0], self.predict_test)

            oof_preds_df = self._prepare_single_seed_results(oof_preds, is_oof_prediction=True)
            self.oof_preds = oof_preds_df

            # Store predictions for test data (if flag is True)
            if self.predict_test:
                sub_preds_df = self._prepare_single_seed_results(sub_preds, is_oof_prediction=False)
                self.sub_preds = sub_preds_df

            # The DF below contains seed number used in the CV run, cv_score averaged over all folds (see above),
            # std of CV score as well as list of CV values (in all folds).
            self.oof_eval_results = pd.DataFrame([self.model_seeds_list[0], cv_score, cv_std, oof_eval_results],
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
        fig, ax = plt.subplots()
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
        ax.set_title('Normalized confusion matrix' if normalize else 'Confusion matrix')
        plt.colorbar(ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xlabel(class_names)
        plt.setp(ax.get_xticklabels(), rotation=0)

        ax.set_yticks(tick_marks)
        ax.set_ylabel(class_names)
        plt.setp(ax.get_xticklabels(), rotation=0)

        fmt = '.4f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()

        print ('Classification report:\n{0}'.format(
            classification_report(true_labels, predicted_labels, target_names=class_names)))

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, self.FIGNAME_CONFUSION_MATRIX)
            print('\nSaving confusion matrix graph into %s' % full_path_to_file)
            fig.savefig(full_path_to_file)

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

    def plot_features_importance(self, n_features=20, figsize_x=10, figsize_y=10, save=False):
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

        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
        sns.barplot(x="importance", y="feature", data=best_features, ax=ax)
        ax.set_title('{0}: features importance (avg over folds/seeds)'.format(self.model_name.upper()))
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)

        if save:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'feat_import']) + '.png')
            print('\nSaving features importance graph into %s' % full_path_to_file)
            fig.savefig(full_path_to_file)

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

        # TODO: think how to visualize shap values in case of multi-class classification task
        # Possible example is here: https://github.com/slundberg/shap (see multi-class SVM example)
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
            full_path_to_file = os.path.join(self.path_output_dir,
                                             '_'.join([self.model_name, 'feat_shap']) + '.png')
            print('\nSaving features shap graph into %s' % full_path_to_file)
            plt.savefig(full_path_to_file)

            full_path_to_file = os.path.join(self.path_output_dir,
                                             '_'.join([self.model_name, 'feat_shap']) + '.csv')
            print('\nSaving {0} shap values into {1}'.format(self.model_name.upper(), full_path_to_file))
            shap_values = shap_values[["feature", "shap_value"]].groupby(
                "feature").mean().sort_values(by="shap_value", ascending=False).reset_index()
            shap_values.to_csv(full_path_to_file, index=False)
        del shap_values;
        gc.collect()

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
            fig.savefig(full_path_to_file)

    def save_oof_results(self):
        float_format = '%.{0}f'.format(str(self.target_decimals)) if self.target_decimals > 0 else None

        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'OOF']) + '.csv')
        print('\nSaving elaborated OOF predictions into %s' % full_path_to_file)
        self.oof_preds.to_csv(full_path_to_file, index=False, float_format=float_format)

        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'CV']) + '.csv')
        print('\nSaving CV results into %s' % full_path_to_file)
        self.oof_eval_results.to_csv(full_path_to_file, index=False, float_format=float_format)

        if self.bagged_oof_preds is not None:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'bagged_OOF']) + '.csv')
            print('\nSaving OOF predictions for each seed into %s' % full_path_to_file)
            self.bagged_oof_preds.to_csv(full_path_to_file, index=False, float_format=float_format)

    def save_submission_results(self):
        float_format = '%.{0}f'.format(str(self.target_decimals)) if self.target_decimals > 0 else None

        if self.sub_preds is None:
            raise ValueError('Submission file is empty. Please set flag predict_test = True in run_cv_and_prediction() '
                             'to generate submission file.')
        full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'SUBM']) + '.csv')
        print('\nSaving submission predictions into %s' % full_path_to_file)
        self.sub_preds.to_csv(full_path_to_file, index=False, float_format=float_format)

        if self.bagged_sub_preds is not None:
            full_path_to_file = os.path.join(self.path_output_dir, '_'.join([self.model_name, 'bagged_SUBM']) + '.csv')
            print('\nSaving submission predictions for each seed into %s' % full_path_to_file)
            self.bagged_sub_preds.to_csv(full_path_to_file, index=False, float_format=float_format)


class Predictor(BaseEstimator):

    def __init__(self, train_df, test_df, target_column, index_column, classifier, predict_probability, class_index,
                 eval_metric, metrics_scorer, metrics_decimals=6, target_decimals=6, cols_to_exclude=[], num_folds=5,
                 stratified=False, kfolds_shuffle=True, cv_verbosity=1000, bagging=False, data_split_seed=789987,
                 model_seeds_list=[27], predict_test=True, project_location='', output_dirname=''):
        """
        This class runs CV and makes OOF and submission predictions. It also allows to run CV in bagging mode using
        different seeds for random generator.
        :param train_df: pandas DF with train data set
        :param test_df: pandas DF with test data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param classifier: wrapped estimator (object of ModelWrapper class)
        :param predict_probability: if True -> use classifier.predict_proba(), else -> classifier.predict() method
        :param class_index: index of the class label for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. It also important to notice that
                            the target should always be label encoded.
                            - if class_index is None -> return probability of all classes in the target
                            - if class_index is int -> return probability of selected class
                            - if class_index is list of int -> return probability of selected classes
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
        :param bagging: if True -> runs CV with different seeds and then average the results
        :param data_split_seed: seed used in splitting train/test data set
        :param model_seeds_list: list of seeds to be used for CV and results prediction (including bagging)
        :param predict_test: IMPORTANT!! If False -> train model and predict OOF (i.e. validation only). Set True
                             if make a prediction for test data set
        :param output_dirname: directory where to save results of CV and prediction
        :return: out_of_fold predictions, submission predictions, oof_eval_results and feature_importance data frame
        """

        # Full path to solution directory
        path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))
        super(Predictor, self).__init__(
            train_df, test_df, target_column, index_column, classifier, predict_probability, class_index, eval_metric,
            metrics_scorer, metrics_decimals, target_decimals, cols_to_exclude, num_folds, stratified, kfolds_shuffle,
            cv_verbosity, bagging, data_split_seed, model_seeds_list, predict_test, path_output_dir
        )


def prepare_lgbm(params):
    from lightgbm import LGBMClassifier
    from modeling.model_wrappers import LightGBMWrapper
    lgbm_wrapped = LightGBMWrapper(model=LGBMClassifier, params=params, seed=27)
    return lgbm_wrapped


def prepare_xgb(params):
    from xgboost import XGBClassifier
    from modeling.model_wrappers import XGBWrapper
    xgb_wrapped = XGBWrapper(model=XGBClassifier, params=params, seed=27)
    return xgb_wrapped


def run_cv_and_prediction_iris(classifier='lgbm'):
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
    class_index = 1  # in Target
    project_location = ''  # 'C:\Kaggle\Iris'
    output_dirname = ''  # 'iris_solution'
    target_column = 'TARGET'
    index_column = ''
    metrics_scorer = accuracy_score
    metrics_decimals = 3
    target_decimals = 0
    num_folds = 3
    stratified = True
    kfolds_shuffle = True
    cv_verbosity = 50
    bagging = True
    predict_test = True
    data_split_seed = 789987
    model_seeds_list = [27, 55, 999999, 123345, 8988, 45789, 65479, 321654]

    # Columns to exclude from input data
    cols_to_exclude = ['TARGET']

    if classifier is 'lgbm':
        params = {
            'boosting_type': 'gbdt',  # gbdt, gbrt, rf, random_forest, dart, goss
            'objective': 'multiclass',
            'num_leaves': 16,
            'max_depth': 4,
            'learning_rate': 0.02,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'min_split_gain': 0.01,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'n_jobs': -1,
            'verbose': -1
        }
        classifier_model = prepare_lgbm(params)
        eval_metric = 'multi_error'  # see https://lightgbm.readthedocs.io/en/latest/Parameters.html
    elif classifier is 'xgb':
        params = {
            'booster': 'gbtree',
            'objective': 'multi:softprob',  # 'binary:logistic'
            'tree_method': 'exact',  # 'exact'
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'early_stopping_rounds': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'verbose': -1
        }
        classifier_model = prepare_xgb(params)
        eval_metric = 'merror'  # see https://xgboost.readthedocs.io/en/latest/parameter.html
    else:
        params = {
            'boosting_type': 'gbdt',  # gbdt, gbrt, rf, random_forest, dart, goss
            'objective': 'multiclass',
            'num_leaves': 16,
            'max_depth': 4,
            'learning_rate': 0.02,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'min_split_gain': 0.01,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'n_jobs': -1,
            'verbose': -1
        }
        classifier_model = prepare_lgbm(params)
        eval_metric = 'multi_error'

    predictor = Predictor(
        train_df=train_data, test_df=test_data, target_column=target_column, index_column=index_column,
        classifier=classifier_model, predict_probability=predict_probability, class_index=class_index,
        eval_metric=eval_metric, metrics_scorer=metrics_scorer, metrics_decimals=metrics_decimals,
        target_decimals=target_decimals, cols_to_exclude=cols_to_exclude, num_folds=num_folds,
        stratified=stratified, kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity, bagging=bagging,
        predict_test=predict_test, data_split_seed=data_split_seed, model_seeds_list=model_seeds_list,
        project_location=project_location, output_dirname=output_dirname
    )
    predictor.run_cv_and_prediction()
    # predictor.save_oof_results()
    # predictor.save_submission_results()

    test_accuracy = round(metrics_scorer(predictor.test_df[target_column], predictor.sub_preds[target_column]),
                          metrics_decimals)
    print ('\nTest: {0}={1}\n'.format(metrics_scorer.func_name.upper(), test_accuracy))


def run_cv_and_prediction_kaggle(classifier='lgbm', debug=False):
    import warnings
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes

    warnings.filterwarnings("ignore")

    # Settings for debug
    num_rows = 2000

    # Input data
    path_to_data = r'C:\Kaggle\kaggle_home_credit_default_risk\feature_selection'
    full_path_to_file = os.path.join(path_to_data, 'train_dataset_lgbm_10.csv')
    train_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
        .reset_index(drop=True)
    full_path_to_file = os.path.join(path_to_data, 'test_dataset_lgbm_10.csv')
    test_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
        .reset_index(drop=True)
    print('df_train shape: {0}'.format(train_data.shape))
    print('df_test shape: {0}'.format(test_data.shape))

    # Parameters
    predict_probability = True  # if True -> use estimator.predict_proba(), otherwise -> estimator.predict()
    class_index = 1  # in Target
    project_location = ''  # 'C:\Kaggle\home_credit'
    output_dirname = ''  # 'solution'
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
    data_split_seed = 789987
    model_seeds_list = [27, 55]
    cols_to_exclude = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']

    if classifier is 'lgbm':
        params = {
            'boosting_type': 'gbdt',  # gbdt, gbrt, rf, random_forest, dart, goss
            'objective': 'binary',
            'num_leaves': 32,  # 32
            'max_depth': 8,  # 8
            'learning_rate': 0.02,  # 0.01
            'n_estimators': 10000,
            'early_stopping_rounds': 200,
            'min_split_gain': 0.02,  # 0.02
            'min_child_weight': 40,  # 40
            'subsample': 0.8,  # 0.87
            'colsample_bytree': 0.8,  # 0.94
            'reg_alpha': 0.04,  # 0.04
            'reg_lambda': 0.07,  # 0.073
            'nthread': -1,
            'verbose': -1
        }
        classifier_model = prepare_lgbm(params)
    elif classifier is 'xgb':
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'tree_method': 'hist',  # 'exact'
            'max_depth': 6,
            'learning_rate': 0.02,
            'n_estimators': 10000,
            'early_stopping_rounds': 200,
            'min_child_weight': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.0,
            'reg_lambda': 1.2,
            'n_jobs': -1,
            'verbose': -1,
            'colsample_bylevel': 0.632,
            'scale_pos_weight': 2.5
        }
        classifier_model = prepare_xgb(params)
    else:
        params = {
            'boosting_type': 'gbdt',  # gbdt, gbrt, rf, random_forest, dart, goss
            'objective': 'binary',
            'num_leaves': 32,  # 32
            'max_depth': 8,  # 8
            'learning_rate': 0.02,  # 0.01
            'n_estimators': 10000,
            'early_stopping_rounds': 200,
            'min_split_gain': 0.02,  # 0.02
            'min_child_weight': 40,  # 40
            'subsample': 0.8,  # 0.87
            'colsample_bytree': 0.8,  # 0.94
            'reg_alpha': 0.04,  # 0.04
            'reg_lambda': 0.07,  # 0.073
            'nthread': -1,
            'verbose': -1
        }
        classifier_model = prepare_lgbm(params)

    predictor = Predictor(
        train_df=train_data, test_df=test_data, target_column=target_column, index_column=index_column,
        classifier=classifier_model, predict_probability=predict_probability, class_index=class_index,
        eval_metric=eval_metric, metrics_scorer=metrics_scorer, metrics_decimals=metrics_decimals,
        target_decimals=target_decimals, cols_to_exclude=cols_to_exclude, num_folds=num_folds,
        stratified=stratified, kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity, bagging=bagging,
        predict_test=predict_test, data_split_seed=data_split_seed, model_seeds_list=model_seeds_list,
        project_location=project_location, output_dirname=output_dirname
    )
    predictor.run_cv_and_prediction()
    # predictor.save_oof_results()
    # predictor.save_submission_results()


if __name__ == '__main__':
    run_cv_and_prediction_iris(classifier='lgbm')
    # run_cv_and_prediction_iris(classifier='xgb')
    # run_cv_and_prediction_kaggle(classifier='lgbm', debug=True)
    # run_cv_and_prediction_kaggle(classifier='xgb', debug=True)
