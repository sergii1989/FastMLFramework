import os
import logging
import warnings
import numpy as np
import pandas as pd
import numpy.testing as npt

from scipy import stats
from shutil import copyfile
from collections import OrderedDict
from modeling.prediction import Predictor
from bayes_opt import BayesianOptimization
from ensembling.ensembler import Ensembler
from generic_tools.loggers import configure_logging
from generic_tools.utils import timing, create_output_dir
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

configure_logging()
_logger = logging.getLogger("ensembling.blender")


class Blender(object):
    FILENAME_TRAIN_OOF_RESULTS = 'train_OOF.csv'
    FILENAME_TEST_RESULTS = 'test.csv'
    FILENAME_CV_RESULTS = 'cv_results.csv'

    def __init__(self, oof_input_files, blend_bagged_results, train_df, test_df, target_column, index_column,
                 metrics_scorer, metrics_decimals=6, target_decimals=6, project_location='', output_dirname=''):
        """
        This is a base class for blending models prediction. The blender is trained on out-of-fold predictions (OOF)
        of the 1st (or 2nd) level models and applied to test submissions. The blender is optimized in a way to maximize
        evaluation metrics.
        :param oof_input_files: dict with locations and names of train and test OOF data sets (to be used in blending)
        :param blend_bagged_results: if True and -> blender will use raw OOF predictions obtained for each seed by the
                                     selected models (and not the mean prediction over all seeds per model)
        :param train_df: pandas DF with train data set
        :param test_df: pandas DF with test data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param metrics_scorer: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        :param metrics_decimals: round precision (decimals) of the metrics (e.g. used in printouts)
        :param target_decimals: round precision (decimals) of the target column
        :param project_location: path to the project
        :param output_dirname: name of directory where to save results of blending procedure
        """

        # Input data
        self.target_column = target_column
        self.index_column = index_column
        self.metrics_scorer = metrics_scorer
        self.metrics_decimals = metrics_decimals
        self.target_decimals = target_decimals

        # Load OOF data (both train and test)
        self.ensembler = Ensembler()
        self.train_oof, self.test_oof = \
            self.ensembler.load_oof_target_and_test_data(oof_input_files, blend_bagged_results, train_df, test_df,
                                                         target_column, index_column, target_decimals, project_location)
        # Blending results
        self.oof_preds = None  # type: pd.DataFrame
        self.sub_preds = None  # type: pd.DataFrame
        self.cv_results = None  # type: pd.DataFrame
        self.cv_score = None  # type: float
        self.cv_std = None  # type: float

        # Full path to solution directory
        self.path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))
        create_output_dir(self.path_output_dir)

    def evaluate_results(self, **params):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def run(self, **kwargs):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def save_oof_results(self):
        float_format = '%.{0}f'.format(str(self.target_decimals)) if self.target_decimals > 0 else None
        full_path_to_file = os.path.join(self.path_output_dir, self.FILENAME_TRAIN_OOF_RESULTS)
        _logger.info('Saving elaborated OOF predictions into %s' % full_path_to_file)
        self.oof_preds.to_csv(full_path_to_file, index=False, float_format=float_format)

        float_format = '%.{0}f'.format(str(self.metrics_decimals)) if self.metrics_decimals > 0 else None
        full_path_to_file = os.path.join(self.path_output_dir, self.FILENAME_CV_RESULTS)
        _logger.info('Saving CV results into %s' % full_path_to_file)
        self.cv_results.to_csv(full_path_to_file, index=False, float_format=float_format)

    def save_submission_results(self):
        float_format = '%.{0}f'.format(str(self.target_decimals)) if self.target_decimals > 0 else None
        if self.sub_preds is None:
            raise ValueError('Submission file is empty. Please set flag predict_test = True in run_cv_and_prediction() '
                             'to generate submission file.')
        full_path_to_file = os.path.join(self.path_output_dir, self.FILENAME_TEST_RESULTS)
        _logger.info('Saving submission predictions into %s' % full_path_to_file)
        self.sub_preds.to_csv(full_path_to_file, index=False, float_format=float_format)

    def save_config(self, project_location, config_directory, config_file):
        """
        This method saves a copy of a config file to the output directory. This helps to identify which configs were
        used to get results stored in the results directory, therefore enhancing traceability of experiments.
        :param project_location: absolute path to project's main directory
        :param config_directory: name of config sub-directory in project directory
        :param config_file: name of config file in the config sub-directory
        :return: None
        """
        src_config = os.path.normpath(os.path.join(project_location, config_directory, config_file))
        if os.path.exists(src_config):
            dst_config = os.path.join(self.path_output_dir, config_file)
            _logger.info('Copying  {0}  into {1}'.format(src_config, dst_config))
            copyfile(src_config, dst_config)
        else:
            raise IOError('No config file found in: %s' % src_config)


class BayesOptimizationBlender(Blender):
    BLENDING_METHOD = 'bayes_opt_blender'

    def __init__(self, oof_input_files, blend_bagged_results, predict_probability, class_label, train_df, test_df,
                 target_column, index_column, metrics_scorer, metrics_decimals=6, target_decimals=6,
                 num_folds=5, stratified=False, kfolds_shuffle=True, data_split_seed=789987,
                 init_points=10, n_iter=15, blender_seed_val=27, project_location='', output_dirname=''):
        """
        This class implements blending method based on Bayes Optimization procedure. It is trained on out-of-fold
        predictions of the 1st (or 2nd) level models and applied to the test submission. The blender is optimized in
        a way to maximize evaluation metrics.

          BayesOptimizationBlender is similar to the sklearn's VotingClassifier but with the bayes optimized weights.

        :param oof_input_files: dict with locations and names of train and test OOF data sets (to be used in blending)
        :param blend_bagged_results: if True and -> blender will use raw OOF predictions obtained for each seed by the
                                     selected models (and not the mean prediction over all seeds per model)
        :param predict_probability: if True -> use model.predict_proba(), else -> model.predict() method
        :param class_label: class label(s) for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. Class label(s) should be
                            selected from the target column.
                            - if class_label is None -> return probability of all class labels in the target
                            - if class_label is int -> return probability of selected class
                            - if class_label is list of int -> return probability of selected classes
        :param train_df: pandas DF with train data set
        :param test_df: pandas DF with test data set
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param metrics_scorer: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        :param metrics_decimals: round precision (decimals) of the metrics (e.g. used in printouts)
        :param target_decimals: round precision (decimals) of the target column
        :param num_folds: number of folds to be used in CV
        :param stratified: if True -> preserves the percentage of samples for each class in a fold
        :param kfolds_shuffle: if True -> shuffle each stratification of the data before splitting into batches
        :param data_split_seed: seed used in splitting train/test data set
        :param init_points: number of initial points in Bayes Optimization procedure
        :param n_iter: number of iteration in Bayes Optimization procedure
        :param blender_seed_val: seed for Bayes Optimization
        :param project_location: path to the project
        :param output_dirname: name of directory where to save results of blending procedure
        """

        super(BayesOptimizationBlender, self).__init__(
            oof_input_files, blend_bagged_results, train_df, test_df, target_column, index_column, metrics_scorer,
            metrics_decimals, target_decimals, project_location, output_dirname
        )

        # Bayes optimization settings
        self.init_points = init_points  # type: int
        self.n_iter = n_iter  # type: int
        self.blender_seed_val = blender_seed_val  # type: int

        # Settings for CV and test prediction
        self.predict_probability = predict_probability  # type: bool
        self.class_label = class_label if predict_probability else None
        self.stratified = stratified  # type: bool
        self.num_folds = num_folds  # type: int
        self.kfolds_shuffle = kfolds_shuffle  # type: bool
        self.data_split_seed = data_split_seed  # type: int

        # Voting classifier settings
        self.blending_opt_history = None  # type: pd.DataFrame
        self.voting_type = self._detect_voting_type()

        # Auxiliary attributes used to pass in-fold train/test data to bayes optimization function
        self._train_x = None  # type: pd.DataFrame
        self._train_y = None  # type: pd.DataFrame

    @staticmethod
    def _normalize_weights(best_params, precision=3):  # type: (OrderedDict, int) -> OrderedDict
        """
        This method normalizes raw weights from the Bayes Optimization process. The sum of all weights should be 1.
        :param best_params: raw weights from Bayes Optimization process
        :param precision: rounding of the weights (decimals)
        :return: optimized weights (normalized and rounded to the requested precision)
        """
        optimal_weights = OrderedDict()
        s = sum(best_params.values())
        for p in best_params:
            optimal_weights[p] = round(best_params[p] / s, precision)
        npt.assert_almost_equal(sum(optimal_weights.values()), 1.0, precision)
        return optimal_weights

    def _detect_voting_type(self):  # type: () -> str
        """
        This method auto-detects the voting type best suited for the provided OOF data.
            - If all OOF results are class labels (i.e. integers [after label-encoding]) -> use "hard" voting
            - If all OOF results are class probabilities -> use "soft" voting
        :return: voting type (either "hard" or "soft")
        """
        float_types = ["float16", "float32", "float64"]
        int_types = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]

        feats = [f for f in self.train_oof.columns if f not in (self.target_column, self.index_column)]
        columns_unique_dtypes = self.train_oof[feats].dtypes.unique()

        int_cols = [True if col_data_type in int_types else False for col_data_type in columns_unique_dtypes]
        float_cols = [True if col_data_type in float_types else False for col_data_type in columns_unique_dtypes]

        if all(int_cols):
            return 'hard'

        elif all(float_cols):
            return 'soft'

        else:
            logging.error("Provided OOF predictions has mixed unique data types: %s. The OOF data frame should contain"
                          "either int- or float- types of predictions" % columns_unique_dtypes)
            raise TypeError()

    def _run_voting(self, oof_data, weights):  # type: (pd.DataFrame, list) -> np.ndarray
        """
        This method performs weighted voting (either 'hard' or 'soft') based on the provided OOF data and weights.
        :param oof_data: pandas DF with the out-of-fold predictions (OOF)
        :param weights: list with weights for each prediction (read column) in the OOF dataframe
        :return: numpy ndarray with the results of voting
        """

        # TODO: Originally, 'soft' voting predicts the class label based on the argmax of the sums of the predicted
        # probabilities, whereas the current implementation returns only weighted average of a probability for
        # each class. Need to think how to pass an explicit flag to use np.argmax() so to compute labels.
        # Maybe, when self.predict_probability=False and self.voting_type='soft' -> use np.argmax()

        if self.voting_type == 'hard':
            return np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=weights)), axis=1, arr=oof_data)
        return np.average(oof_data, axis=1, weights=weights)

    def evaluate_results(self, **params):  # type: (OrderedDict) -> float
        """
        This method evaluates train prediction (in the process of Bayes Optimization) according to the given
        metrics scorer (from sklearn.metrics)
        :param params: dict with the models weight
        :return: metrics score
        """
        feats = [f for f in self.train_oof.columns if f not in (self.target_column, self.index_column)]
        s = sum(params.values())
        if s != 0.0:  # if all weights in dict are zeros...
            weights = [params[feat] / s for feat in feats]  # this is needed because **params not preserves order p2.7
        else:
            n_keys = len(params.keys())
            weights = [1.0 / n_keys for _ in params.keys()]

        train_pred = self._run_voting(oof_data=self._train_x, weights=weights)
        return self.metrics_scorer(self._train_y, train_pred)

    def _prepare_results(self, preds, is_oof_prediction):  # type: (np.ndarray, bool) -> pd.DataFrame
        """
        This method creates pandas DF containing either OOF or test predictions (obtained with the single seed).
        :param preds: array with a predictions (either oof or sub)
        :param is_oof_prediction: if True -> the provided bagged_df contains OOF results, otherwise -> test predictions
        :return: pandas DF with either OOF or test predictions (single seed)
        """
        df = pd.DataFrame()
        target_col = self.target_column + '_OOF' if is_oof_prediction else self.target_column

        if Predictor.verify_index_column_is_defined(self.index_column):
            index_values = self.train_oof[self.index_column].values if is_oof_prediction \
                else self.test_oof[self.index_column].values
            df[self.index_column] = index_values
        df[target_col] = np.round(preds, self.target_decimals)

        # Convert to int if target rounding precision is 0 decimals
        if self.target_decimals == 0:
            df[target_col] = df[target_col].astype(int)

        # Add column with real target values to OOF dataframe
        if is_oof_prediction:
            df[self.target_column] = self.train_oof[self.target_column].values
        return df

    @timing
    def run(self):
        """
        This method runs Bayes Search of optimal weights to the individual model's predictions with the goal of
        maximizing evaluation metrics score on the train data set. After optimal weights are found, apply them to
        the test predictions. Main outcome of this function is two attributes: self.blending_opt_history -> pandas DF
        with the optimal weights and self.sub_preds -> pandas DF with the blended test predictions.
        :return: None
        """
        _logger.info('Running Bayes Optimization...')
        feats = [f for f in self.train_oof.columns if f not in (self.target_column, self.index_column)]

        # If self.train_oof dataframe contains non-bagged out-of-fold predictions by single models, then the names of
        # columns in self.test_oof should be adjusted so to match the names in self.train_oof. In the case of test
        # dataframe, prediction column is called as self.target column, whereas for train - self.target column + '_OOF'
        cols_rename = {col: col + '_OOF' for col in self.test_oof if self.target_column in col}
        if len(cols_rename):
            self.test_oof.rename(columns=cols_rename, inplace=True)

        params = OrderedDict((c, (0, 1)) for c in feats)

        if self.stratified:
            folds = StratifiedKFold(n_splits=self.num_folds,
                                    shuffle=self.kfolds_shuffle,
                                    random_state=self.data_split_seed)
        else:
            folds = KFold(n_splits=self.num_folds,
                          shuffle=self.kfolds_shuffle,
                          random_state=self.data_split_seed)

        if self.predict_probability:
            if self.class_label is None:
                shape = (self.train_oof.shape[0], self.train_oof[self.target_column].unique().size)
            else:
                if isinstance(self.class_label, list) or isinstance(self.class_label, tuple):
                    shape = (self.train_oof.shape[0], len(self.class_label))
                else:
                    shape = self.train_oof.shape[0]
        else:
            shape = self.train_oof.shape[0]

        # Create arrays and data frames to store results. Note: if predict_test is False -> sub_preds = None
        oof_preds = np.zeros(shape=shape)
        sub_preds = []

        # DF with train/validation scores and optimal weights of blender (per each fold)
        blending_opt_history = pd.DataFrame(index=range(1, self.num_folds + 1), columns=['eval_train',
                                                                                         'eval_valid',
                                                                                         'optimal_weights'])
        cv_results = []  # list of cross-validation results per each folder
        train_eval_results = []  # list of training evaluation results
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(self.train_oof[feats], self.train_oof[self.target_column])):
            train_x, train_y = self.train_oof[feats].iloc[train_idx], self.train_oof[self.target_column].iloc[train_idx]
            valid_x, valid_y = self.train_oof[feats].iloc[valid_idx], self.train_oof[self.target_column].iloc[valid_idx]

            # Need this trick because one can't pass train/test data to the BayesianOptimization function
            self._train_x, self._train_y = train_x, train_y

            bo = BayesianOptimization(f=self.evaluate_results, pbounds=params, random_state=self.blender_seed_val)
            bo.maximize(init_points=self.init_points, n_iter=self.n_iter)

            # Extracting training max score and optimal weights (the raw ones, not normalized)
            best_params = OrderedDict((c, bo.res['max']['max_params'][c]) for c in feats)
            best_train_score = round(bo.res['max']['max_val'], self.metrics_decimals)
            train_eval_results.append(best_train_score)
            optimal_weights = self._normalize_weights(best_params)

            # Out-of-fold prediction
            oof_preds[valid_idx] = self._run_voting(oof_data=valid_x[optimal_weights.keys()],
                                                    weights=optimal_weights.values())

            # Make a prediction for test data
            sub_preds.append(self._run_voting(oof_data=self.test_oof[optimal_weights.keys()],
                                              weights=optimal_weights.values()))

            # CV score in each fold
            cv_result = round(self.metrics_scorer(valid_y, oof_preds[valid_idx]), self.metrics_decimals)
            cv_results.append(cv_result)

            # Train / Validation scores and optimal weights in a fold
            blending_opt_history.iloc[n_fold] = [best_train_score, cv_result, optimal_weights]

            _logger.info('Fold {0} {1} : train-[{2}]  valid-[{3}]'.format(n_fold + 1,
                                                                          self.metrics_scorer.__name__.upper(),
                                                                          best_train_score,
                                                                          cv_result))
            _logger.info('Optimal weights:\n{0}'.format(optimal_weights))

        # CV score and STD of CV score over all folds
        self.cv_score = round(self.metrics_scorer(self.train_oof[self.target_column], oof_preds), self.metrics_decimals)
        self.cv_std = round(float(np.std(cv_results)), self.metrics_decimals)
        _logger.info('\n'.join(['', '=' * 70]))
        _logger.info('List of training {0}: {1}'.format(self.metrics_scorer.__name__.upper(), train_eval_results))
        _logger.info('CV: list of OOF {0}: {1}'.format(self.metrics_scorer.__name__.upper(), cv_results))
        _logger.info('CV {0}: {1} +/- {2}'.format(self.metrics_scorer.__name__.upper(), self.cv_score, self.cv_std))

        # Preparing dataframe with the OOF predictions of the blender
        self.oof_preds = self._prepare_results(oof_preds, is_oof_prediction=True)

        # Preparing dataframe with the test predictions
        sub_preds = np.mean(sub_preds, axis=0) if self.voting_type == 'soft' else stats.mode(sub_preds).mode.ravel()
        self.sub_preds = self._prepare_results(sub_preds, is_oof_prediction=False)

        # Persisting pandas DF with train/validation scores and optimal weights of blender for all folds
        self.blending_opt_history = blending_opt_history

        # The DF below contains seed number used in the CV run, cv_score averaged over all folds (see above),
        # std of CV score as well as list of CV values (in all folds).
        self.cv_results = pd.DataFrame([self.blender_seed_val, self.cv_score, self.cv_std, cv_results],
                                       index=['seed', 'cv_mean_score', 'cv_std', 'cv_score_per_each_fold']).T

    def save_weights(self):
        """
        This method saves Bayes Optimized weights to the disc.
        :return: None
        """
        filename = "_".join(['blender_optimal_weights', str(self.cv_score)]) + '.csv'
        output_figname = os.path.join(self.path_output_dir, filename)
        _logger.info('Saving optimal weights DF into %s' % output_figname)
        self.blending_opt_history.to_csv(output_figname)


def run_blender_kaggle_example(debug=True):
    import warnings
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes
    warnings.filterwarnings("ignore")

    # Settings for debug
    num_rows = 2000

    # Input data
    path_to_data = r'C:\Kaggle\kaggle_home_credit_default_risk\feature_selection'

    # Reading train data set
    full_path_to_file = os.path.join(path_to_data, 'train_dataset_lgbm_10.csv')
    train_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
        .reset_index(drop=True)
    print('df_train shape: {0}'.format(train_data.shape))

    # Reading test data set
    full_path_to_file = os.path.join(path_to_data, 'test_dataset_lgbm_10.csv')
    test_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
        .reset_index(drop=True)
    print('df_test shape: {0}'.format(test_data.shape))

    oof_input_files = {
        'lgbm_5249': {
            'path': 'single_model_solution/lightgbm/features_dataset_001/target_permutation_fs_001/bayes_hpo_001/bagging_on',
            'files': ['train_OOF.csv', 'test.csv', 'train_OOF_bagged.csv', 'test_bagged.csv']
        },
        'xgb_2967': {
            'path': 'single_model_solution/xgboost/features_dataset_001/target_permutation_fs_001/bayes_hpo_001/bagging_on',
            'files': ['train_OOF.csv', 'test.csv', 'train_OOF_bagged.csv', 'test_bagged.csv']
        }
    }

    blend_bagged_results = False
    predict_probability = True
    project_location = 'c:\Kaggle\home_credit_default_risk'  # ''
    output_dirname = ''  # 'solution'
    target_column = 'TARGET'
    index_column = 'SK_ID_CURR'
    class_label = 1
    metrics_scorer = roc_auc_score
    target_decimals = 2
    metrics_decimals = 4
    num_folds = 5
    kfolds_shuffle = True
    stratified = True
    data_split_seed = 27
    blender_n_iter = 5
    blender_init_points = 20
    blender_seed_val = 27

    bayes_blender = BayesOptimizationBlender(oof_input_files=oof_input_files,
                                             train_df=train_data, test_df=test_data,
                                             target_column=target_column, index_column=index_column,
                                             blend_bagged_results=blend_bagged_results,
                                             predict_probability=predict_probability,
                                             class_label=class_label,
                                             init_points=blender_init_points, n_iter=blender_n_iter,
                                             blender_seed_val=blender_seed_val,
                                             metrics_scorer=metrics_scorer,
                                             metrics_decimals=metrics_decimals,
                                             target_decimals=target_decimals,
                                             num_folds=num_folds, stratified=stratified,
                                             kfolds_shuffle=kfolds_shuffle,
                                             data_split_seed=data_split_seed,
                                             project_location=project_location,
                                             output_dirname=output_dirname)

    bayes_blender.run()
    bayes_blender.save_weights()


if __name__ == '__main__':
    run_blender_kaggle_example(debug=True)
