import os
import logging
import warnings
import numpy as np
import pandas as pd
import numpy.testing as npt

from shutil import copyfile
from collections import OrderedDict
from modeling.prediction import Predictor
from bayes_opt import BayesianOptimization
from ensembling.ensembler import Ensembler
from generic_tools.loggers import configure_logging
from generic_tools.utils import timing, create_output_dir

warnings.filterwarnings("ignore")

configure_logging()
_logger = logging.getLogger("ensembling.blender")


class Blender(object):
    FILENAME_TRAIN_OOF_RESULTS = 'train_OOF.csv'
    FILENAME_TEST_RESULTS = 'test.csv'

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

        # float_format = '%.{0}f'.format(str(self.metrics_decimals)) if self.metrics_decimals > 0 else None
        # full_path_to_file = os.path.join(self.path_output_dir, self.FILENAME_CV_RESULTS)
        # _logger.info('Saving CV results into %s' % full_path_to_file)
        # self.oof_eval_results.to_csv(full_path_to_file, index=False, float_format=float_format)

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

    def __init__(self, oof_input_files, blend_bagged_results, train_df, test_df, target_column, index_column,
                 metrics_scorer, metrics_decimals=6, target_decimals=6, init_points=10, n_iter=15, seed_val=27,
                 project_location='', output_dirname=''):
        """
        This class implements blending method based on Bayes Optimization procedure. It is trained on out-of-fold
        predictions of the 1st (or 2nd) level models and applied to the test submission. The blender is optimized in
        a way to maximize evaluation metrics.

          BayesOptimizationBlender is similar to the sklearn's VotingClassifier but with the bayes optimized weights.

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
        :param init_points: number of initial points in Bayes Optimization procedure
        :param n_iter: number of iteration in Bayes Optimization procedure
        :param seed_val: seed for numpy random generator
        :param project_location: path to the project
        :param output_dirname: name of directory where to save results of blending procedure
        """

        super(BayesOptimizationBlender, self).__init__(
            oof_input_files, blend_bagged_results, train_df, test_df, target_column, index_column, metrics_scorer,
            metrics_decimals, target_decimals, project_location, output_dirname
        )

        # Bayes optimization settings
        self.init_points = init_points
        self.n_iter = n_iter
        self.seed_val = seed_val

        # Voting classifier settings
        self.voting_type = self._detect_voting_type()

        # Results
        self.blended_train_score = None  # type: float
        self.optimal_weights_df = None  # type: pd.DataFrame

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
        # each class. Need to think how to pass an explicit flag to use np.argmax() so to compute labels

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
        weights = [params[feat] / s for feat in feats]  # this trick is needed because **params not preserves order p2.7
        test_pred = self._run_voting(oof_data=self.train_oof[feats], weights=weights)
        return self.metrics_scorer(self.train_oof[self.target_column], test_pred)

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
        the test predictions. Main outcome of this function is two attributes: self.optimal_weights_df -> pandas DF
        with the optimal weights and self.sub_preds -> pandas DF with the blended test predictions.
        :return: None
        """

        _logger.info('Running Bayes Optimization...')
        feats = [f for f in self.train_oof.columns if f not in (self.target_column, self.index_column)]
        params = OrderedDict((c, (0, 1)) for c in feats)
        bo = BayesianOptimization(self.evaluate_results, params, random_state=self.seed_val)
        bo.maximize(init_points=self.init_points, n_iter=self.n_iter)

        # Extracting max score and optimal weights (the raw ones, not normalized)
        best_params = OrderedDict((c, bo.res['max']['max_params'][c]) for c in feats)
        best_score = round(bo.res['max']['max_val'], self.metrics_decimals)
        optimal_weights = self._normalize_weights(best_params)
        _logger.info('\n'.join(['', '=' * 70, '\nMax score: {0}'.format(best_score)]))
        _logger.info('Optimal weights:\n{0}'.format(optimal_weights))

        # Creating pandas DF with optimal weights
        optimal_weights_df = pd.DataFrame(index=optimal_weights.keys(), data=optimal_weights.values(),
                                          columns=['weights']).sort_values(by='weights',
                                                                           ascending=False).rename_axis("model")
        self.optimal_weights_df = optimal_weights_df

        # Blending train OOF predictions with the optimal weights
        blended_train_preds = self._run_voting(oof_data=self.train_oof[optimal_weights.keys()],
                                               weights=optimal_weights.values())
        oof_preds_df = self._prepare_results(blended_train_preds, is_oof_prediction=True)
        self.oof_preds = oof_preds_df

        # TODO: Think if assert_almost_equal is needed when bayes opt is performed in CV manner
        blended_train_score = round(self.metrics_scorer(self.train_oof[self.target_column], blended_train_preds),
                                    self.metrics_decimals)
        npt.assert_almost_equal(best_score, blended_train_score, self.metrics_decimals)
        self.blended_train_score = blended_train_score  # best metrics score of blended train OOF predictions

        # Blending test OOF predictions with the optimal weights

        # If self.train_oof dataframe contains non-bagged out-of-fold predictions by single models, then the names of
        # columns in self.test_oof should be adjusted so to match the names in self.train_oof. In the case of test
        # dataframe, prediction column is called as self.target column, whereas for train - self.target column + '_OOF'
        cols_rename = {col: col + '_OOF' for col in self.test_oof if self.target_column in col}
        if len(cols_rename):
            self.test_oof.rename(columns=cols_rename, inplace=True)
        blended_test_preds = self._run_voting(oof_data=self.test_oof[optimal_weights.keys()],
                                              weights=optimal_weights.values())
        sub_preds_df = self._prepare_results(blended_test_preds, is_oof_prediction=False)
        self.sub_preds = sub_preds_df

    def save_weights(self):
        """
        This method saves Bayes Optimized weights to the disc.
        :return: None
        """
        filename = "_".join(['blender_optimal_weights', str(self.blended_train_score)]) + '.csv'
        output_figname = os.path.join(self.path_output_dir, filename)
        _logger.info('Saving optimal weights DF into %s' % output_figname)
        self.optimal_weights_df.to_csv(output_figname)


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

    blend_bagged_results = True
    project_location = 'c:\Kaggle\home_credit_default_risk'  # ''
    output_dirname = ''  # 'solution'
    target_column = 'TARGET'
    index_column = 'SK_ID_CURR'
    metrics_scorer = roc_auc_score
    target_decimals = 2
    metrics_decimals = 4
    n_iter = 2
    init_points = 2
    seed_val = 27

    bayes_blender = BayesOptimizationBlender(oof_input_files=oof_input_files,
                                             blend_bagged_results=blend_bagged_results,
                                             train_df=train_data,
                                             test_df=test_data,
                                             target_column=target_column,
                                             index_column=index_column,
                                             metrics_scorer=metrics_scorer,
                                             metrics_decimals=metrics_decimals,
                                             target_decimals=target_decimals,
                                             init_points=init_points,
                                             n_iter=n_iter,
                                             seed_val=seed_val,
                                             project_location=project_location,
                                             output_dirname=output_dirname
                                             )
    bayes_blender.run()
    bayes_blender.save_weights()


if __name__ == '__main__':
    run_blender_kaggle_example(debug=True)
