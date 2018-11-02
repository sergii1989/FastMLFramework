import os
import logging
import warnings
import pandas as pd
import numpy.testing as npt

from collections import OrderedDict
from bayes_opt import BayesianOptimization
from ensembling.ensembler import Ensembler
from generic_tools.loggers import configure_logging
from generic_tools.utils import timing, create_output_dir

warnings.filterwarnings("ignore")

configure_logging()
_logger = logging.getLogger("ensembling.blender")


class Blender(object):

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

        self.ensembler = Ensembler()
        self.train_oof, self.test_oof = \
            self.ensembler.load_oof_target_and_test_data(oof_input_files, blend_bagged_results, train_df, test_df,
                                                         target_column, index_column, target_decimals, project_location)

        self.target_column = target_column
        self.index_column = index_column
        self.metrics_scorer = metrics_scorer
        self.metrics_decimals = metrics_decimals
        self.target_decimals = target_decimals

        # Full path to solution directory
        self.path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))
        create_output_dir(self.path_output_dir)

    def evaluate_results(self, **params):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def run(self, **kwargs):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()


class BayesOptimizationBlender(Blender):

    def __init__(self, oof_input_files, blend_bagged_results, train_df, test_df, target_column, index_column,
                 metrics_scorer, metrics_decimals=6, target_decimals=6, init_points=10, n_iter=15, seed_val=27,
                 project_location='', output_dirname=''):
        """
        This class implements blender based on Bayes Optimization procedure. It is trained on out-of-fold predictions
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

        self.blended_train_score = None  # type: float
        self.optimal_weights_df = None  # type: pd.DataFrame
        self.test_file_blended = None  # type: pd.DataFrame

    @staticmethod
    def _normalize_weights(best_params, precision=3):  # type: (dict, int) -> dict
        """
        This method normalizes raw weights from the Bayes Optimization process. The sum of all weights should be 1.
        :param best_params: raw weights from Bayes Optimization process
        :param precision: rounding of the weights (decimals)
        :return: optimized weights (normalized and rounded to the requested precision)
        """
        optimal_weights = {}
        s = sum(best_params.values())
        for p in best_params:
            optimal_weights[p] = round(best_params[p] / s, precision)
        npt.assert_almost_equal(sum(optimal_weights.values()), 1.0, precision)
        return optimal_weights

    def evaluate_results(self, **params):  # type: (dict) -> float
        """
        This method evaluates train prediction (in the process of Bayes Optimization) according to the given
        metrics scorer (from sklearn.metrics)
        :param params: dict with the models weight
        :return: metrics score
        """
        s = sum(params.values())
        for p in params:
            params[p] = params[p] / s
        test_pred = pd.DataFrame()

        feats = [f for f in self.train_oof.columns if f not in (self.target_column, self.index_column)]
        for f in self.train_oof[feats]:
            test_pred[f] = self.train_oof[f] * params[f]
        return self.metrics_scorer(self.train_oof[self.target_column], test_pred.mean(axis=1))

    @timing
    def run(self):
        """
        This method runs Bayes Search of optimal weights to the individual model's predictions with the goal of
        maximizing evaluation metrics score on the train data set. After optimal weights are found, apply them to
        the test predictions. Main outcome of this function is two attributes: self.optimal_weights_df -> pandas DF
        with the optimal weights and self.test_file_blended -> pandas DF with the blended test predictions.
        :return: None
        """

        _logger.info('Running Bayes Optimization...')
        feats = [f for f in self.train_oof.columns if f not in (self.target_column, self.index_column)]
        params = OrderedDict((c, (0, 1)) for c in self.train_oof[feats])
        bo = BayesianOptimization(self.evaluate_results, params, random_state=self.seed_val)
        bo.maximize(init_points=self.init_points, n_iter=self.n_iter)

        # Extracting max score and optimal weights (the raw ones, not normalized)
        best_params = bo.res['max']['max_params']
        best_score = round(bo.res['max']['max_val'], self.metrics_decimals)
        optimal_weights = self._normalize_weights(best_params)
        _logger.info('\n'.join(['', '=' * 70, '\nMax score: {0}'.format(best_score)]))
        _logger.info('Optimal weights:\n{0}'.format(optimal_weights))

        # Blending train results with optimal weights and verifying the consistency of max score
        prediction_train = self.train_oof[optimal_weights.keys()].mul(optimal_weights.values()).sum(axis=1)
        blended_train_score = round(self.metrics_scorer(self.train_oof[self.target_column], prediction_train),
                                    self.metrics_decimals)
        npt.assert_almost_equal(best_score, blended_train_score, self.metrics_decimals)
        self.blended_train_score = blended_train_score  # best metrics score of blended train OOF predictions

        # Creating pandas DF with optimal weights
        optimal_weights_df = pd.DataFrame(index=optimal_weights.keys(), data=optimal_weights.values(),
                                          columns=['weights']).sort_values(by='weights',
                                                                           ascending=False).rename_axis("model")
        self.optimal_weights_df = optimal_weights_df

        test_file_blended = pd.DataFrame()
        if self.index_column is not None and self.index_column != '':
            test_file_blended[self.index_column] = self.test_oof[self.index_column].values

        # If self.train_oof dataframe contains non-bagged out-of-fold predictions by single models, then the names of
        # columns in self.test_oof should be adjusted so to match the names in self.train_oof. In the case of test
        # dataframe, prediction column is called as self.target column, whereas for train - self.target column + '_OOF'
        cols_rename = {col: col + '_OOF' for col in self.test_oof if self.target_column in col}
        if len(cols_rename):
            self.test_oof.rename(columns=cols_rename, inplace=True)

        test_file_blended[self.target_column] = self.test_oof[optimal_weights.keys()]\
            .mul(optimal_weights.values()).sum(axis=1).round(self.target_decimals).values
        self.test_file_blended = test_file_blended

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
        # 'lgbm_fds1_tp__fts_1_bayes_hpos1'
        'a': {
            'path': r"single_model_solution\lightgbm\features_dataset_001\target_permutation_fts_001\bayes_hpos_001",
            'files': ['lgbm_bagged_OOF.csv', 'lgbm_bagged_SUBM.csv'],
        },
        # 'xgb_fds1_tp__fts_1_bayes_hpos1'
        'b': {
            'path': r"single_model_solution\xgboost\features_dataset_001\target_permutation_fts_001\bayes_hpos_001",
            'files': ['xgb_bagged_OOF.csv', 'xgb_bagged_SUBM.csv'],
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
