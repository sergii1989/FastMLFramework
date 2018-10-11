import os
import warnings
import pandas as pd
import numpy.testing as npt
from collections import OrderedDict
from bayes_opt import BayesianOptimization
from generic_tools.utils import create_output_dir

warnings.filterwarnings("ignore")


class Blender(object):

    def __init__(self, train_oof, test_subm, target_column, index_column, cols_to_use, metrics_scorer,
                 metrics_decimals=6, target_decimals=6, path_output_dir=''):
        """
        This is a base class for blending models prediction. The blender is trained on out-of-fold predictions (OOF)
        of the 1st (or 2nd) level models and applied to test submissions. The blender is optimized in a way to maximize
        evaluation metrics.
        :param train_oof: pandas DF with the train OOF predictions and target variable
        :param test_subm: pandas DF with the test predictions
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param cols_to_use: list of columns to be feed into blender
        :param metrics_scorer: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        :param metrics_decimals: round precision (decimals) of the metrics (e.g. used in printouts)
        :param target_decimals: round precision (decimals) of the target column
        :param path_output_dir: full path to directory where the results of blending to be stored
        """
        self.train_oof = train_oof
        self.test_subm = test_subm
        self.target_column = target_column
        self.index_column = index_column
        self.cols_to_use = cols_to_use
        self.metrics_scorer = metrics_scorer
        self.metrics_decimals = metrics_decimals
        self.target_decimals = target_decimals
        self._verify_input_data_is_correct()

        # Full path to solution directory
        self.path_output_dir = path_output_dir
        create_output_dir(self.path_output_dir)

    def _verify_input_data_is_correct(self):
        """
        This method is used to verify correctness of the provided data
        :return: None
        """
        assert self.target_column in self.train_oof.columns, \
            'Please add {target} column to the train_oof dataframe'.format(target=self.target_column)
        assert callable(self.metrics_scorer), 'metrics_scorer should be callable function'

        if self.index_column is not None and self.index_column != '':
            assert ((self.index_column in self.train_oof.columns) | (self.index_column in self.train_oof.index)), \
                'Please add {index} column to the train_oof dataframe'.format(index=self.index_column)
            assert ((self.index_column in self.test_subm.columns) | (self.index_column in self.test_subm.index)), \
                'Please add {index} column to the test_subm dataframe'.format(index=self.index_column)

        if 'sklearn.metrics' not in self.metrics_scorer.__module__:
            raise TypeError("metrics_scorer should be function from sklearn.metrics module. "
                            "Instead received {0}.".format(self.metrics_scorer.__module__))
        return

    def evaluate_results(self, **params):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def run(self, **kwargs):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()


class BayesOptimizationBlender(Blender):
    def __init__(self, train_oof, test_subm, target_column, index_column, cols_to_use, metrics_scorer,
                 metrics_decimals=6, target_decimals=6, project_location='', output_dirname=''):
        """
        This class implements blender based on Bayes Optimization procedure. It is trained on out-of-fold predictions
        of the 1st (or 2nd) level models and applied to test submissions. The blender is optimized in a way to maximize
        evaluation metrics.
        :param train_oof: pandas DF with the train OOF predictions and target variable
        :param test_subm: pandas DF with the test predictions
        :param target_column: target column (to be predicted)
        :param index_column: unique index column
        :param cols_to_use: list of columns to be feed into blender
        :param metrics_scorer: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        :param metrics_decimals: round precision (decimals) of the metrics (e.g. used in printouts)
        :param target_decimals: round precision (decimals) of the target column
        :param project_location: path to the project
        :param output_dirname: name of directory where to save results of blending procedure
        """

        # Full path to solution directory
        path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))

        super(BayesOptimizationBlender, self).__init__(
            train_oof, test_subm, target_column, index_column, cols_to_use,
            metrics_scorer, metrics_decimals, target_decimals, path_output_dir
        )
        self.blended_train_score = None  # type: float
        self.optimal_weights_df = None  # type: pd.DataFrame
        self.test_file_blended = None  # type: pd.DataFrame

    @staticmethod
    def _normalize_weights(best_params, precision=3):  # type: (dict) -> dict
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
        for f in self.cols_to_use:
            test_pred[f] = self.train_oof[f] * params[f]
        return self.metrics_scorer(self.train_oof[self.target_column], test_pred.mean(axis=1))

    def run(self, init_points, n_iter):
        """
        This method runs Bayes Search of optimal weights to the individual model's predictions with the goal of
        maximizing evaluation metrics score on the train data set. After optimal weights are found, apply them to
        the test predictions. Main outcome of this function is two attributes: self.optimal_weights_df -> pandas DF
        with the optimal weights and self.test_file_blended -> pandas DF with the blended test predictions.
        :param init_points: number of initial points in Bayes Optimization procedure
        :param n_iter: number of iteration in Bayes Optimization procedure
        :return: None
        """

        # TODO: to refactor run() function so to make it general for all type of optimizers

        print('\nRunning Bayes Optimization...')
        params = OrderedDict((c, (0, 1)) for c in self.train_oof[self.cols_to_use])
        bo = BayesianOptimization(self.evaluate_results, params)
        bo.maximize(init_points=init_points, n_iter=n_iter)

        # Extracting max score and optimal weights (the raw ones, not normalized)
        best_params = bo.res['max']['max_params']
        best_score = round(bo.res['max']['max_val'], self.metrics_decimals)
        optimal_weights = self._normalize_weights(best_params)
        print('\n'.join(['', '=' * 70, '\nMax score: {0}'.format(best_score)]))
        print('Optimal weights:\n{0}'.format(optimal_weights))

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
            test_file_blended[self.index_column] = self.test_subm[self.index_column].values

        test_file_blended[self.target_column] = self.test_subm[optimal_weights.keys()]\
            .mul(optimal_weights.values()).sum(axis=1).round(self.target_decimals).values
        self.test_file_blended = test_file_blended

    def save_weights(self):
        """
        This method saves Bayes Optimized weights to the disc.
        :return: None
        """
        filename = '_'.join(list(self.optimal_weights_df.index)) + '_' + str(self.blended_train_score) + '.csv'
        output_figname = ('\\'.join([self.path_output_dir, filename]))
        print('\nSaving optimal weights DF into %s' % output_figname)
        self.optimal_weights_df.to_csv(output_figname)


def main():
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes

    target_decimals = 2
    metrics_decimals = 4

    n_iter = 5
    init_points = 10

    target_column = 'TARGET'
    index_column = 'SK_ID_CURR'
    metrics_scorer = roc_auc_score

    project_location = 'C:\Kaggle\home_credit'  # ''
    output_dirname = 'solution'  # ''

    raw_data_location = r'C:\Kaggle\kaggle_home_credit_default_risk\raw_data'
    path_to_results = r'C:\Kaggle\kaggle_home_credit_default_risk\models\results'

    filenames = \
        [
            'xgb_1',
            'lgbm_13'
        ]

    train_oof = []
    for filename in filenames:
        path_to_data = path_to_results + '\\' + filename + '_OOF.csv'
        df_temp = pd.read_csv(path_to_data).round(metrics_decimals).rename(columns={target_column + '_OOF': filename})
        df_temp[filename + '_rank'] = df_temp[filename].rank(method='min')
        train_oof.append(df_temp)
    train_oof = pd.concat(train_oof, axis=1).reset_index(drop=True)

    test_subm = []
    for filename in filenames:
        path_to_data = path_to_results + '\\' + filename + '.csv'
        df_temp = pd.read_csv(path_to_data)[target_column].round(metrics_decimals).rename(index=filename).to_frame()
        df_temp[filename + '_rank'] = df_temp[filename].rank(method='min')
        test_subm.append(df_temp)
    test_subm = pd.concat(test_subm, axis=1).reset_index(drop=True)

    target_columns_oof = filenames  # raw predicted target (oof)
    # rank_cols_oof = [filename + '_rank' for filename in filenames]  # rank transformed predicted values of target
    print('Shape OOF for train: {0}'.format(train_oof.shape))
    print('Shape preds for test: {0}'.format(test_subm.shape))

    cols_to_use = target_columns_oof

    application_train = downcast_datatypes(pd.read_csv(''.join([raw_data_location, '/application_train.csv']),
                                                       usecols=[index_column, target_column, 'CODE_GENDER']))
    application_train = application_train.loc[application_train['CODE_GENDER'] != 'XNA',
                                              [index_column, target_column]].reset_index(drop=True)
    application_test = downcast_datatypes(pd.read_csv(''.join([raw_data_location, '/application_test.csv']),
                                                      usecols=[index_column])).reset_index(drop=True)

    train_oof = pd.concat([application_train, train_oof], axis=1)
    test_subm = pd.concat([application_test, test_subm], axis=1)

    bayes_blender = BayesOptimizationBlender(train_oof=train_oof, test_subm=test_subm,
                                             target_column=target_column, index_column=index_column,
                                             cols_to_use=cols_to_use, metrics_scorer=metrics_scorer,
                                             metrics_decimals=metrics_decimals,
                                             target_decimals=target_decimals,
                                             project_location=project_location,
                                             output_dirname=output_dirname
                                             )

    bayes_blender.run(init_points, n_iter)
    bayes_blender.save_weights()


if __name__ == '__main__':
    main()
