import warnings
import pandas as pd
import numpy.testing as npt
from collections import OrderedDict
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")

class Blender(object):
    def __init__(self, train_oof, test_subm, target_col, index_col, cols_to_use, metrics_scorer):
        """

        :param train_oof:
        :param test_subm:
        :param target_col:
        :param index_col:
        :param cols_to_use:
        :param metrics_scorer: from sklearn.metrics http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        """
        self.train_oof = train_oof
        self.test_subm = test_subm
        self.target_col = target_col
        self.index_col = index_col
        self.cols_to_use = cols_to_use
        self.metrics_scorer = metrics_scorer
        self._verify_input_data_is_correct()

    def _verify_input_data_is_correct(self):
        assert self.target_col in self.train_oof.columns, \
            'Please add {target} column to the train_oof dataframe'.format(target=self.target_col)
        assert ((self.index_col in self.train_oof.columns) | (self.index_col in self.train_oof.index)), \
            'Please add {index} column to the train_oof dataframe'.format(index=self.index_col)
        assert ((self.index_col in self.test_subm.columns) | (self.index_col in self.test_subm.index)), \
            'Please add {index} column to the test_subm dataframe'.format(index=self.index_col)
        assert callable(self.metrics_scorer), 'metrics_scorer should be callable function'
        if not 'sklearn.metrics' in self.metrics_scorer.__module__:
            raise TypeError("metrics_scorer should be function from sklearn.metrics module. " \
                   "Instead received {0}.".format(self.metrics_scorer.__module__))
        return

    def evaluate_results(self, **params):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def run(self, **kwargs):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()


class BayesOptimizationBlender(Blender):

    def __init__(self, train_oof, test_subm, target_col, index_col, cols_to_use, metrics_scorer):
        super(BayesOptimizationBlender, self).__init__(train_oof, test_subm, target_col, index_col,
                                                       cols_to_use, metrics_scorer)
        self.blended_train_score = None
        self.optimal_weights_df = None

    def evaluate_results(self, **params):
        """

        :param params:
        :return:
        """
        s = sum(params.values())
        for p in params:
            params[p] = params[p] / s
        test_pred = pd.DataFrame()
        for f in self.cols_to_use:
            test_pred[f] = self.train_oof[f] * params[f]
        return self.metrics_scorer(self.train_oof[self.target_col], test_pred.mean(axis=1))

    def _normalize_weights(self, best_params, precision=3):
        """

        :param best_params:
        :param precision:
        :return:
        """
        optimal_weights = {}
        s = sum(best_params.values())
        for p in best_params:
            optimal_weights[p] = round(best_params[p] / s, precision)
        # assert sum(optimal_weights.values()) == 1.0, 'Sum of weights should be 1.0'
        return optimal_weights

    def save_weigths(self, optimal_weights_df, path_to_save_data):
        """

        :param optimal_weights_df:
        :param path_to_save_data:
        :return:
        """
        filename = '_'.join(list(optimal_weights_df.index)) + '_' + str(self.blended_train_score) + '.csv'
        output_figname = ('\\'.join([path_to_save_data, filename]))
        print('\nSaving optimal weights DF into %s' % output_figname)
        optimal_weights_df.to_csv(output_figname)

    def run(self, init_points, n_iter, decimals=3, path_to_save_data=None):
        """

        :param init_points:
        :param n_iter:
        :param decimals:
        :param path_to_save_data:
        :return:
        """
        print '\nRunning Bayes Optimization...'
        if path_to_save_data is None:
            print 'Optimal weights will not be stored to the disk since path_to_save_data={}\n'.format(path_to_save_data)
        else:
            print 'Optimal weights will be stored to {}\n'.format(path_to_save_data)

        params = OrderedDict((c, (0, 1)) for c in self.train_oof[self.cols_to_use])
        bo = BayesianOptimization(self.evaluate_results, params)
        bo.maximize(init_points=init_points, n_iter=n_iter)

        # Extracting max CV score and optimal weights (not normalized yet)
        best_params = bo.res['max']['max_params']
        best_CV_score = round(bo.res['max']['max_val'], decimals)
        optimal_weights = self._normalize_weights(best_params)
        print('\n'.join(['', '=' * 70, '\nMax CV score: {0}'.format(best_CV_score)]))
        print('Optimal weights:\n{0}'.format(optimal_weights))

        # Blending train results with optimal weights and verifying the consistency of max CV score
        prediction_train = self.train_oof[optimal_weights.keys()].mul(optimal_weights.values()).sum(axis=1)
        blended_train_score = round(self.metrics_scorer(self.train_oof[self.target_col], prediction_train), decimals)
        npt.assert_almost_equal(best_CV_score, blended_train_score, decimals)
        self.blended_train_score = blended_train_score

        # Creating pandas DF with optimal weights
        optimal_weights_df = pd.DataFrame(index=optimal_weights.keys(), data=optimal_weights.values(),
                                          columns=['weights']).sort_values(by='weights', ascending=False)\
                                         .rename_axis("model")
        self.optimal_weights_df = optimal_weights_df

        if path_to_save_data is not None:
            self.save_weigths(optimal_weights_df, path_to_save_data)

        test_file_blended = self.test_subm[[self.index_col]].copy()
        test_file_blended[self.target_col] = self.test_subm[optimal_weights.keys()]\
            .mul(optimal_weights.values()).sum(axis=1).round(decimals).values
        self.test_file_blended = test_file_blended

def main():
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes

    decimals = 4
    n_iter = 5
    init_points = 60

    target_col = 'TARGET'
    index_col = 'SK_ID_CURR'
    metrics_scorer = roc_auc_score

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
        df_temp = pd.read_csv(path_to_data).round(decimals).rename(columns={target_col + '_OOF': filename})
        df_temp[filename + '_rank'] = df_temp[filename].rank(method='min')
        train_oof.append(df_temp)
    train_oof = pd.concat(train_oof, axis=1).reset_index(drop=True)

    test_subm = []
    for filename in filenames:
        path_to_data = path_to_results + '\\' + filename + '.csv'
        df_temp = pd.read_csv(path_to_data)[target_col].round(decimals).rename(index=filename).to_frame()
        df_temp[filename + '_rank'] = df_temp[filename].rank(method='min')
        test_subm.append(df_temp)
    test_subm = pd.concat(test_subm, axis=1).reset_index(drop=True)

    target_cols_oof = filenames  # raw predicted target (oof)
    rank_cols_oof = [filename + '_rank' for filename in filenames]  # rank transformed predicted values of target
    print 'Shape OOF for train: ', train_oof.shape
    print 'Shape preds for test: ', test_subm.shape

    cols_to_use = target_cols_oof

    application_train = downcast_datatypes(pd.read_csv(''.join([raw_data_location, '/application_train.csv']),
                                                       usecols=[index_col, target_col, 'CODE_GENDER']))
    application_train = application_train.loc[application_train['CODE_GENDER'] != 'XNA',
                                             [index_col, target_col]].reset_index(drop=True)
    application_test = downcast_datatypes(pd.read_csv(''.join([raw_data_location, '/application_test.csv']),
                                                      usecols=[index_col])).reset_index(drop=True)

    train_oof = pd.concat([application_train, train_oof], axis=1)
    test_subm = pd.concat([application_test, test_subm], axis=1)

    bayes_blender = BayesOptimizationBlender(train_oof, test_subm, target_col, index_col, cols_to_use, metrics_scorer)
    bayes_blender.run(init_points, n_iter, decimals, path_to_save_data=path_to_results)

if __name__ == '__main__':
    main()
