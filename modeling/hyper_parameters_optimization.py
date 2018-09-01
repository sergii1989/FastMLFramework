import gc
import numpy as np
import pandas as pd

from generic_tools.utils import timing
from bayes_opt import BayesianOptimization
from modeling.cross_validation import Predictor


class HyperParamOptimization(Predictor):

    def __call__(self, *args, **kwargs):

        self.num_folds = num_folds
        self.target = target
        self.eval_metric = eval_metric
        self.metrics_scorer = metrics_scorer
        self.cols_to_exclude = cols_to_exclude
        self.stratified = stratified
        self.kfolds_shuffle = kfolds_shuffle
        self.seeds_list = seeds_list
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.predict_test = predict_test

        self.best_params = None
        self.best_CV_score = None
        self.run_bayes_optimization()

    def hp_optimizer(self, **params):
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])

        clf = self.classifier(**params)

        _, _, _, _, cv_score = self._run_cv_one_seed(num_folds, target, eval_metric, metrics_scorer, cols_to_exclude,
                                                     stratified, kfolds_shuffle, seeds_list[0],
                                                     verbose, early_stopping_rounds)
        return cv_score

    @timing
    def run_bayes_optimization(self, init_points, n_iter, decimals=3, path_to_save_data=None):

        params = {'colsample_bytree': (0.8, 1),
                  'learning_rate': (.01, .02),
                  'num_leaves': (33, 35),
                  'subsample': (0.8, 1),
                  'max_depth': (7, 9),
                  'reg_alpha': (.03, .05),
                  'reg_lambda': (.06, .08),
                  'min_split_gain': (.01, .03),
                  'min_child_weight': (38, 40)}

        bo = BayesianOptimization(self.hp_optimizer, params)
        bo.maximize(init_points=init_points, n_iter=n_iter)

        best_params = bo.res['max']['max_params']
        best_CV_score = round(bo.res['max']['max_val'], decimals)
        print('\n'.join(['', '=' * 70, '\nMax CV score: {0}'.format(best_CV_score)]))

        self.best_params = best_params
        self.best_CV_score = best_CV_score


