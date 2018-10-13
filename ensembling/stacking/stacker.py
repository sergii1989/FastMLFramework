import os
import gc
import pandas as pd

from pandas import testing as pdt
from modeling.prediction import BaseEstimator
from data_processing.preprocessing import downcast_datatypes


class Stacker(BaseEstimator):

    def __init__(self, stack_input, train_df, test_df, target_column, index_column, model, predict_probability,
                 class_index, eval_metric, metrics_scorer, metrics_decimals=6, target_decimals=6, cols_to_exclude=[],
                 num_folds=5, stratified=False, kfolds_shuffle=True, cv_verbosity=1000, bagging=False,
                 data_split_seed=789987, model_seeds_list=[27], predict_test=True, project_location='',
                 output_dirname=''):

        # Full path to solution directory
        path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))

        self._verify_completeness_of_stack_input_data(stack_input)
        self.stack_input = stack_input  # type: dict

        train_oof, test_oof = self.load_oof_target_and_test_sets(project_location, train_df, test_df,
                                                                 target_column, index_column, target_decimals)
        self.train_oof = train_oof
        self.test_oof = test_oof

        super(Stacker, self).__init__(
            train_oof, test_oof, target_column, index_column, model, predict_probability, class_index, eval_metric,
            metrics_scorer, metrics_decimals, target_decimals, cols_to_exclude, num_folds, stratified, kfolds_shuffle,
            cv_verbosity, bagging, data_split_seed, model_seeds_list, predict_test, path_output_dir
        )

    @staticmethod
    def _verify_completeness_of_stack_input_data(stack_input):
        for solution in stack_input.values():
            assert len(solution['files']) >= 2, (
                "There should be at least two input files (1 with '_OOF' and 1 with '_SUBM' suffixes) for each "
                "provided path. Instead got: %s" % solution['files'])
            assert len(filter(lambda x: 'OOF' in x, solution['files'])) >= 1, (
                "There should be at least one input file containing train out-of-fold predictions for each "
                "provided path (i.e. file with '_OOF' suffix)")
            assert len(filter(lambda x: 'SUBM' in x, solution['files'])) >= 1, (
                "There should be at least one input file containing test predictions for each "
                "provided path (i.e. file with '_SUBM' suffix)")

    @staticmethod
    def _verify_consistency_of_input_data(meta_data, model_data, target_column, index_column, is_train=True):
        assert meta_data.shape[0] == model_data.shape[0]
        if is_train:
            assert target_column in meta_data
            pdt.assert_series_equal(meta_data[target_column], model_data[target_column])
        if index_column is not None and index_column != '':
            assert index_column in meta_data
            pdt.assert_series_equal(meta_data[index_column], model_data[index_column])

    def _prepare_stacking_input_dataset(self, main_df, index_column, target_column, list_preds_df, target_decimals):
        df = pd.concat(list_preds_df, axis=1).round(target_decimals)

        # Convert to int if target rounding precision is 0 decimals
        if target_decimals == 0:
            df = df.astype(int)

        # Add index column to the beginning of DF if index column is valid
        if index_column is not None and index_column != '':
            index_values = main_df[index_column].values
            df.insert(loc=0, column=index_column, value=index_values)

        # Add column with real target values to OOF dataframe
        if target_column in main_df and main_df[target_column].notnull().all():
            df[target_column] = main_df[target_column].values

        return df

    def load_oof_target_and_test_sets(self, project_location, train_df, test_df, target_column, index_column, target_decimals):
        list_train_oof_df = []
        list_test_preds_df = []

        for results_suffix, solution_details in self.stack_input.iteritems():
            for f in solution_details['files']:
                full_path = os.path.normpath(os.path.join(project_location, solution_details['path'], f))
                if 'OOF' in f:
                    df = downcast_datatypes(pd.read_csv(full_path))
                    self._verify_consistency_of_input_data(df, train_df, target_column, index_column, is_train=True)
                    df = df.loc[:, ~df.columns.isin([index_column, target_column])]
                    df.columns = ['_'.join([results_suffix, col]) for col in df.columns]
                    list_train_oof_df.append(df)
                elif 'SUBM' in f:
                    df = downcast_datatypes(pd.read_csv(full_path))
                    self._verify_consistency_of_input_data(df, test_df, target_column, index_column, is_train=False)
                    df = df.loc[:, df.columns != index_column]
                    df.columns = ['_'.join([results_suffix, col]) for col in df.columns]
                    list_test_preds_df.append(df)
                else:
                    raise (ValueError, 'Filename %s does not contain OOF or SUBM suffix.')

        oof_train_df = self._prepare_stacking_input_dataset(train_df, index_column, target_column,
                                                            list_train_oof_df, target_decimals)
        oof_test_df = self._prepare_stacking_input_dataset(test_df, index_column, target_column,
                                                           list_test_preds_df, target_decimals)

        del list_train_oof_df; list_test_preds_df; gc.collect()
        return oof_train_df, oof_test_df


def run_stacker_kaggle_example(stacker_model='logistic_regression', debug=True):
    import warnings
    from sklearn.metrics import roc_auc_score
    from data_processing.preprocessing import downcast_datatypes
    from solution_pipeline.create_solution import get_wrapped_estimator

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

    # Location and names of data sets with OOF and test predictions to be fed into stacker
    stack_input = {
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

    # Parameters
    class_index = 1  # in Target
    project_location = 'c:\Kaggle\home_credit_default_risk'  # ''
    output_dirname = ''  # 'solution'
    target_column = 'TARGET'
    index_column = 'SK_ID_CURR'
    metrics_scorer = roc_auc_score
    metrics_decimals = 4
    target_decimals = 2
    num_folds = 5
    stratified = True
    kfolds_shuffle = True
    cv_verbosity = 1000
    bagging = False
    predict_test = True
    data_split_seed = 789987
    model_seeds_list = [27]
    cols_to_exclude = ['TARGET', 'SK_ID_CURR']

    if stacker_model is 'lightgbm':
        predict_probability = True  # if True -> use estimator.predict_proba(), otherwise -> estimator.predict()
        eval_metric = 'auc'
        params = {
            'boosting_type': 'gbdt',  # gbdt, gbrt, rf, random_forest, dart, goss
            'objective': 'binary',
            'num_leaves': 12,  # 32
            'max_depth': 4,  # 8
            'learning_rate': 0.1,  # 0.01
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'min_split_gain': 0.001,  # 0.02
            'min_child_weight': 1,  # 40
            'subsample': 0.7,  # 0.87
            'colsample_bytree': 0.5,  # 0.94
            'reg_alpha': 0.0,  # 0.04
            'reg_lambda': 0.0,  # 0.073
            'nthread': -1,
            'verbose': -1
        }
    elif stacker_model is 'logistic_regression':
        predict_probability = True  # if True -> use estimator.predict_proba(), otherwise -> estimator.predict()
        eval_metric = 'auc'
        params = {
            'penalty': 'l1',  # norm used in the penalization ('l1' or 'l2'), default 'l2'
            'tol': 0.00001,  # tolerance for stopping criteria (default 0.0001)
            'C': 0.1,  # inverse of regularization strength, default 1.0. Smaller C values -> stronger regularization
            'fit_intercept': True,  # constant bias to be added to the decision function, default: True
            'class_weight': 'balanced',  # default: None.
            'solver': 'liblinear',  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', default: 'liblinear'
            'max_iter': 10000,  # only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations.
            'multi_class': 'ovr',  # 'ovr', 'multinomial', default: 'ovr' -> a binary problem is fit for each label
            'verbose': 0,  # for the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
            'n_jobs': -1
        }
    else:
        # Linear Regression
        predict_probability = False  # if True -> use estimator.predict_proba(), otherwise -> estimator.predict()
        eval_metric = 'rmse'
        params = {
            'fit_intercept': True,  # constant bias to be added to the decision function, default: True
            'normalize': False,  # If True, regressors will be normalized by subtr. the mean and dividing by l2-norm.
            'n_jobs': -1
        }

    stacker_wrapped = get_wrapped_estimator(stacker_model, params)

    stacker = Stacker(
        stack_input=stack_input, train_df=train_data, test_df=test_data, target_column=target_column,
        index_column=index_column, model=stacker_wrapped, predict_probability=predict_probability,
        class_index=class_index, eval_metric=eval_metric, metrics_scorer=metrics_scorer,
        metrics_decimals=metrics_decimals, target_decimals=target_decimals, cols_to_exclude=cols_to_exclude,
        num_folds=num_folds, stratified=stratified, kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity,
        bagging=bagging, predict_test=predict_test, data_split_seed=data_split_seed, model_seeds_list=model_seeds_list,
        project_location=project_location, output_dirname=output_dirname
    )
    stacker.run_cv_and_prediction()


if __name__ == '__main__':
    # run_stacker_kaggle_example(stacker_model='lightgbm')
    run_stacker_kaggle_example(stacker_model='logistic_regression')
    # run_stacker_kaggle_example(stacker_model='linear_regression')
