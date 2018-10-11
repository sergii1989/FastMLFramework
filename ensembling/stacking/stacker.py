import os
from modeling.prediction import BaseEstimator


class Stacker(BaseEstimator):

    def __init__(self, train_df, test_df, target_column, index_column, classifier, predict_probability, class_index,
                 eval_metric, metrics_scorer, metrics_decimals=6, target_decimals=6, cols_to_exclude=[], num_folds=5,
                 stratified=False, kfolds_shuffle=True, cv_verbosity=1000, bagging=False, data_split_seed=789987,
                 model_seeds_list=[27], predict_test=True, project_location='', output_dirname=''):

        # Full path to solution directory
        path_output_dir = os.path.normpath(os.path.join(project_location, output_dirname))

        super(Stacker, self).__init__(
            train_df, test_df, target_column, index_column, classifier, predict_probability, class_index, eval_metric,
            metrics_scorer, metrics_decimals, target_decimals, cols_to_exclude, num_folds, stratified, kfolds_shuffle,
            cv_verbosity, bagging, data_split_seed, model_seeds_list, predict_test, path_output_dir
        )