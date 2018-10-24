import os
import json
import luigi
import pickle
import warnings
import numpy as np
import pandas as pd

from luigi.util import requires
from pyhocon.config_tree import ConfigTree
from modeling.prediction import Predictor
from ensembling.stacking.stacker import Stacker
from generic_tools.utils import get_metrics_scorer
from generic_tools.config_parser import ConfigFileHandler
from modeling.model_wrappers import get_wrapped_estimator
from modeling.feature_selection import load_feature_selector_class
from modeling.hyper_parameters_optimization import load_hp_optimization_class
from data_processing.preprocessing import downcast_datatypes

warnings.filterwarnings("ignore")


class TrainDataIngestion(luigi.Task):
    project_location = luigi.Parameter()  # type: str
    fg_output_dir = luigi.Parameter()  # type: str

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Settings for debug
        debug = config.get_bool('modeling_settings.debug')
        num_rows = config.get_int('modeling_settings.num_rows')

        # Load train and test data set from feature generation pool and downcast data types
        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.fg_output_dir, 'train.csv'))
        train_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None)) \
            .reset_index(drop=True)

        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.fg_output_dir, 'test.csv'))
        test_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None)) \
            .reset_index(drop=True)

        print('Train DF shape: {0}\n'.format(train_data.shape, train_data.info()))
        print('Test DF shape: {0}'.format(test_data.shape))

        new_train_name = os.path.normpath(os.path.join(self.project_location, self.fg_output_dir, 'train_new.csv'))
        new_test_name = os.path.normpath(os.path.join(self.project_location, self.fg_output_dir, 'test_new.csv'))
        print('\nSaving %s\n' % new_train_name)
        print('\nSaving %s\n' % new_test_name)

        train_data.to_csv(new_train_name, index=False)
        test_data.to_csv(new_test_name, index=False)

    def output(self):
        return {'train_data': luigi.LocalTarget(os.path.normpath(os.path.join(self.project_location,
                                                                              self.fg_output_dir,
                                                                              'train_new.csv'))),
                'test_data': luigi.LocalTarget(os.path.normpath(os.path.join(self.project_location,
                                                                             self.fg_output_dir,
                                                                             'test_new.csv')))}


@requires(TrainDataIngestion)
class FeatureSelection(luigi.Task):
    project_location = luigi.Parameter()  # type: str
    config_directory = luigi.Parameter()  # type: str
    config_file = luigi.Parameter()  # type: str
    fg_output_dir = luigi.Parameter()  # type: str
    fs_output_dir = luigi.Parameter()  # type: str
    feats_select_method = luigi.Parameter()  # type: str
    fs_results_file = 'optimal_features.txt'

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load train data set from feature generation pool
        train_data = pd.read_csv(self.input()['train_data'].path)

        # Categorical features for lgbm in feature_selection process
        categorical_feats = [f for f in train_data.columns if train_data[f].dtype == 'object']
        print('Number of categorical features: {0}'.format(len(categorical_feats)))
        for f_ in categorical_feats:
            train_data[f_], _ = pd.factorize(train_data[f_])
            train_data[f_] = train_data[f_].astype('category')
        cat_features = categorical_feats  # None

        # Extracting settings from config
        feature_selector = load_feature_selector_class(self.feats_select_method)
        target_column = config.get_string('raw_data_settings.target_column')
        index_column = config.get_string('raw_data_settings.index_column')
        int_threshold = config.get_int('features_selection.%s.int_threshold' % self.feats_select_method)
        num_boost_rounds = config.get_int('features_selection.%s.num_boost_rounds' % self.feats_select_method)
        nb_runs = config.get_int('features_selection.%s.nb_target_permutation_runs' % self.feats_select_method)
        fs_seed_val = config.get_int('modeling_settings.fs_seed_value')
        thresholds = config.get_list('features_selection.%s.eval_feats_removal_impact_on_cv_score.'
                                     'thresholds' % self.feats_select_method)
        n_thresholds = config.get_int('features_selection.%s.eval_feats_removal_impact_on_cv_score.'
                                      'n_thresholds' % self.feats_select_method)
        eval_metric = config.get_string('modeling_settings.cv_params.eval_metric')
        metrics_scorer = get_metrics_scorer(config.get('modeling_settings.cv_params.metrics_scorer'))
        metrics_decimals = config.get_int('modeling_settings.cv_params.metrics_decimals')
        num_folds = config.get_int('modeling_settings.cv_params.num_folds')
        stratified = config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = config.get_bool('modeling_settings.cv_params.kfolds_shuffle')

        lgbm_params_feats_exploration = dict(config.get_config('features_selection.%s.lgbm_params.'
                                                               'feats_exploration' % self.feats_select_method))
        lgbm_params_feats_selection = dict(config.get_config('features_selection.%s.lgbm_params.'
                                                             'feats_selection' % self.feats_select_method))
        # Initialize feature selection procedure
        features_selection = feature_selector(
            train_df=train_data, target_column=target_column, index_column=index_column,
            cat_features=cat_features, int_threshold=int_threshold,
            lgbm_params_feats_exploration=lgbm_params_feats_exploration,
            lgbm_params_feats_selection=lgbm_params_feats_selection,
            eval_metric=eval_metric, metrics_scorer=metrics_scorer,
            metrics_decimals=metrics_decimals, num_folds=num_folds,
            stratified=stratified, kfolds_shuffle=kfolds_shuffle,
            seed_val=fs_seed_val, project_location=project_location,
            output_dirname=self.fs_output_dir
        )

        # TODO: think how to pass func_1 and func_2 differently
        # Feature scoring func: option 1
        # See https://www.kaggle.com/ogrellier/feature-selection-with-null-importances/comments
        func_1 = lambda f_act_imps, f_null_imps: np.log(1e-10 + f_act_imps / (1 + np.percentile(f_null_imps, 75)))

        # Feature scoring function: option 2
        # See https://www.kaggle.com/ogrellier/feature-selection-with-null-importances/comments
        func_2 = lambda f_act_imps, f_null_imps: \
            100. * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size

        # Run feature selection process
        features_selection.get_actual_importances_distribution(num_boost_rounds=num_boost_rounds)
        features_selection.get_null_importances_distribution(nb_runs=nb_runs, num_boost_rounds=num_boost_rounds)
        features_selection.score_features(func_2)
        features_selection.eval_feats_removal_impact_on_cv_score(thresholds=thresholds, n_thresholds=n_thresholds)

        # Plot selected figures and store output results
        features_selection.feature_score_comparing_to_importance(ntop_feats=50, figsize_y=12, save=True)
        features_selection.plot_cv_results_vs_feature_threshold(save=True)
        features_selection.save_cv_results_vs_feats_score_thresh()
        features_selection.save_features_scores()

        # Finding best threshold with respect to CV and getting corresponding set of features
        opt_thres = features_selection.get_best_threshold(importance='gain_score', cv_asc_rank=True,
                                                          cv_std_asc_rank=False)
        opt_feats = features_selection.get_list_of_features(importance='gain_score', thresh=opt_thres)

        full_path_to_file = os.path.normpath(os.path.join(self.project_location,
                                                          self.fs_output_dir,
                                                          self.fs_results_file))
        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(opt_feats))

    def output(self):
        return luigi.LocalTarget(os.path.normpath(os.path.join(
            self.project_location, self.fs_output_dir, self.fs_results_file)))


class InitializeSingleModelPredictor(luigi.Task):
    project_location = luigi.Parameter()  # type: str
    config_directory = luigi.Parameter()  # type: str
    config_file = luigi.Parameter()  # type: str
    model = luigi.Parameter()  # type: str
    run_feature_selection = luigi.BoolParameter()  # type: bool
    fg_output_dir = luigi.Parameter()  # type: str
    fs_output_dir = luigi.Parameter()  # type: str
    solution_output_dir = luigi.Parameter()  # type: str
    output_pickle_file = 'predictor_initialized.pickle'

    def requires(self):
        requirements = {'data': self.clone(TrainDataIngestion)}
        if self.run_feature_selection:
            # Parsing config (actually it is cached)
            config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)
            feats_select_method = config.get_string('modeling_settings.%s.fs_method' % self.model)
            requirements['features'] = FeatureSelection(project_location=self.project_location,
                                                        config_directory=self.config_directory,
                                                        config_file = self.config_file,
                                                        fg_output_dir=self.fg_output_dir,
                                                        fs_output_dir=self.fs_output_dir,
                                                        feats_select_method=feats_select_method)
        return requirements

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load train and test data sets
        train_data = pd.read_csv(self.input()['data']['train_data'].path)
        test_data = pd.read_csv(self.input()['data']['test_data'].path)

        opt_feats = []
        if self.run_feature_selection:
            # Load set of features from feature_selection procedure
            with open(self.input()['features'].path, 'r') as f:
                opt_feats = json.load(f)

        # Extracting settings from config
        target_column = config.get_string('raw_data_settings.target_column')
        index_column = config.get_string('raw_data_settings.index_column')
        model_init_params = dict(config.get_config('single_model_init_params.%s' % self.model))
        estimator_wrapped = get_wrapped_estimator(self.model, model_init_params)
        predict_probability = config.get_bool('modeling_settings.%s.predict_probability' % self.model)
        class_label = config.get('modeling_settings.%s.class_label' % self.model)
        cols_to_exclude = config.get_list('modeling_settings.cols_to_exclude')
        bagging = config.get_bool('modeling_settings.%s.run_bagging' % self.model)
        predict_test = config.get_bool('modeling_settings.predict_test')
        eval_metric = config.get_string('modeling_settings.cv_params.eval_metric')
        metrics_scorer = get_metrics_scorer(config.get('modeling_settings.cv_params.metrics_scorer'))
        metrics_decimals = config.get_int('modeling_settings.cv_params.metrics_decimals')
        target_decimals = config.get_int('modeling_settings.cv_params.target_decimals')
        num_folds = config.get_int('modeling_settings.cv_params.num_folds')
        stratified = config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = config.get_bool('modeling_settings.cv_params.kfolds_shuffle')
        cv_verbosity = config.get_int('modeling_settings.cv_params.cv_verbosity')
        data_split_seed = config.get_int('modeling_settings.cv_params.data_split_seed')
        model_seeds_list = config.get_list('modeling_settings.model_seeds_list')

        # Initialize single model predictor
        predictor = Predictor(
            train_df=train_data[opt_feats] if self.run_feature_selection else train_data,
            test_df=test_data[opt_feats] if self.run_feature_selection else test_data,
            target_column=target_column, index_column=index_column,
            model=estimator_wrapped, predict_probability=predict_probability,
            class_label=class_label, eval_metric=eval_metric, metrics_scorer=metrics_scorer,
            metrics_decimals=metrics_decimals, target_decimals=target_decimals,
            cols_to_exclude=cols_to_exclude, num_folds=num_folds, stratified=stratified,
            kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity, bagging=bagging,
            predict_test=predict_test, data_split_seed=data_split_seed,
            model_seeds_list=model_seeds_list, project_location=project_location,
            output_dirname=self.solution_output_dir
        )

        full_path_to_file = os.path.join(self.project_location, self.solution_output_dir, self.output_pickle_file)
        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(predictor, f)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.solution_output_dir, self.output_pickle_file))


@requires(InitializeSingleModelPredictor)
class RunSingleModelHPO(luigi.Task):
    model = luigi.Parameter()  # type: str
    hpo_output_dir = luigi.Parameter()  # type: str
    hpo_results_file = 'optimized_hp.txt'

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load initialized single model predictor
        predictor = pickle.load(open(self.input().path, "rb"))

        # Extracting settings from config
        hpo_method = config.get_string('modeling_settings.%s.hpo_method' % self.model)
        hp_optimizator = load_hp_optimization_class(hpo_method)
        hpo_seed_val = config.get_int('modeling_settings.hpo_seed_value')
        init_points = config.get_int('hp_optimization.%s.init_points' % hpo_method)
        n_iter = config.get_int('hp_optimization.%s.n_iter' % hpo_method)
        hp_optimization_space = dict(config.get_config(
            'hp_optimization.%s.hpo_space.single_model_solution.%s' % (hpo_method, self.model)))

        # Initialize hyper-parameter optimizator
        hpo = hp_optimizator(
            predictor, hp_optimization_space=hp_optimization_space,
            init_points=init_points, n_iter=n_iter, seed_val=hpo_seed_val,
            project_location=project_location, output_dirname=self.hpo_output_dir
        )

        # Run optimization and save output results
        hpo.run()
        hpo.save_hpo_history()

        full_path_to_file = os.path.normpath(os.path.join(self.project_location,
                                                          self.hpo_output_dir,
                                                          self.hpo_results_file))
        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(hpo.best_params))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.hpo_output_dir, self.hpo_results_file))


class RunSingleModelPrediction(luigi.Task):
    project_location = luigi.Parameter()  # absolute path to project's main directory
    config_directory = luigi.Parameter()  # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # name of config file in the config sub-directory
    model = luigi.Parameter()  # name of estimator model
    run_feature_selection = luigi.BoolParameter()  # if True -> run feature selection
    run_hpo = luigi.BoolParameter()  # if True -> run hyper-parameters optimization
    run_bagging = luigi.BoolParameter()  # if True -> run bagging
    fg_output_dir = luigi.Parameter()  # feature generation directory (to be used as input for train data ingestion)
    fs_output_dir = luigi.Parameter()  # feature selection directory (where results of feature selection to be saved)
    hpo_output_dir = luigi.Parameter()  # hyper-parameters optimization directory (where results of hpo to be saved)
    solution_output_dir = luigi.Parameter()  # output directory where results of a single model prediction to be saved
    output_pickle_file = 'predictor_with_results.pickle'

    def requires(self):
        requirements = {'predictor': self.clone(InitializeSingleModelPredictor)}
        if self.run_hpo:
            requirements['hpo'] = self.clone(RunSingleModelHPO)
        return requirements

    def run(self):
        # Load initialized single model predictor
        predictor = pickle.load(open(self.input()['predictor'].path, "rb"))

        if self.run_hpo:
            # Load set of single model's best parameters from hyper-parameters optimization procedure
            with open(self.input()['hpo'].path, 'r') as f:
                best_params = json.load(f)

            # Re-initialize model with optimal parameters
            predictor.model.reinit_model_with_new_params(best_params)

        # Run CV and prediction of test data set
        predictor.run_cv_and_prediction()

        # Plot cv results at different seeds
        predictor.plot_cv_results_vs_used_seeds(save=True)

        # TODO: think how to pass class_names and labels_mapper differently
        # Class names for report & confusion matrix plot
        class_names = ['0', '1']
        labels_mapper = lambda x: 1 if x > 0.5 else 0  # for confusion matrix plot

        # Plot confusion matrix
        predictor.plot_confusion_matrix(class_names, labels_mapper, normalize=True, save=True)

        # Plot features importance
        predictor.plot_features_importance(n_features=40, save=True)

        # Save results
        predictor.save_oof_results()
        predictor.save_submission_results()

        full_path_to_file = os.path.join(self.project_location, self.solution_output_dir, self.output_pickle_file)
        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(predictor, f)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.solution_output_dir, self.output_pickle_file))


class BuildSolution(luigi.WrapperTask):
    project_location = luigi.Parameter()  # type: str
    config_directory = luigi.Parameter()  # type: str
    config_file = luigi.Parameter()  # type: str

    def requires(self):
        config_handler = ConfigFileHandler(self.project_location, self.config_directory, self.config_file)
        collection_input_parameters = config_handler.prepare_input_parameters_for_luigi_pipeline()
        for input_parameters in collection_input_parameters:
            yield RunSingleModelPrediction(**input_parameters)


class InitializeStacker(luigi.Task):

    # stacking_output_dir = config.get_string('stacker.full_path_output')
    # full_path_to_file   = os.path.normpath(os.path.join(project_location, stacking_output_dir, 'stacker.pickle'))

    def requires(self):
        return TrainDataIngestion(self.project_location, self.fg_output_dir)

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load train and test data sets
        train_data = pd.read_csv(self.input()['data']['train_data'].path)
        test_data = pd.read_csv(self.input()['data']['test_data'].path)

        # Read dict with OOF input file names, locations and short labels to be used when naming output files
        oof_input_files = dict(config.get_config('stacker.oof_input_files'))
        for key in oof_input_files:
            oof_input_files[key] = dict(oof_input_files[key])

        # Input settings for Stacker
        target_column = config.get_string('raw_data_settings.target_column')
        index_column = config.get_string('raw_data_settings.index_column')
        stacker_model = config.get_string('stacker.meta_model')
        stacker_init_params = dict(config.get_config('stacker.%s.init_params' % stacker_model))
        stacker_wrapped = get_wrapped_estimator(stacker_model, stacker_init_params)
        stacker_predict_probability = config.get_bool('stacker.%s.predict_probability' % stacker_model)
        class_label = config.get('modeling_settings.class_label')
        stacker_eval_metric = config.get_string('stacker.%s.eval_metric' % stacker_model)
        stacker_metrics_scorer = get_metrics_scorer(config.get('stacker.%s.metrics_scorer' % stacker_model))
        stacker_metrics_decimals = config.get_int('stacker.%s.metrics_decimals' % stacker_model)
        stacker_target_decimals = config.get_int('stacker.%s.target_decimals' % stacker_model)
        cols_to_exclude = config.get_list('modeling_settings.cols_to_exclude')
        num_folds = config.get_int('modeling_settings.cv_params.num_folds')
        stratified = config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = config.get_bool('modeling_settings.cv_params.kfolds_shuffle')
        cv_verbosity = config.get_int('modeling_settings.cv_params.cv_verbosity')
        stacker_bagging = config.get_bool('stacker.%s.run_bagging' % stacker_model)
        data_split_seed = config.get_int('modeling_settings.cv_params.data_split_seed')
        model_seeds_list = config.get_list('modeling_settings.model_seeds_list')

        # TODO: to add this logic
        # If True -> use raw features additionally to out-of-fold results
        stacker_use_raw_features = config.get_bool('stacker.%s.use_raw_features' % stacker_model)

        # Initializing stacker
        stacker = Stacker(
            oof_input_files=oof_input_files, train_df=train_data, test_df=test_data,
            target_column=target_column, index_column=index_column,
            stacker_model=stacker_wrapped, predict_probability=stacker_predict_probability,
            class_label=class_label, eval_metric=stacker_eval_metric, metrics_scorer=stacker_metrics_scorer,
            metrics_decimals=stacker_metrics_decimals, target_decimals=stacker_target_decimals,
            cols_to_exclude=cols_to_exclude, num_folds=num_folds, stratified=stratified,
            kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity, bagging=stacker_bagging,
            data_split_seed=data_split_seed, model_seeds_list=model_seeds_list,
            project_location=self.project_location, output_dirname=self.stacking_output_dir
        )

        print('\nSaving %s\n' % self.full_path_to_file)
        with open(self.full_path_to_file, 'wb') as f:
            pickle.dump(stacker, f)

    def output(self):
        return luigi.LocalTarget(self.full_path_to_file)


class RunStackerHPO(luigi.Task):

    # stacking_output_dir = config.get_string('stacker.full_path_output')
    # full_path_to_file   = os.path.normpath(os.path.join(project_location, stacking_output_dir, 'optim_hp.txt'))

    def requires(self):
        return InitializeStacker()

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load initialized stacker
        stacker = pickle.load(open(self.input().path, "rb"))
        stacker_model = config.get_string('stacker.meta_model')
        stacker_hpo_method = config.get_string('stacker.%s.hpo_method' % stacker_model)
        stacker_hp_optimizator = load_hp_optimization_class(stacker_hpo_method)
        stacker_hpo_space = dict(
            config.get_config('stacker.%s.%s.hpo_space' % (stacker_model, stacker_hpo_method)))
        stacker_hpo_init_points = config.get_int('stacker.%s.%s.init_points' % (stacker_model, stacker_hpo_method))
        stacker_hpo_n_iter = config.get_int('stacker.%s.%s.n_iter' % (stacker_model, stacker_hpo_method))
        stacker_hpo_seed_val = config.get_int('stacker.%s.%s.seed_value' % (stacker_model, stacker_hpo_method))

        # Initialize hyper-parameter optimizator
        stacker_hpo = stacker_hp_optimizator(
            stacker, hp_optimization_space=stacker_hpo_space, init_points=stacker_hpo_init_points,
            n_iter=stacker_hpo_n_iter, seed_val=stacker_hpo_seed_val, output_dir=self.stacking_output_dir
        )

        # Run optimization and save output results
        stacker_hpo.run()
        stacker_hpo.save_optimized_hp()
        stacker_hpo.save_hpo_history()

        print('\nSaving %s\n' % self.full_path_to_file)
        with open(self.full_path_to_file, 'w') as f:
            f.write(json.dumps(stacker_hpo.best_params))

    def output(self):
        return luigi.LocalTarget(self.full_path_to_file)


class RunStackerPrediction(luigi.Task):

    # stacking_output_dir = config.get_string('stacker.full_path_output')
    # full_path_to_file   = os.path.normpath(os.path.join(project_location, stacking_output_dir,
    #                                                    'stacker_with_results.pickle'))

    def requires(self):
        return {'stacker': InitializeStacker(),
                'hpo': RunStackerHPO()}

    def run(self):
        # Load initialized stacker
        stacker = pickle.load(open(self.input()['stacker'].path, "rb"))

        # Load set of stacker's best parameters from hyper-parameters optimization procedure
        with open(self.input()['hpo'].path, 'r') as f:
            best_params = json.load(f)

        # Re-initialize stacker with optimal parameters
        stacker.model.reinit_model_with_new_params(best_params)

        # Run CV and prediction of test data set
        stacker.run_cv_and_prediction()

        # Save results
        stacker.save_oof_results()
        stacker.save_submission_results()

        # Plot features importance
        # stacker.plot_features_importance(n_features=10, save=True)

        # Plot confusion matrix
        # TODO: think how to pass class_names and labels_mapper differently
        # Class names for report & confusion matrix plot
        class_names = ['0', '1']
        labels_mapper = lambda x: 1 if x > 0.5 else 0  # for confusion matrix plot
        stacker.plot_confusion_matrix(class_names, labels_mapper, normalize=True, save=True)

        print('\nSaving %s\n' % self.full_path_to_file)
        with open(self.full_path_to_file, 'wb') as f:
            pickle.dump(stacker, f)

    def output(self):
        return luigi.LocalTarget(self.full_path_to_file)


if __name__ == '__main__':
    # Location of the project and config file
    project_location = r'C:\Kaggle\home_credit_default_risk'  # os.getcwd()
    config_directory = 'configs'
    config_file = 'solution.conf'
    print('Location of project: {0}'.format(project_location))

    # stacker_model = config.get_string('stacker.meta_model')
    # stacker_run_hpo = config.get_bool('stacker.%s.run_hpo' % stacker_model)  # if True -> run HPO of stacker

    # Run pipeline this way
    luigi.build([BuildSolution(project_location, config_directory, config_file)], local_scheduler=False)

    # Or this way
    # luigi.run(main_task_cls=BuildSolution,
    #           cmdline_args=["--BuildSolution-project-location=C:\Kaggle\home_credit_default_risk",
    #                         "--BuildSolution-config-directory=configs",
    #                         "--BuildSolution-config-file=solution.conf",
    #                         "--workers=1"],
    #           local_scheduler=False)
