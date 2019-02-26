import os
import json
import luigi
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

from luigi.util import requires
from modeling.prediction import Predictor
from ensembling.stacking.stacker import Stacker
from generic_tools.loggers import configure_logging
from modeling.model_wrappers import get_wrapped_estimator
from data_processing.preprocessing import downcast_datatypes
from ensembling.blending.blender import BayesOptimizationBlender
from modeling.feature_selection import load_feature_selector_class
from modeling.hyper_parameters_optimization import load_hp_optimization_class
from generic_tools.config_parser import ConfigFileHandler
from generic_tools.utils import (get_metrics_scorer, generate_single_model_solution_id_key,
                                 merge_two_dicts, create_output_dir)
warnings.filterwarnings("ignore")

# Setting up logging interface of luigi
luigi.interface.setup_interface_logging(level_name='INFO')

# Setting up logging interface of FastML Framework
configure_logging()
_logger = logging.getLogger("solution_pipeline")


class TrainDataIngestion(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Settings for debug
        debug = config.get_bool('modeling_settings.debug')
        num_rows = config.get_int('modeling_settings.num_rows')
        train_file = config.get_string('features_generation.train_file')
        test_file = config.get_string('features_generation.test_file')

        # Load train and test data set from feature generation pool and downcast data types
        train_full_path = os.path.normpath(os.path.join(self.project_location, self.fg_output_dir, train_file))
        train_data = downcast_datatypes(pd.read_csv(train_full_path, nrows=num_rows if debug else None)) \
            .reset_index(drop=True)
        _logger.info('Train DF shape: {0}'.format(train_data.shape, train_data.info()))

        test_full_path = os.path.normpath(os.path.join(self.project_location, self.fg_output_dir, test_file))
        test_data = downcast_datatypes(pd.read_csv(test_full_path, nrows=num_rows if debug else None)) \
            .reset_index(drop=True)
        _logger.info('Test DF shape: {0}'.format(test_data.shape))

        new_train_name = os.path.join(self.project_location, self.fg_output_dir, 'train_new.csv')
        new_test_name = os.path.join(self.project_location, self.fg_output_dir, 'test_new.csv')
        _logger.info('Saving %s' % new_train_name)
        _logger.info('Saving %s' % new_test_name)

        train_data.to_csv(new_train_name, index=False)
        test_data.to_csv(new_test_name, index=False)

    def output(self):
        return {'train_data': luigi.LocalTarget(os.path.join(self.project_location, self.fg_output_dir, 'train_new.csv')),
                'test_data': luigi.LocalTarget(os.path.join(self.project_location, self.fg_output_dir, 'test_new.csv'))}


@requires(TrainDataIngestion)
class FeatureSelection(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    fs_output_dir = luigi.Parameter()  # type: str # feat. selection dir (where results of feature select. are saved)
    feats_select_method = luigi.Parameter()  # type: str # # feat. selection method name
    fs_results_file = 'optimal_features.txt'

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load train data set from feature generation pool
        train_data = pd.read_csv(self.input()['train_data'].path)

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
        eval_metric = config.get_string('modeling_settings.lightgbm.eval_metric')  # feature selector is lightgbm-based
        metrics_scorer = get_metrics_scorer(config.get('modeling_settings.cv_params.metrics_scorer'))
        metrics_decimals = config.get_int('modeling_settings.cv_params.metrics_decimals')
        num_folds = config.get_int('modeling_settings.cv_params.num_folds')
        stratified = config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = config.get_bool('modeling_settings.cv_params.kfolds_shuffle')

        lgbm_params_feats_exploration = dict(config.get_config('features_selection.%s.lgbm_params.'
                                                               'feats_exploration' % self.feats_select_method))
        lgbm_params_feats_selection = dict(config.get_config('features_selection.%s.lgbm_params.'
                                                             'feats_selection' % self.feats_select_method))

        # Categorical features for lgbm in feature_selection process
        feat_cols = [f for f in train_data.columns if f not in [target_column, index_column]]
        categorical_feats = [f for f in feat_cols if train_data[f].dtype == 'object']
        _logger.info('Number of categorical features: {0}'.format(len(categorical_feats)))
        for cat_feat in categorical_feats:
            train_data[cat_feat], _ = pd.factorize(train_data[cat_feat])
            train_data[cat_feat] = train_data[cat_feat].astype('category')

        # TODO: how to improve this situation: if cat_features=None, an automatic algo will be used to find cat_features
        # If cat_features = categorical_feats -> categorical feature are only those with dtype = 'object'
        cat_features = categorical_feats  # None

        # Initialize feature selection procedure
        features_selection = feature_selector(
            train_df=train_data, target_column=target_column, index_column=index_column,
            cat_features=cat_features, int_threshold=int_threshold,
            lgbm_params_feats_exploration=lgbm_params_feats_exploration,
            lgbm_params_feats_selection=lgbm_params_feats_selection,
            eval_metric=eval_metric, metrics_scorer=metrics_scorer,
            metrics_decimals=metrics_decimals, num_folds=num_folds,
            stratified=stratified, kfolds_shuffle=kfolds_shuffle,
            seed_val=fs_seed_val, project_location=self.project_location,
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

        full_path_to_file = os.path.join(self.project_location, self.fs_output_dir, self.fs_results_file)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(opt_feats, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.fs_output_dir, self.fs_results_file))


class InitializeSingleModelPredictor(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    model = luigi.Parameter()  # type: str # name of estimator model
    run_feature_selection = luigi.BoolParameter()  # type: bool # if True -> run feature selection
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    fs_output_dir = luigi.Parameter()  # type: str # feat. selection dir (where results of feature select. are saved)
    solution_output_dir = luigi.Parameter()  # type: str # output dir where results of single model prediction are saved
    output_pickle_file = 'predictor_initialized.pickle'

    def requires(self):
        requirements = {'data': self.clone(TrainDataIngestion)}
        if self.run_feature_selection:
            # Parsing config (actually it is cached)
            config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)
            feats_select_method = config.get_string('modeling_settings.%s.fs_method' % self.model)
            requirements['features'] = FeatureSelection(project_location=self.project_location,
                                                        config_directory=self.config_directory,
                                                        config_file=self.config_file,
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
        eval_metric = config.get_string('modeling_settings.%s.eval_metric' % self.model)
        metrics_scorer = get_metrics_scorer(config.get('modeling_settings.cv_params.metrics_scorer'))
        metrics_decimals = config.get_int('modeling_settings.cv_params.metrics_decimals')
        target_decimals = config.get_int('modeling_settings.cv_params.target_decimals')
        num_folds = config.get_int('modeling_settings.cv_params.num_folds')
        stratified = config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = config.get_bool('modeling_settings.cv_params.kfolds_shuffle')
        cv_verbosity = config.get_int('modeling_settings.cv_params.cv_verbosity')
        data_split_seed = config.get_int('modeling_settings.data_split_seed')
        model_seeds_list = config.get_list('modeling_settings.model_seeds_list')

        # Initialize single model predictor
        predictor = Predictor(
            train_df=train_data[opt_feats] if self.run_feature_selection else train_data,
            test_df=test_data[opt_feats] if self.run_feature_selection else test_data,
            target_column=target_column, index_column=index_column, cols_to_exclude=cols_to_exclude,
            model=estimator_wrapped, predict_probability=predict_probability, class_label=class_label,
            bagging=bagging, predict_test=predict_test,
            eval_metric=eval_metric, metrics_scorer=metrics_scorer,
            metrics_decimals=metrics_decimals, target_decimals=target_decimals,
            num_folds=num_folds, stratified=stratified, kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity,
            data_split_seed=data_split_seed, model_seeds_list=model_seeds_list,
            project_location=self.project_location, output_dirname=self.solution_output_dir
        )

        full_path_to_file = os.path.join(self.project_location, self.solution_output_dir, self.output_pickle_file)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(predictor, f)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.solution_output_dir, self.output_pickle_file))


@requires(InitializeSingleModelPredictor)
class RunSingleModelHPO(luigi.Task):
    model = luigi.Parameter()  # type: str # name of estimator model
    hpo_output_dir = luigi.Parameter()  # type: str # hyper-parameters opt. dir (where results of hpo are saved)
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
        init_points = config.get_int('hp_optimization.%s.hpo_space.single_model_solution.init_points' % hpo_method)
        n_iter = config.get_int('hp_optimization.%s.hpo_space.single_model_solution.n_iter' % hpo_method)
        hp_optimization_space = dict(config.get_config(
            'hp_optimization.%s.hpo_space.single_model_solution.%s' % (hpo_method, self.model)))

        # Initialize hyper-parameter optimizator
        hpo = hp_optimizator(predictor=predictor,
                             hp_optimization_space=hp_optimization_space,
                             init_points=init_points,
                             n_iter=n_iter,
                             seed_val=hpo_seed_val,
                             project_location=self.project_location,
                             output_dirname=self.hpo_output_dir)

        # Run optimization and save output results
        hpo.run()
        hpo.save_hpo_history()

        full_path_to_file = os.path.join(self.project_location, self.hpo_output_dir, self.hpo_results_file)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(hpo.best_params, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.hpo_output_dir, self.hpo_results_file))


class RunSingleModelPrediction(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    model = luigi.Parameter()  # type: str # name of estimator model
    run_feature_selection = luigi.BoolParameter()  # type: bool # if True -> run feature selection
    run_hpo = luigi.BoolParameter()  # type: bool # if True -> run hyper-parameters optimization
    run_bagging = luigi.BoolParameter()  # type: bool # if True -> run bagging
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    fs_output_dir = luigi.Parameter()  # type: str # feat. selection dir (where results of feature select. are saved)
    hpo_output_dir = luigi.Parameter()  # type: str # hyper-parameters opt. dir (where results of hpo are saved)
    solution_output_dir = luigi.Parameter()  # type: str # output dir where results of single model prediction are saved
    output_filename = 'single_model_oof_data_info.txt'

    def requires(self):
        requirements = {'predictor': self.clone(InitializeSingleModelPredictor)}
        if self.run_hpo:
            requirements['hpo'] = self.clone(RunSingleModelHPO)
        return requirements

    def run(self):
        # Load initialized single model predictor
        predictor = pickle.load(open(self.input()['predictor'].path, "rb"))

        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

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

        # Plot confusion matrix
        plot_cm = config.get_bool('modeling_settings.plot_confusion_matrix')
        if plot_cm:
            class_names = config.get('modeling_settings.confusion_matrix_labels', default=None)
            class_names = np.array(class_names) if class_names is not None else None
            labels_mapper = config.get('modeling_settings.labels_mapper', default=None)
            labels_mapper = eval(labels_mapper) if labels_mapper is not None else None
            predictor.plot_confusion_matrix(class_names, labels_mapper, normalize=True, save=True)

        # Plot features importance
        predictor.plot_features_importance(n_features=7, save=True)

        # Save results and a copy of the config file
        predictor.save_oof_results()
        predictor.save_submission_results()
        predictor.save_config(self.project_location, self.config_directory, self.config_file)

        # Prepare output file for ensembling
        solution_id = generate_single_model_solution_id_key(predictor.model_name)
        files = [predictor.FILENAME_TRAIN_OOF_RESULTS, predictor.FILENAME_TEST_RESULTS]
        if predictor.bagged_oof_preds is not None:
            files.extend([predictor.FILENAME_TRAIN_OOF_RESULTS_BAGGED, predictor.FILENAME_TEST_RESULTS_BAGGED])
        output = {
            solution_id:
                {
                    'path': os.path.join(self.project_location, self.solution_output_dir),
                    'files': files
                }
        }
        full_path_to_file = os.path.join(self.project_location, self.solution_output_dir, self.output_filename)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(output, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.solution_output_dir, self.output_filename))


class MakeSingleModelsPredictions(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    output_filename = 'single_models_oof_data_info.txt'

    def requires(self):
        config_handler = ConfigFileHandler(self.project_location, self.config_directory, self.config_file)
        collection_input_parameters = config_handler.pipeline_parameters_for_single_models_solutions()
        for input_parameters in collection_input_parameters:
            yield RunSingleModelPrediction(**input_parameters)

    def run(self):
        oof_input_files = {}
        for input_target in self.input():
            with open(input_target.path, 'r') as f:
                oof_input_files = merge_two_dicts(oof_input_files, json.load(f))

        # TODO: to refactor this part
        create_output_dir(os.path.join(self.project_location, 'results_ensembling'))
        full_path_to_file = os.path.join(self.project_location, 'results_ensembling', self.output_filename)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(oof_input_files, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, 'results_ensembling', self.output_filename))


class InitializeStacker(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    stacker_model = luigi.Parameter()  # type: str # name of stacker estimator model
    stacking_output_dir = luigi.Parameter()  # type: str # output dir where results of stacking are saved
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    output_pickle_file = 'stacker_initialized.pickle'

    def requires(self):
        requirements = {'data': self.clone(TrainDataIngestion)}
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)
        use_provided_oof_input_files = config.get_bool('stacker.use_provided_oof_input_files')
        if not use_provided_oof_input_files:
            requirements['single_models_oof_data_info'] = self.clone(MakeSingleModelsPredictions)
        return requirements

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load train and test data sets
        train_data = pd.read_csv(self.input()['data']['train_data'].path)
        test_data = pd.read_csv(self.input()['data']['test_data'].path)

        # Load oof data information (either from single models predictions or from the dictionary in the config file)
        use_provided_oof_input_files = config.get_bool('stacker.use_provided_oof_input_files')
        if use_provided_oof_input_files:
            oof_input_files = dict(config.get('stacker.oof_input_files'))
        else:
            with open(self.input()['single_models_oof_data_info'].path, 'r') as f:
                oof_input_files = json.load(f)

        # Input settings for Stacker
        target_column = config.get_string('raw_data_settings.target_column')
        index_column = config.get_string('raw_data_settings.index_column')

        # Extracting settings from config
        stacker_init_params = dict(config.get_config('stacking_model_init_params.%s' % self.stacker_model))
        stacker_wrapped = get_wrapped_estimator(self.stacker_model, stacker_init_params)
        stacker_predict_probability = config.get_bool('stacker.%s.predict_probability' % self.stacker_model)
        class_label = config.get('stacker.%s.class_label' % self.stacker_model)
        stacker_eval_metric = config.get_string('stacker.%s.eval_metric' % self.stacker_model)
        stacker_metrics_scorer = get_metrics_scorer(config.get('stacker.%s.metrics_scorer' % self.stacker_model))
        stacker_metrics_decimals = config.get_int('stacker.%s.metrics_decimals' % self.stacker_model)
        stacker_target_decimals = config.get_int('stacker.%s.target_decimals' % self.stacker_model)
        cols_to_exclude = config.get_list('modeling_settings.cols_to_exclude')
        num_folds = config.get_int('modeling_settings.cv_params.num_folds')
        stratified = config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = config.get_bool('modeling_settings.cv_params.kfolds_shuffle')
        cv_verbosity = config.get_int('modeling_settings.cv_params.cv_verbosity')
        stacker_bagging = config.get_bool('stacker.%s.run_bagging' % self.stacker_model)
        data_split_seed = config.get_int('modeling_settings.data_split_seed')
        model_seeds_list = config.get_list('modeling_settings.model_seeds_list')
        stack_bagged_results = config.get_bool('modeling_settings.stack_bagged_results')

        # TODO: to add this logic
        # If True -> use raw features additionally to out-of-fold results
        stacker_use_raw_features = config.get_bool('stacker.%s.use_raw_features' % self.stacker_model)

        # Initializing stacker
        stacker = Stacker(
            oof_input_files=oof_input_files, train_df=train_data, test_df=test_data,
            target_column=target_column, index_column=index_column, cols_to_exclude=cols_to_exclude,
            stacker_model=stacker_wrapped, stack_bagged_results=stack_bagged_results, bagging=stacker_bagging,
            predict_probability=stacker_predict_probability, class_label=class_label,
            eval_metric=stacker_eval_metric, metrics_scorer=stacker_metrics_scorer,
            metrics_decimals=stacker_metrics_decimals, target_decimals=stacker_target_decimals,
            num_folds=num_folds, stratified=stratified, kfolds_shuffle=kfolds_shuffle, cv_verbosity=cv_verbosity,
            data_split_seed=data_split_seed, model_seeds_list=model_seeds_list,
            project_location=self.project_location, output_dirname=self.stacking_output_dir
        )

        full_path_to_file = os.path.join(self.project_location, self.stacking_output_dir, self.output_pickle_file)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(stacker, f)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.stacking_output_dir, self.output_pickle_file))


@requires(InitializeStacker)
class RunStackerHPO(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    stacker_model = luigi.Parameter()  # type: str # name of stacker estimator model
    stacking_output_dir = luigi.Parameter()  # type: str # output dir where results of stacking are saved
    stacker_hpo_results_file = 'stacker_optimized_hp.txt'

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load initialized stacker
        stacker = pickle.load(open(self.input().path, "rb"))

        # Extracting settings from config
        stacker_hpo_method = config.get_string('stacker.%s.hpo_method' % self.stacker_model)
        stacker_hp_optimizator = load_hp_optimization_class(stacker_hpo_method)
        stacker_hpo_space = dict(config.get_config(
            'hp_optimization.%s.hpo_space.stacker_model.%s' % (stacker_hpo_method, self.stacker_model)))

        stacker_hpo_init_points = config.get_int('hp_optimization.%s.hpo_space.'
                                                 'stacker_model.init_points' % stacker_hpo_method)
        stacker_hpo_n_iter = config.get_int('hp_optimization.%s.hpo_space.stacker_model.n_iter' % stacker_hpo_method)
        stacker_hpo_seed_val = config.get_int('modeling_settings.stacker_hpo_seed')

        # Initialize hyper-parameter optimizator
        stacker_hpo = stacker_hp_optimizator(predictor=stacker,
                                             hp_optimization_space=stacker_hpo_space,
                                             init_points=stacker_hpo_init_points,
                                             n_iter=stacker_hpo_n_iter,
                                             seed_val=stacker_hpo_seed_val,
                                             project_location=self.project_location,
                                             output_dirname=self.stacking_output_dir)

        # Run optimization and save output results
        stacker_hpo.run()
        stacker_hpo.save_hpo_history()

        full_path_to_file = os.path.join(self.project_location, self.stacking_output_dir, self.stacker_hpo_results_file)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(stacker_hpo.best_params, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location,
                                              self.stacking_output_dir,
                                              self.stacker_hpo_results_file))


class RunSingleStacker(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    stacker_model = luigi.Parameter()  # type: str # name of stacker estimator model
    run_stacker_hpo = luigi.BoolParameter()  # type: bool # if True -> run stacking
    run_bagging = luigi.BoolParameter()  # type: bool # if True -> run bagging
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    stacking_output_dir = luigi.Parameter()  # type: str # output dir where results of stacking are saved
    output_filename = 'stacker_oof_data_info.txt'

    def requires(self):
        requirements = {'stacker': self.clone(InitializeStacker)}
        if self.run_stacker_hpo:
            requirements['stacker_hpo'] = self.clone(RunStackerHPO)
        return requirements

    def run(self):
        # Load initialized stacker
        stacker = pickle.load(open(self.input()['stacker'].path, "rb"))

        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        if self.run_stacker_hpo:
            # Load set of stacker's best parameters from hyper-parameters optimization procedure
            with open(self.input()['stacker_hpo'].path, 'r') as f:
                best_params = json.load(f)

            # Re-initialize stacker with optimal parameters
            stacker.model.reinit_model_with_new_params(best_params)

        # Run CV and prediction of test data set
        stacker.run_cv_and_prediction()

        # Save results and a copy of the config file
        stacker.save_oof_results()
        stacker.save_submission_results()
        stacker.save_config(self.project_location, self.config_directory, self.config_file)

        # Plot confusion matrix
        plot_cm = config.get_bool('modeling_settings.plot_confusion_matrix')
        if plot_cm:
            class_names = config.get('modeling_settings.confusion_matrix_labels', default=None)
            class_names = np.array(class_names) if class_names is not None else None
            labels_mapper = config.get('modeling_settings.labels_mapper', default=None)
            labels_mapper = eval(labels_mapper) if labels_mapper is not None else None
            stacker.plot_confusion_matrix(class_names, labels_mapper, normalize=True, save=True)

        # Prepare output file for ensembling
        solution_id = generate_single_model_solution_id_key(stacker.model_name)
        files = [stacker.FILENAME_TRAIN_OOF_RESULTS, stacker.FILENAME_TEST_RESULTS]
        if stacker.bagged_oof_preds is not None:
            files.extend([stacker.FILENAME_TRAIN_OOF_RESULTS_BAGGED, stacker.FILENAME_TEST_RESULTS_BAGGED])
        output = {
            solution_id:
                {
                    'path': os.path.join(self.project_location, self.stacking_output_dir),
                    'files': files
                }
        }
        full_path_to_file = os.path.join(self.project_location, self.stacking_output_dir, self.output_filename)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(output, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.stacking_output_dir, self.output_filename))


class MakeStackingPredictions(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    output_filename = 'stacking_models_oof_data_info.txt'

    def requires(self):
        config_handler = ConfigFileHandler(self.project_location, self.config_directory, self.config_file)
        collection_stacking_input_pars = config_handler.pipeline_parameters_for_stacked_solutions()
        for input_parameters in collection_stacking_input_pars:
            yield RunSingleStacker(**input_parameters)

    def run(self):
        oof_input_files = {}
        for input_target in self.input():
            with open(input_target.path, 'r') as f:
                oof_input_files = merge_two_dicts(oof_input_files, json.load(f))

        # TODO: to refactor this part
        create_output_dir(os.path.join(self.project_location, 'results_ensembling'))
        full_path_to_file = os.path.join(self.project_location, 'results_ensembling', self.output_filename)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(oof_input_files, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, 'results_ensembling', self.output_filename))


class InitializeBlender(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    blending_method = luigi.Parameter()  # type: str
    blending_output_dir = luigi.Parameter()  # type: str # output dir where results of blending are saved
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    output_pickle_file = 'blender_initialized.pickle'

    def requires(self):
        requirements = {'data': self.clone(TrainDataIngestion)}
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)
        use_provided_oof_input_files = config.get_bool('blender.use_provided_oof_input_files')
        if not use_provided_oof_input_files:
            blend_stacked_results = config.get_bool('modeling_settings.blend_stacked_results')
            if blend_stacked_results:
                requirements['oof_data_info'] = self.clone(MakeStackingPredictions)
            else:
                requirements['oof_data_info'] = self.clone(MakeSingleModelsPredictions)
        return requirements

    def run(self):
        # Parsing config (actually it is cached)
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Load train and test data sets
        train_data = pd.read_csv(self.input()['data']['train_data'].path)
        test_data = pd.read_csv(self.input()['data']['test_data'].path)

        # Load oof data information (either from single models predictions or from the dictionary in the config file)
        use_provided_oof_input_files = config.get_bool('blender.use_provided_oof_input_files')
        if use_provided_oof_input_files:
            oof_input_files = dict(config.get('blender.oof_input_files'))
        else:
            with open(self.input()['oof_data_info'].path, 'r') as f:
                oof_input_files = json.load(f)

        # Input settings for blender
        target_column = config.get_string('raw_data_settings.target_column')
        index_column = config.get_string('raw_data_settings.index_column')

        # Extracting settings from config
        blend_bagged_results = config.get_bool('modeling_settings.blend_bagged_results')
        blender_init_points = config.get_int('blender.%s.init_points' % self.blending_method)
        blender_n_iter = config.get_int('blender.%s.n_iter' % self.blending_method)
        blender_seed_val = config.get_int('modeling_settings.blender_seed')
        class_label = config.get('blender.%s.class_label' % self.blending_method)
        predict_probability = config.get_bool('blender.%s.predict_probability' % self.blending_method)
        stratified = config.get_bool('blender.%s.stratified' % self.blending_method)
        num_folds = config.get_int('blender.%s.num_folds' % self.blending_method)
        kfolds_shuffle = config.get_bool('blender.%s.kfolds_shuffle' % self.blending_method)
        metrics_scorer = get_metrics_scorer(config.get('blender.%s.metrics_scorer' % self.blending_method))
        metrics_decimals = config.get_int('blender.%s.metrics_decimals' % self.blending_method)
        target_decimals = config.get_int('blender.%s.target_decimals' % self.blending_method)
        data_split_seed = config.get_int('blender.%s.data_split_seed' % self.blending_method)

        # Initializing blender
        bayes_blender = BayesOptimizationBlender(
            oof_input_files=oof_input_files, train_df=train_data, test_df=test_data,
            target_column=target_column, index_column=index_column, blend_bagged_results=blend_bagged_results,
            predict_probability=predict_probability, class_label=class_label, init_points=blender_init_points,
            n_iter=blender_n_iter, blender_seed_val=blender_seed_val, metrics_scorer=metrics_scorer,
            metrics_decimals=metrics_decimals, target_decimals=target_decimals, num_folds=num_folds,
            stratified=stratified, kfolds_shuffle=kfolds_shuffle, data_split_seed=data_split_seed,
            project_location=self.project_location, output_dirname=self.blending_output_dir
        )

        full_path_to_file = os.path.join(self.project_location, self.blending_output_dir, self.output_pickle_file)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(bayes_blender, f)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.blending_output_dir, self.output_pickle_file))


class RunSingleBlender(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    blending_method = luigi.Parameter()  # type: str
    fg_output_dir = luigi.Parameter()  # type: str # feat. generation dir (to be used as input for train data ingestion)
    blending_output_dir = luigi.Parameter()  # type: str # output dir where results of stacking are saved
    output_filename = 'blender_oof_data_info.txt'

    def requires(self):
        return {'blender': self.clone(InitializeBlender)}

    def run(self):
        # Load initialized blender
        blender = pickle.load(open(self.input()['blender'].path, "rb"))

        # Parsing config (actually it is cached)
        # config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Run blender and save weights
        blender.run()
        blender.save_weights()

        # Save results and a copy of the config file
        blender.save_oof_results()
        blender.save_submission_results()
        blender.save_config(self.project_location, self.config_directory, self.config_file)

        # Plot confusion matrix
        # plot_cm = config.get_bool('modeling_settings.plot_confusion_matrix')
        # if plot_cm:
        #     class_names = config.get('modeling_settings.confusion_matrix_labels', default=None)
        #     class_names = np.array(class_names) if class_names is not None else None
        #     labels_mapper = config.get('modeling_settings.labels_mapper', default=None)
        #     labels_mapper = eval(labels_mapper) if labels_mapper is not None else None
        #     blender.plot_confusion_matrix(class_names, labels_mapper, normalize=True, save=True)

        # Prepare output file for ensembling
        solution_id = generate_single_model_solution_id_key(blender.BLENDING_METHOD)
        files = [blender.FILENAME_TRAIN_OOF_RESULTS, blender.FILENAME_TEST_RESULTS]
        output = {
            solution_id:
                {
                    'path': os.path.join(self.project_location, self.blending_output_dir),
                    'files': files
                }
        }
        full_path_to_file = os.path.join(self.project_location, self.blending_output_dir, self.output_filename)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(output, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, self.blending_output_dir, self.output_filename))


class MakeBlendingPredictions(luigi.Task):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory
    output_filename = 'blending_models_oof_data_info.txt'

    def requires(self):
        config_handler = ConfigFileHandler(self.project_location, self.config_directory, self.config_file)
        collection_stacking_input_pars = config_handler.pipeline_parameters_for_blending_solutions()
        for input_parameters in collection_stacking_input_pars:
            yield RunSingleBlender(**input_parameters)

    def run(self):
        oof_input_files = {}
        for input_target in self.input():
            with open(input_target.path, 'r') as f:
                oof_input_files = merge_two_dicts(oof_input_files, json.load(f))

        # TODO: to refactor this part
        create_output_dir(os.path.join(self.project_location, 'results_ensembling'))
        full_path_to_file = os.path.join(self.project_location, 'results_ensembling', self.output_filename)
        _logger.info('Saving %s' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(oof_input_files, indent=4))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.project_location, 'results_ensembling', self.output_filename))


class BuildSolution(luigi.WrapperTask):
    project_location = luigi.Parameter()  # type: str # absolute path to project's main directory
    config_directory = luigi.Parameter()  # type: str # name of config sub-directory in project directory
    config_file = luigi.Parameter()  # type: str # name of config file in the config sub-directory

    def requires(self):
        config = ConfigFileHandler.parse_config_file(self.project_location, self.config_directory, self.config_file)

        # Run single model predictions
        yield MakeSingleModelsPredictions(self.project_location, self.config_directory, self.config_file)

        # Run stacking if requested in config
        run_stacking = config.get_bool('modeling_settings.run_stacking')
        if run_stacking:
            yield MakeStackingPredictions(self.project_location, self.config_directory, self.config_file)

        # Run blending if requested in config
        run_blending = config.get_bool('modeling_settings.run_blending')
        if run_blending:
            yield MakeBlendingPredictions(self.project_location, self.config_directory, self.config_file)


def run_full_pipeline(project_abs_path, config_dir, config_file_name, local_scheduler_=True):
    # Run pipeline this way
    luigi.build([BuildSolution(project_abs_path, config_dir, config_file_name)], local_scheduler=local_scheduler_)

    # Or this way
    # luigi.run(main_task_cls=BuildSolution,
    #           cmdline_args=["--BuildSolution-project-location={0}".format(project_abs_path),
    #                         "--BuildSolution-config-directory={0}".format(config_dir),
    #                         "--BuildSolution-config-file={0}".format(config_file_name),
    #                         "--workers=1"],
    #           local_scheduler=local_scheduler_)


def run_stacking_only(project_abs_path, config_dir, config_file_name, local_scheduler_=True):
    # Run pipeline this way
    luigi.build([MakeStackingPredictions(project_abs_path, config_dir, config_file_name)],
                local_scheduler=local_scheduler_)

    # Or this way
    # luigi.run(main_task_cls=MakeStackingPredictions,
    #           cmdline_args=["--MakeStackingPredictions-project-location={0}".format(project_abs_path),
    #                         "--MakeStackingPredictions-config-directory={0}".format(config_dir),
    #                         "--MakeStackingPredictions-config-file={0}".format(config_file_name),
    #                         "--workers=1"],
    #           local_scheduler=local_scheduler_)


if __name__ == '__main__':
    # Location of the project and config file
    project_location = r'c:\Kaggle\FastMLFramework\examples\classification\multiclass\iris'
    # project_location = r'c:\Kaggle\FastMLFramework\examples\classification\binary\credit_scoring'
    # project_location = r'c:\Kaggle\FastMLFramework\examples\classification\multilabel'
    # project_location = r'c:\Kaggle\home_credit_default_risk'

    config_directory = 'configs'
    config_file = 'solution.conf'
    local_scheduler = True  # set True if run on google collab, etc. [no localhost]
                            # set False if run locally -> then run in terminal .luigid -> go to http://localhost:8082
                            # and see the dependencies and pipeline execution flow

    # Run all steps of a pipeline
    run_full_pipeline(project_location, config_directory, config_file, local_scheduler)

    # Run stacking part of the pipeline
    # run_stacking_only(project_location, config_directory, config_file, local_scheduler)
