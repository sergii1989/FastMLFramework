import os
import json
import luigi
import pickle
import warnings
import numpy as np
import pandas as pd

from luigi.util import requires
from pyhocon import ConfigFactory
from pyhocon.config_tree import ConfigTree
from modeling.prediction import Predictor
from ensembling.stacking.stacker import Stacker
from generic_tools.utils import get_metrics_scorer
from generic_tools.config_parser import ResultsLocationManager
from modeling.model_wrappers import get_wrapped_estimator
from modeling.feature_selection import load_feature_selector_class
from modeling.hyper_parameters_optimization import load_hp_optimization_class
from data_processing.preprocessing import downcast_datatypes

warnings.filterwarnings("ignore")


class TrainDataIngestion(luigi.Task):
    project_location = luigi.Parameter()
    ftg_output_dir = luigi.Parameter()
    config = luigi.Parameter()  # type: ConfigTree

    def run(self):
        # Settings for debug
        debug = self.config.get_bool('modeling_settings.debug')
        num_rows = self.config.get_int('modeling_settings.num_rows')

        # Load train and test data set from feature generation pool and downcast data types
        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.ftg_output_dir, 'train.csv'))
        train_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
            .reset_index(drop=True)

        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.ftg_output_dir, 'test.csv'))
        test_data = downcast_datatypes(pd.read_csv(full_path_to_file, nrows=num_rows if debug else None))\
            .reset_index(drop=True)

        print('Train DF shape: {0}\n'.format(train_data.shape, train_data.info()))
        print('Test DF shape: {0}'.format(test_data.shape))

        new_train_name = os.path.normpath(os.path.join(self.project_location, self.ftg_output_dir, 'train_new.csv'))
        new_test_name = os.path.normpath(os.path.join(self.project_location, self.ftg_output_dir, 'test_new.csv'))
        print('\nSaving %s\n' % new_train_name)
        print('\nSaving %s\n' % new_test_name)

        train_data.to_csv(new_train_name, index=False)
        test_data.to_csv(new_test_name, index=False)

    def output(self):
        return {'train_data': luigi.LocalTarget(os.path.normpath(os.path.join(self.project_location,
                                                                              self.ftg_output_dir,
                                                                              'train_new.csv'))),
                'test_data': luigi.LocalTarget(os.path.normpath(os.path.join(self.project_location,
                                                                             self.ftg_output_dir,
                                                                             'test_new.csv')))
                }


@requires(TrainDataIngestion)
class FeatureSelection(luigi.Task):

    fts_output_dir = luigi.Parameter()

    def run(self):
        # Load train data set from feature generation pool
        train_data = pd.read_csv(self.input()['train_data'].path)

        # Categorical features for lgbm in feature_selection process
        categorical_feats = [f for f in train_data.columns if train_data[f].dtype == 'object']
        print('Number of categorical features: {0}'.format(len(categorical_feats)))
        for f_ in categorical_feats:
            train_data[f_], _ = pd.factorize(train_data[f_])
            train_data[f_] = train_data[f_].astype('category')
        cat_features = categorical_feats  # None

        feats_select_method = self.config.get_string('features_selection.method')
        feature_selector = load_feature_selector_class(feats_select_method)
        target_column = self.config.get_string('raw_data_settings.target_column')
        index_column = self.config.get_string('raw_data_settings.index_column')
        int_threshold = self.config.get_int('features_selection.target_permutation.int_threshold')
        num_boost_rounds = self.config.get_int('features_selection.target_permutation.num_boost_rounds')
        nb_runs = self.config.get_int('features_selection.target_permutation.nb_target_permutation_runs')
        fs_seed_val = self.config.get_int('features_selection.target_permutation.seed_value')
        thresholds = self.config.get_list('features_selection.target_permutation.'
                                          'eval_feats_removal_impact_on_cv_score.thresholds')
        n_thresholds = self.config.get_int('features_selection.target_permutation.'
                                           'eval_feats_removal_impact_on_cv_score.n_thresholds')
        eval_metric = self.config.get_string('modeling_settings.cv_params.eval_metric')
        metrics_scorer = get_metrics_scorer(self.config.get('modeling_settings.cv_params.metrics_scorer'))
        metrics_decimals = self.config.get_int('modeling_settings.cv_params.metrics_decimals')
        num_folds = self.config.get_int('modeling_settings.cv_params.num_folds')
        stratified = self.config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = self.config.get_bool('modeling_settings.cv_params.kfolds_shuffle')

        lgbm_params_feats_exploration = dict(self.config.get_config('features_selection.target_permutation.'
                                                                    'lgbm_params.feats_exploration'))
        lgbm_params_feats_selection = dict(self.config.get_config('features_selection.target_permutation.'
                                                                  'lgbm_params.feats_selection'))
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
            output_dirname=self.fts_output_dir
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

        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.fts_output_dir,
                                                          'optimal_features.txt'))

        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(opt_feats))

    def output(self):
        return {
            'opt_feats': luigi.LocalTarget(os.path.normpath(os.path.join(
                self.project_location, self.fts_output_dir, 'optimal_features.txt')))
        }


class InitializeSingleModelPredictor(luigi.Task):

    run_feature_selection = luigi.Parameter()
    project_location = luigi.Parameter()
    ftg_output_dir = luigi.Parameter()
    fts_output_dir = luigi.Parameter()
    solution_output_dir = luigi.Parameter()
    config = luigi.Parameter()  # type: ConfigTree

    def requires(self):
        requirements = {'data': self.clone(TrainDataIngestion)}
        if self.run_feature_selection:
            requirements['features'] = self.clone(FeatureSelection)
        return requirements

    def run(self):
        # Load train and test data sets
        train_data = pd.read_csv(self.input()['data']['train_data'].path)
        test_data = pd.read_csv(self.input()['data']['test_data'].path)

        opt_feats = []
        if self.run_feature_selection:
            # Load set of features from feature_selection procedure
            with open(self.input()['features']['opt_feats'].path, 'r') as f:
                opt_feats = json.load(f)

        target_column = self.config.get_string('raw_data_settings.target_column')
        index_column = self.config.get_string('raw_data_settings.index_column')
        model = self.config.get_string('modeling_settings.model')
        model_init_params = dict(self.config.get_config('model_init_params.%s' % model))
        estimator_wrapped = get_wrapped_estimator(model, model_init_params)
        predict_probability = self.config.get_bool('modeling_settings.predict_probability')
        class_label = self.config.get('modeling_settings.class_label')
        cols_to_exclude = self.config.get_list('modeling_settings.cols_to_exclude')
        bagging = self.config.get_bool('modeling_settings.bagging')
        predict_test = self.config.get_bool('modeling_settings.predict_test')
        eval_metric = self.config.get_string('modeling_settings.cv_params.eval_metric')
        metrics_scorer = get_metrics_scorer(self.config.get('modeling_settings.cv_params.metrics_scorer'))
        metrics_decimals = self.config.get_int('modeling_settings.cv_params.metrics_decimals')
        target_decimals = self.config.get_int('modeling_settings.cv_params.target_decimals')
        num_folds = self.config.get_int('modeling_settings.cv_params.num_folds')
        stratified = self.config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = self.config.get_bool('modeling_settings.cv_params.kfolds_shuffle')
        cv_verbosity = self.config.get_int('modeling_settings.cv_params.cv_verbosity')
        data_split_seed = self.config.get_int('modeling_settings.cv_params.data_split_seed')
        model_seeds_list = self.config.get_list('modeling_settings.model_seeds_list')

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

        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.solution_output_dir,
                                                          'predictor.pickle'))
        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(predictor, f)

    def output(self):
        return luigi.LocalTarget(os.path.normpath(os.path.join(
            self.project_location, self.solution_output_dir, 'predictor.pickle')))


@requires(InitializeSingleModelPredictor)
class RunSingleModelHPO(luigi.Task):

    hpos_output_dir = luigi.Parameter()

    def run(self):
        # Load initialized single model predictor
        predictor = pickle.load(open(self.input().path, "rb"))

        hpo_method = self.config.get_string('hp_optimization.method')
        hp_optimizator = load_hp_optimization_class(hpo_method)
        hpo_seed_val = self.config.get_int('hp_optimization.seed_value')
        init_points = self.config.get_int('hp_optimization.%s.init_points' % hpo_method)
        n_iter = self.config.get_int('hp_optimization.%s.n_iter' % hpo_method)
        model = self.config.get_string('modeling_settings.model')
        hp_optimization_space = dict(self.config.get_config('hp_optimization.%s.hp_optimization_space.%s'
                                                            % (hpo_method, model)))
        # Initialize hyper-parameter optimizator
        hpo = hp_optimizator(
            predictor, hp_optimization_space=hp_optimization_space,
            init_points=init_points, n_iter=n_iter, seed_val=hpo_seed_val,
            project_location=project_location, output_dirname=self.hpos_output_dir
        )

        # Run optimization and save output results
        hpo.run()
        hpo.save_optimized_hp()
        hpo.save_all_hp_results()

        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.hpos_output_dir,
                                                          'optim_hp.txt'))

        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'w') as f:
            f.write(json.dumps(hpo.best_params))

    def output(self):
        return luigi.LocalTarget(os.path.normpath(os.path.join(
            self.project_location, self.hpos_output_dir, 'optim_hp.txt')))


class RunSingleModelPrediction(luigi.Task):

    project_location = luigi.Parameter()
    run_feature_selection = luigi.Parameter()
    run_hpo = luigi.Parameter()
    ftg_output_dir = luigi.Parameter()
    fts_output_dir = luigi.Parameter()
    hpos_output_dir = luigi.Parameter()
    solution_output_dir = luigi.Parameter()
    config = luigi.Parameter()

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

        full_path_to_file = os.path.normpath(os.path.join(self.project_location, self.solution_output_dir,
                                                          'predictor_with_results.pickle'))
        print('\nSaving %s\n' % full_path_to_file)
        with open(full_path_to_file, 'wb') as f:
            pickle.dump(predictor, f)

    def output(self):
        return luigi.LocalTarget(os.path.normpath(os.path.join(
            self.project_location, self.solution_output_dir, 'predictor_with_results.pickle')))


class InitializeStacker(luigi.Task):

    # stacking_output_dir = config.get_string('stacker.full_path_output')
    # full_path_to_file   = os.path.normpath(os.path.join(project_location, stacking_output_dir, 'stacker.pickle'))

    def requires(self):
        return TrainDataIngestion(self.project_location, self.ftg_output_dir)

    def run(self):
        # Load train and test data sets
        train_data = pd.read_csv(self.input()['data']['train_data'].path)
        test_data = pd.read_csv(self.input()['data']['test_data'].path)

        # Read dict with OOF input file names, locations and short labels to be used when naming output files
        oof_input_files = dict(self.config.get_config('stacker.oof_input_files'))
        for key in oof_input_files:
            oof_input_files[key] = dict(oof_input_files[key])

        # Input settings for Stacker
        target_column = self.config.get_string('raw_data_settings.target_column')
        index_column = self.config.get_string('raw_data_settings.index_column')
        stacker_model = self.config.get_string('stacker.meta_model')
        stacker_init_params = dict(self.config.get_config('stacker.%s.init_params' % stacker_model))
        stacker_wrapped = get_wrapped_estimator(stacker_model, stacker_init_params)
        stacker_predict_probability = self.config.get_bool('stacker.%s.predict_probability' % stacker_model)
        class_label = self.config.get('modeling_settings.class_label')
        stacker_eval_metric = self.config.get_string('stacker.%s.eval_metric' % stacker_model)
        stacker_metrics_scorer = get_metrics_scorer(self.config.get('stacker.%s.metrics_scorer' % stacker_model))
        stacker_metrics_decimals = self.config.get_int('stacker.%s.metrics_decimals' % stacker_model)
        stacker_target_decimals = self.config.get_int('stacker.%s.target_decimals' % stacker_model)
        cols_to_exclude = self.config.get_list('modeling_settings.cols_to_exclude')
        num_folds = self.config.get_int('modeling_settings.cv_params.num_folds')
        stratified = self.config.get_bool('modeling_settings.cv_params.stratified')
        kfolds_shuffle = self.config.get_bool('modeling_settings.cv_params.kfolds_shuffle')
        cv_verbosity = self.config.get_int('modeling_settings.cv_params.cv_verbosity')
        stacker_bagging = self.config.get_bool('stacker.%s.bagging' % stacker_model)
        data_split_seed = self.config.get_int('modeling_settings.cv_params.data_split_seed')
        model_seeds_list = self.config.get_list('modeling_settings.model_seeds_list')

        # TODO: to add this logic
        # If True -> use raw features additionally to out-of-fold results
        stacker_use_raw_features = self.config.get_bool('stacker.%s.use_raw_features' % stacker_model)

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
        # Load initialized stacker
        stacker = pickle.load(open(self.input().path, "rb"))
        stacker_model = self.config.get_string('stacker.meta_model')
        stacker_hpo_method = self.config.get_string('stacker.%s.hpo_method' % stacker_model)
        stacker_hp_optimizator = load_hp_optimization_class(stacker_hpo_method)
        stacker_hpo_space = dict(self.config.get_config('stacker.%s.%s.hpo_space' % (stacker_model, stacker_hpo_method)))
        stacker_hpo_init_points = self.config.get_int('stacker.%s.%s.init_points' % (stacker_model, stacker_hpo_method))
        stacker_hpo_n_iter = self.config.get_int('stacker.%s.%s.n_iter' % (stacker_model, stacker_hpo_method))
        stacker_hpo_seed_val = self.config.get_int('stacker.%s.%s.seed_value' % (stacker_model, stacker_hpo_method))

        # Initialize hyper-parameter optimizator
        stacker_hpo = stacker_hp_optimizator(
            stacker, hp_optimization_space=stacker_hpo_space, init_points=stacker_hpo_init_points,
            n_iter=stacker_hpo_n_iter, seed_val=stacker_hpo_seed_val, output_dir=self.stacking_output_dir
        )

        # Run optimization and save output results
        stacker_hpo.run()
        stacker_hpo.save_optimized_hp()
        stacker_hpo.save_all_hp_results()

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
        labels_mapper = lambda x: 1 if x > 0.5 else 0 # for confusion matrix plot
        stacker.plot_confusion_matrix(class_names, labels_mapper, normalize=True, save=True)

        print('\nSaving %s\n' % self.full_path_to_file)
        with open(self.full_path_to_file, 'wb') as f:
            pickle.dump(stacker, f)

    def output(self):
        return luigi.LocalTarget(self.full_path_to_file)


if __name__ == '__main__':

    # Location of the project
    project_location = r'C:\Kaggle\home_credit_default_risk'  # os.getcwd()
    print('Location of project: {0}'.format(project_location))

    # Config file name and its location
    config_file = 'solution.conf'
    path_to_config = os.path.normpath(os.path.join(project_location, 'configs', config_file))
    print('Location of config : {0}'.format(path_to_config))

    # Parsing config file
    config = ConfigFactory.parse_file(path_to_config)

    run_feature_selection = config.get_bool('modeling_settings.run_feature_selection')
    run_hpo = config.get_bool('modeling_settings.run_hpo')

    stacker_model = config.get_string('stacker.meta_model')
    stacker_run_hpo = config.get_bool('stacker.%s.run_hpo' % stacker_model)  # if True -> run HPO of stacking meta-model

    folders_handler = ResultsLocationManager(config, run_feature_selection, run_hpo)
    ftg_output_dir = folders_handler.get_feature_generation_output_dir()[1]
    fts_output_dir = folders_handler.get_feature_selection_output_dir()[1]
    hpos_output_dir = folders_handler.get_hpo_output_dir()[1]
    solution_output_dir = folders_handler.get_solution_output_dir()[1]

    print(ftg_output_dir)
    if run_feature_selection:
        print(fts_output_dir)
    if run_hpo:
        print(hpos_output_dir)
    print(solution_output_dir)

    # luigi.build([RunStackerPrediction()], local_scheduler=True)
    luigi.build([RunSingleModelPrediction(project_location=project_location,
                                          run_feature_selection=run_feature_selection,
                                          run_hpo=run_hpo,
                                          ftg_output_dir=ftg_output_dir,
                                          fts_output_dir=fts_output_dir,
                                          hpos_output_dir=hpos_output_dir,
                                          solution_output_dir=solution_output_dir,
                                          config=config)], local_scheduler=True)
