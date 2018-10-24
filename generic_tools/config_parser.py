import os

from pyhocon import ConfigFactory
from pyhocon.config_tree import ConfigTree
from functools32 import lru_cache


class ConfigFileHandler(object):
    SUBDIR_NO_FEATURE_SELECTION = 'no_feat_selection'  # name of sub-directory if run_feature_selection = False
    SUBDIR_NO_HYPER_PARAMS_OPTIM = 'no_hp_optimization'  # name of sub-directory if run_hpo = False
    SUBDIR_BAGGING_ON = 'bagging_on'  # name of sub-directory if run_bagging = True
    SUBDIR_BAGGING_OFF = 'bagging_off'  # name of sub-directory if run_bagging = False

    def __init__(self, project_location, config_directory, config_file):  # type: (str, str, str) -> None
        """
        This class is designed to parse configuration file and to take care of creating an input and output paths for
        storing and reading pipeline's results. The name of folders and sub-directories as well as proposed project
        structure is provided in the config.
        file.
        :param project_location:
        :param config_directory:
        :param config_file:
        """
        self.project_location = project_location
        self.config_directory = config_directory
        self.config_file = config_file

    @staticmethod
    @lru_cache(2)
    def parse_config_file(project_location, config_directory, config_file):  # type: (str, str, str) -> ConfigTree
        """
        This method returns parsed configuration file with all settings for running the pipeline
        :param project_location:
        :param config_directory:
        :param config_file:
        :return:
        """
        path_to_config = os.path.normpath(os.path.join(project_location, config_directory, config_file))
        if os.path.exists(path_to_config):
            return ConfigFactory.parse_file(path_to_config)
        else:
            raise IOError('No config file found in: %s' % path_to_config)

    def _get_feature_generation_output_dir(self, model):
        """
        This method returns path to a sub-directory in FEATURE_GENERATION_DIR of the project (self.project_structure)
        where is located train and test data sets (to be used in subsequent steps such as feature_selection
        (if run_feature_selection=True), hyper-parameters optimization (if run_hpo=True), single model prediction, etc.)

        Example of feats_generation_results_dir_path:
            ./features_generation/features_dataset_001/

        :param model: name of model to be used for prediction (e.g. lightgbm, xgboost, logistic regression, etc.)
        :return: name of features generation directory, and composed path to it
        """
        config = self.parse_config_file(self.project_location, self.config_directory, self.config_file)
        project_structure = dict(config.get_config('project_structure'))

        feats_generation_dir_name = config.get_string('modeling_settings.%s.name_feats_generation_dir' % model)
        feats_generation_results_dir_path = os.path.join(project_structure['FEATURE_GENERATION_DIR'],
                                                         feats_generation_dir_name)
        return feats_generation_dir_name, feats_generation_results_dir_path

    def _get_feature_selection_output_dir(self, model, run_feature_selection):
        """
        This method returns path to a sub-directory in FEATURE_SELECTION_DIR of the project (self.project_structure)
        where to store feature selection results. This path will then be used in steps such as hyper-parameters
        optimization (if run_hpo=True), single model prediction, etc.).

        Example of feats_selection_results_dir_path:
            ./features_selection/features_dataset_001/target_permutation_fts_001/

        :param model: name of model to be used for prediction (e.g. lightgbm, xgboost, logistic regression, etc.)
        :param run_feature_selection: if True -> run feature selection procedure
        :return: name of features selection output directory, and composed path to it
        """
        config = self.parse_config_file(self.project_location, self.config_directory, self.config_file)
        project_structure = dict(config.get_config('project_structure'))
        feats_generation_dir_name, feats_generation_results_dir_path = self._get_feature_generation_output_dir(model)

        if run_feature_selection:
            feats_selection_method = config.get_string('modeling_settings.%s.fs_method' % model)
            feats_selection_dir_name = config.get_string('features_selection.name_fs_dir')
            feats_selection_output_dir = os.path.join(feats_generation_dir_name,
                                                      '_'.join([feats_selection_method, feats_selection_dir_name]))
        else:
            feats_selection_output_dir = os.path.join(feats_generation_dir_name, self.SUBDIR_NO_FEATURE_SELECTION)
        feats_selection_results_dir_path = os.path.join(project_structure['FEATURE_SELECTION_DIR'],
                                                        feats_selection_output_dir)
        return feats_selection_output_dir, feats_selection_results_dir_path

    def _get_hpo_output_dir(self, model, run_feature_selection, run_hpo):
        """
        This method returns path to a sub-directory in HYPERPARAMS_OPTIM_DIR of the project (self.project_structure)
        where to store hyper-parameters optimization results. This path will then be used in single model prediction.

        Example of feats_selection_results_dir_path:
            ./hyper_parameters_optimization/lightgbm/features_dataset_001/target_permutation_fts_001/bayes_hpos_001/

        :param model: name of model to be used for prediction (e.g. lightgbm, xgboost, logistic regression, etc.)
        :param run_feature_selection: if True -> run feature selection procedure
        :param run_hpo: if True -> run hyper-parameters optimization
        :return: name of hyper-parameters optimization output directory, and composed path to it
        """
        config = self.parse_config_file(self.project_location, self.config_directory, self.config_file)
        project_structure = dict(config.get_config('project_structure'))
        feats_selection_output_dir, feats_selection_results_dir_path = \
            self._get_feature_selection_output_dir(model, run_feature_selection)

        if run_hpo:
            hpo_method = config.get_string('modeling_settings.%s.hpo_method' % model)
            hpo_name_dir = config.get_string('hp_optimization.name_hpo_dir')
            hpo_output_dir = os.path.normpath(os.path.join(model, feats_selection_output_dir,
                                                           '_'.join([hpo_method, hpo_name_dir])))
        else:
            hpo_output_dir = os.path.normpath(os.path.join(model, feats_selection_output_dir,
                                                           self.SUBDIR_NO_HYPER_PARAMS_OPTIM))
        hpo_results_dir_path = os.path.join(project_structure['HYPERPARAMS_OPTIM_DIR'], hpo_output_dir)
        return hpo_output_dir, hpo_results_dir_path

    def _get_solution_output_dir(self, model, run_feature_selection, run_hpo, run_bagging):
        """
        This method returns path to a sub-directory in SOLUTION_DIR of the project (self.project_structure) where
        to store single model prediction results (both OOF and test predictions, CV data, feature importances (if tree
        algorithm has been used), bagged results (if bagging flag has been set to True)). It also worth to mention that
        this path might then be used during stacking/blending processes (to upload OOF predictions...).

        Example of feats_selection_results_dir_path:
            ./single_model_solution/lightgbm/features_dataset_001/target_permutation_fts_001/bayes_hpos_001

        :param model: name of model to be used for prediction (e.g. lightgbm, xgboost, logistic regression, etc.)
        :param run_feature_selection: if True -> run feature selection procedure
        :param run_hpo: if True -> run hyper-parameters optimization
        :param run_bagging: if True -> run bagging over different seeds
        :return: name of single model prediction results directory, and composed path to it
        """
        config = self.parse_config_file(self.project_location, self.config_directory, self.config_file)
        project_structure = dict(config.get_config('project_structure'))
        hpo_output_dir, hpo_results_dir_path = self._get_hpo_output_dir(model, run_feature_selection, run_hpo)
        single_model_solution_output_dir = hpo_output_dir
        single_model_results_dir_path = os.path.join(project_structure['SOLUTION_DIR'],
                                                     single_model_solution_output_dir,
                                                     self.SUBDIR_BAGGING_ON if run_bagging else self.SUBDIR_BAGGING_OFF)
        return single_model_solution_output_dir, single_model_results_dir_path

    def _check_run_settings(self, model):
        config = self.parse_config_file(self.project_location, self.config_directory, self.config_file)
        run_feature_selection = config.get_bool('modeling_settings.%s.run_fs' % model)
        run_hpo = config.get_bool('modeling_settings.%s.run_hpo' % model)
        run_bagging = config.get_bool('modeling_settings.%s.run_bagging' % model)
        return run_feature_selection, run_hpo, run_bagging

    def _prepare_single_model_input_parameters_for_luigi(self, model, run_feature_selection, run_hpo, run_bagging):
        return {
            'project_location': self.project_location,
            'config_directory': self.config_directory,
            'config_file': self.config_file,
            'model': model,
            'run_feature_selection': run_feature_selection,
            'run_hpo': run_hpo,
            'run_bagging': run_bagging,
            'fg_output_dir': self._get_feature_generation_output_dir(model)[1],
            'fs_output_dir': self._get_feature_selection_output_dir(model, run_feature_selection)[1],
            'hpo_output_dir': self._get_hpo_output_dir(model, run_feature_selection, run_hpo)[1],
            'solution_output_dir': self._get_solution_output_dir(model, run_feature_selection, run_hpo, run_bagging)[1]}

    def prepare_input_parameters_for_luigi_pipeline(self):  # type: () -> list
        config = self.parse_config_file(self.project_location, self.config_directory, self.config_file)
        base_models = config.get('modeling_settings.models')

        if isinstance(base_models, list):
            luigi_pipeline_input_parameters = []
            for model in base_models:
                run_feature_selection, run_hpo, run_bagging = self._check_run_settings(model)
                input_parameters = self._prepare_single_model_input_parameters_for_luigi(model,
                                                                                         run_feature_selection,
                                                                                         run_hpo,
                                                                                         run_bagging)
                luigi_pipeline_input_parameters.append(input_parameters)
            return luigi_pipeline_input_parameters

        elif isinstance(base_models, basestring):
            run_feature_selection, run_hpo, run_bagging = self._check_run_settings(base_models)
            input_parameters = self._prepare_single_model_input_parameters_for_luigi(base_models,
                                                                                     run_feature_selection,
                                                                                     run_hpo,
                                                                                     run_bagging)
            return [input_parameters]
        else:
            raise TypeError("modeling_settings.models in config file should be either list or string. "
                            "Instead got %s" % type(base_models))
