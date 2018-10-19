import os
from pyhocon.config_tree import ConfigTree


class ResultsLocationManager(object):
    DIR_NO_FEATURE_SELECTION = 'no_fs'  # name of sub-directory if run_feature_selection = False
    DIR_NO_HYPER_PARAMS_OPTIM = 'no_hpo'  # name of sub-directory if run_hpo = False

    def __init__(self, config, run_feature_selection, run_hpo):  # type: (ConfigTree, bool, bool) -> None
        """
        This class is designed to take care of creating an input and output paths for storing and reading pipeline's
        results. The name of folders and sub-directories as well as proposed project structure is provided in the config
        file.
        :param config: configuration file with all settings needed for running the solution pipeline
        :param run_feature_selection: if True -> run feature selection procedure
        :param run_hpo: if True -> run hyper-parameters optimization
        """
        self.config = config
        self.run_feature_selection = run_feature_selection
        self.run_hpo = run_hpo

        # Name of folders composing the backbone of the project (features_generation, features_selection,
        # hyper_parameters_optimization, single_model_solution, stacking, blending, etc.)
        self.project_structure = dict(config.get_config('project_structure'))

        # Name of model to be used for prediction (e.g. lightgbm, xgboost, logistic regression, etc.)
        self.model = config.get_string('modeling_settings.model')

    def get_feature_generation_output_dir(self):
        """
        This method returns path to a sub-directory in FEATURE_GENERATION_DIR of the project (self.project_structure)
        where is located train and test data sets (to be used in subsequent steps such as feature_selection
        (if run_feature_selection=True), hyper-parameters optimization (if run_hpo=True), single model prediction, etc.)

        Example of feats_generation_results_dir_path:
            ./features_generation/features_dataset_001/

        :return: name of features generation directory, and composed path to it
        """
        feats_generation_dir_name = self.config.get_string('features_generation.feats_generation_dir_name')
        feats_generation_results_dir_path = os.path.join(self.project_structure['FEATURE_GENERATION_DIR'],
                                                         feats_generation_dir_name)
        return feats_generation_dir_name, feats_generation_results_dir_path

    def get_feature_selection_output_dir(self):
        """
        This method returns path to a sub-directory in FEATURE_SELECTION_DIR of the project (self.project_structure)
        where to store feature selection results. This path will then be used in steps such as hyper-parameters
        optimization (if run_hpo=True), single model prediction, etc.).

        Example of feats_selection_results_dir_path:
            ./features_selection/features_dataset_001/target_permutation_fts_001/

        :return: name of features selection output directory, and composed path to it
        """
        feats_generation_dir_name, feats_generation_results_dir_path = self.get_feature_generation_output_dir()

        if self.run_feature_selection:
            feats_selection_method = self.config.get_string('features_selection.method')
            feats_selection_dir_name = self.config.get_string('features_selection.feats_selection_dir_name')
            feats_selection_output_dir = os.path.join(feats_generation_dir_name,
                                                      '_'.join([feats_selection_method, feats_selection_dir_name]))
        else:
            feats_selection_output_dir = os.path.join(feats_generation_dir_name, self.DIR_NO_FEATURE_SELECTION)
        feats_selection_results_dir_path = os.path.join(self.project_structure['FEATURE_SELECTION_DIR'],
                                                        feats_selection_output_dir)
        return feats_selection_output_dir, feats_selection_results_dir_path

    def get_hpo_output_dir(self):
        """
        This method returns path to a sub-directory in HYPERPARAMS_OPTIM_DIR of the project (self.project_structure)
        where to store hyper-parameters optimization results. This path will then be used in single model prediction.

        Example of feats_selection_results_dir_path:
            ./hyper_parameters_optimization/lightgbm/features_dataset_001/target_permutation_fts_001/bayes_hpos_001/

        :return: name of hyper-parameters optimization output directory, and composed path to it
        """
        feats_selection_output_dir, feats_selection_results_dir_path = self.get_feature_selection_output_dir()

        if self.run_hpo:
            hpo_method = self.config.get_string('hp_optimization.method')
            hpo_name_dir = self.config.get_string('hp_optimization.name_hpo_dir')
            hpo_output_dir = os.path.normpath(os.path.join(self.model, feats_selection_output_dir,
                                                           '_'.join([hpo_method, hpo_name_dir])))
        else:
            hpo_output_dir = os.path.normpath(os.path.join(self.model, feats_selection_output_dir,
                                                           self.DIR_NO_HYPER_PARAMS_OPTIM))
        hpo_results_dir_path = os.path.join(self.project_structure['HYPERPARAMS_OPTIM_DIR'], hpo_output_dir)
        return hpo_output_dir, hpo_results_dir_path

    def get_solution_output_dir(self):
        """
        This method returns path to a sub-directory in SOLUTION_DIR of the project (self.project_structure) where
        to store single model prediction results (both OOF and test predictions, CV data, feature importances (if tree
        algorithm has been used), bagged results (if bagging flag has been set to True)). It also worth to mention that
        this path might then be used during stacking/blending processes (to upload OOF predictions...).

        Example of feats_selection_results_dir_path:
            ./single_model_solution/lightgbm/features_dataset_001/target_permutation_fts_001/bayes_hpos_001

        :return: name of single model prediction results directory, and composed path to it
        """
        hpo_output_dir, hpo_results_dir_path = self.get_hpo_output_dir()
        single_model_solution_output_dir = hpo_output_dir
        single_model_results_dir_path = os.path.join(self.project_structure['SOLUTION_DIR'],
                                                     single_model_solution_output_dir)
        return single_model_solution_output_dir, single_model_results_dir_path
