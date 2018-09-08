import json

from generic_tools.utils import timing
from bayes_opt import BayesianOptimization
from modeling.cross_validation import Predictor


class HyperParamsOptimization(object):

    def __init__(self, predictor, seed_val=27, metrics_decimals=6, filename='optim_hp'):
        """
        This is a base class for optimization of model's hyperparameters. Methods of this class can be reused in
        derived classes (as, for instance, in BayesHyperParamsOptimization). These methods allows adjusting of data
        types of the hyperparameters, auto-complete missing parameters, save / read parameters from the disk.
        :param predictor: instance of Predictor class.
        :param seed_val: seed numpy random generator
        :param metrics_decimals: rounding precision for evaluation metrics (e.g. for CV printouts)
        :param filename: name of hyperparameter optimizer (is used when saving results of optimization)
        """
        self.predictor = predictor  # type: Predictor
        self.seed_val = seed_val  # type: int
        self.metrics_decimals = metrics_decimals # type: int
        self.filename = filename  # type: str
        self.best_params = None # type: dict
        self.best_score = None # type: float

    def _adjust_hyperparameters_datatypes(self, hp_optimization_space): # type: (dict) -> dict
        """
        This method adjust data types of model hyperparameters to the requested: int, float, etc.
        :param hp_optimization_space: dict with hyperparameters to be optimized within corresponding variation ranges
        :return: updated hp_optimization_space dict
        """
        map_dict = self.predictor.classifier.HP_DATATYPES # dict with a data types for a model hyperparameters
        return {k: map_dict[k](v) if k in map_dict else v for k, v in hp_optimization_space.items()}

    def _complete_missing_hyperparameters_from_init_params(self, hp_optimization_space):  # type: (dict) -> dict
        """
        This method will add parameters that can not be passed directly to run_bayes_optimization() function.
        These parameters are generally not iterable elements such as, for instance, nthread, verbose, silent lgbm params
        :param hp_optimization_space: dict with hyperparameters to be optimized within corresponding variation ranges
        :return: updated hp_optimization_space dict
        """
        init_params = self.predictor.classifier.params
        for key in set(init_params.keys()).difference(set(hp_optimization_space.keys())):
            hp_optimization_space[key] = init_params[key]
        return hp_optimization_space

    def hp_optimizer(self, **kwargs):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    @timing
    def run(self, **kwargs):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def save_hyperparams(self, path_to_save_data):  # type: (str) -> None
        """
        This method saves Bayes Optimized hyperparameters to the disc.
        :param path_to_save_data: path to be used when storing csv file with the weights
        :return: None
        """
        filename = '_'.join([self.filename, self.predictor.model_name, str(self.best_score)]) + '.txt'
        output_figname = ('\\'.join([path_to_save_data, filename]))
        print('\nSaving optimized hyperparameters into %s' % output_figname)
        with open(output_figname, 'w') as file:
            file.write(json.dumps(self.best_params))

    def read_hyperparams(self, path_to_save_data, filename):  # type: (str, str) -> None
        """
        This method reads optimized hyperparameters from the disc.
        :param path_to_save_data: path to the saved hyperparameters
        :param filename: name of the file with the hyperparameters
        :return: None
        """
        if not self.predictor.model_name in filename:
            raise AttributeError("The model name should be in the name of the file with the hyperparameters. "
                                 "Current model is {0} but the name of the file with parameters is {1}."
                                 .format(self.predictor.model_name, filename))

        output_figname = ('\\'.join([path_to_save_data, filename]))
        print('\nReading optimized hyperparameters from %s...' % output_figname)
        with open(output_figname, 'r') as file:
            self.best_params = json.load(file)
        print('\nNew best_params attribute contains:\n{0}'.format(self.best_params))


class BayesHyperParamsOptimization(HyperParamsOptimization):
    FILENAME = 'bayes_opt_hp'

    def __init__(self, predictor, seed_val=27, metrics_decimals=6):
        """
        This class adopts Bayes Optimization to find set of model's hyperparameters that lead to best CV results.
        :param predictor: instance of Predictor class.
        :param seed_val: seed numpy random generator
        :param metrics_decimals: rounding precision for evaluation metrics (e.g. for CV printouts)
        """
        super(BayesHyperParamsOptimization, self).__init__(predictor, seed_val, metrics_decimals, self.FILENAME)

    def hp_optimizer(self, **hp_optimization_space):
        """
        This method runs cross-validation with the set of hp provided by Bayes procedure and returns CV score
        :param hp_optimization_space: dict with hyperparameters to be optimized within corresponding variation ranges
        :return: CV score according to provided evaluation metrics (this is defined in the instance of class Predictor)
        """
        hp_optimization_space = self._adjust_hyperparameters_datatypes(hp_optimization_space)
        hp_optimization_space = self._complete_missing_hyperparameters_from_init_params(hp_optimization_space)
        self.predictor.classifier.reinit_model_with_new_params(hp_optimization_space)
        _, _, _, _, cv_score = self.predictor._run_cv_one_seed(seed_val=self.seed_val, predict_test=False)
        return cv_score

    @timing
    def run(self, hp_optimization_space, init_points, n_iter, path_to_save_data=None):
        """
        This method runs Bayes Search of optimal hyperparameters that gives max CV score
        :param hp_optimization_space: dict with hyperparameters to be optimized within corresponding variation ranges
        :param init_points: number of initial points in Bayes Optimization procedure
        :param n_iter: number of iteration in Bayes Optimization procedure
        :param path_to_save_data: if provided, will save optimal hyperparameters to the disk
        :return: None
        """

        bo = BayesianOptimization(self.hp_optimizer, hp_optimization_space)
        bo.maximize(init_points=init_points, n_iter=n_iter)

        best_params = self._adjust_hyperparameters_datatypes(bo.res['max']['max_params'])
        self.best_params = self._complete_missing_hyperparameters_from_init_params(best_params)
        self.best_score = round(bo.res['max']['max_val'], self.metrics_decimals)
        print('\n'.join(['', '=' * 70, '\nMax CV score: {0}'.format(self.best_score)]))
        print('Optimal parameters:\n{0}'.format(self.best_params))


