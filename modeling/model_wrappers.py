import numpy as np
import pandas as pd


class ModelWrapper(object):
    def __init__(self, model, params=None, seed=27, model_name=None):
        self.model = model
        self.model_name = model_name
        self._verify_model_name_is_ok()
        self.has_seed_param = self._verify_model_has_seed_param()
        self._add_seed_to_params(params, seed)
        self.params = params
        self.estimator = self.model(**params)

    def _verify_model_name_is_ok(self):
        if self.model_name is None:
            try:
                self.model_name = self.model.__name__
            except Exception as e:
                print('Please provide name of the model. Error %s encountered when trying self.model.__name__' % e)

    def _verify_model_has_seed_param(self):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def reset_seed(self, seed):  # type: (int) -> None
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def fit_estimator(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def run_prediction(self, x, predict_probability, class_label, num_iteration):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def get_best_iteration(self):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def get_features_importance(self):
        # Abstract method, must be implemented by derived classes
        raise NotImplemented()

    def get_name(self):
        return self.model_name


class LightGBMWrapper(ModelWrapper):
    """Class that wraps LightGBM"""
    HP_DATATYPES = {
        'learning_rate': lambda x: round(x, 3),
        'max_depth': lambda x: int(round(x, 0)),
        'num_leaves': lambda x: int(round(x, 0)),
        'reg_alpha': lambda x: round(max(x, 0), 2),
        'reg_lambda': lambda x: round(max(x, 0), 2),
        'min_split_gain': lambda x: round(max(x, 0), 3),
        'subsample': lambda x: round(max(min(x, 1), 0), 2),
        'colsample_bytree': lambda x: round(max(min(x, 1), 0), 2),
        'min_child_weight': lambda x: int(round(x, 0))
    }

    def __init__(self, model, params=None, seed=27, name='lgbm'):
        super(LightGBMWrapper, self).__init__(model, params, seed, name)

    def _verify_model_has_seed_param(self):
        return 'random_state' in self.model().get_params()

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        if self.has_seed_param:
            params['random_state'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_seed(self, seed):  # type: (int) -> None
        self.params['random_state'] = seed
        self.estimator = self.model(**self.params)

    def fit_estimator(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of LGBM estimator.
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted LGBM estimator
        """
        self.estimator.fit(train_x, train_y,
                           eval_set=[(train_x, train_y), (valid_x, valid_y)],
                           eval_metric=eval_metric,
                           verbose=cv_verbosity)

    def run_prediction(self, x, predict_probability, class_label=None, num_iteration=1000
                       ):  # type: (pd.DataFrame, bool, (None, int, list), int) -> np.ndarray
        """
        This method run prediction of target variable. Two options available for prediction: class labels or
        class probabilities.
        :param x: pandas DF with the features for which the prediction of the target to be made
        :param predict_probability: if True -> use estimator.predict_proba() method, else -> estimator.predict()
        :param class_label: class label(s) for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. Class label(s) should be
                            selected from the target column.
                            - if class_label is None -> return probability of all class labels in the target
                            - if class_label is int -> return probability of selected class
                            - if class_label is list of int -> return probability of selected classes
        :param num_iteration: optimum number of iterations (used only for decision trees algorithms)
        :return: np.ndarray with either predicted probability of class(es) or class label(s)
        """
        if predict_probability:
            if class_label is not None:
                return self.estimator.predict_proba(x, num_iteration=num_iteration)[:, class_label]  # return probability of selected class(es)
            return self.estimator.predict_proba(x, num_iteration=num_iteration)  # return probability of all classes
        return self.estimator.predict(x, num_iteration=num_iteration)  # return target class labels (not a probability!)

    def get_best_iteration(self):
        return self.estimator.booster_.best_iteration

    def get_features_importance(self):
        """
        In LightGBM there are two types of features importance:
            - 'split': result contains numbers of times the feature is used in a model.
            - 'gain': result contains total gains of splits which use the feature.
        'Gain' is slightly more informative than the 'split' (thus it is proposed to be used in here by default).
        Note: self.estimator.feature_importances_ is not used here because it reflects 'split' importance by default.
        :return: features_names, features_importances
        """
        features_names = self.estimator.booster_.feature_name()
        features_importances = self.estimator.booster_.feature_importance(importance_type='gain')
        return features_names, features_importances


class XGBWrapper(ModelWrapper):
    """Class that wraps XGBoost"""
    HP_DATATYPES = {
        'learning_rate': lambda x: round(x, 3),
        'max_depth': lambda x: int(round(x, 0)),
        'reg_alpha': lambda x: round(max(x, 0), 2),
        'reg_lambda': lambda x: round(max(x, 0), 2),
        'subsample': lambda x: round(max(min(x, 1), 0), 2),
        'colsample_bytree': lambda x: round(max(min(x, 1), 0), 2),
        'min_child_weight': lambda x: int(round(x, 0)),
        'gamma': lambda x: round(x, 3),
    }

    def __init__(self, model, params=None, seed=27, name='xgb'):
        super(XGBWrapper, self).__init__(model, params, seed, name)

    def _verify_model_has_seed_param(self):
        return 'random_state' in self.model().get_params()

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        if self.has_seed_param:
            params['random_state'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_seed(self, seed):  # type: (int) -> None
        self.params['random_state'] = seed
        self.estimator = self.model(**self.params)

    def fit_estimator(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of XGB estimator. It should be noted that in xgboost v0.72 (the latest
        at the moment of creation of this code), there is no support of early_stopping_rounds directly from general dict
        of parameters (unlike in the case of LightGBM). Therefore, an explicit extraction of this parameter is
        implemented. If the early_stopping_rounds parameter is missing in the dict of parameters, None is used.
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted XGB estimator
        """
        self.estimator.fit(train_x, train_y,
                           eval_set=[(train_x, train_y), (valid_x, valid_y)],
                           eval_metric=eval_metric,
                           early_stopping_rounds=self.params.get("early_stopping_rounds", None),
                           verbose=cv_verbosity)

    def run_prediction(self, x, predict_probability, class_label=None, num_iteration=1000
                       ):  # type: (pd.DataFrame, bool, (None, int, list), int) -> np.ndarray
        """
        This method run prediction of target variable. Two options available for prediction: class labels or
        class probabilities.
        :param x: pandas DF with the features for which the prediction of the target to be made
        :param predict_probability: if True -> use estimator.predict_proba() method, else -> estimator.predict()
        :param class_label: class label(s) for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. Class label(s) should be
                            selected from the target column.
                            - if class_label is None -> return probability of all class labels in the target
                            - if class_label is int -> return probability of selected class
                            - if class_label is list of int -> return probability of selected classes
        :param num_iteration: optimum number of iterations (used only for decision trees algorithms)
        :return: np.ndarray with either predicted probability of class(es) or class label(s)
        """
        if predict_probability:
            if class_label is not None:
                return self.estimator.predict_proba(x, ntree_limit=num_iteration)[:, class_label]  # return probability of selected class(es)
            return self.estimator.predict_proba(x, ntree_limit=num_iteration)  # return probability of all classes
        return self.estimator.predict(x, ntree_limit=num_iteration)  # return target class labels (not a probability!)

    def get_best_iteration(self):
        return self.estimator.get_booster().best_iteration

    def get_features_importance(self):
        """
        The default value of self.estimator.feature_importances_ in XGBoost is 'weight' (not 'gain').
        The following options are available for XGBoost:
            - 'weight': the number of times a feature is used to split the data across all trees.
            - 'gain': the average gain across all splits the feature is used in.
            - 'cover': the average coverage across all splits the feature is used in.
            - 'total_gain': the total gain across all splits the feature is used in.
            - 'total_cover': the total coverage across all splits the feature is used in.
        'Gain' is slightly more informative than the 'split' (thus it is proposed to be used in here by default).
        :return: features_names, features_importances
        """
        features_names = self.estimator.get_booster().get_score(importance_type='gain').keys()
        features_importances = self.estimator.get_booster().get_score(importance_type='gain').values()
        return features_names, features_importances


class SklearnWrapper(ModelWrapper):
    """Class that wraps Sklearn ML algorithms (ET, LinerRegression, LogisticRegression, etc."""
    HP_DATATYPES = {
        # Logistic regression
        'C': lambda x: round(x, 3) if x < 1.0 else round(x, 1),
        'tol': lambda x: round(x, 5),
        'max_iter': lambda x: int(round(x, 0)),

        # ExtraTreesClassifier
        'n_estimators': lambda x: int(round(x, 0)),
        'max_depth': lambda x: int(round(x, 0)),
        'max_features': lambda x: round(max(min(x, 1), 0), 2),
        'max_leaf_nodes': lambda x: int(round(x, 0)),
        'min_samples_split': lambda x: int(round(x, 0)),
        'min_samples_leaf': lambda x: int(round(x, 0)),
        # 'min_impurity_decrease'
    }

    def __init__(self, model, params=None, seed=27, name=None):
        super(SklearnWrapper, self).__init__(model, params, seed, name)

    def _verify_model_has_seed_param(self):
        return 'random_state' in self.model().get_params()

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        if self.has_seed_param:
            params['random_state'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_seed(self, seed):  # type: (int) -> None
        self.params['random_state'] = seed
        self.estimator = self.model(**self.params)

    def fit_estimator(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of Sklearn estimator. It should be noted that fit method in Sklearn API
        does not support valid_x, valid_y, eval_metric, cv_verbosity arguments (thus they are not used here).
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted Sklearn estimator
        """
        self.estimator.fit(train_x, train_y)

    def run_prediction(self, x, predict_probability, class_label=None, num_iteration=1000
                       ):  # type: (pd.DataFrame, bool, (None, int, list), int) -> np.ndarray
        """
        This method run prediction of target variable. Two options available for prediction: class labels or
        class probabilities.
        :param x: pandas DF with the features for which the prediction of the target to be made
        :param predict_probability: if True -> use estimator.predict_proba() method, else -> estimator.predict()
        :param class_label: class label(s) for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. Class label(s) should be
                            selected from the target column.
                            - if class_label is None -> return probability of all class labels in the target
                            - if class_label is int -> return probability of selected class
                            - if class_label is list of int -> return probability of selected classes
        :param num_iteration: optimum number of iterations (used only for decision trees algorithms)
        :return: np.ndarray with either predicted probability of class(es) or class label(s)
        """
        if predict_probability:
            if class_label is not None:
                return self.estimator.predict_proba(x)[:, class_label]  # return probability of selected class(es)
            return self.estimator.predict_proba(x)  # return probability of all classes
        return self.estimator.predict(x)  # return target class labels (not a probability!)

    def get_features_importance(self):
        """
        In Sklearn algorithms, which do have feature_importances_ attribute in base_estimator (e.g. AdaBoostClassifier,
        ExtraTreesClassifier, etc.), features_names are not available and are assigned explicitly from the column names.
        :return: features_names, features_importances
        """
        features_names = None
        features_importances = self.estimator.feature_importances_
        return features_names, features_importances


# TODO: Implement CatBoost wrapper
class CBWrapper(ModelWrapper):
    """Class that wraps CatBoost algorithm"""

    def __init__(self, model, params=None, seed=27, name='cb'):
        super(CBWrapper, self).__init__(model, params, seed, name)

    def _verify_model_has_seed_param(self):
        return 'random_state' in self.model().get_params()

    def _add_seed_to_params(self, params, seed):  # type: (dict, int) -> dict
        if self.has_seed_param:
            params['random_state'] = seed
        return params

    def reinit_model_with_new_params(self, new_params):  # type: (dict) -> None
        self.params = new_params
        self.estimator = self.model(**self.params)

    def reset_seed(self, seed):  # type: (int) -> None
        self.params['random_state'] = seed
        self.estimator = self.model(**self.params)

    def fit_estimator(self, train_x, train_y, valid_x, valid_y, eval_metric, cv_verbosity):
        """
        This method is responsible for fitting of Sklearn estimator. It should be noted that fit method in Sklearn API
        does not support valid_x, valid_y, eval_metric, cv_verbosity arguments (thus they are not used here).
        :param train_x: train dataset with features (not including the target)
        :param train_y: train dataset with target column
        :param valid_x: validation dataset with features (not including the target)
        :param valid_y: validation dataset with target column
        :param eval_metric: 'rmse', 'mae', 'logloss', 'auc', etc.
        :param cv_verbosity: print info about CV training and validation errors every x iterations (e.g. 1000)
        :return: fitted Sklearn estimator
        """
        self.estimator.fit(train_x, train_y)

    def run_prediction(self, x, predict_probability, class_label=None, num_iteration=1000
                       ):  # type: (pd.DataFrame, bool, (None, int, list), int) -> np.ndarray
        """
        This method run prediction of target variable. Two options available for prediction: class labels or
        class probabilities.
        :param x: pandas DF with the features for which the prediction of the target to be made
        :param predict_probability: if True -> use estimator.predict_proba() method, else -> estimator.predict()
        :param class_label: class label(s) for which to predict the probability. Note: it is used only for
                            classification tasks and when the predict_probability=True. Class label(s) should be
                            selected from the target column.
                            - if class_label is None -> return probability of all class labels in the target
                            - if class_label is int -> return probability of selected class
                            - if class_label is list of int -> return probability of selected classes
        :param num_iteration: optimum number of iterations (used only for decision trees algorithms)
        :return: np.ndarray with either predicted probability of class(es) or class label(s)
        """
        if predict_probability:
            if class_label is not None:
                return self.estimator.predict_proba(x)[:, class_label]  # return probability of selected class(es)
            return self.estimator.predict_proba(x)  # return probability of all classes
        return self.estimator.predict(x)  # return target class labels (not a probability!)

    def get_features_importance(self):
        return


def get_wrapped_estimator(model, model_init_params):
    if model == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LightGBMWrapper(model=LGBMClassifier, params=model_init_params)
    elif model == 'xgboost':
        from xgboost import XGBClassifier
        return XGBWrapper(model=XGBClassifier, params=model_init_params)
    elif model == 'et':
        from sklearn.ensemble import ExtraTreesClassifier
        return SklearnWrapper(model=ExtraTreesClassifier, params=model_init_params, name='et')
    elif model == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        return SklearnWrapper(model=LogisticRegression, params=model_init_params, name='logreg')
    elif model == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        return SklearnWrapper(model=LinearRegression, params=model_init_params, name='linreg')
    else:
        raise NotImplemented()
